"""
Detection + multi-object tracking for NearMiss.

Detection uses a PRETRAINED Ultralytics YOLO model (COCO classes: person, bicycle, car, motorcycle,
bus, truck). Tracking uses Ultralytics' built-in ByteTrack. Both are disclosed in the app.

Why the old version highlighted objects only "for a brief moment"
-----------------------------------------------------------------
1. A box was drawn only on frames where ByteTrack returned an ID for it. On every frame the detector
   missed (very common for small, far-away cars), the box simply vanished.
2. yolov8n at 640 px with confidence 0.35 misses far-away road users, so they were only found
   while they were close to the camera.
3. ByteTrack remembered a lost object for just 30 frames, and low-confidence detections were thrown
   away before the tracker could use them to keep a track alive.

What this version does about it
-------------------------------
* PersistentTracks remembers every track. If the detector misses an object for up to `coast_s`
  seconds, its box is carried forward (dashed outline) instead of disappearing. Motion trails show
  where each object has been.
* Detector confidence floor is low (0.12). ByteTrack is designed for this: low-score detections
  can only EXTEND a track that already exists; a NEW track still needs `start_conf`.
* The tracker keeps lost tracks alive for `track_buffer_s` seconds (default 3 s) at the processing rate.
* Larger inference size (960 px) and a selectable model size (n / s / m).
* Uses the Apple-silicon GPU ("mps") or CUDA when available.

Only OBSERVED detections are written to detections.csv. Carried-forward boxes are drawing only; the
analysis fills short gaps by interpolation (engine/trajectories.py).
"""
from __future__ import annotations

import os
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from engine.motion import CameraMotion
from engine.privacy import blur_detections

# COCO class ids we track (street-level models).
TARGET_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# Aerial models (*-obb.pt) are trained on DOTA satellite / drone images: they see vehicles from above but
# have NO person or bicycle class. Their names are mapped onto ours.
AERIAL_NAMES = {"small vehicle": "car", "large vehicle": "truck"}
# A model trained on VisDrone (tools/train_visdrone_colab.ipynb): hand-labelled drone images, so it also finds
# pedestrians and bikes from above. "van" is a car for NearMiss; tricycles are counted with motorcycles.
VISDRONE_NAMES = {"pedestrian": "person", "people": "person", "bicycle": "bicycle", "car": "car", "van": "car",
                  "truck": "truck", "bus": "bus", "motor": "motorcycle", "tricycle": "motorcycle",
                  "awning-tricycle": "motorcycle"}

# BGR colors, one per class, so a class never changes color mid-video.
CLASS_COLORS = {
    "person": (0, 200, 255),
    "bicycle": (90, 220, 90),
    "motorcycle": (80, 230, 200),
    "car": (255, 170, 40),
    "bus": (255, 110, 200),
    "truck": (255, 110, 200),
}

MODEL_NAME = "yolov8n.pt"
MODEL_CHOICES = {
    "yolov8n.pt": "street-level camera, fastest",
    "yolov8s.pt": "street-level camera, finds more far-away road users (about 2x slower)",
    "yolov8m.pt": "street-level camera, most accurate, slowest",
}
AERIAL_MODEL = "yolo11n-obb.pt"
VISDRONE_MODEL = "nearmiss-visdrone.pt"       # made by tools/train_visdrone_colab.ipynb; listed once it is here
AERIAL_CHOICES = {
    "yolo11n-obb.pt": "overhead / drone, vehicles only, fastest",
    "yolo11s-obb.pt": "overhead / drone, vehicles only, slower",
    "yolov8s-obb.pt": "overhead / drone, vehicles only, older model",
}
HERE = Path(__file__).resolve().parent


def aerial_choices() -> Dict[str, str]:
    """Overhead models to offer; the VisDrone-trained one first when its weights are in the app folder."""
    if (HERE / VISDRONE_MODEL).exists():
        return {VISDRONE_MODEL: "overhead / drone, trained on VisDrone: vehicles, pedestrians and bikes",
                **AERIAL_CHOICES}
    return dict(AERIAL_CHOICES)


TRUCK_LENGTH_RATIO = 1.6   # from above: a truck or bus is at least 1.6 x a typical car's length (car ~4.6 m -> 7.4 m)
CAR_ASPECT = 2.5           # a car is about 2.5 x as long as it is wide (4.6 m x 1.8 m)


def size_classes(lengths: np.ndarray, widths: np.ndarray, ref_length: Optional[float]) -> Tuple[List[str], Optional[float]]:
    """Car or truck for overhead detections, from the size of each rotated box.

    The aerial model's own "large vehicle" class is unreliable: SUVs and vans flip between it and
    "small vehicle" from frame to frame. Length is steady and physical. The typical car length in this
    frame is 2.5 x the median box WIDTH: widths barely differ between cars and trucks, and a vehicle cut off
    at the frame edge keeps its width but not its length. Being per frame, it follows a drone that climbs
    or descends. Returns the classes and the reference to reuse on frames with too few vehicles."""
    if len(widths) >= 3:
        ref_length = CAR_ASPECT * float(np.median(widths))
    if not ref_length:
        return ["car"] * len(lengths), ref_length
    return ["truck" if L >= TRUCK_LENGTH_RATIO * ref_length else "car" for L in lengths], ref_length


def is_aerial(model_name: str) -> bool:
    return "-obb" in model_name or "visdrone" in model_name


def finds_people(model_name: str) -> bool:
    """False for the DOTA aerial models, which have no pedestrian or bicycle class."""
    return "-obb" not in model_name


def class_map(names: Dict[int, str]) -> Dict[int, str]:
    """Model class id -> our class name, for the classes NearMiss tracks."""
    if any(n in AERIAL_NAMES for n in names.values()):
        return {i: AERIAL_NAMES[n] for i, n in names.items() if n in AERIAL_NAMES}
    if "pedestrian" in names.values() and "van" in names.values():
        return {i: VISDRONE_NAMES[n] for i, n in names.items() if n in VISDRONE_NAMES}
    return {i: TARGET_CLASSES[i] for i in TARGET_CLASSES if names.get(i) == TARGET_CLASSES[i]}


# ----------------------------------------------------------------------------- settings
@dataclass
class TrackerSettings:
    model_name: str = "yolov8n.pt"
    imgsz: int = 960              # detector input size (px). Bigger finds smaller objects, but is slower.
    detector_conf: float = 0.12   # floor passed to the detector; ByteTrack uses weak detections to extend tracks
    high_conf: float = 0.25       # ByteTrack first-stage match threshold
    start_conf: float = 0.30      # confidence needed to START a new track (higher = fewer false tracks)
    track_buffer_s: float = 3.0   # how long a lost track is kept alive (seconds)
    match_thresh: float = 0.8     # ByteTrack association threshold
    stride: int = 1               # process every Nth frame (1 = all frames)
    device: str = "auto"          # "auto" | "cpu" | "mps" | "cuda:0"
    coast_s: float = 1.0          # carry a box forward this long after a missed detection
    trail_s: float = 2.0          # length of the motion trail
    min_hits_to_coast: int = 3    # never carry forward a track seen fewer times than this (avoids ghosts)
    stabilize: bool = False       # estimate camera motion (drone / drifting camera) and save it for the analysis

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RenderOptions:
    blur_privacy: bool = True     # pixelate the regions where faces / plates usually are
    privacy_level: str = "standard"   # "standard" | "strict" (whole box)
    show_trails: bool = True
    show_coasting: bool = True
    show_confidence: bool = False
    max_width: int = 1280         # annotated frames are scaled down to this width (smaller, faster files)


def auto_stride(fps: float, target_fps: float = 12.0) -> int:
    """Process about `target_fps` frames per second: plenty for walking and driving speeds."""
    return max(1, int(round(fps / target_fps))) if fps > 0 else 1


def pick_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# -------------------------------------------------------------------------------- data
@dataclass
class Detection:
    frame: int
    time_s: float
    track_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DrawItem:
    """One box to draw on one frame."""
    track_id: int
    class_name: str
    box: Tuple[float, float, float, float]
    live: bool                       # True: detected on this frame. False: carried forward from memory.
    trail: List[Tuple[float, float]]
    confidence: float


@dataclass
class FrameResult:
    frame_idx: int                   # index in the ORIGINAL video
    time_s: float
    annotated: np.ndarray            # BGR frame with boxes / trails (and privacy blur if enabled)
    detections: List[Detection]      # observed detections only (these go to detections.csv)
    items: List[DrawItem]            # everything that was drawn (observed + carried forward)
    H: Optional[np.ndarray] = None   # this frame's pixels -> first frame's pixels (only with stabilize)


# ---------------------------------------------------------------------- track memory
@dataclass
class _Mem:
    box: np.ndarray
    vel: np.ndarray
    last: int
    hits: int
    conf: float
    votes: Dict[str, float]
    trail: Deque[Tuple[float, float]]
    label: str = ""

    @property
    def cls(self) -> str:
        """The track's class, by confidence-weighted vote. It is sticky: it only changes when another class has
        clearly more votes, so a close race (an SUV: car or truck?) does not flip the label back and forth."""
        best = max(self.votes, key=self.votes.get)
        if not self.label or self.votes[best] > 1.5 * self.votes.get(self.label, 0.0):
            self.label = best
        return self.label


class PersistentTracks:
    """Keeps every track alive on screen for a short time after the detector stops seeing it."""

    def __init__(self, coast_frames: int, trail_frames: int, min_hits: int = 3):
        self.coast_frames = max(0, int(coast_frames))
        self.trail_frames = max(2, int(trail_frames))
        self.min_hits = min_hits
        self._mem: Dict[int, _Mem] = {}

    def update(self, frame_no: int, dets: List[Detection], frame_wh: Tuple[int, int]) -> List[DrawItem]:
        """`frame_no` counts PROCESSED frames (0, 1, 2, ...), not original video frames."""
        for d in dets:
            box = np.array([d.x1, d.y1, d.x2, d.y2], dtype=float)
            m = self._mem.get(d.track_id)
            if m is None:
                m = _Mem(box=box, vel=np.zeros(4), last=frame_no, hits=0, conf=d.confidence, votes={},
                         trail=deque(maxlen=self.trail_frames))
                self._mem[d.track_id] = m
            else:
                gap = max(1, frame_no - m.last)
                if gap <= self.coast_frames + 1:
                    m.vel = 0.6 * m.vel + 0.4 * (box - m.box) / gap   # smoothed box velocity (px per processed frame)
                else:
                    m.vel = np.zeros(4)
                m.box, m.last = box, frame_no
            m.hits += 1
            m.conf = d.confidence
            m.votes[d.class_name] = m.votes.get(d.class_name, 0.0) + d.confidence
            m.trail.append(((box[0] + box[2]) / 2, box[3]))

        W, H = frame_wh
        items: List[DrawItem] = []
        for tid in list(self._mem):
            m = self._mem[tid]
            age = frame_no - m.last
            if age == 0:
                items.append(DrawItem(tid, m.cls, tuple(m.box), True, list(m.trail), m.conf))
                continue
            if age > self.coast_frames:
                del self._mem[tid]
                continue
            if m.hits < self.min_hits:
                continue
            shift = m.vel * sum(0.85 ** k for k in range(1, age + 1))   # damped, so it does not run away
            box = m.box + shift
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not (0 <= cx <= W and 0 <= cy <= H):
                del self._mem[tid]          # it left the frame
                continue
            trail = list(m.trail) + [((box[0] + box[2]) / 2, box[3])]
            items.append(DrawItem(tid, m.cls, tuple(box), False, trail, m.conf))
        return sorted(items, key=lambda i: i.track_id)


# ---------------------------------------------------------------------------- drawing
def _dashed_rect(img, p1, p2, color, thickness=1, dash=7):
    x1, y1 = p1
    x2, y2 = p2
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = int(max(abs(bx - ax), abs(by - ay)))
        if length == 0:
            continue
        for s in range(0, length, dash * 2):
            e = min(s + dash, length)
            pa = (int(ax + (bx - ax) * s / length), int(ay + (by - ay) * s / length))
            pb = (int(ax + (bx - ax) * e / length), int(ay + (by - ay) * e / length))
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def render_frame(frame: np.ndarray, items: List[DrawItem], opts: RenderOptions, frame_idx: int, time_s: float) -> np.ndarray:
    """Return the annotated frame. Privacy blur happens first, at full resolution, before anything is drawn."""
    img = frame.copy()
    shown = [i for i in items if i.live or opts.show_coasting]
    if opts.blur_privacy and shown:
        cls_map = {"person": "pedestrian"}
        blur_detections(img, [(cls_map.get(i.class_name, i.class_name), i.box) for i in shown], opts.privacy_level)

    scale = min(1.0, opts.max_width / float(img.shape[1]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # translucent fill under live boxes
    fill = img.copy()
    for it in shown:
        if it.live:
            x1, y1, x2, y2 = (int(v * scale) for v in it.box)
            cv2.rectangle(fill, (x1, y1), (x2, y2), CLASS_COLORS.get(it.class_name, (200, 200, 200)), -1)
    img = cv2.addWeighted(fill, 0.16, img, 0.84, 0)

    for it in shown:
        color = CLASS_COLORS.get(it.class_name, (200, 200, 200))
        if opts.show_trails and len(it.trail) >= 2:
            pts = [(int(x * scale), int(y * scale)) for x, y in it.trail]
            n = len(pts)
            for k in range(1, n):
                f = k / n
                shade = tuple(int(c * (0.35 + 0.65 * f)) for c in color)
                cv2.line(img, pts[k - 1], pts[k], shade, 1 + int(2 * f), cv2.LINE_AA)
        x1, y1, x2, y2 = (int(v * scale) for v in it.box)
        if it.live:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        else:
            _dashed_rect(img, (x1, y1), (x2, y2), color, 1)
        label = f"{it.class_name} #{it.track_id}"
        if opts.show_confidence and it.live:
            label += f" {it.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(img, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    live_n = sum(1 for i in shown if i.live)
    hud = f"t = {time_s:6.1f} s   tracked: {len(shown)}  (detected now: {live_n})"
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 360), 22), (0, 0, 0), -1)
    cv2.putText(img, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------- tracker
def _write_tracker_yaml(path: str, s: TrackerSettings, processed_fps: float) -> None:
    buffer_frames = max(30, int(round(s.track_buffer_s * processed_fps)))
    with open(path, "w") as f:
        f.write(
            "tracker_type: bytetrack\n"
            f"track_high_thresh: {s.high_conf}\n"
            "track_low_thresh: 0.05\n"
            f"new_track_thresh: {s.start_conf}\n"
            f"track_buffer: {buffer_frames}\n"
            f"match_thresh: {s.match_thresh}\n"
            "fuse_score: True\n"
        )


def _load_model(name: str):
    """Load a YOLO model; if the weights cannot be downloaded (offline), fall back to the local yolov8n."""
    from ultralytics import YOLO  # imported here so the rest of the app works without ultralytics installed

    local = HERE / name                          # weights in the app folder, wherever the app is started from
    try:
        return YOLO(str(local) if local.exists() else name), name, ""
    except Exception as ex:
        fallback = AERIAL_MODEL if is_aerial(name) else MODEL_NAME
        if name == fallback:
            raise
        model = YOLO(fallback)
        return model, fallback, f"Could not load {name} ({type(ex).__name__}); using {fallback} instead."


class RoadUserTracker:
    """YOLO + ByteTrack over a video. Yields one FrameResult per processed frame."""

    def __init__(self, settings: Optional[TrackerSettings] = None, render: Optional[RenderOptions] = None):
        self.s = settings or TrackerSettings()
        self.render = render or RenderOptions()
        self.model, self.model_used, self.warning = _load_model(self.s.model_name)
        self.device_used = pick_device(self.s.device)
        self.classes = class_map(self.model.names)
        self.motion = None
        self.stats = {"frames": 0, "frames_with_carried_boxes": 0}

    def _stream(self, video_path: str, yaml_path: str, device: str):
        return self.model.track(
            source=video_path,
            classes=list(self.classes),
            conf=self.s.detector_conf,
            imgsz=self.s.imgsz,
            device=device,
            vid_stride=max(1, self.s.stride),
            persist=True,
            stream=True,
            tracker=yaml_path,
            verbose=False,
        )

    def process_video(self, video_path: str, fps: float, progress_cb: Optional[Callable[[int, int], None]] = None
                      ) -> Iterator[FrameResult]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        stride = max(1, self.s.stride)
        processed_fps = (fps / stride) if fps > 0 else 10.0
        memory = PersistentTracks(
            coast_frames=int(round(self.s.coast_s * processed_fps)),
            trail_frames=int(round(self.s.trail_s * processed_fps)),
            min_hits=self.s.min_hits_to_coast,
        )

        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.close()
        _write_tracker_yaml(tmp.name, self.s, processed_fps)
        try:
            device = self.device_used
            try:
                stream = self._stream(video_path, tmp.name, device)
                first = next(stream, None)
            except Exception as ex:
                if device == "cpu":
                    raise
                self.warning = (self.warning + " " if self.warning else "") + \
                    f"{device} failed ({type(ex).__name__}); running on the CPU instead."
                self.device_used = device = "cpu"
                self.model, _, _ = _load_model(self.model_used)     # fresh tracker state
                self.classes = class_map(self.model.names)
                stream = self._stream(video_path, tmp.name, device)
                first = next(stream, None)

            def results():
                if first is not None:
                    yield first
                yield from stream

            self.motion = CameraMotion() if self.s.stabilize else None
            ref_length = None
            for n, result in enumerate(results()):
                frame_idx = n * stride
                time_s = round(frame_idx / fps, 3) if fps > 0 else 0.0
                frame = result.orig_img
                dets: List[Detection] = []
                boxes = result.obb if result.obb is not None else result.boxes   # aerial models give rotated boxes
                if boxes is not None and boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    ids = boxes.id.cpu().numpy().astype(int)
                    names = [self.classes.get(int(c), str(c)) for c in cls_ids]
                    if hasattr(boxes, "xywhr"):     # overhead: car vs truck by length, not by the model's guess
                        wh = boxes.xywhr[:, 2:4].cpu().numpy()
                        names, ref_length = size_classes(wh.max(axis=1), wh.min(axis=1), ref_length)
                    for (x1, y1, x2, y2), conf, name, tid in zip(xyxy, confs, names, ids):
                        dets.append(Detection(frame_idx, time_s, int(tid), name,
                                              round(float(conf), 3), float(x1), float(y1), float(x2), float(y2)))
                H = self.motion.update(frame_idx, frame, [(d.x1, d.y1, d.x2, d.y2) for d in dets]) if self.motion else None
                items = memory.update(n, dets, (frame.shape[1], frame.shape[0]))
                self.stats["frames"] += 1
                if any(not i.live for i in items):
                    self.stats["frames_with_carried_boxes"] += 1
                yield FrameResult(frame_idx, time_s, render_frame(frame, items, self.render, frame_idx, time_s), dets, items, H)
                if progress_cb:
                    progress_cb(min(frame_idx + stride, total_frames), total_frames)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
