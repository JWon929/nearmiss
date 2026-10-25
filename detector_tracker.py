"""
Detection + multi-object tracking for NearMiss.

Detection uses Ultralytics YOLO models, disclosed in the app:
  * street-level cameras: pretrained YOLOv8 (COCO classes: person, bicycle, car, motorcycle, bus, truck);
  * drone footage: pretrained YOLO11-OBB (DOTA aerial photos, vehicles only, rotated boxes), or
    nearmiss-visdrone.pt, a YOLO11s fine-tuned on the VisDrone dataset in tools/train_visdrone_colab.ipynb,
    which also finds pedestrians and bikes from above.
Tracking uses Ultralytics' built-in ByteTrack.

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
import queue
import shutil
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
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
    """Overhead models to offer. The stock aerial model stays the default: on our test clip it kept car/truck
    labels steady (1 flip vs 11) and called only the 3 real trucks trucks (vs 7). The VisDrone-trained model
    comes second when its weights are in the app folder: it tracked more road users and is the only overhead
    model that finds pedestrians and bikes."""
    choices = dict(AERIAL_CHOICES)
    if (HERE / VISDRONE_MODEL).exists():
        first = next(iter(choices))
        choices = {first: choices.pop(first),
                   VISDRONE_MODEL: "overhead / drone, trained on VisDrone: ALSO finds pedestrians and bikes "
                                   "(car/truck labels less steady)", **choices}
    return choices


# Small road users from above are easy to confuse with street furniture: the VisDrone model called lamp posts,
# signal heads and utility boxes "pedestrian" or "motorcycle" on our test clip. None of those ever moves, so from
# above a person / bike box is only drawn once its road user has moved this many times its own box size.
SMALL_CLASSES = ("person", "bicycle", "motorcycle")
CONFIRM_MOVE_BOXES = 3.0      # same rule as Settings.static_clutter_boxes in the analysis

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
    batch: int = 8                # frames detected per model call (GPUs are much faster with a batch)
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
    enabled: bool = True          # False: no tracked video during detection (fastest); event clips are drawn later


def auto_stride(fps: float, target_fps: float = 12.0) -> int:
    """Process about `target_fps` frames per second: plenty for walking and driving speeds."""
    return max(1, int(round(fps / target_fps))) if fps > 0 else 1


_INTEL: Optional[List[str]] = None


def intel_devices() -> List[str]:
    """OpenVINO devices on this computer ("GPU", "NPU", "CPU"), or [] without OpenVINO. Checked once."""
    global _INTEL
    if _INTEL is None:
        try:
            import openvino as ov

            _INTEL = [d.split(".")[0] for d in ov.Core().available_devices]
        except Exception:
            _INTEL = []
    return _INTEL


def pick_device(preference: str = "auto") -> str:
    """auto: an NVIDIA GPU, else an Apple chip, else an Intel GPU through OpenVINO, else the CPU.
    PyTorch cannot use Intel GPUs; OpenVINO ran YOLOv8n about 4x faster on an Arc 130V than PyTorch on the CPU."""
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
    if "GPU" in intel_devices():
        return "intel:gpu"
    return "cpu"


def device_label(device: str) -> str:
    return {"cpu": "CPU", "mps": "Apple chip", "cuda:0": "NVIDIA GPU", "intel:gpu": "Intel GPU (OpenVINO)",
            "intel:npu": "Intel NPU (OpenVINO)", "intel:cpu": "CPU (OpenVINO)"}.get(device, device)


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


def input_shape(imgsz: int, frame_wh: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """(height, width) the model sees: the long side is `imgsz`, the short side keeps the video's shape, rounded up to
    a multiple of 32. A 16:9 video at 960 becomes 544 x 960 instead of a 960 x 960 square that is 43% padding, the
    same shape PyTorch picks itself, so boxes match and the Intel GPU does ~40% less work."""
    if not frame_wh:
        return imgsz, imgsz
    w, h = frame_wh
    s = imgsz / max(w, h)
    pad = lambda v: int(np.ceil(v * s / 32) * 32)
    return (imgsz, pad(w)) if h >= w else (pad(h), imgsz)


def openvino_dir(name: str, shape: Tuple[int, int], batch: int = 1) -> Path:
    """Where the OpenVINO copy of a model is cached (one per input shape and batch: an export's shape is fixed)."""
    return HERE / "openvino_cache" / f"{Path(name).stem}_{shape[0]}x{shape[1]}_b{batch}_openvino_model"  # suffix needed


def _load_openvino(name: str, shape: Tuple[int, int], batch: int = 1):
    """The OpenVINO version of `name` for this input shape and batch size, exported on first use (a few seconds)
    and cached. FP16 weights: as accurate as FP32 here and a little faster on Intel GPUs and NPUs."""
    from ultralytics import YOLO

    target = openvino_dir(name, shape, batch)
    if not (target / "metadata.yaml").exists():
        torch_model, used, _ = _load_model(name)
        exported = Path(torch_model.export(format="openvino", imgsz=list(shape), half=True, batch=batch,
                                           verbose=False))
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(exported), str(target))
    import yaml                                  # comes with ultralytics

    # class names straight from the export's metadata: asking the model for .names would first compile it on the CPU
    names = {int(k): v for k, v in yaml.safe_load((target / "metadata.yaml").read_text())["names"].items()}
    return YOLO(str(target), task="obb" if "-obb" in name else "detect"), names


class FastOpenVINO:
    """Batched detection with the image conversion done on the Intel GPU.

    Ultralytics turns every frame into 32-bit floats on the CPU before inference; at 960 px that cost 12 ms per frame,
    twice the GPU's own 6 ms. Here the frames go to OpenVINO as they are (8-bit BGR, one letterbox per frame), and an
    OpenVINO pre-processing step converts, reorders the colour channels and scales them on the GPU. Everything after
    inference (non-max suppression with Ultralytics' defaults, box scaling, Results) is Ultralytics' own code, so the
    boxes match predict() (tests/test_fast_openvino.py)."""

    def __init__(self, folder: Path, device: str, names: Dict[int, str], task: str, shape: Tuple[int, int],
                 batch: int):
        import openvino as ov
        from ultralytics.data.augment import LetterBox

        core = ov.Core()
        model = core.read_model(str(next(Path(folder).glob("*.xml"))))
        ppp = ov.preprocess.PrePostProcessor(model)
        ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")) \
            .set_color_format(ov.preprocess.ColorFormat.BGR)
        ppp.input().preprocess().convert_element_type(ov.Type.f32) \
            .convert_color(ov.preprocess.ColorFormat.RGB).scale(255.0)
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        self.compiled = core.compile_model(ppp.build(), device.split(":", 1)[1].upper(), {"PERFORMANCE_HINT": "LATENCY"})
        self.request = self.compiled.create_infer_request()
        self.output = self.compiled.output(0)
        self.names, self.task, self.shape, self.batch = names, task, tuple(shape), int(batch)
        self.letterbox = LetterBox(list(shape), auto=False, stride=32)

    def __call__(self, frames: List[np.ndarray], conf: float, classes: List[int]):
        import torch
        from ultralytics.engine.results import Results
        from ultralytics.utils import nms, ops

        n = len(frames)
        padded = list(frames) + [frames[-1]] * (self.batch - n)       # the model takes exactly `batch` frames
        imgs = np.stack([self.letterbox(image=f) for f in padded])
        out = self.request.infer({0: imgs})[self.output]
        preds = nms.non_max_suppression(torch.from_numpy(np.asarray(out)), conf, 0.7, classes, False, max_det=300,
                                        nc=0 if self.task == "detect" else len(self.names),
                                        rotated=self.task == "obb")
        results = []
        for pred, orig in zip(preds[:n], frames):
            if self.task == "obb":
                rboxes = torch.cat([pred[:, :4], pred[:, -1:]], dim=-1)
                rboxes[:, :4] = ops.scale_boxes(self.shape, rboxes[:, :4], orig.shape, xywh=True)
                results.append(Results(orig, path="", names=self.names, obb=torch.cat([rboxes, pred[:, 4:6]], dim=-1)))
            else:
                pred[:, :4] = ops.scale_boxes(self.shape, pred[:, :4], orig.shape)
                results.append(Results(orig, path="", names=self.names, boxes=pred[:, :6]))
        return results


class RoadUserTracker:
    """YOLO + ByteTrack over a video. Yields one FrameResult per processed frame."""

    def __init__(self, settings: Optional[TrackerSettings] = None, render: Optional[RenderOptions] = None):
        self.s = settings or TrackerSettings()
        self.render = render or RenderOptions()
        self.model, self.model_used, self.warning = _load_model(self.s.model_name)
        self.device_used = pick_device(self.s.device)
        self.classes = class_map(self.model.names)       # reading names from the PyTorch weights compiles nothing
        self.shape = (self.s.imgsz, self.s.imgsz)
        self.fast = None
        self.motion = None
        self.stats = {"frames": 0, "frames_with_carried_boxes": 0}
        self.timing = {"model": 0.0, "motion": 0.0, "render": 0.0}   # seconds spent in each stage (tools/benchmark.py)

    def _prepare_intel(self, frame_wh: Tuple[int, int]) -> None:
        """Swap in the OpenVINO model once the video's shape is known; on any problem stay on the CPU."""
        if not self.device_used.startswith("intel:"):
            return
        self.shape = input_shape(self.s.imgsz, frame_wh)
        try:
            self.model, names = _load_openvino(self.model_used, self.shape, max(1, int(self.s.batch)))
            self.classes = class_map(names)
            try:
                self.fast = FastOpenVINO(openvino_dir(self.model_used, self.shape, max(1, int(self.s.batch))),
                                         self.device_used, names, "obb" if "-obb" in self.model_used else "detect",
                                         self.shape, max(1, int(self.s.batch)))
            except Exception:
                self.fast = None                     # Ultralytics' own (slower) path still works
        except Exception as ex:
            self.warning = (self.warning + " " if self.warning else "") + \
                f"The fast {device_label(self.device_used)} model could not be prepared ({type(ex).__name__}); " \
                "running on the CPU instead."
            self.device_used = "cpu"
            self.shape = (self.s.imgsz, self.s.imgsz)

    def _detect(self, frames: List[np.ndarray]):
        """Detect road users on several frames in one call: the GPU is much better used with a batch (on the Intel
        Arc 130V, 12 ms per frame in batches of 8 against 18 ms one by one). Tracking happens afterwards, in order."""
        intel = self.device_used.startswith("intel:")
        if intel and getattr(self, "fast", None) is not None:
            t0 = time.perf_counter()
            res = self.fast(frames, self.s.detector_conf, list(self.classes))
            self.timing["fast_openvino"] = self.timing.get("fast_openvino", 0.0) + time.perf_counter() - t0
            return res
        imgsz = list(self.shape) if intel else self.s.imgsz
        n = len(frames)
        if intel and n < self.s.batch:                 # the exported model takes exactly `batch` frames: pad the last
            frames = list(frames) + [frames[-1]] * (self.s.batch - n)
        res = self.model.predict(frames, classes=list(self.classes), conf=self.s.detector_conf, imgsz=imgsz,
                                 device=self.device_used, verbose=False)[:n]
        if res:                                       # Ultralytics' own split, ms per image (tools/benchmark.py)
            for k in ("preprocess", "inference", "postprocess"):
                self.timing[k] = self.timing.get(k, 0.0) + res[0].speed.get(k, 0.0) / 1000 * n
        return res

    def _new_tracker(self, yaml_path: str):
        """ByteTrack fed one frame at a time with the batch's detections, exactly as Ultralytics' own track() does."""
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace

        cfg = IterableSimpleNamespace(**YAML.load(yaml_path))
        try:
            return BYTETracker(args=cfg, frame_rate=30)     # older Ultralytics versions take frame_rate (30 = no scaling)
        except TypeError:
            return BYTETracker(args=cfg)

    @staticmethod
    def _apply_tracks(result, tracker):
        """result -> the same result keeping only tracked boxes, with their track IDs (ultralytics/trackers/track.py)."""
        import torch

        is_obb = result.obb is not None
        src = result.obb if is_obb else result.boxes
        tracks = tracker.update(src.cpu().numpy(), result.orig_img)
        if len(tracks) == 0:
            return result[:0]
        out = result[tracks[:, -1].astype(int)]
        out.update(**{"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1], device=src.data.device)})
        return out

    def process_video(self, video_path: str, fps: float, progress_cb: Optional[Callable[[int, int], None]] = None,
                      start_frame: int = 0, id_offset: int = 0) -> Iterator[FrameResult]:
        """Yield one FrameResult per processed frame (every `stride`-th frame from `start_frame`), in order.

        Three things run at once: a thread decodes the next frames, the model detects a batch of `batch` frames, and
        a small thread pool draws and blurs the previous batch's output frames. Tracking stays in frame order.
        `start_frame` and `id_offset` resume an interrupted run: ByteTrack starts over at ID 1, so its IDs are shifted
        past the ones already saved. Camera motion is always measured against frame 0, as in an uninterrupted run."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        self._prepare_intel(frame_wh)
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
        B = max(1, int(self.s.batch))
        reader = FrameReader(video_path, stride, start_frame, ahead=2 * B + 2)
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nearmiss-draw")
        pending: Deque[Tuple[FrameResult, "Future"]] = deque()
        self.timing.setdefault("track", 0.0)

        def draw(frame, items, frame_idx, time_s):
            t0 = time.perf_counter()
            img = render_frame(frame, items, self.render, frame_idx, time_s)
            self.timing["render"] += time.perf_counter() - t0      # summed over the drawing threads
            return img

        def emit():
            fr, fut = pending.popleft()
            fr.annotated = fut.result() if fut is not None else None
            if progress_cb:
                progress_cb(min(fr.frame_idx + stride, total_frames), total_frames)
            return fr

        try:
            self.motion = CameraMotion() if self.s.stabilize else None
            if self.motion is not None and start_frame > 0:
                self.motion.update(0, grab_first_frame(video_path))   # the same reference frame as the first part
            tracker = self._new_tracker(tmp.name)
            ref_length = None
            hide_still = is_aerial(self.model_used) and finds_people(self.model_used)
            first_seen: Dict[int, Tuple[float, float]] = {}     # small road user -> first centre (reference pixels)
            confirmed = set()
            n = 0
            for batch in _batches(reader, B):
                clock = time.perf_counter()
                try:
                    results = self._detect([f for _, f in batch])
                except Exception as ex:
                    if self.device_used == "cpu" or n > 0:
                        raise
                    self.warning = (self.warning + " " if self.warning else "") + \
                        f"{device_label(self.device_used)} failed ({type(ex).__name__}); running on the CPU instead."
                    self.device_used = "cpu"
                    self.fast = None
                    self.model, _, _ = _load_model(self.model_used)
                    self.classes = class_map(self.model.names)
                    results = self._detect([f for _, f in batch])
                self.timing["model"] += time.perf_counter() - clock
                for (frame_idx, frame), result in zip(batch, results):
                    clock = time.perf_counter()
                    result = self._apply_tracks(result, tracker)
                    self.timing["track"] += time.perf_counter() - clock
                    dets, items, H, ref_length, time_s = self._frame(
                        result, frame_idx, frame, fps, id_offset, n, memory, hide_still, first_seen, confirmed,
                        ref_length)
                    fut = pool.submit(draw, frame, items, frame_idx, time_s) if self.render.enabled else None
                    pending.append((FrameResult(frame_idx, time_s, None, dets, items, H), fut))
                    n += 1
                while len(pending) > B:       # the last batch keeps drawing while the model works on the next one
                    yield emit()
            while pending:
                yield emit()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            reader.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _frame(self, result, frame_idx, frame, fps, id_offset, n, memory, hide_still, first_seen, confirmed,
               ref_length):
        """Detections, camera motion and the boxes to draw for one tracked frame."""
        time_s = round(frame_idx / fps, 3) if fps > 0 else 0.0
        dets: List[Detection] = []
        boxes = result.obb if result.obb is not None else result.boxes   # aerial models give rotated boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) + id_offset
            names = [self.classes.get(int(c), str(c)) for c in cls_ids]
            if hasattr(boxes, "xywhr"):     # overhead: car vs truck by length, not by the model's guess
                wh = boxes.xywhr[:, 2:4].cpu().numpy()
                names, ref_length = size_classes(wh.max(axis=1), wh.min(axis=1), ref_length)
            for (x1, y1, x2, y2), conf, name, tid in zip(xyxy, confs, names, ids):
                dets.append(Detection(frame_idx, time_s, int(tid), name,
                                      round(float(conf), 3), float(x1), float(y1), float(x2), float(y2)))
        t = time.perf_counter()
        H = self.motion.update(frame_idx, frame, [(d.x1, d.y1, d.x2, d.y2) for d in dets]) if self.motion else None
        self.timing["motion"] += time.perf_counter() - t
        items = memory.update(n, dets, (frame.shape[1], frame.shape[0]))
        if hide_still:
            for d in dets:
                if d.class_name not in SMALL_CLASSES or d.track_id in confirmed:
                    continue
                c = np.array([d.center])
                if H is not None:
                    c = cv2.perspectiveTransform(c[None], H)[0]
                start = first_seen.setdefault(d.track_id, tuple(c[0]))
                size = max(d.x2 - d.x1, d.y2 - d.y1)
                if np.hypot(c[0][0] - start[0], c[0][1] - start[1]) > CONFIRM_MOVE_BOXES * size:
                    confirmed.add(d.track_id)
            items = [i for i in items if i.class_name not in SMALL_CLASSES or i.track_id in confirmed]
        self.stats["frames"] += 1
        if any(not i.live for i in items):
            self.stats["frames_with_carried_boxes"] += 1
        return dets, items, H, ref_length, time_s


def _batches(it, size: int):
    """Group an iterator into lists of up to `size` items."""
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def render_from_detections(video_path: str, det, out_path: str, fps: float, stride: int, render: RenderOptions,
                           t0: float = 0.0, t1: Optional[float] = None, coast_s: float = 1.0, trail_s: float = 2.0,
                           progress: Optional[Callable[[float], None]] = None) -> int:
    """Draw the tracked video (blur, boxes, trails) from the original video and saved detections, for [t0, t1] or
    the whole clip. Used when detection ran without drawing (fast mode): event clips are drawn when opened, and the
    full tracked video only if asked for. Returns the number of frames written."""
    from video_utils import AnnotatedVideoWriter

    stride = max(1, int(stride))
    start = max(0, int(np.floor(t0 * fps / stride)) * stride)
    stop = int(np.ceil(t1 * fps)) if t1 is not None else None
    by_frame: Dict[int, List[Detection]] = {}
    names = {"pedestrian": "person"}                  # the analysis says pedestrian, the drawing colours use person
    for i in range(len(det)):
        f = int(det.frame[i])
        if f < start - 3 * int(coast_s * fps) or (stop is not None and f > stop):
            continue
        x1, y1, x2, y2 = (float(v) for v in det.box[i])
        by_frame.setdefault(f, []).append(Detection(f, float(det.t[i]), int(det.tid[i]),
                                                     names.get(str(det.cls[i]), str(det.cls[i])),
                                                     float(det.conf[i]), x1, y1, x2, y2))
    pfps = fps / stride
    memory = PersistentTracks(int(round(coast_s * pfps)), int(round(trail_s * pfps)))
    # warm the memory up with the frames just before t0, so boxes and trails are already there at the start
    for n, f in enumerate(range(max(0, start - 3 * int(coast_s * fps) // stride * stride), start, stride)):
        memory.update(n, by_frame.get(f, []), (1, 1))
    n0 = len(range(max(0, start - 3 * int(coast_s * fps) // stride * stride), start, stride))
    writer = AnnotatedVideoWriter(out_path, pfps)
    reader = FrameReader(video_path, stride, start)
    written = 0
    try:
        for k, (f, frame) in enumerate(reader):
            if stop is not None and f > stop:
                break
            items = memory.update(n0 + k, by_frame.get(f, []), (frame.shape[1], frame.shape[0]))
            writer.write(render_frame(frame, items, render, f, round(f / fps, 3)))
            written += 1
            if progress and stop:
                progress(min(1.0, (f - start) / max(stop - start, 1)))
    finally:
        reader.close()
        writer.release()
    return written


def grab_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read the first frame of {video_path}")
    return frame


class FrameReader:
    """Decodes every `stride`-th frame from `start` on a background thread, a few frames ahead of the detector.

    Skipped frames are only grabbed (not decoded into images), which is most of the cost of reading. Iterating yields
    (frame index in the original video, BGR frame)."""

    def __init__(self, path: str, stride: int = 1, start: int = 0, ahead: int = 6):
        self.path, self.stride, self.start = path, max(1, int(stride)), max(0, int(start))
        self._q: "queue.Queue" = queue.Queue(maxsize=ahead)
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self.path}")
            idx = 0
            if self.start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
                idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if idx != self.start:                      # inexact seek: start over and grab forward
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    idx = 0
                    while idx < self.start and cap.grab():
                        idx += 1
            while not self._stop.is_set():
                if not cap.grab():
                    break
                if (idx - self.start) % self.stride == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    self._put((idx, frame))
                idx += 1
        except BaseException as ex:                        # handed to the consumer, never lost on the thread
            self._error = ex
        finally:
            cap.release()
            self._put(None)

    def _put(self, item) -> None:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                if self._error is not None:
                    raise self._error
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)
