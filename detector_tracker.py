"""
Detection + multi-object tracking for NearMiss Phase 1.

Uses a PRETRAINED YOLOv8 model (COCO classes) — this is disclosed to the user in
app.py per the build plan's requirement that pretrained detection be clearly stated.
Tracking uses Ultralytics' built-in ByteTrack integration to assign persistent IDs.
"""

from dataclasses import dataclass

import cv2
from ultralytics import YOLO

# COCO class ids we care about for road users, per the build plan's required set.
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    5: "bus",
    7: "truck",
}

BOX_COLOR = (0, 200, 255)  # BGR
TEXT_COLOR = (255, 255, 255)

MODEL_NAME = "yolov8n.pt"  # small/fast pretrained model; auto-downloads on first run


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


class RoadUserTracker:
    """Runs YOLOv8 + ByteTrack frame-by-frame over a video, yielding Detections
    and an annotated (boxes + IDs drawn) copy of each frame."""

    def __init__(self, model_name: str = MODEL_NAME, conf_threshold: float = 0.35):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def process_video(self, video_path: str, fps: float, progress_cb=None):
        """
        Generator yielding (frame_index, annotated_frame_bgr, list[Detection]) per frame.
        progress_cb(frame_idx, total_frames) is called if provided, for UI progress bars.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        # Ultralytics' .track() with stream=True processes lazily and keeps track
        # state (ByteTrack) across calls on the same model instance.
        results_stream = self.model.track(
            source=video_path,
            classes=list(TARGET_CLASSES.keys()),
            conf=self.conf_threshold,
            persist=True,
            stream=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        for result in results_stream:
            frame = result.orig_img.copy()
            time_s = frame_idx / fps if fps > 0 else 0.0
            detections = []

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cls_id, tid in zip(
                    xyxy, confs, cls_ids, track_ids
                ):
                    class_name = TARGET_CLASSES.get(int(cls_id), str(cls_id))
                    det = Detection(
                        frame=frame_idx,
                        time_s=round(time_s, 3),
                        track_id=int(tid),
                        class_name=class_name,
                        confidence=round(float(conf), 3),
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    )
                    detections.append(det)
                    _draw_box(frame, det)

            yield frame_idx, frame, detections

            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx, total_frames)

        cap.release()


def _draw_box(frame, det: Detection):
    p1 = (int(det.x1), int(det.y1))
    p2 = (int(det.x2), int(det.y2))
    cv2.rectangle(frame, p1, p2, BOX_COLOR, 2)
    label = f"{det.class_name} #{det.track_id} {det.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), BOX_COLOR, -1)
    cv2.putText(
        frame, label, (p1[0] + 2, p1[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )
