"""Video I/O helpers: read metadata, write annotated frames back out."""

import cv2


def read_metadata(video_path: str) -> dict:
    """Return basic metadata for a video file: duration, fps, resolution, frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": round(duration_s, 2),
    }


class AnnotatedVideoWriter:
    """Thin wrapper around cv2.VideoWriter so app.py doesn't touch fourcc/codec details."""

    def __init__(self, out_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, max(fps, 1.0), (width, height))
        self.out_path = out_path

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
