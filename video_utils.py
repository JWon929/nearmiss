"""Video I/O helpers: read metadata, grab frames, write annotated video that plays in a browser."""
from __future__ import annotations

import shutil
import subprocess
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


def read_metadata(video_path: str) -> dict:
    """Return basic metadata for a video file: duration, fps, resolution, frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}. Use an MP4 or MOV (H.264 works best).")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0:                      # some containers do not report a count: count by decoding
        frame_count = 0
        while cap.grab():
            frame_count += 1
    cap.release()
    if fps <= 0 or fps > 240:
        fps = 30.0
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError("The video has no readable frames.")

    return {
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": round(frame_count / fps, 2),
    }


def grab_frame(video_path: str, t_s: float = 0.0) -> np.ndarray:
    """One BGR frame at time t_s (falls back to the first frame)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Could not read a frame from the video.")
    return frame


def iter_frames(video_path: str, stride: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_index, frame) for every `stride`-th frame. Skipped frames are only grabbed, not decoded."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def ffmpeg_exe() -> Optional[str]:
    """Path to an ffmpeg binary: the one bundled with imageio-ffmpeg, else one on the PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class AnnotatedVideoWriter:
    """Writes frames to an MP4 that plays in every browser (H.264 through ffmpeg).

    If ffmpeg is not available it falls back to OpenCV's mp4v, which many browsers cannot play
    (you can still open the file in VLC or QuickTime). `self.codec` says which one was used.
    The size is taken from the first frame, so annotated frames may be smaller than the source.
    """

    def __init__(self, out_path: str, fps: float, width: Optional[int] = None, height: Optional[int] = None):
        self.out_path = out_path
        self.fps = max(float(fps), 1.0)
        self.size: Optional[Tuple[int, int]] = (width, height) if width and height else None
        self.codec = ""
        self._proc: Optional[subprocess.Popen] = None
        self._cv: Optional[cv2.VideoWriter] = None
        self.frames_written = 0

    def _open(self, w: int, h: int) -> None:
        self.size = (w, h)
        exe = ffmpeg_exe()
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{self.fps:.4f}", "-i", "-", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", self.out_path]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self.codec = "h264"
                return
            except OSError:
                self._proc = None
        self._cv = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.codec = "mp4v"

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._proc is None and self._cv is None:
            self._open(w, h)
        if (w, h) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self._proc is not None:
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            except BrokenPipeError:
                err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
                raise RuntimeError("ffmpeg stopped while writing the video: " + err[-300:])
        elif self._cv is not None:
            self._cv.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.wait()
            self._proc = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None
