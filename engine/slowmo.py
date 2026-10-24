"""
Slow-motion near-miss video.

Takes the annotated (already blurred) video and slows it down around every flagged event, so a viewer can see what
happened. Frames inside an event window are written `factor` times; the in-between copies are blended with the next
frame, so motion stays smooth instead of stuttering. A banner names the event while it plays slowly.

    make_slowmo(..., highlights_only=False)  the whole clip, normal speed except the slowed events
    make_slowmo(..., highlights_only=True)   only the slowed events, one after another (a highlight reel)

Uses only OpenCV plus the app's H.264 writer, so no extra install is needed.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SEV_BGR = {"candidate": (0, 200, 255), "moderate": (0, 140, 255), "serious": (60, 60, 235)}


def slow_windows(events: Sequence[Dict], before_s: float, after_s: float) -> List[Tuple[float, float, List[Dict]]]:
    """Time windows to slow down, one per event, merged where they overlap."""
    spans = sorted((max(0.0, e["t_start"] - before_s), e["t_end"] + after_s, e) for e in events)
    out: List[Tuple[float, float, List[Dict]]] = []
    for t0, t1, e in spans:
        if out and t0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], t1), out[-1][2] + [e])
        else:
            out.append((t0, t1, [e]))
    return out


def _banner(img: np.ndarray, evs: List[Dict], factor: int) -> np.ndarray:
    worst = max(evs, key=lambda e: e.get("severity_level", 0))
    col = SEV_BGR.get(worst["severity"], (0, 200, 255))
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, max(4, w // 160))
    text = f"SLOW MOTION 1/{factor}x   " + "  ".join(f"{e['id']} {e['kind_label']} ({e['severity']})" for e in evs[:2])
    scale = max(0.5, w / 1600)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    y = h - 12
    cv2.rectangle(img, (0, y - th - 14), (min(w, tw + 24), h), (0, 0, 0), -1)
    cv2.putText(img, text, (12, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
    return img


def make_slowmo(src: str, out: str, events: Sequence[Dict], factor: int = 4, before_s: float = 2.0,
                after_s: float = 1.0, highlights_only: bool = False,
                progress: Optional[Callable[[float], None]] = None) -> Dict:
    """Write the slowed video to `out`. Returns what was done (windows, length in seconds, codec)."""
    from video_utils import AnnotatedVideoWriter

    factor = max(1, int(factor))
    windows = slow_windows(events, before_s, after_s)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError(f"Could not open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    writer = AnnotatedVideoWriter(out, fps)
    ok, cur = cap.read()
    i, w_idx = 0, 0
    try:
        while ok:
            ok_next, nxt = cap.read()
            t = i / fps
            while w_idx < len(windows) and t > windows[w_idx][1]:
                w_idx += 1
            win = windows[w_idx] if w_idx < len(windows) and windows[w_idx][0] <= t <= windows[w_idx][1] else None
            if win is None:
                if not highlights_only:
                    writer.write(cur)
            else:
                for k in range(factor):
                    a = k / factor
                    frame = cur if a == 0 or not ok_next else cv2.addWeighted(cur, 1 - a, nxt, a, 0)
                    writer.write(_banner(frame.copy(), win[2], factor))
            if progress and i % 20 == 0:
                progress(min(1.0, i / total))
            cur, ok, i = nxt, ok_next, i + 1
    finally:
        cap.release()
        writer.release()
    if progress:
        progress(1.0)
    return {"windows": len(windows), "events": sum(len(w[2]) for w in windows),
            "seconds": round(writer.frames_written / fps, 1), "codec": writer.codec}
