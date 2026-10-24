"""
Privacy blurring for anything shown or exported.

NearMiss measures conflicts, not people. We never need faces or plates, so:
  * event frames and the top-down background are blurred before they are saved
  * reports contain no raw video frames at all

HOW IT WORKS (and its limits - say this in your Limitations section):
We do not run a face or plate detector. We pixelate the REGIONS where faces and
plates usually are, inside boxes the road-user detector found:
    pedestrian  -> top 30% of the box (head)
    vehicle     -> bottom 40% of the box (plate band) and the upper 45% (windshield)
    "strict"    -> the whole box
Anything the detector misses is NOT blurred. Event frames are also downscaled,
which removes small detail, but treat the raw video as sensitive and delete it
when you are done (Settings -> "Delete source video").
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def _pixelate(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    bw, bh = max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def regions_for(cls: str, box: Sequence[float], level: str = "standard") -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = 0.08 * max(w, h)
    if level == "strict":
        return [(int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y2 + pad))]
    if cls == "pedestrian":
        return [(int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y1 + 0.30 * h + pad))]
    if cls in ("car", "bus", "truck", "motorcycle"):
        return [
            (int(x1 - pad), int(y2 - 0.40 * h - pad), int(x2 + pad), int(y2 + pad)),
            (int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y1 + 0.45 * h)),
        ]
    return []  # bicycles: the rider is its own "pedestrian" detection


def blur_detections(img: np.ndarray, dets: Iterable[Tuple[str, Sequence[float]]], level: str = "standard") -> np.ndarray:
    """In-place pixelation of face/plate regions for (cls, box) pairs. Returns img."""
    for cls, box in dets:
        for r in regions_for(cls, box, level):
            _pixelate(img, *r)
    return img
