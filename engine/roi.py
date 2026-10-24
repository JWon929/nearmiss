"""
Road area (region of interest).

The user outlines the part of the image where conflicts can happen (the roadway and crosswalks). Ground points
outside it (parked cars in a lot, people on a sidewalk, a car park next door) are ignored, so they cannot create
false events. The polygon is in pixels of the reference frame: the first frame when the camera moves, which is the
frame that camera-motion compensation lines everything up with.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

MIN_POINTS = 3


def as_polygon(points: Optional[Sequence[Sequence[float]]]) -> Optional[np.ndarray]:
    """(n,2) float array, or None when there are too few points to enclose an area."""
    if not points or len(points) < MIN_POINTS:
        return None
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def inside(pts: np.ndarray, poly: Optional[np.ndarray]) -> np.ndarray:
    """Even-odd ray casting for many points at once. Everything is inside when there is no polygon."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if poly is None:
        return np.ones(len(pts), dtype=bool)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    hit = crosses & (x < x_at)
    return (hit.sum(axis=1) % 2) == 1
