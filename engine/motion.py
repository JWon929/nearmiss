"""
Camera-motion compensation (for drones and cameras that drift, zoom or wobble).

The calibration maps pixels of ONE frame to metres. If the camera moves, a parked car slides across
the image, looks like it is driving, and every position after the first second is off. So for each
processed frame we estimate a homography that maps its pixels onto the FIRST (reference) frame:

    1. Find ORB corner features in a downscaled grey copy of the frame, ignoring the boxes of
       detected road users (they move on their own and would pull the estimate with them).
    2. Match them with the reference frame's features and fit a homography with RANSAC, which
       ignores the matches that disagree (other moving things, rooftops with parallax).
    3. If too few features match the reference (the view has changed a lot), match the previous
       frame instead and chain the two transforms.

The road is assumed flat, so a homography is the right model for points on it. Tall buildings do
not fit the model, which is why RANSAC is used. Detections stay in raw pixels in detections.csv;
the analysis applies the transform to each ground point.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

MIN_INLIERS = 40


class CameraMotion:
    def __init__(self, work_width: int = 960, n_features: int = 3000):
        self.work_width = work_width
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.scale = 1.0
        self._ref = None            # (keypoints, descriptors) of the reference frame
        self._prev = None           # (keypoints, descriptors, H_prev_to_ref)
        self.H: Dict[int, np.ndarray] = {}
        self.inliers: Dict[int, int] = {}
        self.lost: List[int] = []   # frames where no transform could be found (previous one reused)
        self.chained: List[int] = []  # frames matched to the previous frame only (small errors add up)

    def _features(self, frame: np.ndarray, boxes: Sequence[Sequence[float]]):
        h, w = frame.shape[:2]
        self.scale = min(1.0, self.work_width / float(w))
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            pad = 0.15 * max(x2 - x1, y2 - y1)
            cv2.rectangle(mask, (int((x1 - pad) * self.scale), int((y1 - pad) * self.scale)),
                          (int((x2 + pad) * self.scale), int((y2 + pad) * self.scale)), 0, -1)
        return self.orb.detectAndCompute(gray, mask)

    def _match(self, a, b) -> Optional[tuple]:
        """Homography mapping frame `a` points onto frame `b` (downscaled pixels), plus inlier count."""
        (ka, da), (kb, db) = a, b
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return None
        pairs = self.matcher.knnMatch(da, db, k=2)
        good = [m for m, *n in pairs if n and m.distance < 0.75 * n[0].distance]
        if len(good) < MIN_INLIERS:
            return None
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or int(mask.sum()) < MIN_INLIERS:
            return None
        return H, int(mask.sum())

    def update(self, frame_idx: int, frame: np.ndarray, boxes: Sequence[Sequence[float]] = ()) -> np.ndarray:
        """Return (and remember) the 3x3 full-resolution homography: this frame's pixels -> reference pixels."""
        feats = self._features(frame, boxes)
        if self._ref is None:
            self._ref = feats
            self._prev = (*feats, np.eye(3))
            self.H[frame_idx], self.inliers[frame_idx] = np.eye(3), len(feats[0])
            return self.H[frame_idx]
        H_small, n = None, 0
        direct = self._match(feats, self._ref)
        if direct is not None:
            H_small, n = direct
        else:
            step = self._match(feats, self._prev[:2])
            if step is not None:
                H_small, n = self._prev[2] @ step[0], step[1]
                self.chained.append(frame_idx)
        if H_small is None:
            H_small = self._prev[2]
            self.lost.append(frame_idx)
        self._prev = (*feats, H_small)
        S = np.diag([self.scale, self.scale, 1.0])
        H = np.linalg.inv(S) @ H_small @ S
        self.H[frame_idx], self.inliers[frame_idx] = H / H[2, 2], n
        return self.H[frame_idx]

    def summary(self, frame_size) -> Dict:
        """How far the image wandered: largest move of a frame corner or the centre (reference pixels).
        Corners matter because a drone that climbs or descends zooms about the centre."""
        w, h = frame_size
        probe = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1], [w / 2.0, h / 2.0, 1]], dtype=np.float64)
        shifts = []
        for H in self.H.values():
            p = probe @ H.T
            shifts.append(float(np.max(np.hypot(*(p[:, :2] / p[:, 2:3] - probe[:, :2]).T))))
        return {"frames": len(self.H), "max_shift_px": round(max(shifts, default=0.0), 1),
                "lost_frames": len(self.lost), "chained_frames": len(self.chained),
                "direct_frames": len(self.H) - len(self.lost) - len(self.chained),
                "median_inliers": int(np.median(list(self.inliers.values()))) if self.inliers else 0}

    def to_dict(self, frame_size) -> Dict:
        return {"reference_frame": min(self.H) if self.H else 0, "summary": self.summary(frame_size),
                "H": {str(k): np.round(v, 9).tolist() for k, v in self.H.items()}}


def load_motion(d: Optional[Dict]) -> Optional[Dict[int, np.ndarray]]:
    """motion.json -> {frame: 3x3}. None if the camera was treated as fixed."""
    if not d or not d.get("H"):
        return None
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in d["H"].items()}


def to_reference(points: np.ndarray, frames: np.ndarray, motion: Optional[Dict[int, np.ndarray]]) -> np.ndarray:
    """Map each (x, y) pixel seen on frames[i] into reference-frame pixels. Frames without a
    transform use the nearest frame that has one."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not motion:
        return points
    keys, stack = _stacked(motion)
    frames = np.asarray(frames)
    # nearest frame with a transform: compare the neighbours on each side of the insertion point
    j = np.clip(np.searchsorted(keys, frames), 1, len(keys) - 1) if len(keys) > 1 else np.zeros(len(frames), int)
    if len(keys) > 1:
        j = np.where(np.abs(keys[j - 1] - frames) <= np.abs(keys[j] - frames), j - 1, j)
    ph = np.einsum("nij,nj->ni", stack[j], np.c_[points, np.ones(len(points))])
    return ph[:, :2] / ph[:, 2:3]


_STACK_CACHE: Dict[int, tuple] = {}


def _stacked(motion: Dict[int, np.ndarray]):
    """Sorted frame numbers and their transforms as one (K, 3, 3) array, built once per motion table."""
    key = id(motion)
    hit = _STACK_CACHE.get(key)
    if hit is None or hit[0] is not motion:
        keys = np.array(sorted(motion))
        hit = (motion, keys, np.stack([motion[int(k)] for k in keys]))
        _STACK_CACHE.clear()                  # keep one: a table is reused for every track of one recording
        _STACK_CACHE[key] = hit
    return hit[1], hit[2]
