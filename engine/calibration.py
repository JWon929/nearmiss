"""
Scene calibration: image pixels -> ground-plane meters.

Idea
----
A camera looking at a flat road sees the road through a *homography*: a 3x3
matrix H with  p_ground ~ H * p_image  (in homogeneous coordinates).
Four point correspondences determine H. The user clicks the four corners of a
rectangle on the road whose real size they know (a crosswalk, a painted box, a
lane section measured on an aerial map) and types its two side lengths.

Ground frame (meters):  P1=(0,0)  P2=(w,0)  P3=(w,l)  P4=(0,l)
where w = |P1P2| and l = |P2P3|.

IMPORTANT: one known distance is NOT enough to fix a homography from four
arbitrary points. A rectangle needs *two* lengths (width and length).

"Map" frame: the ground frame rotated so that "up on screen" in the camera view
is "up" in the top-down map. It is a rigid rotation, so distances are unchanged.
All trajectories, events and plots use map meters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


class CalibrationError(ValueError):
    """The clicked points / lengths cannot define a usable calibration."""


def _signed_area(pts: np.ndarray) -> float:
    """Shoelace area with image coordinates (y down): > 0 means clockwise on screen."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _is_convex(pts: np.ndarray) -> bool:
    signs = []
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-9:
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)


@dataclass
class Calibration:
    image_points: np.ndarray   # (4,2) full-resolution pixels, normalized to clockwise order
    width_m: float             # |P1P2|
    length_m: float            # |P2P3|
    H: np.ndarray              # 3x3 image -> ground
    rot: np.ndarray            # 2x2 ground -> map rotation
    frame_size: Tuple[int, int]  # (width, height) of the video frame used for clicking

    # ------------------------------------------------------------------ build
    @classmethod
    def from_points(
        cls,
        points: List[List[float]],
        width_m: float,
        length_m: float,
        frame_size: Tuple[int, int],
    ) -> "Calibration":
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if pts.shape != (4, 2):
            raise CalibrationError("Exactly four points are required.")
        if not (width_m > 0 and length_m > 0):
            raise CalibrationError("Both real-world lengths must be positive numbers.")
        if not _is_convex(pts):
            raise CalibrationError(
                "The four points must be the corners of a rectangle, clicked in order "
                "around its edge (no crossing lines)."
            )
        # Normalise to clockwise-on-screen so the top-down map is not mirrored.
        if _signed_area(pts) < 0:
            pts = pts[[0, 3, 2, 1]]
            width_m, length_m = length_m, width_m

        ground = np.array(
            [[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]], dtype=np.float64
        )
        H = cv2.getPerspectiveTransform(pts.astype(np.float32), ground.astype(np.float32))
        H = np.asarray(H, dtype=np.float64)
        # getPerspectiveTransform works in float32 internally; polish in float64 with a
        # direct linear solve so tiny rounding does not leak into distances.
        H = _solve_homography(pts, ground)

        cal = cls(
            image_points=pts,
            width_m=float(width_m),
            length_m=float(length_m),
            H=H,
            rot=np.eye(2),
            frame_size=(int(frame_size[0]), int(frame_size[1])),
        )
        cal._w_sign = cal._centroid_w_sign()
        cal.rot = cal._compute_up_rotation()
        return cal

    @classmethod
    def from_two_points(cls, p1, p2, distance_m: float, frame_size: Tuple[int, int]) -> "Calibration":
        """Calibration from ONE known distance. Only valid for a camera looking straight down, where
        the image is a scaled, rotated copy of the road (no perspective). A square is built on the
        segment p1-p2 and handed to from_points."""
        a, b = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
        d = b - a
        if np.hypot(*d) < 5:
            raise CalibrationError("The two points are too close together. Pick two points farther apart.")
        if not distance_m > 0:
            raise CalibrationError("The real-world distance must be a positive number.")
        n = np.array([-d[1], d[0]])
        return cls.from_points([a, b, b + n, a + n], distance_m, distance_m, frame_size)

    # --------------------------------------------------------------- mapping
    def _centroid_w_sign(self) -> float:
        c = self.image_points.mean(axis=0)
        w = (self.H @ np.array([c[0], c[1], 1.0]))[2]
        return 1.0 if w > 0 else -1.0

    def img_to_ground(self, pts) -> Tuple[np.ndarray, np.ndarray]:
        """Pixels -> ground meters. Returns (xy, valid). Invalid = at/over the horizon."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        ph = np.c_[p, np.ones(len(p))] @ self.H.T
        w = ph[:, 2]
        valid = (np.sign(w) == self._w_sign) & (np.abs(w) > 1e-9)
        safe = np.where(valid, w, 1.0)
        xy = ph[:, :2] / safe[:, None]
        xy[~valid] = np.nan
        return xy, valid

    def ground_to_map(self, g) -> np.ndarray:
        g = np.asarray(g, dtype=np.float64).reshape(-1, 2)
        return g @ self.rot.T

    def img_to_map(self, pts) -> Tuple[np.ndarray, np.ndarray]:
        g, valid = self.img_to_ground(pts)
        return self.ground_to_map(g), valid

    @property
    def center_map(self) -> np.ndarray:
        return self.ground_to_map([[self.width_m / 2.0, self.length_m / 2.0]])[0]

    def _compute_up_rotation(self) -> np.ndarray:
        """Rotation that makes 'up on screen' point up (-y) on the top-down map."""
        c = self.image_points.mean(axis=0)
        g, ok = self.img_to_ground([c, c + np.array([0.0, -20.0])])
        if not ok.all():
            return np.eye(2)
        d = g[1] - g[0]
        phi = math.atan2(d[1], d[0])
        alpha = -math.pi / 2.0 - phi
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array([[ca, -sa], [sa, ca]])

    # ----------------------------------------------------------- measurement
    def distance_between(self, p1, p2) -> float:
        g, ok = self.img_to_ground([p1, p2])
        if not ok.all():
            raise CalibrationError("A point lies at or above the horizon of the calibrated plane.")
        return float(np.linalg.norm(g[1] - g[0]))

    def pixel_scale_at_center(self) -> float:
        """Approximate pixels per meter near the calibrated rectangle (resolution check)."""
        c = self.image_points.mean(axis=0)
        g0, _ = self.img_to_ground([c])
        # step 1 m in ground x and find pixel distance by inverting H
        Hinv = np.linalg.inv(self.H)

        def to_img(g):
            q = Hinv @ np.array([g[0], g[1], 1.0])
            return q[:2] / q[2]

        a, b = to_img(g0[0]), to_img(g0[0] + np.array([1.0, 0.0]))
        return float(np.linalg.norm(b - a))

    def quality_hints(self) -> List[str]:
        """Plain-language warnings about calibration quality."""
        hints: List[str] = []
        fw, fh = self.frame_size
        area = abs(_signed_area(self.image_points))
        frac = area / float(fw * fh)
        if frac < 0.03:
            hints.append(
                "The rectangle covers only %.1f%% of the frame. A larger rectangle spread over the "
                "area you care about gives a more accurate scale." % (frac * 100)
            )
        scale = self.pixel_scale_at_center()
        if scale < 4:
            hints.append(
                "About %.1f pixels per meter at the calibration box: positions will be coarse. "
                "Use higher-resolution footage or a closer camera." % scale
            )
        ratio = max(self.width_m, self.length_m) / min(self.width_m, self.length_m)
        if ratio > 12:
            hints.append("The rectangle is very long and thin; a squarer one is better conditioned.")
        return hints

    # ------------------------------------------------------------- top-down
    def map_transform(self, origin_xy, px_per_m: float) -> np.ndarray:
        """3x3 matrix: image pixel -> top-down raster pixel (for warpPerspective)."""
        rot3 = np.eye(3)
        rot3[:2, :2] = self.rot
        tr = np.eye(3)
        tr[0, 2], tr[1, 2] = -origin_xy[0], -origin_xy[1]
        sc = np.diag([px_per_m, px_per_m, 1.0])
        return sc @ tr @ rot3 @ self.H

    def preview(self, frame_bgr: np.ndarray, margin: float = 0.6, max_px: int = 900) -> np.ndarray:
        """Bird's-eye preview of one frame. Lane lines should look parallel and the
        calibration rectangle should look like a true rectangle."""
        corners_map = self.ground_to_map(
            [[0, 0], [self.width_m, 0], [self.width_m, self.length_m], [0, self.length_m]]
        )
        lo, hi = corners_map.min(axis=0), corners_map.max(axis=0)
        span = hi - lo
        pad = margin * max(span)
        lo, hi = lo - pad, hi + pad
        px_per_m = max_px / float(max(hi - lo))
        w_px, h_px = int(math.ceil((hi[0] - lo[0]) * px_per_m)), int(math.ceil((hi[1] - lo[1]) * px_per_m))
        M = self.map_transform(lo, px_per_m)
        out = cv2.warpPerspective(frame_bgr, M, (w_px, h_px), flags=cv2.INTER_LINEAR)
        # draw the calibration rectangle
        quad = ((corners_map - lo) * px_per_m).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [quad], True, (0, 220, 255), 2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict:
        return {
            "image_points": self.image_points.tolist(),
            "width_m": self.width_m,
            "length_m": self.length_m,
            "H": self.H.tolist(),
            "rot": self.rot.tolist(),
            "frame_size": list(self.frame_size),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Calibration":
        cal = cls(
            image_points=np.array(d["image_points"], dtype=np.float64),
            width_m=float(d["width_m"]),
            length_m=float(d["length_m"]),
            H=np.array(d["H"], dtype=np.float64),
            rot=np.array(d["rot"], dtype=np.float64),
            frame_size=tuple(d["frame_size"]),
        )
        cal._w_sign = cal._centroid_w_sign()
        return cal


def _solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct Linear Transform from exactly 4 correspondences (h33 fixed to 1)."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64))
    return np.append(h, 1.0).reshape(3, 3)
