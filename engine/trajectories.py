"""
Detections -> clean ground-plane trajectories.

Steps
-----
1. Take each box's bottom-center (where the road user touches the ground).
2. Map it from pixels to map meters with the calibration homography.
3. Drop points that are out of range or over the horizon.
4. Split tracks at long detection gaps; resample to a uniform time grid.
5. Estimate position AND velocity with a local linear regression over a short
   window. We do NOT differentiate frame to frame: a pixel of detector jitter
   becomes a huge speed spike that way (noise is amplified by 1/dt).
6. Remove riders that the detector reports as a separate "person" on a bicycle.
7. Drop "pedestrians" and "bikes" that never move: street furniture the model mistook for one.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .calibration import Calibration
from .config import CLASS_GROUP, Settings
from .motion import to_reference
from .roi import inside as roi_inside

CSV_HEADER = ["frame", "t", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]


# ----------------------------------------------------------------- detections
class Detections:
    """Detections table in memory (numpy columns)."""

    def __init__(self, rows: List[tuple]):
        n = len(rows)
        self.frame = np.array([r[0] for r in rows], dtype=np.int64) if n else np.zeros(0, np.int64)
        self.t = np.array([r[1] for r in rows], dtype=np.float64) if n else np.zeros(0)
        self.tid = np.array([r[2] for r in rows], dtype=np.int64) if n else np.zeros(0, np.int64)
        self.cls = np.array([r[3] for r in rows], dtype=object) if n else np.zeros(0, object)
        self.conf = np.array([r[4] for r in rows], dtype=np.float64) if n else np.zeros(0)
        self.box = np.array([r[5:9] for r in rows], dtype=np.float64).reshape(-1, 4) if n else np.zeros((0, 4))

    def __len__(self) -> int:
        return len(self.frame)

    def track_ids(self) -> List[int]:
        return sorted(set(int(i) for i in self.tid))

    def rows_of(self, tid: int) -> np.ndarray:
        return np.flatnonzero(self.tid == tid)

    def track_class(self, tid: int) -> str:
        idx = self.rows_of(tid)
        score: Dict[str, float] = {}
        for c, p in zip(self.cls[idx], self.conf[idx]):
            score[c] = score.get(c, 0.0) + float(p)
        return max(score, key=score.get)


def save_detections(path: Path, rows: List[tuple]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", r[2], r[3], f"{r[4]:.3f}"] + [f"{v:.1f}" for v in r[5:9]])


def load_detections(path: Path) -> Detections:
    rows = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(
                (int(r["frame"]), float(r["t"]), int(r["track_id"]), r["cls"], float(r["conf"]),
                 float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            )
    return Detections(rows)


# --------------------------------------------------------------------- tracks
@dataclass
class Track:
    id: int
    seg: int
    cls: str
    group: str
    k0: int                 # index of first sample on the global time grid (t = k * dt)
    t: np.ndarray           # (n,) seconds
    xy: np.ndarray          # (n,2) smoothed position, map meters
    v: np.ndarray           # (n,2) velocity m/s
    speed: np.ndarray       # (n,) m/s
    n_dets: int

    @property
    def k1(self) -> int:
        return self.k0 + len(self.t) - 1

    @property
    def path_length_m(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))) if len(self.t) > 1 else 0.0

    @property
    def name(self) -> str:
        return "#%d" % self.id


def local_linear(t: np.ndarray, p: np.ndarray, half: int) -> Tuple[np.ndarray, np.ndarray]:
    """Position and velocity at every sample from a least-squares line over +/- `half` samples."""
    n = len(t)
    pos = np.empty_like(p)
    vel = np.zeros_like(p)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        m = hi - lo
        if m < 3:
            pos[i] = p[i]
            continue
        tt = t[lo:hi] - t[i]
        pp = p[lo:hi]
        st, stt = tt.sum(), (tt * tt).sum()
        den = m * stt - st * st
        if den <= 1e-12:
            pos[i] = p[i]
            continue
        sp = pp.sum(axis=0)
        stp = (tt[:, None] * pp).sum(axis=0)
        b = (m * stp - st * sp) / den
        a = (sp - b * st) / m
        pos[i], vel[i] = a, b
    return pos, vel


def _resample(ts: np.ndarray, ps: np.ndarray, dt: float) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
    k_first = int(math.ceil(ts[0] / dt - 1e-9))
    k_last = int(math.floor(ts[-1] / dt + 1e-9))
    if k_last - k_first < 2:
        return None
    ks = np.arange(k_first, k_last + 1)
    tg = ks * dt
    x = np.interp(tg, ts, ps[:, 0])
    y = np.interp(tg, ts, ps[:, 1])
    return k_first, tg, np.c_[x, y]


# ---------------------------------------------------------------------- riders
def _iomin(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    amin = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / amin if amin > 0 else 0.0


def find_rider_tracks(det: Detections) -> Set[int]:
    """A cyclist is detected twice (person + bicycle). Mark person tracks that ride a bike/motorcycle."""
    persons: Dict[int, Dict[int, np.ndarray]] = {}
    rides: Dict[int, Dict[int, np.ndarray]] = {}
    for tid in det.track_ids():
        cls = det.track_class(tid)
        if cls not in ("pedestrian", "bicycle", "motorcycle"):
            continue
        idx = det.rows_of(tid)
        table = {int(f): b for f, b in zip(det.frame[idx], det.box[idx])}
        (persons if cls == "pedestrian" else rides)[tid] = table
    riders: Set[int] = set()
    for pid, pf in persons.items():
        for bid, bf in rides.items():
            common = pf.keys() & bf.keys()
            if len(common) < max(3, 0.5 * len(pf)):
                continue
            hits = sum(1 for f in common if _iomin(pf[f], bf[f]) >= 0.35)
            if hits >= 0.6 * len(common):
                riders.add(pid)
                break
    return riders


# ----------------------------------------------------------------- build tracks
def ground_pixels(boxes: np.ndarray, mode: str = "bottom") -> np.ndarray:
    """Pixel where each box touches the road (see Settings.ground_point)."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    y = (boxes[:, 1] + boxes[:, 3]) / 2.0 if mode == "center" else boxes[:, 3]
    return np.c_[cx, y]


STILL_CLASSES = ("pedestrian", "bicycle", "motorcycle")


def is_still(boxes: np.ndarray, frames: np.ndarray, motion: Optional[Dict[int, np.ndarray]], max_boxes: float) -> bool:
    """True when a track's box centre never moves more than `max_boxes` of its own (median) box size, in
    reference-frame pixels. Scale-free, so a wrong calibration cannot hide or create movement."""
    centres = to_reference(np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], frames, motion)
    size = float(np.median(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])))
    return float(np.linalg.norm(np.ptp(centres, axis=0))) < max_boxes * max(size, 1.0)


def likely_id_switches(tracks: List[Track], max_gap_s: float = 1.0, max_dist_m: float = 3.0) -> int:
    """Tracks that end where another track of the same kind starts a moment later: most likely one road user
    whose ID changed (the tracker lost it and started a new track). An estimate, not a count of real switches."""
    n = 0
    for a in tracks:
        end_t, end_p = a.t[-1], a.xy[-1]
        for b in tracks:
            if b is a or b.group != a.group:
                continue
            gap = b.t[0] - end_t
            if 0 < gap <= max_gap_s and np.linalg.norm(b.xy[0] - (end_p + a.v[-1] * gap)) <= max_dist_m:
                n += 1
                break
    return n


def build_tracks(det: Detections, calib: Calibration, S: Settings,
                 motion: Optional[Dict[int, np.ndarray]] = None,
                 roi: Optional[np.ndarray] = None) -> Tuple[List[Track], Dict]:
    """`motion` (frame -> homography onto the first frame) undoes camera movement before calibrating.
    `roi` (polygon in reference-frame pixels, see engine/roi.py) drops ground points outside the road area."""
    dt = 1.0 / S.analysis_hz
    half = max(1, int(round(S.smooth_window_s / dt / 2.0)))
    riders = find_rider_tracks(det)
    stats = {"raw_tracks": len(det.track_ids()), "dropped_riders": len(riders), "dropped_short": 0,
             "dropped_out_of_range": 0, "dropped_outside_roi": 0, "dropped_static_clutter": 0, "track_breaks": 0}
    tracks: List[Track] = []
    center = calib.center_map
    for tid in det.track_ids():
        if tid in riders:
            continue
        idx = det.rows_of(tid)
        order = np.argsort(det.t[idx])
        idx = idx[order]
        cls = det.track_class(tid)
        boxes = det.box[idx]
        foot = to_reference(ground_pixels(boxes, S.ground_point), det.frame[idx], motion)
        on_road = roi_inside(foot, roi)
        if not on_road.any():
            stats["dropped_outside_roi"] += 1
            continue
        mp, valid = calib.img_to_map(foot)
        valid &= on_road
        rng = np.linalg.norm(mp - center, axis=1)
        valid = valid & np.isfinite(rng) & (rng <= S.max_range_m)
        if valid.sum() < S.min_hits:
            stats["dropped_out_of_range" if len(idx) >= S.min_hits else "dropped_short"] += 1
            continue
        if S.static_clutter_boxes > 0 and cls in STILL_CLASSES and is_still(boxes, det.frame[idx], motion,
                                                                            S.static_clutter_boxes):
            stats["dropped_static_clutter"] += 1      # never moved: a pole or a box, not a person
            continue
        t, mp = det.t[idx][valid], mp[valid]
        cuts = [0] + list(np.flatnonzero(np.diff(t) > S.max_gap_s) + 1) + [len(t)]
        made = False
        for seg, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            ts, ps = t[a:b], mp[a:b]
            if len(ts) < S.min_hits or ts[-1] - ts[0] < S.min_track_s:
                continue
            res = _resample(ts, ps, dt)
            if res is None:
                continue
            k0, tg, pg = res
            pos, vel = local_linear(tg, pg, half)
            speed = np.linalg.norm(vel, axis=1)
            tracks.append(Track(tid, seg, cls, CLASS_GROUP[cls], k0, tg, pos, vel, speed, len(ts)))
            stats["track_breaks"] += made
            made = True
        if not made:
            stats["dropped_short"] += 1
    stats["tracks"] = len(tracks)
    stats["likely_id_switches"] = likely_id_switches(tracks)
    stats["median_track_s"] = round(float(np.median([t.t[-1] - t.t[0] for t in tracks])), 1) if tracks else 0.0
    return tracks, stats
