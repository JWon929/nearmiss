"""
Conflict engine: which pairs of road users had a potential traffic conflict?

For two road users A and B at one instant:

    r = pB - pA            relative position
    v = vB - vA            relative velocity
    d = |r|                separation

Under a short-term constant-velocity assumption the pair is predicted to be
closest at

    t* = -(r . v) / |v|^2          (only meaningful when t* > 0)
    d* = | r + v t* |

Two indicators are computed for every pair; the event severity is the worse:

 1. TTC  - time until the predicted separation first drops to a "collision
           radius" R. Solve |r + v tau| = R for the smallest tau >= 0.
 2. Proximity - the actual minimum separation d_min reached during the encounter.

Also reported (not used to flag in this version): PET, the post-encroachment
time for paths that cross.

Filters against common false alarms:
  * Two tracks whose centres are closer than half their combined widths are one
    road user detected twice, not a collision.
  * A TTC flag must last at least ttc_min_duration_s (heading noise lasts one or two samples).
  * If one road user is stopped (a parked car, a waiting pedestrian), only the other one
    could avoid the collision. A TTC flag then needs evidence: the mover braked hard, or
    they really came close. A car that drives on at the same speed and passes by was never
    on a collision course; the straight-line prediction was wrong (curved road, another lane).

This is a SCREENING measurement. It does not predict crashes, assign fault, or
replace a traffic engineer. Positions come from box bottom-centers, so expect
roughly 0.3-1 m of noise; thresholds are tunable in Settings.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import HALF_WIDTH_M, KIND_LABEL, SEVERITY_LEVELS, THREE_FEET_M, Settings
from .trajectories import Track

_ELIGIBLE = {("ped", "veh"): "ped-veh", ("bike", "veh"): "bike-veh", ("veh", "veh"): "veh-veh"}


def pair_kind(ga: str, gb: str) -> Optional[str]:
    return _ELIGIBLE.get(tuple(sorted((ga, gb))))


def ttc_to_radius(r: np.ndarray, v: np.ndarray, R: float) -> np.ndarray:
    """Smallest tau >= 0 with |r + v tau| = R, per row. 0 if already inside, inf if never."""
    a = (v * v).sum(axis=1)
    b = 2.0 * (r * v).sum(axis=1)
    c = (r * r).sum(axis=1) - R * R
    ttc = np.full(len(a), np.inf)
    inside = c <= 0
    ttc[inside] = 0.0
    disc = b * b - 4.0 * a * c
    ok = (~inside) & (a > 1e-9) & (b < 0) & (disc >= 0)
    ttc[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2.0 * a[ok])
    return ttc


def closest_approach(r: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(t*, d*) for each row under constant velocity. t* is nan where the pair has no relative motion."""
    vv = (v * v).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = -(r * v).sum(axis=1) / vv
    ts = np.where(vv > 1e-9, ts, np.nan)
    ds = np.linalg.norm(r + v * np.nan_to_num(ts)[:, None], axis=1)
    return ts, ds


def _long_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only runs of True that are at least min_len samples long."""
    out = np.zeros_like(mask)
    idx = np.flatnonzero(mask)
    if len(idx):
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run) >= min_len:
                out[run] = True
    return out


def _peak_decel(track: Track, k: int, horizon_steps: int, dt: float, min_span_s: float = 0.5) -> float:
    """Hardest average braking (m/s^2) of `track` from grid step k over the next horizon_steps.
    Averaged over at least min_span_s, because speed noise over one step looks like hard braking."""
    i = k - track.k0
    seg = track.speed[i : i + horizon_steps + 1]
    first = max(1, int(round(min_span_s / dt)))
    if len(seg) <= first:
        return 0.0
    steps = np.arange(first, len(seg))
    return max(0.0, float(np.max((seg[0] - seg[first:]) / (steps * dt))))


def _level_from_ttc(ttc: float, S: Settings) -> int:
    if ttc <= S.ttc_serious_s:
        return 3
    if ttc <= S.ttc_moderate_s:
        return 2
    if ttc <= S.ttc_candidate_s:
        return 1
    return 0


def _level_from_prox(d: float, thr: List[float]) -> int:
    cand, mod, ser = thr
    if d <= ser:
        return 3
    if d <= mod:
        return 2
    if d <= cand:
        return 1
    return 0


def post_encroachment_time(A: Track, B: Track, R: float, min_angle_deg: float = 30.0) -> Optional[float]:
    """Time gap between the two users passing the point where their paths nearly meet.
    Only for crossing paths (headings differ by >= 30 deg); None if the paths never come within R."""
    D = np.linalg.norm(A.xy[:, None, :] - B.xy[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(D), D.shape)
    if D[i, j] > R:
        return None
    va, vb = A.v[i], B.v[j]
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 0.5 or nb < 0.5:
        return None
    cosang = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    if np.degrees(np.arccos(cosang)) < min_angle_deg:
        return None
    return float(abs(A.t[i] - B.t[j]))


def _analyze_pair(A: Track, B: Track, kind: str, S: Settings, dt: float, dropped: Dict[str, int]) -> List[Dict]:
    k0, k1 = max(A.k0, B.k0), min(A.k1, B.k1)
    if k1 - k0 + 1 < 3:
        return []
    sa = slice(k0 - A.k0, k1 - A.k0 + 1)
    sb = slice(k0 - B.k0, k1 - B.k0 + 1)
    pA, pB, vA, vB = A.xy[sa], B.xy[sb], A.v[sa], B.v[sb]
    spA, spB = A.speed[sa], B.speed[sb]
    r, v = pB - pA, vB - vA
    d = np.linalg.norm(r, axis=1)
    vmag = np.linalg.norm(v, axis=1)
    R = S.collision_radius_m[kind]
    prox_thr = S.proximity_m[kind]

    # cheap reject: even closing at full relative speed for the TTC horizon, could they get close?
    if (d - S.ttc_candidate_s * vmag).min() > max(R, prox_thr[0]):
        return []

    # one road user detected twice: bodies would overlap by more than half
    overlap_d = 0.5 * (HALF_WIDTH_M[A.cls] + HALF_WIDTH_M[B.cls])
    if np.median(d) < overlap_d or (d < overlap_d).mean() > 0.3:
        if (d <= prox_thr[0]).any():
            dropped["duplicate_detection"] += 1
        return []

    ttc = ttc_to_radius(r, v, R)
    with np.errstate(divide="ignore", invalid="ignore"):
        closing = -(r * v).sum(axis=1) / np.maximum(d, 1e-6)

    # gates that remove obvious non-conflicts
    gate = (spA <= S.max_speed_mps) & (spB <= S.max_speed_mps)
    if kind == "veh-veh":
        gate &= np.maximum(spA, spB) >= S.min_vehicle_speed_mps
    else:
        vehicle_speed = spA if A.group == "veh" else spB
        gate &= vehicle_speed >= S.min_vehicle_speed_mps
    valid_ttc = gate & (closing >= S.min_closing_speed_mps) & (ttc <= S.ttc_candidate_s)
    raw_ttc = valid_ttc.copy()
    valid_ttc = _long_runs(valid_ttc, max(1, int(round(S.ttc_min_duration_s / dt))))
    stopped = np.minimum(spA, spB) < S.stopped_mps
    valid_prox = gate & (d <= prox_thr[0])
    if kind == "veh-veh":
        valid_prox &= ~stopped   # driving past a parked or queued car is routine
    valid = valid_ttc | valid_prox
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return []

    merge_steps = max(1, int(round(S.merge_gap_s / dt)))
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > merge_steps) + 1)
    events = []
    for run in runs:
        lo, hi = int(run[0]), int(run[-1])
        if closing[run].max() < S.min_closing_speed_mps:
            continue  # never actually approached each other
        seg_d = d[lo : hi + 1]
        ip = lo + int(np.argmin(seg_d))
        d_min = float(d[ip])
        vt = valid_ttc[run]
        ttc_min = float(ttc[run][vt].min()) if vt.any() else float("inf")
        lvl_t = _level_from_ttc(ttc_min, S) if np.isfinite(ttc_min) else 0
        braked = None
        if lvl_t:
            it = int(run[vt][np.argmin(ttc[run][vt])])
            if stopped[it]:
                mover = A if spA[it] >= spB[it] else B
                horizon = int(np.ceil((ttc_min + 1.0) / dt))
                decel = _peak_decel(mover, k0 + it, horizon, dt)
                came_close = d[it : it + horizon + 1].min() <= R
                if decel >= S.evasive_decel_mps2:
                    braked = (mover.cls, decel)
                elif not came_close:
                    lvl_t = 0
                    dropped["no_evasive_action"] += 1
        lvl_p = _level_from_prox(d_min, prox_thr) if valid_prox[run].any() else 0
        level = max(lvl_t, lvl_p)
        if level == 0:
            if raw_ttc[run].any() and not vt.any():
                dropped["ttc_too_brief"] += 1
            continue

        # vehicle first in the pair
        if kind == "veh-veh":
            a_first = bool(spA[ip] >= spB[ip])  # faster vehicle listed first
        else:
            a_first = A.group == "veh"
        X, Y = (A, B) if a_first else (B, A)
        spX, spY = (spA[ip], spB[ip]) if a_first else (spB[ip], spA[ip])
        vX, vY = (vA[ip], vB[ip]) if a_first else (vB[ip], vA[ip])

        # heading relationship + 3-ft screening for vehicle/bicycle passes
        nX, nY = np.linalg.norm(vX), np.linalg.norm(vY)
        same_dir = None
        clearance = None
        below_3ft = None
        if nX > 1.0 and nY > 0.3:
            cosang = float(np.clip(np.dot(vX, vY) / (nX * nY), -1, 1))
            same_dir = bool(np.degrees(np.arccos(cosang)) <= 30.0)
        if kind == "bike-veh" and same_dir and spX > spY + 1.0:
            clearance = d_min - (HALF_WIDTH_M[X.cls] + HALF_WIDTH_M[Y.cls])
            below_3ft = bool(clearance < THREE_FEET_M)

        reasons = []
        if lvl_t:
            thr = [S.ttc_candidate_s, S.ttc_moderate_s, S.ttc_serious_s][lvl_t - 1]
            reasons.append("Predicted collision in %.1f s (<= %.1f s, %s threshold)" % (ttc_min, thr, SEVERITY_LEVELS[lvl_t]))
        if lvl_p:
            thr = prox_thr[lvl_p - 1]
            reasons.append("Came within %.1f m (<= %.1f m, %s threshold for %s)" % (d_min, thr, SEVERITY_LEVELS[lvl_p], KIND_LABEL[kind].lower()))
        if braked:
            reasons.append("The %s braked at %.1f m/s^2 toward a stopped road user" % braked)
        if below_3ft:
            reasons.append("Estimated body-to-body clearance %.2f m is under 3 ft (%.2f m)" % (clearance, THREE_FEET_M))

        events.append(
            {
                "kind": kind,
                "kind_label": KIND_LABEL[kind],
                "a": {"track": X.id, "seg": X.seg, "cls": X.cls, "speed_mps": round(float(spX), 2)},
                "b": {"track": Y.id, "seg": Y.seg, "cls": Y.cls, "speed_mps": round(float(spY), 2)},
                "t_start": round(float((k0 + lo) * dt), 2),
                "t_peak": round(float((k0 + ip) * dt), 2),
                "t_end": round(float((k0 + hi) * dt), 2),
                "ttc_min_s": None if not np.isfinite(ttc_min) else round(ttc_min, 2),
                "d_min_m": round(d_min, 2),
                "closing_speed_mps": round(float(closing[run].max()), 2),
                "pet_s": None,
                "same_direction": same_dir,
                "clearance_est_m": None if clearance is None else round(float(clearance), 2),
                "below_3ft": below_3ft,
                "x": round(float((pA[ip, 0] + pB[ip, 0]) / 2), 2),
                "y": round(float((pA[ip, 1] + pB[ip, 1]) / 2), 2),
                "severity": SEVERITY_LEVELS[level],
                "severity_level": level,
                "severity_from_ttc": SEVERITY_LEVELS[lvl_t],
                "severity_from_proximity": SEVERITY_LEVELS[lvl_p],
                "reasons": reasons,
                "merged_duplicates": 0,
                "_tracks": (A, B),
            }
        )
    return events


def _dedupe(events: List[Dict], S: Settings) -> List[Dict]:
    """ID switches can split one encounter into several pairs; keep the worst of each cluster."""
    events = sorted(events, key=lambda e: e["t_peak"])
    kept: List[Dict] = []
    for e in events:
        dup = None
        for k in kept:
            if (
                k["kind"] == e["kind"]
                and abs(k["t_peak"] - e["t_peak"]) <= S.dedupe_time_s
                and np.hypot(k["x"] - e["x"], k["y"] - e["y"]) <= S.dedupe_dist_m
            ):
                dup = k
                break
        if dup is None:
            kept.append(e)
            continue
        better = (e["severity_level"], -(e["ttc_min_s"] or 99)) > (dup["severity_level"], -(dup["ttc_min_s"] or 99))
        n = dup["merged_duplicates"] + 1
        if better:
            kept[kept.index(dup)] = e
            e["merged_duplicates"] = n
        else:
            dup["merged_duplicates"] = n
    return kept


def detect_conflicts(tracks: List[Track], S: Settings, stats: Optional[Dict] = None) -> List[Dict]:
    """Flagged events. If `stats` is given, it receives counts of candidate flags each filter removed."""
    dt = 1.0 / S.analysis_hz
    events: List[Dict] = []
    dropped = {"duplicate_detection": 0, "ttc_too_brief": 0, "no_evasive_action": 0}
    for i in range(len(tracks)):
        A = tracks[i]
        for j in range(i + 1, len(tracks)):
            B = tracks[j]
            if A.id == B.id:
                continue
            if min(A.k1, B.k1) - max(A.k0, B.k0) < 2:
                continue
            kind = pair_kind(A.group, B.group)
            if kind is None:
                continue
            events.extend(_analyze_pair(A, B, kind, S, dt, dropped))
    if stats is not None:
        stats.update({"filtered_" + k: v for k, v in dropped.items()})
    events = _dedupe(events, S)
    for e in events:
        A, B = e.pop("_tracks")
        e["pet_s"] = post_encroachment_time(A, B, S.collision_radius_m[e["kind"]])
        if e["pet_s"] is not None:
            e["pet_s"] = round(e["pet_s"], 2)
    events.sort(key=lambda e: e["t_peak"])
    for n, e in enumerate(events, 1):
        e["id"] = "E%03d" % n
    return events
