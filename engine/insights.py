"""
Extra measurements built on the tracks and events the engine already computes.

  * explain_event       one plain-language sentence per event, for readers who are not traffic engineers
  * speed_summary       vehicle speed distribution (85th percentile, share over a posted limit)
  * sensitivity         how the event count changes when a threshold moves (the thresholds are tunable, not magic)
  * tuning_curve        precision of your OWN labelled events at stricter cutoffs, and a suggested cutoff
  * vehicle_size_check  median car length in metres from overhead boxes: a hint that the scale is off

None of these is a safety standard. Tuning in particular describes your clips only.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from .calibration import Calibration
from .config import KIND_LABEL, Settings
from .conflicts import detect_conflicts
from .motion import to_reference
from .trajectories import Detections, Track

MPS_TO_MPH = 2.23694
NOUN = {"pedestrian": "pedestrian", "bicycle": "cyclist", "motorcycle": "motorcycle", "car": "car", "bus": "bus",
        "truck": "truck"}


def _a(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def _speed(mps: float) -> str:
    return f"{mps:.1f} m/s, {mps * MPS_TO_MPH:.0f} mph"


# ------------------------------------------------------------------ explanations
def explain_event(e: Dict) -> str:
    """'A car (9.3 m/s, 21 mph) and a pedestrian (1.4 m/s, 3 mph) came within 1.4 m ...'"""
    a, b = NOUN.get(e["a"]["cls"], e["a"]["cls"]), NOUN.get(e["b"]["cls"], e["b"]["cls"])
    if e.get("flag_source") == "pet":           # never close at the same instant: only their paths crossed
        return (f"{_a(a).capitalize()} ({_speed(e['a']['speed_mps'])}) and {_a(b)} ({_speed(e['b']['speed_mps'])}) "
                f"used the same crossing point {e['pet_s']:.1f} s apart, at {e['t_peak']:.1f} s into the video: the "
                "second one arrived just after the first one cleared it.")
    s = (f"{_a(a).capitalize()} ({_speed(e['a']['speed_mps'])}) and {_a(b)} ({_speed(e['b']['speed_mps'])}) came "
         f"within {e['d_min_m']:.1f} m of each other at {e['t_peak']:.1f} s into the video")
    if e.get("closing_speed_mps"):
        s += f", closing at up to {e['closing_speed_mps']:.1f} m/s"
    s += "."
    if e.get("ttc_min_s") is not None and e["ttc_min_s"] < 0.05:
        s += " At one point they were already inside the collision distance the check allows for."
    elif e.get("ttc_min_s") is not None:
        s += (f" If neither had slowed or turned, they would have been within touching distance in "
              f"{e['ttc_min_s']:.1f} s.")
    if e.get("drac_max_mps2"):
        s += (f" Avoiding it needed about {e['drac_max_mps2']:.1f} m/s² of braking"
              + (" (harder than comfortable braking)." if e["drac_max_mps2"] >= 3.35 else "."))
    if e.get("pet_s") is not None:
        s += f" Their paths crossed, and the second one reached the crossing point {e['pet_s']:.1f} s after the first left it."
    if e.get("below_3ft"):
        s += " The estimated gap while passing the cyclist was under 3 feet."
    return s


# ------------------------------------------------------------------ speeds
def speed_summary(tracks: Sequence[Track], S: Settings, limit_mps: Optional[float] = None) -> Dict:
    """One speed per vehicle: the 85th percentile of its own moving samples (its cruising speed, not a spike).
    Then the distribution over vehicles, as in a spot-speed study."""
    per_id: Dict[int, List[np.ndarray]] = {}
    for t in tracks:
        if t.group == "veh":
            per_id.setdefault(t.id, []).append(t.speed)
    speeds = []
    for parts in per_id.values():
        sp = np.concatenate(parts)
        moving = sp[(sp >= S.stopped_mps) & (sp <= S.max_speed_mps)]
        if len(moving) >= 5:
            speeds.append(float(np.percentile(moving, 85)))
    v = np.array(speeds)
    out = {"vehicles": len(v), "speeds_mps": [round(x, 2) for x in speeds]}
    if not len(v):
        return out
    out.update(mean_mps=round(float(v.mean()), 2), median_mps=round(float(np.median(v)), 2),
               p85_mps=round(float(np.percentile(v, 85)), 2), max_mps=round(float(v.max()), 2))
    if limit_mps:
        out["limit_mps"] = limit_mps
        out["share_over_limit"] = round(float((v > limit_mps).mean()), 3)
        out["share_over_limit_by_5mph"] = round(float((v > limit_mps + 5 / MPS_TO_MPH).mean()), 3)
    return out


# ------------------------------------------------------------------ sensitivity
def _with_candidate_ttc(S: Settings, ttc: float) -> Settings:
    mod = min(S.ttc_moderate_s, ttc)
    return replace(S, ttc_candidate_s=ttc, ttc_moderate_s=mod, ttc_serious_s=min(S.ttc_serious_s, mod))


def _with_proximity_scale(S: Settings, k: float) -> Settings:
    return replace(S, proximity_m={kind: [round(v * k, 3) for v in thr] for kind, thr in S.proximity_m.items()})


def sensitivity(tracks: List[Track], S: Settings, ttc_values: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
                prox_scales: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5)) -> Dict[str, List[Dict]]:
    """Event counts (total and per kind) when one threshold changes and everything else stays as in S."""
    def row(name, value, S2):
        ev = detect_conflicts(tracks, S2)
        r = {name: value, "events": len(ev), "current": False}
        r.update({KIND_LABEL[k]: sum(1 for e in ev if e["kind"] == k) for k in KIND_LABEL})
        return r

    ttc_rows = [row("ttc_candidate_s", v, _with_candidate_ttc(S, v)) for v in ttc_values]
    prox_rows = [row("proximity_scale", k, _with_proximity_scale(S, k)) for k in prox_scales]
    for r in ttc_rows:
        r["current"] = abs(r["ttc_candidate_s"] - S.ttc_candidate_s) < 1e-9
    for r in prox_rows:
        r["current"] = abs(r["proximity_scale"] - 1.0) < 1e-9
    return {"ttc": ttc_rows, "proximity": prox_rows}


# ------------------------------------------------------------------ tuning from labels
MIN_LABELLED = 10


def tuning_curve(events: Sequence[Dict], S: Settings, keep_real: float = 0.8) -> Dict:
    """Would a stricter cutoff have kept your real near misses and dropped your false alarms?

    Uses only events you labelled real or false alarm. Two one-dimensional scans:
      * TTC: keep an event if its min TTC <= cutoff (events flagged only by distance are unaffected)
      * distance: keep an event if it came within scale x the candidate distance for its kind
        (events flagged by TTC are unaffected)
    The suggestion is the cutoff with the best precision that still keeps `keep_real` of the real ones."""
    lab = [e for e in events if e.get("label") in ("tp", "fp")]
    n_tp = sum(e["label"] == "tp" for e in lab)
    out = {"labelled": len(lab), "real": n_tp, "enough": len(lab) >= MIN_LABELLED and n_tp >= 3,
           "ttc": [], "distance": [], "suggest_ttc": None, "suggest_scale": None,
           "caveat": "Tuned on your labelled clips only. It is not a general standard and may not carry over to "
                     "other intersections, cameras or lighting."}
    if not lab:
        return out

    def scan(values, keep, name):
        rows = []
        for x in values:
            kept = [e for e in lab if keep(e, x)]
            tp = sum(e["label"] == "tp" for e in kept)
            fp = len(kept) - tp
            rows.append({name: round(float(x), 2), "kept_real": tp, "kept_false": fp,
                         "precision": round(tp / len(kept), 3) if kept else None,
                         "real_kept_share": round(tp / n_tp, 3) if n_tp else None})
        return rows

    def by_ttc(e, x):
        return e.get("ttc_min_s") is None or e["ttc_min_s"] <= x

    def by_dist(e, k):
        if e.get("severity_from_ttc", "none") != "none":
            return True
        return e["d_min_m"] <= k * S.proximity_m[e["kind"]][0]

    out["ttc"] = scan(np.arange(0.5, S.ttc_candidate_s + 1e-9, 0.25), by_ttc, "ttc_cutoff_s")
    out["distance"] = scan(np.arange(0.4, 1.0001, 0.1), by_dist, "distance_scale")

    def best(rows, name):
        ok = [r for r in rows if r["precision"] is not None and (r["real_kept_share"] or 0) >= keep_real]
        if not ok:
            return None
        top = max(r["precision"] for r in ok)
        return max(r[name] for r in ok if r["precision"] == top)      # loosest cutoff among the best

    if out["enough"]:
        out["suggest_ttc"] = best(out["ttc"], "ttc_cutoff_s")
        out["suggest_scale"] = best(out["distance"], "distance_scale")
    return out


# ------------------------------------------------------------------ calibration hint from car size
TYPICAL_CAR_M = (3.3, 6.0)      # most cars are 4-5 m long; the box of a turning car is a little longer


def vehicle_size_check(det: Detections, calib: Calibration, S: Settings,
                       motion: Optional[Dict[int, np.ndarray]] = None, max_samples: int = 3000) -> Optional[Dict]:
    """Overhead footage only: the median long side of car boxes, in metres. A median far from 4-5 m means the scale
    is probably off (wrong typed length, points clicked on the wrong marks). A hint, not a measurement."""
    if S.ground_point != "center":
        return None
    idx = np.flatnonzero(det.cls == "car")
    if len(idx) < 20:
        return None
    idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), max_samples)).astype(int)]
    b, f = det.box[idx], det.frame[idx]
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    ends = [np.c_[b[:, 0], cy], np.c_[b[:, 2], cy], np.c_[cx, b[:, 1]], np.c_[cx, b[:, 3]]]
    g = [calib.img_to_ground(to_reference(p, f, motion))[0] for p in ends]
    sides = np.c_[np.linalg.norm(g[1] - g[0], axis=1), np.linalg.norm(g[3] - g[2], axis=1)]
    long_side = sides.max(axis=1)
    long_side = long_side[np.isfinite(long_side)]
    if not len(long_side):
        return None
    med = float(np.median(long_side))
    lo, hi = TYPICAL_CAR_M
    warning = None
    if not lo <= med <= hi:
        warning = (f"Cars measure {med:.1f} m long on average with this calibration; real cars are about 4-5 m. "
                   f"The scale may be off by about {abs(med / 4.5 - 1):.0%}: re-check the typed length and the clicked "
                   "points. (This is a hint from typical car sizes, not a measurement.)")
    return {"median_car_length_m": round(med, 2), "cars_sampled": int(len(long_side)), "warning": warning}
