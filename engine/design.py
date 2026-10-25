"""
Proposed designs: screen an intersection that is not built yet.

There is no video of a design that does not exist. The FHWA Surrogate Safety Assessment Model (SSAM) approach is
to run a traffic microsimulation (VISSIM, SUMO, Aimsun...) of each design alternative, export the simulated
vehicle and pedestrian trajectories, and count conflicts in them. This module does the counting part with the same
conflict engine as the video workflow, so alternatives can be compared on equal terms.

Input: one CSV per alternative with columns
    t_s, id, class, x_m, y_m
(time in seconds, a road-user ID, its class: pedestrian / bicycle / motorcycle / car / bus / truck, and its position
in metres on a flat ground plane). Most simulators can export this (SUMO: fcd-output; VISSIM: vehicle record).

Caveat (show it with the results): a simulation only has the conflicts its driver and pedestrian models produce.
Compare alternatives with each other under the same demand and model settings, never with counts from video.
"""
from __future__ import annotations

import csv
import io
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .analysis import exposure
from .config import CLASS_GROUP, Settings
from .conflicts import detect_conflicts
from .trajectories import Track, _resample, local_linear

ALIASES = {"person": "pedestrian", "ped": "pedestrian", "bike": "bicycle", "cyclist": "bicycle", "vehicle": "car",
           "passenger": "car", "motorbike": "motorcycle", "hgv": "truck", "lorry": "truck"}
COLUMNS = ("t_s", "id", "class", "x_m", "y_m")


def read_trajectories(raw: bytes) -> Tuple[List[tuple], List[str]]:
    """CSV bytes -> rows (t, id, class, x, y) and problems found. Bad rows are skipped and reported."""
    problems: List[str] = []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [], ["The file is not a UTF-8 text (CSV) file."]
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        return [], [f"Missing column(s): {', '.join(missing)}. Expected: {', '.join(COLUMNS)}."]
    rows, bad, unknown = [], 0, set()
    for r in reader:
        try:
            cls = ALIASES.get(r["class"].strip().lower(), r["class"].strip().lower())
            if cls not in CLASS_GROUP:
                unknown.add(r["class"])
                continue
            rows.append((float(r["t_s"]), str(r["id"]).strip(), cls, float(r["x_m"]), float(r["y_m"])))
        except (TypeError, ValueError):
            bad += 1
    if bad:
        problems.append(f"{bad} row(s) with a non-numeric time or position were skipped.")
    if unknown:
        problems.append(f"Unknown class(es) skipped: {', '.join(sorted(unknown))[:200]}. Use: "
                        f"{', '.join(CLASS_GROUP)}.")
    return rows, problems


def build_design_tracks(rows: Sequence[tuple], S: Settings) -> List[Track]:
    """Simulated positions are exact, so tracks are only resampled to the engine's grid and lightly smoothed."""
    dt = 1.0 / S.analysis_hz
    half = max(1, int(round(S.smooth_window_s / dt / 2.0)))
    by_id: Dict[str, List[tuple]] = {}
    for r in rows:
        by_id.setdefault(r[1], []).append(r)
    tracks: List[Track] = []
    for n, (rid, rs) in enumerate(sorted(by_id.items()), start=1):
        rs.sort(key=lambda r: r[0])
        ts = np.array([r[0] for r in rs])
        ps = np.array([[r[3], r[4]] for r in rs])
        keep = np.r_[True, np.diff(ts) > 1e-9]
        ts, ps = ts[keep], ps[keep]
        if len(ts) < 3 or ts[-1] - ts[0] < S.min_track_s:
            continue
        res = _resample(ts, ps, dt)
        if res is None:
            continue
        k0, tg, pg = res
        pos, vel = local_linear(tg, pg, half)
        cls = rs[0][2]
        tracks.append(Track(n, 0, cls, CLASS_GROUP[cls], k0, tg, pos, vel, np.linalg.norm(vel, axis=1), len(ts)))
    return tracks


def evaluate(rows: Sequence[tuple], S: Settings) -> Dict:
    tracks = build_design_tracks(rows, S)
    events = detect_conflicts(tracks, S)
    span = (max(r[0] for r in rows) - min(r[0] for r in rows)) if rows else 0.0
    ex = exposure(tracks, events, 1.0 / S.analysis_hz, span)
    sev = {s: sum(e["severity"] == s for e in events) for s in ("serious", "moderate", "candidate")}
    kinds = {k: sum(e["kind"] == k for e in events) for k in ("ped-veh", "bike-veh", "veh-veh")}
    return {"tracks": tracks, "events": events, "exposure": ex, "severity": sev, "kinds": kinds}


def demo_alternatives() -> Dict[str, str]:
    """Two sample alternatives made from the demo scene, as CSV text: the existing design, and the same traffic with
    a 3 s leading pedestrian interval on the first crosswalk (ped1 starts 3 s earlier) and a bike lane moved 1.5 m
    away from the general lane. They show the file format and the comparison, nothing more."""
    from .demo import truth

    out = {}
    for name, changed in (("existing", False), ("lpi_and_bike_buffer", True)):
        lines = ["t_s,id,class,x_m,y_m"]
        for a in truth()["actors"]:
            wp = np.array(a["waypoints"], dtype=float)
            if changed and a["name"] == "ped1":
                wp[:, 0] -= 3.0
            if changed and a["cls"] == "bicycle":
                wp[:, 1] += 1.5
            for t in np.arange(wp[0, 0], wp[-1, 0] + 1e-9, 0.1):
                x, y = np.interp(t, wp[:, 0], wp[:, 1]), np.interp(t, wp[:, 0], wp[:, 2])
                lines.append(f"{t:.2f},{a['name']},{a['cls']},{x:.3f},{y:.3f}")
        out[name] = "\n".join(lines) + "\n"
    return out
