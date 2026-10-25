"""
Movements and signal phases: what traffic engineers count.

A flag count alone does not say WHICH movement is in trouble. Each track gets
  * its approach direction (northbound, eastbound, ...), from its heading when it first moves, and
  * for vehicles, its turning movement (through, left, right, U-turn), from how much its heading changed.
"North" is the top of the screen unless the user says otherwise (`north_deg`: the screen direction of north,
clockwise from up, e.g. 90 when north is to the right).

The map frame is a rotation of the image (calibration.py keeps it unmirrored, with y pointing down like the
screen), so a heading's angle clockwise from screen-up is atan2(vx, -vy), and a clockwise heading change is a right
turn.

Signal phases are typed by the user as lines "start-end name" (seconds into the video), e.g. "0-28 NS green".
Events are tagged with the phase they happened in. There is no signal detection: the phases are only as good as
the timing typed in.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .trajectories import Track

BOUNDS = ["northbound", "eastbound", "southbound", "westbound"]
SHORT = {"northbound": "NB", "eastbound": "EB", "southbound": "SB", "westbound": "WB"}
# A stopped vehicle's position jitters (up to ~1.5 m/s in random directions on drone footage), so "moving" needs a
# real speed, and headings come from distance actually travelled, not from single velocity samples.
MOVING_MPS = {"veh": 2.0, "bike": 1.0, "ped": 0.6}
HEADING_DIST_M = {"veh": 3.0, "bike": 2.0, "ped": 1.0}


def _bearing(v: np.ndarray, north_deg: float) -> float:
    """Compass bearing (degrees clockwise from north) of a map-frame velocity."""
    return (math.degrees(math.atan2(v[0], -v[1])) - north_deg) % 360.0


def _bound(bearing: float) -> str:
    return BOUNDS[int(((bearing + 45.0) % 360.0) // 90.0)]


def classify_track(t: Track, north_deg: float = 0.0, dt: float = 0.1) -> Dict:
    """{'bound', 'turn', 'label'}; 'stationary' when the road user never travelled the heading distance while
    moving. The entry heading is the direction of its first HEADING_DIST_M of travel, the exit heading that of its
    last, so jitter while waiting cannot create a turn."""
    moving = np.flatnonzero(t.speed >= MOVING_MPS.get(t.group, 1.0))
    dist = HEADING_DIST_M.get(t.group, 2.0)
    if len(moving) < 2:
        return {"bound": None, "turn": None, "label": "stationary"}
    pts = t.xy[moving[0]: moving[-1] + 1]
    from_start = np.linalg.norm(pts - pts[0], axis=1)
    to_end = np.linalg.norm(pts - pts[-1], axis=1)
    if from_start.max() < dist:
        return {"bound": None, "turn": None, "label": "stationary"}
    v_in = pts[int(np.argmax(from_start >= dist))] - pts[0]
    far = np.flatnonzero(to_end >= dist)
    # every point within `dist` of the end (it overshot past the end and came back): use the whole path's direction
    v_out = pts[-1] - pts[int(far.max())] if len(far) else pts[-1] - pts[0]
    if not np.any(v_out):
        v_out = v_in
    b_in, b_out = _bearing(v_in, north_deg), _bearing(v_out, north_deg)
    bound = _bound(b_in)
    if t.group != "veh":
        return {"bound": bound, "turn": None, "label": f"{SHORT[bound]} {'walking' if t.group == 'ped' else 'riding'}"}
    change = (b_out - b_in + 180.0) % 360.0 - 180.0          # + = clockwise = right turn
    if abs(change) < 35:
        turn = "through"
    elif abs(change) > 145:
        turn = "U-turn"
    else:
        turn = "right" if change > 0 else "left"
    return {"bound": bound, "turn": turn, "label": f"{SHORT[bound]} {turn}"}


def track_movements(tracks: Sequence[Track], north_deg: float = 0.0) -> Dict[Tuple[int, int], Dict]:
    return {(t.id, t.seg): classify_track(t, north_deg) for t in tracks}


def annotate_events(events: List[Dict], tracks: Sequence[Track], north_deg: float = 0.0,
                    phases: Sequence[Tuple[float, float, str]] = ()) -> None:
    """Add 'movement' to each side of every event, a 'movements' summary, and the signal 'phase'."""
    mv = track_movements(tracks, north_deg)
    for e in events:
        for side in ("a", "b"):
            e[side]["movement"] = mv.get((e[side]["track"], e[side]["seg"]), {}).get("label", "unknown")
        e["movements"] = f"{e['a']['movement']} x {e['b']['movement']}"
        e["phase"] = phase_at(e["t_peak"], phases)


def movement_counts(tracks: Sequence[Track], events: Sequence[Dict], north_deg: float = 0.0) -> List[Dict]:
    """Per movement: road users (exposure) and events involving it, so a busy movement can be told apart from a
    risky one."""
    mv = track_movements(tracks, north_deg)
    users: Dict[str, set] = {}
    for t in tracks:
        users.setdefault(mv[(t.id, t.seg)]["label"], set()).add(t.id)
    rows = []
    for label, ids in sorted(users.items()):
        n_ev = sum(1 for e in events if label in (e["a"].get("movement"), e["b"].get("movement")))
        rows.append({"movement": label, "road users": len(ids), "events": n_ev,
                     "events per 100 road users": round(100.0 * n_ev / len(ids), 1) if ids else None})
    return rows


# ------------------------------------------------------------------ signal phases
PHASE_LINE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s+(.+?)\s*$")


def parse_phases(text: str) -> Tuple[List[Tuple[float, float, str]], List[str]]:
    """'0-28 NS green' lines -> [(0, 28, 'NS green')] plus a message for every line that could not be read."""
    phases, problems = [], []
    for n, line in enumerate((text or "").splitlines(), 1):
        if not line.strip():
            continue
        m = PHASE_LINE.match(line)
        if not m or float(m.group(2)) <= float(m.group(1)):
            problems.append(f"Line {n} ('{line.strip()}') was ignored: write it as 'start-end name', e.g. '0-28 NS green'.")
            continue
        phases.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return sorted(phases), problems


def phase_at(t: float, phases: Sequence[Tuple[float, float, str]]) -> Optional[str]:
    for t0, t1, name in phases:
        if t0 <= t < t1:
            return name
    return None
