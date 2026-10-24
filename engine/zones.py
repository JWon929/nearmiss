"""
Named zones: crosswalks, stop-line areas, bike lanes, the intersection box...

The user outlines each zone on the calibration frame (same reference frame as the road area). The outline is mapped
to metres with the calibration (a homography keeps straight edges straight, so a polygon stays a polygon), and then:

  * every event is tagged with the zone its closest-approach point falls in ("" if none)
  * every zone gets the number of road users that passed through it, by type, so its event count can be read
    against how busy it was (events per 100 road users through the zone)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .calibration import Calibration
from .roi import inside
from .trajectories import Track

ZONE_KINDS = ["crosswalk", "stop line area", "bike lane", "intersection box", "driveway", "other"]
# BGR, one per zone in order of creation (wraps around)
ZONE_COLORS = [(255, 160, 40), (200, 90, 255), (60, 200, 120), (40, 200, 255), (255, 90, 160), (160, 160, 255)]


def zone_polygons_map(zones: Sequence[Dict], calib: Calibration) -> List[Optional[np.ndarray]]:
    """Each zone outline in map metres, or None when a corner is at or over the horizon."""
    out = []
    for z in zones:
        pts, ok = calib.img_to_map(z["points"])
        out.append(pts if len(pts) >= 3 and ok.all() else None)
    return out


def tag_events(events: List[Dict], zones: Sequence[Dict], calib: Calibration) -> None:
    """Set e['zone'] to the name of the first zone containing the event point."""
    polys = zone_polygons_map(zones, calib)
    for e in events:
        e["zone"] = ""
        p = np.array([[e["x"], e["y"]]])
        for z, poly in zip(zones, polys):
            if poly is not None and inside(p, poly)[0]:
                e["zone"] = z["name"]
                break


def densify(xy: np.ndarray, step_m: float = 0.5) -> np.ndarray:
    """Points every step_m along a path, so a fast road user cannot skip over a narrow zone between two samples."""
    if len(xy) < 2:
        return xy
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    at = np.arange(0.0, s[-1] + step_m, step_m)
    return np.c_[np.interp(at, s, xy[:, 0]), np.interp(at, s, xy[:, 1])]


def zone_summary(events: Sequence[Dict], tracks: Sequence[Track], zones: Sequence[Dict], calib: Calibration) -> List[Dict]:
    """One row per zone: road users through it, events in it (by severity), and events per 100 road users."""
    polys = zone_polygons_map(zones, calib)
    rows = []
    for z, poly in zip(zones, polys):
        users = {"ped": set(), "bike": set(), "veh": set()}
        if poly is not None:
            for t in tracks:
                if t.group in users and inside(densify(t.xy), poly).any():
                    users[t.group].add(t.id)
        evs = [e for e in events if e.get("zone") == z["name"]]
        n_users = sum(len(s) for s in users.values())
        rows.append({
            "zone": z["name"], "type": z.get("kind", "other"), "valid": poly is not None,
            "pedestrians": len(users["ped"]), "bicycles": len(users["bike"]), "vehicles": len(users["veh"]),
            "events": len(evs),
            "serious": sum(e["severity"] == "serious" for e in evs),
            "moderate": sum(e["severity"] == "moderate" for e in evs),
            "candidate": sum(e["severity"] == "candidate" for e in evs),
            "events_per_100_users": round(100.0 * len(evs) / n_users, 1) if n_users else None,
        })
    return rows
