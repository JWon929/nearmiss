"""
How far the numbers of one recording can be trusted, in one place.

Every page used to show its own warnings, so a recording could show two "serious" flags with an unchecked
calibration and a car-size warning further down the page. Here the problems are collected once, ranked, and given
one overall verdict:

  * calibration_status   good / unchecked / problem. Severity depends on distances and speeds, so with anything
                         but "good" every severity is PROVISIONAL and is shown that way.
  * quality_items        every limitation (calibration, blind spots, camera alignment, tracking, review status),
                         each with what to do about it and which page fixes it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

CHECK_LIMIT_PCT = 10.0


def calibration_status(cal: Optional[Dict], size_check: Optional[Dict] = None, synthetic: bool = False) -> Dict:
    """`cal` is calibration.json; `size_check` is insights.vehicle_size_check. 'provisional' means severities should
    not be quoted as final."""
    if synthetic:
        return {"level": "good", "title": "Scripted demo (exact calibration)", "detail": "", "provisional": False}
    if not cal:
        return {"level": "problem", "title": "Not calibrated", "detail": "Distances and speeds cannot be measured.",
                "provisional": True}
    checks = cal.get("checks") or []
    size_warn = (size_check or {}).get("warning")
    if len(checks) > 1:                              # several known distances: judge by their typical (RMS) error
        rms = float(np.sqrt(np.mean([c["error_pct"] ** 2 for c in checks])))
        worst = max(checks, key=lambda c: abs(c["error_pct"]))
        if rms > CHECK_LIMIT_PCT:
            return {"level": "problem", "title": f"Calibration off by about {rms:.0f}% ({len(checks)} checks)",
                    "detail": f"The worst: a known {worst['known_m']} m measured {worst['measured_m']} m. Fit the scale "
                              "to all distances on Calibrate, or re-click the calibration points.", "provisional": True}
        if size_warn:
            return {"level": "problem", "title": "Calibration scale looks wrong", "detail": size_warn,
                    "provisional": True}
        return {"level": "good", "title": f"Calibration checked ({len(checks)} distances, typical error {rms:.1f}%)",
                "detail": f"The worst: a known {worst['known_m']} m measured {worst['measured_m']} m.",
                "provisional": False}
    if checks and abs(checks[0]["error_pct"]) > CHECK_LIMIT_PCT:
        c = checks[0]
        return {"level": "problem", "title": f"Calibration off by {c['error_pct']:+.0f}%",
                "detail": f"A known {c['known_m']} m measured {c['measured_m']} m. Distances, speeds and severities "
                          "are off by about that much. Re-click the calibration points or fix the typed lengths.",
                "provisional": True}
    if size_warn:
        return {"level": "problem", "title": "Calibration scale looks wrong", "detail": size_warn, "provisional": True}
    if not checks:
        return {"level": "unchecked", "title": "Calibration not checked",
                "detail": "Nobody measured a second known distance, so the size of distance and speed errors is "
                          "unknown. Severities are provisional until the accuracy check is done on Calibrate.",
                "provisional": True}
    c = checks[0]
    return {"level": "good", "title": f"Calibration checked ({c['error_pct']:+.1f}%)",
            "detail": f"A known {c['known_m']} m measured {c['measured_m']} m.", "provisional": False}


def quality_items(*, cal_status: Dict, vehicles_only: bool, motion_summary: Optional[Dict], stats: Optional[Dict],
                  summary: Optional[Dict], privacy: Optional[Dict]) -> List[Dict]:
    """[{level: problem|warn|info, area, message, action, page}], most serious first."""
    items: List[Dict] = []
    if cal_status["level"] != "good":
        items.append({"level": "problem" if cal_status["level"] == "problem" else "warn", "area": "Calibration",
                      "message": f"{cal_status['title']}. {cal_status['detail']}",
                      "action": "Do the accuracy check" if cal_status["level"] == "unchecked" else "Fix the calibration",
                      "page": "Calibrate"})
    if vehicles_only:
        items.append({"level": "warn", "area": "Blind spot",
                      "message": "Pedestrians and bicycles were not detected (vehicles-only model). Conflicts with them "
                                 "are NOT MEASURED, which is not the same as none.",
                      "action": "Re-run Detect", "page": "Detect"})
    if motion_summary and motion_summary.get("frames"):
        lost = motion_summary.get("lost_frames", 0)
        if lost > 0.05 * motion_summary["frames"]:
            items.append({"level": "warn", "area": "Camera alignment",
                          "message": f"{lost} of {motion_summary['frames']} frames could not be lined up with the first "
                                     "one. Positions on those frames are less reliable.",
                          "action": None, "page": None})
    if stats:
        n = stats.get("analysed_tracks", 0)
        if n and stats.get("likely_id_switches", 0) > 0.15 * n:
            items.append({"level": "warn", "area": "Tracking",
                          "message": f"About {stats['likely_id_switches']} road users were split into two tracks. An "
                                     "encounter can then be counted twice or missed.",
                          "action": "Keep lost tracks longer", "page": "Detect"})
        if stats.get("dropped_static_clutter"):
            items.append({"level": "info", "area": "Tracking",
                          "message": f"{stats['dropped_static_clutter']} pedestrians/bikes that never moved were treated "
                                     "as street furniture and left out. Check them; restore any real person.",
                          "action": "Check them", "page": "Results"})
    if summary and summary.get("events_flagged"):
        if summary.get("unreviewed"):
            items.append({"level": "info", "area": "Review",
                          "message": f"{summary['unreviewed']} of {summary['events_flagged']} flags are not reviewed. "
                                     "Only reviewed real near misses are confirmed.",
                          "action": "Review them", "page": "Event Review"})
    if privacy and not privacy.get("blur", True):
        items.append({"level": "warn", "area": "Privacy",
                      "message": "Blur was turned off for this recording's videos: faces and plates in them are NOT "
                                 "blurred. Do not share them.", "action": None, "page": None})
    order = {"problem": 0, "warn": 1, "info": 2}
    return sorted(items, key=lambda i: order[i["level"]])
