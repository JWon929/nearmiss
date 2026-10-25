"""
Settings for NearMiss.

Every number that changes what gets flagged lives here, so that the flags are
explainable ("flagged because TTC 0.8 s <= 1.0 s") and tunable from the UI.
Defaults are *starting points* for screening, not safety standards. Validate
them on your own clips (see the Event Review screen) before trusting them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional

# ---- road-user classes -------------------------------------------------------------
# Detector class name -> group used by the conflict engine.
CLASS_GROUP: Dict[str, str] = {
    "pedestrian": "ped",
    "bicycle": "bike",
    "motorcycle": "veh",
    "car": "veh",
    "bus": "veh",
    "truck": "veh",
}

# Rough half-widths (m). Only used to estimate body-to-body clearance for passes.
HALF_WIDTH_M: Dict[str, float] = {
    "pedestrian": 0.3,
    "bicycle": 0.3,
    "motorcycle": 0.4,
    "car": 0.9,
    "bus": 1.3,
    "truck": 1.3,
}

# Rough half-lengths (m), along the direction of travel. Used for post-encroachment time: a road user occupies a
# point from when its front reaches it until its back clears it.
HALF_LENGTH_M: Dict[str, float] = {
    "pedestrian": 0.3,
    "bicycle": 0.9,
    "motorcycle": 1.1,
    "car": 2.3,
    "bus": 6.0,
    "truck": 4.0,
}

KIND_LABEL: Dict[str, str] = {
    "ped-veh": "Vehicle-Pedestrian",
    "bike-veh": "Vehicle-Bicycle",
    "veh-veh": "Vehicle-Vehicle",
}

SEVERITY_LEVELS = ["none", "candidate", "moderate", "serious"]

# California "Three Feet for Safety Act" (AB 1371): 3 ft = 0.9144 m.
THREE_FEET_M = 0.9144


@dataclass
class Settings:
    """Analysis settings. All distances in meters, times in seconds."""

    # --- trajectory building ---------------------------------------------------
    analysis_hz: float = 10.0        # resampling rate for trajectories
    smooth_window_s: float = 0.8     # local-linear-regression window for position/velocity
    max_gap_s: float = 0.6           # detection gaps longer than this split a track
    min_track_s: float = 1.0         # ignore tracks shorter than this
    min_hits: int = 6                # ...or with fewer detections than this
    max_range_m: float = 80.0        # ignore ground points farther than this from the calibrated area
    max_speed_mps: float = 40.0      # drop implausible speeds (calibration / tracking errors)
    # Which pixel of a box touches the road: "bottom" (bottom-centre) for a camera that looks at the road at an
    # angle, "center" for a camera looking straight down (drone), where the bottom edge is half a car away.
    ground_point: str = "bottom"
    # Pedestrian, bicycle and motorcycle tracks that never move more than this many of their own box sizes are street
    # furniture the model mistook for a road user (lamp posts, signal boxes, bins). Measured in pixels after camera-
    # motion compensation, so it works even when the calibration is off. On our drone clip those objects moved at
    # most 1.4 box sizes (tall ones seem to sway as the drone moves); real pedestrians and cyclists moved 7-11.
    # 0 turns the check off. Only applied to overhead footage (ground_point "center"): street-level models rarely
    # mistake furniture for people, and there a waiting pedestrian is common and must not be dropped.
    static_clutter_boxes: float = 3.0
    # Track IDs the user confirmed as real road users (e.g. a pedestrian who waited the whole clip): never dropped.
    keep_tracks: List[int] = field(default_factory=list)

    # --- conflict engine -------------------------------------------------------
    # Indicator 1: time-to-collision (TTC), constant-velocity model, to a "collision radius".
    ttc_candidate_s: float = 3.0
    ttc_moderate_s: float = 2.0
    ttc_serious_s: float = 1.0
    collision_radius_m: Dict[str, float] = field(
        default_factory=lambda: {"ped-veh": 1.5, "bike-veh": 1.5, "veh-veh": 2.5}
    )
    # Indicator 2: actual minimum separation (center-to-center) = [candidate, moderate, serious].
    # bike-veh candidate 2.1 m ~ the 3-ft passing rule: 0.91 m gap + ~1.2 m of half-widths.
    proximity_m: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "ped-veh": [2.0, 1.5, 1.0],
            "bike-veh": [2.1, 1.5, 1.0],
            "veh-veh": [3.0, 2.5, 2.0],
        }
    )
    # Gates that remove obvious non-conflicts.
    min_closing_speed_mps: float = 1.0   # the pair must actually be approaching
    min_vehicle_speed_mps: float = 2.0   # a stopped car with a pedestrian walking past is not a conflict
    merge_gap_s: float = 1.0             # join flagged instants closer than this into one event
    ttc_min_duration_s: float = 0.5      # a TTC flag must last this long (one noisy heading is not a conflict)
    stopped_mps: float = 1.0             # slower than this counts as stopped (parked car, waiting pedestrian)
    evasive_decel_mps2: float = 2.0      # braking this hard counts as evasive action
    dedupe_time_s: float = 1.0           # merge same-kind events within this time...
    dedupe_dist_m: float = 3.0           # ...and this distance (ID switches can double-count)
    # Indicator 3 (optional): post-encroachment time (FHWA SSAM), for paths that CROSS. Catches near misses where the
    # two were never on a collision course at the same instant (one cleared the crossing point just before the other
    # reached it), which TTC cannot see. [candidate, moderate, serious] seconds. Off until validated on your labels.
    pet_flagging: bool = False
    pet_s: List[float] = field(default_factory=lambda: [3.0, 2.0, 1.0])
    # Indicator 4 (optional): DRAC, the braking needed to avoid the collision, = closing speed / (2 x TTC). Always
    # reported; flags only when on. 3.35 m/s^2 is the usual "conflict" line (comfortable braking limit in AASHTO).
    drac_flagging: bool = False
    drac_mps2: List[float] = field(default_factory=lambda: [3.35, 4.5, 6.0])

    # --- event clips / privacy ---------------------------------------------------
    privacy_level: str = "standard"      # "standard" | "strict"
    clip_before_s: float = 3.0
    clip_after_s: float = 1.5
    clip_fps: float = 8.0
    clip_max_width: int = 800
    max_clip_events: int = 80

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None) -> "Settings":
        base = cls()
        if not d:
            return base
        known = {f.name for f in fields(cls)}
        for k, v in d.items():
            if k not in known:
                continue
            cur = getattr(base, k)
            if isinstance(cur, dict):
                merged = dict(cur)
                merged.update(v)
                setattr(base, k, merged)
            elif isinstance(cur, bool):
                setattr(base, k, bool(v))
            elif isinstance(cur, int):
                setattr(base, k, int(v))
            elif isinstance(cur, float):
                setattr(base, k, float(v))
            else:
                setattr(base, k, v)
        return base


# ---- threshold presets ---------------------------------------------------------------
# Starting points for common kinds of site. They change only the conflict thresholds, never camera, privacy or
# clean-up settings. They are NOT standards: check them against your own labelled events (Report > Tuning).
THRESHOLD_FIELDS = ["ttc_candidate_s", "ttc_moderate_s", "ttc_serious_s", "collision_radius_m", "proximity_m",
                    "min_closing_speed_mps", "min_vehicle_speed_mps", "ttc_min_duration_s", "stopped_mps",
                    "evasive_decel_mps2", "pet_flagging", "pet_s", "drac_flagging", "drac_mps2"]
PRESETS: Dict[str, Dict[str, Any]] = {
    "Urban intersection (default)": {},
    "Parking lot or slow street": {
        "ttc_candidate_s": 2.0, "ttc_moderate_s": 1.5, "ttc_serious_s": 0.8, "min_vehicle_speed_mps": 1.0,
        "stopped_mps": 0.5,
        "proximity_m": {"ped-veh": [1.5, 1.0, 0.7], "bike-veh": [1.8, 1.2, 0.8], "veh-veh": [2.0, 1.5, 1.0]},
    },
    "Roundabout": {
        # curved paths: a straight-line prediction points at the island and at circulating cars, so require a
        # shorter, longer-lasting TTC
        "ttc_candidate_s": 2.0, "ttc_moderate_s": 1.5, "ttc_serious_s": 0.8, "ttc_min_duration_s": 0.8,
    },
    "Arterial or high-speed road": {
        "ttc_candidate_s": 4.0, "ttc_moderate_s": 3.0, "ttc_serious_s": 1.5, "min_vehicle_speed_mps": 3.0,
        "proximity_m": {"ped-veh": [2.5, 2.0, 1.5], "bike-veh": [2.5, 2.0, 1.5], "veh-veh": [4.0, 3.0, 2.5]},
    },
}
PRESET_NOTES = {
    "Urban intersection (default)": "The app's defaults: signalised or stop-controlled crossings at city speeds.",
    "Parking lot or slow street": "Walking-pace traffic: shorter times and distances, so slow manoeuvres are not all "
                                  "flagged.",
    "Roundabout": "Curving paths make straight-line TTC point at things drivers will steer around; this asks for a "
                  "shorter TTC that lasts longer.",
    "Arterial or high-speed road": "Higher speeds need more time and room, so the thresholds are wider.",
}


def apply_preset(S: "Settings", name: str) -> "Settings":
    """S with its threshold fields reset to the defaults, then the preset's values. Everything else is kept."""
    base = Settings()
    out = Settings.from_dict(S.to_dict())
    for f in THRESHOLD_FIELDS:
        setattr(out, f, getattr(base, f))
    for k, v in PRESETS[name].items():
        setattr(out, k, {kk: list(vv) if isinstance(vv, list) else vv for kk, vv in v.items()} if isinstance(v, dict)
                else v)
    return out


def matching_preset(S: "Settings") -> Optional[str]:
    """The preset S currently equals on every threshold field, or None if it was changed by hand."""
    for name in PRESETS:
        P = apply_preset(S, name)
        if all(getattr(P, f) == getattr(S, f) for f in THRESHOLD_FIELDS):
            return name
    return None


@dataclass
class DetectionSettings:
    """How detection/tracking runs. Changing these requires re-running detection."""

    detector: str = "auto"       # "auto" | "yolo" | "motion"
    model: str = "yolo11n.pt"    # any Ultralytics detection weights
    conf: float = 0.25
    imgsz: int = 960
    target_fps: float = 10.0     # frames per second actually processed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None) -> "DetectionSettings":
        base = cls()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, type(getattr(base, k))(v))
        return base
