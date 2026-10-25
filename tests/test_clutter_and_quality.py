"""Street-furniture filter, tracker-quality stats, validation totals, calibration hints and the default overhead
model, on cases with a known answer."""
import numpy as np

from engine.calibration import Calibration
from engine.config import Settings
from engine.trajectories import Detections, build_tracks, is_still, likely_id_switches
from engine.validation import combine, summarize

CAL = Calibration.from_two_points([0, 0], [100, 0], 10.0, (1000, 1000))     # 10 px per metre, straight down


def rows_for(tid, cls, xs, ys, t0=0.0, dt=0.1, size=20):
    return [(k, t0 + k * dt, tid, cls, 0.9, x - size / 2, y - size / 2, x + size / 2, y + size / 2)
            for k, (x, y) in enumerate(zip(xs, ys))]


def overhead():
    S = Settings()
    S.ground_point = "center"
    return S


# ------------------------------------------------------------------ street furniture
def test_a_pedestrian_that_never_moves_is_street_furniture_and_a_walking_one_is_kept():
    n = 40
    pole = rows_for(1, "pedestrian", 300 + np.random.default_rng(0).normal(0, 3, n), np.full(n, 300))   # jitter only
    walker = rows_for(2, "pedestrian", np.linspace(300, 360, n), np.full(n, 500), size=8)   # 6 m in 4 s, 0.8 m box
    tracks, stats = build_tracks(Detections(pole + walker), CAL, overhead())
    assert [t.id for t in tracks] == [2] and stats["dropped_static_clutter"] == 1


def test_a_parked_car_is_not_dropped_as_furniture():
    n = 40
    tracks, stats = build_tracks(Detections(rows_for(1, "car", np.full(n, 300), np.full(n, 300))), CAL, overhead())
    assert len(tracks) == 1 and stats["dropped_static_clutter"] == 0


def test_stillness_is_measured_in_box_sizes_so_calibration_does_not_matter():
    boxes = np.array([[0, 0, 50, 50], [60, 0, 110, 50]], float)           # moved 60 px = 1.2 box sizes
    frames = np.array([0, 1])
    assert is_still(boxes, frames, None, 3.0) and not is_still(boxes, frames, None, 1.0)
    S = overhead()
    S.static_clutter_boxes = 0                                              # 0 turns the check off
    n = 40
    tracks, _ = build_tracks(Detections(rows_for(1, "pedestrian", np.full(n, 300), np.full(n, 300))), CAL, S)
    assert len(tracks) == 1


# ------------------------------------------------------------------ tracker quality
def test_a_track_that_restarts_where_another_ended_is_a_likely_id_switch():
    first = rows_for(1, "car", np.linspace(100, 300, 21), np.full(21, 300))          # 0-2 s at 10 m/s
    second = rows_for(2, "car", np.linspace(330, 530, 21), np.full(21, 300), t0=2.3)   # picks up 0.3 s later
    other = rows_for(3, "car", np.linspace(100, 300, 21), np.full(21, 600), t0=2.3)    # another car, elsewhere
    tracks, stats = build_tracks(Detections(first + second + other), CAL, overhead())
    assert len(tracks) == 3 and likely_id_switches(tracks) == 1 and stats["likely_id_switches"] == 1


# ------------------------------------------------------------------ validation totals
def ev(kind, t):
    return {"kind": kind, "t_peak": t, "severity": "candidate"}


def test_totals_over_recordings_sum_counts_instead_of_averaging_precision():
    a = summarize([ev("veh-veh", 1.0)], {"events": [{"kind": "veh-veh", "t_peak": 1.0, "label": "tp"}]})
    b = summarize([ev("veh-veh", t) for t in (1, 5, 9)],
                  {"events": [{"kind": "veh-veh", "t_peak": t, "label": "fp"} for t in (1, 5, 9)]})
    tot = combine([a, b])
    assert tot["precision"] == 0.25 and tot["recordings"] == 2      # 1 / 4, not the mean of 100% and 0%
    assert tot["by_kind"]["veh-veh"]["flagged"] == 4


# ------------------------------------------------------------------ calibration hints
def test_a_short_known_distance_and_an_impossible_rectangle_are_warned_about():
    short = Calibration.from_two_points([500, 500], [560, 500], 10.0, (1920, 1080))
    assert any("close together" in h for h in short.quality_hints(check_shape=False))
    good = Calibration.from_two_points([200, 500], [1600, 500], 30.0, (1920, 1080))
    assert not any("close together" in h for h in good.quality_hints(check_shape=False))
    # a far-away trapezoid with the long side typed as the short one: the frame bottom lands beyond the horizon
    bad = Calibration.from_points([[900, 100], [1000, 100], [1400, 150], [500, 150]], 3.0, 200.0, (1920, 1080))
    assert any("horizon" in h for h in bad.quality_hints())


# ------------------------------------------------------------------ models
def test_stock_aerial_model_stays_the_default_when_the_visdrone_model_is_present(tmp_path, monkeypatch):
    import detector_tracker as dt
    monkeypatch.setattr(dt, "HERE", tmp_path)
    assert dt.VISDRONE_MODEL not in dt.aerial_choices()
    (tmp_path / dt.VISDRONE_MODEL).write_bytes(b"")
    order = list(dt.aerial_choices())
    assert order[0] == dt.AERIAL_MODEL and order[1] == dt.VISDRONE_MODEL
