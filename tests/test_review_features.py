"""Road area mask, exposure rates, per-kind validation, calibration geometry warnings and the report notes."""
import json

import numpy as np
import pytest

from engine import demo, validation
from engine.analysis import build_report_html, exposure
from engine.calibration import Calibration
from engine.config import Settings
from engine.conflicts import detect_conflicts
from engine.privacy import regions_for
from engine.roi import as_polygon, inside
from engine.trajectories import Detections, Track, build_tracks


# ------------------------------------------------------------------ road area
def test_point_in_polygon_handles_a_concave_outline():
    # an L-shaped road: the notch at the top right is not road
    poly = as_polygon([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]])
    pts = np.array([[2, 2], [8, 2], [2, 8], [8, 8], [-1, 5], [11, 1]])
    assert inside(pts, poly).tolist() == [True, True, True, False, False, False]


def test_too_few_points_means_no_mask():
    assert as_polygon([[0, 0], [5, 5]]) is None
    assert inside(np.zeros((3, 2)), None).all()


@pytest.fixture(scope="module")
def scene():
    if not demo.demo_available():
        pytest.skip("demo clip not generated")
    truth = json.loads(demo.DEMO_TRUTH.read_text())
    cal = Calibration.from_points(truth["calibration_points_px"], truth["calibration_width_m"],
                                  truth["calibration_length_m"], tuple(truth["size"]))
    return Detections(demo.scripted_detections()), cal, tuple(truth["size"])


def test_whole_frame_road_area_changes_nothing(scene):
    det, cal, (w, h) = scene
    full = as_polygon([[-1, -1], [w + 1, -1], [w + 1, h + 1], [-1, h + 1]])
    a, _ = build_tracks(det, cal, Settings())
    b, stats = build_tracks(det, cal, Settings(), roi=full)
    assert len(a) == len(b) and stats["dropped_outside_roi"] == 0


def test_road_area_away_from_everyone_drops_every_track(scene):
    det, cal, _ = scene
    corner = as_polygon([[0, 0], [3, 0], [3, 3], [0, 3]])
    tracks, stats = build_tracks(det, cal, Settings(), roi=corner)
    assert tracks == [] and stats["dropped_outside_roi"] == stats["raw_tracks"]


# ------------------------------------------------------------------ exposure
DT = 0.1


def track(tid, group, k0, n):
    cls = {"veh": "car", "ped": "pedestrian", "bike": "bicycle"}[group]
    t = (k0 + np.arange(n)) * DT
    xy = np.c_[np.arange(n, dtype=float), np.zeros(n)]
    v = np.tile([10.0, 0.0], (n, 1))
    return Track(tid, 0, cls, group, k0, t, xy, v, np.linalg.norm(v, axis=1), n)


def test_exposure_counts_pairs_that_were_on_the_road_together():
    tracks = [track(1, "veh", 0, 50), track(2, "veh", 10, 50), track(3, "ped", 0, 30),
              track(4, "veh", 200, 20),            # alone: overlaps nobody
              track(2, "veh", 70, 10)]             # second segment of road user 2: not a new road user or pair
    events = [{"kind": "veh-veh"}, {"kind": "ped-veh"}]
    x = exposure(tracks, events, DT)
    assert x["road_users"] == {"veh": 3, "ped": 1}
    assert x["pairs_together"] == {"ped-veh": 2, "bike-veh": 0, "veh-veh": 1}
    assert x["events_per_100_vehicles"] == pytest.approx(66.67)
    assert x["events_per_1000_pairs"] == pytest.approx(666.67)
    assert x["events_per_1000_pairs_by_kind"]["bike-veh"] is None


def test_exposure_with_no_tracks_gives_no_rates():
    x = exposure([], [], DT)
    assert x["events_per_100_vehicles"] is None and x["events_per_1000_pairs"] is None


# ------------------------------------------------------------------ validation by kind
def test_precision_is_reported_per_conflict_kind():
    events = [{"kind": "ped-veh", "t_peak": 5.0, "severity": "serious"},
              {"kind": "ped-veh", "t_peak": 9.0, "severity": "candidate"},
              {"kind": "veh-veh", "t_peak": 20.0, "severity": "candidate"},
              {"kind": "veh-veh", "t_peak": 40.0, "severity": "moderate"}]
    store = {}
    validation.set_label(store, "ped-veh", 5.0, "tp")
    validation.set_label(store, "ped-veh", 9.0, "fp")
    validation.set_label(store, "veh-veh", 20.0, "tp")
    s = validation.summarize(events, store)
    assert s["by_kind"]["ped-veh"]["precision"] == 0.5
    assert s["by_kind"]["veh-veh"]["precision"] == 1.0 and s["by_kind"]["veh-veh"]["unreviewed"] == 1
    assert "bike-veh" not in s["by_kind"]


# ------------------------------------------------------------------ calibration geometry
FRAME = (1000, 500)


def test_a_well_spread_rectangle_has_no_shape_warnings():
    cal = Calibration.from_points([[300, 150], [700, 150], [820, 400], [180, 400]], 10.0, 6.0, FRAME)
    assert cal._shape_hints() == []


def test_corners_clicked_close_together_are_warned_about():
    cal = Calibration.from_points([[500, 300], [530, 300], [532, 320], [498, 320]], 3.0, 2.0, FRAME)
    assert any("pixels apart" in h for h in cal.quality_hints())


def test_nearly_collinear_corners_are_warned_about():
    cal = Calibration.from_points([[100, 300], [500, 290], [900, 300], [500, 330]], 10.0, 10.0, FRAME)
    assert any("nearly in a line" in h for h in cal.quality_hints())


def test_far_edge_squeezed_toward_the_horizon_is_warned_about():
    cal = Calibration.from_points([[480, 100], [520, 100], [900, 450], [100, 450]], 8.0, 30.0, FRAME)
    assert any("horizon" in h for h in cal.quality_hints())


def test_two_point_calibration_skips_the_shape_checks():
    cal = Calibration.from_two_points([100, 100], [300, 100], 10.0, (1000, 800))
    assert not any("corner" in h.lower() for h in cal.quality_hints(check_shape=False))


# ------------------------------------------------------------------ privacy and report
def test_strict_blur_covers_the_whole_box_for_overhead_footage():
    box = (100, 100, 140, 180)
    (x1, y1, x2, y2), = regions_for("car", box, "strict")
    assert x1 <= 100 and y1 <= 100 and x2 >= 140 and y2 >= 180


def test_report_states_blind_spots_and_unreviewed_events():
    summ = validation.summarize([{"kind": "veh-veh", "t_peak": 1.0, "severity": "candidate"}], {})
    page = build_report_html("t", {}, {}, Settings(), [], summ, "", "", notes=["Vehicles only."],
                             exposure={"events_per_100_vehicles": 1.5})
    assert "Vehicles only." in page and "were not reviewed" in page and "events_per_100_vehicles" in page
