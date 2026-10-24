"""End-to-end maths check on the scripted scene (known answer key).

Perfect-but-noisy detections are fed through calibration -> trajectories -> conflict engine.
Expected (see tools/make_synthetic_video.py):
  carA-ped1   serious car-pedestrian conflict at ~12 s (closest approach ~1.4 m, PET ~1.0 s)
  carC-bike1  close pass under 3 ft at ~30 s
  carB-ped2, carD-bike2  controls: must NOT be flagged

This proves the MATHS on a known answer. It says nothing about real-world detector accuracy.
"""
import json

import pytest

from engine import demo
from engine.calibration import Calibration
from engine.config import Settings
from engine.conflicts import detect_conflicts
from engine.trajectories import Detections, build_tracks

pytestmark = pytest.mark.skipif(not demo.demo_available(), reason="demo clip not generated")


@pytest.fixture(scope="module")
def run():
    truth = json.loads(demo.DEMO_TRUTH.read_text())
    cal = Calibration.from_points(truth["calibration_points_px"], truth["calibration_width_m"],
                                  truth["calibration_length_m"], tuple(truth["size"]))
    det = Detections(demo.scripted_detections())
    tracks, stats = build_tracks(det, cal, Settings())
    events = detect_conflicts(tracks, Settings())
    names = demo.actor_names()
    by_pair = {frozenset((names[e["a"]["track"]], names[e["b"]["track"]])): e for e in events}
    return tracks, stats, events, by_pair


def test_all_scripted_actors_become_tracks(run):
    tracks, stats, _, _ = run
    assert stats["raw_tracks"] == 8 and len({t.id for t in tracks}) == 8


def test_car_pedestrian_near_miss_is_flagged_serious(run):
    e = run[3][frozenset(("carA", "ped1"))]
    assert e["severity"] == "serious" and e["kind"] == "ped-veh"
    assert e["t_peak"] == pytest.approx(12.0, abs=0.5)
    assert e["d_min_m"] == pytest.approx(1.4, abs=0.4)       # script: ped crossed 1.0 s earlier, 1.4 m offset
    assert e["pet_s"] == pytest.approx(1.0, abs=0.3)


def test_close_pass_is_flagged_and_marked_under_three_feet(run):
    e = run[3][frozenset(("carC", "bike1"))]
    assert e["kind"] == "bike-veh" and e["below_3ft"] is True
    assert e["t_peak"] == pytest.approx(30.0, abs=0.5)
    assert e["d_min_m"] == pytest.approx(1.4, abs=0.4)


def test_control_scenes_are_not_flagged(run):
    by_pair = run[3]
    assert frozenset(("carB", "ped2")) not in by_pair
    assert frozenset(("carD", "bike2")) not in by_pair


def test_no_extra_events(run):
    assert len(run[2]) == 2


def test_speeds_match_the_script(run):
    tracks = {t.id: t for t in run[0]}
    names = {v: k for k, v in demo.actor_names().items()}
    # carA drives 60 m in 6 s = 10 m/s ; ped1 walks 14 m in 10 s = 1.4 m/s
    assert float(tracks[names["carA"]].speed[5:-5].mean()) == pytest.approx(10.0, rel=0.1)
    assert float(tracks[names["ped1"]].speed[5:-5].mean()) == pytest.approx(1.4, rel=0.2)


def test_stricter_thresholds_can_remove_events():
    truth = json.loads(demo.DEMO_TRUTH.read_text())
    cal = Calibration.from_points(truth["calibration_points_px"], truth["calibration_width_m"],
                                  truth["calibration_length_m"], tuple(truth["size"]))
    tracks, _ = build_tracks(Detections(demo.scripted_detections()), cal, Settings())
    S = Settings.from_dict({"ttc_candidate_s": 0.05, "ttc_moderate_s": 0.04, "ttc_serious_s": 0.03,
                            "collision_radius_m": {"ped-veh": 0.1, "bike-veh": 0.1, "veh-veh": 0.1},
                            "proximity_m": {"ped-veh": [0.2, 0.15, 0.1], "bike-veh": [0.2, 0.15, 0.1], "veh-veh": [0.2, 0.15, 0.1]}})
    assert len(detect_conflicts(tracks, S)) == 0
