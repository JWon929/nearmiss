"""Plain-language explanations, speeds, sensitivity, tuning from labels, the car-size hint and the review pack."""
import json

import numpy as np
import pytest

from engine import demo, review_pack
from engine.calibration import Calibration
from engine.config import Settings
from engine.insights import (MPS_TO_MPH, explain_event, sensitivity, speed_summary, tuning_curve,
                             vehicle_size_check)
from engine.trajectories import Detections, Track, build_tracks


def event(kind="ped-veh", t=5.0, ttc=0.8, d=1.4, label=None, sev_ttc="serious", eid="E001"):
    return {"id": eid, "kind": kind, "kind_label": "Vehicle-Pedestrian", "t_peak": t, "ttc_min_s": ttc, "d_min_m": d,
            "pet_s": None, "closing_speed_mps": 4.0, "below_3ft": None, "severity": "serious",
            "severity_from_ttc": sev_ttc, "a": {"cls": "car", "speed_mps": 9.0}, "b": {"cls": "pedestrian",
                                                                                   "speed_mps": 1.4},
            "label": label, "reasons": []}


# ------------------------------------------------------------------ explanations
def test_explanation_reads_like_a_sentence_with_the_key_numbers():
    s = explain_event(event())
    assert s.startswith("A car (9.0 m/s, 20 mph) and a pedestrian")
    assert "within 1.4 m" in s and "in 0.8 s" in s


def test_explanation_skips_ttc_when_there_was_none():
    assert "touching distance" not in explain_event(event(ttc=None))


# ------------------------------------------------------------------ speeds
def vehicle(tid, speed, n=40, k0=0, cls="car", group="veh"):
    t = (k0 + np.arange(n)) * 0.1
    v = np.tile([speed, 0.0], (n, 1))
    return Track(tid, 0, cls, group, k0, t, np.c_[np.arange(n) * speed * 0.1, np.zeros(n)], v,
                 np.full(n, float(speed)), n)


def test_speed_summary_is_per_vehicle_and_ignores_parked_cars_and_pedestrians():
    tracks = [vehicle(1, 10), vehicle(2, 12), vehicle(3, 14), vehicle(4, 0.0),
              vehicle(5, 1.4, cls="pedestrian", group="ped")]
    sp = speed_summary(tracks, Settings(), limit_mps=25 / MPS_TO_MPH)
    assert sp["vehicles"] == 3 and sp["median_mps"] == pytest.approx(12.0)
    assert sp["share_over_limit"] == pytest.approx(2 / 3, abs=1e-3)     # 12 and 14 m/s are over 25 mph (11.2 m/s)


def test_two_segments_of_one_vehicle_count_once():
    assert speed_summary([vehicle(1, 10), vehicle(1, 10, k0=100)], Settings())["vehicles"] == 1


# ------------------------------------------------------------------ sensitivity
def test_looser_thresholds_never_find_fewer_events():
    if not demo.demo_available():
        pytest.skip("demo clip not generated")
    truth = json.loads(demo.DEMO_TRUTH.read_text())
    cal = Calibration.from_points(truth["calibration_points_px"], truth["calibration_width_m"],
                                  truth["calibration_length_m"], tuple(truth["size"]))
    tracks, _ = build_tracks(Detections(demo.scripted_detections()), cal, Settings())
    res = sensitivity(tracks, Settings())
    counts = [r["events"] for r in res["proximity"]]
    assert counts == sorted(counts)
    assert sum(r["current"] for r in res["ttc"]) == 1
    assert next(r for r in res["ttc"] if r["current"])["events"] == 2         # matches the default run


# ------------------------------------------------------------------ tuning
def test_tuning_suggests_a_stricter_ttc_when_false_alarms_have_long_ttc():
    events = ([event(t=i, ttc=0.8, label="tp") for i in range(6)]
              + [event(t=10 + i, ttc=2.6, label="fp") for i in range(6)])
    t = tuning_curve(events, Settings())
    assert t["enough"] and t["suggest_ttc"] is not None and 0.75 <= t["suggest_ttc"] < 2.6
    row = next(r for r in t["ttc"] if r["ttc_cutoff_s"] == t["suggest_ttc"])
    assert row["precision"] == 1.0 and row["real_kept_share"] == 1.0


def test_tuning_with_few_labels_makes_no_suggestion():
    t = tuning_curve([event(label="tp"), event(t=9, label="fp")], Settings())
    assert not t["enough"] and t["suggest_ttc"] is None


# ------------------------------------------------------------------ car size hint
def test_car_size_hint_flags_a_scale_that_is_twice_too_big():
    rows = [(f, f * 0.1, i, "car", 0.9, 100 + 60 * i, 100.0, 100 + 60 * i + 45, 120.0)
            for f in range(10) for i in range(5)]                         # cars 45 px long
    det = Detections(rows)
    S = Settings(ground_point="center")
    good = Calibration.from_two_points([0, 0], [100, 0], 10.0, (1000, 800))   # 10 px/m -> 4.5 m cars
    bad = Calibration.from_two_points([0, 0], [100, 0], 20.0, (1000, 800))    # typed 20 m -> 9 m cars
    assert vehicle_size_check(det, good, S)["warning"] is None
    assert "re-check" in vehicle_size_check(det, bad, S)["warning"]
    assert vehicle_size_check(det, good, Settings()) is None                  # angled camera: no check


# ------------------------------------------------------------------ review pack
def test_review_pack_is_standalone_and_leaves_out_my_labels():
    ev = [event(label="tp"), event(t=9.0, eid="E002")]
    page = review_pack.build_pack_html("Main & 5th", ev, {"E001": "QUJD"}, explain_event, "abc")
    assert "<script" in page and "https://" not in page                      # nothing loaded from the internet
    assert "Main &amp; 5th" in page and "data:image/jpeg;base64,QUJD" in page
    assert '"tp"' not in page and "label\": \"tp" not in page                # my own label is not in the file


def test_reviewer_answers_are_validated():
    good = json.dumps({"format": review_pack.FORMAT, "reviewer": "Ms. Lee", "pack": "abc",
                       "events": [{"kind": "ped-veh", "t_peak": 5.0, "label": "fp"},
                                  {"kind": "ped-veh", "t_peak": 9.0, "label": "bogus"}]})
    got = review_pack.read_pack_answers(good.encode())
    assert got["reviewer"] == "Ms. Lee" and len(got["events"]) == 1
    with pytest.raises(ValueError):
        review_pack.read_pack_answers(b'{"events": []}')


def test_agreement_and_kappa():
    events = [event(t=float(i), eid=f"E{i:03d}") for i in range(10)]
    mine = [{"kind": "ped-veh", "t_peak": float(i), "label": "tp" if i < 5 else "fp"} for i in range(10)]
    same = review_pack.agreement(events, mine, mine)
    assert same["agree"] == 1.0 and same["kappa"] == 1.0 and same["disagreements"] == []
    theirs = [dict(m, label="tp" if i < 6 else "fp") for i, m in enumerate(mine)]     # one disagreement
    ag = review_pack.agreement(events, mine, theirs)
    assert ag["agree"] == 0.9 and 0.7 < ag["kappa"] < 0.9 and len(ag["disagreements"]) == 1
    assert review_pack.kappa_words(ag["kappa"]) == "substantial"
