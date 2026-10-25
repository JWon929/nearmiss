"""PET flagging, DRAC, threshold presets, fitting the scale to several known distances, and the multi-check status."""
import numpy as np
import pytest

from engine.calibration import Calibration, fit_scale
from engine.config import PRESETS, Settings, apply_preset, matching_preset
from engine.conflicts import detect_conflicts
from engine.insights import explain_event
from engine.quality import calibration_status
from engine.trajectories import Track

DT = 0.1


def track(tid, cls, start, vel, k0=0, n=60):
    xy = np.asarray(start, float) + np.arange(n)[:, None] * DT * np.asarray(vel, float)
    v = np.tile(np.asarray(vel, float), (n, 1))
    group = {"car": "veh", "pedestrian": "ped", "bicycle": "bike"}[cls]
    return Track(tid, 0, cls, group, k0, (k0 + np.arange(n)) * DT, xy, v, np.linalg.norm(v, axis=1), n)


def crossing_after(start_s):
    """A car drives east through (20, 0) at t = 2 s and leaves the scene at 2.3 s. A pedestrian appears at `start_s`
    1 m south of that point and walks north across it, so the two are never in the scene together."""
    car = track(1, "car", [0, 0], [10, 0], k0=0, n=24)
    ped = track(2, "pedestrian", [20, -1.0], [0, 1.4], k0=int(round(start_s / DT)), n=40)
    return car, ped


def test_pet_flags_a_crossing_the_ttc_check_cannot_see():
    car, ped = crossing_after(2.5)
    assert detect_conflicts([car, ped], Settings()) == []                    # never in the scene together
    ev = detect_conflicts([car, ped], Settings(pet_flagging=True))
    assert len(ev) == 1 and ev[0]["flag_source"] == "pet" and ev[0]["kind"] == "ped-veh"
    # car's back clears the point at 2.0 + 2.3/10 = 2.23 s; the pedestrian's front reaches it ~0.5 s after 2.5 s
    assert ev[0]["pet_s"] == pytest.approx(0.77, abs=0.25) and ev[0]["severity"] == "serious"
    assert "crossing point" in explain_event(ev[0]) and "came within" not in explain_event(ev[0])


def test_long_pet_is_not_flagged():
    car, ped = crossing_after(8.0)
    assert detect_conflicts([car, ped], Settings(pet_flagging=True)) == []


def test_drac_is_reported_and_can_raise_the_severity():
    lead = track(1, "car", [10, 0], [9, 0])
    follow = track(2, "car", [0, 0], [12.5, 0])                              # closes at 3.5 m/s from 10 m
    base = detect_conflicts([lead, follow], Settings())[0]
    assert base["drac_max_mps2"] > 0
    S = Settings(drac_flagging=True, drac_mps2=[0.5, 1.0, 1.5])
    raised = detect_conflicts([lead, follow], S)[0]
    assert raised["severity_level"] >= base["severity_level"] and raised["severity_from_drac"] != "none"


def test_presets_change_only_thresholds_and_are_recognised():
    S = Settings(ground_point="center", privacy_level="strict", keep_tracks=[4])
    for name in PRESETS:
        P = apply_preset(S, name)
        assert (P.ground_point, P.privacy_level, P.keep_tracks) == ("center", "strict", [4])
        assert matching_preset(P) == name
        assert P.ttc_serious_s < P.ttc_moderate_s < P.ttc_candidate_s
    P = apply_preset(S, "Parking lot or slow street")
    P.ttc_candidate_s = 5.5
    assert matching_preset(P) is None                                        # changed by hand


def test_fit_scale_corrects_a_mistyped_length_and_reports_an_honest_error():
    pts = [[300, 150], [700, 150], [820, 400], [180, 400]]
    cal = Calibration.from_points(pts, 12.0, 7.2, (1000, 500))               # true 10 x 6 m, typed 20% too big
    checks = [{"p1": pts[0], "p2": pts[1], "known_m": 10.0}, {"p1": pts[1], "p2": pts[2], "known_m": 6.0},
              {"p1": pts[0], "p2": pts[2], "known_m": float(np.hypot(10, 6))}]
    fit = fit_scale(cal, checks)
    assert fit["scale"] == pytest.approx(1 / 1.2, rel=1e-6)
    assert fit["rms_before_pct"] == pytest.approx(20.0, abs=0.1) and fit["rms_after_pct"] < 0.01
    assert fit["rms_leave_one_out_pct"] < 0.01
    fixed = cal.rescaled(fit["scale"])
    assert fixed.distance_between(pts[0], pts[1]) == pytest.approx(10.0, rel=1e-6)


def test_status_with_several_checks_uses_their_typical_error():
    ok = [{"known_m": 3.5, "measured_m": 3.6, "error_pct": 2.9}, {"known_m": 10, "measured_m": 9.8, "error_pct": -2.0}]
    assert calibration_status({"checks": ok}, {"warning": None})["level"] == "good"
    bad = ok + [{"known_m": 5, "measured_m": 6.5, "error_pct": 30.0}]
    assert calibration_status({"checks": bad}, {"warning": None})["level"] == "problem"
