"""Regression tests for the reviewed bugs: label matching, de-duplication, PET, closest distance, waiting
pedestrians, rate denominators, label validation, movements, signal phases and proposed designs."""
import numpy as np
import pytest

from engine import validation as v
from engine.analysis import distance_series, exposure
from engine.calibration import Calibration
from engine.config import Settings
from engine.conflicts import _dedupe, detect_conflicts, post_encroachment_time
from engine.design import demo_alternatives, evaluate, read_trajectories
from engine.movements import classify_track, parse_phases, phase_at
from engine.quality import calibration_status
from engine.trajectories import Detections, Track, build_tracks

DT = 0.1


def track(tid, cls, xy, t0=0.0, v=None):
    xy = np.asarray(xy, dtype=float)
    vel = np.gradient(xy, DT, axis=0) if v is None else np.asarray(v, dtype=float)
    group = {"car": "veh", "pedestrian": "ped", "bicycle": "bike"}[cls]
    t = t0 + np.arange(len(xy)) * DT
    return Track(tid, 0, cls, group, int(round(t0 / DT)), t, xy, vel, np.linalg.norm(vel, axis=1), len(xy))


def straight(start, vel, n, dt=DT):
    return np.asarray(start, float) + np.arange(n)[:, None] * dt * np.asarray(vel, float)


def event(kind, t, a, b, sev="serious", level=3, ttc=None, x=0.0, y=0.0):
    return {"kind": kind, "t_peak": t, "severity": sev, "severity_level": level, "ttc_min_s": ttc, "x": x, "y": y,
            "a": {"track": a, "seg": 0}, "b": {"track": b, "seg": 0}, "merged_duplicates": 0}


# ------------------------------------------------------------------ 1. labels
def test_one_label_is_never_counted_for_two_events_a_moment_apart():
    events = [event("ped-veh", 5.0, 1, 2), event("ped-veh", 5.6, 3, 4)]
    store = {}
    v.set_label(store, "ped-veh", 5.0, "tp", pair=[1, 2], events=events)
    s = v.summarize(events, store)
    assert (s["tp"], s["unreviewed"]) == (1, 1)
    v.apply_labels(events, store)
    assert [e["label"] for e in events] == ["tp", None]


def test_labelling_the_second_event_does_not_overwrite_the_first():
    events = [event("ped-veh", 5.0, 1, 2), event("ped-veh", 5.6, 3, 4)]
    store = {}
    v.set_label(store, "ped-veh", 5.0, "tp", pair=[1, 2], events=events)
    v.set_label(store, "ped-veh", 5.6, "fp", pair=[3, 4], events=events)
    v.apply_labels(events, store)
    assert [e["label"] for e in events] == ["tp", "fp"] and len(store["events"]) == 2


def test_old_labels_without_road_users_still_match_one_event_each():
    events = [event("ped-veh", 5.0, 1, 2), event("ped-veh", 5.6, 3, 4)]
    store = {"events": [{"kind": "ped-veh", "t_peak": 5.3, "label": "tp"}]}
    assert v.summarize(events, store)["tp"] == 1


def test_malformed_label_files_are_cleaned_not_trusted():
    store, problems = v.clean_store({"events": [{"kind": "ped-veh", "t_peak": "x", "label": "tp"},
                                                {"kind": "ped-veh", "t_peak": 3, "label": "maybe"},
                                                {"kind": "veh-veh", "t_peak": 4, "label": "fp", "pair": [9, 2]}],
                                     "missed": [{"t": 7}, {"note": "no time"}]})
    assert len(store["events"]) == 1 and store["events"][0]["pair"] == [2, 9] and len(store["missed"]) == 1
    assert len(problems) == 3
    assert v.clean_store([1, 2])[1]


# ------------------------------------------------------------------ 2. de-duplication
def test_independent_encounters_at_the_same_place_and_time_are_both_kept():
    kept = _dedupe([event("veh-veh", 5.0, 1, 2), event("veh-veh", 5.2, 3, 4, x=1.0)], Settings())
    assert len(kept) == 2


def test_an_id_switch_duplicate_keeps_the_worse_event_and_ttc_zero_counts_as_worst():
    a = event("veh-veh", 5.0, 1, 2, ttc=1.0)
    b = event("veh-veh", 5.2, 1, 7, ttc=0.0)          # same car 1, the other car got a new ID
    kept = _dedupe([a, b], Settings())
    assert len(kept) == 1 and kept[0]["ttc_min_s"] == 0.0


# ------------------------------------------------------------------ 5. PET
def test_pet_runs_from_the_first_body_clearing_to_the_second_arriving():
    car = track(1, "car", straight([-20, 0], [10, 0], 60))               # centre at the origin at t = 2.0 s
    ped = track(2, "pedestrian", straight([0, -7.5], [0, 1.5], 80))      # centre at the origin at t = 5.0 s
    pet = post_encroachment_time(car, ped, 1.5)
    # car's back (2.3 m behind its centre) clears at 2.23 s; pedestrian's front (0.3 m) arrives at 4.8 s
    assert pet == pytest.approx(5.0 - 0.3 / 1.5 - (2.0 + 2.3 / 10), abs=0.05)


def test_pet_is_zero_when_both_are_in_the_conflict_area_at_once():
    car = track(1, "car", straight([-20, 0], [10, 0], 60))
    ped = track(2, "pedestrian", straight([0, -3.0], [0, 1.5], 60))      # at the origin at t = 2.0 s too
    assert post_encroachment_time(car, ped, 1.5) == 0.0


# ------------------------------------------------------------------ 6. closest distance agrees with the chart
def test_reported_closest_distance_is_the_minimum_of_the_distance_chart():
    lead = track(1, "car", straight([12, 0], [8, 0], 80))
    follow = track(2, "car", straight([0, 0.5], [11, 0], 80))           # closes from 12 m at 3 m/s
    events = detect_conflicts([lead, follow], Settings())
    assert events
    for e in events:
        ds = distance_series([lead, follow], e)
        assert min(ds["distance_m"]) == pytest.approx(e["d_min_m"], abs=0.011)
        assert ds["t"][int(np.argmin(ds["distance_m"]))] == pytest.approx(e["t_peak"], abs=0.051)


# ------------------------------------------------------------------ 7. waiting pedestrians
CAL = Calibration.from_two_points([0, 0], [100, 0], 10.0, (1000, 1000))


def waiting_pedestrian(n=40):
    return [(k, k * 0.1, 1, "pedestrian", 0.9, 295.0, 295.0, 305.0, 305.0) for k in range(n)]


def test_a_waiting_pedestrian_is_kept_for_street_level_cameras():
    S = Settings()                                   # ground_point "bottom": a camera looking at an angle
    tracks, stats = build_tracks(Detections(waiting_pedestrian()), CAL, S)
    assert len(tracks) == 1 and stats["dropped_static_clutter"] == 0


def test_a_still_pedestrian_from_above_can_be_restored_by_the_user():
    S = Settings()
    S.ground_point = "center"
    tracks, stats = build_tracks(Detections(waiting_pedestrian()), CAL, S)
    assert not tracks and stats["static_clutter_ids"] == [1]
    S.keep_tracks = [1]
    tracks, _ = build_tracks(Detections(waiting_pedestrian()), CAL, S)
    assert len(tracks) == 1


# ------------------------------------------------------------------ 8. calibration status
def test_severities_are_provisional_until_the_calibration_is_checked():
    assert calibration_status({"checks": []})["provisional"]
    assert calibration_status({"checks": [{"error_pct": 25.0, "known_m": 3.5, "measured_m": 4.4}]})["level"] == "problem"
    assert calibration_status({"checks": [{"error_pct": 2.0, "known_m": 3.5, "measured_m": 3.6}]},
                              {"warning": "Cars measure 9 m"})["level"] == "problem"
    good = calibration_status({"checks": [{"error_pct": 2.0, "known_m": 3.5, "measured_m": 3.6}]}, {"warning": None})
    assert good["level"] == "good" and not good["provisional"]


# ------------------------------------------------------------------ 9. rate denominators
def test_events_per_hour_use_the_observation_period_not_the_track_span():
    car = track(1, "car", straight([0, 0], [10, 0], 100))                # 10 s of traffic in a 10 min recording
    ev = [{"kind": "veh-veh"}]
    assert exposure([car], ev, DT, observation_s=600.0)["events_per_hour"] == 6.0
    assert exposure([car], ev, DT)["events_per_hour"] > 300                # what the old span-based rate said


# ------------------------------------------------------------------ 11. movements and phases
def test_turning_movements_and_approach_directions():
    # map frame: y down like the screen, so "up the screen" (north) is -y
    north_then_east = np.r_[straight([0, 0], [0, -8], 30), straight([0, -24], [8, 0], 30)[1:]]
    north_then_west = np.r_[straight([0, 0], [0, -8], 30), straight([0, -24], [-8, 0], 30)[1:]]
    assert classify_track(track(1, "car", north_then_east))["label"] == "NB right"
    assert classify_track(track(2, "car", north_then_west))["label"] == "NB left"
    assert classify_track(track(3, "car", straight([0, 0], [10, 0], 40)))["label"] == "EB through"
    # north to the right of the screen: moving right is northbound
    assert classify_track(track(4, "car", straight([0, 0], [10, 0], 40)), north_deg=90)["label"] == "NB through"
    assert classify_track(track(5, "car", np.zeros((40, 2))))["label"] == "stationary"


def test_signal_phases_are_parsed_and_bad_lines_reported():
    phases, problems = parse_phases("0-28 NS green\n28-31 NS yellow\nnonsense\n40-35 backwards")
    assert phases == [(0.0, 28.0, "NS green"), (28.0, 31.0, "NS yellow")] and len(problems) == 2
    assert phase_at(29.0, phases) == "NS yellow" and phase_at(50.0, phases) is None


# ------------------------------------------------------------------ 12. proposed designs
def test_design_alternatives_are_compared_with_the_same_engine():
    alts = demo_alternatives()
    existing = evaluate(read_trajectories(alts["existing"].encode())[0], Settings())
    changed = evaluate(read_trajectories(alts["lpi_and_bike_buffer"].encode())[0], Settings())
    assert existing["kinds"]["ped-veh"] == 1 and existing["kinds"]["bike-veh"] == 1
    assert sum(changed["kinds"].values()) == 0


def test_bad_trajectory_files_are_explained():
    rows, problems = read_trajectories(b"time,x,y\n1,2,3\n")
    assert not rows and "Missing column" in problems[0]
    rows, problems = read_trajectories(b"t_s,id,class,x_m,y_m\n0,a,car,0,0\nx,a,car,1,1\n0,b,tram,0,0\n")
    assert len(rows) == 1 and len(problems) == 2


def test_imported_labels_are_merged_one_per_event():
    store = {"events": [{"kind": "veh-veh", "t_peak": 3.0, "label": "fp", "note": "", "pair": [1, 2]}], "missed": []}
    incoming, _ = v.clean_store({"events": [{"kind": "veh-veh", "t_peak": 3.1, "label": "tp", "pair": [1, 2]},
                                            {"kind": "veh-veh", "t_peak": 3.2, "label": "unsure", "pair": [5, 6]}],
                                 "missed": [{"t": 7.0}, {"t": 7.0}]})
    assert v.merge_labels(store, incoming) == (2, 1)
    events = [event("veh-veh", 3.0, 1, 2), event("veh-veh", 3.2, 5, 6)]
    v.apply_labels(events, store)
    assert [e["label"] for e in events] == ["tp", "unsure"]


def test_a_path_that_ends_near_where_it_turned_back_does_not_crash():
    # 4 m out, then 2 m back: every point is within 3 m of the end, yet it travelled 3 m from the start
    xy = np.r_[straight([0, 0], [8, 0], 6), straight([4.0, 0], [-8, 0], 4)[1:]]
    assert classify_track(track(1, "car", xy))["label"] in ("EB through", "EB U-turn", "stationary")
