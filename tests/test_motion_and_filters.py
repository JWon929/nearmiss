"""Camera-motion compensation, overhead calibration and the false-alarm filters, on cases with a known answer."""
import cv2
import numpy as np
import pytest

from detector_tracker import class_map
from engine.calibration import Calibration, CalibrationError
from engine.config import Settings
from engine.conflicts import detect_conflicts
from engine.motion import CameraMotion, to_reference
from engine.trajectories import Track, ground_pixels


# ------------------------------------------------------------------ camera motion
def textured(w=1280, h=720, seed=1):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (h // 8, w // 8), dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def test_camera_motion_recovers_a_known_drift_and_zoom():
    ref = textured()
    # the camera drifts right/down and zooms in 10 %: frame = warp(ref) by M, so frame -> ref is M^-1
    M = np.array([[1.1, 0.0, -90.0], [0.0, 1.1, -40.0], [0.0, 0.0, 1.0]])
    moved = cv2.warpPerspective(ref, M, (ref.shape[1], ref.shape[0]))
    cm = CameraMotion()
    cm.update(0, ref)
    H = cm.update(2, moved)
    pts = np.array([[300.0, 200.0], [900.0, 500.0], [640.0, 360.0]])
    seen = cv2.perspectiveTransform(pts[None], M)[0]                 # where the camera sees those road points now
    back = to_reference(seen, np.array([2, 2, 2]), cm.H)
    assert np.allclose(back, pts, atol=2.0)
    assert cm.summary((1280, 720))["max_shift_px"] > 50


def test_to_reference_without_motion_is_identity():
    p = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(to_reference(p, np.array([0, 5]), None), p)


# ------------------------------------------------------------------ overhead set-up
def test_ground_point_is_box_centre_for_overhead_and_bottom_for_angled():
    box = np.array([[100.0, 200.0, 140.0, 300.0]])
    assert ground_pixels(box, "bottom").tolist() == [[120.0, 300.0]]
    assert ground_pixels(box, "center").tolist() == [[120.0, 250.0]]


def test_one_known_distance_calibration_is_a_uniform_scale():
    cal = Calibration.from_two_points([100, 100], [300, 100], 10.0, (1000, 800))      # 20 px per metre
    assert cal.distance_between([100, 100], [300, 100]) == pytest.approx(10.0)
    assert cal.distance_between([500, 200], [500, 400]) == pytest.approx(10.0)          # same scale vertically
    assert cal.distance_between([0, 0], [300, 400]) == pytest.approx(25.0)
    with pytest.raises(CalibrationError):
        Calibration.from_two_points([100, 100], [101, 100], 10.0, (1000, 800))


def test_aerial_model_classes_map_to_vehicles_only():
    dota = {0: "plane", 1: "ship", 9: "large vehicle", 10: "small vehicle"}
    assert class_map(dota) == {9: "truck", 10: "car"}
    coco = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 7: "truck"}
    assert class_map(coco) == {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ------------------------------------------------------------------ false-alarm filters
DT = 0.1


def track(tid, cls, xy, v=None):
    """Track on the 10 Hz grid from positions; velocity from the positions unless given."""
    xy = np.asarray(xy, dtype=float)
    vel = np.gradient(xy, DT, axis=0) if v is None else np.asarray(v, dtype=float)
    group = {"car": "veh", "pedestrian": "ped", "bicycle": "bike"}[cls]
    t = np.arange(len(xy)) * DT
    return Track(tid, 0, cls, group, 0, t, xy, vel, np.linalg.norm(vel, axis=1), len(xy))


def straight(start, vel, n=60):
    return np.asarray(start, float) + np.arange(n)[:, None] * DT * np.asarray(vel, float)


def test_one_car_detected_twice_is_not_a_conflict():
    a = track(1, "car", straight([0, 0], [8, 0]))
    b = track(2, "car", straight([0.2, 0.1], [8, 0]))            # same car, second box
    stats = {}
    assert detect_conflicts([a, b], Settings(), stats) == []
    assert stats["filtered_duplicate_detection"] == 1


def test_driving_past_a_parked_car_at_constant_speed_is_not_flagged():
    # heading briefly points at a parked car 20 m ahead and 3 m to the side, then the road bends away
    xy = straight([0, 0], [9, 0], 40)
    vel = np.tile([9.0, 0.0], (40, 1))
    vel[5:15] = [9.0, 1.4]                                       # 0.1 s heading noise towards the parked car
    a = track(1, "car", xy, vel)
    parked = track(2, "car", np.tile([20.0, 3.0], (40, 1)), np.zeros((40, 2)))
    stats = {}
    assert detect_conflicts([a, parked], Settings(), stats) == []
    assert stats["filtered_no_evasive_action"] == 1


def test_braking_hard_towards_a_stopped_car_is_flagged():
    # 10 m/s towards a stopped car 25 m ahead, braking at 4 m/s^2 from t = 1 s: stops ~2.5 m short
    n, x, v, xs, vs = 50, 0.0, 10.0, [], []
    for k in range(n):
        xs.append(x)
        vs.append(v)
        if k * DT >= 1.0:
            v = max(0.0, v - 4.0 * DT)
        x += v * DT
    a = track(1, "car", np.c_[xs, np.zeros(n)], np.c_[vs, np.zeros(n)])
    stopped = track(2, "car", np.tile([xs[-1] + 4.5, 0.0], (n, 1)), np.zeros((n, 2)))
    ev = detect_conflicts([a, stopped], Settings())
    assert len(ev) == 1 and any("braked" in r for r in ev[0]["reasons"])


def test_close_following_between_moving_cars_is_flagged():
    lead = track(1, "car", straight([10, 0], [9, 0]))
    follow = track(2, "car", straight([0, 0], [12.5, 0]))      # closes at 3.5 m/s from 10 m
    ev = detect_conflicts([lead, follow], Settings())
    assert len(ev) == 1 and ev[0]["ttc_min_s"] is not None


def test_overhead_trucks_are_told_apart_by_length_not_by_the_model():
    from detector_tracker import size_classes
    widths = np.array([50, 52, 48, 51, 49, 55])            # typical car ~ 2.5 x median 50.5 = 126 px long
    lengths = np.array([120, 130, 150, 60, 260, 210])      # SUV 150 and a cut-off car 60 stay cars
    names, ref = size_classes(lengths, widths, None)
    assert names == ["car", "car", "car", "car", "truck", "truck"] and ref == pytest.approx(126.25)
    names, _ = size_classes(np.array([300]), np.array([60]), ref)   # too few to re-measure: reuse ref
    assert names == ["truck"]


def test_visdrone_model_classes_include_people_and_merge_vans_into_cars():
    from detector_tracker import finds_people
    visdrone = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van", 5: "truck", 6: "tricycle",
                7: "awning-tricycle", 8: "bus", 9: "motor"}
    m = class_map(visdrone)
    assert m[0] == m[1] == "person" and m[3] == m[4] == "car" and m[9] == "motorcycle" and m[8] == "bus"
    assert finds_people("nearmiss-visdrone.pt") and not finds_people("yolo11n-obb.pt")


def test_track_label_does_not_flip_on_a_close_vote():
    from detector_tracker import Detection, PersistentTracks
    mem = PersistentTracks(coast_frames=5, trail_frames=5)
    seq = ["car", "car", "truck", "truck", "car", "truck", "truck"]       # an SUV the model cannot decide on
    shown = []
    for n, c in enumerate(seq):
        items = mem.update(n, [Detection(n, n / 10, 7, c, 0.5, 0, 0, 10, 20)], (100, 100))
        shown.append(items[0].class_name)
    assert set(shown) == {"car"}                                             # 4 truck votes vs 3 car: not 1.5x
    for n in range(7, 12):
        items = mem.update(n, [Detection(n, n / 10, 7, "truck", 0.5, 0, 0, 10, 20)], (100, 100))
    assert items[0].class_name == "truck"                                    # a clear majority does switch
