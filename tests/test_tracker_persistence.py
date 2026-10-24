"""The original bug: boxes flashed on for a moment. These tests check the carry-forward logic."""
from detector_tracker import Detection, PersistentTracks

WH = (1000, 600)


def det(tid, x, frame=0):
    return Detection(frame, frame / 10, tid, "car", 0.6, x, 100, x + 80, 160)


def test_box_stays_visible_after_the_detector_loses_it():
    pt = PersistentTracks(coast_frames=10, trail_frames=20, min_hits=3)
    for f in range(5):
        pt.update(f, [det(1, 100 + 10 * f, f)], WH)
    seen = [pt.update(f, [], WH) for f in range(5, 12)]
    assert all(len(items) == 1 and not items[0].live for items in seen[:6])   # coasts for 6 frames
    assert pt.update(16, [], WH) == []                                              # then disappears


def test_coasting_box_keeps_moving_in_the_same_direction():
    pt = PersistentTracks(coast_frames=10, trail_frames=20, min_hits=3)
    for f in range(5):
        last = pt.update(f, [det(1, 100 + 10 * f, f)], WH)
    x_seen = last[0].box[0]
    x_coast = pt.update(5, [], WH)[0].box[0]
    assert x_coast > x_seen


def test_flickery_one_off_detection_does_not_leave_a_ghost():
    pt = PersistentTracks(coast_frames=10, trail_frames=20, min_hits=3)
    pt.update(0, [det(7, 300, 0)], WH)
    assert pt.update(1, [], WH) == []


def test_track_that_leaves_the_frame_is_dropped():
    pt = PersistentTracks(coast_frames=30, trail_frames=20, min_hits=3)
    for f in range(5):
        pt.update(f, [det(1, 700 + 60 * f, f)], WH)
    gone = [pt.update(f, [], WH) for f in range(5, 25)]
    assert gone[-1] == []
