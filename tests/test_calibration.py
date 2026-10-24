"""Calibration maths on a known rectangle: the homography must recover real distances."""
import numpy as np
import pytest

from engine.calibration import Calibration, CalibrationError

# A 10 m x 6 m rectangle seen at an angle (pixel corners clicked clockwise on screen: TL, TR, BR, BL).
PTS = [[300, 150], [700, 150], [820, 400], [180, 400]]
FRAME = (1000, 500)


def make():
    return Calibration.from_points(PTS, 10.0, 6.0, FRAME)


def test_corners_map_to_the_real_rectangle():
    cal = make()
    g, ok = cal.img_to_ground(PTS)
    assert ok.all()
    expected = np.array([[0, 0], [10, 0], [10, 6], [0, 6]], float)
    assert np.allclose(g, expected, atol=1e-6)


def test_side_lengths_are_recovered_in_meters():
    cal = make()
    assert cal.distance_between(PTS[0], PTS[1]) == pytest.approx(10.0, abs=1e-6)
    assert cal.distance_between(PTS[1], PTS[2]) == pytest.approx(6.0, abs=1e-6)
    # the diagonal is NOT used to build H, so it is an honest accuracy check
    assert cal.distance_between(PTS[0], PTS[2]) == pytest.approx(np.hypot(10, 6), abs=1e-6)


def test_counter_clockwise_clicking_gives_the_same_distances():
    """Clicking the corners in the other direction must not mirror or rescale the scene."""
    ccw = [PTS[0], PTS[3], PTS[2], PTS[1]]
    # going TL -> BL -> BR -> TR: first edge is the 6 m side, second edge the 10 m side
    cal = Calibration.from_points(ccw, 6.0, 10.0, FRAME)
    assert cal.distance_between(PTS[0], PTS[1]) == pytest.approx(10.0, abs=1e-6)
    assert cal.distance_between(PTS[1], PTS[2]) == pytest.approx(6.0, abs=1e-6)


def test_wrong_scale_is_detected_by_a_second_known_distance():
    """If the user types the wrong width, a check measurement exposes it as a large % error."""
    bad = Calibration.from_points(PTS, 12.0, 6.0, FRAME)   # claimed 12 m instead of 10 m
    measured = bad.distance_between(PTS[0], PTS[1])
    assert abs(100 * (measured - 10.0) / 10.0) > 10        # a >10% error is visible


def test_bowtie_and_degenerate_quads_are_rejected():
    bowtie = [[300, 150], [820, 400], [700, 150], [180, 400]]
    with pytest.raises(CalibrationError):
        Calibration.from_points(bowtie, 10, 6, FRAME)
    with pytest.raises(CalibrationError):
        Calibration.from_points(PTS, 0, 6, FRAME)
    with pytest.raises(CalibrationError):
        Calibration.from_points(PTS[:3], 10, 6, FRAME)


def test_json_round_trip_keeps_the_mapping():
    cal = make()
    again = Calibration.from_dict(cal.to_dict())
    p = [[512.0, 300.0]]
    assert np.allclose(cal.img_to_ground(p)[0], again.img_to_ground(p)[0])
