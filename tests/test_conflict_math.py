"""The closest-approach / TTC maths from the plan, checked against hand-worked answers."""
import numpy as np
import pytest

from engine.conflicts import closest_approach, ttc_to_radius


def arr(*rows):
    return np.array(rows, dtype=float)


def test_head_on_ttc():
    # 10 m apart, closing at 5 m/s, collision radius 2 m: reaches 2 m after (10-2)/5 = 1.6 s
    ttc = ttc_to_radius(arr([10, 0]), arr([-5, 0]), 2.0)
    assert ttc[0] == pytest.approx(1.6)


def test_moving_apart_or_parallel_never_collides():
    assert np.isinf(ttc_to_radius(arr([10, 0]), arr([5, 0]), 2.0)[0])      # separating
    assert np.isinf(ttc_to_radius(arr([10, 5]), arr([-5, 0]), 2.0)[0])     # passes 5 m to the side
    assert np.isinf(ttc_to_radius(arr([10, 0]), arr([0, 0]), 2.0)[0])      # no relative motion


def test_already_inside_radius_is_zero():
    assert ttc_to_radius(arr([1, 0]), arr([-1, 0]), 2.0)[0] == 0.0


def test_glancing_path_uses_circle_geometry():
    # offset 1.2 m, radius 2 m: enters the circle when forward distance = sqrt(2^2 - 1.2^2) = 1.6 m
    # so r_forward = 10 -> (10 - 1.6)/4 = 2.1 s
    ttc = ttc_to_radius(arr([10, 1.2]), arr([-4, 0]), 2.0)
    assert ttc[0] == pytest.approx((10 - 1.6) / 4.0)


def test_closest_approach_formula():
    # r = (10, 3), v = (-2, 0): t* = -(r.v)/|v|^2 = 5 s, d* = 3 m
    t_star, d_star = closest_approach(arr([10, 3]), arr([-2, 0]))
    assert t_star[0] == pytest.approx(5.0)
    assert d_star[0] == pytest.approx(3.0)


def test_closest_approach_with_no_relative_motion_is_nan_time():
    t_star, d_star = closest_approach(arr([4, 3]), arr([0, 0]))
    assert np.isnan(t_star[0])
    assert d_star[0] == pytest.approx(5.0)
