"""The vectorised versions must give the same numbers as the straightforward loops they replaced."""
import numpy as np

from engine.motion import to_reference
from engine.trajectories import local_linear


def local_linear_loop(t, p, half):
    """The original per-sample least-squares fit (reference implementation)."""
    n = len(t)
    pos, vel = np.empty_like(p), np.zeros_like(p)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        m = hi - lo
        if m < 3:
            pos[i] = p[i]
            continue
        tt, pp = t[lo:hi] - t[i], p[lo:hi]
        st, stt = tt.sum(), (tt * tt).sum()
        den = m * stt - st * st
        if den <= 1e-12:
            pos[i] = p[i]
            continue
        sp, stp = pp.sum(axis=0), (tt[:, None] * pp).sum(axis=0)
        b = (m * stp - st * sp) / den
        pos[i], vel[i] = (sp - b * st) / m, b
    return pos, vel


def test_vectorised_local_linear_matches_the_loop():
    rng = np.random.default_rng(3)
    for n, half in ((1, 4), (2, 4), (3, 1), (7, 4), (300, 4), (50, 10)):
        t = 412.3 + np.arange(n) * 0.1                 # late in a long video: large absolute times
        p = np.c_[np.cumsum(rng.normal(1, 0.3, n)), rng.normal(0, 2, n)]
        a, b = local_linear(t, p, half), local_linear_loop(t, p, half)
        assert np.allclose(a[0], b[0], atol=1e-7) and np.allclose(a[1], b[1], atol=1e-6)


def to_reference_loop(points, frames, motion):
    keys = np.array(sorted(motion))
    out = np.empty_like(points)
    for f in np.unique(frames):
        sel = frames == f
        k = keys[np.argmin(np.abs(keys - f))]
        ph = np.c_[points[sel], np.ones(sel.sum())] @ motion[int(k)].T
        out[sel] = ph[:, :2] / ph[:, 2:3]
    return out


def test_vectorised_to_reference_matches_the_loop_including_missing_frames():
    rng = np.random.default_rng(5)
    motion = {f: np.array([[1 + 0.001 * f, 0.0005 * f, 2.0 * f], [0, 1, -1.0 * f], [1e-6 * f, 0, 1]])
              for f in range(0, 200, 3)}                 # only every 3rd frame has a transform
    frames = rng.integers(0, 230, 500)                   # including frames past the last transform
    pts = rng.uniform(0, 1000, (500, 2))
    assert np.allclose(to_reference(pts, frames, motion), to_reference_loop(pts, frames, motion))
    one = {7: motion[6]}
    assert np.allclose(to_reference(pts, frames, one), to_reference_loop(pts, frames, one))
