"""Slow-motion near-miss video: windows merge correctly and the output is longer by the expected amount."""
import cv2
import numpy as np
import pytest

from engine.slowmo import make_slowmo, slow_windows


def ev(eid, t0, t1, sev="serious"):
    return {"id": eid, "t_start": t0, "t_end": t1, "t_peak": (t0 + t1) / 2, "kind_label": "Vehicle-Pedestrian",
            "severity": sev, "severity_level": 3}


def test_overlapping_windows_merge():
    w = slow_windows([ev("E1", 5, 6), ev("E2", 7, 8), ev("E3", 20, 21)], before_s=2, after_s=1)
    assert [(a, b, [e["id"] for e in es]) for a, b, es in w] == [(3, 9, ["E1", "E2"]), (18, 22, ["E3"])]


@pytest.fixture
def clip(tmp_path):
    path = str(tmp_path / "in.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(100):                                          # 10 s at 10 fps, a square moving right
        f = np.zeros((120, 160, 3), np.uint8)
        cv2.rectangle(f, (i, 50), (i + 10, 60), (255, 255, 255), -1)
        out.write(f)
    out.release()
    return path


def frames(path):
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.read()[0]:
        n += 1
    return n


def test_slowed_window_is_written_factor_times(clip, tmp_path):
    out = str(tmp_path / "slow.mp4")
    info = make_slowmo(clip, out, [ev("E1", 4.0, 5.0)], factor=4, before_s=1.0, after_s=1.0)
    # window 3.0-6.0 s = 31 frames at 10 fps -> 31 x 4; the other 69 frames once
    assert frames(out) == 69 + 31 * 4 and info["windows"] == 1


def test_highlights_only_keeps_just_the_slowed_parts(clip, tmp_path):
    out = str(tmp_path / "hl.mp4")
    make_slowmo(clip, out, [ev("E1", 4.0, 5.0)], factor=2, before_s=1.0, after_s=1.0, highlights_only=True)
    assert frames(out) == 31 * 2
