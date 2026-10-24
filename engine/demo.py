"""Synthetic demo: scripted detections for the scene in tools/make_synthetic_video.py.

HONESTY RULE: this checks the MATHS (calibration -> trajectories -> conflicts) against a known
answer. It says nothing about how well the detector works on real traffic.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEMO_VIDEO = ROOT / "sample_data" / "synthetic_intersection.mp4"
DEMO_TRUTH = ROOT / "sample_data" / "synthetic_intersection.truth.json"
DEMO_STRIDE = 3


def demo_available() -> bool:
    return DEMO_VIDEO.exists() and DEMO_TRUTH.exists()


def _scene():
    path = ROOT / "tools" / "make_synthetic_video.py"
    spec = importlib.util.spec_from_file_location("make_synthetic_video", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scripted_detections(seed: int = 3, stride: int = DEMO_STRIDE, jitter_px: float = 1.5, dropout: float = 0.03) -> List[tuple]:
    """Rows (frame, t, track_id, cls, conf, x1, y1, x2, y2) a perfect-but-noisy detector would produce."""
    scene = _scene()
    rng = np.random.default_rng(seed)
    Hgt = scene.camera_homography()
    rows: List[tuple] = []
    for k in range(int(scene.DURATION * scene.FPS)):
        if k % stride:
            continue
        t = k / scene.FPS
        for tid, (_n, cls, size, _c, wp) in enumerate(scene.ACTORS, start=1):
            box = scene.actor_box(Hgt, size, wp, t)
            if box is None or rng.random() < dropout:
                continue
            box = np.asarray(box) + rng.normal(0.0, jitter_px, 4)
            rows.append((k, t, tid, cls, float(rng.uniform(0.7, 0.95)), *map(float, box)))
    return rows


def actor_names() -> Dict[int, str]:
    return {i: a[0] for i, a in enumerate(_scene().ACTORS, start=1)}


def truth() -> dict:
    return json.loads(DEMO_TRUTH.read_text())
