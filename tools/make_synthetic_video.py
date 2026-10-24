"""
Synthetic intersection clip with KNOWN ground truth.

Why: real footage has no answer key. Here we simulate road users moving on a
ground plane (meters), project them through a known camera, and render a video.
The pipeline must then recover the right speeds, distances and conflicts.

Scenario (X = lateral, Y = along the road, meters; camera looks north from the south)
  carA  north-bound, 10 m/s, passes the crosswalk 1.0 s after ped1 crosses its lane -> SERIOUS car-ped conflict
  ped1  crosses west->east at 1.4 m/s
  carB  south-bound, 8 m/s, passes 3 s before ped2 crosses its lane                  -> NO conflict (control)
  ped2  crosses east->west at 1.4 m/s
  bike1 north-bound, 5 m/s, at X=3.5
  carC  north-bound, 12 m/s, overtakes bike1 with 1.4 m center-to-center             -> close pass, under 3 ft
  bike2 north-bound, 5 m/s, at X=3.5
  carD  north-bound, 12 m/s, overtakes bike2 with 2.8 m center-to-center             -> legal pass (control)

This is a TEST FIXTURE. Do not report results from it as accuracy of the real system.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
FPS = 30.0

# name, class, size (w_m, h_m for drawing), BGR color, waypoints [(t, X, Y)]
ACTORS = [
    ("carA", "car", (1.8, 1.5), (40, 40, 200), [(10.0, 2.0, 0.0), (16.0, 2.0, 60.0)]),
    ("ped1", "pedestrian", (0.5, 1.9), (30, 30, 30), [(5.29, -6.0, 20.0), (15.29, 8.0, 20.0)]),
    ("carB", "car", (1.8, 1.5), (200, 120, 30), [(31.0, -2.0, 60.0), (38.5, -2.0, 0.0)]),
    ("ped2", "pedestrian", (0.5, 1.9), (60, 20, 90), [(31.86, 8.0, 20.0), (43.29, -8.0, 20.0)]),
    ("bike1", "bicycle", (1.3, 1.5), (40, 160, 40), [(26.0, 3.5, 0.0), (38.0, 3.5, 60.0)]),
    ("carC", "car", (1.8, 1.5), (30, 180, 220), [(28.33, 2.1, 0.0), (33.33, 2.1, 60.0)]),
    ("bike2", "bicycle", (1.3, 1.5), (40, 160, 40), [(44.0, 3.5, 0.0), (56.0, 3.5, 60.0)]),
    ("carD", "car", (1.8, 1.5), (180, 60, 160), [(46.33, 0.7, 0.0), (51.33, 0.7, 60.0)]),
]
DURATION = 58.0

# calibration rectangle painted on the road (ground meters): X in [-4,4], Y in [10,30]
CAL_RECT = dict(x0=-4.0, x1=4.0, y0=10.0, y1=30.0)


def camera_homography() -> np.ndarray:
    """Ground (X,Y,0) -> image pixels for a pinhole camera 14 m up, 30 m south of the origin."""
    C = np.array([0.0, -30.0, 14.0])
    T = np.array([0.0, 25.0, 0.0])
    f = (T - C) / np.linalg.norm(T - C)
    right = np.cross(f, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    Rm = np.vstack([right, down, f])
    t = -Rm @ C
    fx = 780.0
    K = np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0 - 40], [0, 0, 1.0]])
    return K @ np.column_stack([Rm[:, 0], Rm[:, 1], t])


def project(Hgt: np.ndarray, X: float, Y: float) -> np.ndarray:
    p = Hgt @ np.array([X, Y, 1.0])
    return p[:2] / p[2]


def position(waypoints, t):
    ts = [w[0] for w in waypoints]
    if t < ts[0] or t > ts[-1]:
        return None
    return float(np.interp(t, ts, [w[1] for w in waypoints])), float(np.interp(t, ts, [w[2] for w in waypoints]))


def actor_box(Hgt: np.ndarray, size, waypoints, t: float):
    """Image box (x1,y1,x2,y2) of an actor at time t, with its bottom-center on the projected ground point."""
    pos = position(waypoints, t)
    if pos is None:
        return None
    X, Y = pos
    foot = project(Hgt, X, Y)
    scale = np.linalg.norm(project(Hgt, X + 0.5, Y) - foot) * 2.0  # px per meter here
    wpx, hpx = size[0] * scale, size[1] * scale
    return (foot[0] - wpx / 2, foot[1] - hpx, foot[0] + wpx / 2, foot[1])


def background(Hgt: np.ndarray) -> np.ndarray:
    img = np.full((H, W, 3), (95, 140, 95), np.uint8)  # grass

    def poly(pts, color):
        q = np.array([project(Hgt, *p) for p in pts], np.int32)
        cv2.fillConvexPoly(img, q, color, cv2.LINE_AA)

    poly([(-6, -40), (6, -40), (6, 90), (-6, 90)], (110, 110, 112))  # road
    for y in range(-40, 90, 6):  # dashed center line
        poly([(-0.08, y), (0.08, y), (0.08, y + 3), (-0.08, y + 3)], (200, 200, 200))
    for x in np.arange(-5.0, 5.5, 1.0):  # crosswalk stripes
        poly([(x - 0.25, 18.5), (x + 0.25, 18.5), (x + 0.25, 21.5), (x - 0.25, 21.5)], (225, 225, 225))
    r = CAL_RECT  # painted calibration box
    q = np.array([project(Hgt, *p) for p in [(r["x0"], r["y1"]), (r["x1"], r["y1"]), (r["x1"], r["y0"]), (r["x0"], r["y0"])]], np.int32)
    cv2.polylines(img, [q], True, (0, 220, 255), 2, cv2.LINE_AA)
    return img


def render(path: Path, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    Hgt = camera_homography()
    bg = background(Hgt)
    tmp = path.with_suffix(".raw.avi")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    n = int(DURATION * FPS)
    noise_bank = [np.rint(rng.normal(0, 3.0, (H, W, 3))).astype(np.int16) for _ in range(24)]
    for k in range(n):
        t = k / FPS
        frame = bg.copy()
        for name, cls, size, color, wp in ACTORS:
            box = actor_box(Hgt, size, wp, t)
            if box is None:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        noise = noise_bank[k % len(noise_bank)]
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        vw.write(frame)
    vw.release()
    # re-encode to H.264 MP4 if ffmpeg exists (smaller, and what real clips look like)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
            check=True,
        )
        tmp.unlink()
    except Exception:
        tmp.rename(path.with_suffix(".avi"))
        path = path.with_suffix(".avi")

    r = CAL_RECT
    corners_ground = [(r["x0"], r["y1"]), (r["x1"], r["y1"]), (r["x1"], r["y0"]), (r["x0"], r["y0"])]  # TL, TR, BR, BL on screen
    truth = {
        "video": str(path),
        "fps": FPS,
        "size": [W, H],
        "H_ground_to_image": Hgt.tolist(),
        "calibration_points_px": [project(Hgt, *p).tolist() for p in corners_ground],
        "calibration_width_m": r["x1"] - r["x0"],
        "calibration_length_m": r["y1"] - r["y0"],
        "actors": [{"name": a[0], "cls": a[1], "waypoints": a[4]} for a in ACTORS],
        "expected_events": [
            {"name": "carA-ped1", "kind": "ped-veh", "t_peak_approx": 12.0, "min_severity": "serious"},
            {"name": "carC-bike1", "kind": "bike-veh", "t_peak_approx": 30.0, "min_severity": "moderate", "below_3ft": True},
        ],
        "expected_no_event": ["carB-ped2", "carD-bike2"],
    }
    path.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2))
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make a synthetic intersection clip with ground truth.")
    ap.add_argument("out", nargs="?", default="synthetic_intersection.mp4")
    a = ap.parse_args()
    tr = render(Path(a.out))
    print("wrote", tr["video"], "and", Path(a.out).with_suffix(".truth.json"))
    print("calibration corners (px, clicked TL,TR,BR,BL):", [[round(v, 1) for v in p] for p in tr["calibration_points_px"]])
    print("calibration size: %.1f m x %.1f m" % (tr["calibration_width_m"], tr["calibration_length_m"]))
