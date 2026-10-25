"""
Compare detection models on one clip, through the same pipeline the app uses (tracking + camera-motion
compensation), so you can decide which model NearMiss should use.

There is no answer key for your own footage, so these are evidence, not accuracy:
  * road users per frame, by class      more is better only if they are real: check the snapshot
  * tracks of 1 s or more               road users followed long enough to analyse
  * moving tracks                       tracks that travel more than 3 car lengths (the ones conflicts come from)
  * label switches                      times a track's per-frame class changed (car <-> truck flicker)
A side-by-side snapshot of the same frame is saved for each model. Look at it: count the vehicles in the
intersection yourself and see which model finds them.

Usage:  python tools/compare_models.py CLIP.mp4 OUT_DIR MODEL [MODEL ...] [--seconds 30] [--snapshot 12.5]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector_tracker import RenderOptions, RoadUserTracker, TrackerSettings, auto_stride  # noqa: E402
from engine.motion import to_reference  # noqa: E402
from video_utils import read_metadata  # noqa: E402


def evaluate(model: str, clip: str, seconds: float, snapshot_s: float, out: Optional[Path] = None,
             device: str = "auto", stabilize: bool = True, imgsz: int = 960, blur: bool = False,
             progress: Optional[Callable[[float], None]] = None) -> dict:
    """Run one model on the first `seconds` of the clip. Also used by the app's model comparison (Detect page)."""
    meta = read_metadata(clip)
    stride = auto_stride(meta["fps"])
    tracker = RoadUserTracker(TrackerSettings(model_name=model, stride=stride, stabilize=stabilize, device=device,
                                              imgsz=imgsz),
                              RenderOptions(blur_privacy=blur, privacy_level="strict", show_confidence=False))
    per_class, frames = Counter(), 0
    seen = defaultdict(list)                      # track -> [(frame, class, x, y)]
    snap = None
    t0 = time.perf_counter()
    for res in tracker.process_video(clip, meta["fps"]):
        if res.time_s > seconds:
            break
        frames += 1
        if progress:
            progress(min(res.time_s / seconds, 1.0))
        for d in res.detections:
            per_class[d.class_name] += 1
            seen[d.track_id].append((d.frame, d.class_name, (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2))
        if snap is None and res.time_s >= snapshot_s:
            snap = res.annotated.copy()
            cv2.rectangle(snap, (0, snap.shape[0] - 40), (snap.shape[1], snap.shape[0]), (0, 0, 0), -1)
            cv2.putText(snap, Path(model).name, (12, snap.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if out is not None:
                cv2.imwrite(str(out / f"snapshot_{Path(model).stem}.jpg"), snap)
    took = time.perf_counter() - t0
    motion = tracker.motion.H if tracker.motion else None
    min_len = max(3, int(round(1.0 * meta["fps"] / stride)))
    tracks = {t: v for t, v in seen.items() if len(v) >= min_len}
    switches = sum(sum(a[1] != b[1] for a, b in zip(v, v[1:])) for v in tracks.values())
    car_px = 0.04 * max(meta["width"], meta["height"])   # rough car length in a drone shot of an intersection
    moving = 0
    for v in tracks.values():
        p = to_reference(np.array([[r[2], r[3]] for r in v]), np.array([r[0] for r in v]), motion)
        if np.linalg.norm(p[-1] - p[0]) > 3 * car_px:
            moving += 1
    return {"model": tracker.model_used, "warning": tracker.warning, "frames": frames,
            "per_frame": {k: round(n / max(frames, 1), 1) for k, n in per_class.most_common()},
            "tracks_1s": len(tracks), "moving_tracks": moving, "label_switches": switches,
            "device": tracker.device_used, "ms_per_frame": round(1000 * took / max(frames, 1)), "snapshot": snap}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seconds", type=float, default=1e9)
    ap.add_argument("--snapshot", type=float, default=10.0, help="time (s) of the side-by-side snapshot")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [evaluate(m, a.clip, a.seconds, a.snapshot, out) for m in a.models]
    print(f"\n{'model':28s} {'tracks>=1s':>10s} {'moving':>7s} {'label switches':>15s}   road users per frame")
    for r in rows:
        print(f"{r['model']:28s} {r['tracks_1s']:>10d} {r['moving_tracks']:>7d} {r['label_switches']:>15d}   {r['per_frame']}")
        if r["warning"]:
            print("   warning:", r["warning"])
    snaps = [cv2.imread(str(out / f"snapshot_{Path(m).stem}.jpg")) for m in a.models]
    snaps = [s for s in snaps if s is not None]
    if snaps:
        h = min(s.shape[0] for s in snaps)
        cv2.imwrite(str(out / "side_by_side.jpg"),
                    np.vstack([cv2.resize(s, (int(s.shape[1] * h / s.shape[0]), h)) for s in snaps]))
        print("\nSnapshots:", out / "side_by_side.jpg")


if __name__ == "__main__":
    main()
