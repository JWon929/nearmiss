"""
Time where NearMiss spends its time, so speed-ups are measured, not guessed.

  python tools/benchmark.py analysis [--tracks 400] [--minutes 10]
      Analysis only (no model): a synthetic recording with many road users, with and without camera motion.

  python tools/benchmark.py detect VIDEO [--seconds 30] [--model yolov8n.pt] [--stabilize]
      The detection loop on the first seconds of a real clip, split into model, camera motion, drawing and writing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def synthetic(n_tracks: int, minutes: float, seed: int = 0):
    """Straight-line road users with box jitter, 12 fps, 1920x1080, spread over the recording."""
    from engine.trajectories import Detections

    rng = np.random.default_rng(seed)
    fps, dur = 12.0, minutes * 60
    rows = []
    kinds = rng.choice(["car", "pedestrian", "bicycle", "truck"], n_tracks, p=[0.6, 0.25, 0.1, 0.05])
    for tid in range(1, n_tracks + 1):
        cls = kinds[tid - 1]
        t0 = rng.uniform(0, dur - 30)
        life = rng.uniform(8, 30)
        start = rng.uniform([200, 300], [1700, 1000])
        vel = rng.normal(0, 60 if cls in ("car", "truck") else 15, 2)
        w, h = {"car": (90, 60), "truck": (140, 90), "pedestrian": (25, 60), "bicycle": (40, 55)}[cls]
        for f in range(int(t0 * fps), int((t0 + life) * fps)):
            t = f / fps
            cx, cy = start + vel * (t - t0) + rng.normal(0, 1.5, 2)
            if not (0 < cx < 1920 and 250 < cy < 1080):
                continue
            rows.append((f, t, tid, str(cls), 0.8, cx - w / 2, cy - h, cx + w / 2, cy))
    rows.sort(key=lambda r: r[0])
    return Detections(rows), int(dur * fps)


def timed(label, fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    print(f"  {label:<34s} {time.perf_counter() - t:7.2f} s")
    return out


def bench_analysis(n_tracks: int, minutes: float) -> None:
    from engine.calibration import Calibration
    from engine.config import Settings
    from engine.conflicts import detect_conflicts
    from engine.insights import sensitivity, vehicle_size_check
    from engine.trajectories import build_tracks

    det, n_frames = synthetic(n_tracks, minutes)
    cal = Calibration.from_points([[600, 500], [1300, 500], [1700, 1000], [200, 1000]], 12.0, 20.0, (1920, 1080))
    motion = {f: np.array([[1, 0, 0.01 * f], [0, 1, 0.005 * f], [0, 0, 1.0]]) for f in range(n_frames)}
    S = Settings()
    print(f"{len(det)} detections, {n_tracks} tracks, {minutes} min")
    for name, mo in (("fixed camera", None), ("moving camera", motion)):
        print(name)
        tracks, _ = timed("build_tracks", build_tracks, det, cal, S, mo)
        timed(f"detect_conflicts ({len(tracks)} tracks)", detect_conflicts, tracks, S)
    timed("sensitivity (12 re-runs)", sensitivity, tracks, S)
    timed("vehicle_size_check", vehicle_size_check, det, cal, Settings(ground_point="center"), motion)


def bench_detect(video: str, seconds: float, model: str, stabilize: bool, device: str = "cpu",
                 analysis_fps: float = 12.0, batch: int = 8, draw_video: bool = True) -> None:
    import cv2

    from detector_tracker import RenderOptions, RoadUserTracker, TrackerSettings, auto_stride
    from video_utils import AnnotatedVideoWriter, read_metadata

    m = read_metadata(video)
    clip = str(Path(__file__).resolve().parents[1] / "runs" / "bench_clip.mp4")
    Path(clip).parent.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(video)
    out = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), m["fps"], (m["width"], m["height"]))
    for _ in range(int(seconds * m["fps"])):
        ok, f = cap.read()
        if not ok:
            break
        out.write(f)
    out.release()
    cap.release()

    stride = auto_stride(m["fps"], analysis_fps)
    t_load = time.perf_counter()
    tracker = RoadUserTracker(TrackerSettings(model_name=model, stride=stride, stabilize=stabilize, device=device,
                                              batch=batch), RenderOptions(enabled=draw_video))
    t_load = time.perf_counter() - t_load
    writer =AnnotatedVideoWriter(str(Path(clip).with_name("bench_out.mp4")), m["fps"] / stride)
    t0 = time.perf_counter()
    n = dets = 0
    t_write = 0.0
    t_first = None
    seen = {}                                          # track id -> (first time, last time, class)
    for res in tracker.process_video(clip, m["fps"]):
        if t_first is None:
            t_first = time.perf_counter() - t0
        t = time.perf_counter()
        if res.annotated is not None:
            writer.write(res.annotated)
        t_write += time.perf_counter() - t
        n += 1
        dets += len(res.detections)
        for d in res.detections:
            a = seen.get(d.track_id)
            seen[d.track_id] = (a[0] if a else d.time_s, d.time_s, d.class_name)
    long = [v for v in seen.values() if v[1] - v[0] >= 1.0]
    print(f"  tracks: {len(seen)} in total, {len(long)} lasting 1 s or more "
          f"({sum(v[2] == 'car' for v in long)} cars, {sum(v[2] == 'person' for v in long)} people), "
          f"median length {np.median([v[1] - v[0] for v in long]) if long else 0:.1f} s")
    writer.release()
    total = time.perf_counter() - t0
    print(f"{n} frames ({seconds:g} s of {m['width']}x{m['height']} video at {m['fps']:g} fps, stride {stride}), "
          f"{dets} detections, model {tracker.model_used} on {tracker.device_used}"
          + (f"  [{tracker.warning}]" if tracker.warning else ""))
    steady = total - (t_first or 0)
    print(f"  model load {t_load:.1f} s, first frame {t_first or 0:.1f} s, then {1000 * steady / max(n - 1, 1):.0f} "
          f"ms/frame = {(n - 1) / max(steady, 1e-9):.1f} frames/s")
    print(f"  total {total:.1f} s; {total / seconds:.2f} s of processing per s of video "
          f"({steady / seconds:.2f} s without the first frame)")
    for k, v in tracker.timing.items():
        print(f"  {k:<10s} {1000 * v / max(n, 1):6.1f} ms/frame")
    print(f"  {'write':<10s} {1000 * t_write / max(n, 1):6.1f} ms/frame")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analysis")
    a.add_argument("--tracks", type=int, default=400)
    a.add_argument("--minutes", type=float, default=10)
    d = sub.add_parser("detect")
    d.add_argument("video")
    d.add_argument("--seconds", type=float, default=30)
    d.add_argument("--model", default="yolov8n.pt")
    d.add_argument("--stabilize", action="store_true")
    d.add_argument("--device", default="cpu", help="cpu, auto, intel:gpu, intel:npu, cuda:0, mps")
    d.add_argument("--fps", type=float, default=12.0, help="frames analysed per second of video")
    d.add_argument("--batch", type=int, default=8)
    d.add_argument("--no-video", action="store_true", help="skip drawing and writing the tracked video")
    args = ap.parse_args()
    if args.cmd == "analysis":
        bench_analysis(args.tracks, args.minutes)
    else:
        bench_detect(args.video, args.seconds, args.model, args.stabilize, args.device, args.fps, args.batch,
                     not args.no_video)
