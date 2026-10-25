"""
One detection run that survives interruptions.

The run folder gets a `partial/` folder while detection is going on:

    partial/job.json         settings, upload name, and progress: the next frame to process and the highest track ID
    partial/detections.csv   rows of every finished chunk
    partial/motion.json      camera-motion transforms of every finished chunk (drone / moving camera only)
    partial/chunk_000.mp4    the annotated video, one file per chunk of `chunk_s` seconds of video

Every `chunk_s` seconds of video the chunk is closed and the progress written, so a crash, a closed laptop or the
Stop button loses at most one chunk. `run_job` on the same folder picks up where it stopped. When the video is done,
the chunks are joined into annotated.mp4 (ffmpeg stream copy, no re-encoding), detections.csv and motion.json are
written, and `partial/` is removed. The app writes meta.json last, as before.

A road user visible across a resume point gets a new track ID after it (ByteTrack starts over); the analysis treats
the two halves as two tracks, as it does after any long detection gap.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

CSV_HEADER = ["frame", "time_s", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]
CHUNK_S = 30.0


def partial_dir(run: Path) -> Path:
    return Path(run) / "partial"


def load_job(run: Path) -> Optional[Dict]:
    try:
        return json.loads((partial_dir(run) / "job.json").read_text())
    except Exception:
        return None


def _save_job(run: Path, job: Dict) -> None:
    p = partial_dir(run) / "job.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=1))
    tmp.replace(p)                                     # atomic: a crash mid-write never leaves a broken job.json


def new_job(run: Path, *, name: str, view: str, tracker: Dict, render: Dict, meta: Dict, chunk_s: float = CHUNK_S) -> Dict:
    """Start the bookkeeping for a fresh run (the video is already at run/source.mp4)."""
    pd = partial_dir(run)
    pd.mkdir(parents=True, exist_ok=True)
    job = {"name": name, "view": view, "tracker": tracker, "render": render, "meta": meta, "chunk_s": chunk_s,
           "next_frame": 0, "max_track_id": 0, "chunks": [], "rows": 0, "seconds": 0.0, "started": time.time(),
           "resumes": 0, "warning": "", "model_used": None, "device_used": None, "codec": None}
    with open(pd / "detections.csv", "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)
    _save_job(run, job)
    return job


def _row(x) -> list:
    return [x.frame, f"{x.time_s:.4f}", x.track_id, x.class_name, f"{x.confidence:.3f}", f"{x.x1:.1f}", f"{x.y1:.1f}",
            f"{x.x2:.1f}", f"{x.y2:.1f}", f"{(x.x1 + x.x2) / 2:.1f}", f"{(x.y1 + x.y2) / 2:.1f}"]


def run_job(run: Path, progress: Optional[Callable[[int, int, float], None]] = None,
            preview: Optional[Callable[[np.ndarray], None]] = None, tracker_factory=None) -> Dict:
    """Detect from job['next_frame'] to the end, checkpointing every chunk. Returns the finished job.

    `progress(frame, total, seconds_left_or_-1)`; `preview(annotated_frame)` about once a second;
    `tracker_factory(tracker_settings_dict, render_dict)` -> RoadUserTracker (injectable for tests)."""
    from detector_tracker import RenderOptions, RoadUserTracker, TrackerSettings
    from video_utils import AnnotatedVideoWriter

    run = Path(run)
    job = load_job(run)
    if job is None:
        raise ValueError("No unfinished detection in this folder.")
    pd = partial_dir(run)
    m = job["meta"]
    fps, stride = float(m["fps"]), int(job["tracker"]["stride"])
    if tracker_factory is None:
        def tracker_factory(ts, ro):
            return RoadUserTracker(TrackerSettings(**ts), RenderOptions(**ro))
    tracker = tracker_factory(job["tracker"], job["render"])
    job["model_used"], job["device_used"] = tracker.model_used, tracker.device_used
    if job["next_frame"] > 0:
        job["resumes"] += 1

    chunk_frames = max(stride, int(round(job["chunk_s"] * fps / stride)) * stride)
    total = int(m["frame_count"])
    motion_all = json.loads((pd / "motion.json").read_text()) if (pd / "motion.json").exists() else {}
    t_start, done_at_start = time.time(), job["next_frame"]
    last_preview = 0.0
    writer, rows, chunk_end = None, [], None

    def close_chunk(next_frame: int) -> None:
        nonlocal writer, rows
        if writer is not None:
            writer.release()
            job["codec"] = writer.codec
            if writer.frames_written:
                job["chunks"].append(writer.out_path)
        with open(pd / "detections.csv", "a", newline="") as f:
            csv.writer(f).writerows(rows)
        job["rows"] += len(rows)
        job["max_track_id"] = max([job["max_track_id"]] + [int(r[2]) for r in rows])
        if tracker.motion is not None:
            motion_all.update({str(k): np.round(v, 9).tolist() for k, v in tracker.motion.H.items()})
            (pd / "motion.json").write_text(json.dumps(motion_all))
            ms = job.setdefault("motion_stats", {"lost": [], "chained": [], "inliers": {}})
            ms["lost"] = sorted(set(ms["lost"]) | {int(f) for f in tracker.motion.lost})
            ms["chained"] = sorted(set(ms["chained"]) | {int(f) for f in tracker.motion.chained})
            ms["inliers"].update({str(k): int(v) for k, v in tracker.motion.inliers.items()})
        job["next_frame"] = next_frame
        job["seconds"] = round(job["seconds"] + (time.time() - t_start_chunk[0]), 1)
        t_start_chunk[0] = time.time()
        _save_job(run, job)
        writer, rows = None, []

    t_start_chunk = [time.time()]
    try:
        for res in tracker.process_video(str(run / "source.mp4"), fps, start_frame=job["next_frame"],
                                         id_offset=job["max_track_id"]):
            if chunk_end is not None and res.frame_idx >= chunk_end:
                close_chunk(res.frame_idx)
            if writer is None:
                k = len(job["chunks"])
                writer = AnnotatedVideoWriter(str(pd / f"chunk_{k:03d}.mp4"), fps / stride)
                chunk_end = res.frame_idx + chunk_frames
            writer.write(res.annotated)
            rows.extend(_row(x) for x in res.detections)
            if progress:
                done = res.frame_idx - done_at_start
                rate = done / max(time.time() - t_start, 1e-6)
                left = (total - res.frame_idx) / rate if done > 3 * stride and time.time() - t_start > 3 else -1
                progress(res.frame_idx, total, left)
            if preview and time.time() - last_preview > 1.0:     # sending a picture to the browser costs time
                preview(res.annotated)
                last_preview = time.time()
        close_chunk(total)
    except BaseException:
        # Stop button, page change, crash: keep everything up to the last finished chunk; drop the half-done one
        if writer is not None:
            writer.release()
            Path(writer.out_path).unlink(missing_ok=True)
        job["warning"] = tracker.warning
        _save_job(run, job)
        raise
    job["warning"] = tracker.warning
    job["stats"] = dict(tracker.stats)
    if tracker.motion is not None:
        from engine.motion import CameraMotion

        cm = CameraMotion()
        ms = job.get("motion_stats", {"lost": [], "chained": [], "inliers": {}})
        cm.H = {int(k): np.asarray(v) for k, v in motion_all.items()}
        cm.lost, cm.chained = list(ms["lost"]), list(ms["chained"])
        cm.inliers = {int(k): v for k, v in ms["inliers"].items()}
        job["motion_summary"] = cm.summary((m["width"], m["height"]))
    _finish(run, job, motion_all)
    return job


def _finish(run: Path, job: Dict, motion_all: Dict) -> None:
    pd = partial_dir(run)
    join_videos([Path(c) for c in job["chunks"]], run / "annotated.mp4", fps=job["meta"]["fps"] / job["tracker"]["stride"])
    shutil.copy(pd / "detections.csv", run / "detections.csv")
    if motion_all:
        (run / "motion.json").write_text(json.dumps({"reference_frame": 0, "summary": job.get("motion_summary", {}),
                                                     "H": motion_all}))
    (run / "job_done.json").write_text(json.dumps({k: v for k, v in job.items() if k not in ("chunks",)}, indent=1))
    for _ in range(20):          # OneDrive, antivirus or a closing ffmpeg can hold a chunk for a moment on Windows
        shutil.rmtree(pd, ignore_errors=True)
        if not pd.exists():
            break
        time.sleep(0.25)


def join_videos(parts, out: Path, fps: float) -> None:
    """Join chunk videos. ffmpeg's concat demuxer copies the streams (seconds, no quality loss); without ffmpeg the
    frames are re-written with OpenCV."""
    from video_utils import AnnotatedVideoWriter, ffmpeg_exe

    parts = [p for p in parts if Path(p).exists()]
    if not parts:
        return
    if len(parts) == 1:
        shutil.copy(parts[0], out)
        return
    exe = ffmpeg_exe()
    if exe:
        lst = Path(out).with_suffix(".txt")
        lst.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in parts))
        r = subprocess.run([exe, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                            "-movflags", "+faststart", str(out)], timeout=600)
        lst.unlink(missing_ok=True)
        if r.returncode == 0 and Path(out).exists():
            return
    import cv2

    w = AnnotatedVideoWriter(str(out), fps)
    for p in parts:
        cap = cv2.VideoCapture(str(p))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            w.write(f)
        cap.release()
    w.release()
