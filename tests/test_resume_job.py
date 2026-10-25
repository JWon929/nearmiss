"""A detection that is stopped part-way can be resumed, and the result matches an uninterrupted run."""
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

import detect_job
from detector_tracker import Detection, FrameReader, FrameResult


class Interrupted(BaseException):
    """Like Streamlit's stop: not an Exception, so nothing swallows it."""


class FakeTracker:
    """Stands in for the YOLO tracker: one road user per frame, on frames read with the real FrameReader."""

    def __init__(self, video=True):
        self.model_used, self.device_used, self.warning = "fake.pt", "cpu", ""
        self.motion, self.stats, self.video = None, {"frames": 0}, video

    def process_video(self, path, fps, progress_cb=None, start_frame=0, id_offset=0):
        reader = FrameReader(path, 2, start_frame)
        try:
            for idx, frame in reader:
                self.stats["frames"] += 1
                d = Detection(idx, idx / fps, 1 + id_offset, "car", 0.9, 10, 10, 30, 20)
                yield FrameResult(idx, idx / fps, frame if self.video else None, [d], [])
        finally:
            reader.close()


def make_clip(path, n=300, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        f = np.zeros((48, 64, 3), np.uint8)
        f[:, i % 64] = 255
        w.write(f)
    w.release()


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def count_frames(path):
    cap, n = cv2.VideoCapture(str(path)), 0
    while cap.read()[0]:
        n += 1
    return n


@pytest.fixture
def run(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    make_clip(d / "source.mp4")                                    # 30 s at 10 fps
    meta = {"fps": 10.0, "frame_count": 300, "width": 64, "height": 48, "duration_s": 30.0}
    detect_job.new_job(d, name="clip.mp4", view="angled", tracker={"stride": 2}, render={}, meta=meta, chunk_s=5.0)
    return d


def test_uninterrupted_run_writes_everything_and_cleans_up(run):
    job = detect_job.run_job(run, tracker_factory=lambda ts, ro: FakeTracker())
    r = rows(run / "detections.csv")
    assert [int(x["frame"]) for x in r] == list(range(0, 300, 2))
    assert count_frames(run / "annotated.mp4") == 150 and job["rows"] == 150
    assert not (run / "partial").exists() and (run / "job_done.json").exists()


def test_stop_then_resume_gives_the_same_frames_with_new_ids_after_the_break(run):
    def stop_at_frame_130(i, total, left):
        if i >= 130:
            raise Interrupted()

    with pytest.raises(Interrupted):
        detect_job.run_job(run, progress=stop_at_frame_130, tracker_factory=lambda ts, ro: FakeTracker())
    job = detect_job.load_job(run)
    assert job["next_frame"] == 100 and len(job["chunks"]) == 2      # two finished 5 s chunks kept, the third dropped
    assert not (run / "detections.csv").exists()                     # not finished yet

    job = detect_job.run_job(run, tracker_factory=lambda ts, ro: FakeTracker())
    r = rows(run / "detections.csv")
    frames = [int(x["frame"]) for x in r]
    assert frames == list(range(0, 300, 2))                          # nothing lost, nothing twice
    ids = {int(x["track_id"]) for x in r if int(x["frame"]) < 100} , {int(x["track_id"]) for x in r if int(x["frame"]) >= 100}
    assert ids == ({1}, {2})                                         # IDs after the resume do not collide
    assert count_frames(run / "annotated.mp4") == 150 and job["resumes"] == 1


def test_fast_mode_without_tracked_video_still_saves_every_detection_and_resumes(run):
    def stop_at_frame_130(i, total, left):
        if i >= 130:
            raise Interrupted()

    with pytest.raises(Interrupted):
        detect_job.run_job(run, progress=stop_at_frame_130, tracker_factory=lambda ts, ro: FakeTracker(video=False))
    assert detect_job.load_job(run)["chunks"] == [] and detect_job.load_job(run)["next_frame"] == 100
    job = detect_job.run_job(run, tracker_factory=lambda ts, ro: FakeTracker(video=False))
    assert [int(x["frame"]) for x in rows(run / "detections.csv")] == list(range(0, 300, 2))
    assert not (run / "annotated.mp4").exists() and job["resumes"] == 1


def test_event_clip_can_be_drawn_later_from_the_original_video(tmp_path):
    from detector_tracker import RenderOptions, render_from_detections
    from engine import demo
    from engine.trajectories import Detections

    if not demo.demo_available():
        pytest.skip("demo clip not generated")
    det = Detections(demo.scripted_detections())
    fps = cv2.VideoCapture(str(demo.DEMO_VIDEO)).get(cv2.CAP_PROP_FPS)
    out = tmp_path / "clip.mp4"
    n = render_from_detections(str(demo.DEMO_VIDEO), det, str(out), fps, 1, RenderOptions(), 10.0, 13.0)
    assert n == pytest.approx(3 * fps, abs=2) and count_frames(out) == n


def test_frame_reader_starts_mid_video_with_the_right_stride(tmp_path):
    make_clip(tmp_path / "c.mp4", n=40)
    r = FrameReader(str(tmp_path / "c.mp4"), stride=3, start=10)
    got = [i for i, _ in r]
    r.close()
    assert got == list(range(10, 40, 3))
