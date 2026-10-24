import csv

import pytest

from engine import demo
from engine.analysis import analyze, load_tracker_csv, topdown_map, events_table, distance_series
from engine.calibration import Calibration
from engine.config import Settings

pytestmark = pytest.mark.skipif(not demo.demo_available(), reason="demo clip not generated")


def _write_user_csv(path, rows):
    names = {"pedestrian": "person"}          # the Detect page writes COCO names
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], names.get(r[3], r[3]), r[4], *r[5:9], (r[5] + r[7]) / 2, (r[6] + r[8]) / 2])


def test_user_csv_round_trip_finds_the_known_events(tmp_path):
    t = demo.truth()
    cal = Calibration.from_points(t["calibration_points_px"], t["calibration_width_m"], t["calibration_length_m"], tuple(t["size"]))
    p = tmp_path / "detections.csv"
    _write_user_csv(p, demo.scripted_detections())
    det = load_tracker_csv(p)
    assert "pedestrian" in set(det.cls) and "person" not in set(det.cls)
    tracks, stats, events = analyze(det, cal, Settings())
    assert len(events) == 2
    img = topdown_map(tracks, events, cal)
    assert img.ndim == 3 and img.shape[0] > 100
    assert events_table(events)[0]["id"] == "E001"
    ds = distance_series(tracks, events[0])
    assert ds["t"] and min(ds["distance_m"]) == pytest.approx(events[0]["d_min_m"], abs=0.3)
