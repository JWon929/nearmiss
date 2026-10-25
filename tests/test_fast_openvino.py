"""The fast OpenVINO path must find the same boxes as Ultralytics' own predict() on the same exported model.

Needs model weights, OpenVINO and an Intel GPU, so it is skipped elsewhere (and on GitHub Actions)."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "sample_data" / "synthetic_intersection.mp4"


def _needs():
    try:
        import openvino as ov
        import ultralytics  # noqa: F401
    except Exception:
        return "OpenVINO or Ultralytics not installed"
    if "GPU" not in [d.split(".")[0] for d in ov.Core().available_devices]:
        return "no Intel GPU"
    if not (ROOT / "yolov8n.pt").exists():
        return "yolov8n.pt not in the app folder"
    return None


@pytest.mark.skipif(_needs() is not None, reason=str(_needs()))
def test_fast_path_matches_ultralytics_predict():
    import cv2

    from detector_tracker import FastOpenVINO, _load_openvino, class_map, input_shape, openvino_dir

    cap = cv2.VideoCapture(str(CLIP))
    frames = []
    while len(frames) < 8:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    shape = input_shape(960, (frames[0].shape[1], frames[0].shape[0]))
    model, names = _load_openvino("yolov8n.pt", shape, 8)
    classes = list(class_map(names))
    fast = FastOpenVINO(openvino_dir("yolov8n.pt", shape, 8), "intel:gpu", names, "detect", shape, 8)
    a = fast(frames, 0.12, classes)
    b = model.predict(frames, imgsz=list(shape), device="intel:gpu", conf=0.12, classes=classes, verbose=False)
    for ra, rb in zip(a, b):
        xa, xb = ra.boxes.xyxy.numpy(), rb.boxes.xyxy.cpu().numpy()
        assert len(xa) == len(xb)
        if len(xa):
            assert np.abs(np.sort(xa, axis=0) - np.sort(xb, axis=0)).max() < 2.0      # pixels
