"""
Build a fine-tuning dataset for the overhead (aerial OBB) vehicle detector from unlabelled drone clips.

There are no hand-drawn labels, so this uses SELF-TRAINING: the current aerial model plus tracking labels the
clips, and the labels are cleaned with information a single frame does not have:
  * one class per vehicle for its whole track (car or truck by length, see detector_tracker.size_classes),
    so the model learns not to flip a car between "small vehicle" and "large vehicle";
  * a vehicle the model missed for up to 1 s inside a track gets an interpolated box, so misses are not
    taught as background;
  * weak detections are kept when they belong to a long, steady track, so the model learns to find them
    with confidence;
  * ghost tracks (seen only a few times) and duplicate boxes on one vehicle are removed;
  * close-up cars that the stock model calls "ship" (it was trained on satellite photos, where cars are tiny)
    are labelled as vehicles, so the new model learns low-altitude footage too.
The labels are NOT checked by a person. Spot-check the preview images this writes before trusting a model
trained on them, and always compare against the original model on a clip that was not used for training.

Usage:  python tools/make_aerial_dataset.py OUT_DIR clip1.mp4 clip2.mp4 ...
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector_tracker import AERIAL_MODEL, AERIAL_NAMES, TRUCK_LENGTH_RATIO, size_classes  # noqa: E402

NAMES = ["small vehicle", "large vehicle"]      # kept as DOTA names so the app maps them like the stock model
SAMPLE_EVERY_S = 0.33     # one training image per this many seconds
TRACK_FPS = 8.0           # tracking runs at about this rate
MIN_HITS = 5              # a track must be seen this often to be trusted
MIN_MEAN_CONF = 0.30
SAVE_WIDTH = 1920


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _align(poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
    """poly_b with its corners rotated into the order that best matches poly_a (so they can be blended)."""
    shifts = [np.roll(poly_b, k, axis=0) for k in range(4)]
    return min(shifts, key=lambda q: float(np.abs(q - poly_a).sum()))


def label_clip(model, path: Path, out: Path, split_of) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, int(round(fps / TRACK_FPS)))
    keep_every = max(1, int(round(SAMPLE_EVERY_S * fps / stride)))
    max_gap = int(round(TRACK_FPS * 1.0))          # fill misses of up to 1 s inside a track
    # the stock model calls close-up cars (low drone) "ship": its boxes are right, its class is not, and there
    # are no ships at an intersection. Class comes from size and the track below, not from these names.
    ids = {i for i, n in model.names.items() if n in AERIAL_NAMES or n == "ship"}
    images = {}                                    # sampled n -> image
    seen = defaultdict(dict)                       # tid -> {n: (conf, poly)}
    ref = {}                                       # n -> typical car length (px) in that frame
    last_ref = None
    for n, r in enumerate(model.track(str(path), stream=True, imgsz=960, conf=0.12, classes=sorted(ids),
                                      vid_stride=stride, persist=True, verbose=False, tracker="bytetrack.yaml")):
        if n % keep_every == 0:
            images[n] = r.orig_img.copy()
        o = r.obb
        if o is None or o.id is None:
            ref[n] = last_ref
            continue
        wh = o.xywhr[:, 2:4].cpu().numpy()
        _, last_ref = size_classes(wh.max(axis=1), wh.min(axis=1), last_ref)
        ref[n] = last_ref
        for t, c, poly in zip(o.id.cpu().numpy().astype(int), o.conf.cpu().numpy(), o.xyxyxyxy.cpu().numpy()):
            seen[t][n] = (float(c), poly)
    h, w = next(iter(images.values())).shape[:2] if images else (1, 1)

    good, cls_of = set(), {}
    for t, obs in seen.items():
        if len(obs) >= MIN_HITS and np.mean([c for c, _ in obs.values()]) >= MIN_MEAN_CONF:
            good.add(t)
        votes = []
        for n, (_, poly) in obs.items():
            L = max(np.linalg.norm(poly[1] - poly[0]), np.linalg.norm(poly[2] - poly[1]))
            if ref.get(n):
                votes.append(L >= TRUCK_LENGTH_RATIO * ref[n])
        cls_of[t] = int(np.mean(votes) > 0.5) if votes else 0

    stats = {"images": 0, "boxes": 0, "trucks": 0, "filled_gaps": 0, "dropped_ghost": 0, "dropped_duplicate": 0}
    for n, img in images.items():
        cand = []                                  # (track length, tid, poly)
        for t, obs in seen.items():
            if n in obs:
                if t not in good:
                    stats["dropped_ghost"] += 1
                    continue
                cand.append((len(obs), t, obs[n][1], False))
            elif t in good:
                before = [k for k in obs if n - max_gap <= k < n]
                after = [k for k in obs if n < k <= n + max_gap]
                if before and after:               # missed in this frame but seen on both sides: interpolate
                    a, b = max(before), min(after)
                    f = (n - a) / float(b - a)
                    pa = obs[a][1]
                    cand.append((len(obs), t, (1 - f) * pa + f * _align(pa, obs[b][1]), True))
        cand.sort(key=lambda c: -c[0])             # longest tracks first win duplicates
        kept = []
        for L, t, poly, filled in cand:
            box = [*poly.min(axis=0), *poly.max(axis=0)]
            if any(iou(box, k[3]) > 0.5 for k in kept):
                stats["dropped_duplicate"] += 1
                continue
            kept.append((t, poly, filled, box))
            stats["filled_gaps"] += filled
        s = min(1.0, SAVE_WIDTH / float(max(w, h)))
        split = split_of(path.stem, n)
        name = f"{path.stem}_{n:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"),
                    cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = [f"{cls_of[t]} " + " ".join(f"{v:.6f}" for v in np.clip(poly / [w, h], 0, 1).reshape(-1))
                 for t, poly, _, _ in kept]
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["boxes"] += len(kept)
        stats["trucks"] += sum(cls_of[t] for t, *_ in kept)
        if n == sorted(images)[len(images) // 2]:   # one preview per clip to spot-check by eye
            prev = img.copy()
            for t, poly, filled, _ in kept:
                color = (0, 0, 255) if cls_of[t] else ((0, 220, 0) if filled else (255, 160, 0))
                cv2.polylines(prev, [poly.astype(np.int32)], True, color, 3)
            cv2.imwrite(str(out / "previews" / f"{name}.jpg"), cv2.resize(prev, None, fx=0.5 * s, fy=0.5 * s))
    return stats


def main(out_dir: str, clips: list) -> None:
    from ultralytics import YOLO

    out = Path(out_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "previews"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    # validation = the last 15 % of every clip (later frames, so it is not a copy of a training frame)
    lengths = {}
    for c in clips:
        cap = cv2.VideoCapture(c)
        lengths[Path(c).stem] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / TRACK_FPS)))
        cap.release()

    def split_of(stem, n):
        return "val" if n >= 0.85 * lengths[stem] else "train"

    for c in clips:
        model = YOLO(AERIAL_MODEL)                               # fresh tracker state per clip
        print(Path(c).name, label_clip(model, Path(c), out, split_of), flush=True)
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                                   f"names:\n  0: {NAMES[0]}\n  1: {NAMES[1]}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
