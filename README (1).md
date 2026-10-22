# NearMiss — Proof of Concept (Phase 1)

Turns fixed-camera intersection video into detected/tracked road users, drawn on the
video, with detections saved to a structured file.

This is **Phase 1** from the build plan (Sep 24–27):
> Set up repo; video upload; run detector; draw boxes; save detections; choose a simple web UI.
> **Exit criterion:** one video processes start-to-finish with visible detections.

Detection/tracking are not custom-built — this uses a **pretrained YOLOv8 model**
(COCO classes) and Ultralytics' built-in ByteTrack integration, clearly disclosed here
and in the app UI, per your plan's requirement.

## Project layout

```
nearmiss/
├── app.py                 # Streamlit UI — upload, run, view, download
├── src/
│   ├── detector_tracker.py  # YOLOv8 detection + ByteTrack tracking wrapper
│   └── video_utils.py       # metadata reading, annotated-video writer
├── requirements.txt
├── data/                  # put local test clips here (gitignored)
└── outputs/                # annotated videos + detections CSVs land here
```

## Setup (run locally — needs more disk/CPU than this chat sandbox has)

```bash
git init nearmiss && cd nearmiss     # or clone the repo if you already made one
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads the pretrained `yolov8n.pt` weights (~6MB) automatically from
Ultralytics' GitHub releases.

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Upload an MP4/MOV of a stationary-camera
intersection. You'll see metadata, then an annotated video with boxes + track IDs,
plus a downloadable `detections.csv`.

## What's implemented vs. still to build

| Component | Status |
|---|---|
| Video upload + metadata | ✅ done |
| Detection (person, bicycle, car, bus/truck) | ✅ done (pretrained YOLOv8n, disclosed) |
| Multi-object tracking with persistent IDs | ✅ done (ByteTrack via Ultralytics) |
| Trajectories saved to structured file | ✅ done (`detections.csv`: frame, time, track_id, class, conf, bbox, center) |
| Draw boxes + IDs on output video | ✅ done |
| Calibration (4-point homography) | ⬜ Phase 2 (Oct 3–6 in your schedule) |
| Conflict engine (TTC/PET, closest approach) | ⬜ Phase 3 (Oct 7–10) |
| Heatmap, severity, privacy blur, report | ⬜ Phase 4+ |

## Detections CSV schema

`frame, time_s, track_id, class_name, confidence, x1, y1, x2, y2, center_x, center_y`

This is the exact table the calibration and conflict-engine phases will consume —
`center_x, center_y` are the pixel points you'll run through the homography in Phase 2
to get ground-plane coordinates for the `r`, `v`, `t*`, `d*` math in your plan.

## Notes / honesty checklist (per your plan's validation section)

- Pretrained model = COCO classes; disclosed in-app.
- No accuracy numbers are shown yet — Phase 1 only proves the pipeline runs
  start-to-finish. Real precision/recall comes in the Oct 15–18 validation phase,
  against manually labeled clips, not fabricated.
- No face/plate blurring yet — flagged in-app as a known Phase-4 gap (privacy item).
