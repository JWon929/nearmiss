# NearMiss

Screening tool for potential traffic conflicts at an intersection. It does **not** predict crashes.

Fixed-camera video -> detect and track road users -> calibrate to metres -> flag close calls
(closest approach, time to collision, post-encroachment time) -> review each flag yourself -> report.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Click **Load synthetic demo** in the sidebar to try every page without footage.

## What changed from Phase 1
- Boxes no longer flash: weak detections extend existing tracks, lost tracks are kept 3 s, and a missed box is
  carried forward (dashed) for 1 s with a trail. Only real detections go to `detections.csv`.
- Runs on an Apple chip (mps) or GPU when available; bigger detection size finds farther road users.
- Output video is H.264, so it plays in the browser.
- New: calibration, conflict detection, top-down map and heat layer, event review with your own labels,
  privacy blur, HTML report. The Phase 1 code is commit `878aa7b` in the git history.

## Drone / overhead footage
- On **Detect**, choose *From above (drone looking straight down)*. That picks an aerial model (YOLO11-OBB, trained on
  DOTA aerial photos) and turns on camera-motion compensation. Street-level COCO models find only 1-3 of the 30+
  vehicles in a top-down frame. Aerial models find vehicles only: **no pedestrians or bicycles**.
- A hovering drone drifts and zooms. NearMiss lines every frame up with the first one (ORB features + RANSAC
  homography), so you calibrate on the first frame. Without this, parked cars look like they are moving.
- On **Calibrate**, *One known distance (2 clicks)* is enough for a straight-down view. Measure a lane line or
  crosswalk on Google Maps.

## Better drone model: train on VisDrone (free, on Google Colab)
The stock aerial model misses some moving cars and finds no pedestrians or bikes. To train a better one:
1. Open `tools/train_visdrone_colab.ipynb` in Google Colab (colab.research.google.com -> Upload).
2. Runtime -> Change runtime type -> T4 GPU, then Runtime -> Run all. About 2-3 hours.
3. Put the downloaded `nearmiss-visdrone.pt` next to `app.py`. It appears first in the overhead model list.
4. Check it beats the old model on your own clip before relying on it:
   `python tools/compare_models.py your_clip.mp4 compare_out yolo11n-obb.pt nearmiss-visdrone.pt`
VisDrone: Zhu et al., "Detection and Tracking Meet Drones Challenge", IEEE TPAMI 2021 (hand-labelled drone images).

- Aerial models find vehicles only, so the app warns that pedestrian and bicycle conflicts were **not measured**
  (zero is a blind spot, not a safe result), and privacy blur covers each whole road user.

## False-alarm filters
One road user detected twice, TTC flags shorter than 0.5 s, and a car driving past a stopped one without braking
are not flagged. The Results page shows how many flags each filter removed.

**Road area (optional).** On **Calibrate**, choose *Road area outline* and click around the roadway and crosswalks.
Road users outside it (parked cars in a lot, people on the sidewalk) are ignored.

## Comparing clips fairly
Raw event counts grow with traffic. Results also shows events per 100 vehicles, per 1,000 road-user pairs that were
on the road at the same time (overall and per conflict type), and per hour.

## Extra measurements (Results and Report pages)
- **Vehicle speeds:** median and 85th-percentile speed, and the share over an optional posted limit.
- **Sensitivity:** event counts re-computed at other TTC and distance thresholds, so the effect of each setting is visible.
- **Threshold tuning from your labels:** precision at stricter cutoffs and a suggested setting once at least
  10 events are labelled. It describes your clips only, not a general standard.
- **Car-size check (drone footage):** if cars measure far from 4-5 m long, the calibration scale is probably off.
- Every event has a plain-language sentence, e.g. *"A car (9.3 m/s, 21 mph) and a pedestrian came within 1.4 m..."*

## Second reviewer
Report -> *Second reviewer* builds `review_pack.html`: one file with blurred frames and label buttons that anyone can
open in a browser. Your own labels are left out. Load the JSON they send back to see percent agreement and Cohen's
kappa, plus the events you disagree on.

## Privacy
Report -> *Privacy* lists every file stored for a recording. *Delete original video* removes the only unblurred file;
the analysis, labels, blurred video and report keep working.

## Reviewing and exporting
- Event Review keys: **T** real near miss, **F** false alarm, **U** unsure, **←/→** previous/next.
  Filter by type, severity and review status; add a note to any label.
- Every page shows a settings line (model, calibration and its check error, thresholds, road area), and Report has a
  recording-details sheet (date, place, weather, lighting, camera) that is printed in the report.
- The Report page shows precision per conflict type and how many events are still unreviewed.
- Results -> *Export this run*: `run.json` (recording, calibration, road area, settings, stats, events, labels),
  `tracks.csv` (every trajectory sample in metres), and `labels.json`, which can be imported back.

## Tests
`python -m pytest tests` checks the maths against a scripted scene with a known answer. That proves the
calculations, **not** accuracy on real traffic. Measure that yourself on Event Review (real / false alarm / missed).
GitHub Actions runs the tests on every push (`.github/workflows/tests.yml`).

## Disclosure (Congressional App Challenge)
Detection uses pretrained models I did not train:
- **Street-level video:** YOLOv8 (Ultralytics, COCO classes: person, bicycle, motorcycle, car, bus, truck).
- **Drone video:** YOLO11-OBB (Ultralytics, trained on the DOTA aerial dataset). Vehicles only.
- **Optional:** `nearmiss-visdrone.pt`, fine-tuned on VisDrone with `tools/train_visdrone_colab.ipynb`.
- **Tracking:** ByteTrack.

Write down which parts you wrote yourself and where AI helped; the Report page has a template.
