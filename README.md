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

## False-alarm filters
One road user detected twice, TTC flags shorter than 0.5 s, and a car driving past a stopped one without braking
are not flagged. The Results page shows how many flags each filter removed.

## Tests
`python -m pytest tests` checks the maths against a scripted scene with a known answer. That proves the
calculations, **not** accuracy on real traffic. Measure that yourself on Event Review (real / false alarm / missed).

## Disclosure (Congressional App Challenge)
Detection uses pretrained YOLOv8 (Ultralytics, COCO) and ByteTrack. Write down which parts you wrote yourself and
where AI helped; the Report page has a template.
