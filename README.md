# NearMiss

Screening tool for potential traffic conflicts at an intersection. It does **not** predict crashes.

Fixed-camera video -> detect and track road users -> calibrate to metres -> flag close calls
(closest approach, time to collision, post-encroachment time) -> review each flag yourself -> report.

## Run
Windows: `.\setup.ps1`. Mac or Linux: `./setup.sh`. Or by hand:
```
pip install -r requirements.txt
streamlit run app.py
```
Click **Try the demo** on Home (or *Manage recordings > Load synthetic demo* in the sidebar) to try every page
without footage. See `ARCHITECTURE.md` for how it is
built, `CONTRIBUTING.md` before changing it, and `DATA.md` for the data statement to fill in.

## Speed
Detect has three speed modes. Measured on 2 min of a 1280x720 clip, Intel Core Ultra 5 226V **on mains power**
(on a nearly empty battery Windows throttles the GPU and it was 5-10x slower):

| Mode | Frames analysed | Tracked video | Speed vs real time | Tracks of 1 s+ on the test clip |
| --- | --- | --- | --- | --- |
| Full | 12 / s | drawn while detecting | ~4.5x | 96 (80 cars, 13 people) |
| Fast (default) | 12 / s | drawn later, only if asked | ~6x | 96, identical to Full |
| Fastest | 6 / s | drawn later, only if asked | ~15x | 60 (55 cars, 5 people) |

How: *auto* uses an NVIDIA GPU, an Apple chip, or an **Intel GPU through OpenVINO** (exported once into
`openvino_cache/`). Frames go to the model 8 at a time, and the image conversion runs on the Intel GPU instead of the
CPU (it used to take twice as long as the model itself; the boxes are identical, see `tests/test_fast_openvino.py`).
Frames are decoded on one thread and drawn on others while the model works. Without the tracked video, event clips are
drawn from the original video when you open them, and Results > Slow-motion video can draw the whole video later.
- **Resume:** every 30 s of video is saved as it finishes. A stopped, crashed or closed run shows on Detect with
  *Resume*; at most 30 s is redone. A road user visible across the resume point gets a new ID after it.
- **Batch:** choose several clips at once; they run one after another, each as its own recording.
- `python tools/benchmark.py analysis` and `python tools/benchmark.py detect CLIP --device auto` time each stage.

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
3. Put the downloaded `nearmiss-visdrone.pt` next to `app.py`. It appears second in the overhead model list; the
   stock model stays the default.
4. Check it beats the old model on your own clip before relying on it:
   `python tools/compare_models.py your_clip.mp4 compare_out yolo11n-obb.pt nearmiss-visdrone.pt`
VisDrone: Zhu et al., "Detection and Tracking Meet Drones Challenge", IEEE TPAMI 2021 (hand-labelled drone images).

How the two compared on our drone clip (30 epochs, val mAP50 0.49): the VisDrone model tracked more road users
(67 vs 48) but flipped car/truck labels more (11 vs 1), and called lamp posts, signal boxes and a utility cabinet
"pedestrian" or "motorcycle". None of those ever moves, so pedestrians, bikes and motorcycles that never move more
than 3 of their own box sizes are dropped from the analysis and hidden in the drone video. Real riders on other
clips moved 7-11 box sizes; the furniture at most 1.4. Results -> Data quality shows how many were dropped.

- Aerial models find vehicles only, so the app warns that pedestrian and bicycle conflicts were **not measured**
  (zero is a blind spot, not a safe result), and privacy blur covers each whole road user.

## False-alarm filters
One road user detected twice, TTC flags shorter than 0.5 s, and a car driving past a stopped one without braking
are not flagged. The Results page shows how many flags each filter removed.

**Road area (optional).** On **Calibrate**, choose *Road area outline* and click around the roadway and crosswalks.
Road users outside it (parked cars in a lot, people on the sidewalk) are ignored.

## How far to trust the numbers
- **Calibration status** (good / not checked / problem) is shown on every page. Until a second known distance is
  checked, or when it is more than 10% off, or cars measure an implausible length, every severity is marked
  *provisional*. Results and Report list every limitation in one quality panel, each with the page that fixes it.
- **Labels** belong to exactly one event: matched by the two road users involved, then by time. One label can no
  longer count for two events a moment apart. Imported label files are checked and bad entries are listed.
- **De-duplication** merges two flagged pairs only when they share a road user (an ID switch). A TTC of 0 counts as
  the worst TTC, not a missing one.
- **Closest distance** is the minimum over the whole encounter; the number, the picture and the distance chart agree.
- **PET** follows FHWA: from the first road user's back clearing the conflict point to the second one's front
  reaching it.
- **Rates** divide by the whole observation period; *flagged* and *confirmed* (labelled real) are kept apart.
- **Street furniture** (drone footage only): pedestrians and bikes that never move are left out, and shown on
  Results > Quality so a real waiting person can be restored. Street-level footage is never filtered this way.
- **Reports** are marked out of date when labels, calibration, settings or details change after they were built.

## Extra indicators, presets and calibration fit
- **PET flagging** (Results > Thresholds, off by default): flags crossing paths where the second road user reached the
  crossing point soon after the first left it, even if they were never in the scene together. TTC cannot see these.
- **DRAC** (braking needed to avoid the collision) is shown for every event; it can also raise the severity.
- **Presets** for urban intersections, parking lots, roundabouts and arterials set the thresholds to a starting point.
  They change only thresholds; check them on your labels (Report > Tuning).
- **Several known distances** on Calibrate: keep each check, then fit the scale to all of them. With three or more,
  the app reports the leave-one-out error, the honest accuracy estimate.
- **Compare two models** on the start of a clip from Detect, with pictures side by side.
- **Precision by condition** (Report > Validation > All recordings): grouped by lighting, weather, camera view, camera
  or model, from the recording details.
- In Event Review the distance chart marks the frame being viewed; click the chart to jump there.

## Movements, signal phases and before/after
Results > Traffic & movements names each road user's approach (northbound...) and turn (through, left, right),
counts road users and events per movement, and tags events with signal phases you type (`0-28 NS green`).
Report > Compare recordings adds up a *before* and an *after* set of recordings and compares rates, warning when
thresholds, models or calibration quality differ.

## Proposed designs (not built yet)
The *Proposed designs* page runs the same conflict check on simulated trajectories of each design alternative (the
FHWA SSAM approach): export `t_s, id, class, x_m, y_m` per road user from SUMO, VISSIM or Aimsun, one CSV per
alternative. Two sample alternatives are included. Compare alternatives with each other, never with video counts.

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

## Named zones
On **Calibrate**, choose *Named zone outline*, click around a crosswalk, stop-line area or bike lane, and name it.
Every event is tagged with its zone, and Results -> *Zones* counts the road users through each zone and the events per
100 of them. Zones show on the top-down map and in the report.

## Slow-motion near-miss video
Results -> *Slow-motion video* writes a copy of the blurred result video that slows down (2-8x) around each flagged
event, with a coloured border and a banner naming the event. *Highlights only* keeps just the slowed parts.

## Reusing a calibration
A fixed camera filmed several times from the same spot: on **Calibrate**, *Reuse a calibration* copies the calibration,
road area and zones from another recording with the same frame size.

## Second reviewer
Report -> *Second reviewer* builds `review_pack.html`: one file with blurred frames and label buttons that anyone can
open in a browser. Your own labels are left out. Load the JSON they send back to see percent agreement and Cohen's
kappa, plus the events you disagree on.

## Privacy
Report -> *Privacy* lists every file stored for a recording. *Delete original video* removes the only unblurred file;
the analysis, labels, blurred video and report keep working.

## Reviewing and exporting
- Event Review keys: **T** real near miss, **F** false alarm, **U** unsure, **←/→** previous/next.
  Filter by type, severity, zone and review status; *Next unreviewed* jumps to the next event without a label; add a
  note to any label.
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
- **Optional:** `nearmiss-visdrone.pt`, a YOLO11s we fine-tuned for 30 epochs on VisDrone with
  `tools/train_visdrone_colab.ipynb` on Google Colab.
- **Tracking:** ByteTrack.

Write down which parts you wrote yourself and where AI helped; the Report page has a template.
