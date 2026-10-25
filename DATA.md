# Data statement

What footage and models NearMiss results are based on, where they came from, and on what terms. Fill in the
footage table before you share results, and keep it up to date.

## Footage

| Recording | Where and when | Who filmed it, with what | Permission / terms | Kept where, until when |
| --- | --- | --- | --- | --- |
| `sample_data/synthetic_intersection.mp4` | Not real: drawn by `tools/make_synthetic_video.py` | Generated | Part of this repository | In the repository |
| *(your clip)* | *e.g. 150th Ave SE & SE 38th St, Bellevue WA, 2017-09-10 18:08* | *e.g. city traffic camera / own drone, model* | *e.g. public data portal licence, or filmed from public land* | *e.g. this computer only, deleted after review* |

Notes to include for each real clip:
- Faces and plates: was the original video deleted after analysis (Report > Privacy)? Was anything shared unblurred?
- Was anyone filmed who could object (private property, schools)? How was that handled?

## Models

| Model | Trained on | Terms to check before sharing results or code |
| --- | --- | --- |
| YOLOv8 / YOLO11 weights (Ultralytics) | COCO (street-level), DOTA (aerial) | Ultralytics code and weights are AGPL-3.0 (or an Ultralytics commercial licence). |
| `nearmiss-visdrone.pt` (optional) | VisDrone 2019, fine-tuned by this project | VisDrone is published for academic research; check its terms. |
| ByteTrack (in Ultralytics) | No training data | Part of Ultralytics. |

## Labels

Event labels (`labels.json`) are one person's judgement of each flagged event. Say who labelled, how many events,
and whether a second reviewer was used (Report > Second reviewer).
