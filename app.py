"""
NearMiss — Phase 1 Proof of Concept
Upload a fixed-camera intersection video -> detect + track road users -> draw
boxes -> save detections to a structured CSV.

Detection uses a PRETRAINED YOLOv8n model (COCO classes: person, bicycle, car,
bus, truck), disclosed here per the build plan's requirement.
"""

import os
import tempfile

import pandas as pd
import streamlit as st

from detector_tracker import RoadUserTracker
from video_utils import AnnotatedVideoWriter, read_metadata

st.set_page_config(page_title="NearMiss — Phase 1", layout="wide")

st.title("NearMiss")
st.caption("Screening tool for potential traffic conflicts — not a crash predictor.")

st.info(
    "**Disclosure:** road-user detection uses a pretrained YOLOv8n model "
    "(COCO classes). Tracking uses ByteTrack. This build does not yet calibrate "
    "coordinates, detect conflicts, blur faces/plates, or generate a full report — "
    "those are later phases. See README.md for the roadmap."
)

uploaded = st.file_uploader("Upload intersection video (MP4/MOV, stationary camera)", type=["mp4", "mov"])

if uploaded is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, uploaded.name)
        with open(in_path, "wb") as f:
            f.write(uploaded.read())

        meta = read_metadata(in_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration (s)", meta["duration_s"])
        c2.metric("FPS", meta["fps"])
        c3.metric("Resolution", f'{meta["width"]}x{meta["height"]}')
        c4.metric("Frames", meta["frame_count"])

        conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.9, 0.35, 0.05)
        run = st.button("Run detection + tracking", type="primary")

        if run:
            out_video_path = os.path.join(tmpdir, "annotated.mp4")
            writer = AnnotatedVideoWriter(out_video_path, meta["fps"], meta["width"], meta["height"])

            tracker = RoadUserTracker(conf_threshold=conf_threshold)
            progress_bar = st.progress(0, text="Loading model...")
            all_rows = []

            def on_progress(frame_idx, total_frames):
                if total_frames > 0:
                    progress_bar.progress(
                        min(frame_idx / total_frames, 1.0),
                        text=f"Processing frame {frame_idx}/{total_frames}",
                    )

            for frame_idx, annotated_frame, detections in tracker.process_video(
                in_path, meta["fps"], progress_cb=on_progress
            ):
                writer.write(annotated_frame)
                for det in detections:
                    cx, cy = det.center
                    all_rows.append({
                        "frame": det.frame,
                        "time_s": det.time_s,
                        "track_id": det.track_id,
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2,
                        "center_x": round(cx, 1), "center_y": round(cy, 1),
                    })

            writer.release()
            progress_bar.progress(1.0, text="Done")

            df = pd.DataFrame(all_rows)

            st.success(
                f"Processed {meta['frame_count']} frames. "
                f"{df['track_id'].nunique() if not df.empty else 0} unique tracked road users, "
                f"{len(df)} total detections."
            )

            if not df.empty:
                st.subheader("Detected road users by class")
                st.bar_chart(df.groupby("class_name")["track_id"].nunique())

            st.subheader("Annotated video")
            with open(out_video_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button("Download annotated video", video_bytes, file_name="annotated.mp4")

            st.subheader("Detections table")
            st.dataframe(df.head(200))
            st.download_button(
                "Download full detections.csv",
                df.to_csv(index=False).encode("utf-8"),
                file_name="detections.csv",
                mime="text/csv",
            )
else:
    st.write("Upload a video to begin. Try a short (10–30s) clip first to iterate quickly.")
