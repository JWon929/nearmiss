"""
NearMiss: screening tool for potential traffic conflicts at an intersection (not a crash predictor).

Pages: Detect -> Calibrate -> Results -> Event Review -> Report -> Method.
Detection uses a pretrained YOLO model (COCO classes for street-level cameras, DOTA aerial classes for
drones) and ByteTrack; everything after that (camera-motion compensation, calibration, TTC / closest
approach / PET, privacy blur, validation) is in engine/.

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import csv
import json
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detector_tracker import (MODEL_CHOICES, RenderOptions, RoadUserTracker, TrackerSettings, aerial_choices,
                              auto_stride, finds_people, pick_device)
from engine import demo
from engine import validation
from engine.analysis import (analyze, build_report_html, cut_clip, distance_series, event_frame, events_table,
                             load_tracker_csv, topdown_map)
from engine.calibration import Calibration, CalibrationError
from engine.config import Settings
from engine.motion import load_motion
from video_utils import AnnotatedVideoWriter, grab_frame, read_metadata

RUNS = Path(__file__).parent / "runs"
RUNS.mkdir(exist_ok=True)
CSV_HEADER = ["frame", "time_s", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]
DISPLAY_W = 900
PAGES = ["Detect", "Calibrate", "Results", "Event Review", "Report", "Method"]
LABEL_TEXT = {"tp": "Real near miss", "fp": "False alarm", "unsure": "Unsure"}

st.set_page_config(page_title="NearMiss", layout="wide")


# ------------------------------------------------------------------ run storage
def run_dir(rid: str) -> Path:
    return RUNS / rid


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1))


def run_label(p: Path, m: dict) -> str:
    """Name shown in the recording picker. File names repeat (same clip run twice), so add view, date and status."""
    if m.get("synthetic"):
        return f"Synthetic demo  ·  {p.name}"
    view = {"overhead": "drone", "angled": "street-level"}.get(m.get("view"), "old version")
    when = time.strftime("%b %d %H:%M", time.localtime(m.get("created", 0)))
    done = "calibrated" if (p / "calibration.json").exists() else "needs calibration"
    return f"{when}  ·  {view}  ·  {done}  ·  {m.get('name', p.name)}"     # short, distinct parts first


def list_runs() -> list:
    """(run id, label) pairs, newest first by creation time (not file time, which changes on every save)."""
    found = [(p, read_json(p / "meta.json")) for p in RUNS.iterdir() if p.is_dir()]
    found = sorted([(p, m) for p, m in found if m], key=lambda pm: pm[1].get("created", 0), reverse=True)
    return [(p.name, run_label(p, m)) for p, m in found]


def current():
    rid = st.session_state.get("rid")
    if rid and (run_dir(rid) / "meta.json").exists():
        return rid, read_json(run_dir(rid) / "meta.json")
    return None, None


def source_path(rid: str) -> str:
    return str(run_dir(rid) / "source.mp4")


def load_calibration(rid: str):
    d = read_json(run_dir(rid) / "calibration.json")
    return (Calibration.from_dict(d), d) if d else (None, None)


def load_settings(rid: str) -> Settings:
    return Settings.from_dict(read_json(run_dir(rid) / "settings.json"))


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Analysing tracks...")
def cached_analysis(rid: str, csv_mtime: float, cal_json: str, settings_json: str, motion_mtime: float):
    det = load_tracker_csv(run_dir(rid) / "detections.csv")
    cal = Calibration.from_dict(json.loads(cal_json))
    S = Settings.from_dict(json.loads(settings_json))
    tracks, stats, events = analyze(det, cal, S, load_motion(read_json(run_dir(rid) / "motion.json")))
    return det, tracks, stats, events


def get_analysis(rid: str):
    cal, d = load_calibration(rid)
    if cal is None or not (run_dir(rid) / "detections.csv").exists():
        return None
    S = load_settings(rid)
    core = {k: d[k] for k in ("image_points", "width_m", "length_m", "H", "rot", "frame_size")}  # not the checks
    det, tracks, stats, events = cached_analysis(rid, mtime(run_dir(rid) / "detections.csv"), json.dumps(core),
                                                 json.dumps(S.to_dict()), mtime(run_dir(rid) / "motion.json"))
    events = [dict(e) for e in events]
    validation.apply_labels(events, read_json(run_dir(rid) / "labels.json", {}))
    return cal, S, det, tracks, stats, events


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def jpg_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ demo
def create_demo_run() -> str:
    rid = "demo-" + uuid.uuid4().hex[:6]
    d = run_dir(rid)
    d.mkdir(parents=True)
    shutil.copy(demo.DEMO_VIDEO, d / "source.mp4")
    shutil.copy(demo.DEMO_VIDEO, d / "annotated.mp4")
    t = demo.truth()
    meta = read_metadata(source_path(rid))
    meta.update(name="Synthetic demo intersection", synthetic=True, created=time.time(),
                note="Scripted road users on a drawn road. Checks the maths only, NOT real-world accuracy.")
    write_json(d / "meta.json", meta)
    names = {"pedestrian": "person"}
    with open(d / "detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in demo.scripted_detections():
            w.writerow([r[0], f"{r[1]:.4f}", r[2], names.get(r[3], r[3]), f"{r[4]:.3f}",
                        *[f"{v:.1f}" for v in r[5:9]], f"{(r[5] + r[7]) / 2:.1f}", f"{(r[6] + r[8]) / 2:.1f}"])
    cal = Calibration.from_points(t["calibration_points_px"], t["calibration_width_m"], t["calibration_length_m"],
                                  tuple(t["size"]))
    cd = cal.to_dict()
    cd["input"] = {"points": t["calibration_points_px"], "width_m": t["calibration_width_m"],
                   "length_m": t["calibration_length_m"]}
    write_json(d / "calibration.json", cd)
    return rid


# ------------------------------------------------------------------ sidebar
def go(page: str, **state) -> None:
    """Button callback: switch page (and set any other session values) before the next run draws the sidebar."""
    st.session_state["page"] = page
    st.session_state.update(state)


def next_button(page: str, text: str) -> None:
    st.button(f"Next: {text} ->", on_click=go, args=(page,), type="primary")


def steps_done(rid) -> dict:
    if not rid:
        return {}
    d = run_dir(rid)
    labels = read_json(d / "labels.json", {})
    return {"Detect": (d / "detections.csv").exists(), "Calibrate": (d / "calibration.json").exists(),
            "Results": (d / "calibration.json").exists(), "Event Review": bool(labels.get("events"))}


def pick_recording() -> None:
    st.session_state["rid"] = st.session_state["rec_pick"]


with st.sidebar:
    st.title("NearMiss")
    st.caption("Screening tool for potential traffic conflicts. Not a crash predictor.")
    runs = list_runs()
    if runs:
        ids = [r[0] for r in runs]
        if st.session_state.get("rid") not in ids:
            st.session_state["rid"] = ids[0]
        st.session_state["rec_pick"] = st.session_state["rid"]
        st.selectbox("Recording", ids, key="rec_pick", format_func=lambda i: dict(runs)[i], on_change=pick_recording)
    done = steps_done(st.session_state.get("rid"))
    page = st.radio("Steps", PAGES, key="page",
                    format_func=lambda p: ("✓ " if done.get(p) else "") + (f"{PAGES.index(p) + 1}. {p}" if p != "Method" else p))
    st.divider()
    if demo.demo_available() and st.button("Load synthetic demo"):
        st.session_state["rid"] = create_demo_run()
        st.rerun()
    if st.session_state.get("rid"):
        with st.expander("Delete this recording"):
            st.caption("Removes its video, detections, calibration and labels from this computer. Cannot be undone.")
            if st.button("Delete it permanently", type="primary"):
                shutil.rmtree(run_dir(st.session_state.pop("rid")), ignore_errors=True)
                st.rerun()
    st.caption("All processing happens on this computer. Footage is not uploaded anywhere.")

rid, meta = current()


def need_run() -> bool:
    if rid is None:
        st.info("Start on the **Detect** page: upload a clip. Or click *Load synthetic demo* in the sidebar to try the "
                "whole app without footage.")
        return True
    st.caption(f"Recording: **{dict(list_runs()).get(rid, rid)}**  (change it in the sidebar)")
    if meta.get("synthetic"):
        st.warning("Synthetic demo: scripted road users on a drawn road. It checks the maths on a known answer. "
                   "It says nothing about accuracy on real traffic.")
    else:
        frames = meta.get("frame_count", 0) / max(1, meta.get("stride", 1))
        per_frame = meta.get("detection_rows", 0) / frames if frames and "detection_rows" in meta else None
        if per_frame is not None and per_frame < 2:
            st.warning(f"Only {per_frame:.1f} road users were detected per frame in this recording, so few conflicts can "
                       "be found. If it is drone footage, run **Detect** again and choose *From above*.")
    return False


# ================================================================== DETECT
def page_detect():
    st.header("1. Detect and track road users")
    st.write("Upload an intersection clip. Works best from a camera that stays still (tripod, pole, hovering drone). "
             "If the camera drifts or zooms, NearMiss lines every frame up with the first one.")
    up = st.file_uploader("Intersection video", type=["mp4", "mov", "m4v"])
    view = st.radio("How was it filmed?", ["angled", "overhead"], horizontal=True, key="det_view",
                    format_func=lambda v: {"angled": "From the side or a pole (looking at the road at an angle)",
                                           "overhead": "From above (drone looking straight down)"}[v])
    overhead = view == "overhead"
    choices = aerial_choices() if overhead else MODEL_CHOICES
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", list(choices), format_func=lambda m: f"{m}  ({choices[m]})")
    if overhead and not finds_people(model):
        st.caption("This model was trained on aerial photos. It finds cars, buses and trucks seen from above, but "
                   "**not pedestrians or bicycles**, so only vehicle-vehicle conflicts can be flagged. To find them "
                   "too, train the VisDrone model (tools/train_visdrone_colab.ipynb) and put `nearmiss-visdrone.pt` "
                   "in the app folder.")
    imgsz = c2.select_slider("Detection size (px)", [640, 800, 960, 1280], value=960,
                             help="Bigger finds smaller or farther road users but is slower. "
                                  "For overhead models 960 works best: bigger sizes start calling cars 'large vehicles'.")
    device = c3.selectbox("Device", ["auto", "cpu", "mps", "cuda:0"], help="auto picks an Apple chip (mps) or GPU if present.")
    with st.expander("More settings"):
        a, b, c = st.columns(3)
        high = a.slider("Detection confidence", 0.10, 0.60, 0.25, 0.05,
                        help="Lower finds more, but adds false boxes. Weak detections only extend existing tracks.")
        start = b.slider("Confidence to start a new track", 0.15, 0.70, 0.30, 0.05)
        buffer_s = c.slider("Keep lost tracks (s)", 0.5, 6.0, 3.0, 0.5)
        a, b, c = st.columns(3)
        coast = a.slider("Carry box forward after a miss (s)", 0.0, 2.0, 1.0, 0.25)
        trail = b.slider("Trail length (s)", 0.0, 5.0, 2.0, 0.5)
        stride_auto = c.checkbox("Auto-skip frames (about 12 fps)", True)
        stabilize = st.checkbox("Camera moves (drone, wind, zoom): line every frame up with the first one", overhead,
                                help="Without this, a drifting camera makes parked cars look like they are driving "
                                     "and puts every position after the first seconds in the wrong place. "
                                     "Adds about 50 ms per frame.")
        a, b = st.columns(2)
        blur = a.checkbox("Blur faces and plates in the output video", True)
        level = b.selectbox("Blur strength", ["standard", "strict"], help="strict pixelates each whole road user.")
        conf_show = st.checkbox("Show confidence on labels", False)

    if up is None:
        return
    tmp = RUNS / ".incoming"
    tmp.mkdir(exist_ok=True)
    src_tmp = tmp / "upload.mp4"
    src_tmp.write_bytes(up.getbuffer())
    try:
        m = read_metadata(str(src_tmp))
    except ValueError as e:
        st.error(str(e))
        return
    cols = st.columns(4)
    cols[0].metric("Duration (s)", m["duration_s"])
    cols[1].metric("FPS", m["fps"])
    cols[2].metric("Resolution", f'{m["width"]}x{m["height"]}')
    cols[3].metric("Frames", m["frame_count"])
    stride = auto_stride(m["fps"]) if stride_auto else 1
    st.caption(f"Will analyse every {stride} frame(s). Device: {pick_device(device)}.")
    if not st.button("Run detection + tracking", type="primary"):
        return

    new_id = time.strftime("%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    d = run_dir(new_id)
    d.mkdir(parents=True)
    src = d / "source.mp4"
    shutil.move(str(src_tmp), str(src))
    ts = TrackerSettings(model_name=model, imgsz=imgsz, high_conf=high, start_conf=start, track_buffer_s=buffer_s,
                         stride=stride, device=device, coast_s=coast, trail_s=trail, stabilize=stabilize)
    ro = RenderOptions(blur_privacy=blur, privacy_level=level, show_trails=trail > 0, show_confidence=conf_show)
    prog = st.progress(0.0, text="Loading model...")
    try:
        tracker = RoadUserTracker(ts, ro)
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        st.error(f"Could not load the model: {e}")
        return
    writer = AnnotatedVideoWriter(str(d / "annotated.mp4"), m["fps"] / stride)
    preview = st.empty()
    rows = []
    t0 = time.time()
    n = 0
    try:
        for res in tracker.process_video(str(src), m["fps"],
                                         lambda i, tot: prog.progress(min(i / max(tot, 1), 1.0), text=f"Frame {i}/{tot}")):
            writer.write(res.annotated)
            for x in res.detections:
                rows.append([x.frame, f"{x.time_s:.4f}", x.track_id, x.class_name, f"{x.confidence:.3f}",
                             f"{x.x1:.1f}", f"{x.y1:.1f}", f"{x.x2:.1f}", f"{x.y2:.1f}",
                             f"{(x.x1 + x.x2) / 2:.1f}", f"{(x.y1 + x.y2) / 2:.1f}"])
            n += 1
            if n % 5 == 0:
                preview.image(rgb(res.annotated), channels="RGB", width="stretch")
    except Exception as e:
        writer.release()
        shutil.rmtree(d, ignore_errors=True)
        st.error(f"Detection stopped: {e}")
        return
    writer.release()
    if tracker.motion is not None:
        write_json(d / "motion.json", tracker.motion.to_dict((m["width"], m["height"])))
    if overhead:
        S = Settings()
        S.ground_point = "center"
        write_json(d / "settings.json", S.to_dict())
    with open(d / "detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
    m.update(name=up.name, created=time.time(), synthetic=False, stride=stride, settings=ts.to_dict(),
             codec=writer.codec, seconds=round(time.time() - t0, 1), detection_rows=len(rows),
             model_used=tracker.model_used, device_used=tracker.device_used, view=view)
    write_json(d / "meta.json", m)
    st.session_state["rid"] = new_id
    prog.progress(1.0, text="Done")
    st.success(f"Processed {n} frames in {m['seconds']} s. {len(rows)} detections saved.")
    if tracker.warning:
        st.warning(tracker.warning)
    if tracker.motion is not None:
        ms = tracker.motion.summary((m["width"], m["height"]))
        if ms["max_shift_px"] < 3:
            st.caption("The camera stayed still.")
        else:
            st.info(f"The camera moved up to {ms['max_shift_px']:.0f} px during the clip. Every frame was lined up with "
                    "the first one, so you calibrate on the first frame.")
        if ms["lost_frames"]:
            st.warning(f"{ms['lost_frames']} frames could not be lined up (too little texture or too much change). "
                       "Positions on those frames are less reliable.")
    if writer.codec != "h264":
        st.warning("ffmpeg was not found, so the video uses a format some browsers cannot play. "
                   "Run `pip install imageio-ffmpeg`, or open the file in VLC.")
    st.video(str(d / "annotated.mp4"))
    st.download_button("Download detections.csv", (d / "detections.csv").read_bytes(), "detections.csv", "text/csv")
    st.write("Next, calibrate so distances and speeds are in metres.")
    next_button("Calibrate", "Calibrate")


# ================================================================== CALIBRATE
def draw_points(img: np.ndarray, pts, scale: float, color, closed: bool) -> None:
    for i, (x, y) in enumerate(pts):
        p = (int(x * scale), int(y * scale))
        cv2.circle(img, p, 6, color, -1)
        cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if i:
            cv2.line(img, (int(pts[i - 1][0] * scale), int(pts[i - 1][1] * scale)), p, color, 2)
    if closed and len(pts) > 2:
        cv2.line(img, (int(pts[-1][0] * scale), int(pts[-1][1] * scale)),
                 (int(pts[0][0] * scale), int(pts[0][1] * scale)), color, 2)


def point_editor(pts: list, n: int, key: str) -> list:
    """Table of n points (full-resolution pixels) the user can fine-tune. Returns the edited points."""
    df = pd.DataFrame(pts + [[None, None]] * (n - len(pts)), columns=["x", "y"])
    df = st.data_editor(df, hide_index=True, width="stretch", key=f"{key}_{len(pts)}")
    return [[float(r.x), float(r.y)] for r in df.itertuples() if pd.notna(r.x) and pd.notna(r.y)][:n]


def page_calibrate():
    st.header("2. Calibrate (turn pixels into metres)")
    if need_run():
        return
    S = load_settings(rid)
    _, saved = load_calibration(rid)
    saved_in = (saved or {}).get("input", {})
    moving = (run_dir(rid) / "motion.json").exists()

    a, b = st.columns(2)
    view = a.radio("Camera view", ["angled", "overhead"], index=1 if S.ground_point == "center" else 0,
                   format_func=lambda v: {"angled": "Angled (side or pole)", "overhead": "Straight down (drone)"}[v],
                   help="Sets which point of each box touches the road: the bottom edge for an angled camera, "
                        "the centre for a camera looking straight down.")
    if (view == "overhead") != (S.ground_point == "center"):
        S.ground_point = "center" if view == "overhead" else "bottom"
        write_json(run_dir(rid) / "settings.json", S.to_dict())
    methods = ["rect", "line"] if view == "overhead" else ["rect"]
    method = b.radio("Calibrate with", methods, horizontal=True,
                     index=methods.index(saved_in.get("method", "rect")) if saved_in.get("method", "rect") in methods else 0,
                     format_func=lambda m: {"rect": "A rectangle of known size (4 clicks)",
                                            "line": "One known distance (2 clicks)"}[m])
    n_pts = 4 if method == "rect" else 2
    if method == "rect":
        st.write("Click the **four corners of a rectangle on the road** whose side lengths you know (a crosswalk, "
                 "a lane width and a stop-bar gap, a parking space). Go around the edge in order.")
    else:
        st.write("Click **two points on the road** a known distance apart (the ends of a long lane line, a crosswalk "
                 "edge). Only valid when the camera looks straight down.")
    st.caption("Don't know the size? Measure it on Google Maps: right-click a spot, choose *Measure distance*, "
               "then click the other end.")

    if moving:
        t_s = 0.0
        st.info("The camera moves in this clip, so you calibrate on the **first frame**. Every other frame has been "
                "lined up with it.")
    else:
        t_s = st.slider("Frame time (s)", 0.0, float(meta["duration_s"]), 0.0, 0.5,
                        help="Pick a frame where the road markings are clearly visible.")
    frame = grab_frame(source_path(rid), t_s)
    H, W = frame.shape[:2]
    scale = DISPLAY_W / W

    key = f"pts_{rid}_{method}"
    if key not in st.session_state:
        st.session_state[key] = [list(p) for p in saved_in.get("points", [])] if saved_in.get("method", "rect") == method else []
    pts = st.session_state[key]
    chk_key = f"chk_{rid}"
    chk = st.session_state.setdefault(chk_key, [])
    target = "cal"
    if saved is not None:
        target = st.radio("Clicks on the image add", ["cal", "chk"], horizontal=True, key=f"target_{rid}",
                          format_func=lambda t: {"cal": "Calibration points (yellow)",
                                                 "chk": "Accuracy-check points (pink)"}[t])

    c1, c2 = st.columns([3, 2])
    with c2:
        if method == "rect":
            w_m = st.number_input("Length of side 1->2 (m)", 0.5, 500.0, float(saved_in.get("width_m", 8.0)), 0.1)
            l_m = st.number_input("Length of side 2->3 (m)", 0.5, 500.0, float(saved_in.get("length_m", 20.0)), 0.1)
        else:
            w_m = st.number_input("Distance from point 1 to point 2 (m)", 0.5, 500.0,
                                  float(saved_in.get("distance_m", 10.0)), 0.1)
        if st.button("Clear calibration points"):
            st.session_state[key] = []
            st.rerun()
        st.caption("Use the biggest shape you can see. Points close together make distances less accurate.")

    shown = cv2.resize(frame, (DISPLAY_W, int(H * scale)))
    draw_points(shown, pts, scale, (0, 220, 255), closed=len(pts) == 4)
    draw_points(shown, chk, scale, (200, 80, 255), closed=False)
    with c1:
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            click = streamlit_image_coordinates(rgb(shown), key=f"img_{rid}", width="stretch")
        except ImportError:
            click = None
            st.warning("Install the click component: `pip install streamlit-image-coordinates`, "
                       "or type the points in the table.")
    if click and click.get("unix_time") != st.session_state.get(f"last_{rid}"):
        st.session_state[f"last_{rid}"] = click.get("unix_time")
        sx = W / click["width"]                     # the component reports the displayed size
        p = [round(click["x"] * sx, 1), round(click["y"] * sx, 1)]
        if target == "chk":
            st.session_state[chk_key] = (chk + [p])[-2:]
        elif len(pts) < n_pts:
            pts.append(p)
        st.rerun()

    with c2:
        if len(pts) < n_pts:
            st.info(f"Click {n_pts - len(pts)} more point(s) on the image.")
        elif st.button("Save calibration", type="primary", width="stretch"):
            try:
                if method == "rect":
                    cal = Calibration.from_points(pts, w_m, l_m, (W, H))
                    inp = {"method": "rect", "points": pts, "width_m": w_m, "length_m": l_m}
                else:
                    cal = Calibration.from_two_points(pts[0], pts[1], w_m, (W, H))
                    inp = {"method": "line", "points": pts, "distance_m": w_m}
            except CalibrationError as e:
                st.error(str(e))
            else:
                d = cal.to_dict()
                d["input"] = inp
                write_json(run_dir(rid) / "calibration.json", d)
                st.session_state[f"just_saved_{rid}"] = True
                st.rerun()
        with st.expander(f"Fine-tune calibration points ({len(pts)} of {n_pts})"):
            edited = point_editor(pts, n_pts, f"ed_{rid}_{method}")
            if edited != pts:
                st.session_state[key] = pts = edited

    cal, d = load_calibration(rid)
    if cal is None:
        return
    a = get_analysis(rid)
    if a is not None:
        n_tracks, n_events = a[4]["analysed_tracks"], len(a[5])
        msg = (f"Calibration saved for this recording. With it, {n_tracks} road users are tracked and "
               f"{n_events} events are flagged.")
        if st.session_state.pop(f"just_saved_{rid}", False):
            st.toast("Calibration saved")
        if n_tracks == 0:
            st.error(msg + " No road users could be placed on the road: check the camera view and the points.")
        else:
            st.success(msg)
        next_button("Results", "See the results")
    st.subheader("Check it")
    for h in cal.quality_hints():
        st.warning(h)
    a, b = st.columns(2)
    a.image(rgb(cal.preview(frame)), caption="Bird's-eye view of the road. Straight lines should look straight and "
            "lane edges parallel.", width="stretch")
    with b:
        st.write("**Accuracy check.** Choose *Accuracy-check points* above, click the two ends of a second known "
                 "distance (for example a lane width), and type its real length.")
        known = st.number_input("Known distance (m)", 0.5, 500.0, 3.5, 0.1)
        if len(chk) == 2:
            try:
                est = cal.distance_between(chk[0], chk[1])
            except CalibrationError as e:
                st.error(str(e))
            else:
                err = 100 * (est - known) / known
                st.metric("Measured", f"{est:.2f} m", f"{err:+.1f}% vs {known} m", delta_color="off")
                if abs(err) > 10:
                    st.warning("More than 10% off. Re-click the calibration points more carefully, or use a larger shape.")
                check = [{"known_m": known, "measured_m": round(est, 3), "error_pct": round(err, 1)}]
                if d.get("checks") != check:
                    d["checks"] = check
                    write_json(run_dir(rid) / "calibration.json", d)
        else:
            st.caption(f"{len(chk)} of 2 check points placed.")


# ================================================================== RESULTS
def thresholds_ui(S: Settings) -> None:
    with st.expander("Thresholds (change what gets flagged)"):
        st.caption("Defaults are starting points for screening, not safety standards. Validate them on your own clips.")
        a, b, c = st.columns(3)
        S.ttc_candidate_s = a.slider("Candidate if TTC under (s)", 1.0, 6.0, S.ttc_candidate_s, 0.5)
        S.ttc_moderate_s = b.slider("Moderate if TTC under (s)", 0.5, 4.0, S.ttc_moderate_s, 0.5)
        S.ttc_serious_s = c.slider("Serious if TTC under (s)", 0.2, 2.0, S.ttc_serious_s, 0.1)
        a, b = st.columns(2)
        S.min_vehicle_speed_mps = a.slider("Ignore vehicles slower than (m/s)", 0.0, 5.0, S.min_vehicle_speed_mps, 0.5)
        S.min_track_s = b.slider("Ignore tracks shorter than (s)", 0.5, 4.0, S.min_track_s, 0.5)
        ok = S.ttc_serious_s < S.ttc_moderate_s < S.ttc_candidate_s
        if not ok:
            st.error("TTC thresholds must go serious < moderate < candidate.")
        a, b = st.columns(2)
        if a.button("Save thresholds", disabled=not ok):
            write_json(run_dir(rid) / "settings.json", S.to_dict())
            st.rerun()
        if b.button("Reset to defaults"):
            D = Settings()
            D.ground_point = S.ground_point    # that one belongs to the camera, not the thresholds
            write_json(run_dir(rid) / "settings.json", D.to_dict())
            st.rerun()


def page_results():
    st.header("3. Results")
    if need_run():
        return
    a = get_analysis(rid)
    if a is None:
        st.info("This recording is not calibrated yet, so distances and speeds cannot be measured.")
        st.button("Go to Calibrate ->", type="primary", on_click=go, args=("Calibrate",))
        return
    cal, S, det, tracks, stats, events = a
    sev = pd.Series([e["severity"] for e in events]).value_counts() if events else pd.Series(dtype=int)
    c = st.columns(5)
    c[0].metric("Tracked road users", stats["analysed_tracks"])
    c[1].metric("Flagged events", len(events))
    c[2].metric("Serious", int(sev.get("serious", 0)))
    c[3].metric("Moderate", int(sev.get("moderate", 0)))
    c[4].metric("Candidate", int(sev.get("candidate", 0)))
    st.caption(f"Tracks left out: {stats['dropped_riders']} riders (person on a bike counted once), "
               f"{stats['dropped_short']} too short (mostly flicker or duplicate boxes), {stats['dropped_out_of_range']} out of range. "
               f"Flags filtered out as likely false alarms: {stats.get('filtered_duplicate_detection', 0)} one road user "
               f"detected twice, {stats.get('filtered_ttc_too_brief', 0)} too brief, "
               f"{stats.get('filtered_no_evasive_action', 0)} driving past a stopped road user without braking.")
    if meta.get("view") == "overhead" and not finds_people(meta.get("model_used", "-obb")):
        st.caption("Overhead footage with a vehicles-only model: pedestrians and bicycles are not detected, so only "
                   "vehicle-vehicle conflicts appear.")
    thresholds_ui(Settings.from_dict(S.to_dict()))
    l, r = st.columns([3, 2])
    with l:
        heat = st.checkbox("Heat layer", True)
        st.image(rgb(topdown_map(tracks, events, cal, heat=heat)),
                 caption="Top-down map. Thin lines are paths (green pedestrians, purple bikes, blue vehicles). "
                         "Dots are flagged events.", width="stretch")
    with r:
        if events:
            df = pd.DataFrame(events_table(events))
            df["label"] = df["label"].map(lambda v: LABEL_TEXT.get(v, ""))
            st.dataframe(df.drop(columns=["why"]), hide_index=True, width="stretch")
            st.download_button("Download events.csv", df.to_csv(index=False), "events.csv", "text/csv")
            ids = [e["id"] for e in events]
            pick = st.selectbox("Look at an event", range(len(ids)), format_func=lambda i: ids[i])
            st.button("Open it in Event Review ->", type="primary", on_click=go,
                      args=("Event Review",), kwargs={f"ev_{rid}": pick})
        else:
            st.success("No events flagged with these thresholds. Try loosening them, or check the calibration.")


# ================================================================== EVENT REVIEW
def step_event(key: str, delta: int, n: int) -> None:
    st.session_state[key] = min(max(st.session_state.get(key, 0) + delta, 0), n - 1)


def label_event(ev: dict, lab: str, key: str, go_to: int) -> None:
    """Button callback: save the label, then move on (go_to = next event's index)."""
    store = read_json(run_dir(rid) / "labels.json", {})
    validation.set_label(store, ev["kind"], ev["t_peak"], lab)
    write_json(run_dir(rid) / "labels.json", store)
    st.session_state[key] = go_to


def event_name(e: dict) -> str:
    lab = LABEL_TEXT.get(e.get("label"), "not reviewed")
    return f"{e['id']}  ·  {e['severity']}  ·  {e['kind_label']}  ·  t = {e['t_peak']} s  ·  {lab}"


def page_review():
    st.header("4. Review each flag yourself")
    if need_run():
        return
    a = get_analysis(rid)
    if a is None or not a[5]:
        st.info("Nothing to review yet. Detect, calibrate, and check Results.")
        return
    cal, S, det, tracks, stats, events = a
    key = f"ev_{rid}"
    if st.session_state.get(key, 0) >= len(events):
        st.session_state[key] = 0
    n1, n2, n3 = st.columns([1, 1, 6])
    n1.button("< Previous", on_click=step_event, args=(key, -1, len(events)), disabled=st.session_state.get(key, 0) == 0)
    n2.button("Next >", on_click=step_event, args=(key, 1, len(events)),
              disabled=st.session_state.get(key, 0) == len(events) - 1)
    idx = n3.selectbox("Event", range(len(events)), key=key, format_func=lambda i: event_name(events[i]),
                       label_visibility="collapsed")
    ev = events[idx]
    reviewed = sum(1 for e in events if e.get("label"))
    st.progress(reviewed / len(events), text=f"Reviewed {reviewed} of {len(events)}")

    c1, c2 = st.columns([3, 2])
    with c1:
        blur = not st.toggle("Show unblurred (this computer only)", False)
        st.image(rgb(event_frame(source_path(rid), det, ev, S.privacy_level, blur)), width="stretch",
                 caption="The moment of closest approach. The two road users are outlined.")
        t0, t1 = ev["t_start"] - S.clip_before_s, ev["t_end"] + S.clip_after_s
        clip = run_dir(rid) / f"clip_{max(t0, 0):.2f}_{t1:.2f}.mp4"     # by time: event IDs change with thresholds
        ann = run_dir(rid) / "annotated.mp4"
        if not clip.exists() and ann.exists():
            with st.spinner("Cutting the clip..."):
                cut_clip(str(ann), str(clip), t0, t1)
        if clip.exists():
            st.video(str(clip))
    with c2:
        st.metric("Severity", ev["severity"].capitalize())
        m = st.columns(3)
        m[0].metric("Closest", f"{ev['d_min_m']} m", help="Centre-to-centre distance at the closest moment.")
        m[1].metric("Min TTC", f"{ev['ttc_min_s']} s" if ev["ttc_min_s"] is not None else "n/a",
                    help="Time to collision: seconds until they would touch if both kept going straight at the same speed.")
        m[2].metric("PET", f"{ev['pet_s']} s" if ev["pet_s"] is not None else "n/a",
                    help="Post-encroachment time: gap between the first leaving and the second reaching the crossing point.")
        st.write(f"**{ev['kind_label']}** at t = {ev['t_peak']} s. Speeds: {ev['a']['cls']} {ev['a']['speed_mps']} m/s, "
                 f"{ev['b']['cls']} {ev['b']['speed_mps']} m/s.")
        if ev.get("below_3ft"):
            st.warning("Estimated clearance is under 3 ft (0.91 m), the California passing rule for bikes.")
        st.caption("Why it was flagged: " + "; ".join(ev["reasons"]))
        ds = distance_series(tracks, ev)
        if ds["t"]:
            st.line_chart(pd.DataFrame({"distance between them (m)": ds["distance_m"]}, index=ds["t"]), height=180)
        st.write("**Was this a real near miss?** Watch the clip, then choose.")
        b = st.columns(3)
        for col, lab in zip(b, ["tp", "fp", "unsure"]):
            col.button(LABEL_TEXT[lab], key=f"{lab}_{ev['id']}", type="primary" if ev.get("label") == lab else "secondary",
                       width="stretch", on_click=label_event, args=(ev, lab, key, idx + 1 if idx < len(events) - 1 else idx))
        if ev.get("label"):
            st.button("Remove my label", key=f"clear_{ev['id']}", on_click=label_event, args=(ev, "clear", key, idx))
    with st.expander("Add a near miss the app missed"):
        st.caption("While watching, note real near misses that were not flagged. They count against recall in the report.")
        store = read_json(run_dir(rid) / "labels.json", {})
        a, b = st.columns([1, 3])
        t_m = a.number_input("Time in the video (s)", 0.0, float(meta["duration_s"]), 0.0, 0.5)
        note = b.text_input("What happened?", placeholder="e.g. car turning right nearly hit a cyclist")
        if st.button("Add to missed list"):
            store.setdefault("missed", []).append({"t": t_m, "note": note})
            write_json(run_dir(rid) / "labels.json", store)
            st.rerun()
        missed = store.get("missed", [])
        if missed:
            st.dataframe(pd.DataFrame(missed).rename(columns={"t": "time (s)", "note": "what happened"}), width="stretch")
            gone = st.selectbox("Remove one", range(len(missed)),
                                format_func=lambda i: f"{missed[i]['t']} s: {missed[i]['note']}")
            if st.button("Remove"):
                missed.pop(gone)
                write_json(run_dir(rid) / "labels.json", store)
                st.rerun()


# ================================================================== REPORT
AI_TEXT = """Fill this in honestly before you submit. Example structure:
- Tools: [e.g. Claude] helped with [what: debugging, explaining the homography maths, drafting the UI layout].
- Written by me: [which files / functions you wrote or rewrote yourself].
- Pretrained models: YOLOv8 (Ultralytics, COCO classes) and ByteTrack for detection and tracking. I did not train them.
- Libraries: Streamlit, OpenCV, NumPy, pandas, Ultralytics.
- I can explain: [calibration, TTC, closest approach, PET, validation]."""


def page_report():
    st.header("5. Report")
    if need_run():
        return
    a = get_analysis(rid)
    if a is None:
        st.info("Finish Detect and Calibrate first.")
        return
    cal, S, det, tracks, stats, events = a
    store = read_json(run_dir(rid) / "labels.json", {})
    summ = validation.summarize(events, store)
    st.subheader("Validation of the flags")
    c = st.columns(6)
    c[0].metric("Flagged", summ["events_flagged"])
    c[1].metric("Real", summ["tp"])
    c[2].metric("False alarms", summ["fp"])
    c[3].metric("Not reviewed", summ["unreviewed"])
    c[4].metric("Precision", "n/a" if summ["precision"] is None else f"{summ['precision']:.0%}")
    c[5].metric("Recall (at most)", "n/a" if summ["recall_upper_bound"] is None else f"{summ['recall_upper_bound']:.0%}")
    st.caption("Precision = real / (real + false alarms). Recall can only be bounded from above, because the app cannot "
               "know about events nobody marked as missed.")
    if summ["unreviewed"]:
        st.button(f"Review the {summ['unreviewed']} remaining event(s) first", on_click=go, args=("Event Review",))
    with st.expander("All numbers"):
        st.json(summ)
    _, cd = load_calibration(rid)
    disclosure = st.text_area("AI-use disclosure (the Congressional App Challenge requires it)", AI_TEXT, height=200)
    if st.button("Build report", type="primary"):
        frames = {e["id"]: jpg_b64(event_frame(source_path(rid), det, e, S.privacy_level, True)) for e in events[:30]}
        page = build_report_html(meta.get("name", rid), {k: v for k, v in meta.items() if k != "settings"},
                                 {k: cd[k] for k in ("width_m", "length_m", "input", "checks") if k in cd}, S, events, summ,
                                 disclosure, png_b64(topdown_map(tracks, events, cal)), frames)
        st.session_state[f"report_{rid}"] = page
    if f"report_{rid}" in st.session_state:
        st.download_button("Download report.html (print to PDF)", st.session_state[f"report_{rid}"],
                           "nearmiss_report.html", "text/html")
        st.download_button("Download events.json", json.dumps(events, indent=1, default=str), "events.json")


# ================================================================== METHOD
def page_method():
    st.header("Method and limits")
    st.markdown("""
**Pipeline.** For street-level cameras a pretrained YOLOv8 model (COCO) finds people, bicycles, motorcycles, cars,
buses and trucks. For drone footage a pretrained aerial model (YOLO11-OBB, trained on the DOTA aerial dataset) finds
vehicles seen from above; it has no pedestrian or bicycle class. A model trained on VisDrone (hand-labelled drone
images, see tools/train_visdrone_colab.ipynb) also finds pedestrians and bikes from above. Each track keeps one class
label, which only changes when another class clearly outvotes it, so labels do not flicker. ByteTrack links detections into tracks. Boxes that go
missing for a moment are carried forward (dashed) so they do not flicker. Only real detections are saved.

**Moving camera.** If the camera drifts or zooms (a hovering drone does both), every frame is lined up with the first
one: corner features on the road are matched between frames, and a homography is fitted with RANSAC, which ignores
matches that disagree (moving cars, rooftops). Without this, parked cars look like they are driving.

**Calibration.** Four clicks on a rectangle of known size give a homography that maps the point where each box touches
the road to metres on the road plane: the bottom centre for an angled camera, the box centre for a camera looking
straight down. For a straight-down camera one known distance is enough. A second known distance checks the error.

**Indicators.** *Time to collision (TTC):* seconds until two road users would reach a collision radius if both kept their
current velocity. *Closest approach:* the minimum separation, and when. *PET (post-encroachment time):* the gap between one
road user leaving a crossing point and the next arriving. Velocities come from a local linear fit over a short window, not
from frame-to-frame differences, which are too noisy.

**Severity.** Candidate, moderate and serious are set by the thresholds on the Results page. They are screening labels.

**False-alarm filters.** Two tracks closer than half their combined widths are one road user detected twice. A TTC flag
must last at least 0.5 s. If one road user is stopped (a parked car), a TTC flag also needs evidence that it was real:
the other one braked hard, or they actually came within the collision radius. Otherwise the straight-line prediction was
wrong (a curving path, a different lane). The Results page says how many flags each filter removed.

**Limits.** One fixed camera, flat road assumed, no night or heavy rain testing, no head or gaze tracking. A pretrained
model misses far or small road users and can confuse classes. Every flag needs a human look, which is what Event Review is for.
This tool finds where road users came close. It does **not** predict crashes or certify an intersection as safe.

**Privacy.** Faces and plates are pixelated by position heuristics (head and lower-box regions), which is not perfect.
Footage stays on this computer.
""")


{"Detect": page_detect, "Calibrate": page_calibrate, "Results": page_results, "Event Review": page_review,
 "Report": page_report, "Method": page_method}[page]()
