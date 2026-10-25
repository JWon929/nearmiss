"""
NearMiss: screening tool for potential traffic conflicts at an intersection (not a crash predictor).

Pages: Detect -> Calibrate -> Results -> Event Review -> Report -> Method.
Detection uses YOLO models (YOLOv8 / COCO for street-level cameras; YOLO11-OBB / DOTA, or a YOLO11s fine-tuned on
VisDrone, for drones) and ByteTrack; everything after that (camera-motion compensation, calibration, TTC / closest
approach / PET, privacy blur, validation) is in engine/.

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import csv
import json
from html import escape as html_escape
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from detector_tracker import (MODEL_CHOICES, VISDRONE_MODEL, RenderOptions, RoadUserTracker, TrackerSettings,
                              aerial_choices, auto_stride, finds_people, pick_device)
from engine import demo
from engine import validation
from engine.analysis import (analyze, build_report_html, cut_clip, distance_series, event_frame, events_table,
                             load_tracker_csv, topdown_map)
from engine import review_pack
from engine.calibration import Calibration, CalibrationError
from engine.config import KIND_LABEL, SEVERITY_LEVELS, Settings
from engine.insights import (MPS_TO_MPH, explain_event, sensitivity, speed_summary, tuning_curve,
                             vehicle_size_check)
from engine.motion import load_motion
from engine.roi import as_polygon as roi_polygon
from engine.slowmo import make_slowmo
from engine.zones import ZONE_COLORS, ZONE_KINDS, tag_events, zone_polygons_map, zone_summary
from video_utils import AnnotatedVideoWriter, grab_frame, read_metadata

RUNS = Path(__file__).parent / "runs"
RUNS.mkdir(exist_ok=True)
CSV_HEADER = ["frame", "time_s", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]
DISPLAY_W = 900
PAGES = ["Detect", "Calibrate", "Results", "Event Review", "Report", "Method"]
PAGE_ICONS = {"Detect": ":material/videocam:", "Calibrate": ":material/straighten:", "Results": ":material/analytics:",
              "Event Review": ":material/fact_check:", "Report": ":material/description:", "Method": ":material/menu_book:"}
LABEL_TEXT = {"tp": "Real near miss", "fp": "False alarm", "unsure": "Unsure"}
SEV_HEX = {"serious": "#c62828", "moderate": "#ef6c00", "candidate": "#b28704", "none": "#6b7280"}

st.set_page_config(page_title="NearMiss", page_icon=":material/traffic:", layout="wide")
st.markdown("""<style>
/* metric cards */
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: 10px 14px;
  background: rgba(128,128,128,.04);}
div[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .8;}
/* severity badge and page intro */
.nm-badge {display: inline-block; padding: 2px 10px; border-radius: 999px; color: #fff; font-weight: 600;
  font-size: .85rem; letter-spacing: .02em;}
.nm-intro {opacity: .8; margin-top: -.6rem; margin-bottom: .6rem;}
.nm-step {font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .6;
  margin-bottom: -.4rem;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {padding: 3px 0;}
</style>""", unsafe_allow_html=True)


def page_header(page: str, title: str, intro: str = "") -> None:
    """Step number, title and a one-line explanation, the same on every page."""
    if page in PAGES[:-1]:
        st.markdown(f"<div class='nm-step'>Step {PAGES.index(page) + 1} of {len(PAGES) - 1}</div>",
                    unsafe_allow_html=True)
    st.title(title)
    if intro:
        st.markdown(f"<div class='nm-intro'>{intro}</div>", unsafe_allow_html=True)


def badge(severity: str) -> str:
    return f"<span class='nm-badge' style='background:{SEV_HEX.get(severity, '#6b7280')}'>{severity.capitalize()}</span>"


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


def has_source(rid: str) -> bool:
    """False once the user deleted the original (unblurred) video to protect privacy."""
    return (run_dir(rid) / "source.mp4").exists()


def recording_info(rid: str) -> dict:
    """The recording metadata sheet (date, place, weather...) filled in on the Report page."""
    return read_json(run_dir(rid) / "recording_info.json", {})


def load_calibration(rid: str):
    d = read_json(run_dir(rid) / "calibration.json")
    return (Calibration.from_dict(d), d) if d else (None, None)


def load_settings(rid: str) -> Settings:
    return Settings.from_dict(read_json(run_dir(rid) / "settings.json"))


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def load_roi(rid: str) -> list:
    return read_json(run_dir(rid) / "roi.json", {}).get("points", [])


def load_zones(rid: str) -> list:
    """Named zones: [{"name", "kind", "points"}], drawn on the calibration frame."""
    return read_json(run_dir(rid) / "zones.json", {}).get("zones", [])


def zone_overlays(cal) -> list:
    zones = load_zones(rid)
    return [(z["name"], p, ZONE_COLORS[i % len(ZONE_COLORS)])
            for i, (z, p) in enumerate(zip(zones, zone_polygons_map(zones, cal))) if p is not None]


@st.cache_data(show_spinner="Analysing tracks...")
def cached_analysis(rid: str, csv_mtime: float, cal_json: str, settings_json: str, motion_mtime: float,
                    roi_json: str = "[]"):
    det = load_tracker_csv(run_dir(rid) / "detections.csv")
    cal = Calibration.from_dict(json.loads(cal_json))
    S = Settings.from_dict(json.loads(settings_json))
    tracks, stats, events = analyze(det, cal, S, load_motion(read_json(run_dir(rid) / "motion.json")),
                                    roi_polygon(json.loads(roi_json)))
    return det, tracks, stats, events


def get_analysis(rid: str):
    cal, d = load_calibration(rid)
    if cal is None or not (run_dir(rid) / "detections.csv").exists():
        return None
    S = load_settings(rid)
    core = {k: d[k] for k in ("image_points", "width_m", "length_m", "H", "rot", "frame_size")}  # not the checks
    det, tracks, stats, events = cached_analysis(rid, mtime(run_dir(rid) / "detections.csv"), json.dumps(core),
                                                 json.dumps(S.to_dict()), mtime(run_dir(rid) / "motion.json"),
                                                 json.dumps(load_roi(rid)))
    events = [dict(e) for e in events]
    validation.apply_labels(events, read_json(run_dir(rid) / "labels.json", {}))
    tag_events(events, load_zones(rid), cal)
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
    st.markdown("## :material/traffic: NearMiss")
    st.caption("Screening tool for potential traffic conflicts. Not a crash predictor.")
    runs = list_runs()
    if runs:
        ids = [r[0] for r in runs]
        if st.session_state.get("rid") not in ids:
            st.session_state["rid"] = ids[0]
        st.session_state["rec_pick"] = st.session_state["rid"]
        st.selectbox("Recording", ids, key="rec_pick", format_func=lambda i: dict(runs)[i], on_change=pick_recording)
    done = steps_done(st.session_state.get("rid"))
    # The option labels must not change between runs: Streamlit treats a radio whose labels changed as a new widget
    # and resets it to the first page (labelling the first event used to jump back to Detect). So the ticks go below.
    page = st.radio("Steps", PAGES, key="page",
                    format_func=lambda p: f"{PAGE_ICONS[p]} " + (f"{PAGES.index(p) + 1}. {p}" if p != "Method" else p))
    n_done = sum(bool(done.get(p)) for p in PAGES[:4])
    if st.session_state.get("rid"):
        st.progress(n_done / 4, text=f"{n_done} of 4 steps done" + (": " + ", ".join(
            p for p in PAGES if done.get(p)) if n_done else ""))
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


def blind_spot(m: dict):
    """Warning text when the model cannot see pedestrians or bikes, so '0 ped/bike conflicts' means 'not looked for'."""
    if not m or m.get("synthetic") or finds_people(m.get("model_used") or m.get("settings", {}).get("model_name", "")):
        return None
    return ("**Vehicles only.** The aerial model used for this recording cannot detect pedestrians or bicycles, so "
            "vehicle-pedestrian and vehicle-bicycle conflicts were **not measured**. Zero of them is a blind spot, "
            "not a safe result.")


def limit_notes(a) -> list:
    """Plain warnings about what limits this recording's numbers (besides the vehicles-only blind spot), shown on
    Results and Report and printed in the report: an unchecked or poor calibration, frames that could not be lined
    up, and cars that measure the wrong size."""
    if meta.get("synthetic"):
        return []
    notes = []
    _, d = load_calibration(rid)
    checks = (d or {}).get("checks") or []
    if d is not None and not checks:
        notes.append("The calibration has not been checked against a second known distance, so how far off distances "
                     "and speeds are is unknown. Do the accuracy check on the Calibrate page.")
    elif checks and abs(checks[0]["error_pct"]) > 10:
        notes.append(f"The calibration check measured a known {checks[0]['known_m']} m as {checks[0]['measured_m']} m "
                     f"({checks[0]['error_pct']:+.1f}%). Distances and speeds are off by about that much.")
    ms = (read_json(run_dir(rid) / "motion.json") or {}).get("summary")
    if ms and ms.get("frames") and ms.get("lost_frames", 0) > 0.05 * ms["frames"]:
        notes.append(f"{ms['lost_frames']} of {ms['frames']} frames could not be lined up with the first frame "
                     "(camera moved too much, or too little texture). Positions on those frames are less reliable.")
    if a is not None:
        size = vehicle_size_check(a[2], a[0], a[1], load_motion(read_json(run_dir(rid) / "motion.json")))
        if size and size["warning"]:
            notes.append(size["warning"])
    return notes


def settings_banner() -> str:
    """One line tying every page's numbers to the settings that produced them."""
    S = load_settings(rid)
    model = meta.get("model_used") or meta.get("settings", {}).get("model_name") or ("scripted" if meta.get("synthetic") else "?")
    view = {"center": "straight down", "bottom": "angled"}[S.ground_point]
    _, cd = load_calibration(rid)
    if cd is None:
        cal = "not calibrated"
    else:
        inp = cd.get("input", {})
        cal = (f"{inp['distance_m']} m line" if inp.get("method") == "line"
               else f"{cd['width_m']:g} x {cd['length_m']:g} m rectangle")
        checks = cd.get("checks") or []
        cal += f", check error {checks[0]['error_pct']:+.1f}%" if checks else ", no accuracy check"
    return (f"Settings: model **{model}** · camera {view} · calibration {cal} · TTC candidate/moderate/serious "
            f"{S.ttc_candidate_s:g}/{S.ttc_moderate_s:g}/{S.ttc_serious_s:g} s · road area "
            f"{'on' if load_roi(rid) else 'off'} · blur {S.privacy_level}"
            + ("" if has_source(rid) else " · original video deleted"))


def need_run() -> bool:
    if rid is None:
        st.info("Start on the **Detect** page: upload a clip. Or click *Load synthetic demo* in the sidebar to try the "
                "whole app without footage.")
        return True
    st.caption(f"Recording: **{dict(list_runs()).get(rid, rid)}**  (change it in the sidebar)")
    st.caption(settings_banner())
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
def model_error(name: str, e: Exception) -> str:
    return (f"Could not load the model {name} ({type(e).__name__}: {e}). What to try: standard models are downloaded "
            "the first time they are used, so check the internet connection. A custom model (such as "
            f"{VISDRONE_MODEL}) must be in the app folder next to app.py.")


def quick_test(path: str, m: dict, ts: TrackerSettings, ro: RenderOptions, seconds: float = 10.0) -> None:
    """Run the chosen settings on the first few seconds only, so a bad model or threshold shows up in a minute,
    not after a full run. Nothing is saved."""
    prog = st.progress(0.0, text="Loading model...")
    try:
        tracker = RoadUserTracker(ts, ro)
    except Exception as e:
        prog.empty()
        st.error(model_error(ts.model_name, e))
        return
    counts, frames, snap, tracks = {}, 0, None, set()
    for res in tracker.process_video(path, m["fps"]):
        if res.time_s > seconds:
            break
        frames += 1
        for d in res.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
            tracks.add(d.track_id)
        if snap is None or res.time_s <= seconds / 2:
            snap = res.annotated
        prog.progress(min(res.time_s / seconds, 1.0), text=f"Testing: {res.time_s:.1f} of {seconds:.0f} s")
    prog.empty()
    if not frames:
        st.error("No frames could be read from the video. Re-export it as H.264 MP4 and upload it again.")
        return
    per = ", ".join(f"{n / frames:.1f} {c}" for c, n in sorted(counts.items(), key=lambda kv: -kv[1])) or "nothing"
    (st.success if counts else st.error)(f"First {seconds:.0f} s with {tracker.model_used}: {len(tracks)} tracks; "
                                         f"per frame {per}.")
    if tracker.warning:
        st.warning(tracker.warning)
    if snap is not None:
        st.image(rgb(snap), width="stretch", caption=f"Frame at about {seconds / 2:.0f} s. Are the road users boxed, "
                 "with the right labels? If not, try another model or a lower confidence before the full run.")


def page_detect():
    page_header("Detect", "Detect and track road users",
                "Upload an intersection clip. Works best from a camera that stays still (tripod, pole, hovering "
                "drone). If the camera drifts or zooms, NearMiss lines every frame up with the first one.")
    up = st.file_uploader("Intersection video", type=["mp4", "mov", "m4v"])
    view = st.radio("How was it filmed?", ["angled", "overhead"], horizontal=True, key="det_view",
                    format_func=lambda v: {"angled": "From the side or a pole (looking at the road at an angle)",
                                           "overhead": "From above (drone looking straight down)"}[v])
    overhead = view == "overhead"
    choices = aerial_choices() if overhead else MODEL_CHOICES
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", list(choices), key=f"model_{view}", format_func=lambda m: f"{m}  ({choices[m]})")
    if overhead and not finds_people(model):
        msg = ("This model finds cars, buses and trucks seen from above, but **not pedestrians or bicycles**, so only "
               "vehicle-vehicle conflicts can be flagged.")
        msg += (f" Choose **{VISDRONE_MODEL}** to include them." if VISDRONE_MODEL in choices else
                f" To find them too, train the VisDrone model (tools/train_visdrone_colab.ipynb) and put "
                f"`{VISDRONE_MODEL}` in the app folder.")
        st.warning(msg)
    elif model == VISDRONE_MODEL:
        st.info("Also finds pedestrians, bikes and motorcycles from above. It mistakes some street furniture (lamp "
                "posts, signal boxes) for people, so a pedestrian or bike that **never moves** is hidden in the video "
                "and left out of the analysis. Its car/truck labels are less steady than the default model's. Use "
                "*Try on the first 10 s* to check.")
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
        # Seen from straight above there is no "head" or "plate band" in a box, so the region heuristics miss;
        # blur each whole road user instead.
        level = b.selectbox("Blur strength", ["standard", "strict"], index=1 if overhead else 0,
                            help="strict pixelates each whole road user. Use it for drone footage: the head and "
                                 "plate regions that 'standard' blurs only make sense for a camera looking from the side.")
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
        st.error(f"{e} If it plays elsewhere, the codec may be unsupported (HEVC from some phones): re-export it as "
                 "H.264 MP4 and upload it again.")
        return
    cols = st.columns(4)
    cols[0].metric("Duration (s)", m["duration_s"])
    cols[1].metric("FPS", m["fps"])
    cols[2].metric("Resolution", f'{m["width"]}x{m["height"]}')
    cols[3].metric("Frames", m["frame_count"])
    stride = auto_stride(m["fps"]) if stride_auto else 1
    st.caption(f"Will analyse every {stride} frame(s). Device: {pick_device(device)}.")
    ts = TrackerSettings(model_name=model, imgsz=imgsz, high_conf=high, start_conf=start, track_buffer_s=buffer_s,
                         stride=stride, device=device, coast_s=coast, trail_s=trail, stabilize=stabilize)
    ro = RenderOptions(blur_privacy=blur, privacy_level=level, show_trails=trail > 0, show_confidence=conf_show)
    a, b = st.columns([1, 3])
    test = a.button("Try on the first 10 s", icon=":material/science:",
                    help="Quick check of the model and settings before the full run. Nothing is saved.")
    run = b.button("Run detection + tracking", type="primary")
    if test:
        quick_test(str(src_tmp), m, ts, ro)
    if not run:
        return

    new_id = time.strftime("%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    d = run_dir(new_id)
    d.mkdir(parents=True)
    src = d / "source.mp4"
    shutil.move(str(src_tmp), str(src))
    prog = st.progress(0.0, text="Loading model...")
    try:
        tracker = RoadUserTracker(ts, ro)
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        st.error(model_error(model, e))
        return
    writer = AnnotatedVideoWriter(str(d / "annotated.mp4"), m["fps"] / stride)
    preview = st.empty()
    rows = []
    t0 = time.time()
    n = 0

    def progress(i: int, tot: int) -> None:
        done = min(i / max(tot, 1), 1.0)
        took = time.time() - t0
        text = f"Frame {i}/{tot}"
        if done > 0.02 and took > 3:            # the first frames include model warm-up, so wait before estimating
            left = took * (1 - done) / done
            text += f" · about {left / 60:.0f} min left" if left >= 90 else f" · about {left:.0f} s left"
        prog.progress(done, text=text)

    try:
        for res in tracker.process_video(str(src), m["fps"], progress):
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
        S.privacy_level = "strict"
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
    if n == 0:
        st.error("No frames could be read from this video. Try re-exporting it as H.264 MP4 (for example with "
                 "HandBrake or your phone's share/export option) and upload it again.")
    elif not rows:
        st.error(f"The model ran on {n} frames but found **no road users**, so there is nothing to analyse. Things to "
                 "try: check *How was it filmed?* matches the footage (drone footage needs *From above*), lower "
                 "*Detection confidence* under More settings, or raise *Detection size*.")
    else:
        st.success(f"Processed {n} frames in {m['seconds']} s. {len(rows)} detections saved.")
    if tracker.warning:
        st.warning(tracker.warning)
    if blind_spot(m):
        st.warning(blind_spot(m))
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


def show_calibration_hints(method: str, pts: list, w_m: float, l_m: float, size) -> None:
    """Geometry warnings for the clicked points before they are saved."""
    try:
        cal = (Calibration.from_points(pts, w_m, l_m, size) if method == "rect"
               else Calibration.from_two_points(pts[0], pts[1], w_m, size))
    except CalibrationError as e:
        st.error(str(e))
        return
    for h in cal.quality_hints(check_shape=method == "rect"):
        st.warning(h)


def save_zone(zones: list, name: str, kind: str, draft: list) -> None:
    """Button callback (runs before the widgets are drawn, so the name box can be emptied for the next zone)."""
    zones.append({"name": name, "kind": kind, "points": list(draft)})
    write_json(run_dir(rid) / "zones.json", {"zones": zones})
    draft.clear()
    st.session_state[f"zone_name_{rid}"] = ""


def zones_editor(draft: list, open_it: bool) -> None:
    """Name the outlined zone and keep a list of saved zones (crosswalks, bike lanes...)."""
    zones = load_zones(rid)
    with st.expander(f"Named zones: {len(zones)} saved", expanded=open_it, icon=":material/pin_drop:"):
        st.caption("Optional. Choose *Named zone outline* above, click around a crosswalk, bike lane or other area, "
                   "then name it. Events are tagged with their zone and Results counts events and road users per zone.")
        a, b = st.columns(2)
        name = a.text_input("Zone name", key=f"zone_name_{rid}", placeholder="e.g. North crosswalk")
        kind = b.selectbox("Type", ZONE_KINDS, key=f"zone_kind_{rid}")
        a, b, c = st.columns(3)
        taken = name.strip() in {z["name"] for z in zones}
        a.button("Save zone", type="primary", disabled=len(draft) < 3 or not name.strip() or taken, width="stretch",
                 on_click=save_zone, args=(zones, name.strip(), kind, draft))
        if b.button("Undo point", key=f"zone_undo_{rid}", disabled=not draft, width="stretch"):
            draft.pop()
            st.rerun()
        if c.button("Clear outline", key=f"zone_clear_{rid}", disabled=not draft, width="stretch"):
            draft.clear()
            st.rerun()
        if taken:
            st.caption("A zone with that name already exists.")
        elif draft and len(draft) < 3:
            st.caption(f"{len(draft)} point(s) so far: click at least 3.")
        for i, z in enumerate(zones):
            l, r = st.columns([4, 1])
            l.markdown(f"**{z['name']}** · {z.get('kind', 'other')} · {len(z['points'])} points")
            if r.button("Delete", key=f"zone_del_{rid}_{i}"):
                zones.pop(i)
                write_json(run_dir(rid) / "zones.json", {"zones": zones})
                st.rerun()


def reuse_calibration_ui() -> None:
    """Fixed cameras: copy the calibration (and road area and zones) from another recording of the same spot."""
    size = [meta.get("width"), meta.get("height")]
    same = []
    for r, label in list_runs():
        cd = read_json(run_dir(r) / "calibration.json") if r != rid else None
        if cd and list(cd.get("frame_size", [])) == size:
            same.append((r, label))
    if not same:
        return
    with st.expander(f"Reuse a calibration from another recording ({len(same)} with the same frame size)"):
        st.caption("Only for a camera that stayed in exactly the same place and zoom. Copies the calibration, road "
                   "area and zones. Check the result with the accuracy check below.")
        src = st.selectbox("Copy from", [r for r, _ in same], format_func=dict(same).get, key=f"reuse_{rid}")
        if st.button("Copy calibration, road area and zones"):
            for f in ("calibration.json", "roi.json", "zones.json"):
                if (run_dir(src) / f).exists():
                    shutil.copy(run_dir(src) / f, run_dir(rid) / f)
            for k in [k for k in st.session_state if rid in k and k.startswith(("pts_", "roi_", "chk_"))]:
                del st.session_state[k]
            st.rerun()


def page_calibrate():
    page_header("Calibrate", "Calibrate", "Turn pixels into metres. Optionally outline the road area and name zones "
                "such as crosswalks.")
    if need_run():
        return
    if not has_source(rid):
        st.info("The original video of this recording was deleted for privacy, so the calibration cannot be redone. "
                "The saved calibration is still used for the analysis.")
        return
    reuse_calibration_ui()
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
        if view == "overhead":
            S.privacy_level = "strict"      # head / plate regions do not exist from straight above
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
    roi_key = f"roi_{rid}"
    roi_pts = st.session_state.setdefault(roi_key, [list(p) for p in load_roi(rid)])
    zone_draft = st.session_state.setdefault(f"zone_draft_{rid}", [])
    targets = ["cal"] + (["chk"] if saved is not None else []) + ["roi", "zone"]
    target = st.radio("Clicks on the image add", targets, horizontal=True, key=f"target_{rid}",
                      format_func=lambda t: {"cal": "Calibration points (yellow)",
                                             "chk": "Accuracy-check points (pink)",
                                             "roi": "Road area outline (green, optional)",
                                             "zone": "Named zone outline (white, optional)"}[t])

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
    draw_points(shown, roi_pts, scale, (80, 220, 80), closed=len(roi_pts) >= 3)
    for i, z in enumerate(load_zones(rid)):
        col = ZONE_COLORS[i % len(ZONE_COLORS)]
        q = (np.asarray(z["points"]) * scale).astype(np.int32)
        cv2.polylines(shown, [q], True, col, 2, cv2.LINE_AA)
        c = q.mean(axis=0).astype(int)
        cv2.putText(shown, z["name"], (int(c[0]) - 4 * len(z["name"]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    col, 2, cv2.LINE_AA)
    draw_points(shown, zone_draft, scale, (255, 255, 255), closed=len(zone_draft) >= 3)
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
        elif target == "roi":
            roi_pts.append(p)
        elif target == "zone":
            zone_draft.append(p)
        elif len(pts) < n_pts:
            pts.append(p)
        st.rerun()

    with c2:
        if len(pts) < n_pts:
            st.info(f"Click {n_pts - len(pts)} more point(s) on the image.")
        elif pts != saved_in.get("points"):
            show_calibration_hints(method, pts, w_m, l_m if method == "rect" else w_m, (W, H))
        if len(pts) == n_pts and st.button("Save calibration", type="primary", width="stretch"):
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
    saved_roi = load_roi(rid)
    with st.expander(f"Road area: {'saved, ' if saved_roi else ''}{len(roi_pts)} points", expanded=target == "roi",
                     icon=":material/road:"):
        st.caption("Optional. Choose *Road area outline* above and click around the roadway and crosswalks. "
                   "Road users outside it (parked cars in a lot, people on the sidewalk) are ignored, so they "
                   "cannot cause false alarms. Leave it empty to use the whole frame.")
        a, b, c = st.columns(3)
        if a.button("Save area", disabled=len(roi_pts) < 3 or roi_pts == saved_roi, type="primary", width="stretch"):
            write_json(run_dir(rid) / "roi.json", {"points": roi_pts})
            st.rerun()
        if b.button("Undo point", disabled=not roi_pts, width="stretch"):
            roi_pts.pop()
            st.rerun()
        if c.button("Clear area", disabled=not roi_pts and not saved_roi, width="stretch"):
            roi_pts.clear()
            (run_dir(rid) / "roi.json").unlink(missing_ok=True)
            st.rerun()
        if roi_pts != saved_roi:
            st.caption("Not saved yet.")
    zones_editor(zone_draft, target == "zone")

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
    for h in cal.quality_hints(check_shape=d.get("input", {}).get("method", "rect") == "rect"):
        st.warning(h)
    if a is not None:
        size = vehicle_size_check(a[2], cal, a[1], load_motion(read_json(run_dir(rid) / "motion.json")))
        if size and size["warning"]:
            st.warning(size["warning"])
        elif size:
            st.caption(f"Sanity check: cars measure {size['median_car_length_m']} m long on average with this "
                       f"calibration ({size['cars_sampled']} boxes), in line with real cars (about 4-5 m).")
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
    with st.container(border=True):
        st.subheader("Thresholds", anchor=False)
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
            D.ground_point = S.ground_point    # these belong to the camera, not the thresholds
            D.privacy_level = S.privacy_level
            write_json(run_dir(rid) / "settings.json", D.to_dict())
            st.rerun()


def page_results():
    page_header("Results", "Results", "What was flagged, where, and how it compares with the amount of traffic.")
    if need_run():
        return
    a = get_analysis(rid)
    if a is None:
        st.info("This recording is not calibrated yet, so distances and speeds cannot be measured.")
        st.button("Go to Calibrate ->", type="primary", on_click=go, args=("Calibrate",))
        return
    cal, S, det, tracks, stats, events = a
    sev = pd.Series([e["severity"] for e in events]).value_counts() if events else pd.Series(dtype=int)
    rate = lambda v: "n/a" if v is None else str(v)
    c = st.columns(3)
    c[0].metric("Flagged events", len(events))
    c[1].metric("Events per 100 vehicles", rate(stats["events_per_100_vehicles"]),
                help="Raw counts grow with traffic. Rates let you compare a busy clip with a quiet one.")
    c[2].metric("Road users tracked", stats["analysed_tracks"])
    c = st.columns(3)
    c[0].metric("Serious", int(sev.get("serious", 0)))
    c[1].metric("Moderate", int(sev.get("moderate", 0)))
    c[2].metric("Candidate", int(sev.get("candidate", 0)))
    if blind_spot(meta):
        st.warning(blind_spot(meta))
    for note in limit_notes(a):
        st.warning(note)
    tabs = st.tabs([":material/map: Map & events", ":material/insights: Traffic & rates", ":material/pin_drop: Zones",
                    ":material/speed: Speeds", ":material/tune: Thresholds", ":material/slow_motion_video: Slow-motion video",
                    ":material/fact_check: Data quality", ":material/download: Export"])
    with tabs[0]:
        l, r = st.columns([3, 2])
        with l:
            heat = st.toggle("Heat layer", True)
            st.image(rgb(topdown_map(tracks, events, cal, heat=heat, zones=zone_overlays(cal))),
                     caption="Top-down map. Thin lines are paths (green pedestrians, purple bikes, blue vehicles). "
                             "Dots are flagged events; shaded areas are named zones.", width="stretch")
        with r:
            if events:
                df = pd.DataFrame(events_table(events))
                df["label"] = df["label"].map(lambda v: LABEL_TEXT.get(v, ""))
                if not load_zones(rid):
                    df = df.drop(columns=["zone"])
                st.dataframe(df.drop(columns=["why"]).style.format(precision=2, na_rep="").map(
                    lambda v: f"color: {SEV_HEX[v]}; font-weight: 600" if v in SEV_HEX else "", subset=["severity"]),
                    hide_index=True, width="stretch")
                ids = [e["id"] for e in events]
                pick = st.selectbox("Look at an event", range(len(ids)), format_func=lambda i: event_name(events[i]))
                st.button("Open it in Event Review", type="primary", icon=":material/arrow_forward:", on_click=go,
                          args=("Event Review",), kwargs={f"ev_{rid}": pick})
            else:
                st.success("No events flagged with these thresholds. Try loosening them, or check the calibration.")
    with tabs[1]:
        c = st.columns(4)
        c[0].metric("Vehicles", stats["road_users"].get("veh", 0))
        c[1].metric("Pedestrians", stats["road_users"].get("ped", 0))
        c[2].metric("Events per 1,000 pairs", rate(stats["events_per_1000_pairs"]),
                    help="Pairs = two road users (of a kind the engine checks) on the road at the same time.")
        c[3].metric("Events per hour", rate(stats["events_per_hour"]))
        per_kind = stats["events_per_1000_pairs_by_kind"]
        st.dataframe(pd.DataFrame([{"type": KIND_LABEL[k], "pairs on the road together": stats["pairs_together"][k],
                                    "events": sum(e["kind"] == k for e in events),
                                    "events per 1,000 pairs": rate(per_kind[k])} for k in per_kind]),
                     hide_index=True, width="stretch")
        st.caption("Which tracks were left out, how well tracking and camera alignment went, and what each "
                   "false-alarm filter removed: see the *Data quality* tab.")
    with tabs[2]:
        zones_ui(cal, tracks, events)
    with tabs[3]:
        speed_ui(S, tracks)
    with tabs[4]:
        thresholds_ui(Settings.from_dict(S.to_dict()))
        sensitivity_ui(S, tracks)
    with tabs[5]:
        slowmo_ui(S, events)
    with tabs[6]:
        quality_ui(S, stats)
    with tabs[7]:
        export_ui(S, stats, tracks, events)


FILTERS = [("filtered_duplicate_detection", "One road user detected twice",
            "Two boxes on the same road user look like a collision. Pairs closer than half their combined widths "
            "are dropped."),
           ("filtered_ttc_too_brief", "Collision course too brief",
            "A predicted collision that lasted under 0.5 s: one noisy heading, not a real conflict."),
           ("filtered_no_evasive_action", "Passing a stopped road user",
            "Heading toward a parked or waiting road user, but no hard braking and no close pass: the straight-line "
            "prediction was wrong (a curve, another lane).")]


def quality_ui(S: Settings, stats: dict) -> None:
    """Tracker quality, camera-motion alignment quality, and what each false-alarm filter removed."""
    st.write("**Tracking**")
    rows = [("Track IDs from the tracker", stats.get("raw_tracks", 0), ""),
            ("Analysed road users", stats["analysed_tracks"], "long enough to measure"),
            ("Too short", stats.get("dropped_short", 0), f"under {S.min_track_s:g} s: mostly flicker or duplicate boxes"),
            ("Never moved (street furniture)", stats.get("dropped_static_clutter", 0),
             f"a pedestrian, bike or motorcycle that never moved {S.static_clutter_boxes:g} of its own box sizes: "
             "a lamp post, signal box or bin the model mistook for one"),
            ("Outside the road area", stats.get("dropped_outside_roi", 0), "only if a road area is drawn"),
            ("Out of range", stats.get("dropped_out_of_range", 0), "too far from the calibrated area"),
            ("Riders merged", stats.get("dropped_riders", 0), "a person on a bike, counted once"),
            ("Track breaks", stats.get("track_breaks", 0), "a track split by a detection gap"),
            ("Likely ID switches", stats.get("likely_id_switches", 0),
             "a track ends where a new one of the same kind starts within 1 s and 3 m (estimate)"),
            ("Median track length (s)", stats.get("median_track_s", "-"), "")]
    st.dataframe(pd.DataFrame(rows, columns=["", "count", "meaning"]), hide_index=True, width="stretch")
    if stats["analysed_tracks"] and stats.get("likely_id_switches", 0) > 0.15 * stats["analysed_tracks"]:
        st.warning("Many tracks look like one road user split into several. A conflict can then be counted twice or "
                   "missed. Try a longer *Keep lost tracks* time on Detect.")
    st.write("**Camera alignment**")
    ms = (read_json(run_dir(rid) / "motion.json") or {}).get("summary")
    if not ms:
        st.caption("The camera was treated as fixed (no alignment).")
    else:
        rows = [("Frames", ms["frames"], ""),
                ("Matched to the first frame", ms.get("direct_frames", ms["frames"] - ms["lost_frames"]), "most reliable"),
                ("Matched through the previous frame", ms.get("chained_frames", "-"),
                 "small errors can add up over many frames"),
                ("Could not be lined up", ms["lost_frames"], "the previous alignment was reused"),
                ("Largest camera movement (px)", ms["max_shift_px"], ""),
                ("Median matched features", ms.get("median_inliers", "-"), "under 40 is weak")]
        st.dataframe(pd.DataFrame(rows, columns=["", "count", "meaning"]).astype({"count": str}), hide_index=True,
                     width="stretch")
        if ms["lost_frames"] > 0.05 * max(ms["frames"], 1):
            st.warning("More than 5% of frames could not be lined up. Positions there are less reliable.")
    st.write("**False-alarm filters** (flags removed before you see them)")
    st.dataframe(pd.DataFrame([(name, stats.get(k, 0), why) for k, name, why in FILTERS],
                              columns=["filter", "flags removed", "why"]), hide_index=True, width="stretch")


def zones_ui(cal, tracks, events) -> None:
    zones = load_zones(rid)
    if not zones:
        st.info("No zones yet. On **Calibrate**, choose *Named zone outline*, click around a crosswalk or bike lane, "
                "and name it. Each event is then tagged with its zone.")
        st.button("Go to Calibrate", on_click=go, args=("Calibrate",))
        return
    rows = zone_summary(events, tracks, zones, cal)
    df = pd.DataFrame(rows).rename(columns={"events_per_100_users": "events per 100 road users"})
    st.dataframe(df.drop(columns=["valid"]), hide_index=True, width="stretch")
    if not all(r["valid"] for r in rows):
        st.warning("Some zones reach above the horizon of the calibration and cannot be measured: "
                   + ", ".join(r["zone"] for r in rows if not r["valid"]) + ". Redraw them on the road surface.")
    outside = sum(1 for e in events if not e.get("zone"))
    st.caption(f"{outside} event(s) are outside every zone. Road users through a zone are counted once each, even if "
               "they pass through it twice. A busy zone has more events simply because more people use it: compare "
               "*events per 100 road users*.")
    st.download_button("Download zones.csv", df.to_csv(index=False), f"zones_{rid}.csv", "text/csv")


def slowmo_ui(S: Settings, events) -> None:
    st.write("Make a version of the blurred result video that **slows down around every flagged event**, so a "
             "viewer can see what happened. A coloured border and banner mark the slowed part.")
    ann = run_dir(rid) / "annotated.mp4"
    if not ann.exists():
        st.info("This recording has no annotated video.")
        return
    if not events:
        st.info("No events are flagged, so there is nothing to slow down.")
        return
    a, b, c, d = st.columns(4)
    factor = a.select_slider("Slow down by", [2, 3, 4, 6, 8], value=4, format_func=lambda f: f"{f}x")
    before = b.number_input("Start before the event (s)", 0.0, 10.0, 2.0, 0.5)
    after = c.number_input("Keep slow after (s)", 0.0, 10.0, 1.0, 0.5)
    only = d.toggle("Highlights only", help="Only the slowed parts, one after another.")
    sevs = st.multiselect("Events to include", [s for s in ("serious", "moderate", "candidate")
                                                if any(e["severity"] == s for e in events)],
                          default=[s for s in ("serious", "moderate", "candidate") if any(e["severity"] == s for e in events)],
                          format_func=str.capitalize)
    chosen = [e for e in events if e["severity"] in sevs]
    out = run_dir(rid) / f"slowmo_{factor}x_{'hl' if only else 'full'}.mp4"
    if st.button(f"Make slow-motion video ({len(chosen)} events)", type="primary", disabled=not chosen,
                 icon=":material/slow_motion_video:"):
        bar = st.progress(0.0, text="Writing video...")
        info = make_slowmo(str(ann), str(out), chosen, factor, before, after, only,
                           lambda f: bar.progress(f, text=f"Writing video... {f:.0%}"))
        bar.empty()
        st.session_state[f"slowmo_{rid}"] = (str(out), info)
    made = st.session_state.get(f"slowmo_{rid}")
    if made and Path(made[0]).exists():
        path, info = made
        st.success(f"{info['events']} event(s) in {info['windows']} slowed section(s), {info['seconds']} s long.")
        st.video(path)
        st.download_button("Download video", Path(path).read_bytes(), Path(path).name, "video/mp4",
                           icon=":material/download:")
        if info["codec"] != "h264":
            st.caption("ffmpeg was not found, so some browsers cannot play this file. Open it in VLC.")


def speed_ui(S: Settings, tracks) -> None:
    with st.container():
        info = recording_info(rid)
        limit = st.number_input("Posted speed limit (mph, optional)", 0, 80, int(info.get("posted_limit_mph", 0)), 5,
                                help="Leave at 0 if you do not know it.")
        if limit != info.get("posted_limit_mph", 0):
            info["posted_limit_mph"] = limit
            write_json(run_dir(rid) / "recording_info.json", info)
        sp = speed_summary(tracks, S, limit / MPS_TO_MPH if limit else None)
        if not sp["vehicles"]:
            st.caption("No moving vehicle was tracked long enough to measure its speed.")
            return
        mph = lambda v: f"{v * MPS_TO_MPH:.0f} mph"
        c = st.columns(4)
        c[0].metric("Vehicles measured", sp["vehicles"])
        c[1].metric("Median speed", mph(sp["median_mps"]))
        c[2].metric("85th percentile", mph(sp["p85_mps"]),
                    help="The speed 85% of drivers stay under: the usual summary of a spot-speed study.")
        if limit:
            c[3].metric("Over the limit", f"{sp['share_over_limit']:.0%}",
                        help=f"{sp['share_over_limit_by_5mph']:.0%} were more than 5 mph over.")
        else:
            c[3].metric("Fastest", mph(sp["max_mps"]))
        counts, edges = np.histogram(np.array(sp["speeds_mps"]) * MPS_TO_MPH, bins=np.arange(0, 85, 5))
        keep = slice(0, int(np.flatnonzero(counts).max()) + 2)          # stop just past the fastest bin
        st.bar_chart(pd.DataFrame({"vehicles": counts[keep]}, index=edges[:-1][keep].astype(int)),
                     height=180, x_label="speed (mph, 5 mph bins: 25 means 25-30)", y_label="vehicles")
        st.caption("Each vehicle's speed is the 85th percentile of its own moving samples, so a brief tracking spike "
                   "does not count. Speeds are only as accurate as the calibration (a 5% scale error is a 5% speed error).")


@st.cache_data(show_spinner="Re-running the conflict check at other thresholds...")
def cached_sensitivity(rid: str, csv_mtime: float, cal_json: str, settings_json: str, motion_mtime: float,
                       roi_json: str):
    _, tracks, _, _ = cached_analysis(rid, csv_mtime, cal_json, settings_json, motion_mtime, roi_json)
    return sensitivity(tracks, Settings.from_dict(json.loads(settings_json)))


def sensitivity_ui(S: Settings, tracks) -> None:
    with st.container(border=True):
        st.subheader("Sensitivity: how the count depends on the thresholds", anchor=False)
        st.caption("The thresholds are tunable starting points, not magic numbers. This re-runs the check with one "
                   "threshold changed at a time, so you can see how much the event count depends on it.")
        if not st.toggle("Run it", key=f"sens_{rid}"):
            return
        _, d = load_calibration(rid)
        core = {k: d[k] for k in ("image_points", "width_m", "length_m", "H", "rot", "frame_size")}
        res = cached_sensitivity(rid, mtime(run_dir(rid) / "detections.csv"), json.dumps(core),
                                 json.dumps(S.to_dict()), mtime(run_dir(rid) / "motion.json"), json.dumps(load_roi(rid)))
        a, b = st.columns(2)
        with a:
            st.write("**Candidate TTC threshold**")
            df = pd.DataFrame(res["ttc"]).rename(columns={"ttc_candidate_s": "TTC under (s)"})
            df["current"] = df["current"].map({True: "◀ now", False: ""})
            st.dataframe(df, hide_index=True, width="stretch")
        with b:
            st.write("**Distance thresholds, scaled**")
            df = pd.DataFrame(res["proximity"]).rename(columns={"proximity_scale": "x current distances"})
            df["current"] = df["current"].map({True: "◀ now", False: ""})
            st.dataframe(df, hide_index=True, width="stretch")


def tracks_csv(tracks) -> str:
    """Every smoothed trajectory sample in map metres, so a reviewer can recompute any number."""
    out = ["track_id,segment,class,group,time_s,x_m,y_m,vx_mps,vy_mps,speed_mps"]
    for t in tracks:
        for k in range(len(t.t)):
            out.append(f"{t.id},{t.seg},{t.cls},{t.group},{t.t[k]:.2f},{t.xy[k, 0]:.3f},{t.xy[k, 1]:.3f},"
                       f"{t.v[k, 0]:.3f},{t.v[k, 1]:.3f},{t.speed[k]:.3f}")
    return "\n".join(out) + "\n"


def export_ui(S: Settings, stats: dict, tracks, events) -> None:
    with st.container():
        st.caption("Everything needed to check the numbers or reproduce the run. The bundle holds no video frames.")
        _, cd = load_calibration(rid)
        bundle = {"recording": {k: v for k, v in meta.items()}, "calibration": cd, "road_area": load_roi(rid),
                  "zones": load_zones(rid), "settings": S.to_dict(), "stats": stats, "events": events,
                  "labels": read_json(run_dir(rid) / "labels.json", {}), "recording_info": recording_info(rid),
                  "speeds": speed_summary(tracks, S), "other_reviewers": [
                      read_json(p, {}) for p in sorted(run_dir(rid).glob("review_*.json"))]}
        a, b, c, d = st.columns(4)
        a.download_button("run.json (everything)", json.dumps(bundle, indent=1, default=str),
                          f"nearmiss_{rid}.json", "application/json", icon=":material/download:")
        b.download_button("events.csv", pd.DataFrame(events_table(events)).to_csv(index=False), f"events_{rid}.csv",
                          "text/csv", icon=":material/download:")
        c.download_button("tracks.csv", tracks_csv(tracks), f"tracks_{rid}.csv", "text/csv", icon=":material/download:")
        d.download_button("labels.json", json.dumps(bundle["labels"], indent=1), f"labels_{rid}.json",
                          "application/json", icon=":material/download:")
        up = st.file_uploader("Import labels.json (restores Event Review labels, e.g. from another computer)",
                              type=["json"], key=f"lab_up_{rid}")
        if up is not None and st.button("Replace this recording's labels with the file"):
            try:
                store = json.loads(up.getvalue())
                assert isinstance(store, dict) and isinstance(store.get("events", []), list)
            except Exception:
                st.error("That file is not a NearMiss labels.json.")
            else:
                write_json(run_dir(rid) / "labels.json", store)
                st.success(f"Imported {len(store.get('events', []))} labels and {len(store.get('missed', []))} "
                           "missed events.")


# ================================================================== EVENT REVIEW
# T / F / U label the event, left / right arrows move between events. Streamlit has no key bindings, so a tiny
# script in a zero-height frame listens on the page and clicks the matching button. It is added once per tab,
# skips keys typed into fields or the video player, and does nothing on pages without those buttons.
REVIEW_KEYS_JS = """<script>
const doc = window.parent.document;
if (!doc.nearmissKeys) {
  doc.nearmissKeys = true;
  const keys = {t: "%s", f: "%s", u: "%s", ArrowLeft: "< Previous", ArrowRight: "Next >"};
  doc.addEventListener("keydown", (e) => {
    const t = e.target, tag = (t.tagName || "").toLowerCase();
    if (["input", "textarea", "select", "video"].includes(tag) || t.isContentEditable) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const want = keys[e.key.length === 1 ? e.key.toLowerCase() : e.key];
    if (!want) return;
    const btn = [...doc.querySelectorAll("button")].find(b => b.innerText.trim() === want && !b.disabled);
    if (btn) { e.preventDefault(); btn.click(); }
  });
}
</script>""" % (LABEL_TEXT["tp"], LABEL_TEXT["fp"], LABEL_TEXT["unsure"])


def step_event(key: str, delta: int, n: int) -> None:
    st.session_state[key] = min(max(st.session_state.get(key, 0) + delta, 0), n - 1)


def label_event(ev: dict, lab: str, key: str, go_to: int) -> None:
    """Button callback: save the label (with the note typed for this event), then move on (go_to = next index)."""
    store = read_json(run_dir(rid) / "labels.json", {})
    validation.set_label(store, ev["kind"], ev["t_peak"], lab, st.session_state.get(f"note_{rid}_{ev['id']}", ""))
    write_json(run_dir(rid) / "labels.json", store)
    st.session_state[key] = go_to


def review_filters(events: list) -> list:
    """Narrow the review list by type, severity and review status."""
    with st.expander("Filter events", icon=":material/filter_list:"):
        a, b, c, d = st.columns(4)
        kinds = a.multiselect("Type", sorted({e["kind"] for e in events}), format_func=KIND_LABEL.get,
                              key=f"f_kind_{rid}")
        sevs = b.multiselect("Severity", [s for s in SEVERITY_LEVELS if any(e["severity"] == s for e in events)],
                             key=f"f_sev_{rid}")
        status = c.selectbox("Status", ["all", "not reviewed", "real", "false alarm", "unsure"], key=f"f_stat_{rid}")
        zone_names = sorted({e.get("zone", "") for e in events})
        zones = d.multiselect("Zone", zone_names, format_func=lambda z: z or "(outside every zone)",
                              key=f"f_zone_{rid}", disabled=zone_names == [""])
        want = {"not reviewed": None, "real": "tp", "false alarm": "fp", "unsure": "unsure"}
        return [e for e in events if (not kinds or e["kind"] in kinds) and (not sevs or e["severity"] in sevs)
                and (status == "all" or e.get("label") == want[status]) and (not zones or e.get("zone", "") in zones)]


def next_unreviewed(key: str, events: list, idx: int) -> None:
    """Button callback: jump to the next event without a label, wrapping around."""
    order = list(range(idx + 1, len(events))) + list(range(0, idx + 1))
    st.session_state[key] = next((i for i in order if not events[i].get("label")), idx)


def event_name(e: dict) -> str:
    lab = LABEL_TEXT.get(e.get("label"), "not reviewed")
    return f"{e['id']}  ·  {e['severity']}  ·  {e['kind_label']}  ·  t = {e['t_peak']} s  ·  {lab}"


def page_review():
    page_header("Event Review", "Event review", "Watch each flagged moment and decide whether it was a real near miss.")
    if need_run():
        return
    a = get_analysis(rid)
    if a is None or not a[5]:
        st.info("Nothing to review yet. Detect, calibrate, and check Results.")
        return
    cal, S, det, tracks, stats, events = a
    all_events = events
    events = review_filters(events)
    if not events:
        st.info("No events match the filter.")
        return
    key = f"ev_{rid}"
    if st.session_state.get(key, 0) >= len(events):
        st.session_state[key] = 0
    cur = st.session_state.get(key, 0)
    n1, n2, n3, n4 = st.columns([1, 1, 1.4, 5])
    n1.button("< Previous", on_click=step_event, args=(key, -1, len(events)), disabled=cur == 0, width="stretch")
    n2.button("Next >", on_click=step_event, args=(key, 1, len(events)), disabled=cur == len(events) - 1,
              width="stretch")
    n3.button("Next unreviewed", on_click=next_unreviewed, args=(key, events, cur), icon=":material/skip_next:",
              disabled=all(e.get("label") for e in events), width="stretch")
    idx = n4.selectbox("Event", range(len(events)), key=key, format_func=lambda i: event_name(events[i]),
                       label_visibility="collapsed")
    ev = events[idx]
    reviewed = sum(1 for e in all_events if e.get("label"))
    st.progress(reviewed / len(all_events), text=f"Reviewed {reviewed} of {len(all_events)}"
                + ("" if len(events) == len(all_events) else f" · showing {len(events)} that match the filter"))
    if hasattr(st, "iframe"):                      # Streamlit >= 1.5x; components.html is deprecated there
        st.iframe(REVIEW_KEYS_JS, height=1)
    else:
        components.html(REVIEW_KEYS_JS, height=0)
    st.caption("Keyboard: **T** real near miss, **F** false alarm, **U** unsure, **←/→** previous/next event. "
               "Click an empty part of the page first if a key does nothing.")

    c1, c2 = st.columns([3, 2])
    with c1:
        if has_source(rid):
            blur = not st.toggle("Show unblurred (this computer only)", False)
            st.image(rgb(event_frame(source_path(rid), det, ev, S.privacy_level, blur)), width="stretch",
                     caption="The moment of closest approach. The two road users are outlined.")
        else:
            st.caption("The original video was deleted for privacy, so only the blurred clip is shown.")
        t0, t1 = ev["t_start"] - S.clip_before_s, ev["t_end"] + S.clip_after_s
        clip = run_dir(rid) / f"clip_{max(t0, 0):.2f}_{t1:.2f}.mp4"     # by time: event IDs change with thresholds
        ann = run_dir(rid) / "annotated.mp4"
        if not clip.exists() and ann.exists():
            with st.spinner("Cutting the clip..."):
                cut_clip(str(ann), str(clip), t0, t1)
        if clip.exists():
            st.video(str(clip))
    with c2:
        st.markdown(f"{badge(ev['severity'])} &nbsp; **{ev['id']} · {ev['kind_label']}** · t = {ev['t_peak']} s"
                    + (f" · zone: {html_escape(ev['zone'])}" if ev.get("zone") else ""), unsafe_allow_html=True)
        m = st.columns(3)
        m[0].metric("Closest", f"{ev['d_min_m']} m", help="Centre-to-centre distance at the closest moment.")
        m[1].metric("Min TTC", f"{ev['ttc_min_s']} s" if ev["ttc_min_s"] is not None else "n/a",
                    help="Time to collision: seconds until they would touch if both kept going straight at the same speed.")
        m[2].metric("PET", f"{ev['pet_s']} s" if ev["pet_s"] is not None else "n/a",
                    help="Post-encroachment time: gap between the first leaving and the second reaching the crossing point.")
        st.write(explain_event(ev))
        if ev.get("below_3ft"):
            st.warning("Estimated clearance is under 3 ft (0.91 m), the California passing rule for bikes.")
        st.caption("Why it was flagged: " + "; ".join(ev["reasons"]))
        ds = distance_series(tracks, ev)
        if ds["t"]:
            st.line_chart(pd.DataFrame({"distance between them (m)": ds["distance_m"]}, index=ds["t"]), height=180)
        st.write("**Was this a real near miss?** Watch the clip, then choose.")
        note_key = f"note_{rid}_{ev['id']}"
        if note_key not in st.session_state:
            st.session_state[note_key] = ev.get("label_note", "")
        st.text_input("Note (optional, saved with your label)", key=note_key,
                      placeholder="e.g. car stopped late at the crosswalk")
        b = st.columns(3)
        # with the "not reviewed" filter a labelled event leaves the list, so the next one moves up to this index
        nxt = idx if st.session_state.get(f"f_stat_{rid}") == "not reviewed" else min(idx + 1, len(events) - 1)
        for col, lab in zip(b, ["tp", "fp", "unsure"]):
            col.button(LABEL_TEXT[lab], key=f"{lab}_{ev['id']}", type="primary" if ev.get("label") == lab else "secondary",
                       width="stretch", on_click=label_event, args=(ev, lab, key, nxt))
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
- Pretrained models: YOLOv8 (Ultralytics, COCO classes) for street-level video, YOLO11-OBB (Ultralytics, trained on the
  DOTA aerial dataset; vehicles only) for drone video, and ByteTrack for tracking. I did not train them.
  [If you used nearmiss-visdrone.pt: I fine-tuned YOLO11s for 30 epochs on the VisDrone dataset (Zhu et al., IEEE
  TPAMI 2021) on Google Colab with tools/train_visdrone_colab.ipynb.]
- Libraries: Streamlit, OpenCV, NumPy, pandas, Ultralytics.
- I can explain: [calibration, TTC, closest approach, PET, validation]."""


def page_report():
    page_header("Report", "Report", "How good the flags were, a second opinion, and a shareable report.")
    if need_run():
        return
    a = get_analysis(rid)
    if a is None:
        st.info("Finish Detect and Calibrate first.")
        return
    cal, S, det, tracks, stats, events = a
    store = read_json(run_dir(rid) / "labels.json", {})
    summ = validation.summarize(events, store)
    c = st.columns(3)
    c[0].metric("Precision", "n/a" if summ["precision"] is None else f"{summ['precision']:.0%}")
    c[1].metric("Recall (at most)", "n/a" if summ["recall_upper_bound"] is None else f"{summ['recall_upper_bound']:.0%}")
    c[2].metric("Not reviewed", summ["unreviewed"])
    c = st.columns(3)
    c[0].metric("Flagged", summ["events_flagged"])
    c[1].metric("Real", summ["tp"])
    c[2].metric("False alarms", summ["fp"])
    if summ["unreviewed"]:
        st.warning(f"{summ['unreviewed']} of {summ['events_flagged']} flagged events are not reviewed yet, so the "
                   "precision above describes only the ones you looked at.")
        st.button(f"Review the {summ['unreviewed']} remaining event(s) first", on_click=go, args=("Event Review",),
                  icon=":material/fact_check:")
    if blind_spot(meta):
        st.warning(blind_spot(meta))
    for note in limit_notes(a):
        st.warning(note)
    tabs = st.tabs([":material/verified: Validation", ":material/tune: Tuning", ":material/group: Second reviewer",
                    ":material/edit_note: Recording details", ":material/shield: Privacy",
                    ":material/description: Build report"])
    with tabs[0]:
        st.caption("Precision = real / (real + false alarms). Recall can only be bounded from above, because the app "
                   "cannot know about events nobody marked as missed.")
        if summ["by_kind"]:
            st.write("**By type of conflict**")
            st.dataframe(pd.DataFrame([
                {"type": KIND_LABEL[k], "flagged": v["flagged"], "real": v["tp"], "false alarms": v["fp"],
                 "unsure": v["unsure"], "not reviewed": v["unreviewed"],
                 "precision": "n/a" if v["precision"] is None else f"{v['precision']:.0%}"}
                for k, v in summ["by_kind"].items()]), hide_index=True, width="stretch")
        with st.expander("All numbers"):
            st.json(summ)
        all_recordings_ui()
    with tabs[1]:
        tuning_ui(S, events)
    with tabs[2]:
        second_review_ui(det, S, events, store)
    with tabs[3]:
        metadata_ui()
    with tabs[4]:
        privacy_ui()
    with tabs[5]:
        build_report_ui(cal, S, det, tracks, stats, events, summ)


def kind_rows(by_kind: dict) -> list:
    return [{"type": KIND_LABEL.get(k, k), "flagged": v["flagged"], "real": v["tp"], "false alarms": v["fp"],
             "unsure": v["unsure"], "not reviewed": v["unreviewed"],
             "precision": "n/a" if v["precision"] is None else f"{v['precision']:.0%}"} for k, v in by_kind.items()]


def all_recordings_ui() -> None:
    """Validation over every reviewed real recording, so the submission can quote one set of numbers."""
    st.write("**All recordings**")
    st.caption("Every real (not synthetic) recording that is calibrated and has labels. Precision comes from the "
               "summed counts, not an average of percentages. Each recording uses its own current thresholds.")
    if not st.toggle("Compute it", key="all_val"):
        return
    rows, summs = [], []
    for r, label in list_runs():
        m = read_json(run_dir(r) / "meta.json") or {}
        store = read_json(run_dir(r) / "labels.json", {})
        if m.get("synthetic") or not (store.get("events") or store.get("missed")):
            continue
        a = get_analysis(r)
        if a is None:
            continue
        s = validation.summarize(a[5], store)
        summs.append(s)
        rows.append({"recording": label, "flagged": s["events_flagged"], "real": s["tp"], "false alarms": s["fp"],
                     "unsure": s["unsure"], "not reviewed": s["unreviewed"], "missed": s["missed"],
                     "precision": "n/a" if s["precision"] is None else f"{s['precision']:.0%}"})
    if not rows:
        st.info("No reviewed real recordings yet. Label events on Event Review first.")
        return
    tot = validation.combine(summs)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    st.write(f"Total over {tot['recordings']} recording(s): {tot['events_flagged']} flagged, {tot['tp']} real, "
             f"{tot['fp']} false alarms, {tot['unreviewed']} not reviewed, {tot['missed']} missed. Precision "
             f"{pct(tot['precision'])}, recall at most {pct(tot['recall_upper_bound'])}.")
    st.dataframe(pd.DataFrame(kind_rows(tot["by_kind"])), hide_index=True, width="stretch")


def build_report_ui(cal, S, det, tracks, stats, events, summ) -> None:
    _, cd = load_calibration(rid)
    disclosure = st.text_area("AI-use disclosure (the Congressional App Challenge requires it)", AI_TEXT, height=200)
    if st.button("Build report", type="primary", icon=":material/description:"):
        frames = event_frames(det, S, events[:30])
        page = build_report_html(meta.get("name", rid), {k: v for k, v in meta.items() if k != "settings"},
                                 {k: cd[k] for k in ("width_m", "length_m", "input", "checks") if k in cd}, S, events, summ,
                                 disclosure, png_b64(topdown_map(tracks, events, cal, zones=zone_overlays(cal))),
                                 frames,
                                 notes=[n.replace("**", "") for n in [blind_spot(meta)] if n]
                                       + limit_notes((cal, S, det, tracks, stats, events)),
                                 exposure={k: stats[k] for k in ("road_users", "pairs_together", "events_per_100_vehicles",
                                                                 "events_per_1000_pairs",
                                                                 "events_per_1000_pairs_by_kind", "events_per_hour")},
                                 info=recording_info(rid), explain=explain_event,
                                 speeds=speed_summary(tracks, S, (recording_info(rid).get("posted_limit_mph") or 0)
                                                      / MPS_TO_MPH or None),
                                 zone_rows=zone_summary(events, tracks, load_zones(rid), cal) if load_zones(rid) else None)
        st.session_state[f"report_{rid}"] = page
    if f"report_{rid}" in st.session_state:
        a, b = st.columns(2)
        a.download_button("Download report.html (print to PDF)", st.session_state[f"report_{rid}"],
                          "nearmiss_report.html", "text/html", icon=":material/download:")
        b.download_button("Download events.json", json.dumps(events, indent=1, default=str), "events.json",
                          icon=":material/download:")


def event_frames(det, S: Settings, events) -> dict:
    """Blurred closest-moment frames as base64 JPEGs; none once the original video has been deleted."""
    if not has_source(rid):
        return {}
    return {e["id"]: jpg_b64(event_frame(source_path(rid), det, e, S.privacy_level, True)) for e in events}


def tuning_ui(S: Settings, events) -> None:
    with st.container():
        t = tuning_curve(events, S)
        st.caption(t["caveat"])
        if not t["labelled"]:
            st.info("Label some events as real or false alarm on Event Review first.")
            return
        if not t["enough"]:
            st.info(f"{t['labelled']} labelled events ({t['real']} real) so far. At least 10, with 3 or more real, are "
                    "needed before a suggestion means anything. The tables below are shown for interest.")
        else:
            parts = []
            if t["suggest_ttc"] is not None:
                parts.append(f"flagging by TTC only under **{t['suggest_ttc']} s**")
            if t["suggest_scale"] is not None:
                parts.append(f"distance thresholds at **{t['suggest_scale']:g} x** the current ones")
            if parts:
                st.success("On your labelled events, the best precision that still keeps at least 80% of the real near "
                           "misses comes from " + " and ".join(parts) + ". Re-check it on a different clip before "
                           "trusting it.")
            else:
                st.info("No stricter cutoff keeps 80% of your real near misses, so the current thresholds stay.")
        a, b = st.columns(2)
        a.write("**TTC cutoff**")
        a.dataframe(pd.DataFrame(t["ttc"]), hide_index=True, width="stretch")
        b.write("**Distance cutoff (x current)**")
        b.dataframe(pd.DataFrame(t["distance"]), hide_index=True, width="stretch")


def second_review_ui(det, S: Settings, events, store: dict) -> None:
    with st.container():
        st.caption("Export a single HTML page someone else can open in any browser to label the events, without "
                   "installing anything. Your own labels are left out so their review is independent. They send back "
                   "a small JSON file; load it here to see how often you agree.")
        if st.button("Build review pack"):
            with st.spinner("Rendering blurred frames..."):
                frames = event_frames(det, S, events)
                st.session_state[f"pack_{rid}"] = review_pack.build_pack_html(
                    meta.get("name", rid), events, frames, explain_event, f"{rid}-{uuid.uuid4().hex[:6]}")
        if f"pack_{rid}" in st.session_state:
            st.download_button("Download review_pack.html", st.session_state[f"pack_{rid}"],
                               f"nearmiss_review_pack_{rid}.html", "text/html")
            st.caption("The pack contains blurred frames only. Check them before sending it to anyone.")
        up = st.file_uploader("Load a reviewer's answers (.json)", type=["json"], key=f"rev_up_{rid}")
        if up is None:
            return
        try:
            theirs = review_pack.read_pack_answers(up.getvalue())
        except ValueError as e:
            st.error(str(e))
            return
        ag = review_pack.agreement(events, store.get("events", []), theirs["events"])
        who = theirs["reviewer"] or "the other reviewer"
        if not ag["both_labelled"]:
            st.info(f"No event was labelled by both you and {who} yet.")
            return
        c = st.columns(3)
        c[0].metric("Labelled by both", ag["both_labelled"])
        c[1].metric("Agreement", f"{ag['agree']:.0%}")
        c[2].metric("Cohen's kappa", ag["kappa"], help="Agreement corrected for chance: 1 is perfect, 0 is no better "
                                                       "than guessing. " + review_pack.kappa_words(ag["kappa"]) + ".")
        st.caption(f"Kappa {ag['kappa']} is usually described as *{review_pack.kappa_words(ag['kappa'])}* agreement "
                   "(Landis and Koch).")
        st.write("Rows: your label · columns: " + who)
        names = {l: review_pack.LABEL_TEXT[l] for l in review_pack.LABELS}
        st.dataframe(pd.DataFrame(ag["table"]).T.rename(index=names, columns=names), width="stretch")
        if ag["disagreements"]:
            st.write("**Where you disagree** (worth watching again together)")
            st.dataframe(pd.DataFrame(ag["disagreements"]).replace(names), hide_index=True, width="stretch")
        if st.button("Save these answers with this recording"):
            write_json(run_dir(rid) / f"review_{uuid.uuid4().hex[:6]}.json", theirs)
            st.success("Saved. They are included in run.json exports.")


INFO_FIELDS = [("date", "Date recorded"), ("time", "Time of day"), ("location", "Location / intersection"),
               ("weather", "Weather"), ("lighting", "Lighting"), ("camera", "Camera (phone, action cam, drone model)"),
               ("camera_height_m", "Camera height or drone altitude (m)"), ("notes", "Anything unusual")]


def metadata_ui() -> None:
    with st.container():
        st.caption("Saved with the recording and printed in the report, so old results stay interpretable.")
        info = recording_info(rid)
        with st.form(f"info_{rid}"):
            vals = {}
            cols = st.columns(2)
            for i, (k, label) in enumerate(INFO_FIELDS):
                if k == "lighting":
                    opts = ["", "daylight", "overcast", "dusk / dawn", "night, street lights", "night, dark"]
                    vals[k] = cols[i % 2].selectbox(label, opts, index=opts.index(info.get(k, "")) if info.get(k, "") in opts else 0)
                else:
                    vals[k] = cols[i % 2].text_input(label, info.get(k, ""))
            if st.form_submit_button("Save details"):
                info.update(vals)
                write_json(run_dir(rid) / "recording_info.json", info)
                st.success("Saved.")


def privacy_ui() -> None:
    with st.container():
        st.caption("Everything below is on this computer only. The original video is the only file with unblurred "
                   "faces and plates.")
        rows = []
        for p in sorted(run_dir(rid).iterdir()):
            what = {"source.mp4": "original video (NOT blurred)", "annotated.mp4": "annotated video (blurred)",
                    "detections.csv": "boxes per frame (no images)", "labels.json": "your labels",
                    "calibration.json": "calibration", "roi.json": "road area", "settings.json": "thresholds",
                    "motion.json": "camera motion", "meta.json": "recording facts",
                    "recording_info.json": "recording details", "zones.json": "named zones"}.get(p.name)
            if what is None and p.name.startswith("slowmo_"):
                what = "slow-motion video (blurred)"
            if what is None:
                what = "event clip (blurred)" if p.name.startswith("clip_") else (
                    "second reviewer's answers" if p.name.startswith("review_") else "other")
            rows.append({"file": p.name, "contains": what, "size (MB)": round(p.stat().st_size / 1e6, 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if has_source(rid):
            st.write("**Delete the original video.** The analysis, labels, blurred video and report keep working. "
                     "You can no longer redo detection or calibration, see unblurred frames, or put frames in a report "
                     "or review pack.")
            ok = st.checkbox("I understand this cannot be undone", key=f"del_src_ok_{rid}")
            if st.button("Delete original video", disabled=not ok):
                (run_dir(rid) / "source.mp4").unlink(missing_ok=True)
                st.rerun()


# ================================================================== METHOD
def page_method():
    page_header("Method", "Method and limits")
    st.markdown("""
**Pipeline.** For street-level cameras a pretrained YOLOv8 model (COCO) finds people, bicycles, motorcycles, cars,
buses and trucks. For drone footage the default is a pretrained aerial model (YOLO11-OBB, trained on the DOTA aerial
dataset), which finds vehicles seen from above and has no pedestrian or bicycle class; car or truck is decided by
vehicle length. A YOLO11s fine-tuned on VisDrone (hand-labelled drone images, see tools/train_visdrone_colab.ipynb)
also finds pedestrians and bikes from above, but mistakes some street furniture for them. Real pedestrians and riders
move, so one that never moves more than 3 of its own box sizes is dropped (and hidden in the drone video). Each track
keeps one class label, which only changes when another class clearly outvotes it, so labels do not flicker. ByteTrack links detections into tracks. Boxes that go
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

**Privacy.** For street-level footage, faces and plates are pixelated by position heuristics (head and lower-box
regions), which is not perfect: anything the detector misses is not blurred. For drone footage those regions do not
exist, so each whole road user is pixelated instead. Footage stays on this computer.
""")


{"Detect": page_detect, "Calibrate": page_calibrate, "Results": page_results, "Event Review": page_review,
 "Report": page_report, "Method": page_method}[page]()
