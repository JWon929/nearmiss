"""
NearMiss: screening tool for potential traffic conflicts at an intersection (not a crash predictor).

Steps: Detect -> Calibrate -> Results -> Event Review -> Report. Separate workflow: Proposed designs (simulated
trajectories of an intersection that is not built yet). Method explains everything.

Detection uses YOLO models (YOLOv8 / COCO for street-level cameras; YOLO11-OBB / DOTA, or a YOLO11s fine-tuned on
VisDrone, for drones) and ByteTrack; everything after that (camera-motion compensation, calibration, TTC / closest
approach / PET, movements, privacy blur, validation) is in engine/.

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import time
import uuid
import zipfile
from dataclasses import asdict, replace
from html import escape as html_escape
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import figures

from detect_job import load_job, new_job, run_job
from detector_tracker import (MODEL_CHOICES, VISDRONE_MODEL, RenderOptions, RoadUserTracker, TrackerSettings,
                              aerial_choices, auto_stride, device_label, finds_people, intel_devices, pick_device,
                              render_frame, render_from_detections)
from engine import demo, design, review_pack, validation
from engine.analysis import (analyze, build_report_html, cut_clip, distance_series, event_frame, events_table,
                             load_tracker_csv, topdown_map)
from engine.calibration import Calibration, CalibrationError, fit_scale
from engine.config import (KIND_LABEL, PRESET_NOTES, PRESETS, SEVERITY_LEVELS, Settings, apply_preset,
                           matching_preset)
from engine.conflicts import ENCOUNTER_PAD_S
from engine.insights import (MPS_TO_MPH, explain_event, sensitivity, speed_summary, tuning_curve,
                             vehicle_size_check)
from engine.motion import load_motion
from engine.movements import annotate_events, movement_counts, parse_phases
from engine.quality import calibration_status, quality_items
from engine.roi import as_polygon as roi_polygon
from engine.slowmo import make_slowmo
from engine.zones import ZONE_COLORS, ZONE_KINDS, tag_events, zone_polygons_map, zone_summary
from video_utils import AnnotatedVideoWriter, grab_frame, read_metadata

RUNS = Path(__file__).parent / "runs"
RUNS.mkdir(exist_ok=True)
TRASH = RUNS / ".trash"
CSV_HEADER = ["frame", "time_s", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]
DISPLAY_W = 900
STEPS = ["Detect", "Calibrate", "Results", "Event Review", "Report"]
LEARN = ["Purpose", "Method", "Proposed designs", "Help & contact"]
PAGES = ["Home"] + STEPS + LEARN
PAGE_ICONS = {"Home": ":material/home:", "Detect": ":material/videocam:", "Calibrate": ":material/straighten:", "Results": ":material/analytics:",
              "Event Review": ":material/fact_check:", "Report": ":material/description:",
              "Purpose": ":material/flag:", "Proposed designs": ":material/architecture:",
              "Method": ":material/menu_book:", "Help & contact": ":material/support_agent:"}
# One accent colour per page: its banner, its sidebar button and its buttons use it, so each step is easy to tell apart.
PAGE_COLORS = {"Home": "#0f766e", "Detect": "#2563eb", "Calibrate": "#7c3aed", "Results": "#0d9488", "Event Review": "#ea580c",
               "Report": "#16a34a", "Purpose": "#db2777", "Method": "#4f46e5", "Proposed designs": "#0891b2",
               "Help & contact": "#64748b"}
INQUIRIES = Path(__file__).parent / "inquiries"          # questions and feedback typed into the app (git-ignored)
PREFS_FILE = Path(__file__).parent / "preferences.json"   # this computer's defaults (git-ignored)
LABEL_TEXT = {"tp": "Real near miss", "fp": "False alarm", "unsure": "Unsure"}
SEV_HEX = {"serious": "#c62828", "moderate": "#ef6c00", "candidate": "#b28704", "none": "#6b7280"}
LEVEL_HEX = {"good": "#2e7d32", "unchecked": "#b28704", "problem": "#c62828"}
GROUP_NAME = {"ped": "Pedestrians", "bike": "Bicycles", "veh": "Vehicles"}

st.set_page_config(page_title="NearMiss", page_icon=":material/traffic:", layout="wide")
st.markdown("""<style>
/* ---- base: system fonts only (no web requests), calmer spacing ---- */
html, body, [class*="st-"], .stMarkdown, button, input, textarea {
  font-family: "Segoe UI Variable", "Segoe UI", Inter, -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue",
               Arial, sans-serif;}
[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"], [data-testid*="IconMaterial"],
span[data-testid$="Icon"], .material-symbols-rounded {
  font-family: "Material Symbols Rounded" !important;}          /* keep Streamlit's icon font */
.block-container {padding-top: 1.6rem; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -.01em;}
/* ---- page hero ---- */
.nm-hero {position: relative; overflow: hidden; border-radius: 16px; padding: 18px 22px 16px 22px; margin: 0 0 14px 0;
  background: linear-gradient(120deg, rgba(31,95,191,.16) 0%, rgba(31,95,191,.06) 45%, rgba(0,150,136,.10) 100%);
  border: 1px solid rgba(31,95,191,.22);}
.nm-hero:after {content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
  border-radius: 50%; background: radial-gradient(circle, rgba(31,95,191,.18), rgba(31,95,191,0) 70%);}
.nm-hero .row {display: flex; align-items: center; gap: 14px;}
.nm-hero .icon {flex: 0 0 auto; width: 46px; height: 46px; border-radius: 12px; display: flex; align-items: center;
  justify-content: center; font-size: 26px; background: #1f5fbf; color: #fff; box-shadow: 0 4px 14px rgba(31,95,191,.35);
  font-family: "Material Symbols Rounded"; font-feature-settings: "liga";}
.nm-hero h1 {font-size: 1.75rem; font-weight: 700; margin: 0; padding: 0; line-height: 1.2;}
.nm-hero .intro {opacity: .82; margin-top: 6px; max-width: 900px; font-size: .97rem;}
.nm-hero .pill {display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; color: #1f5fbf; background: rgba(31,95,191,.12); border-radius: 999px; padding: 2px 9px;
  margin-bottom: 4px;}
.nm-steps {display: flex; gap: 4px; margin-top: 12px;}
.nm-steps span {flex: 1; height: 5px; border-radius: 3px; background: rgba(128,128,128,.22);}
.nm-steps span.on {background: #1f5fbf;}
.nm-steps span.done {background: rgba(31,95,191,.45);}
/* ---- sidebar brand ---- */
.nm-brand {display: flex; align-items: center; gap: 10px; margin: -6px 0 6px 0;}
.nm-brand .logo {width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #1f5fbf, #00897b);
  display: flex; align-items: center; justify-content: center; color: #fff; font-family: "Material Symbols Rounded";
  font-size: 24px; box-shadow: 0 3px 10px rgba(31,95,191,.35);}
.nm-brand .name {font-size: 1.3rem; font-weight: 750; letter-spacing: -.01em; line-height: 1.1;}
.nm-brand .tag {font-size: .74rem; opacity: .7;}
/* ---- cards, tabs, buttons ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px !important;}
div[data-testid="stExpander"] details {border-radius: 12px;}
button[kind="primary"], button[kind="secondary"] {border-radius: 10px !important; font-weight: 600 !important;}
button[kind="primary"] {box-shadow: 0 3px 10px rgba(31,95,191,.25);}
div[data-baseweb="tab-list"] {gap: 2px;}
button[data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 6px 12px;}
.nm-navhead {font-size: .72rem; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; opacity: .6;
  margin: 12px 0 4px 2px;}
section[data-testid="stSidebar"] div[class*="st-key-nav_"] {margin-bottom: -8px;}
section[data-testid="stSidebar"] div[class*="st-key-nav_"] button {box-shadow: none !important; min-height: 2.1rem;}
section[data-testid="stSidebar"] div[class*="st-key-nav_"] button p {font-weight: 600;}
.nm-pager-gap {height: 18px;}
/* ---- home ---- */
.nm-homehero {display: flex; align-items: center; gap: 24px; border-radius: 22px; padding: 28px 30px; margin-bottom: 18px;
  background: linear-gradient(125deg, rgba(15,118,110,.22) 0%, rgba(37,99,235,.14) 55%, rgba(219,39,119,.12) 100%);
  border: 1px solid rgba(15,118,110,.28); box-shadow: 0 8px 28px rgba(15,23,42,.08); position: relative; overflow: hidden;}
.nm-homehero .left {flex: 1 1 auto; min-width: 0;}
.nm-homehero .brandrow {display: flex; align-items: center; gap: 18px;}
.nm-homehero .biglogo {flex: 0 0 auto; width: 92px; height: 92px; border-radius: 24px; color: #fff; font-size: 60px;
  display: flex; align-items: center; justify-content: center; font-family: "Material Symbols Rounded";
  background: linear-gradient(135deg, #2563eb, #0f766e); box-shadow: 0 10px 26px rgba(37,99,235,.38);}
.nm-homehero .kicker {font-size: .85rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: #0f766e;}
.nm-homehero .big {font-size: 3.1rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.05;}
.nm-homehero .lead {font-size: 1.12rem; opacity: .85; margin-top: 14px; max-width: 640px; line-height: 1.5;}
.nm-homehero .promises {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;}
.nm-homehero .promises span {display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px;
  font-size: .85rem; font-weight: 600; background: rgba(255,255,255,.6); border: 1px solid rgba(15,118,110,.25);}
.nm-homehero .promises i {font-style: normal; font-family: "Material Symbols Rounded"; font-size: 18px; color: #0f766e;}
.nm-homehero .art {flex: 0 0 300px;}
.nm-homehero .art img {width: 100%; height: auto; display: block;}
@media (max-width: 1000px) {.nm-homehero .art {display: none;} .nm-homehero .big {font-size: 2.4rem;}}
@media (prefers-color-scheme: dark) {.nm-homehero .promises span {background: rgba(255,255,255,.08);}}
.nm-act {min-height: 92px; line-height: 1.35;}
.nm-act .i {float: left; width: 40px; height: 40px; border-radius: 11px; margin: 2px 12px 6px 0; color: #fff;
  display: flex; align-items: center; justify-content: center; font-family: "Material Symbols Rounded"; font-size: 23px;}
.nm-act b {font-size: 1.05rem;}
.nm-dots {display: flex; gap: 4px; margin: 8px 0 4px 0;}
.nm-dots span {flex: 1; height: 6px; border-radius: 3px;}
.nm-rec b {display: block; font-size: 1rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.nm-rec small {opacity: .7;}
.st-key-pager {background: rgba(128,128,128,.04);}
/* ---- method and purpose pages ---- */
.nm-card {border-radius: 14px; padding: 14px 16px; height: 100%; border: 1px solid rgba(128,128,128,.2);}
.nm-card h4 {margin: 0 0 4px 0; font-size: 1.02rem;}
.nm-card p {margin: 0; font-size: .92rem; opacity: .9;}
div[data-testid="stVerticalBlockBorderWrapper"] {background: rgba(255,255,255,.55);}
@media (prefers-color-scheme: dark) {
  div[data-testid="stVerticalBlockBorderWrapper"], .nm-fig {background: rgba(255,255,255,.04) !important;}
}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
div[data-testid="stImage"] img, div[data-testid="stVideo"] video {border-radius: 12px;}
.nm-fig {border-radius: 14px; border: 1px solid rgba(128,128,128,.18); padding: 10px; background: rgba(255,255,255,.55);
  text-align: center; margin: 4px 0 6px 0;}
.nm-fig img {max-width: 100%; height: auto;}
.nm-fig figcaption {font-size: .8rem; opacity: .75; margin-top: 4px;}
.nm-section {display: flex; align-items: center; gap: 10px; margin: 22px 0 8px 0;}
.nm-section .dot {width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center; justify-content: center;
  color: #fff; font-family: "Material Symbols Rounded"; font-size: 19px;}
.nm-section h3 {margin: 0; padding: 0; font-size: 1.25rem;}
.nm-footer {margin-top: 28px; padding-top: 10px; border-top: 1px solid rgba(128,128,128,.2); font-size: .78rem;
  opacity: .65;}
.nm-faq-q {font-weight: 600;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: 8px 12px;
  background: rgba(128,128,128,.04);}
div[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .8;}
.nm-badge {display: inline-block; padding: 2px 10px; border-radius: 999px; color: #fff; font-weight: 600;
  font-size: .85rem; letter-spacing: .02em;}
.nm-intro {opacity: .8; margin-top: -.6rem; margin-bottom: .4rem;}
.nm-step {font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .6;
  margin-bottom: -.4rem;}
.nm-chips {display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 6px 0;}
.nm-chip {border: 1px solid rgba(128,128,128,.35); border-radius: 999px; padding: 1px 10px; font-size: .8rem;
  white-space: nowrap;}
.nm-chip b {font-weight: 600;}
.nm-kpis {display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 10px 0;}
.nm-kpi {flex: 1 1 105px; border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: 8px 12px;
  background: linear-gradient(180deg, rgba(31,95,191,.06), rgba(128,128,128,.02)); min-width: 0;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);}
.nm-kpi .v {font-size: 1.25rem; font-weight: 650; line-height: 1.3; overflow-wrap: anywhere;}
.nm-kpi .l {font-size: .74rem; opacity: .75;}
.nm-stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 6px; margin: 6px 0;}
.nm-stats > div {border: 1px solid rgba(128,128,128,.25); border-radius: 8px; padding: 4px 8px;}
.nm-stats .v {font-size: 1.1rem; font-weight: 650; overflow-wrap: anywhere;}
.nm-stats .l {font-size: .72rem; opacity: .75;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {padding: 2px 0;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ small UI pieces
def hex_rgb(c: str) -> str:
    return ",".join(str(int(c[i:i + 2], 16)) for i in (1, 3, 5))


def page_header(page: str, title: str, intro: str = "") -> None:
    """The banner at the top of every page, in the page's own colour: icon, step number or section, title, a one-line
    explanation and, on the procedure steps, which steps are done. Also tints this page's buttons and tabs."""
    icon = PAGE_ICONS.get(page, ":material/info:").split("/")[-1].strip(":")
    c = PAGE_COLORS.get(page, "#2563eb")
    rgb_ = hex_rgb(c)
    pill = f"Procedure · step {STEPS.index(page) + 1} of {len(STEPS)}" if page in STEPS else \
        ("Home" if page == "Home" else "Learn & help")
    bars = ""
    if page in STEPS:
        done = steps_done(st.session_state.get("rid")) if st.session_state.get("rid") else {}
        bars = "<div class='nm-steps'>" + "".join(
            f"<span style='background:{PAGE_COLORS[p] if (p == page or done.get(p)) else 'rgba(128,128,128,.22)'};"
            f"opacity:{1 if p == page else (.55 if done.get(p) else 1)}' title='{p}'></span>" for p in STEPS) + "</div>"
    intro_html = f"<div class='intro'>{intro}</div>" if intro else ""
    st.markdown(f"""<style>
section.main button[kind="primary"], div[data-testid="stMain"] button[kind="primary"] {{
  background: {c} !important; border-color: {c} !important; box-shadow: 0 3px 10px rgba({rgb_},.3) !important;}}
div[data-testid="stMain"] button[data-baseweb="tab"][aria-selected="true"] p {{color: {c} !important;}}
div[data-testid="stMain"] div[data-baseweb="tab-highlight"] {{background-color: {c} !important;}}
</style>
<div class='nm-hero' style='background: linear-gradient(120deg, rgba({rgb_},.20) 0%, rgba({rgb_},.07) 50%,
  rgba({rgb_},.14) 100%); border-color: rgba({rgb_},.28)'>
<div class='row'><div class='icon' style='background:{c}; box-shadow: 0 4px 14px rgba({rgb_},.4)'>{icon}</div>
<div><div class='pill' style='color:{c}; background: rgba({rgb_},.13)'>{pill}</div><h1>{html_escape(title)}</h1></div>
</div>{intro_html}{bars}</div>""", unsafe_allow_html=True)


SCROLL_JS = """<script>
const d = window.parent.document;
function toTop() {
  for (const el of [d.querySelector('[data-testid="stMain"]'), d.querySelector('[data-testid="stAppViewContainer"]'),
                    d.querySelector('section.main'), d.scrollingElement]) {
    if (el && el.scrollTop > 0) el.scrollTo({top: 0, behavior: 'instant'});
  }
}
if (%s) { toTop(); setTimeout(toTop, 80); setTimeout(toTop, 400); }
if (!d.nearmissTabScroll) {      // clicking a tab further down the page brings the tab row to the top
  d.nearmissTabScroll = true;
  d.addEventListener('click', (e) => {
    const tab = e.target.closest('button[data-baseweb="tab"]');
    if (!tab) return;
    const bar = tab.closest('[data-baseweb="tab-list"]');
    if (bar && bar.getBoundingClientRect().top < 0) bar.scrollIntoView({block: 'start', behavior: 'smooth'});
  }, true);
}
</script>"""


def scroll_helper(new_page: bool, page: str) -> None:
    """Start each page at the top (Streamlit keeps the old scroll position), and keep tab switches in view.
    The page name is in the script so that every page change gives a new script, which the browser then runs."""
    html = SCROLL_JS % ("true" if new_page else "false") + f"<!-- {html_escape(page)} -->"
    if hasattr(st, "iframe"):
        st.iframe(html, height=1)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=0)


def step_pager(page: str) -> None:
    """Back / Next between the procedure steps, at the bottom of each step."""
    if page not in STEPS:
        return
    i = STEPS.index(page)
    st.markdown("<div class='nm-pager-gap'></div>", unsafe_allow_html=True)
    with st.container(border=True, key="pager"):
        a, mid, b = st.columns([2, 3, 2], vertical_alignment="center")
        if i > 0:
            a.button(f"Back: {STEPS[i - 1]}", icon=":material/arrow_back:", on_click=go, args=(STEPS[i - 1],),
                     width="stretch", key="pager_back")
        mid.markdown(f"<div style='text-align:center;opacity:.75;font-size:.85rem'>Step {i + 1} of {len(STEPS)}: "
                     f"<b style='color:{PAGE_COLORS[page]}'>{page}</b></div>", unsafe_allow_html=True)
        if i < len(STEPS) - 1:
            b.button(f"Next: {STEPS[i + 1]}", icon=":material/arrow_forward:", on_click=go,
                     args=(STEPS[i + 1],), type="primary", width="stretch", key="pager_next")


def badge(severity: str, provisional: bool = False) -> str:
    text = severity.capitalize() + (" · provisional" if provisional else "")
    return f"<span class='nm-badge' style='background:{SEV_HEX.get(severity, '#6b7280')}'>{text}</span>"


def chips(items) -> None:
    """Compact status chips: (label, value, colour or None)."""
    html = "".join(f"<span class='nm-chip' style='{f'border-color:{c};color:{c}' if c else ''}'>{html_escape(l)}: "
                   f"<b>{html_escape(str(v))}</b></span>" for l, v, c in items)
    st.markdown(f"<div class='nm-chips'>{html}</div>", unsafe_allow_html=True)


def kpis(items) -> None:
    """A strip of small number cards that wrap instead of truncating: (label, value)."""
    html = "".join(f"<div class='nm-kpi'><div class='v'>{html_escape(str(v))}</div><div class='l'>{html_escape(l)}</div>"
                   "</div>" for l, v in items)
    st.markdown(f"<div class='nm-kpis'>{html}</div>", unsafe_allow_html=True)


def stat_grid(items) -> None:
    html = "".join(f"<div title='{html_escape(h)}'><div class='v'>{html_escape(str(v))}</div><div class='l'>"
                   f"{html_escape(l)}</div></div>" for l, v, h in items)
    st.markdown(f"<div class='nm-stats'>{html}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ run storage
def run_dir(rid: str) -> Path:
    return RUNS / rid


def read_json(path: Path, default=None):
    """Missing file -> default. A file that exists but cannot be read is NOT silently replaced by defaults: it is
    reported at the top of the page (see damaged_files_ui)."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        st.session_state.setdefault("_damaged", set()).add(str(path))
        return default


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1))


def display_name(m: dict, p: Path) -> str:
    return m.get("display_name") or m.get("name", p.name)


def run_label(p: Path, m: dict) -> str:
    """Name shown in the recording picker. File names repeat (same clip run twice), so add view, date and status."""
    if m.get("synthetic"):
        return f"Synthetic demo  ·  {m.get('display_name') or p.name}"
    view = {"overhead": "drone", "angled": "street-level"}.get(m.get("view"), "old version")
    when = time.strftime("%b %d %H:%M", time.localtime(m.get("created", 0)))
    done = "calibrated" if (p / "calibration.json").exists() else "needs calibration"
    return f"{display_name(m, p)}  ·  {when}  ·  {view}  ·  {done}"


def list_runs(search: str = "") -> list:
    """(run id, label) pairs, newest first by creation time. Folders without meta.json (interrupted runs, the trash)
    are not recordings."""
    found = []
    for p in RUNS.iterdir():
        if p.is_dir() and not p.name.startswith(".") and (p / "meta.json").exists():
            m = read_json(p / "meta.json")
            if m:
                found.append((p, m))
    found.sort(key=lambda pm: pm[1].get("created", 0), reverse=True)
    rows = [(p.name, run_label(p, m)) for p, m in found]
    return [r for r in rows if search.lower() in r[1].lower()] if search else rows


def current():
    rid = st.session_state.get("rid")
    if rid and (run_dir(rid) / "meta.json").exists():
        return rid, read_json(run_dir(rid) / "meta.json", {})
    return None, None


def source_path(rid: str) -> str:
    return str(run_dir(rid) / "source.mp4")


def has_source(rid: str) -> bool:
    """False once the user deleted the original (unblurred) video to protect privacy."""
    return (run_dir(rid) / "source.mp4").exists()


def recording_info(rid: str) -> dict:
    """The recording details (date, place, weather, north, signal phases...)."""
    return read_json(run_dir(rid) / "recording_info.json", {}) or {}


def load_calibration(rid: str):
    d = read_json(run_dir(rid) / "calibration.json")
    return (Calibration.from_dict(d), d) if d else (None, None)


def load_settings(rid: str) -> Settings:
    return Settings.from_dict(read_json(run_dir(rid) / "settings.json"))


def load_labels(rid: str) -> tuple:
    """(cleaned labels store, problems). Malformed entries are dropped and reported, never trusted."""
    raw = read_json(run_dir(rid) / "labels.json", {})
    return validation.clean_store(raw or {})


def save_labels(rid: str, store: dict) -> None:
    write_json(run_dir(rid) / "labels.json", store)


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def load_roi(rid: str) -> list:
    return (read_json(run_dir(rid) / "roi.json", {}) or {}).get("points", [])


def load_zones(rid: str) -> list:
    """Named zones: [{"name", "kind", "points"}], drawn on the calibration frame."""
    return (read_json(run_dir(rid) / "zones.json", {}) or {}).get("zones", [])


def motion_summary(rid: str):
    return (read_json(run_dir(rid) / "motion.json") or {}).get("summary")


def camera_moved(rid: str) -> bool:
    """Measured, not assumed: a stabilisation file exists for every drone run, even when the camera stayed still."""
    ms = motion_summary(rid)
    return bool(ms) and ms.get("max_shift_px", 0) >= 3


def privacy_of(m: dict):
    """What blur was actually applied to the derived videos, as saved at detection (None for older recordings)."""
    if m.get("synthetic"):
        return {"blur": True, "level": "n/a (drawn scene, no people)"}
    return m.get("privacy")


def video_privacy_text(m: dict) -> str:
    p = privacy_of(m)
    if p is None:
        return "blur setting not recorded (made before it was saved)"
    return f"blurred ({p['level']})" if p.get("blur") else "NOT blurred"


def zone_overlays(cal, rid_: str) -> list:
    zones = load_zones(rid_)
    return [(z["name"], p, ZONE_COLORS[i % len(ZONE_COLORS)])
            for i, (z, p) in enumerate(zip(zones, zone_polygons_map(zones, cal))) if p is not None]


def cal_core(d: dict) -> dict:
    return {k: d[k] for k in ("image_points", "width_m", "length_m", "H", "rot", "frame_size")}  # not the checks


@st.cache_data(show_spinner="Analysing tracks...")
def cached_analysis(rid: str, csv_mtime: float, cal_json: str, settings_json: str, motion_mtime: float,
                    roi_json: str = "[]", observation_s: float = 0.0):
    det = load_tracker_csv(run_dir(rid) / "detections.csv")
    cal = Calibration.from_dict(json.loads(cal_json))
    S = Settings.from_dict(json.loads(settings_json))
    tracks, stats, events = analyze(det, cal, S, load_motion(read_json(run_dir(rid) / "motion.json")),
                                    roi_polygon(json.loads(roi_json)), observation_s or None)
    return det, tracks, stats, events


@st.cache_data(show_spinner=False)
def cached_size_check(rid: str, csv_mtime: float, cal_json: str, settings_json: str, motion_mtime: float):
    det = load_tracker_csv(run_dir(rid) / "detections.csv")
    return vehicle_size_check(det, Calibration.from_dict(json.loads(cal_json)),
                              Settings.from_dict(json.loads(settings_json)),
                              load_motion(read_json(run_dir(rid) / "motion.json")))


def analysis_keys(rid_: str):
    _, d = load_calibration(rid_)
    S = load_settings(rid_)
    return (rid_, mtime(run_dir(rid_) / "detections.csv"), json.dumps(cal_core(d)), json.dumps(S.to_dict()),
            mtime(run_dir(rid_) / "motion.json"))


def cal_status(rid_: str, m: dict) -> dict:
    _, d = load_calibration(rid_)
    size = None
    if d is not None and (run_dir(rid_) / "detections.csv").exists():
        size = cached_size_check(*analysis_keys(rid_))
    return calibration_status(d, size, bool(m.get("synthetic")))


def get_analysis(rid_: str):
    """(cal, S, det, tracks, stats, events) with labels, zones, movements, signal phases and the provisional flag
    applied, or None when the recording is not calibrated."""
    cal, d = load_calibration(rid_)
    if cal is None or not (run_dir(rid_) / "detections.csv").exists():
        return None
    m = read_json(run_dir(rid_) / "meta.json", {}) or {}
    keys = analysis_keys(rid_)
    det, tracks, stats, events = cached_analysis(*keys, json.dumps(load_roi(rid_)), float(m.get("duration_s") or 0))
    events = [json.loads(json.dumps(e)) for e in events]          # deep copy: the cache must not be modified
    store, _ = load_labels(rid_)
    validation.apply_labels(events, store)
    tag_events(events, load_zones(rid_), cal)
    info = recording_info(rid_)
    phases, _ = parse_phases(info.get("signal_phases", ""))
    annotate_events(events, tracks, float(info.get("north_deg", 0) or 0), phases)
    provisional = cal_status(rid_, m)["provisional"]
    for e in events:
        e["key"] = validation.event_key(e)
        e["provisional"] = provisional
    return cal, load_settings(rid_), det, tracks, stats, events


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def jpg_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def blind_spot(m: dict) -> bool:
    """True when the model could not see pedestrians or bikes, so their conflicts were NOT MEASURED."""
    if not m or m.get("synthetic"):
        return False
    return not finds_people(m.get("model_used") or (m.get("settings") or {}).get("model_name", ""))


def measured(m: dict, group_or_kind: str) -> bool:
    return not (blind_spot(m) and group_or_kind in ("ped", "bike", "ped-veh", "bike-veh"))


# ------------------------------------------------------------------ demo
def create_demo_run() -> str:
    rid_ = "demo-" + uuid.uuid4().hex[:6]
    d = run_dir(rid_)
    d.mkdir(parents=True)
    shutil.copy(demo.DEMO_VIDEO, d / "source.mp4")
    shutil.copy(demo.DEMO_VIDEO, d / "annotated.mp4")
    t = demo.truth()
    m = read_metadata(source_path(rid_))
    m.update(name="Synthetic demo intersection", synthetic=True, created=time.time(),
             note="Scripted road users on a drawn road. Checks the maths only, NOT real-world accuracy.")
    write_json(d / "meta.json", m)
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
    return rid_


# ------------------------------------------------------------------ report freshness
def report_signature(rid_: str) -> str:
    """Changes whenever anything the report shows could change: labels, calibration, settings, areas, details."""
    parts = [str(mtime(run_dir(rid_) / f)) for f in ("labels.json", "calibration.json", "settings.json", "roi.json",
                                                     "zones.json", "recording_info.json", "detections.csv",
                                                     "meta.json")]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def report_state(rid_: str):
    """('none' | 'current' | 'outdated', meta of the saved report)."""
    meta_r = read_json(run_dir(rid_) / "report.json")
    if not meta_r or not (run_dir(rid_) / "report.html").exists():
        return "none", None
    return ("current" if meta_r.get("signature") == report_signature(rid_) else "outdated"), meta_r


# ------------------------------------------------------------------ sidebar
def go(page: str, **state) -> None:
    """Button callback: switch page (and set any other session values) before the next run draws the sidebar."""
    st.session_state["page"] = page
    st.session_state.update(state)


def next_button(page: str, text: str) -> None:
    st.button(f"Next: {text}", on_click=go, args=(page,), type="primary", icon=":material/arrow_forward:")


def steps_done(rid_) -> dict:
    if not rid_:
        return {}
    d = run_dir(rid_)
    calibrated = (d / "calibration.json").exists()
    out = {"Detect": (d / "detections.csv").exists(), "Calibrate": calibrated, "Results": calibrated}
    a = get_analysis(rid_) if calibrated else None
    if a is not None:
        out["Event Review"] = all(e.get("label") for e in a[5])        # every flag reviewed (or nothing flagged)
    out["Report"] = report_state(rid_)[0] == "current"
    return out


def pick_recording() -> None:
    st.session_state["rid"] = st.session_state["rec_pick"]


def trash_recording(rid_: str) -> None:
    TRASH.mkdir(exist_ok=True)
    dest = TRASH / rid_
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(run_dir(rid_)), str(dest))
    write_json(dest / "deleted.json", {"deleted_at": time.time()})
    st.session_state.pop("rid", None)


def recording_options_ui(rid_: str, m: dict) -> None:
    with st.expander("This recording", icon=":material/settings:"):
        new = st.text_input("Name", display_name(m, run_dir(rid_)), key=f"rename_{rid_}")
        if new.strip() and new.strip() != display_name(m, run_dir(rid_)) and st.button("Rename"):
            m["display_name"] = new.strip()
            write_json(run_dir(rid_) / "meta.json", m)
            st.rerun()
        st.caption("Delete moves it to *Recently deleted*, where it can be restored.")
        if st.button("Delete this recording", icon=":material/delete:"):
            trash_recording(rid_)
            st.rerun()


def trash_ui() -> None:
    if not TRASH.exists():
        return
    items = sorted([p for p in TRASH.iterdir() if p.is_dir()], key=lambda p: -mtime(p / "deleted.json"))
    if not items:
        return
    with st.expander(f"Recently deleted ({len(items)})", icon=":material/restore_from_trash:"):
        for p in items:
            m = read_json(p / "meta.json", {}) or {}
            l, r = st.columns([3, 2])
            l.caption(display_name(m, p))
            if r.button("Restore", key=f"restore_{p.name}"):
                if not run_dir(p.name).exists():
                    shutil.move(str(p), str(run_dir(p.name)))
                    (run_dir(p.name) / "deleted.json").unlink(missing_ok=True)
                    st.session_state["rid"] = p.name
                st.rerun()
        if st.button("Empty it (cannot be undone)", type="primary"):
            shutil.rmtree(TRASH, ignore_errors=True)
            st.rerun()


def clean_interrupted() -> int:
    """Folders left by a detection that was stopped or crashed (no meta.json), older than 10 minutes, that cannot be
    resumed. Ones with a checkpoint (partial/job.json) are kept and offered on Detect."""
    n = 0
    for p in RUNS.iterdir():
        if p.is_dir() and not p.name.startswith(".") and not (p / "meta.json").exists() \
                and load_job(p) is None and time.time() - p.stat().st_mtime > 600:
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    return n


with st.sidebar:
    st.markdown("<div class='nm-brand'><div class='logo'>traffic</div><div><div class='name'>NearMiss</div>"
                "<div class='tag'>Traffic conflict screening</div></div></div>", unsafe_allow_html=True)
    st.caption("Finds where road users came close. Not a crash predictor.")
    if "_cleaned" not in st.session_state:
        st.session_state["_cleaned"] = clean_interrupted()
    all_runs = list_runs()
    if all_runs:
        search = st.text_input("Find a recording", placeholder="name, date, drone...", label_visibility="collapsed",
                               key="run_search")
        runs = list_runs(search) or all_runs
        ids = [r[0] for r in runs]
        if search and not list_runs(search):
            st.caption("No recording matches; showing all.")
        if st.session_state.get("rid") not in ids:         # a search that excludes the current one opens the first match
            st.session_state["rid"] = ids[0]
        st.session_state["rec_pick"] = st.session_state["rid"]
        st.selectbox("Recording", ids, key="rec_pick", format_func=lambda i: dict(runs)[i], on_change=pick_recording)
    if st.session_state.get("page") not in PAGES:
        st.session_state["page"] = "Home"
    page = st.session_state["page"]
    done = steps_done(st.session_state["rid"]) if st.session_state.get("rid") else {}
    nav_css = "".join(
        f".st-key-nav_{i} button {{border-left: 4px solid {PAGE_COLORS[p]} !important; justify-content: flex-start;}}"
        f".st-key-nav_{i} button[kind='primary'] {{background: {PAGE_COLORS[p]} !important; border-color: "
        f"{PAGE_COLORS[p]} !important; color: #fff !important;}}" for i, p in enumerate(PAGES))
    st.markdown(f"<style>{nav_css}</style>", unsafe_allow_html=True)

    def nav(p: str, label: str) -> None:
        st.button(label, key=f"nav_{PAGES.index(p)}", icon=PAGE_ICONS[p], on_click=go, args=(p,), width="stretch",
                  type="primary" if p == page else "secondary")

    nav("Home", "Home")
    st.markdown("<div class='nm-navhead'>Procedure</div>", unsafe_allow_html=True)
    for i, p in enumerate(STEPS):
        nav(p, f"{i + 1}. {p}" + ("  ✓" if done.get(p) else ""))
    if st.session_state.get("rid"):
        n_done = sum(bool(done.get(p)) for p in STEPS)
        st.progress(n_done / len(STEPS), text=f"{n_done} of {len(STEPS)} steps done for this recording")
    st.markdown("<div class='nm-navhead'>Learn & help</div>", unsafe_allow_html=True)
    for p in LEARN:
        nav(p, p)
    st.divider()
    if demo.demo_available() and st.button("Load synthetic demo"):
        st.session_state["rid"] = create_demo_run()
        st.rerun()
    if st.session_state.get("rid") and (run_dir(st.session_state["rid"]) / "meta.json").exists():
        recording_options_ui(st.session_state["rid"], read_json(run_dir(st.session_state["rid"]) / "meta.json", {}))
    trash_ui()
    if st.session_state.get("_cleaned"):
        st.caption(f"Removed {st.session_state['_cleaned']} unfinished run folder(s) left by a stopped detection.")
    st.caption(":material/lock: All processing happens on this computer. Footage is not uploaded anywhere.")
    scroll_helper(st.session_state.get("_shown_page") != page, page)
    st.session_state["_shown_page"] = page

rid, meta = current()


def damaged_files_ui() -> None:
    bad = sorted(p for p in st.session_state.get("_damaged", set()) if Path(p).exists())
    if not bad:
        return
    with st.container(border=True):
        st.error("These files are damaged and could not be read, so the app is NOT using what they contained (it "
                 "uses defaults instead): " + ", ".join(f"`{Path(p).parent.name}/{Path(p).name}`" for p in bad)
                 + ". Restore them from a backup, or set the file aside and redo that step.")
        if st.button("Set damaged files aside (renamed to .damaged)"):
            for p in bad:
                Path(p).rename(p + ".damaged")
            st.session_state["_damaged"] = set()
            st.rerun()


def settings_chips(rid_: str, m: dict) -> None:
    """Compact status chips with an expandable details panel, instead of a paragraph on every page."""
    S = load_settings(rid_)
    model = m.get("model_used") or (m.get("settings") or {}).get("model_name") or (
        "scripted" if m.get("synthetic") else "?")
    status = cal_status(rid_, m)
    p = privacy_of(m)
    chips([("Model", model, None),
           ("Camera", {"center": "straight down", "bottom": "angled"}[S.ground_point], None),
           ("Calibration", status["title"], LEVEL_HEX[status["level"]]),
           ("Peds & bikes", "not measured" if blind_spot(m) else "measured", "#c62828" if blind_spot(m) else None),
           ("Road area", "on" if load_roi(rid_) else "whole frame", None),
           ("Videos", "not blurred" if p and not p.get("blur") else ("blurred" if p else "blur unknown"),
            "#c62828" if p and not p.get("blur") else None)])
    with st.expander("Settings details", icon=":material/tune:"):
        _, cd = load_calibration(rid_)
        if cd is None:
            cal = "not calibrated"
        else:
            inp = cd.get("input", {})
            cal = (f"{inp['distance_m']} m line" if inp.get("method") == "line"
                   else f"{cd['width_m']:g} x {cd['length_m']:g} m rectangle")
        st.markdown(f"- Calibration: {cal}. {status['detail']}\n"
                    f"- TTC thresholds (candidate / moderate / serious): {S.ttc_candidate_s:g} / {S.ttc_moderate_s:g}"
                    f" / {S.ttc_serious_s:g} s\n"
                    f"- Ignore vehicles slower than {S.min_vehicle_speed_mps:g} m/s; tracks shorter than "
                    f"{S.min_track_s:g} s\n"
                    f"- Named zones: {len(load_zones(rid_))}; recording length {m.get('duration_s', '?')} s\n"
                    f"- Derived videos: {video_privacy_text(m)}; event pictures in the app and reports: blurred "
                    f"({S.privacy_level})\n"
                    f"- Original video: {'kept on this computer' if has_source(rid_) else 'deleted'}")


def need_run() -> bool:
    damaged_files_ui()
    if rid is None:
        st.info("Start on the **Detect** page: upload a clip. Or click *Load synthetic demo* in the sidebar to try the "
                "whole app without footage.")
        return True
    st.caption(f"Recording: **{display_name(meta, run_dir(rid))}**  (change it in the sidebar)")
    settings_chips(rid, meta)
    if meta.get("synthetic"):
        st.info("Synthetic demo: scripted road users on a drawn road. It checks the maths on a known answer. It says "
                "nothing about accuracy on real traffic.", icon=":material/science:")
    else:
        frames = meta.get("frame_count", 0) / max(1, meta.get("stride", 1))
        per_frame = meta.get("detection_rows", 0) / frames if frames and "detection_rows" in meta else None
        if per_frame is not None and per_frame < 2:
            st.warning(f"Only {per_frame:.1f} road users were detected per frame in this recording, so few conflicts can "
                       "be found. If it is drone footage, run **Detect** again and choose *From above*.")
    return False


def quality_panel(a, summ=None, key: str = "q") -> None:
    """Every limitation of this recording in one actionable list, most serious first."""
    stats = a[4] if a else None
    items = quality_items(cal_status=cal_status(rid, meta), vehicles_only=blind_spot(meta),
                          motion_summary=motion_summary(rid), stats=stats, summary=summ, privacy=privacy_of(meta))
    if not items:
        st.success("No quality problems found for this recording.", icon=":material/verified:")
        return
    icon = {"problem": ":material/error:", "warn": ":material/warning:", "info": ":material/info:"}
    worst = items[0]["level"]
    n_issues = sum(i["level"] != "info" for i in items)
    title = (f"Quality: {items[0]['area']}: {items[0]['message'].split('.')[0]}"
             + (f" (+{len(items) - 1} more)" if len(items) > 1 else "")) if worst != "info" else \
        f"Quality: {len(items)} note(s), no problems"
    with st.expander(title, expanded=False, icon=icon[worst]):
        if n_issues:
            st.caption("These limit what the numbers below mean. Fix the problems first; the rest are notes.")
        for n, it in enumerate(items):
            l, r = st.columns([5, 1.3])
            l.markdown(f"{icon[it['level']]} **{it['area']}.** {it['message']}")
            if it["page"] and it["page"] != st.session_state.get("page"):
                r.button(it["action"] or f"Go to {it['page']}", key=f"{key}_{n}", on_click=go, args=(it["page"],),
                         width="stretch")


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
    damaged_files_ui()
    prewarm_detector()
    if load_prefs().get("tips", True) and not list_runs():
        st.info("**New here?** 1) Upload an intersection clip below and press Run. 2) Calibrate with a rectangle you "
                "know the size of. 3) Look at Results, then review each flag. Or try *Load synthetic demo* in the "
                "sidebar first. Tips can be turned off under Help & contact > Preferences.", icon=":material/lightbulb:")
    unfinished_ui()
    ups = st.file_uploader("Intersection video(s)", type=["mp4", "mov", "m4v"], accept_multiple_files=True,
                           help="Choose several clips to process them one after another, each as its own recording.")
    view = st.radio("How was it filmed?", ["angled", "overhead"], horizontal=True, key="det_view",
                    format_func=lambda v: {"angled": "From the side or a pole (looking at the road at an angle)",
                                           "overhead": "From above (drone looking straight down)"}[v])
    overhead = view == "overhead"
    choices = aerial_choices() if overhead else MODEL_CHOICES
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", list(choices), key=f"model_{view}", format_func=lambda m: f"{m}  ({choices[m]})")
    if overhead and not finds_people(model):
        msg = ("This model finds cars, buses and trucks seen from above, but **not pedestrians or bicycles**: their "
               "conflicts will show as *not measured*.")
        msg += (f" Choose **{VISDRONE_MODEL}** to include them." if VISDRONE_MODEL in choices else
                f" To find them too, train the VisDrone model (tools/train_visdrone_colab.ipynb) and put "
                f"`{VISDRONE_MODEL}` in the app folder.")
        st.warning(msg)
    elif model == VISDRONE_MODEL:
        st.info("Also finds pedestrians, bikes and motorcycles from above. It mistakes some street furniture (lamp "
                "posts, signal boxes) for people, so a pedestrian or bike that never moves is hidden in the video and "
                "left out of the analysis; you can restore any real one on Results. Use *Try on the first 10 s*.")
    imgsz = c2.select_slider("Detection size (px)", [640, 800, 960, 1280], value=960,
                             help="Bigger finds smaller or farther road users but is slower. "
                                  "For overhead models 960 works best: bigger sizes start calling cars 'large vehicles'.")
    devices = ["auto", "cpu"] + [f"intel:{d.lower()}" for d in ("GPU", "NPU") if d in intel_devices()] + ["mps", "cuda:0"]
    prefs = load_prefs()
    device = c3.selectbox("Device", devices, index=devices.index(prefs["device"]) if prefs.get("device") in devices else 0,
                          format_func=lambda d: f"auto ({device_label(pick_device('auto'))})"
                          if d == "auto" else device_label(d),
                          help="auto picks the fastest one found: an NVIDIA GPU, an Apple chip, or an Intel GPU "
                               "(through OpenVINO, about 4x faster than the CPU on an Intel Arc laptop chip).")
    speed = st.radio("Speed", list(SPEED_MODES), horizontal=True, key="det_speed",
                     index=list(SPEED_MODES).index(prefs["speed"]) if prefs.get("speed") in SPEED_MODES else 1,
                     format_func=lambda k: SPEED_MODES[k]["label"],
                     help="How many frames per second are analysed, and whether the tracked video of the whole clip "
                          "is drawn during detection. Event clips are always available: without the tracked video "
                          "they are drawn when you open them.")
    st.caption(SPEED_MODES[speed]["note"])
    analysis_fps, draw_video = SPEED_MODES[speed]["fps"], SPEED_MODES[speed]["video"]
    with st.expander("More settings"):
        a, b, c = st.columns(3)
        high = a.slider("Detection confidence", 0.10, 0.60, 0.25, 0.05,
                        help="Lower finds more, but adds false boxes. Weak detections only extend existing tracks.")
        start = b.slider("Confidence to start a new track", 0.15, 0.70, 0.30, 0.05)
        buffer_s = c.slider("Keep lost tracks (s)", 0.5, 6.0, 3.0, 0.5)
        a, b, c = st.columns(3)
        coast = a.slider("Carry box forward after a miss (s)", 0.0, 2.0, 1.0, 0.25)
        trail = b.slider("Trail length (s)", 0.0, 5.0, 2.0, 0.5)
        stride_auto = c.checkbox("Skip frames as the Speed setting says", True,
                                 help="Off: analyse every frame of the video (slowest; rarely needed).")
        stabilize = st.checkbox("Camera moves (drone, wind, zoom): line every frame up with the first one", overhead,
                                help="Without this, a drifting camera makes parked cars look like they are driving "
                                     "and puts every position after the first seconds in the wrong place. "
                                     "Adds about 50 ms per frame.")
        a, b = st.columns(2)
        blur = a.checkbox("Blur faces and plates in the output video", True,
                          help="If you turn this off, the tracked video, event clips, slow-motion video and review "
                               "packs made from it are NOT blurred, and the app says so wherever they appear.")
        level = b.selectbox("Blur strength", ["standard", "strict"], index=1 if overhead else 0,
                            help="strict pixelates each whole road user. Use it for drone footage: the head and "
                                 "plate regions that 'standard' blurs only make sense for a camera looking from the side.")
        conf_show = st.checkbox("Show confidence on labels", False)
        if not blur:
            st.warning("Blur is off: the videos this run makes will show faces and plates. Do not share them.")

    if not ups:
        return
    files = []                                       # (upload, saved path, metadata)
    for up in ups:
        path = save_upload(up)
        try:
            files.append((up, path, read_metadata(str(path))))
        except ValueError as e:
            st.error(f"{up.name}: {e} If it plays elsewhere, the codec may be unsupported (HEVC from some phones): "
                     "re-export it as H.264 MP4 and upload it again.")
    if not files:
        return
    up, src_tmp, m = files[0]
    stride = auto_stride(m["fps"], analysis_fps) if stride_auto else 1
    dev = pick_device(device)
    if len(files) == 1:
        kpis([("Duration", f"{m['duration_s']} s"), ("FPS", m["fps"]), ("Resolution", f"{m['width']}x{m['height']}"),
              ("Frames", m["frame_count"]), ("Analysed", f"every {stride} frame(s)"), ("Device", device_label(dev))])
    else:
        st.dataframe(pd.DataFrame([{"clip": u.name, "duration (s)": mm["duration_s"], "fps": mm["fps"],
                                    "resolution": f"{mm['width']}x{mm['height']}"} for u, _, mm in files]),
                     hide_index=True, width="stretch")
        st.caption(f"{len(files)} clips, {sum(mm['duration_s'] for _, _, mm in files) / 60:.1f} min in total. They run "
                   f"one after another with the same settings, on {device_label(dev)}; each becomes its own recording.")
    if dev == "cpu" and "GPU" not in intel_devices():
        st.caption("Running on the CPU. On a computer with an NVIDIA, Apple or Intel graphics chip, detection is several "
                   "times faster (Intel needs `pip install openvino`).")

    def settings_for(mm):
        st_ = auto_stride(mm["fps"], analysis_fps) if stride_auto else 1
        return (TrackerSettings(model_name=model, imgsz=imgsz, high_conf=high, start_conf=start,
                                track_buffer_s=buffer_s, stride=st_, device=device, coast_s=coast, trail_s=trail,
                                stabilize=stabilize),
                RenderOptions(blur_privacy=blur, privacy_level=level, show_trails=trail > 0, show_confidence=conf_show,
                              enabled=draw_video))

    a, b = st.columns([1, 3])
    test = a.button("Try on the first 10 s", icon=":material/science:",
                    help="Quick check of the model and settings before the full run. Nothing is saved.")
    run = b.button("Run detection + tracking" if len(files) == 1 else f"Run all {len(files)} clips", type="primary")
    if test:
        quick_test(str(src_tmp), m, *settings_for(m))
    compare_models_ui(str(src_tmp), m, choices, model, imgsz, device, stabilize)
    if not run:
        return
    queued = []
    for u, path, mm in files:
        ts, ro = settings_for(mm)
        queued.append(start_job(u.name, path, mm, ts, ro, view, blur, level))
    st.button("Stop processing", icon=":material/stop_circle:",
              help="Stops detection. Finished 30-second parts are kept, so you can resume it later from this page.")
    done = []
    for k, run_id in enumerate(queued):
        if len(queued) > 1:
            st.subheader(f"Clip {k + 1} of {len(queued)}: {load_job(run_dir(run_id))['name']}", anchor=False)
        job = process_job_ui(run_id)
        if job is not None:
            done.append((run_id, job))
    if len(done) == 1:
        after_run_ui(*done[0])
    elif done:
        st.success(f"Finished {len(done)} of {len(queued)} recordings. Pick one in the sidebar to calibrate it.")
        st.dataframe(pd.DataFrame([{"recording": j["name"], "detections": j["rows"], "minutes taken":
                                    round(j["seconds"] / 60, 1)} for _, j in done]), hide_index=True, width="stretch")
        st.session_state["rid"] = done[0][0]


def compare_models_ui(path: str, m: dict, choices: dict, current: str, imgsz: int, device: str,
                      stabilize: bool) -> None:
    """Run two detection models on the first seconds of this clip and compare what they find (tools/compare_models.py
    on the command line). There is no answer key, so these are clues, not accuracy: look at the two pictures."""
    with st.expander("Compare two models on this clip", icon=":material/compare:"):
        st.caption("Runs both models on the start of the clip with the same settings. More road users is better only "
                   "if they are real: check the two pictures. Nothing is saved.")
        names = list(choices)
        a, b, c = st.columns([2, 2, 1])
        m1 = a.selectbox("Model A", names, index=names.index(current), key="cmp_a",
                         format_func=lambda x: f"{x} ({choices[x]})")
        m2 = b.selectbox("Model B", names, index=(names.index(current) + 1) % len(names), key="cmp_b",
                         format_func=lambda x: f"{x} ({choices[x]})")
        secs = c.number_input("Seconds", 5, 120, 20, 5, key="cmp_s")
        if not st.button("Compare", icon=":material/compare_arrows:", disabled=m1 == m2):
            return
        from tools.compare_models import evaluate

        rows = []
        for mdl in (m1, m2):
            bar = st.progress(0.0, text=f"Running {mdl}...")
            try:
                rows.append(evaluate(mdl, path, float(secs), float(secs) / 2, device=device, stabilize=stabilize,
                                     imgsz=imgsz, blur=True, progress=lambda f: bar.progress(f, text=f"Running {mdl}...")))
            except Exception as e:
                st.error(model_error(mdl, e))
            bar.empty()
        if not rows:
            return
        st.dataframe(pd.DataFrame([{"model": r["model"], "tracks of 1 s or more": r["tracks_1s"],
                                    "moving tracks": r["moving_tracks"], "label switches": r["label_switches"],
                                    "road users per frame": ", ".join(f"{v} {k}" for k, v in r["per_frame"].items()),
                                    "speed": f"{r['ms_per_frame']} ms/frame on {device_label(r['device'])}"}
                                   for r in rows]), hide_index=True, width="stretch")
        st.caption("*Moving tracks* travel more than 3 car lengths: conflicts come from those. *Label switches* count "
                   "car/truck flicker: fewer is steadier.")
        cols = st.columns(len(rows))
        for col, r in zip(cols, rows):
            if r["snapshot"] is not None:
                col.image(rgb(r["snapshot"]), width="stretch", caption=f"{r['model']} at {secs / 2:.0f} s (blurred)")
            if r["warning"]:
                col.warning(r["warning"])


# Speeds measured on 2 min of a 1280x720 intersection clip, Intel Core Ultra 5 226V on mains power (Arc GPU).
SPEED_MODES = {
    "full": {"label": "Full (12 fps + tracked video)", "fps": 12.0, "video": True,
             "note": "12 frames analysed per second, and the tracked video of the whole clip is drawn while detecting. "
                     "About 4.5x faster than real time on an Intel Arc laptop GPU."},
    "fast": {"label": "Fast (12 fps, video later)", "fps": 12.0, "video": False,
             "note": "The same 12 frames per second, so exactly the same tracks and events, but the whole tracked "
                     "video is not drawn (event clips are drawn when opened). About 6x faster than real time."},
    "fastest": {"label": "Fastest (6 fps, video later)", "fps": 6.0, "video": False,
                "note": "6 frames per second: about 15x faster than real time, but on our test clip it kept 5 "
                        "pedestrian tracks instead of 13 and 55 cars instead of 80. Good for a first pass over long "
                        "footage; use Fast for anything you report."},
}


def save_upload(up) -> Path:
    """Copy an uploaded clip to disk once. Streamlit re-runs the page on every click; re-writing a 2 GB upload each
    time used to cost seconds per click."""
    tmp = RUNS / ".incoming"
    tmp.mkdir(exist_ok=True)
    path = tmp / f"upload_{abs(hash((up.name, up.size))) % 10 ** 10}.mp4"
    if not path.exists() or path.stat().st_size != up.size:
        with open(path, "wb") as f:
            shutil.copyfileobj(up, f, length=16 * 1024 * 1024)
        up.seek(0)
    return path


@st.cache_resource
def prewarm_detector():
    """Import the detection libraries in the background while the user picks a clip and settings, once per app
    start: PyTorch and Ultralytics take 5-10 s to import, which used to be paid after pressing Run."""
    import threading

    def load():
        try:
            import ultralytics  # noqa: F401
            from ultralytics.trackers import byte_tracker  # noqa: F401
        except Exception:
            pass

    t = threading.Thread(target=load, daemon=True)
    t.start()
    return t


def start_job(name: str, upload_path: Path, m: dict, ts: TrackerSettings, ro: RenderOptions, view: str,
              blur: bool, level: str) -> str:
    """A new run folder with the video and a checkpointed job, ready for process_job_ui."""
    new_id = time.strftime("%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    d = run_dir(new_id)
    d.mkdir(parents=True)
    shutil.copy(str(upload_path), str(d / "source.mp4"))   # keep the upload until the run finishes
    new_job(d, name=name, view=view, tracker=ts.to_dict(), render=asdict(ro),
            meta=dict(m, privacy={"blur": bool(blur), "level": level if blur else "off"}))
    return new_id


def process_job_ui(run_id: str):
    """Run (or resume) one job with a progress bar and a live preview. Returns the finished job, or None."""
    d = run_dir(run_id)
    job = load_job(d)
    resumed = job["next_frame"] > 0
    prog = st.progress(job["next_frame"] / max(job["meta"]["frame_count"], 1),
                       text="Loading model..." if not resumed else "Loading model to resume...")
    preview = st.empty()

    def progress(i: int, tot: int, left: float) -> None:
        text = f"Frame {i}/{tot}"
        if left >= 0:
            text += f" · about {left / 60:.0f} min left" if left >= 90 else f" · about {left:.0f} s left"
        prog.progress(min(i / max(tot, 1), 1.0), text=text)

    try:
        job = run_job(d, progress, lambda img: preview.image(rgb(img), channels="RGB", width="stretch"))
    except Exception as e:
        st.error(f"Detection stopped ({type(e).__name__}: {e}). The finished parts are kept: resume it at the top of "
                 "this page. What to try: choose Device *cpu*, a smaller detection size, or re-export the video as "
                 "H.264 MP4.")
        return None
    prog.progress(1.0, text="Done")
    preview.empty()
    finish_run(run_id, job)
    return job


def finish_run(run_id: str, job: dict) -> None:
    """settings.json and meta.json for a finished detection (meta.json last: without it a run is unfinished)."""
    d = run_dir(run_id)
    S = Settings()
    if job["view"] == "overhead":
        S.ground_point = "center"
        S.privacy_level = "strict"
    write_json(d / "settings.json", S.to_dict())
    m = dict(job["meta"])
    m.update(name=job["name"], created=time.time(), synthetic=False, stride=job["tracker"]["stride"],
             settings=job["tracker"], codec=job.get("codec") or "", seconds=job["seconds"], detection_rows=job["rows"],
             model_used=job["model_used"], device_used=job["device_used"], view=job["view"],
             resumed=job.get("resumes", 0))
    write_json(d / "meta.json", m)
    for p in (RUNS / ".incoming").glob("upload_*.mp4"):
        if p.stat().st_size == (d / "source.mp4").stat().st_size:
            p.unlink(missing_ok=True)


def unfinished_ui() -> None:
    """Detections that were stopped or crashed, with their progress: resume or discard each."""
    for msg in st.session_state.pop("_resumed_msgs", []):
        st.success(msg)
    jobs = [(p.name, load_job(p)) for p in sorted(RUNS.iterdir()) if p.is_dir() and not (p / "meta.json").exists()]
    jobs = [(r, j) for r, j in jobs if j]
    if not jobs:
        return
    with st.container(border=True):
        st.markdown(f":material/pause_circle: **{len(jobs)} unfinished detection(s)**. Finished parts were kept.")
        for r, j in jobs:
            pct = j["next_frame"] / max(j["meta"]["frame_count"], 1)
            a, b, c = st.columns([4, 1, 1])
            a.markdown(f"**{j['name']}** · {pct:.0%} done · {j['meta']['duration_s'] * pct / 60:.1f} of "
                       f"{j['meta']['duration_s'] / 60:.1f} min")
            if b.button("Resume", key=f"resume_{r}", type="primary", width="stretch"):
                st.session_state["_resume"] = [r]
            if c.button("Discard", key=f"discard_{r}", width="stretch"):
                shutil.rmtree(run_dir(r), ignore_errors=True)
                st.rerun()
        if len(jobs) > 1 and st.button("Resume all", icon=":material/play_arrow:"):
            st.session_state["_resume"] = [r for r, _ in jobs]
        todo = st.session_state.pop("_resume", None)
        if todo:
            st.button("Stop processing", icon=":material/stop_circle:", key="stop_resume")
            msgs = []
            for r in todo:
                st.write(f"Resuming **{load_job(run_dir(r))['name']}**")
                job = process_job_ui(r)
                if job is None:
                    return                             # the error is shown; the job stays listed for another try
                msgs.append(f"{job['name']}: finished, {job['rows']} detections. It is now selected in the sidebar.")
                st.session_state["rid"] = r
            st.session_state["_resumed_msgs"] = msgs
            st.rerun()                                 # redraw without the finished ones in the list


def after_run_ui(run_id: str, job: dict) -> None:
    """Messages after one finished detection: what was found, and what to check."""
    d = run_dir(run_id)
    m = read_json(d / "meta.json", {})
    st.session_state["rid"] = run_id
    frames = job.get("stats", {}).get("frames", 0)
    if frames == 0:
        st.error("No frames could be read from this video. Try re-exporting it as H.264 MP4 (for example with "
                 "HandBrake or your phone's share/export option) and upload it again.")
    elif job["rows"] == 0:
        st.error(f"The model ran on {frames} frames but found **no road users**, so there is nothing to analyse. "
                 "Things to try: check *How was it filmed?* matches the footage (drone footage needs *From above*), "
                 "lower *Detection confidence* under More settings, or raise *Detection size*.")
    else:
        st.success(f"Processed {frames} frames in {job['seconds']:.0f} s on {device_label(job['device_used'])}. "
                   f"{job['rows']} detections saved." + (f" (Resumed {job['resumes']} time(s).)" if job.get("resumes") else ""))
    if job.get("warning"):
        st.warning(job["warning"])
    if blind_spot(m):
        st.warning("Vehicles only: pedestrian and bicycle conflicts are **not measured** in this recording.")
    ms = job.get("motion_summary")
    if ms:
        if ms["max_shift_px"] < 3:
            st.caption("The camera stayed still.")
        else:
            st.info(f"The camera moved up to {ms['max_shift_px']:.0f} px during the clip. Every frame was lined up with "
                    "the first one, so you calibrate on the first frame.")
        if ms["lost_frames"]:
            st.warning(f"{ms['lost_frames']} frames could not be lined up (too little texture or too much change). "
                       "Positions on those frames are less reliable.")
    if job.get("codec") and job["codec"] != "h264":
        st.warning("ffmpeg was not found, so the video uses a format some browsers cannot play. "
                   "Run `pip install imageio-ffmpeg`, or open the file in VLC.")
    if (d / "annotated.mp4").exists():
        st.video(str(d / "annotated.mp4"))
    else:
        st.caption("Fast mode: no tracked video of the whole clip was drawn. Event clips are drawn when you open them, "
                   "and Results > Slow-motion video can draw the whole tracked video later.")
    st.download_button("Download detections.csv", (d / "detections.csv").read_bytes(), "detections.csv", "text/csv")
    st.write("Next, calibrate so distances and speeds are in metres.")
    next_button("Calibrate", "Calibrate")


# ================================================================== CALIBRATE
def draw_points(img: np.ndarray, pts, scale: float, color, closed: bool) -> None:
    for i, (x, y) in enumerate(pts):
        p = (int(x * scale), int(y * scale))
        cv2.circle(img, p, 7, color, -1)
        cv2.circle(img, p, 7, (0, 0, 0), 1)
        cv2.putText(img, str(i + 1), (p[0] + 9, p[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
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
    """Fixed cameras: copy the calibration, road area and zones from another recording of the same spot."""
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
                   "area and zones, and REPLACES this recording's own road area and zones (also when the other "
                   "recording has none). Re-do the accuracy check: a check belongs to the clip it was measured on.")
        src = st.selectbox("Copy from", [r for r, _ in same], format_func=dict(same).get, key=f"reuse_{rid}")
        if st.button("Copy calibration, road area and zones"):
            cd = read_json(run_dir(src) / "calibration.json")
            cd.pop("checks", None)
            write_json(run_dir(rid) / "calibration.json", cd)
            for f in ("roi.json", "zones.json"):
                if (run_dir(src) / f).exists():
                    shutil.copy(run_dir(src) / f, run_dir(rid) / f)
                else:
                    (run_dir(rid) / f).unlink(missing_ok=True)         # do not leave this recording's old ones behind
            for k in [k for k in st.session_state if rid in k and k.startswith(("pts_", "roi_", "chk_", "zone_draft_"))]:
                del st.session_state[k]
            st.rerun()


def page_calibrate():
    page_header("Calibrate", "Calibrate", "Turn pixels into metres, check the result against a second known distance, "
                "and optionally outline the road area and name zones such as crosswalks.")
    if need_run():
        return
    if not has_source(rid):
        st.info("The original video of this recording was deleted for privacy, so the calibration cannot be redone. "
                "The saved calibration is still used for the analysis.")
        return
    status = cal_status(rid, meta)
    reuse_calibration_ui()
    S = load_settings(rid)
    _, saved = load_calibration(rid)
    saved_in = (saved or {}).get("input", {})
    moving = camera_moved(rid)

    a, b, c = st.columns([2, 2, 1])
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
    method = b.radio("Calibrate with", methods,
                     index=methods.index(saved_in.get("method", "rect")) if saved_in.get("method", "rect") in methods else 0,
                     format_func=lambda m: {"rect": "A rectangle of known size (4 clicks)",
                                            "line": "One known distance (2 clicks)"}[m])
    big = c.toggle("Large canvas", key=f"bigcanvas_{rid}", help="Use the full page width for the picture.")
    n_pts = 4 if method == "rect" else 2

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
    a, b = st.columns([4, 2])
    target = a.radio("Clicks on the image add", targets, horizontal=True, key=f"target_{rid}",
                     format_func=lambda t: {"cal": "Calibration points (yellow)",
                                            "chk": "Accuracy-check points (pink)",
                                            "roi": "Road area outline (green)",
                                            "zone": "Named zone outline (white)"}[t])
    move = None
    if target == "cal" and pts:
        opts = ["add"] + [f"move {i + 1}" for i in range(len(pts))]
        mode = b.selectbox("Click mode", opts, key=f"cmode_{rid}_{method}",
                           format_func=lambda o: "Add the next point" if o == "add" else f"Move point {o.split()[1]}",
                           help="Pick a point, then click where it should be. Easier than starting over.")
        move = None if mode == "add" else int(mode.split()[1]) - 1

    canvas, side = (st.container(), st.container()) if big else st.columns([3, 2])
    shown = cv2.resize(frame, (DISPLAY_W, int(H * scale)))
    draw_points(shown, pts, scale, (0, 220, 255), closed=len(pts) == 4)
    draw_points(shown, chk, scale, (200, 80, 255), closed=False)
    draw_points(shown, roi_pts, scale, (80, 220, 80), closed=len(roi_pts) >= 3)
    for i, z in enumerate(load_zones(rid)):
        col = ZONE_COLORS[i % len(ZONE_COLORS)]
        q = (np.asarray(z["points"]) * scale).astype(np.int32)
        cv2.polylines(shown, [q], True, col, 2, cv2.LINE_AA)
        cc = q.mean(axis=0).astype(int)
        cv2.putText(shown, z["name"], (int(cc[0]) - 4 * len(z["name"]), int(cc[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    col, 2, cv2.LINE_AA)
    draw_points(shown, zone_draft, scale, (255, 255, 255), closed=len(zone_draft) >= 3)
    with canvas:
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
        elif move is not None and move < len(pts):
            pts[move] = p
        elif len(pts) < n_pts:
            pts.append(p)
        st.rerun()

    with side:
        if method == "rect":
            a, b = st.columns(2)
            w_m = a.number_input("Side 1->2 (m)", 0.5, 500.0, float(saved_in.get("width_m", 8.0)), 0.1)
            l_m = b.number_input("Side 2->3 (m)", 0.5, 500.0, float(saved_in.get("length_m", 20.0)), 0.1)
            st.caption("Click the **four corners of a rectangle on the road** whose side lengths you know (a crosswalk, "
                       "a lane width and a stop-bar gap, a parking space), in order around the edge.")
        else:
            w_m = st.number_input("Distance from point 1 to point 2 (m)", 0.5, 500.0,
                                  float(saved_in.get("distance_m", 10.0)), 0.1)
            l_m = w_m
            st.caption("Click **two points on the road** a known distance apart (the ends of a long lane line, a "
                       "crosswalk edge). Only valid when the camera looks straight down.")
        st.caption("Don't know the size? Measure it on Google Maps: right-click, *Measure distance*.")
        a, b = st.columns(2)
        if a.button("Undo last point", disabled=not pts, width="stretch"):
            pts.pop()
            st.rerun()
        if b.button("Clear points", disabled=not pts, width="stretch"):
            st.session_state[key] = []
            st.rerun()
        if len(pts) < n_pts:
            st.info(f"Click {n_pts - len(pts)} more point(s) on the image.")
        elif pts != saved_in.get("points"):
            show_calibration_hints(method, pts, w_m, l_m, (W, H))
        if len(pts) == n_pts and st.button("Save calibration", type="primary", width="stretch"):
            try:
                if method == "rect":
                    cal = Calibration.from_points(pts, w_m, l_m, (W, H))
                    inp = {"method": "rect", "points": pts, "width_m": w_m, "length_m": l_m}
                else:
                    cal = Calibration.from_two_points(pts[0], pts[1], w_m, (W, H))
                    inp = {"method": "line", "points": pts, "distance_m": w_m}
            except CalibrationError as e:
                st.error(f"{e} What to try: move a point (Click mode) or clear and click the corners again in order.")
            else:
                d = cal.to_dict()
                d["input"] = inp
                write_json(run_dir(rid) / "calibration.json", d)       # a new calibration needs a new accuracy check
                st.session_state[chk_key] = []
                st.session_state[f"just_saved_{rid}"] = True
                st.rerun()
        with st.expander(f"Type the points ({len(pts)} of {n_pts})"):
            edited = point_editor(pts, n_pts, f"ed_{rid}_{method}")
            if edited != pts:
                st.session_state[key] = pts = edited
        # live bird's-eye preview of the points placed so far, before saving
        if len(pts) == n_pts and pts != saved_in.get("points"):
            try:
                draft = (Calibration.from_points(pts, w_m, l_m, (W, H)) if method == "rect"
                         else Calibration.from_two_points(pts[0], pts[1], w_m, (W, H)))
                st.image(rgb(draft.preview(frame, max_px=500)), width="stretch",
                         caption="Preview (not saved): lane edges should look parallel.")
            except CalibrationError:
                pass

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
    status = cal_status(rid, meta)
    a = get_analysis(rid)
    if st.session_state.pop(f"just_saved_{rid}", False):
        st.toast("Calibration saved")
    with st.container(border=True):
        st.markdown(f"**Calibration status:** <span style='color:{LEVEL_HEX[status['level']]};font-weight:600'>"
                    f"{status['title']}</span>. {status['detail']}", unsafe_allow_html=True)
        if a is not None:
            n_tracks, n_events = a[4]["analysed_tracks"], len(a[5])
            msg = f"With it, {n_tracks} road users are tracked and {n_events} events are flagged."
            if n_tracks == 0:
                st.error(msg + " No road users could be placed on the road: check the camera view, the points and the "
                         "road area.")
            else:
                st.write(msg + (" Severities stay **provisional** until the calibration is checked."
                                if status["provisional"] and not meta.get("synthetic") else ""))
            next_button("Results", "See the results")
    st.subheader("Check it")
    for h in cal.quality_hints(check_shape=d.get("input", {}).get("method", "rect") == "rect"):
        st.warning(h)
    if a is not None:
        size = cached_size_check(*analysis_keys(rid))
        if size and not size["warning"]:
            st.caption(f"Sanity check: cars measure {size['median_car_length_m']} m long on average with this "
                       f"calibration ({size['cars_sampled']} boxes), in line with real cars (about 4-5 m).")
    a1, b1 = st.columns(2)
    a1.image(rgb(cal.preview(frame)), caption="Bird's-eye view of the road. Straight lines should look straight and "
             "lane edges parallel.", width="stretch")
    with b1:
        accuracy_check_ui(cal, d, chk, chk_key)


def accuracy_check_ui(cal, d: dict, chk: list, chk_key: str) -> None:
    """One or more known distances measured with the calibration. The pink pair being placed is the 'current' check;
    'Keep it and add another' saves it so a new pair can be clicked. With two or more, the scale can be fitted to all
    of them (least squares) and the leave-one-out error says how accurate distances really are."""
    st.write("**Accuracy check.** Choose *Accuracy-check points* above, click the two ends of a second known "
             "distance (for example a lane width), and type its real length. More distances give a better check.")
    kept = [c for c in d.get("checks", []) if not c.get("current")]
    known = st.number_input("Known distance (m)", 0.5, 500.0, 3.5, 0.1, key=f"known_{rid}")
    if len(chk) == 2:
        try:
            est = cal.distance_between(chk[0], chk[1])
        except CalibrationError as e:
            st.error(str(e))
            return
        err = 100 * (est - known) / known
        st.metric("Measured", f"{est:.2f} m", f"{err:+.1f}% vs {known} m", delta_color="off")
        if abs(err) > 10:
            st.warning("More than 10% off. Re-click the calibration points more carefully, or use a larger shape.")
        cur = {"known_m": known, "measured_m": round(est, 3), "error_pct": round(err, 1), "p1": chk[0], "p2": chk[1],
               "current": True}
        checks = [cur] + kept
        if d.get("checks") != checks:
            d["checks"] = checks
            write_json(run_dir(rid) / "calibration.json", d)
            st.rerun()
        if st.button("Keep it and add another distance", icon=":material/add:"):
            cur.pop("current")
            write_json(run_dir(rid) / "calibration.json", d)
            st.session_state[chk_key] = []
            st.rerun()
    else:
        st.caption(f"{len(chk)} of 2 check points placed.")
    all_checks = [c for c in d.get("checks", []) if "p1" in c]
    if len(d.get("checks", [])) > 1:
        st.dataframe(pd.DataFrame([{"known (m)": c["known_m"], "measured (m)": c["measured_m"],
                                    "error": f"{c['error_pct']:+.1f}%", "": "being placed" if c.get("current") else ""}
                                   for c in d["checks"]]), hide_index=True, width="stretch")
        if st.button("Remove the kept distances"):
            d["checks"] = [c for c in d["checks"] if c.get("current")]
            write_json(run_dir(rid) / "calibration.json", d)
            st.rerun()
    if len(all_checks) >= 2:
        fit = fit_scale(cal, all_checks)
        loo = fit["rms_leave_one_out_pct"]
        st.caption(f"Fitting the scale to all {fit['n']} distances would multiply distances by {fit['scale']:.3f}: typical "
                   f"error {fit['rms_before_pct']:.1f}% now, {fit['rms_after_pct']:.1f}% after"
                   + (f"; tested on each distance left out of the fit, {loo:.1f}% (the honest estimate)." if loo is not None
                      else ". With 3 or more distances the app can also test the fit on each one left out."))
        if abs(fit["scale"] - 1) > 0.005 and st.button("Apply the fitted scale", icon=":material/straighten:"):
            new = cal.rescaled(fit["scale"])
            nd = new.to_dict()
            nd["input"] = d.get("input", {})
            nd["scale_fit"] = {k: fit[k] for k in ("scale", "n", "rms_before_pct", "rms_after_pct",
                                                    "rms_leave_one_out_pct")}
            nd["checks"] = []
            for c in d["checks"]:
                if "p1" in c:
                    m_ = new.distance_between(c["p1"], c["p2"])
                    nd["checks"].append(dict(c, measured_m=round(m_, 3),
                                             error_pct=round(100 * (m_ - c["known_m"]) / c["known_m"], 1)))
            write_json(run_dir(rid) / "calibration.json", nd)
            st.rerun()
    if d.get("scale_fit"):
        sf = d["scale_fit"]
        st.caption(f"Scale fitted to {sf['n']} known distances (x{sf['scale']:.3f}).")


# ================================================================== RESULTS
def thresholds_ui(S: Settings) -> None:
    with st.container(border=True):
        st.subheader("Thresholds", anchor=False)
        st.caption("Defaults are starting points for screening, not safety standards. Validate them on your own clips.")
        now = matching_preset(S)
        a, b = st.columns([3, 1])
        pick = a.selectbox("Start from a preset for this kind of site", list(PRESETS),
                           index=list(PRESETS).index(now) if now else 0, key=f"preset_pick_{rid}",
                           help="Changes only the conflict thresholds. Check the result on your labelled events "
                                "(Report > Tuning) before relying on it.")
        st.caption(PRESET_NOTES[pick] + ("" if now else " Your current thresholds were changed by hand."))
        b.write("")
        if b.button("Apply preset", disabled=pick == now, width="stretch"):
            write_json(run_dir(rid) / "settings.json", apply_preset(S, pick).to_dict())
            st.rerun()
        a, b, c = st.columns(3)
        S.ttc_candidate_s = a.slider("Candidate if TTC under (s)", 1.0, 6.0, S.ttc_candidate_s, 0.5)
        S.ttc_moderate_s = b.slider("Moderate if TTC under (s)", 0.5, 4.0, S.ttc_moderate_s, 0.5)
        S.ttc_serious_s = c.slider("Serious if TTC under (s)", 0.2, 2.0, S.ttc_serious_s, 0.1)
        a, b = st.columns(2)
        S.min_vehicle_speed_mps = a.slider("Ignore vehicles slower than (m/s)", 0.0, 5.0, S.min_vehicle_speed_mps, 0.5)
        S.min_track_s = b.slider("Ignore tracks shorter than (s)", 0.5, 4.0, S.min_track_s, 0.5)
        st.markdown("**Extra indicators** (off by default: turn on, then check them against your labelled events)")
        a, b = st.columns(2)
        S.pet_flagging = a.toggle("Flag crossing paths by PET", S.pet_flagging,
                                  help="Post-encroachment time: how soon the second road user reached a crossing point "
                                       "after the first one left it. Catches near misses where the two were never on a "
                                       "collision course at the same moment, which TTC cannot see.")
        if S.pet_flagging:
            S.pet_s = list(a.slider("PET candidate / serious (s)", 0.5, 5.0, (S.pet_s[2], S.pet_s[0]), 0.5))[::-1]
            S.pet_s = [S.pet_s[0], round((S.pet_s[0] + S.pet_s[1]) / 2, 2), S.pet_s[1]]
        S.drac_flagging = b.toggle("Flag by required braking (DRAC)", S.drac_flagging,
                                   help="Deceleration rate to avoid the crash: the braking either road user would have "
                                        "needed. Always shown on each event; this also lets it raise the severity. "
                                        "3.35 m/s² is the usual line (comfortable braking).")
        if S.drac_flagging:
            S.drac_mps2 = list(b.slider("DRAC candidate / serious (m/s²)", 1.0, 9.0,
                                        (S.drac_mps2[0], S.drac_mps2[2]), 0.25))
            S.drac_mps2 = [S.drac_mps2[0], round((S.drac_mps2[0] + S.drac_mps2[1]) / 2, 2), S.drac_mps2[1]]
        ok = S.ttc_serious_s < S.ttc_moderate_s < S.ttc_candidate_s
        if not ok:
            st.error("TTC thresholds must go serious < moderate < candidate.")
        a, b = st.columns(2)
        if a.button("Save thresholds", disabled=not ok):
            write_json(run_dir(rid) / "settings.json", S.to_dict())
            st.rerun()
        if b.button("Reset to defaults"):
            D = Settings()
            D.ground_point, D.privacy_level, D.keep_tracks = S.ground_point, S.privacy_level, S.keep_tracks
            write_json(run_dir(rid) / "settings.json", D.to_dict())
            st.rerun()


def event_name(e: dict) -> str:
    lab = LABEL_TEXT.get(e.get("label"), "not reviewed")
    sev = e["severity"] + (" (provisional)" if e.get("provisional") else "")
    return f"{e['id']}  ·  {sev}  ·  {e['kind_label']}  ·  t = {e['t_peak']} s  ·  {lab}"


def open_in_review(key: str) -> None:
    """Button callback: open one event in Event Review by its stable key (filters there are cleared so it is found)."""
    for k in [k for k in st.session_state if k.startswith(("f_kind_", "f_sev_", "f_stat_", "f_zone_"))]:
        del st.session_state[k]
    go("Event Review", **{f"ev_target_{rid}": key})


def selected_event(events: list):
    key = st.session_state.get(f"sel_{rid}")
    return next((e for e in events if e["key"] == key), None)


def overview_tab(cal, S, tracks, events) -> None:
    """Map, table and the selected event, all connected: clicking a row or a map marker selects that event."""
    if not events:
        st.success("No events flagged with these thresholds. Try loosening them, or check the calibration.")
    l, r = st.columns([1, 1])
    with r:
        if events:
            df = pd.DataFrame(events_table(events))
            df.insert(1, "movements", [e["movements"] for e in events])
            df["severity"] = [e["severity"] + ("*" if e.get("provisional") else "") for e in events]
            df["label"] = df["label"].map(lambda v: LABEL_TEXT.get(v, ""))
            drop = ["why"] + ([] if load_zones(rid) else ["zone"])
            sel = st.dataframe(df.drop(columns=drop), hide_index=True, width="stretch", on_select="rerun",
                               selection_mode="single-row", key=f"evtable_{rid}", height=320)
            rows = list(sel.selection.rows) if sel and sel.selection else []
            if rows != st.session_state.get(f"_tblsel_{rid}"):
                st.session_state[f"_tblsel_{rid}"] = rows
                if rows:
                    st.session_state[f"sel_{rid}"] = events[rows[0]]["key"]
            if any(e.get("provisional") for e in events):
                st.caption("\\* provisional: the calibration is not checked or looks wrong, so distances, speeds and "
                           "severities may be off.")
    ev = selected_event(events)
    with l:
        heat = st.toggle("Heat layer", True)
        markers = {}
        img = topdown_map(tracks, events, cal, heat=heat, zones=zone_overlays(cal, rid),
                          highlight=ev["id"] if ev else None, markers_out=markers)
        from streamlit_image_coordinates import streamlit_image_coordinates
        click = streamlit_image_coordinates(rgb(img), key=f"map_{rid}", width="stretch")
        st.caption("Top-down map. Lines are paths (green pedestrians, purple bikes, blue vehicles); dots are events. "
                   "Click a dot or a table row to select it.")
        if click and click.get("unix_time") != st.session_state.get(f"_mapclick_{rid}"):
            st.session_state[f"_mapclick_{rid}"] = click.get("unix_time")
            s = img.shape[1] / click["width"]
            x, y = click["x"] * s, click["y"] * s
            near = min(markers.items(), key=lambda kv: (kv[1][0] - x) ** 2 + (kv[1][1] - y) ** 2, default=None)
            if near and (near[1][0] - x) ** 2 + (near[1][1] - y) ** 2 <= (25 * s) ** 2:
                st.session_state[f"sel_{rid}"] = next(e["key"] for e in events if e["id"] == near[0])
                st.rerun()
    if ev is not None:
        with st.container(border=True):
            st.markdown(f"{badge(ev['severity'], ev.get('provisional'))} &nbsp; **{ev['id']} · {ev['kind_label']}** · "
                        f"t = {ev['t_peak']} s · {html_escape(ev['movements'])}", unsafe_allow_html=True)
            st.write(explain_event(ev))
            st.button("Open it in Event Review", type="primary", icon=":material/arrow_forward:",
                      on_click=open_in_review, args=(ev["key"],))
    elif events:
        st.caption("Select an event in the table or on the map to see it here.")


def rate(v):
    return "n/a" if v is None else str(v)


def directions_ui() -> tuple:
    """North on screen and signal phases, saved with the recording details."""
    info = recording_info(rid)
    with st.expander("Directions and signal phases", icon=":material/explore:"):
        a, b = st.columns([1, 2])
        opts = {0: "up", 90: "right", 180: "down", 270: "left"}
        cur = int(info.get("north_deg", 0) or 0)
        north = a.selectbox("Where is north on the screen?", list(opts), index=list(opts).index(cur) if cur in opts else 0,
                            format_func=lambda d: opts[d], help="Used to name approaches (northbound, ...).")
        phases_txt = b.text_area("Signal phases (optional)", info.get("signal_phases", ""), height=110,
                                 placeholder="0-28 NS green\n28-31 NS yellow\n31-60 EW green",
                                 help="One line per phase: start-end (seconds into the video) and a name. Events are "
                                      "tagged with the phase they happened in. Type them from the video; NearMiss does "
                                      "not detect signals.")
        _, problems = parse_phases(phases_txt)
        for p in problems:
            st.warning(p)
        if north != cur or phases_txt != info.get("signal_phases", ""):
            if st.button("Save directions and phases"):
                info.update(north_deg=north, signal_phases=phases_txt)
                write_json(run_dir(rid) / "recording_info.json", info)
                st.rerun()
    return info


def traffic_tab(cal, S, tracks, stats, events, summ) -> None:
    info = directions_ui()
    st.subheader("Exposure and rates", anchor=False)
    obs = stats.get("observation_s") or meta.get("duration_s") or 0
    users = stats["road_users"]
    confirmed = summ["tp"]
    hours = obs / 3600.0 if obs else 0
    kpis([("Observed", f"{obs / 60:.1f} min"), ("Vehicles", users.get("veh", 0)),
          ("Pedestrians", users.get("ped", 0) if measured(meta, "ped") else "Not measured"),
          ("Bicycles", users.get("bike", 0) if measured(meta, "bike") else "Not measured"),
          ("Flagged per hour", rate(stats["events_per_hour"])),
          ("Confirmed per hour", f"{confirmed / hours:.1f}" if hours and summ["reviewed"] else "not reviewed"),
          ("Flagged per 100 vehicles", rate(stats["events_per_100_vehicles"]))])
    st.caption(f"Rates divide by the whole observation period ({obs:.0f} s of video), quiet stretches included. "
               "*Flagged* is what the app found; *confirmed* is what you labelled a real near miss.")
    per_kind = stats["events_per_1000_pairs_by_kind"]
    rows = []
    for k in per_kind:
        ok = measured(meta, k)
        n_conf = sum(1 for e in events if e["kind"] == k and e.get("label") == "tp")
        rows.append({"type": KIND_LABEL[k], "pairs on the road together": stats["pairs_together"][k] if ok else "Not measured",
                     "flagged": sum(e["kind"] == k for e in events) if ok else "Not measured",
                     "confirmed": n_conf if ok else "Not measured",
                     "flagged per 1,000 pairs": rate(per_kind[k]) if ok else "Not measured"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Movements", anchor=False)
    st.caption("Each road user's approach (from its heading when it starts moving, north = "
               f"{ {0: 'up', 90: 'right', 180: 'down', 270: 'left'}.get(int(info.get('north_deg', 0) or 0), 'up')} on the "
               "screen) and, for vehicles, its turn. A busy movement has more events simply because more road users "
               "use it: compare events per 100 road users.")
    mc = movement_counts(tracks, events, float(info.get("north_deg", 0) or 0))
    if mc:
        st.dataframe(pd.DataFrame(mc), hide_index=True, width="stretch")
    if events:
        pairs = pd.Series([e["movements"] for e in events]).value_counts().rename_axis("movement pair").reset_index(
            name="events")
        st.dataframe(pairs, hide_index=True, width="stretch")
    phases, _ = parse_phases(info.get("signal_phases", ""))
    if phases:
        st.subheader("Signal phases", anchor=False)
        st.dataframe(pd.DataFrame([{"phase": n, "from (s)": t0, "to (s)": t1, "length (s)": round(t1 - t0, 1),
                                    "events": sum(1 for e in events if e.get("phase") == n)}
                                   for t0, t1, n in phases]), hide_index=True, width="stretch")
        st.caption(f"{sum(1 for e in events if not e.get('phase'))} event(s) fall outside the phases you typed.")
    st.subheader("Zones", anchor=False)
    zones_ui(cal, tracks, events)
    st.subheader("Speeds", anchor=False)
    speed_ui(S, tracks)


def zones_ui(cal, tracks, events) -> None:
    zones = load_zones(rid)
    if not zones:
        st.caption("No named zones. On **Calibrate**, choose *Named zone outline* to outline a crosswalk or bike lane.")
        return
    rows = zone_summary(events, tracks, zones, cal)
    df = pd.DataFrame(rows).rename(columns={"events_per_100_users": "events per 100 road users"})
    st.dataframe(df.drop(columns=["valid"]), hide_index=True, width="stretch")
    if not all(r["valid"] for r in rows):
        st.warning("Some zones reach above the horizon of the calibration and cannot be measured: "
                   + ", ".join(r["zone"] for r in rows if not r["valid"]) + ". Redraw them on the road surface.")
    st.caption(f"{sum(1 for e in events if not e.get('zone'))} event(s) are outside every zone.")


def speed_ui(S: Settings, tracks) -> None:
    info = recording_info(rid)
    limit = st.number_input("Posted speed limit (mph, optional)", 0, 80, int(info.get("posted_limit_mph", 0) or 0), 5,
                            help="Leave at 0 if you do not know it.")
    if limit != (info.get("posted_limit_mph", 0) or 0):
        info["posted_limit_mph"] = limit
        write_json(run_dir(rid) / "recording_info.json", info)
    sp = speed_summary(tracks, S, limit / MPS_TO_MPH if limit else None)
    if not sp["vehicles"]:
        st.caption("No moving vehicle was tracked long enough to measure its speed.")
        return
    mph = lambda v: f"{v * MPS_TO_MPH:.0f} mph"
    items = [("Vehicles measured", sp["vehicles"]), ("Median speed", mph(sp["median_mps"])),
             ("85th percentile", mph(sp["p85_mps"]))]
    items.append(("Over the limit", f"{sp['share_over_limit']:.0%}") if limit else ("Fastest", mph(sp["max_mps"])))
    kpis(items)
    counts, edges = np.histogram(np.array(sp["speeds_mps"]) * MPS_TO_MPH, bins=np.arange(0, 85, 5))
    keep = slice(0, int(np.flatnonzero(counts).max()) + 2)
    st.bar_chart(pd.DataFrame({"vehicles": counts[keep]}, index=edges[:-1][keep].astype(int)),
                 height=180, x_label="speed (mph, 5 mph bins: 25 means 25-30)", y_label="vehicles")
    st.caption("Each vehicle's speed is the 85th percentile of its own moving samples. Speeds are only as accurate as "
               "the calibration (a 5% scale error is a 5% speed error).")


FILTERS = [("filtered_duplicate_detection", "One road user detected twice",
            "Two boxes on the same road user look like a collision. Pairs closer than half their combined widths "
            "are dropped."),
           ("filtered_ttc_too_brief", "Collision course too brief",
            "A predicted collision that lasted under 0.5 s: one noisy heading, not a real conflict."),
           ("filtered_no_evasive_action", "Passing a stopped road user",
            "Heading toward a parked or waiting road user, but no hard braking and no close pass: the straight-line "
            "prediction was wrong (a curve, another lane).")]


@st.cache_data(show_spinner=False, max_entries=200)
def object_crop(path: str, t: float, box: tuple) -> np.ndarray:
    img = grab_frame(path, t)
    x1, y1, x2, y2 = (int(v) for v in box)
    p = max(40, int(1.5 * max(x2 - x1, y2 - y1)))
    crop = img[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p].copy()
    cv2.rectangle(crop, (min(p, x1), min(p, y1)), (min(p, x1) + x2 - x1, min(p, y1) + y2 - y1), (0, 0, 255), 2)
    return rgb(crop)


def still_objects_ui(S: Settings, det, stats: dict) -> None:
    """Show what the street-furniture filter dropped, so a real waiting pedestrian can be restored."""
    ids = stats.get("static_clutter_ids", [])
    restored = [t for t in S.keep_tracks if t in set(det.track_ids())]
    if not ids and not restored:
        return
    st.write(f"**Never moved: {len(ids)} left out as street furniture**")
    st.caption("From above, the model sometimes calls lamp posts and signal boxes pedestrians. These never moved, so "
               "they were left out. If one is a real person (someone waiting the whole clip), tick it and restore it.")
    if not has_source(rid):
        st.caption("The original video was deleted, so they cannot be shown.")
    cols = st.columns(4)
    pick = []
    for n, tid in enumerate(ids[:16]):
        idx = det.rows_of(tid)
        k = idx[len(idx) // 2]
        with cols[n % 4]:
            if has_source(rid):
                st.image(object_crop(source_path(rid), float(det.t[k]), tuple(float(v) for v in det.box[k])),
                         width="stretch")
            if st.checkbox(f"#{tid} ({det.track_class(tid)}) is real", key=f"keep_{rid}_{tid}"):
                pick.append(tid)
    if len(ids) > 16:
        st.caption(f"... and {len(ids) - 16} more.")
    a, b = st.columns(2)
    if a.button("Restore the ticked ones", disabled=not pick):
        S.keep_tracks = sorted(set(S.keep_tracks) | set(pick))
        write_json(run_dir(rid) / "settings.json", S.to_dict())
        st.rerun()
    if restored and b.button(f"Undo restores ({len(restored)})"):
        S.keep_tracks = []
        write_json(run_dir(rid) / "settings.json", S.to_dict())
        st.rerun()


def quality_tab(S: Settings, det, stats: dict, tracks) -> None:
    st.subheader("Tracking", anchor=False)
    rows = [("Track IDs from the tracker", stats.get("raw_tracks", 0), ""),
            ("Analysed road users", stats["analysed_tracks"], "long enough to measure"),
            ("Too short", stats.get("dropped_short", 0), f"under {S.min_track_s:g} s: mostly flicker or duplicate boxes"),
            ("Never moved (street furniture)", stats.get("dropped_static_clutter", 0),
             "overhead only: a pedestrian/bike that never moved 3 of its own box sizes"),
            ("Outside the road area", stats.get("dropped_outside_roi", 0), "only if a road area is drawn"),
            ("Out of range", stats.get("dropped_out_of_range", 0), "too far from the calibrated area"),
            ("Riders merged", stats.get("dropped_riders", 0), "a person on a bike, counted once"),
            ("Track breaks", stats.get("track_breaks", 0), "a track split by a detection gap"),
            ("Likely ID switches", stats.get("likely_id_switches", 0),
             "a track ends where a new one of the same kind starts within 1 s and 3 m (estimate)"),
            ("Median track length (s)", stats.get("median_track_s", "-"), "")]
    st.dataframe(pd.DataFrame([(a, str(b), c) for a, b, c in rows], columns=["", "count", "meaning"]),
                 hide_index=True, width="stretch")
    still_objects_ui(S, det, stats)
    st.subheader("Camera alignment", anchor=False)
    ms = motion_summary(rid)
    if not ms:
        st.caption("The camera was treated as fixed (no alignment).")
    else:
        rows = [("Frames", ms["frames"], ""),
                ("Matched to the first frame", ms.get("direct_frames", ms["frames"] - ms["lost_frames"]), "most reliable"),
                ("Matched through the previous frame", ms.get("chained_frames", "-"), "small errors can add up"),
                ("Could not be lined up", ms["lost_frames"], "the previous alignment was reused"),
                ("Largest camera movement (px)", ms["max_shift_px"], "under 3 px: the camera did not move"),
                ("Median matched features", ms.get("median_inliers", "-"), "under 40 is weak")]
        st.dataframe(pd.DataFrame([(a, str(b), c) for a, b, c in rows], columns=["", "count", "meaning"]),
                     hide_index=True, width="stretch")
    st.subheader("False-alarm filters", anchor=False)
    st.dataframe(pd.DataFrame([(name, stats.get(k, 0), why) for k, name, why in FILTERS],
                              columns=["filter", "flags removed", "why"]), hide_index=True, width="stretch")
    thresholds_ui(Settings.from_dict(S.to_dict()))
    sensitivity_ui(S)


@st.cache_data(show_spinner="Re-running the conflict check at other thresholds...")
def cached_sensitivity(rid_: str, csv_mtime: float, cal_json: str, settings_json: str, motion_mtime: float,
                       roi_json: str, observation_s: float):
    _, tracks, _, _ = cached_analysis(rid_, csv_mtime, cal_json, settings_json, motion_mtime, roi_json, observation_s)
    return sensitivity(tracks, Settings.from_dict(json.loads(settings_json)))


def sensitivity_ui(S: Settings) -> None:
    with st.container(border=True):
        st.subheader("Sensitivity: how the count depends on the thresholds", anchor=False)
        st.caption("Re-runs the check with one threshold changed at a time, so you can see how much the event count "
                   "depends on it.")
        if not st.toggle("Run it", key=f"sens_{rid}"):
            return
        res = cached_sensitivity(*analysis_keys(rid), json.dumps(load_roi(rid)), float(meta.get("duration_s") or 0))
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


def draw_tracked_video_ui() -> None:
    """Fast-mode recordings have no tracked video; draw it now from the original video and the saved detections."""
    if not has_source(rid):
        st.info("This recording has no tracked video, and the original video was deleted, so it cannot be drawn.")
        return
    mins = meta.get("duration_s", 0) / 60
    st.info(f"Detection ran in fast mode, so there is no tracked video of the whole clip yet (event clips are drawn "
            f"when you open them). Drawing it takes roughly {max(1, round(mins / 6))} min for this "
            f"{mins:.0f}-min clip.")
    if st.button("Draw the tracked video now", icon=":material/movie:"):
        a = get_analysis(rid)
        det = a[2] if a else load_tracker_csv(run_dir(rid) / "detections.csv")
        priv = meta.get("privacy") or {}
        ro = RenderOptions(blur_privacy=priv.get("blur", True),
                           privacy_level=priv.get("level") if priv.get("level") in ("standard", "strict") else "standard")
        bar = st.progress(0.0, text="Drawing the tracked video...")
        tmp = run_dir(rid) / "annotated.partial.mp4"
        render_from_detections(source_path(rid), det, str(tmp), float(meta["fps"]), int(meta.get("stride", 1)), ro,
                               0.0, float(meta.get("duration_s", 0)),
                               progress=lambda f: bar.progress(f, text=f"Drawing the tracked video... {f:.0%}"))
        tmp.replace(run_dir(rid) / "annotated.mp4")
        st.rerun()


def slowmo_ui(S: Settings, events) -> None:
    st.write("Make a version of the result video that **slows down around every flagged event**. A coloured border and "
             f"banner mark the slowed part. Made from the tracked video, which is **{video_privacy_text(meta)}**.")
    ann = run_dir(rid) / "annotated.mp4"
    if not ann.exists():
        draw_tracked_video_ui()
        return
    if not events:
        st.info("No events are flagged, so there is nothing to slow down.")
        return
    a, b, c, d = st.columns(4)
    factor = a.select_slider("Slow down by", [2, 3, 4, 6, 8], value=4, format_func=lambda f: f"{f}x")
    before = b.number_input("Start before the event (s)", 0.0, 10.0, 2.0, 0.5)
    after = c.number_input("Keep slow after (s)", 0.0, 10.0, 1.0, 0.5)
    only = d.toggle("Highlights only", help="Only the slowed parts, one after another.")
    present = [s for s in ("serious", "moderate", "candidate") if any(e["severity"] == s for e in events)]
    sevs = st.multiselect("Events to include", present, default=present, format_func=str.capitalize)
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


def tracks_csv(tracks) -> str:
    """Every smoothed trajectory sample in map metres, so a reviewer can recompute any number."""
    out = ["track_id,segment,class,group,time_s,x_m,y_m,vx_mps,vy_mps,speed_mps"]
    for t in tracks:
        for k in range(len(t.t)):
            out.append(f"{t.id},{t.seg},{t.cls},{t.group},{t.t[k]:.2f},{t.xy[k, 0]:.3f},{t.xy[k, 1]:.3f},"
                       f"{t.v[k, 0]:.3f},{t.v[k, 1]:.3f},{t.speed[k]:.3f}")
    return "\n".join(out) + "\n"


REPRODUCE_PY = '''"""Recompute this recording's events from the inputs in this bundle with the NearMiss engine.
Run from the NearMiss folder:  python reproduce.py path/to/unzipped_bundle"""
import json, sys
from pathlib import Path
from engine.analysis import analyze, load_tracker_csv
from engine.calibration import Calibration
from engine.config import Settings
from engine.motion import load_motion
from engine.roi import as_polygon

b = Path(sys.argv[1]) / "inputs"
rd = lambda n, d=None: json.loads((b / n).read_text()) if (b / n).exists() else d
meta = rd("meta.json", {})
_, stats, events = analyze(load_tracker_csv(b / "detections.csv"), Calibration.from_dict(rd("calibration.json")),
                           Settings.from_dict(rd("settings.json")), load_motion(rd("motion.json")),
                           as_polygon(rd("roi.json", {}).get("points")), meta.get("duration_s"))
want = json.loads((Path(sys.argv[1]) / "outputs" / "events.json").read_text())
print(f"{len(events)} events recomputed, {len(want)} in the bundle:", "MATCH" if len(events) == len(want) else "DIFFERENT")
'''


def bundle_zip(S: Settings, stats: dict, tracks, events) -> bytes:
    """A reproducible bundle: the exact inputs, the outputs, versions, and a script that recomputes the events."""
    import ultralytics
    buf = io.BytesIO()
    d = run_dir(rid)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("detections.csv", "calibration.json", "settings.json", "motion.json", "roi.json", "zones.json",
                  "labels.json", "meta.json", "recording_info.json"):
            if (d / f).exists():
                z.write(d / f, f"inputs/{f}")
        z.writestr("outputs/events.json", json.dumps(events, indent=1, default=str))
        z.writestr("outputs/events.csv", pd.DataFrame(events_table(events)).to_csv(index=False))
        z.writestr("outputs/tracks.csv", tracks_csv(tracks))
        z.writestr("outputs/stats.json", json.dumps(stats, indent=1, default=str))
        z.writestr("outputs/speeds.json", json.dumps(speed_summary(tracks, S), indent=1))
        z.writestr("manifest.json", json.dumps({
            "recording": display_name(meta, d), "made": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
            "ultralytics": ultralytics.__version__, "privacy_of_videos": privacy_of(meta),
            "contains": "inputs needed to recompute the analysis, the outputs, no video frames"}, indent=1))
        z.writestr("reproduce.py", REPRODUCE_PY)
    return buf.getvalue()


def labels_io_ui() -> None:
    store, problems = load_labels(rid)
    for p in problems:
        st.warning(f"labels.json: {p}")
    a, b = st.columns(2)
    a.download_button("labels.json", json.dumps(store, indent=1), f"labels_{rid}.json", "application/json",
                      icon=":material/download:")
    up = b.file_uploader("Import labels.json", type=["json"], key=f"lab_up_{rid}",
                         help="Restores Event Review labels, e.g. from another computer or a re-run of the same clip.")
    if up is None:
        return
    try:
        incoming, probs = validation.clean_store(json.loads(up.getvalue()))
    except ValueError:
        st.error("That file is not JSON, so nothing was imported.")
        return
    for p in probs[:5]:
        st.warning(p)
    if len(probs) > 5:
        st.warning(f"... and {len(probs) - 5} more problems.")
    st.caption(f"Usable: {len(incoming['events'])} labels and {len(incoming['missed'])} missed notes.")
    a, b = st.columns(2)
    if a.button("Add them to my labels", disabled=not (incoming["events"] or incoming["missed"])):
        n_lab, n_missed = validation.merge_labels(store, incoming)
        save_labels(rid, store)
        st.success(f"Added or updated {n_lab} labels and {n_missed} missed notes.")
    if b.button("Replace my labels with them"):
        save_labels(rid, incoming)
        st.success("Replaced.")


def outputs_tab(S, stats, tracks, events) -> None:
    st.subheader("Slow-motion video", anchor=False)
    slowmo_ui(S, events)
    st.subheader("Downloads", anchor=False)
    st.caption("Built from the current analysis each time, so they are never out of date. None contains video frames.")
    a, b, c = st.columns(3)
    a.download_button("Reproducible bundle (.zip)", bundle_zip(S, stats, tracks, events), f"nearmiss_{rid}.zip",
                      "application/zip", icon=":material/download:",
                      help="Exact inputs, outputs, versions and a reproduce.py that recomputes the events.")
    b.download_button("events.csv", pd.DataFrame(events_table(events)).to_csv(index=False), f"events_{rid}.csv",
                      "text/csv", icon=":material/download:")
    c.download_button("tracks.csv", tracks_csv(tracks), f"tracks_{rid}.csv", "text/csv", icon=":material/download:")
    st.subheader("Labels", anchor=False)
    labels_io_ui()


def page_results():
    page_header("Results", "Results", "What was flagged, where, by which movements, and how far to trust it.")
    if need_run():
        return
    a = get_analysis(rid)
    if a is None:
        st.info("This recording is not calibrated yet, so distances and speeds cannot be measured.")
        st.button("Go to Calibrate", type="primary", on_click=go, args=("Calibrate",))
        return
    cal, S, det, tracks, stats, events = a
    store, _ = load_labels(rid)
    summ = validation.summarize(events, store)
    sev = pd.Series([e["severity"] for e in events]).value_counts() if events else pd.Series(dtype=int)
    prov = "*" if events and events[0].get("provisional") else ""
    kpis([("Flagged events", len(events)), (f"Serious{prov}", int(sev.get("serious", 0))),
          (f"Moderate{prov}", int(sev.get("moderate", 0))), (f"Candidate{prov}", int(sev.get("candidate", 0))),
          ("Confirmed real", f"{summ['tp']} of {summ['reviewed'] + summ['unsure']} reviewed" if summ["reviewed"] + summ["unsure"]
           else "not reviewed"),
          ("Road users", stats["analysed_tracks"]), ("Flagged per hour", rate(stats["events_per_hour"]))])
    quality_panel(a, summ, key="qr")
    tabs = st.tabs([":material/map: Overview", ":material/traffic: Traffic & movements",
                    ":material/fact_check: Quality & thresholds", ":material/download: Outputs"])
    with tabs[0]:
        overview_tab(cal, S, tracks, events)
    with tabs[1]:
        traffic_tab(cal, S, tracks, stats, events, summ)
    with tabs[2]:
        quality_tab(S, det, stats, tracks)
    with tabs[3]:
        outputs_tab(S, stats, tracks, events)


# ================================================================== EVENT REVIEW
def step_event(key: str, delta: int, n: int) -> None:
    st.session_state[key] = min(max(st.session_state.get(key, 0) + delta, 0), n - 1)


def label_event(ev: dict, lab: str, key: str, go_to: int, all_events: list) -> None:
    """Button callback: save the label for THIS event (matched by its two road users), then move on."""
    store, _ = load_labels(rid)
    validation.set_label(store, ev["kind"], ev["t_peak"], lab, st.session_state.get(f"note_{rid}_{ev['key']}", ""),
                         pair=validation.event_pair(ev), events=all_events)
    save_labels(rid, store)
    st.session_state[key] = go_to


def review_filters(events: list) -> list:
    """Narrow the review list by type, severity, review status and zone."""
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


def missed_ui() -> None:
    """Near misses the app did not flag. Available even when nothing was flagged: that is exactly when they matter."""
    with st.expander("Add a near miss the app missed", icon=":material/add_alert:"):
        st.caption("While watching the video, note real near misses that were not flagged. They count against recall "
                   "in the report.")
        store, _ = load_labels(rid)
        a, b = st.columns([1, 3])
        t_m = a.number_input("Time in the video (s)", 0.0, float(meta.get("duration_s") or 0), 0.0, 0.5)
        note = b.text_input("What happened?", placeholder="e.g. car turning right nearly hit a cyclist")
        if st.button("Add to missed list"):
            store.setdefault("missed", []).append({"t": t_m, "note": note})
            save_labels(rid, store)
            st.rerun()
        missed = store.get("missed", [])
        if missed:
            st.dataframe(pd.DataFrame(missed).rename(columns={"t": "time (s)", "note": "what happened"}), width="stretch")
            gone = st.selectbox("Remove one", range(len(missed)),
                                format_func=lambda i: f"{missed[i]['t']} s: {missed[i]['note']}")
            if st.button("Remove"):
                missed.pop(gone)
                save_labels(rid, store)
                st.rerun()


def distance_chart(tracks, ev) -> None:
    """Separation over time with a line at the frame shown in *Step through frames*; clicking the chart moves the
    frame there."""
    import altair as alt

    ds = distance_series(tracks, ev)
    if not ds["t"]:
        return
    tkey = f"tstep_{rid}_{ev['key']}"
    now = float(st.session_state.get(tkey, ev["t_peak"]))
    df = pd.DataFrame({"t": ds["t"], "distance": ds["distance_m"]})
    pick = alt.selection_point(fields=["t"], nearest=True, on="click", name="pick")
    line = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=18, opacity=0.4)).encode(
        x=alt.X("t:Q", title="time in the video (s)", scale=alt.Scale(zero=False)),
        y=alt.Y("distance:Q", title="distance between them (m)", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("t:Q", format=".1f"), alt.Tooltip("distance:Q", format=".2f")]).add_params(pick)
    rule = alt.Chart(pd.DataFrame({"t": [now]})).mark_rule(color="#c62828", strokeWidth=2).encode(x="t:Q")
    low = df.loc[[int(df["distance"].idxmin())]]
    dot = alt.Chart(low).mark_point(color="#c62828", size=70, filled=True).encode(x="t:Q", y="distance:Q")
    got = st.altair_chart((line + rule + dot).properties(height=170), width="stretch", on_select="rerun",
                          key=f"dchart_{rid}_{ev['key']}")
    pts = (got or {}).get("selection", {}).get("pick") or []
    if pts and "t" in pts[0] and abs(float(pts[0]["t"]) - now) > 1e-6:
        st.session_state[f"_tjump_{rid}"] = (ev["key"], float(pts[0]["t"]))   # applied before the slider is drawn
        st.rerun()
    st.caption(f"Lowest point (red dot): {low['distance'].iloc[0]} m at {low['t'].iloc[0]} s, the same moment as "
               f"*Closest*. The red line is the frame shown under *Step through frames* ({now:.1f} s); click the chart "
               "to move it.")


def review_player(det, S, tracks, ev) -> None:
    """One large player: the looping clip, or step through the frames with the two road users emphasised."""
    tabs = st.tabs([":material/play_circle: Clip", ":material/frame_inspect: Step through frames"])
    t0, t1 = ev["t_start"] - S.clip_before_s, ev["t_end"] + S.clip_after_s
    with tabs[0]:
        clip = run_dir(rid) / f"clip_{max(t0, 0):.2f}_{t1:.2f}.mp4"     # by time: event IDs change with thresholds
        if not clip.exists():
            with st.spinner("Making the clip..."):
                event_clip(det, max(t0, 0), t1, clip)
        if clip.exists():
            st.video(str(clip), loop=True)
            st.caption(f"Clip {max(t0, 0):.1f}-{t1:.1f} s, looping. From the tracked video, which is "
                       f"{video_privacy_text(meta)}.")
        else:
            st.caption("No clip: the tracked video and the original video are both missing, or ffmpeg is missing.")
    with tabs[1]:
        if not has_source(rid):
            st.caption("The original video was deleted for privacy, so frames cannot be stepped through.")
            return
        lo, hi = round(max(0.0, ev["t_start"] - ENCOUNTER_PAD_S), 1), round(ev["t_end"] + ENCOUNTER_PAD_S, 1)
        tkey = f"tstep_{rid}_{ev['key']}"
        if tkey not in st.session_state:
            st.session_state[tkey] = float(round(ev["t_peak"], 1))
        jump = st.session_state.pop(f"_tjump_{rid}", None)        # a click on the distance chart
        if jump and jump[0] == ev["key"]:
            st.session_state[tkey] = float(min(hi, max(lo, round(jump[1], 1))))
        a, b, c, d = st.columns([1, 1, 1, 1])
        a.button("-0.1 s", key=f"tb_{tkey}", width="stretch",
                 on_click=lambda: st.session_state.__setitem__(tkey, round(max(lo, st.session_state[tkey] - 0.1), 1)))
        b.button("+0.1 s", key=f"tf_{tkey}", width="stretch",
                 on_click=lambda: st.session_state.__setitem__(tkey, round(min(hi, st.session_state[tkey] + 0.1), 1)))
        zoom = c.toggle("Zoom", key=f"zoom_{rid}", help="Crop to the two road users.")
        dim = d.toggle("Dim the rest", True, key=f"dim_{rid}")
        t = st.slider("Time (s)", lo, hi, key=tkey, step=0.1)
        blur = not st.toggle("Show unblurred (this computer only)", False, key=f"unblur_{rid}")
        st.image(rgb(event_frame(source_path(rid), det, ev, S.privacy_level, blur, t=t, emphasis=dim, zoom=zoom)),
                 width="stretch", caption=f"t = {t:.1f} s. The closest moment is t = {ev['t_peak']} s.")


def page_review():
    page_header("Event Review", "Event review", "Watch each flagged moment and decide whether it was a real near miss.")
    if need_run():
        return
    a = get_analysis(rid)
    if a is None:
        st.info("Calibrate this recording first: events are found after calibration.")
        missed_ui()
        return
    cal, S, det, tracks, stats, events = a
    _, problems = load_labels(rid)
    for p in problems:
        st.warning(f"labels.json: {p} (it was ignored)")
    if not events:
        st.info("Nothing was flagged with the current thresholds. If you saw a near miss in the video, add it below: "
                "it counts against recall, and a recording with no flags but real near misses is important evidence.")
        missed_ui()
        return
    all_events = events
    target = st.session_state.pop(f"ev_target_{rid}", None)
    key = f"ev_{rid}"
    events = review_filters(all_events)
    if target is not None:
        st.session_state[key] = next((i for i, e in enumerate(events) if e["key"] == target), 0)
    if not events:
        st.info("No events match the filter.")
        missed_ui()
        return
    if st.session_state.get(key, 0) >= len(events):
        st.session_state[key] = 0
    cur = st.session_state.get(key, 0)
    idx_row = st.columns([3, 2])
    n1, n2, n3 = idx_row[1].columns(3)
    n1.button("Prev", on_click=step_event, args=(key, -1, len(events)), disabled=cur == 0, width="stretch",
              shortcut="Left", help="Previous event (left arrow)")
    n2.button("Next", on_click=step_event, args=(key, 1, len(events)), disabled=cur == len(events) - 1,
              width="stretch", shortcut="Right", help="Next event (right arrow)")
    n3.button("To do", on_click=next_unreviewed, args=(key, events, cur), icon=":material/skip_next:",
              disabled=all(e.get("label") for e in events), width="stretch", help="Next event without a label")
    idx = idx_row[0].selectbox("Event", range(len(events)), key=key, format_func=lambda i: event_name(events[i]),
                       label_visibility="collapsed")
    ev = events[idx]
    reviewed = sum(1 for e in all_events if e.get("label"))
    st.progress(reviewed / len(all_events), text=f"Reviewed {reviewed} of {len(all_events)}"
                + ("" if len(events) == len(all_events) else f" · showing {len(events)} that match the filter"))

    c1, c2 = st.columns([3, 2])
    with c1:
        review_player(det, S, tracks, ev)
    with c2:
        st.markdown(f"{badge(ev['severity'], ev.get('provisional'))} &nbsp; **{ev['id']} · {ev['kind_label']}**"
                    + (f" · zone: {html_escape(ev['zone'])}" if ev.get("zone") else ""), unsafe_allow_html=True)
        stat_grid([("Closest", f"{ev['d_min_m']} m", "Centre-to-centre distance at the closest moment of the encounter."),
                   ("Min TTC", f"{ev['ttc_min_s']} s" if ev["ttc_min_s"] is not None else "n/a",
                    "Time to collision: seconds until they would touch if both kept going straight at the same speed."),
                   ("PET", f"{ev['pet_s']} s" if ev["pet_s"] is not None else "n/a",
                    "Post-encroachment time: from the first one leaving the shared conflict point to the second one "
                    "reaching it."),
                   ("Braking needed", f"{ev['drac_max_mps2']} m/s²" if ev.get("drac_max_mps2") else "n/a",
                    "DRAC: the braking either road user would have needed to avoid the collision. Above 3.35 m/s² is "
                    "harder than comfortable braking."),
                   ("Time", f"{ev['t_peak']} s", "Seconds into the video."),
                   ("Movements", ev["movements"], "Approach and turn of each road user."),
                   ("Phase", ev.get("phase") or "-", "Signal phase, if typed on Results.")])
        if ev.get("provisional"):
            st.caption("Severity is provisional: the calibration is not checked or looks wrong.")
        st.write(explain_event(ev))
        if ev.get("below_3ft"):
            st.warning("Estimated clearance is under 3 ft (0.91 m), the California passing rule for bikes.")
        st.caption("Why it was flagged: " + "; ".join(ev["reasons"]))
        distance_chart(tracks, ev)
        st.write("**Was this a real near miss?**")
        note_key = f"note_{rid}_{ev['key']}"
        if note_key not in st.session_state:
            st.session_state[note_key] = ev.get("label_note", "")
        st.text_input("Note (optional, saved with your label)", key=note_key,
                      placeholder="e.g. car stopped late at the crosswalk")
        b = st.columns(3)
        nxt = idx if st.session_state.get(f"f_stat_{rid}") == "not reviewed" else min(idx + 1, len(events) - 1)
        for col, lab, sc in zip(b, ["tp", "fp", "unsure"], ["T", "F", "U"]):
            col.button(LABEL_TEXT[lab], key=f"{lab}_{ev['key']}", type="primary" if ev.get("label") == lab else "secondary",
                       width="stretch", shortcut=sc, on_click=label_event, args=(ev, lab, key, nxt, all_events))
        st.caption("Keyboard: **T** real, **F** false alarm, **U** unsure, **← / →** previous / next.")
        if ev.get("label"):
            st.button("Remove my label", key=f"clear_{ev['key']}", on_click=label_event,
                      args=(ev, "clear", key, idx, all_events))
    missed_ui()


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


def kind_rows(by_kind: dict, m: dict) -> list:
    rows = []
    for k in ("ped-veh", "bike-veh", "veh-veh"):
        v = by_kind.get(k)
        if not measured(m, k):
            rows.append({"type": KIND_LABEL[k], "flagged": "Not measured", "real": "", "false alarms": "", "unsure": "",
                         "not reviewed": "", "precision": ""})
        elif v:
            rows.append({"type": KIND_LABEL[k], "flagged": v["flagged"], "real": v["tp"], "false alarms": v["fp"],
                         "unsure": v["unsure"], "not reviewed": v["unreviewed"],
                         "precision": "n/a" if v["precision"] is None else f"{v['precision']:.0%}"})
    return rows


def recording_summary(r: str):
    """Totals for one recording, for comparisons: None if it is not calibrated."""
    a = get_analysis(r)
    if a is None:
        return None
    m = read_json(run_dir(r) / "meta.json", {}) or {}
    store, _ = load_labels(r)
    s = validation.summarize(a[5], store)
    stats = a[4]
    return {"rid": r, "name": display_name(m, run_dir(r)), "meta": m, "settings": a[1], "summary": s, "stats": stats,
            "observation_s": stats.get("observation_s") or m.get("duration_s") or 0, "status": cal_status(r, m)}


def compare_ui() -> None:
    """Before/after: totals and rates for two sets of recordings, with what makes them not comparable."""
    runs = [(r, l) for r, l in list_runs() if not (read_json(run_dir(r) / "meta.json", {}) or {}).get("synthetic")
            and (run_dir(r) / "calibration.json").exists()]
    if len(runs) < 2:
        st.info("Needs at least two calibrated real recordings, for example before and after a street change.")
        return
    a, b = st.columns(2)
    names = dict(runs)
    before = a.multiselect("Before", [r for r, _ in runs], format_func=names.get, key="cmp_before")
    after = b.multiselect("After", [r for r, _ in runs], format_func=names.get, key="cmp_after")
    if not before or not after:
        st.caption("Pick the recordings of each period. Several recordings per period are added together.")
        return
    cols, warn = {}, []
    groups = {"Before": [recording_summary(r) for r in before], "After": [recording_summary(r) for r in after]}
    for g, rs in groups.items():
        rs = [x for x in rs if x]
        obs = sum(x["observation_s"] for x in rs)
        veh = sum(x["stats"]["road_users"].get("veh", 0) for x in rs)
        flagged = sum(x["summary"]["events_flagged"] for x in rs)
        conf = sum(x["summary"]["tp"] for x in rs)
        reviewed = sum(x["summary"]["reviewed"] for x in rs)
        pairs = sum(sum(x["stats"]["pairs_together"].values()) for x in rs)
        vru = any(blind_spot(x["meta"]) for x in rs)
        cols[g] = {"recordings": len(rs), "observed (min)": round(obs / 60, 1), "vehicles": veh,
                   "pedestrians": "Not measured" if vru else sum(x["stats"]["road_users"].get("ped", 0) for x in rs),
                   "flagged": flagged, "confirmed real": conf if reviewed else "not reviewed",
                   "flagged per hour": round(flagged / (obs / 3600), 1) if obs else "n/a",
                   "confirmed per hour": round(conf / (obs / 3600), 1) if obs and reviewed else "n/a",
                   "flagged per 100 vehicles": round(100 * flagged / veh, 1) if veh else "n/a",
                   "flagged per 1,000 pairs": round(1000 * flagged / pairs, 1) if pairs else "n/a"}
        if any(x["status"]["provisional"] for x in rs):
            warn.append(f"{g}: at least one calibration is not checked or looks wrong.")
        if any(x["summary"]["unreviewed"] for x in rs):
            warn.append(f"{g}: not every flag is reviewed, so 'confirmed' counts are incomplete.")
    st.dataframe(pd.DataFrame(cols).astype(str), width="stretch")
    allr = [x for rs in groups.values() for x in rs if x]
    thr = {json.dumps({k: v for k, v in x["settings"].to_dict().items() if k.startswith(("ttc_", "proximity",
                                                                                          "min_", "collision"))},
                      sort_keys=True) for x in allr}
    if len(thr) > 1:
        warn.append("The recordings use different thresholds, so their flags are not comparable. Use the same ones.")
    models = {x["meta"].get("model_used") for x in allr}
    if len(models) > 1:
        warn.append(f"Different detection models were used ({', '.join(str(m) for m in models)}).")
    o = [cols[g]["observed (min)"] for g in cols]
    if min(o) and max(o) / min(o) > 2:
        warn.append("One period was observed more than twice as long as the other. Rates allow a comparison, but the "
                    "shorter one is less certain.")
    for w in warn:
        st.warning(w)
    st.caption("Compare rates, not raw counts: more traffic or a longer recording means more flags with no change in "
               "risk. Same time of day and weekday make periods more comparable.")


def page_report():
    page_header("Report", "Report", "How good the flags were, comparisons, a second opinion, and a shareable report.")
    if need_run():
        return
    a = get_analysis(rid)
    if a is None:
        st.info("Finish Detect and Calibrate first.")
        return
    cal, S, det, tracks, stats, events = a
    store, problems = load_labels(rid)
    for p in problems:
        st.warning(f"labels.json: {p} (it was ignored)")
    summ = validation.summarize(events, store)
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    kpis([("Flagged", summ["events_flagged"]), ("Real", summ["tp"]), ("False alarms", summ["fp"]),
          ("Not reviewed", summ["unreviewed"]),
          ("Precision", pct(summ["precision"]) + (f" (of {summ['reviewed']} reviewed)" if summ["unreviewed"] else "")),
          ("Recall (at most)", pct(summ["recall_upper_bound"]))])
    if summ["unreviewed"]:
        st.button(f"Review the {summ['unreviewed']} remaining event(s)", on_click=go, args=("Event Review",),
                  icon=":material/fact_check:")
    quality_panel(a, summ, key="qp")
    tabs = st.tabs([":material/verified: Validation", ":material/compare_arrows: Compare recordings",
                    ":material/tune: Tuning", ":material/group: Second reviewer",
                    ":material/edit_note: Recording details", ":material/shield: Privacy",
                    ":material/description: Build report"])
    with tabs[0]:
        st.caption("Precision = real / (real + false alarms), over the events you reviewed. Recall can only be bounded "
                   "from above, because the app cannot know about events nobody marked as missed.")
        st.dataframe(pd.DataFrame(kind_rows(summ["by_kind"], meta)), hide_index=True, width="stretch")
        with st.expander("All numbers"):
            st.json(summ)
        all_recordings_ui()
    with tabs[1]:
        compare_ui()
    with tabs[2]:
        tuning_ui(S, events)
    with tabs[3]:
        second_review_ui(det, S, events, store)
    with tabs[4]:
        metadata_ui()
    with tabs[5]:
        privacy_ui()
    with tabs[6]:
        build_report_ui(cal, S, det, tracks, stats, events, summ)


def all_recordings_ui() -> None:
    """Validation over every reviewed real recording, so the submission can quote one set of numbers."""
    st.write("**All recordings**")
    st.caption("Every real (not synthetic) recording that is calibrated and has labels. Precision comes from the "
               "summed counts, not an average of percentages.")
    if not st.toggle("Compute it", key="all_val"):
        return
    rows, summs, conds = [], [], []
    for r, label in list_runs():
        m = read_json(run_dir(r) / "meta.json", {}) or {}
        store, _ = load_labels(r)
        if m.get("synthetic") or not (store.get("events") or store.get("missed")):
            continue
        x = recording_summary(r)
        if x is None:
            continue
        s = x["summary"]
        summs.append(s)
        info = read_json(run_dir(r) / "recording_info.json", {}) or {}
        conds.append({"lighting": info.get("lighting") or "not recorded", "weather": info.get("weather") or "not recorded",
                      "camera view": {"overhead": "drone", "angled": "street-level"}.get(m.get("view"), "unknown"),
                      "camera": info.get("camera") or "not recorded", "detection model": m.get("model_used") or "?",
                      "summary": s})
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
    by_condition_ui(conds)


def by_condition_ui(conds: list) -> None:
    """Precision per recording condition (lighting, weather, view...): where the tool works and where it breaks."""
    st.write("**By condition**")
    dim = st.radio("Group recordings by", ["lighting", "weather", "camera view", "camera", "detection model"],
                   horizontal=True, key="cond_dim")
    groups = {}
    for c in conds:
        groups.setdefault(c[dim], []).append(c["summary"])
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    rows = []
    for value, ss in sorted(groups.items()):
        t = validation.combine(ss)
        rows.append({dim: value, "recordings": t["recordings"], "reviewed": t["tp"] + t["fp"], "real": t["tp"],
                     "false alarms": t["fp"], "precision": pct(t["precision"]),
                     "recall (at most)": pct(t["recall_upper_bound"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    small = [r[dim] for r in rows if r["reviewed"] < 10]
    st.caption("Fill in *Recording details* for each recording to group by lighting, weather or camera."
               + (f" Fewer than 10 reviewed events in: {', '.join(map(str, small))}; their precision can swing a lot "
                  "with one more label." if small else ""))


def event_frames(det, S: Settings, events) -> dict:
    """Blurred closest-moment frames as base64 JPEGs; none once the original video has been deleted."""
    if not has_source(rid):
        return {}
    return {e["id"]: jpg_b64(event_frame(source_path(rid), det, e, S.privacy_level, True)) for e in events}


def report_sections(cal, S, tracks, stats, events, summ) -> list:
    """Plain-language sections a planner reads first."""
    status = cal_status(rid, meta)
    obs = stats.get("observation_s") or 0
    sev = {s: sum(e["severity"] == s for e in events) for s in ("serious", "moderate", "candidate")}
    top = pd.Series([e["movements"] for e in events]).value_counts().head(3) if events else pd.Series(dtype=int)
    lines = [f"{len(events)} potential conflicts were flagged in {obs / 60:.1f} minutes of video "
             f"({sev['serious']} serious, {sev['moderate']} moderate, {sev['candidate']} candidate"
             + (", all provisional because the calibration is not confirmed" if status["provisional"] else "") + ").",
             f"{summ['tp']} were confirmed as real near misses by a reviewer, {summ['fp']} were false alarms, and "
             f"{summ['unreviewed']} are not reviewed yet."]
    if blind_spot(meta):
        lines.append("Pedestrian and bicycle conflicts were not measured (vehicles-only model).")
    if len(top):
        lines.append("Most flags involved: " + "; ".join(f"{k} ({v})" for k, v in top.items()) + ".")
    summary_html = "<ul>" + "".join(f"<li>{html_escape(x)}</li>" for x in lines) + "</ul>"
    info = recording_info(rid)
    mc = movement_counts(tracks, events, float(info.get("north_deg", 0) or 0))
    mov_html = ("<table><tr><th>Movement</th><th>Road users</th><th>Events</th><th>Events per 100</th></tr>"
                + "".join(f"<tr><td>{html_escape(r['movement'])}</td><td>{r['road users']}</td><td>{r['events']}</td>"
                          f"<td>{r['events per 100 road users']}</td></tr>" for r in mc) + "</table>")
    out = [("Summary", summary_html), ("Calibration", f"<p><b>{html_escape(status['title'])}.</b> "
                                                      f"{html_escape(status['detail'])}</p>"), ("Movements", mov_html)]
    phases, _ = parse_phases(info.get("signal_phases", ""))
    if phases:
        out.append(("Signal phases", "<table><tr><th>Phase</th><th>From (s)</th><th>To (s)</th><th>Events</th></tr>"
                    + "".join(f"<tr><td>{html_escape(n)}</td><td>{t0:g}</td><td>{t1:g}</td>"
                              f"<td>{sum(1 for e in events if e.get('phase') == n)}</td></tr>" for t0, t1, n in phases)
                    + "</table>"))
    return out


def build_report_ui(cal, S, det, tracks, stats, events, summ) -> None:
    preset = st.radio("Report for", ["planner", "competition"], horizontal=True, key=f"preset_{rid}",
                      format_func={"planner": "A planner or engineer (summary first, settings in an appendix)",
                                   "competition": "The Congressional App Challenge (with method details and the "
                                                  "AI-use disclosure)"}.get)
    disclosure = ""
    if preset == "competition":
        disclosure = st.text_area("AI-use disclosure (required by the competition)", AI_TEXT, height=200)
    state, rmeta = report_state(rid)
    if state == "outdated":
        st.warning("The saved report is **out of date**: labels, calibration, settings or details changed after it was "
                   "built. Build it again before sharing it.")
    if st.button("Build report", type="primary", icon=":material/description:"):
        _, cd = load_calibration(rid)
        status = cal_status(rid, meta)
        notes = []
        if blind_spot(meta):
            notes.append("Vehicles only: the model used cannot detect pedestrians or bicycles, so their conflicts were "
                         "NOT MEASURED. Zero of them is a blind spot, not a safe result.")
        if status["level"] != "good":
            notes.append(f"{status['title']}. {status['detail']}")
        p = privacy_of(meta)
        if p and not p.get("blur"):
            notes.append("The videos of this recording were made without blur. This report's pictures are blurred.")
        page = build_report_html(
            display_name(meta, run_dir(rid)), {k: v for k, v in meta.items() if k != "settings"},
            {k: cd[k] for k in ("width_m", "length_m", "input", "checks") if k in cd}, S, events, summ, disclosure,
            png_b64(topdown_map(tracks, events, cal, zones=zone_overlays(cal, rid))), event_frames(det, S, events[:30]),
            notes=notes,
            exposure={k: stats[k] for k in ("road_users", "pairs_together", "events_per_100_vehicles",
                                            "events_per_1000_pairs", "events_per_1000_pairs_by_kind", "events_per_hour",
                                            "observation_s")},
            info={k: v for k, v in recording_info(rid).items() if k not in ("signal_phases",)}, explain=explain_event,
            speeds=speed_summary(tracks, S, (recording_info(rid).get("posted_limit_mph") or 0) / MPS_TO_MPH or None),
            zone_rows=zone_summary(events, tracks, load_zones(rid), cal) if load_zones(rid) else None,
            sections=report_sections(cal, S, tracks, stats, events, summ), planner=preset == "planner")
        (run_dir(rid) / "report.html").write_text(page)
        write_json(run_dir(rid) / "report.json", {"signature": report_signature(rid), "built": time.time(),
                                                  "preset": preset})
        st.rerun()
    state, rmeta = report_state(rid)
    if state == "current":
        st.success(f"Report built {time.strftime('%b %d %H:%M', time.localtime(rmeta['built']))} "
                   f"({rmeta.get('preset', 'competition')} version) and up to date.")
        st.download_button("Download report.html", (run_dir(rid) / "report.html").read_text(), "nearmiss_report.html",
                           "text/html", icon=":material/download:")
        st.caption("For a PDF: open the file in a browser and choose File > Print > Save as PDF.")


def tuning_ui(S: Settings, events) -> None:
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
                       "misses comes from " + " and ".join(parts) + ". Re-check it on a different clip before trusting it.")
        else:
            st.info("No stricter cutoff keeps 80% of your real near misses, so the current thresholds stay.")
    a, b = st.columns(2)
    a.write("**TTC cutoff**")
    a.dataframe(pd.DataFrame(t["ttc"]), hide_index=True, width="stretch")
    b.write("**Distance cutoff (x current)**")
    b.dataframe(pd.DataFrame(t["distance"]), hide_index=True, width="stretch")


def pack_clips(events) -> dict:
    """Small blurred clips for the review pack (base64)."""
    out = {}
    a = get_analysis(rid)
    if a is None or not ((run_dir(rid) / "annotated.mp4").exists() or has_source(rid)):
        return out
    S = load_settings(rid)
    for e in events:
        t0, t1 = max(0.0, e["t_start"] - S.clip_before_s), e["t_end"] + S.clip_after_s
        path = run_dir(rid) / f"packclip_{t0:.2f}_{t1:.2f}.mp4"
        if path.exists() or event_clip(a[2], t0, t1, path, max_width=640):
            out[e["id"]] = base64.b64encode(path.read_bytes()).decode()
    return out


def event_clip(det, t0: float, t1: float, out: Path, max_width: Optional[int] = None) -> bool:
    """A blurred clip of [t0, t1]: cut from the tracked video, or, when detection ran in fast mode without one,
    drawn now from the original video and the saved detections (same blur, boxes and trails)."""
    ann = run_dir(rid) / "annotated.mp4"
    if ann.exists():
        return cut_clip(str(ann), str(out), t0, t1, max_width=max_width)
    if not has_source(rid):
        return False
    priv = meta.get("privacy") or {}
    ro = RenderOptions(blur_privacy=priv.get("blur", True),
                       privacy_level=priv.get("level") if priv.get("level") in ("standard", "strict") else "standard",
                       max_width=max_width or 1280)
    return render_from_detections(source_path(rid), det, str(out), float(meta["fps"]), int(meta.get("stride", 1)),
                                  ro, t0, t1) > 0


def second_review_ui(det, S: Settings, events, store: dict) -> None:
    st.caption("Export a single HTML page someone else can open in any browser to label the events, without "
               "installing anything. Your own labels are left out so their review is independent. They send back a "
               "small JSON file; load it here to see how often you agree.")
    p = privacy_of(meta)
    with_clips = st.toggle("Include short video clips", True, help="About 6 s per event at 640 px, so the reviewer "
                           "can see who yielded. Makes the file larger.")
    if with_clips and p and not p.get("blur"):
        st.warning("This recording's videos are NOT blurred, so the clips would show faces and plates. Turn clips off "
                   "or re-run detection with blur.")
    if st.button("Build review pack"):
        with st.spinner("Rendering blurred frames and clips..."):
            frames = event_frames(det, S, events)
            clips = pack_clips(events) if with_clips else {}
            note = ("The pictures are blurred." if not clips or (p and p.get("blur"))
                    else "The pictures are blurred; the clips' blur setting was not recorded." if p is None
                    else "The clips are NOT blurred.")
            st.session_state[f"pack_{rid}"] = (review_pack.build_pack_html(
                display_name(meta, run_dir(rid)), events, frames, explain_event, f"{rid}-{uuid.uuid4().hex[:6]}",
                clips, note), report_signature(rid))
    if f"pack_{rid}" in st.session_state:
        html_doc, sig = st.session_state[f"pack_{rid}"]
        if sig != report_signature(rid):
            st.warning("Events or settings changed after this pack was built. Build it again.")
        else:
            st.download_button("Download review_pack.html", html_doc, f"nearmiss_review_pack_{rid}.html", "text/html")
            st.caption(f"{len(html_doc) / 1e6:.1f} MB. Check the pictures and clips before sending it to anyone.")
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
    kpis([("Labelled by both", ag["both_labelled"]), ("Agreement", f"{ag['agree']:.0%}"),
          ("Cohen's kappa", f"{ag['kappa']} ({review_pack.kappa_words(ag['kappa'])})")])
    st.write("Rows: your label · columns: " + who)
    names = {l: review_pack.LABEL_TEXT[l] for l in review_pack.LABELS}
    st.dataframe(pd.DataFrame(ag["table"]).T.rename(index=names, columns=names), width="stretch")
    if ag["disagreements"]:
        st.write("**Where you disagree** (worth watching again together)")
        st.dataframe(pd.DataFrame(ag["disagreements"]).replace(names), hide_index=True, width="stretch")
    if st.button("Save these answers with this recording"):
        write_json(run_dir(rid) / f"review_{uuid.uuid4().hex[:6]}.json", theirs)
        st.success("Saved. They are included in the reproducible bundle's inputs list.")


INFO_FIELDS = [("date", "Date recorded"), ("time", "Time of day"), ("location", "Location / intersection"),
               ("weather", "Weather"), ("lighting", "Lighting"), ("camera", "Camera (phone, action cam, drone model)"),
               ("camera_height_m", "Camera height or drone altitude (m)"), ("notes", "Anything unusual")]


def metadata_ui() -> None:
    st.caption("Saved with the recording and printed in the report, so old results stay interpretable. Directions and "
               "signal phases are set on Results > Traffic & movements.")
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
    st.caption("Everything below is on this computer only. Each file says what it actually contains.")
    vid = video_privacy_text(meta)
    rows = []
    for p in sorted(run_dir(rid).iterdir()):
        what = {"source.mp4": "original video (NOT blurred)", "annotated.mp4": f"tracked video ({vid})",
                "detections.csv": "boxes per frame (no images)", "labels.json": "your labels",
                "calibration.json": "calibration", "roi.json": "road area", "settings.json": "thresholds",
                "motion.json": "camera motion", "meta.json": "recording facts", "report.html": "report (blurred pictures)",
                "report.json": "report freshness", "recording_info.json": "recording details",
                "zones.json": "named zones"}.get(p.name)
        if what is None and p.name.startswith("slowmo_"):
            what = f"slow-motion video ({vid})"
        if what is None and p.name.startswith(("clip_", "packclip_")):
            what = f"event clip ({vid})"
        if what is None:
            what = "second reviewer's answers" if p.name.startswith("review_") else "other"
        rows.append({"file": p.name, "contains": what, "size (MB)": round(p.stat().st_size / 1e6, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if privacy_of(meta) is None:
        st.caption("This recording was made before the blur setting was saved, so whether its videos are blurred is not "
                   "known. Check them before sharing.")
    if has_source(rid):
        st.write("**Delete the original video.** The analysis, labels, tracked video and report keep working. You can no "
                 "longer redo detection or calibration, see unblurred frames, or step through frames.")
        ok = st.checkbox("I understand this cannot be undone", key=f"del_src_ok_{rid}")
        if st.button("Delete original video", disabled=not ok):
            (run_dir(rid) / "source.mp4").unlink(missing_ok=True)
            st.rerun()


# ================================================================== PROPOSED DESIGNS
def design_map(tracks, events) -> np.ndarray:
    """Top-down picture of simulated trajectories (north up: simulators use y pointing north)."""
    if not tracks:
        return np.full((200, 400, 3), 245, np.uint8)
    allp = np.vstack([t.xy for t in tracks])
    x0, y1 = allp[:, 0].min(), allp[:, 1].max()
    flip = lambda p: np.c_[p[:, 0] - x0, y1 - p[:, 1]]
    shown = [replace(t, xy=flip(t.xy)) for t in tracks]
    ev = [dict(e, x=float(flip(np.array([[e["x"], e["y"]]]))[0, 0]), y=float(flip(np.array([[e["x"], e["y"]]]))[0, 1]))
          for e in events]
    span = flip(allp).max(axis=0)
    cal = Calibration.from_points([[0, 0], [span[0], 0], [span[0], span[1]], [0, span[1]]], max(span[0], 0.5),
                                  max(span[1], 0.5), (int(span[0]) + 1, int(span[1]) + 1))
    return topdown_map(shown, ev, cal, heat=False)


def page_design():
    st.title("Proposed designs")
    st.markdown("<div class='nm-intro'>Screen an intersection that is not built yet. There is no video of a design, so "
                "this uses simulated trajectories of each alternative (the FHWA SSAM approach) and runs the same "
                "conflict check on them.</div>", unsafe_allow_html=True)
    with st.expander("How to get trajectory files", expanded=False, icon=":material/help:"):
        st.markdown("""
1. Model each design alternative in a traffic simulator (SUMO is free; VISSIM and Aimsun are common in agencies), with
   the **same demand** (volumes, pedestrian counts) and settings for every alternative.
2. Export every road user's position over time as a CSV with the columns `t_s, id, class, x_m, y_m`
   (seconds, an ID, one of pedestrian / bicycle / motorcycle / car / bus / truck, position in metres).
   SUMO: `--fcd-output`, then convert; VISSIM: vehicle record.
3. Upload one file per alternative below.

**Caveat:** a simulation only has the conflicts its driver and pedestrian models produce. Compare alternatives with
each other, never with counts from video.""")
    S = Settings()
    files = st.file_uploader("Trajectory CSV, one per alternative", type=["csv"], accept_multiple_files=True)
    alts = {}
    for f in files or []:
        rows, problems = design.read_trajectories(f.getvalue())
        for p in problems:
            st.warning(f"{f.name}: {p}")
        if rows:
            alts[Path(f.name).stem] = rows
    if st.toggle("Use the two sample alternatives", value=not alts,
                 help="Made from the synthetic demo scene: the existing design, and the same traffic with a leading "
                      "pedestrian interval and a buffered bike lane."):
        for name, text in design.demo_alternatives().items():
            alts["sample: " + name] = design.read_trajectories(text.encode())[0]
        st.download_button("Download the sample files (format example)", "\n".join(
            f"# {n}\n{t}" for n, t in design.demo_alternatives().items()), "sample_alternatives.txt")
    if not alts:
        return
    results = {name: design.evaluate(rows, S) for name, rows in alts.items()}
    table = {}
    for name, r in results.items():
        ex = r["exposure"]
        table[name] = {"simulated (min)": round(ex["observation_s"] / 60, 1),
                       "vehicles": ex["road_users"].get("veh", 0), "pedestrians": ex["road_users"].get("ped", 0),
                       "bicycles": ex["road_users"].get("bike", 0), "conflicts": len(r["events"]),
                       **{f"{s}": n for s, n in r["severity"].items()},
                       **{KIND_LABEL[k]: n for k, n in r["kinds"].items()},
                       "per hour": rate(ex["events_per_hour"]), "per 1,000 pairs": rate(ex["events_per_1000_pairs"])}
    st.subheader("Comparison", anchor=False)
    st.dataframe(pd.DataFrame(table).astype(str), width="stretch")
    demand = {name: (t["vehicles"], t["pedestrians"], t["bicycles"]) for name, t in table.items()}
    if len(set(demand.values())) > 1:
        st.warning("The alternatives do not have the same numbers of road users. Compare rates, and check that the "
                   "simulations used the same demand.")
    cols = st.columns(min(len(results), 3))
    for n, (name, r) in enumerate(results.items()):
        with cols[n % len(cols)]:
            st.image(rgb(design_map(r["tracks"], r["events"])), width="stretch", caption=name)
            for e in r["events"]:
                st.markdown(f"{badge(e['severity'])} {e['kind_label']} at {e['t_peak']} s: {explain_event(e)}",
                            unsafe_allow_html=True)


# ================================================================== METHOD
def section(title: str, icon: str, color: str) -> None:
    """A coloured section heading with an icon, used on the Method and Purpose pages."""
    st.markdown(f"<div class='nm-section'><div class='dot' style='background:{color}'>{icon}</div><h3>{title}</h3>"
                "</div>", unsafe_allow_html=True)


def card(title: str, text: str, color: str) -> str:
    rgb_ = hex_rgb(color)
    return (f"<div class='nm-card' style='background: rgba({rgb_},.08); border-color: rgba({rgb_},.3)'>"
            f"<h4 style='color:{color}'>{title}</h4><p>{text}</p></div>")


def quick_progress(r: str) -> dict:
    """Which steps a recording has done, from its files only (no analysis), so Home stays fast with many recordings."""
    d = run_dir(r)
    labels = read_json(d / "labels.json", {}) or {}
    cal = (d / "calibration.json").exists()
    return {"Detect": (d / "detections.csv").exists(), "Calibrate": cal, "Results": cal,
            "Event Review": bool(labels.get("events")), "Report": report_state(r)[0] != "none"}


def next_step(prog: dict) -> str:
    return next((p for p in STEPS if not prog.get(p)), "Report")


def open_recording(r: str, page: str) -> None:
    """Button callback: select a recording and go to a page."""
    st.session_state["rid"] = r
    st.session_state["page"] = page


def step_dots(prog: dict) -> str:
    return "<div class='nm-dots'>" + "".join(
        f"<span title='{p}' style='background:{PAGE_COLORS[p] if prog.get(p) else 'rgba(128,128,128,.25)'}'></span>"
        for p in STEPS) + "</div>"


def home_banner() -> None:
    """The big welcome banner: large logo, name, what it does, three promises, and an intersection illustration."""
    c = PAGE_COLORS["Home"]
    chips_html = "".join(f"<span><i>{i}</i>{t}</span>" for i, t in (
        ("lock", "Runs on this computer"), ("blur_on", "Faces and plates blurred"),
        ("fact_check", "Every flag checked by you")))
    st.markdown(f"""<div class='nm-homehero'>
<div class='left'>
  <div class='brandrow'><div class='biglogo'>traffic</div>
    <div><div class='kicker'>Welcome to</div><div class='big'>NearMiss</div></div></div>
  <div class='lead'>Film an intersection, and NearMiss finds every moment two road users came dangerously close, for
  you to check and report.</div>
  <div class='promises'>{chips_html}</div>
</div>
<div class='art'>{figures.hero_intersection()}</div>
</div>""", unsafe_allow_html=True)
    st.markdown(f"<style>div[data-testid='stMain'] button[kind='primary'] {{background: {c} !important; "
                f"border-color: {c} !important;}}</style>", unsafe_allow_html=True)


def page_home():
    home_banner()
    runs = list_runs()
    metas = {r: read_json(run_dir(r) / "meta.json", {}) or {} for r, _ in runs}
    last = st.session_state.get("rid") if st.session_state.get("rid") in metas else (runs[0][0] if runs else None)

    # ---- what to do now
    a, b, c = st.columns(3)
    with a.container(border=True, key="home_new"):
        st.markdown(f"<div class='nm-act'><span class='i' style='background:{PAGE_COLORS['Detect']}'>add_a_photo</span>"
                    "<b>New recording</b><br><small>Upload one or more intersection clips.</small></div>",
                    unsafe_allow_html=True)
        st.button("Start: Detect", type="primary", icon=":material/arrow_forward:", on_click=go, args=("Detect",),
                  width="stretch", key="home_go_detect")
    with b.container(border=True, key="home_continue"):
        if last:
            prog = quick_progress(last)
            nxt = next_step(prog)
            st.markdown(f"<div class='nm-act'><span class='i' style='background:{PAGE_COLORS[nxt]}'>play_arrow</span>"
                        f"<b>Continue</b><br><small>{html_escape(display_name(metas[last], run_dir(last)))}: next is "
                        f"<b style='color:{PAGE_COLORS[nxt]}'>{STEPS.index(nxt) + 1}. {nxt}</b></small></div>"
                        + step_dots(prog), unsafe_allow_html=True)
            st.button(f"Open {nxt}", icon=":material/arrow_forward:", on_click=open_recording, args=(last, nxt),
                      width="stretch", key="home_continue_btn")
        else:
            st.markdown(f"<div class='nm-act'><span class='i' style='background:{PAGE_COLORS['Results']}'>play_arrow"
                        "</span><b>Continue</b><br><small>No recordings yet. Your work will show up here.</small></div>",
                        unsafe_allow_html=True)
    with c.container(border=True, key="home_demo"):
        st.markdown(f"<div class='nm-act'><span class='i' style='background:{PAGE_COLORS['Purpose']}'>science</span>"
                    "<b>Try the demo</b><br><small>A scripted intersection with two near misses, to see every page."
                    "</small></div>", unsafe_allow_html=True)
        if demo.demo_available():
            st.button("Open the demo", icon=":material/science:", width="stretch", key="home_demo_btn",
                      on_click=lambda: st.session_state.update(rid=create_demo_run(), page="Results"))

    # ---- unfinished detections
    jobs = [(p.name, load_job(p)) for p in sorted(RUNS.iterdir()) if p.is_dir() and not (p / "meta.json").exists()]
    jobs = [(r, j) for r, j in jobs if j]
    if jobs:
        st.warning(f"**{len(jobs)} detection(s) stopped part-way**: "
                   + ", ".join(f"{j['name']} ({j['next_frame'] / max(j['meta']['frame_count'], 1):.0%})"
                               for _, j in jobs) + ". The finished parts were kept.", icon=":material/pause_circle:")
        st.button("Resume on Detect", icon=":material/play_arrow:", on_click=go, args=("Detect",), key="home_resume")

    # ---- your work so far
    real = [r for r in metas if not metas[r].get("synthetic")]
    hours = sum(metas[r].get("duration_s", 0) for r in real) / 3600
    labelled = sum(len((read_json(run_dir(r) / "labels.json", {}) or {}).get("events", [])) for r in real)
    section("Your work so far", "insights", PAGE_COLORS["Home"])
    kpis([("Recordings", len(real)), ("Footage analysed", f"{hours:.1f} h" if hours >= 1 else f"{hours * 60:.0f} min"),
          ("Road-user detections", f"{sum(metas[r].get('detection_rows', 0) for r in real):,}"),
          ("Flags you reviewed", labelled), ("Reports made", sum(report_state(r)[0] != "none" for r in real))])

    # ---- recent recordings
    if runs:
        section("Recent recordings", "video_library", PAGE_COLORS["Detect"])
        cols = st.columns(3)
        for k, (r, _) in enumerate(runs[:6]):
            m = metas[r]
            prog = quick_progress(r)
            nxt = next_step(prog)
            view = "demo" if m.get("synthetic") else {"overhead": "drone", "angled": "street-level"}.get(m.get("view"), "")
            when = time.strftime("%b %d, %H:%M", time.localtime(m.get("created", 0)))
            with cols[k % 3].container(border=True, key=f"home_rec_{k}"):
                st.markdown(f"<div class='nm-rec'><b>{html_escape(display_name(m, run_dir(r)))}</b>"
                            f"<small>{when} · {view} · {m.get('duration_s', 0) / 60:.1f} min</small></div>"
                            + step_dots(prog)
                            + f"<small style='opacity:.75'>{sum(prog.values())} of {len(STEPS)} steps · next: "
                              f"<b style='color:{PAGE_COLORS[nxt]}'>{nxt}</b></small>", unsafe_allow_html=True)
                st.button(f"Continue: {nxt}", key=f"home_open_{r}", on_click=open_recording, args=(r, nxt),
                          width="stretch", icon=":material/arrow_forward:")
        if len(runs) > 6:
            st.caption(f"{len(runs) - 6} older recording(s): pick them in the sidebar.")

    # ---- learn
    section("Learn", "school", PAGE_COLORS["Method"])
    cols = st.columns(3)
    for col, (p, text) in zip(cols, [("Purpose", "Why close calls tell you about danger long before crashes do."),
                                     ("Method", "How detection, calibration and the indicators work, with diagrams."),
                                     ("Help & contact", "Answers to common questions, and a way to ask the team.")]):
        with col.container(border=True, key=f"home_learn_{PAGES.index(p)}"):
            st.markdown(card(p, text, PAGE_COLORS[p]), unsafe_allow_html=True)
            st.button(f"Open {p}", key=f"home_learn_btn_{PAGES.index(p)}", on_click=go, args=(p,), width="stretch")


def page_purpose():
    page_header("Purpose", "Why NearMiss exists",
                "Crashes are what we want to prevent, but they are too rare at one intersection to learn from quickly. "
                "Close calls are not.")
    a, b, c = st.columns(3)
    a.markdown(card("The problem", "At any single intersection serious crashes are rare, so crash records take years "
                    "to show a pattern, and by then people have been hurt.", PAGE_COLORS["Purpose"]),
               unsafe_allow_html=True)
    b.markdown(card("The idea", "Near misses happen far more often than crashes. Counting them, a traffic-conflict "
                    "study, shows where danger is building up within days of filming.", PAGE_COLORS["Method"]),
               unsafe_allow_html=True)
    c.markdown(card("What NearMiss does", "It watches ordinary video of an intersection, measures how close road users "
                    "came and how little time they had, and lists every close call for a person to check.",
                    PAGE_COLORS["Results"]), unsafe_allow_html=True)

    section("Who it is for", "groups", PAGE_COLORS["Detect"])
    a, b, c = st.columns(3)
    a.markdown(card("Residents and students", "Turn a phone or drone video of a crossing you worry about into evidence: "
                    "how often, where, and between whom.", PAGE_COLORS["Detect"]), unsafe_allow_html=True)
    b.markdown(card("Schools and community groups", "Show a school board or council a clear, checked report instead of "
                    "anecdotes.", PAGE_COLORS["Calibrate"]), unsafe_allow_html=True)
    c.markdown(card("Planners and engineers", "Screen many hours of footage quickly and compare before and after a "
                    "change, or alternative designs.", PAGE_COLORS["Proposed designs"]), unsafe_allow_html=True)

    section("What you get", "inventory_2", PAGE_COLORS["Results"])
    a, b, c, d = st.columns(4)
    for col, (t, x, k) in zip((a, b, c, d), [
            ("A map of close calls", "Where they happened, with a heat layer and named zones.", "Results"),
            ("Fair rates", "Per 100 vehicles and per hour, so busy and quiet clips compare fairly.", "Results"),
            ("Your own check", "Every flag reviewed by a person: real, false alarm or unsure.", "Event Review"),
            ("A shareable report", "Precision, limits and blurred pictures, ready to print.", "Report")]):
        col.markdown(card(t, x, PAGE_COLORS[k]), unsafe_allow_html=True)

    section("What it is not", "block", "#b91c1c")
    st.markdown(card("A screening tool, not a verdict", "NearMiss does not predict crashes, assign blame or certify an "
                     "intersection as safe. A flag is a measurement that needs a human look; zero flags with a "
                     "vehicles-only drone model means pedestrians were not measured, not that they are safe.",
                     "#b91c1c"), unsafe_allow_html=True)

    section("How it works, in five steps", "route", PAGE_COLORS["Method"])
    st.markdown(figures.figure(figures.pipeline()), unsafe_allow_html=True)
    cols = st.columns(len(STEPS))
    for i, (col, p) in enumerate(zip(cols, STEPS)):
        with col:
            st.markdown(card(f"{i + 1}. {p}", {"Detect": "Find and follow every road user in the video.",
                                               "Calibrate": "Click a shape of known size to turn pixels into metres.",
                                               "Results": "See the close calls, rates, zones and speeds.",
                                               "Event Review": "Decide which flags were real near misses.",
                                               "Report": "Check the numbers and share a report."}[p], PAGE_COLORS[p]),
                        unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    a, b, _ = st.columns([1, 1, 2])
    a.button("Start: 1. Detect", type="primary", icon=":material/arrow_forward:", on_click=go, args=("Detect",),
             width="stretch")
    if demo.demo_available():
        b.button("Try the demo first", icon=":material/science:", width="stretch",
                 on_click=lambda: st.session_state.update(rid=create_demo_run(), page="Results"))
    st.caption(":material/lock: Footage never leaves this computer. Faces and plates are pixelated in every video the "
               "app makes.")


@st.cache_resource(show_spinner="Preparing the pictures from the sample clip...")
def method_images() -> Optional[dict]:
    """Real pictures for the Method page, made with the app's own code on the bundled sample clip (scripted road
    users on a drawn intersection): detection boxes, the calibration shape, its bird's-eye view, a flagged near miss and
    the top-down map, plus a short clip with boxes. Made once per app start; None if the sample clip is missing."""
    if not demo.demo_available():
        return None
    from detector_tracker import DrawItem
    from engine.analysis import boxes_near
    from engine.conflicts import detect_conflicts as _dc
    from engine.trajectories import Detections as _Det, build_tracks as _bt

    t = demo.truth()
    cal = Calibration.from_points(t["calibration_points_px"], t["calibration_width_m"], t["calibration_length_m"],
                                  tuple(t["size"]))
    det = _Det(demo.scripted_detections())
    S = Settings()
    tracks, _ = _bt(det, cal, S)
    events = _dc(tracks, S)
    ev = events[0] if events else None
    t_show = ev["t_peak"] - 1.0 if ev else 11.0
    W, H = (int(v) for v in t["size"])

    # The sample's road users are small in a big frame: show the part of the picture where things happen
    near = [b for _, _, b in boxes_near(det, t_show, max_dt=3.0)] + [np.r_[np.min(t["calibration_points_px"], 0),
                                                                          np.max(t["calibration_points_px"], 0)]]
    x0, y0 = np.min([b[:2] for b in near], 0) - 90
    x1, y1 = np.max([b[2:] for b in near], 0) + 90
    w = max(x1 - x0, (y1 - y0) * 16 / 9)                   # keep a wide 16:9 view
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = min(w, W)
    h = min(w * 9 / 16, H)
    cx0 = int(np.clip(cx - w / 2, 0, W - w)) // 2 * 2
    cy0 = int(np.clip(cy - h / 2, 0, H - h)) // 2 * 2
    cw, ch = int(w) // 2 * 2, int(h) // 2 * 2

    def crop(img):
        return img[cy0:cy0 + ch, cx0:cx0 + cw]
    # 1. what detection sees: boxes and trails on one frame
    frame = grab_frame(str(demo.DEMO_VIDEO), t_show)
    items = []
    for tid, cls, box in boxes_near(det, t_show):
        rows = det.rows_of(tid)
        rows = rows[(det.t[rows] <= t_show) & (det.t[rows] >= t_show - 1.5)]
        trail = [((det.box[i][0] + det.box[i][2]) / 2, det.box[i][3]) for i in rows]
        items.append(DrawItem(tid, "person" if cls == "pedestrian" else str(cls), tuple(box), True, trail, 0.9))
    out = {"detect": rgb(crop(render_frame(frame, items, RenderOptions(max_width=10000), 0, t_show)))}
    # 2. calibration: the clicked rectangle on the camera view, and the road from above
    cam = frame.copy()
    pts = np.asarray(t["calibration_points_px"], dtype=np.int32)
    cv2.polylines(cam, [pts], True, (0, 220, 255), 3, cv2.LINE_AA)
    for k, (x, y) in enumerate(pts):
        cv2.circle(cam, (int(x), int(y)), 9, (0, 220, 255), -1)
        cv2.putText(cam, str(k + 1), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    out["cal_camera"] = rgb(crop(cam))
    out["cal_top"] = rgb(cal.preview(frame, margin=0.25))
    # 3. a flagged near miss and the map of all events
    if ev:
        out["event"] = rgb(event_frame(str(demo.DEMO_VIDEO), det, ev, "standard", True, emphasis=True, zoom=True))
        out["event_text"] = explain_event(ev)
    out["map"] = rgb(topdown_map(tracks, events, cal))
    # 4. a few seconds of video with the boxes drawn, as Detect makes it, cropped the same way
    clip = RUNS / ".method_clip_v2.mp4"
    if not clip.exists():
        from video_utils import ffmpeg_exe

        fps = cv2.VideoCapture(str(demo.DEMO_VIDEO)).get(cv2.CAP_PROP_FPS) or 10.0
        full = RUNS / ".method_clip_full.mp4"
        render_from_detections(str(demo.DEMO_VIDEO), det, str(full), fps, 1, RenderOptions(max_width=10000),
                               max(0.0, t_show - 3), t_show + 3)
        exe = ffmpeg_exe()
        if exe and full.exists():
            import subprocess

            subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(full), "-vf",
                            f"crop={cw}:{ch}:{cx0}:{cy0}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(clip)], timeout=120)
        if not clip.exists() and full.exists():
            full.replace(clip)
        full.unlink(missing_ok=True)
    out["clip"] = str(clip) if clip.exists() else None
    return out


def sample_note() -> None:
    st.caption(":material/movie: From the bundled sample clip: scripted road users on a drawn intersection, analysed "
               "with the app's own code. Real footage looks busier.")


def page_method():
    page_header("Method", "How NearMiss works",
                "From video to a reviewed list of close calls, step by step, and what the numbers can and cannot tell "
                "you.")
    st.markdown(figures.figure(figures.pipeline()), unsafe_allow_html=True)
    imgs = method_images()
    tabs = st.tabs([":material/videocam: Detection", ":material/straighten: Calibration",
                    ":material/insights: Indicators", ":material/filter_alt: Filters & severity",
                    ":material/verified: Validation", ":material/shield: Privacy & limits", ":material/notes: Full text"])
    with tabs[0]:
        section("Finding and following road users", "videocam", PAGE_COLORS["Detect"])
        a, b = st.columns(2)
        a.markdown(card("Street-level cameras", "A pretrained YOLOv8 model (COCO classes) finds people, bicycles, "
                        "motorcycles, cars, buses and trucks in every analysed frame.", PAGE_COLORS["Detect"]),
                   unsafe_allow_html=True)
        b.markdown(card("Drone footage", "An aerial model (YOLO11-OBB, DOTA) finds vehicles from above, with no "
                        "pedestrian class; the optional VisDrone model also finds pedestrians and bikes.",
                        PAGE_COLORS["Method"]), unsafe_allow_html=True)
        a, b = st.columns(2)
        a.markdown(card("Tracking", "ByteTrack links the boxes of one road user from frame to frame. A box that is "
                        "missed for a moment is carried forward, so tracks do not flicker.", PAGE_COLORS["Results"]),
                   unsafe_allow_html=True)
        b.markdown(card("Moving camera", "If the camera drifts or zooms, every frame is lined up with the first one "
                        "(ORB features and a RANSAC homography), so parked cars stay parked.",
                        PAGE_COLORS["Event Review"]), unsafe_allow_html=True)
        if imgs:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            l, r = st.columns(2)
            if imgs.get("clip"):
                l.video(imgs["clip"], loop=True, autoplay=True, muted=True)
                l.caption("A few seconds of tracked video: every road user boxed, with a trail of where it has been.")
            r.image(imgs["detect"], width="stretch",
                    caption="One frame: a box and an ID for each road user, kept the same from frame to frame.")
            sample_note()
    with tabs[1]:
        section("Turning pixels into metres", "straighten", PAGE_COLORS["Calibrate"])
        l, r = st.columns([3, 2])
        l.markdown(figures.figure(figures.calibration(), "Four clicks on a rectangle of known size give the mapping "
                                  "from the camera view to the road seen from above."), unsafe_allow_html=True)
        r.markdown(card("Why it matters", "Every distance, speed and time to collision comes from this mapping. A 10% "
                        "error in the scale is a 10% error in every speed.", PAGE_COLORS["Calibrate"]),
                   unsafe_allow_html=True)
        r.markdown("<div style='height:8px'></div>" + card(
            "Checking it", "Measure one or more other known distances. Until that check is done, or when it is more than "
            "10% off, or cars measure an implausible length, every severity is shown as provisional.",
            PAGE_COLORS["Report"]), unsafe_allow_html=True)
        if imgs:
            l, r = st.columns(2)
            l.image(imgs["cal_camera"], width="stretch", caption="The four clicked corners of a rectangle of known size "
                    "on the road.")
            r.image(imgs["cal_top"], width="stretch", caption="The same frame seen from above after calibration: the "
                    "rectangle is square again and lane lines are parallel.")
            sample_note()
    with tabs[2]:
        section("Measuring how close a call was", "insights", PAGE_COLORS["Results"])
        a, b = st.columns(2)
        a.markdown(figures.figure(figures.ttc(), "<b>Time to collision (TTC)</b>: seconds until two road users would "
                                  "reach a collision distance if both kept their speed and direction."),
                   unsafe_allow_html=True)
        b.markdown(figures.figure(figures.closest(), "<b>Closest approach</b>: the smallest distance over the whole "
                                  "encounter. The event picture, the number and the chart all use that moment."),
                   unsafe_allow_html=True)
        a, b = st.columns(2)
        a.markdown(figures.figure(figures.pet(), "<b>Post-encroachment time (PET, FHWA)</b>: from the first road user "
                                  "clearing a crossing point to the second reaching it. Optional flag."),
                   unsafe_allow_html=True)
        b.markdown(figures.figure(figures.drac(), "<b>DRAC</b>: the braking either road user would have needed to "
                                  "avoid the collision. Always shown; optional flag."), unsafe_allow_html=True)
        if imgs:
            section("A real flag from the sample clip", "flag", PAGE_COLORS["Event Review"])
            l, r = st.columns(2)
            if imgs.get("event") is not None:
                l.image(imgs["event"], width="stretch", caption="The moment of closest approach: the two road users "
                        "outlined, everything else dimmed.")
                l.caption(imgs["event_text"])
            r.image(imgs["map"], width="stretch", caption="Top-down map: paths (green pedestrians, purple bikes, blue "
                    "vehicles), flagged events as dots and a heat layer.")
            sample_note()
    with tabs[3]:
        section("Cutting false alarms, rating what is left", "filter_alt", PAGE_COLORS["Event Review"])
        c = st.columns(3)
        c[0].markdown(card("Detected twice", "Two tracks closer than half their widths are one road user seen twice, "
                           "not a collision.", PAGE_COLORS["Event Review"]), unsafe_allow_html=True)
        c[1].markdown(card("Too brief", "A time-to-collision flag must last at least 0.5 s; one noisy heading is not a "
                           "conflict.", PAGE_COLORS["Event Review"]), unsafe_allow_html=True)
        c[2].markdown(card("Passing a stopped car", "Driving past a parked car is only a conflict if the driver braked "
                           "hard or they really came close.", PAGE_COLORS["Event Review"]), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        c = st.columns(3)
        for col, (sev, text) in zip(c, [("candidate", "Worth a look."), ("moderate", "Closer, or less time to react."),
                                        ("serious", "Very close, or almost no time to react.")]):
            col.markdown(card(sev.capitalize(), text + " Set by the thresholds on Results; screening labels, not "
                              "verdicts.", SEV_HEX[sev]), unsafe_allow_html=True)
        st.caption("Two flagged pairs are merged only when they share a road user (an ID switch). Rates per hour use "
                   "the whole observation period; *flagged* is what the app found, *confirmed* what a reviewer labelled "
                   "real.")
    with tabs[4]:
        section("Checking the app against people", "verified", PAGE_COLORS["Report"])
        a, b = st.columns(2)
        a.markdown(card("Precision", "Of the flags you reviewed, the share that were real near misses. Labels belong to "
                        "exactly one event, matched by its two road users, then by time.", PAGE_COLORS["Report"]),
                   unsafe_allow_html=True)
        b.markdown(card("Recall (at most)", "Real near misses found divided by real ones found plus the ones you noticed "
                        "were missed. Only an upper bound: unnoticed misses cannot be counted.", PAGE_COLORS["Report"]),
                   unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>" + card(
            "Proposed designs", "For an intersection that is not built yet, simulated trajectories of each alternative go "
            "through the same conflict check (the FHWA SSAM approach). Compare alternatives with each other, never with "
            "video counts.", PAGE_COLORS["Proposed designs"]), unsafe_allow_html=True)
    with tabs[5]:
        section("Privacy and limits", "shield", PAGE_COLORS["Purpose"])
        l, r = st.columns([3, 2])
        l.markdown(figures.figure(figures.privacy(), "What is pixelated in the videos the app makes."),
                   unsafe_allow_html=True)
        r.markdown(card("Privacy", "Everything runs on this computer. Anything the detector misses is not blurred, so "
                        "check videos before sharing them; the original video can be deleted on Report.",
                        PAGE_COLORS["Purpose"]), unsafe_allow_html=True)
        r.markdown("<div style='height:8px'></div>" + card(
            "Limits", "One camera, a flat road, no night or heavy-rain testing. Models miss far or small road users. "
            "Every flag needs a human look. NearMiss does <b>not</b> predict crashes or certify an intersection as "
            "safe.", "#b91c1c"), unsafe_allow_html=True)
    with tabs[6]:
        method_full_text()


def method_full_text():
    st.markdown("""
**Pipeline.** For street-level cameras a pretrained YOLOv8 model (COCO) finds people, bicycles, motorcycles, cars,
buses and trucks. For drone footage the default is a pretrained aerial model (YOLO11-OBB, trained on the DOTA aerial
dataset), which finds vehicles seen from above and has no pedestrian or bicycle class (their conflicts show as *not
measured*); car or truck is decided by vehicle length. A YOLO11s fine-tuned on VisDrone also finds pedestrians and
bikes from above, but mistakes some street furniture for them: from above only, a pedestrian or bike that never moves
more than 3 of its own box sizes is left out, and Results shows each one so a real waiting person can be restored.
ByteTrack links detections into tracks.

**Moving camera.** If the camera drifts or zooms, every frame is lined up with the first one (ORB features + RANSAC
homography). Whether the camera moved is measured (more than 3 px), not assumed.

**Calibration.** Four clicks on a rectangle of known size (or, straight down, one known distance) map the point where
each box touches the road to metres. A second known distance checks the error. Until that check is done, or when it is
more than 10% off, or cars measure an implausible length, every severity is shown as **provisional**.

**Indicators.** *Time to collision (TTC):* seconds until two road users would reach a collision radius if both kept their
current velocity. *Closest approach:* the minimum centre-to-centre distance over the whole encounter (the flagged period
plus 1.5 s either side); the event picture, the number and the distance chart all use that same moment. *PET
(post-encroachment time, FHWA):* from the first road user's back clearing the point where the paths cross to the second
one's front reaching it; 0 if both occupied it at once. Reported, not used to flag.

**Severity.** Candidate, moderate and serious come from the thresholds on Results. They are screening labels.

**False-alarm filters.** One road user detected twice; TTC flags under 0.5 s; driving past a stopped road user without
braking or a close pass. Two flagged pairs are merged only when they share a road user (an ID switch); independent
encounters at the same place and time are kept apart.

**Rates.** Per hour uses the whole observation period. *Flagged* is what the app found; *confirmed* is what a reviewer
labelled real. Movements (approach and turn) come from each track's headings; signal phases are typed by the user.

**Validation.** A label belongs to exactly one event (matched by its two road users, then time) and one event has at most
one label. Imported label files are checked; bad entries are skipped and listed. A saved report is marked out of date
when anything it shows changes.

**Proposed designs.** For an intersection that is not built, simulated trajectories of each alternative go through the
same conflict check (the FHWA SSAM approach). Compare alternatives with each other, never with video counts.

**Limits.** One camera, flat road assumed, no night or heavy rain testing. Models miss far or small road users and can
confuse classes. Every flag needs a human look. This tool finds where road users came close; it does **not** predict
crashes or certify an intersection as safe.

**Privacy.** Street-level: heads and plate areas inside each detected box are pixelated; drone: each whole road user.
Anything the detector misses is not blurred. If blur was turned off at detection, every derived video says so.
""")


# ================================================================== HELP & CONTACT
INQUIRY_TYPES = {"question": "Ask a question", "problem": "Report a problem", "idea": "Suggest a feature",
                 "privacy": "Privacy or data request", "other": "Something else"}
FAQ = [
    ("Is my footage uploaded anywhere?",
     "No. Detection, analysis and reports all run on this computer. Nothing is sent unless you choose to email or "
     "share a file yourself. The original video stays in the app's runs folder; Report > Privacy can delete it."),
    ("How long will a video take?",
     "On a laptop with an Intel Arc GPU, about 6x faster than real time in Fast mode (a 60-min clip in ~10 min) and "
     "about 15x in Fastest mode. On a computer without a graphics chip it is slower than real time. Keep a laptop "
     "plugged in: on a low battery Windows slows the GPU down a lot."),
    ("Which speed mode should I use?",
     "Fast for anything you report: it finds exactly the same road users as Full, it just does not draw the tracked "
     "video of the whole clip. Fastest skips more frames and misses some road users, especially pedestrians; use it "
     "to scan long footage first."),
    ("Why does drone footage show no pedestrian conflicts?",
     "The standard aerial model only knows vehicles. Zero pedestrian conflicts there means 'not measured', not "
     "'safe'. The VisDrone model (nearmiss-visdrone.pt) finds pedestrians and bikes from above."),
    ("Why are my severities marked provisional?",
     "Every distance and speed depends on the calibration. Until a second known distance has been checked on "
     "Calibrate (or when it is more than 10% off), severities are provisional."),
    ("What do TTC, PET and DRAC mean?",
     "TTC (time to collision): seconds until two road users would touch if neither changed course. PET (post-"
     "encroachment time): the gap between one leaving a crossing point and the other reaching it. DRAC: how hard "
     "someone would have needed to brake to avoid the collision. The Method page explains each."),
    ("A detection stopped or my computer turned off. Is the work lost?",
     "No. Progress is saved every 30 s of video. The unfinished run is listed at the top of Detect with Resume."),
    ("Does a flagged event mean a crash almost happened?",
     "Not necessarily. Each flag is a measurement that needs a human look; that is what Event Review is for. "
     "Precision on the Report page says how many of your reviewed flags were real."),
]


def load_prefs() -> dict:
    return read_json(PREFS_FILE, {}) or {}


def save_prefs(p: dict) -> None:
    PREFS_FILE.write_text(json.dumps(p, indent=1))


def app_version() -> str:
    """The git commit the app runs from (best effort; 'unknown' outside a git checkout)."""
    try:
        head = (Path(__file__).parent / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = Path(__file__).parent / ".git" / head.split(" ", 1)[1]
            return f"{head.rsplit('/', 1)[-1]} @ {ref.read_text().strip()[:7]}" if ref.exists() else head
        return head[:7]
    except Exception:
        return "unknown"


def diagnostics() -> dict:
    """What helps someone answer a problem report. No footage, no file names, no labels."""
    from detector_tracker import intel_devices as _intel

    d = {"app_version": app_version(), "python": platform.python_version(), "os": platform.platform(terse=True),
         "streamlit": st.__version__, "detection_device": device_label(pick_device("auto")),
         "intel_devices": _intel(), "page_before": st.session_state.get("_last_page")}
    if rid and meta:
        d["recording"] = {k: meta.get(k) for k in ("view", "model_used", "device_used", "stride", "fps", "width",
                                                    "height", "duration_s", "detection_rows", "seconds", "resumed")}
        d["recording"]["calibrated"] = (run_dir(rid) / "calibration.json").exists()
    return d


def contact_email() -> str:
    try:
        sec = st.secrets.get("contact_email", "")
    except Exception:
        sec = ""
    return load_prefs().get("contact_email") or os.environ.get("NEARMISS_CONTACT_EMAIL", "") or sec


def inquiry_text(q: dict) -> str:
    lines = [f"NearMiss inquiry: {INQUIRY_TYPES.get(q['type'], q['type'])}", f"From: {q.get('name') or '(no name)'}"
             + (f" <{q['email']}>" if q.get("email") else ""), f"Sent: {q['created_text']}", "", q["message"]]
    if q.get("diagnostics"):
        lines += ["", "--- diagnostics ---", json.dumps(q["diagnostics"], indent=1)]
    return "\n".join(lines)


def page_help():
    page_header("Help & contact", "Help, questions & feedback",
                "Quick answers, a way to ask the team something or report a problem, and your preferences.")
    tabs = st.tabs([":material/rocket_launch: Quick start", ":material/quiz: FAQ", ":material/mail: Contact the team",
                    ":material/inbox: Inquiries", ":material/settings: Preferences"])
    with tabs[0]:
        steps = [("videocam", "Detect", "Upload a clip (or several) and press Run. Fast mode is a good default."),
                 ("straighten", "Calibrate", "Click four corners of a rectangle you know the size of, then check a "
                                             "second known distance."),
                 ("analytics", "Results", "See the flagged events on the map, the rates, zones, speeds and movements."),
                 ("fact_check", "Event Review", "Watch each flag and mark it real, false alarm or unsure."),
                 ("description", "Report", "Precision, a second reviewer, and a report to share.")]
        cols = st.columns(len(steps))
        for col, (icon, name, text) in zip(cols, steps):
            with col.container(border=True):
                st.markdown(f"<div style='font-family:\"Material Symbols Rounded\";font-size:28px;color:#1f5fbf'>"
                            f"{icon}</div><b>{name}</b>", unsafe_allow_html=True)
                st.caption(text)
                st.button("Open", key=f"qs_{name}", on_click=go, args=(name,), width="stretch")
        st.caption("No footage yet? Click *Load synthetic demo* in the sidebar to try every page with a scripted scene.")
    with tabs[1]:
        search = st.text_input("Search the questions", placeholder="e.g. battery, drone, provisional", key="faq_q")
        shown = [(q, a) for q, a in FAQ if not search or search.lower() in (q + a).lower()]
        for q, a in shown:
            with st.expander(q):
                st.write(a)
        if not shown:
            st.info("Nothing matches. Ask the team on the *Contact the team* tab.")
    with tabs[2]:
        contact_form()
    with tabs[3]:
        inbox_ui()
    with tabs[4]:
        preferences_ui()


def contact_form() -> None:
    st.write("Questions, problems, ideas, or a request about data you appear in. Your message is **saved on this "
             "computer**; nothing is sent until you choose to email it or share the file.")
    with st.form("inquiry", clear_on_submit=True):
        a, b = st.columns(2)
        kind = a.selectbox("What is it about?", list(INQUIRY_TYPES), format_func=INQUIRY_TYPES.get)
        name = b.text_input("Your name (optional)")
        email = a.text_input("Your email, if you want an answer (optional)")
        subject = b.text_input("Subject", max_chars=120)
        message = st.text_area("Message", height=160, max_chars=4000,
                               placeholder="What happened, what you expected, and which recording or page it was on.")
        diag = st.checkbox("Attach technical details (app version, device, settings of the open recording; no "
                           "footage, no file names)", value=kind == "problem")
        sent = st.form_submit_button("Save message", type="primary", icon=":material/send:")
    if sent:
        if not message.strip():
            st.error("Please write a message.")
            return
        if email and "@" not in email:
            st.error("That email address does not look right.")
            return
        INQUIRIES.mkdir(exist_ok=True)
        now = time.time()
        q = {"id": time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + uuid.uuid4().hex[:4], "type": kind,
             "name": name.strip(), "email": email.strip(), "subject": subject.strip() or INQUIRY_TYPES[kind],
             "message": message.strip(), "created": now,
             "created_text": time.strftime("%Y-%m-%d %H:%M", time.localtime(now)), "status": "open",
             "diagnostics": diagnostics() if diag else None}
        write_json(INQUIRIES / f"{q['id']}.json", q)
        st.session_state["_last_inquiry"] = q["id"]
    last = st.session_state.get("_last_inquiry")
    q = read_json(INQUIRIES / f"{last}.json") if last else None
    if q:
        st.success("Saved. To get it to the team, email it or send the file:")
        a, b = st.columns(2)
        to = contact_email()
        if to:
            from urllib.parse import quote

            a.link_button(f"Email it to {to}", f"mailto:{to}?subject={quote('NearMiss: ' + q['subject'])}"
                                                f"&body={quote(inquiry_text(q)[:1800])}", icon=":material/mail:")
        else:
            a.caption("No team email is set yet: add one under *Preferences*.")
        b.download_button("Download as a text file", inquiry_text(q), f"nearmiss_inquiry_{q['id']}.txt",
                          icon=":material/download:")


def inbox_ui() -> None:
    """Messages saved on this computer, newest first, with a status so the team can keep track of them."""
    items = sorted([read_json(p, {}) for p in INQUIRIES.glob("*.json")] if INQUIRIES.exists() else [],
                   key=lambda q: -q.get("created", 0))
    items = [q for q in items if q.get("id")]
    if not items:
        st.info("No messages saved on this computer yet.")
        return
    a, b = st.columns([2, 1])
    show = a.segmented_control("Show", ["open", "answered", "all"], default="open", key="inbox_show")
    open_n = sum(q.get("status") == "open" for q in items)
    b.metric("Open", open_n)
    for q in items:
        if show and show != "all" and q.get("status") != show:
            continue
        icon = {"open": ":material/mark_email_unread:", "answered": ":material/mark_email_read:"}.get(q["status"], "")
        with st.expander(f"{q['created_text']} · {INQUIRY_TYPES.get(q['type'], q['type'])} · {q['subject']}",
                         icon=icon or None):
            st.write(q["message"])
            st.caption(f"From {q.get('name') or '(no name)'}" + (f" · {q['email']}" if q.get("email") else ""))
            if q.get("diagnostics"):
                st.json(q["diagnostics"], expanded=False)
            c1, c2, c3 = st.columns(3)
            if q["status"] == "open" and c1.button("Mark answered", key=f"ans_{q['id']}"):
                q["status"] = "answered"
                write_json(INQUIRIES / f"{q['id']}.json", q)
                st.rerun()
            if q["status"] != "open" and c1.button("Reopen", key=f"reo_{q['id']}"):
                q["status"] = "open"
                write_json(INQUIRIES / f"{q['id']}.json", q)
                st.rerun()
            c2.download_button("Download", inquiry_text(q), f"nearmiss_inquiry_{q['id']}.txt", key=f"dl_{q['id']}")
            if c3.button("Delete", key=f"del_{q['id']}"):
                (INQUIRIES / f"{q['id']}.json").unlink(missing_ok=True)
                st.rerun()


def preferences_ui() -> None:
    p = load_prefs()
    st.write("Defaults for this computer. They are saved in `preferences.json` next to the app.")
    with st.form("prefs"):
        a, b = st.columns(2)
        speed = a.selectbox("Default speed mode", list(SPEED_MODES), index=list(SPEED_MODES).index(
            p.get("speed", "fast")) if p.get("speed", "fast") in SPEED_MODES else 1,
            format_func=lambda k: SPEED_MODES[k]["label"])
        devices = ["auto", "cpu", "intel:gpu", "intel:npu", "cuda:0", "mps"]
        device = b.selectbox("Default device", devices, index=devices.index(p.get("device", "auto"))
                             if p.get("device", "auto") in devices else 0,
                             format_func=lambda d: "auto (fastest found)" if d == "auto" else device_label(d))
        tips = a.toggle("Show beginner tips on Detect", p.get("tips", True))
        email = b.text_input("Team email for inquiries", p.get("contact_email", ""),
                             help="Messages saved on Contact the team get an 'Email it' button addressed here.")
        if st.form_submit_button("Save preferences", type="primary"):
            if email and "@" not in email:
                st.error("That email address does not look right.")
            else:
                save_prefs({"speed": speed, "device": device, "tips": tips, "contact_email": email.strip()})
                st.success("Saved. They apply the next time you open Detect.")


{"Detect": page_detect, "Calibrate": page_calibrate, "Results": page_results, "Event Review": page_review,
 "Report": page_report, "Proposed designs": page_design, "Method": page_method, "Help & contact": page_help,
 "Purpose": page_purpose, "Home": page_home}[page]()
step_pager(page)
st.markdown("<div class='nm-footer'>NearMiss · a screening tool for traffic conflicts, not a crash predictor or a "
            "safety certification · everything runs on this computer</div>", unsafe_allow_html=True)
