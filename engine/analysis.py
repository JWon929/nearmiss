"""Glue between the tracker's CSV and the conflict engine, plus the pictures the app shows.

Everything here works on files, so it is easy to test without Streamlit.
"""
from __future__ import annotations

import csv
import html
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .calibration import Calibration
from .config import Settings, KIND_LABEL
from .conflicts import detect_conflicts
from .privacy import blur_detections
from .trajectories import Detections, Track, build_tracks

# The tracker writes COCO names; the engine wants "pedestrian".
CLASS_ALIASES = {"person": "pedestrian"}

SEV_COLOR = {"candidate": (0, 200, 255), "moderate": (0, 140, 255), "serious": (60, 60, 235)}  # BGR
GROUP_COLOR = {"ped": (90, 200, 90), "bike": (180, 80, 160), "veh": (200, 130, 70)}   # BGR: green, purple, blue


# ------------------------------------------------------------------ loading
def load_tracker_csv(path) -> Detections:
    """Read the CSV written by the Detect page (frame,time_s,track_id,class_name,confidence,x1..y2,...)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            cls = CLASS_ALIASES.get(r["class_name"], r["class_name"])
            rows.append((int(r["frame"]), float(r["time_s"]), int(r["track_id"]), cls, float(r["confidence"]),
                         float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
    return Detections(rows)


def analyze(det: Detections, calib: Calibration, S: Settings,
            motion: Optional[Dict[int, np.ndarray]] = None,
            roi: Optional[np.ndarray] = None, observation_s: Optional[float] = None) -> Tuple[List[Track], Dict, List[Dict]]:
    """`observation_s`: how long the camera watched (the recording's duration). Rates per hour use it; without it
    they fall back to the time span of the detections."""
    tracks, stats = build_tracks(det, calib, S, motion, roi)
    events = detect_conflicts(tracks, S, stats)
    stats["analysed_tracks"] = len(tracks)
    if observation_s is None and len(det):
        observation_s = float(det.t.max() - det.t.min())
    stats.update(exposure(tracks, events, 1.0 / S.analysis_hz, observation_s))
    return tracks, stats, events


def exposure(tracks: Sequence[Track], events: Sequence[Dict], dt: float, observation_s: Optional[float] = None) -> Dict:
    """How much traffic there was to have conflicts with, so busy and quiet clips can be compared.

    Raw event counts grow with traffic: twice the cars, about twice the events, with no change in risk.
    So events are also given per 100 vehicles and per 1,000 road-user pairs that were on the road at the
    same time (only the pair kinds the engine checks: ped-veh, bike-veh, veh-veh).

    Events per hour divide by the OBSERVATION period (how long the camera watched), not by the time between the
    first and last kept track, which leaves out quiet stretches and so inflates the rate."""
    from .conflicts import pair_kind

    ids = {}
    for t in tracks:                      # one road user can be split into segments; count it once
        ids.setdefault(t.group, set()).add(t.id)
    counts = {g: len(s) for g, s in ids.items()}
    pairs = {"ped-veh": 0, "bike-veh": 0, "veh-veh": 0}
    seen = set()
    for i, A in enumerate(tracks):
        for B in tracks[i + 1:]:
            kind = pair_kind(A.group, B.group)
            key = tuple(sorted((A.id, B.id)))
            if kind is None or A.id == B.id or key in seen:
                continue
            if min(A.k1, B.k1) - max(A.k0, B.k0) >= 2:     # on the road together for a few samples
                seen.add(key)
                pairs[kind] += 1
    n_pairs, n_veh = sum(pairs.values()), counts.get("veh", 0)
    if observation_s is None:
        observation_s = (max((t.k1 for t in tracks), default=0) - min((t.k0 for t in tracks), default=0)) * dt
    hours = observation_s / 3600.0
    by_kind = {k: sum(1 for e in events if e["kind"] == k) for k in pairs}
    return {
        "road_users": counts,
        "pairs_together": pairs,
        "events_per_100_vehicles": round(100.0 * len(events) / n_veh, 2) if n_veh else None,
        "events_per_1000_pairs": round(1000.0 * len(events) / n_pairs, 2) if n_pairs else None,
        "events_per_1000_pairs_by_kind": {k: round(1000.0 * by_kind[k] / pairs[k], 2) if pairs[k] else None
                                          for k in pairs},
        "events_per_hour": round(len(events) / hours, 1) if hours > 0 else None,
        "observation_s": round(float(observation_s), 1),
    }


# ------------------------------------------------------------------ boxes at a time
def boxes_near(det: Detections, t: float, max_dt: float = 0.5) -> List[Tuple[int, str, np.ndarray]]:
    """For every track, its observed box nearest in time to t (within max_dt)."""
    out = []
    for tid in det.track_ids():
        idx = det.rows_of(tid)
        j = idx[np.argmin(np.abs(det.t[idx] - t))]
        if abs(det.t[j] - t) <= max_dt:
            out.append((tid, str(det.cls[j]), det.box[j]))
    return out


def event_frame(video_path: str, det: Detections, ev: Dict, privacy_level: str = "standard",
                blur: bool = True, t: Optional[float] = None, emphasis: bool = False, zoom: bool = False) -> np.ndarray:
    """Full-resolution frame at the moment of closest approach (or at time `t`): road users blurred, the two involved
    ones outlined. `emphasis` dims everything except the two road users; `zoom` crops to them."""
    from video_utils import grab_frame

    t = ev["t_peak"] if t is None else t
    img = grab_frame(video_path, t)
    near = boxes_near(det, t)
    if blur:
        img = blur_detections(img, [(c, b) for _, c, b in near], privacy_level)
    col = SEV_COLOR.get(ev["severity"], (0, 200, 255))
    centers, mine = [], []
    for tid, c, b in near:
        if tid in (ev["a"]["track"], ev["b"]["track"]):
            mine.append(b)
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
    if len(centers) == 2:
        cv2.line(img, centers[0], centers[1], col, 2, cv2.LINE_AA)
    if mine and (emphasis or zoom):
        h, w = img.shape[:2]
        b = np.array(mine)
        side = max(b[:, 2].max() - b[:, 0].min(), b[:, 3].max() - b[:, 1].min())
        pad = max(0.6 * side, 0.08 * max(w, h))
        x1, y1 = int(max(0, b[:, 0].min() - pad)), int(max(0, b[:, 1].min() - pad))
        x2, y2 = int(min(w, b[:, 2].max() + pad)), int(min(h, b[:, 3].max() + pad))
        if emphasis:
            dim = (img * 0.45).astype(img.dtype)
            dim[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            img = dim
        if zoom:
            img = img[y1:y2, x1:x2].copy()
            s = 900.0 / max(img.shape[:2])
            if s > 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            return img
    label = f"{ev['id']}  {ev['kind_label']}  closest {ev['d_min_m']} m"
    if ev.get("ttc_min_s") is not None:
        label += f"  TTC {ev['ttc_min_s']} s"
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def cut_clip(src_video: str, out_path: str, t0: float, t1: float, max_width: Optional[int] = None) -> bool:
    """Cut [t0, t1] out of a video as browser-playable H.264 (no wider than `max_width`). Returns False if ffmpeg
    is missing or fails."""
    from video_utils import ffmpeg_exe

    exe = ffmpeg_exe()
    if not exe:
        return False
    t0 = max(0.0, t0)
    cmd = [exe, "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-i", src_video, "-t", f"{max(0.5, t1 - t0):.3f}",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24" if not max_width else "28",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    if max_width:
        cmd[-1:-1] = ["-vf", f"scale='min({int(max_width)},iw)':-2"]
    try:
        return subprocess.run(cmd, timeout=120).returncode == 0 and Path(out_path).exists()
    except Exception:
        return False


# ------------------------------------------------------------------ top-down map
def topdown_map(tracks: Sequence[Track], events: Sequence[Dict], calib: Calibration,
                size: int = 900, heat: bool = True, show_tracks: bool = True,
                zones: Sequence[Tuple[str, np.ndarray, Tuple[int, int, int]]] = (), highlight: Optional[str] = None,
                markers_out: Optional[Dict[str, Tuple[int, int]]] = None) -> np.ndarray:
    """Bird's-eye picture: trajectories, a heat layer of events, event markers, a metre scale bar.
    `zones` are (name, polygon in map metres, BGR colour) outlines drawn under everything else. `highlight` rings one
    event (by id); `markers_out` receives each event's marker position in pixels, so a click can be matched to it."""
    pts = [t.xy for t in tracks if len(t.xy)]
    pts += [np.array([[e["x"], e["y"]] for e in events])] if events else []
    corners, _ = calib.img_to_map(calib.image_points)
    pts.append(np.asarray(corners))
    pts += [np.asarray(p) for _, p, _ in zones]
    allp = np.vstack(pts)
    lo, hi = np.percentile(allp, 1, axis=0) - 3.0, np.percentile(allp, 99, axis=0) + 3.0
    span = np.maximum(hi - lo, 1.0)
    s = (size - 40) / float(max(span))
    w, h = int(span[0] * s) + 40, int(span[1] * s) + 40

    def px(p):
        p = np.atleast_2d(p)
        # map metres have y pointing DOWN like the screen (calibration.rot keeps 'up on screen' = -y), so y is not
        # flipped here: flipping it made the map a mirror image of the camera view
        return np.c_[20 + (p[:, 0] - lo[0]) * s, 20 + (p[:, 1] - lo[1]) * s]

    img = np.full((h, w, 3), 245, np.uint8)
    for name, poly, col in zones:
        q = px(poly).astype(np.int32)
        tint = img.copy()
        cv2.fillPoly(tint, [q], col)
        img = cv2.addWeighted(tint, 0.18, img, 0.82, 0)
        cv2.polylines(img, [q], True, col, 2, cv2.LINE_AA)
        c = q.mean(axis=0).astype(int)
        cv2.putText(img, name, (int(c[0]) - 4 * len(name), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)
    cv2.polylines(img, [px(corners).astype(np.int32)], True, (170, 170, 170), 2, cv2.LINE_AA)
    if show_tracks:
        for t in tracks:
            if len(t.xy) < 2:
                continue
            cv2.polylines(img, [px(t.xy).astype(np.int32)], False, GROUP_COLOR.get(t.group, (150, 150, 150)), 1,
                          cv2.LINE_AA)
    if heat and events:
        acc = np.zeros((h, w), np.float32)
        for e in events:
            x, y = px([e["x"], e["y"]])[0]
            cv2.circle(acc, (int(x), int(y)), 1, float(1 + e["severity_level"]), -1)
        acc = cv2.GaussianBlur(acc, (0, 0), max(8.0, 1.5 * s))
        if acc.max() > 0:
            hm = cv2.applyColorMap((255 * acc / acc.max()).astype(np.uint8), cv2.COLORMAP_INFERNO)
            a = (acc / acc.max() * 0.75)[..., None]
            img = (img * (1 - a) + hm * a).astype(np.uint8)
    for e in events:
        x, y = px([e["x"], e["y"]])[0]
        if markers_out is not None:
            markers_out[e["id"]] = (int(x), int(y))
        if e["id"] == highlight:
            cv2.circle(img, (int(x), int(y)), 15, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 7, SEV_COLOR.get(e["severity"], (0, 200, 255)), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, e["id"], (int(x) + 9, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1,
                    cv2.LINE_AA)
    bar_m = 5 if span[0] < 60 else 10
    x0, y0 = 24, h - 8
    cv2.line(img, (x0, y0), (int(x0 + bar_m * s), y0), (40, 40, 40), 3)
    cv2.putText(img, f"{bar_m} m", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def distance_series(tracks: Sequence[Track], ev: Dict, pad_s: float = 1.5) -> Dict[str, List[float]]:
    """Distance between the two road users of an event over time (for the chart on the review page)."""
    by = {(t.id, t.seg): t for t in tracks}
    A, B = by.get((ev["a"]["track"], ev["a"]["seg"])), by.get((ev["b"]["track"], ev["b"]["seg"]))
    if A is None or B is None:
        return {"t": [], "distance_m": []}
    t0, t1 = ev["t_start"] - pad_s, ev["t_end"] + pad_s
    ts, ds = [], []
    for k, tt in enumerate(A.t):
        if tt < t0 or tt > t1:
            continue
        j = np.where(np.isclose(B.t, tt, atol=1e-6))[0]
        if len(j):
            ts.append(round(float(tt), 2))
            ds.append(round(float(np.linalg.norm(A.xy[k] - B.xy[j[0]])), 2))
    return {"t": ts, "distance_m": ds}


# ------------------------------------------------------------------ report
def events_table(events: Sequence[Dict]) -> List[Dict]:
    rows = []
    for e in events:
        rows.append({
            "id": e["id"], "time_s": e["t_peak"], "type": e["kind_label"], "severity": e["severity"],
            "closest_m": e["d_min_m"], "ttc_min_s": e["ttc_min_s"], "pet_s": e["pet_s"],
            "below_3ft": e.get("below_3ft"), "zone": e.get("zone", ""), "why": "; ".join(e.get("reasons", [])),
            "label": e.get("label") or "",
        })
    return rows


def build_report_html(title: str, meta: Dict, calib_info: Dict, S: Settings, events: Sequence[Dict],
                      validation: Dict, ai_disclosure: str, map_png_b64: str,
                      frames_b64: Optional[Dict[str, str]] = None, notes: Sequence[str] = (),
                      exposure: Optional[Dict] = None, info: Optional[Dict] = None, explain=None,
                      speeds: Optional[Dict] = None, zone_rows: Optional[Sequence[Dict]] = None,
                      sections: Sequence[Tuple[str, str]] = (), planner: bool = False) -> str:
    """One self-contained HTML page (print it to PDF from the browser). `notes` are blind spots to state up front,
    `info` the recording details sheet, `explain` turns an event into a plain-language sentence. `sections` are
    extra (title, html) blocks shown right after the notes (summary, movements, phases). `planner` puts the plain
    summary first and moves the raw settings into an appendix; an empty `ai_disclosure` leaves that section out."""
    esc = html.escape
    frames_b64 = frames_b64 or {}
    info_rows = "".join(f"<tr><th>{esc(k.replace('_', ' '))}</th><td>{esc(str(v))}</td></tr>"
                        for k, v in (info or {}).items() if v not in ("", None, 0))
    info_html = f"<h2>Recording details</h2><table>{info_rows}</table>" if info_rows else ""
    zones_html = ""
    if zone_rows:
        cols = ["zone", "type", "pedestrians", "bicycles", "vehicles", "events", "serious", "moderate", "candidate",
                "events_per_100_users"]
        zones_html = ("<h2>Zones</h2><table><tr>" + "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols) + "</tr>"
                      + "".join("<tr>" + "".join(f"<td>{esc(str(r[c] if r[c] is not None else 'n/a'))}</td>"
                                                 for c in cols) + "</tr>" for r in zone_rows) + "</table>")
    speed_html = ""
    if speeds and speeds.get("vehicles"):
        mph = lambda v: f"{v * 2.23694:.0f} mph"
        speed_html = (f"<h2>Vehicle speeds</h2><p>{speeds['vehicles']} vehicles measured. Median {mph(speeds['median_mps'])}, "
                      f"85th percentile {mph(speeds['p85_mps'])}, fastest {mph(speeds['max_mps'])}."
                      + (f" {speeds['share_over_limit']:.0%} over the posted limit of {mph(speeds['limit_mps'])}."
                         if speeds.get("limit_mps") else "") + "</p>")
    notes_html = "".join(f"<p class='warn'><b>Limitation:</b> {esc(n)}</p>" for n in notes)
    kind_rows = "".join(
        f"<tr><td>{esc(KIND_LABEL.get(k, k))}</td><td>{v['flagged']}</td><td>{v['tp']}</td><td>{v['fp']}</td>"
        f"<td>{v['unsure']}</td><td>{v['unreviewed']}</td>"
        f"<td>{'n/a' if v['precision'] is None else format(v['precision'], '.0%')}</td></tr>"
        for k, v in validation.get("by_kind", {}).items())
    kind_table = ("<table><tr><th>Type</th><th>Flagged</th><th>Real</th><th>False alarms</th><th>Unsure</th>"
                  f"<th>Not reviewed</th><th>Precision</th></tr>{kind_rows}</table>") if kind_rows else ""
    unreviewed = (f"<p class='warn'>{validation['unreviewed']} of {validation['events_flagged']} flagged events were not "
                  "reviewed; precision covers reviewed events only.</p>") if validation.get("unreviewed") else ""
    exposure_html = (f"<h2>Traffic exposure</h2><p>Event counts grow with traffic, so rates are given to compare "
                     f"clips.</p><pre>{esc(json.dumps(exposure, indent=1))}</pre>") if exposure else ""
    rows = ""
    for r in events_table(events):
        rows += "<tr>" + "".join(f"<td>{esc(str(r[k]))}</td>" for k in
                                 ["id", "time_s", "type", "severity", "closest_m", "ttc_min_s", "pet_s", "zone",
                                  "label"]) + "</tr>"
    cards = ""
    for e in events:
        img = frames_b64.get(e["id"])
        cards += (f"<div class='card'><b>{esc(e['id'])}</b> {esc(e['kind_label'])}, {esc(e['severity'])}, "
                  f"t = {e['t_peak']} s<br>closest {e['d_min_m']} m, TTC {e['ttc_min_s']} s<br>"
                  + (f"{esc(explain(e))}<br>" if explain else "")
                  + f"<small>{esc('; '.join(e.get('reasons', [])))}</small>"
                  + (f"<br><img src='data:image/jpeg;base64,{img}'>" if img else "") + "</div>")
    v = json.dumps(validation, indent=1)
    extra_html = "".join(f"<h2>{esc(t)}</h2>{body}" for t, body in sections)
    technical = (f"<h2>Recording</h2><pre>{esc(json.dumps(meta, indent=1))}</pre>"
                 f"<h2>Calibration</h2><pre>{esc(json.dumps(calib_info, indent=1))}</pre>"
                 f"<h2>Thresholds used</h2><pre>{esc(json.dumps(S.to_dict(), indent=1))}</pre>")
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
.card{{border:1px solid #ccc;border-radius:8px;padding:10px;margin:10px 0;break-inside:avoid}}
img{{max-width:100%;border-radius:6px}}pre{{background:#f4f4f4;padding:10px;overflow:auto}}
.warn{{background:#fff4e0;border-left:4px solid #e08a00;padding:8px 12px}}</style></head><body>
<h1>{esc(title)}</h1>
<p><b>Screening report, not a safety certification.</b> Flags show where road users came close or crossed paths in
this one recording. They do not predict crashes. Footage is a single fixed camera; distances are only as good as the
calibration below.</p>{notes_html}{extra_html}
{info_html}{"" if planner else technical}{exposure_html}{speed_html}
<h2>Top-down map</h2><img src='data:image/png;base64,{map_png_b64}'>{zones_html}
<h2>Events ({len(events)})</h2><table><tr><th>ID</th><th>t (s)</th><th>Type</th><th>Severity</th><th>Closest (m)</th>
<th>Min TTC (s)</th><th>PET (s)</th><th>Zone</th><th>My label</th></tr>{rows}</table>{cards}
<h2>Validation (my own review of the flags)</h2>{unreviewed}{kind_table}{"" if planner else f"<pre>{esc(v)}</pre>"}
{f"<h2>Appendix: settings used</h2>{technical}" if planner else ""}
{f"<h2>AI use disclosure</h2><p>{esc(ai_disclosure).replace(chr(10), '<br>')}</p>" if ai_disclosure.strip() else ""}
</body></html>"""
