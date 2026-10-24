"""
Honest validation: compare what NearMiss flagged with what a human decides.

Workflow (Event Review screen):
  1. Watch every flagged event clip. Mark it  true positive / false positive / unsure.
  2. While watching the clip, note REAL conflicts the system missed ("missed" list).
  3. The numbers below are computed ONLY from your labels. Nothing is invented.

Labels are stored against (kind, t_peak) rather than event number, so they survive
re-running the analysis with different thresholds.

Interpretation caveats (put them in your submission):
  * precision = TP / (TP + FP) over the events you reviewed
  * recall    = TP / (TP + missed) is an UPPER BOUND on true recall: you can only
    list misses you happened to notice.
  * one clip is an anecdote, not a benchmark. Say how many events and clips you reviewed.
"""
from __future__ import annotations

from typing import Dict, List, Optional

MATCH_WINDOW_S = 1.0


def find_label(labels: List[Dict], kind: str, t_peak: float) -> Optional[Dict]:
    best, best_dt = None, MATCH_WINDOW_S + 1
    for lab in labels:
        dt = abs(lab["t_peak"] - t_peak)
        if lab["kind"] == kind and dt <= MATCH_WINDOW_S and dt < best_dt:
            best, best_dt = lab, dt
    return best


def apply_labels(events: List[Dict], store: Dict) -> None:
    labels = store.get("events", [])
    for e in events:
        lab = find_label(labels, e["kind"], e["t_peak"])
        e["label"] = lab["label"] if lab else None
        e["label_note"] = lab.get("note", "") if lab else ""


def set_label(store: Dict, kind: str, t_peak: float, label: Optional[str], note: str = "") -> Dict:
    labels = store.setdefault("events", [])
    lab = find_label(labels, kind, t_peak)
    if label in (None, "", "clear"):
        if lab:
            labels.remove(lab)
        return store
    if label not in ("tp", "fp", "unsure"):
        raise ValueError("label must be tp, fp, unsure or clear")
    if lab:
        lab.update({"label": label, "note": note, "t_peak": t_peak})
    else:
        labels.append({"kind": kind, "t_peak": t_peak, "label": label, "note": note})
    return store


def summarize(events: List[Dict], store: Dict) -> Dict:
    labels = store.get("events", [])
    missed = store.get("missed", [])
    n = len(events)
    tp = fp = unsure = 0
    by_sev: Dict[str, Dict[str, int]] = {}
    by_kind: Dict[str, Dict] = {}
    matched = set()
    for e in events:
        lab = find_label(labels, e["kind"], e["t_peak"])
        s = by_sev.setdefault(e["severity"], {"tp": 0, "fp": 0, "unsure": 0, "unreviewed": 0})
        k = by_kind.setdefault(e["kind"], {"flagged": 0, "tp": 0, "fp": 0, "unsure": 0, "unreviewed": 0})
        k["flagged"] += 1
        if lab is None:
            s["unreviewed"] += 1
            k["unreviewed"] += 1
            continue
        matched.add(id(lab))
        if lab["label"] == "tp":
            tp += 1
            s["tp"] += 1
            k["tp"] += 1
        elif lab["label"] == "fp":
            fp += 1
            s["fp"] += 1
            k["fp"] += 1
        else:
            unsure += 1
            s["unsure"] += 1
            k["unsure"] += 1
    for k in by_kind.values():
        k["precision"] = round(k["tp"] / (k["tp"] + k["fp"]), 3) if k["tp"] + k["fp"] else None
    reviewed = tp + fp
    precision = tp / reviewed if reviewed else None
    recall_ub = tp / (tp + len(missed)) if (tp + len(missed)) else None
    return {
        "events_flagged": n,
        "reviewed": reviewed,
        "unsure": unsure,
        "unreviewed": n - reviewed - unsure,
        "tp": tp,
        "fp": fp,
        "missed": len(missed),
        "precision": None if precision is None else round(precision, 3),
        "recall_upper_bound": None if recall_ub is None else round(recall_ub, 3),
        "by_severity": by_sev,
        "by_kind": by_kind,
        "orphaned_labels": sum(1 for lab in labels if id(lab) not in matched),
        "caveat": "Precision is over reviewed events only. Recall is an upper bound: it counts only misses you noticed.",
    }
