"""
Honest validation: compare what NearMiss flagged with what a human decides.

Workflow (Event Review screen):
  1. Watch every flagged event clip. Mark it  true positive / false positive / unsure.
  2. While watching the clip, note REAL conflicts the system missed ("missed" list).
  3. The numbers below are computed ONLY from your labels. Nothing is invented.

Labels are stored against the event's kind, its time of closest approach and the two track IDs involved, rather
than the event number, so they survive re-running the analysis with different thresholds.

Matching is ONE-TO-ONE: a label belongs to at most one event and an event has at most one label. (An earlier version
matched every event of the same kind within 1 s, so one "real near miss" could be counted for two different events.)
A label first goes to the event with the same two road users; otherwise to the nearest event of the same kind within
1 s that no other label claims.

Interpretation caveats (put them in your submission):
  * precision = TP / (TP + FP) over the events you reviewed
  * recall    = TP / (TP + missed) is an UPPER BOUND on true recall: you can only
    list misses you happened to notice.
  * one clip is an anecdote, not a benchmark. Say how many events and clips you reviewed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

MATCH_WINDOW_S = 1.0
LABELS = ("tp", "fp", "unsure")
KINDS = ("ped-veh", "bike-veh", "veh-veh")


def event_pair(e: Dict) -> Optional[List[int]]:
    """The two track IDs of an event, sorted, or None for an event without them."""
    try:
        return sorted([int(e["a"]["track"]), int(e["b"]["track"])])
    except (KeyError, TypeError, ValueError):
        return None


def event_key(e: Dict) -> str:
    """A name for an event that does not change when other events appear or disappear (unlike E001, E002...)."""
    p = event_pair(e)
    return f"{e['kind']}:{p[0]}-{p[1]}:{e['t_peak']:.1f}" if p else f"{e['kind']}:{e['t_peak']:.1f}"


def match(events: Sequence[Dict], labels: Sequence[Dict]) -> Dict[int, int]:
    """One-to-one assignment {event index: label index}. Same road users first, then closest in time."""
    cands: List[Tuple[int, float, int, int]] = []
    for i, e in enumerate(events):
        pe = event_pair(e)
        for j, lab in enumerate(labels):
            if lab.get("kind") != e["kind"]:
                continue
            dt = abs(float(lab["t_peak"]) - float(e["t_peak"]))
            if dt > MATCH_WINDOW_S:
                continue
            same_users = pe is not None and lab.get("pair") == pe
            cands.append((0 if same_users else 1, dt, i, j))
    out: Dict[int, int] = {}
    used = set()
    for _, _, i, j in sorted(cands):
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def find_label(labels: List[Dict], kind: str, t_peak: float) -> Optional[Dict]:
    """The nearest label of this kind within the match window (for one event on its own, e.g. a reviewer file)."""
    best, best_dt = None, MATCH_WINDOW_S + 1
    for lab in labels:
        dt = abs(lab["t_peak"] - t_peak)
        if lab["kind"] == kind and dt <= MATCH_WINDOW_S and dt < best_dt:
            best, best_dt = lab, dt
    return best


def apply_labels(events: List[Dict], store: Dict) -> None:
    labels = store.get("events", [])
    m = match(events, labels)
    for i, e in enumerate(events):
        lab = labels[m[i]] if i in m else None
        e["label"] = lab["label"] if lab else None
        e["label_note"] = lab.get("note", "") if lab else ""


def set_label(store: Dict, kind: str, t_peak: float, label: Optional[str], note: str = "",
              pair: Optional[List[int]] = None, events: Optional[Sequence[Dict]] = None) -> Dict:
    """Label (or with "clear", unlabel) one event. Pass the event's `pair` and the current `events` so the label that
    is updated is the one matched to THIS event, never a neighbour's."""
    labels = store.setdefault("events", [])
    if label not in (None, "", "clear") and label not in LABELS:
        raise ValueError("label must be tp, fp, unsure or clear")
    lab = None
    if events is not None:
        m = match(events, labels)
        for i, e in enumerate(events):
            if e["kind"] == kind and abs(e["t_peak"] - t_peak) < 1e-6 and (pair is None or event_pair(e) == pair):
                lab = labels[m[i]] if i in m else None
                break
    else:
        same = [x for x in labels if x["kind"] == kind and abs(x["t_peak"] - t_peak) <= MATCH_WINDOW_S
                and (pair is None or x.get("pair") in (None, pair))]
        lab = min(same, key=lambda x: abs(x["t_peak"] - t_peak)) if same else None
    if label in (None, "", "clear"):
        if lab is not None:
            labels.remove(lab)
        return store
    entry = {"kind": kind, "t_peak": t_peak, "label": label, "note": note}
    if pair is not None:
        entry["pair"] = list(pair)
    if lab is not None:
        lab.clear()
        lab.update(entry)
    else:
        labels.append(entry)
    return store


def clean_store(raw) -> Tuple[Dict, List[str]]:
    """A labels file as the app can trust it, plus what was wrong with it. Bad entries are dropped, not guessed."""
    problems: List[str] = []
    if not isinstance(raw, dict):
        return {"events": [], "missed": []}, ["The file is not a NearMiss labels file (expected a JSON object)."]
    events, missed = [], []
    for n, e in enumerate(raw.get("events", []) if isinstance(raw.get("events", []), list) else []):
        try:
            entry = {"kind": str(e["kind"]), "t_peak": float(e["t_peak"]), "label": str(e["label"]),
                     "note": str(e.get("note", ""))[:500]}
            if entry["kind"] not in KINDS or entry["label"] not in LABELS or entry["t_peak"] < 0:
                raise ValueError
            if e.get("pair") is not None:
                entry["pair"] = sorted(int(x) for x in e["pair"])[:2]
            events.append(entry)
        except (KeyError, TypeError, ValueError):
            problems.append(f"Label {n + 1} was skipped: it needs a kind ({', '.join(KINDS)}), a time t_peak and a "
                            f"label ({', '.join(LABELS)}).")
    for n, m in enumerate(raw.get("missed", []) if isinstance(raw.get("missed", []), list) else []):
        try:
            missed.append({"t": float(m["t"]), "note": str(m.get("note", ""))[:500]})
        except (KeyError, TypeError, ValueError):
            problems.append(f"Missed note {n + 1} was skipped: it needs a time t.")
    if not isinstance(raw.get("events", []), list) or not isinstance(raw.get("missed", []), list):
        problems.append("'events' and 'missed' must be lists.")
    return {"events": events, "missed": missed}, problems


def summarize(events: List[Dict], store: Dict) -> Dict:
    labels = store.get("events", [])
    missed = store.get("missed", [])
    n = len(events)
    tp = fp = unsure = 0
    by_sev: Dict[str, Dict[str, int]] = {}
    by_kind: Dict[str, Dict] = {}
    m = match(events, labels)
    for i, e in enumerate(events):
        lab = labels[m[i]] if i in m else None
        s = by_sev.setdefault(e["severity"], {"tp": 0, "fp": 0, "unsure": 0, "unreviewed": 0})
        k = by_kind.setdefault(e["kind"], {"flagged": 0, "tp": 0, "fp": 0, "unsure": 0, "unreviewed": 0})
        k["flagged"] += 1
        key = "unreviewed" if lab is None else (lab["label"] if lab["label"] in ("tp", "fp") else "unsure")
        tp += key == "tp"
        fp += key == "fp"
        unsure += key == "unsure"
        s[key] += 1
        k[key] += 1
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
        "orphaned_labels": len(labels) - len(m),
        "caveat": "Precision is over reviewed events only. Recall is an upper bound: it counts only misses you noticed.",
    }


def combine(summaries: List[Dict]) -> Dict:
    """Totals over several recordings: precision is recomputed from the summed counts, never averaged."""
    tot = {"events_flagged": 0, "tp": 0, "fp": 0, "unsure": 0, "unreviewed": 0, "missed": 0}
    by_kind: Dict[str, Dict] = {}
    for s in summaries:
        for k in tot:
            tot[k] += s[k]
        for kind, row in s.get("by_kind", {}).items():
            acc = by_kind.setdefault(kind, {"flagged": 0, "tp": 0, "fp": 0, "unsure": 0, "unreviewed": 0})
            for k in acc:
                acc[k] += row[k]
    for row in by_kind.values():
        row["precision"] = round(row["tp"] / (row["tp"] + row["fp"]), 3) if row["tp"] + row["fp"] else None
    tot["precision"] = round(tot["tp"] / (tot["tp"] + tot["fp"]), 3) if tot["tp"] + tot["fp"] else None
    tot["recall_upper_bound"] = round(tot["tp"] / (tot["tp"] + tot["missed"]), 3) if tot["tp"] + tot["missed"] else None
    tot["by_kind"] = by_kind
    tot["recordings"] = len(summaries)
    return tot


def merge_labels(store: Dict, incoming: Dict) -> Tuple[int, int]:
    """Add CLEANED labels (see clean_store) to `store`; an incoming label replaces the one for the same event.
    Missed notes are added unless the same one is already there. Returns (labels added or updated, missed added)."""
    n_lab = 0
    for lab in incoming.get("events", []):
        set_label(store, lab["kind"], lab["t_peak"], lab["label"], lab.get("note", ""), pair=lab.get("pair"))
        n_lab += 1
    have = {(round(m["t"], 1), m.get("note", "")) for m in store.setdefault("missed", [])}
    n_missed = 0
    for m in incoming.get("missed", []):
        k = (round(m["t"], 1), m.get("note", ""))
        if k not in have:
            have.add(k)
            store["missed"].append({"t": m["t"], "note": m.get("note", "")})
            n_missed += 1
    return n_lab, n_missed
