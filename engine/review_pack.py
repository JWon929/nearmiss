"""
Review pack: let someone else (a teacher, a traffic engineer) label the flagged events without installing anything.

build_pack_html  one standalone HTML file: a short clip (when available) and a frame, the numbers and a plain-language
                 sentence per event, three
                 label buttons, and a button that downloads the reviewer's answers as JSON. Your own labels are NOT
                 included, so the second review is blind. Nothing is sent anywhere; answers stay in that browser
                 until the reviewer downloads them.
read_pack_answers  parse the JSON the reviewer sends back.
agreement        how often the two of you agree, and Cohen's kappa (agreement corrected for chance).
"""
from __future__ import annotations

import html
import json
from typing import Dict, List, Optional, Sequence

from .validation import match

LABELS = ("tp", "fp", "unsure")
LABEL_TEXT = {"tp": "Real near miss", "fp": "False alarm", "unsure": "Unsure"}
FORMAT = "nearmiss-review-pack/1"


def build_pack_html(title: str, events: Sequence[Dict], frames_b64: Dict[str, str], explain, pack_id: str,
                    clips_b64: Optional[Dict[str, str]] = None, privacy_note: str = "The pictures are blurred.") -> str:
    """`clips_b64`: event id -> base64 MP4. A still frame cannot show who yielded to whom; a few seconds of video
    can, so a reviewer can judge the event the way you did."""
    esc = html.escape
    clips_b64 = clips_b64 or {}
    cards = []
    for e in events:
        img = frames_b64.get(e["id"])
        clip = clips_b64.get(e["id"])
        ttc = "n/a" if e.get("ttc_min_s") is None else f"{e['ttc_min_s']} s"
        buttons = "".join(f"<button data-l='{l}'>{LABEL_TEXT[l]}</button>" for l in LABELS)
        cards.append(
            f"<section class='ev' data-kind='{esc(e['kind'])}' data-t='{e['t_peak']}' data-id='{esc(e['id'])}'>"
            f"<h2>{esc(e['id'])} · {esc(e['kind_label'])} · t = {e['t_peak']} s</h2>"
            + (f"<video controls loop muted playsinline src='data:video/mp4;base64,{clip}'></video>" if clip else "")
            + (f"<img alt='Frame at the closest moment of {esc(e['id'])}' src='data:image/jpeg;base64,{img}'>"
               if img else "")
            + f"<p>{esc(explain(e))}</p><p class='num'>Closest {e['d_min_m']} m · minimum TTC {ttc}</p>"
              f"<div class='btns'>{buttons}</div><input class='note' placeholder='Note (optional)'></section>")
    data = json.dumps({"format": FORMAT, "pack": pack_id, "recording": title})
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>NearMiss review: {esc(title)}</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px;color:#222;background:#fff}}
.ev{{border:1px solid #ccc;border-radius:10px;padding:12px 16px;margin:16px 0}}.ev.done{{border-color:#2a7}}
img,video{{max-width:100%;border-radius:6px;display:block;margin:6px 0}}.num{{color:#555}}.btns button{{font-size:15px;margin:4px 6px 4px 0;padding:8px 14px;
border:1px solid #888;border-radius:6px;background:#f6f6f6;cursor:pointer}}.btns button.on{{background:#1f5fbf;color:#fff;
border-color:#1f5fbf}}.note{{width:100%;box-sizing:border-box;padding:6px;margin-top:6px}}
header{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd}}#dl{{font-size:16px;padding:8px 16px}}
</style></head><body><header><h1>Review: {esc(title)}</h1>
<p>For each flagged moment, decide whether it was a <b>real near miss</b>. {esc(privacy_note)} Your answers stay in
this browser until you press the button below, then send the downloaded file back.</p>
<label>Your name <input id='who'></label> <button id='dl'>Download my answers (<span id='n'>0</span>/{len(events)})</button>
</header>{''.join(cards)}
<script>
const META = {data};
const KEY = "nearmiss-pack-" + META.pack;
let saved = {{}};
try {{ saved = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
const who = document.getElementById("who");
who.value = saved._who || "";
function store() {{ try {{ localStorage.setItem(KEY, JSON.stringify(saved)); }} catch (e) {{}} }}
function count() {{ document.getElementById("n").textContent = Object.keys(saved).filter(k => k !== "_who").length; }}
document.querySelectorAll(".ev").forEach(sec => {{
  const id = sec.dataset.id, note = sec.querySelector(".note");
  const show = () => {{
    sec.querySelectorAll("button").forEach(b => b.classList.toggle("on", saved[id] && saved[id].label === b.dataset.l));
    sec.classList.toggle("done", !!saved[id]);
  }};
  if (saved[id]) note.value = saved[id].note || "";
  sec.querySelectorAll("button").forEach(b => b.onclick = () => {{
    saved[id] = {{kind: sec.dataset.kind, t_peak: parseFloat(sec.dataset.t), label: b.dataset.l, note: note.value}};
    store(); show(); count();
  }});
  note.oninput = () => {{ if (saved[id]) {{ saved[id].note = note.value; store(); }} }};
  show();
}});
who.oninput = () => {{ saved._who = who.value; store(); }};
count();
document.getElementById("dl").onclick = () => {{
  const events = Object.keys(saved).filter(k => k !== "_who").map(k => saved[k]);
  const out = Object.assign({{}}, META, {{reviewer: who.value, events: events}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {{type: "application/json"}}));
  a.download = "nearmiss_review_" + (who.value || "reviewer").replace(/[^a-z0-9]+/gi, "_") + ".json";
  a.click();
}};
</script></body></html>"""


def read_pack_answers(raw: bytes) -> Dict:
    """Parse a reviewer's answers. Raises ValueError for anything that is not a NearMiss review file."""
    try:
        d = json.loads(raw)
    except Exception as ex:
        raise ValueError("The file is not JSON.") from ex
    if not isinstance(d, dict) or d.get("format") != FORMAT or not isinstance(d.get("events"), list):
        raise ValueError("The file is not a NearMiss review-pack answer file.")
    clean = []
    for e in d["events"]:
        if isinstance(e, dict) and e.get("label") in LABELS and isinstance(e.get("t_peak"), (int, float)):
            clean.append({"kind": str(e.get("kind")), "t_peak": float(e["t_peak"]), "label": e["label"],
                          "note": str(e.get("note", ""))[:500]})
    return {"reviewer": str(d.get("reviewer", ""))[:80], "pack": d.get("pack"), "events": clean}


def agreement(events: Sequence[Dict], mine: List[Dict], theirs: List[Dict]) -> Dict:
    """Agreement on the events both reviewers labelled. Kappa: 1 = perfect, 0 = no better than chance."""
    pairs = []
    m1, m2 = match(events, mine), match(events, theirs)       # one label per event, one event per label
    for i, e in enumerate(events):
        if i in m1 and i in m2:
            pairs.append((e, mine[m1[i]]["label"], theirs[m2[i]]["label"]))
    n = len(pairs)
    table = {x: {y: 0 for y in LABELS} for x in LABELS}
    for _, x, y in pairs:
        table[x][y] += 1
    out: Dict = {"both_labelled": n, "table": table, "agree": None, "kappa": None,
                 "disagreements": [{"id": e["id"], "t_peak": e["t_peak"], "mine": x, "theirs": y}
                                   for e, x, y in pairs if x != y]}
    if not n:
        return out
    po = sum(table[l][l] for l in LABELS) / n
    pe = sum((sum(table[l].values()) / n) * (sum(table[x][l] for x in LABELS) / n) for l in LABELS)
    out["agree"] = round(po, 3)
    out["kappa"] = round((po - pe) / (1 - pe), 3) if pe < 1 else (1.0 if po == 1 else 0.0)
    return out


def kappa_words(k: Optional[float]) -> str:
    """Landis & Koch (1977) wording, the usual rough guide."""
    if k is None:
        return "n/a"
    for cut, word in ((0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k <= cut:
            return word
    return "almost perfect"
