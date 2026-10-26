# Contributing to NearMiss

Thanks for helping. NearMiss is a screening tool that people may use to argue for street changes, so the rule above
all others is: **never make a number look more certain than it is.**

## Set up

Windows: `.\setup.ps1`. Mac or Linux: `./setup.sh`. Both make a virtual environment in `.venv`, install
`requirements.txt` and start the app. Or by hand:

```
python -m venv .venv
.venv/Scripts/activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Before you open a pull request

- `python -m pytest tests` passes. Add a test for anything that changes a number: a scripted scene with a known
  answer (see `tests/test_engine_synthetic.py`) is better than a test of the code's own output.
- Speed changes: measure with `python tools/benchmark.py analysis` and `python tools/benchmark.py detect CLIP` before
  and after, and put both numbers in the pull request. Vectorised code needs a test that it matches the simple loop.
- New indicators or thresholds start **off** or at the old default, and say in the UI that they are not validated
  until checked against labelled events.
- No footage, labels or `runs/` in commits. Model weights (`*.pt`) stay out too.

## Style

- Keep `engine/` free of Streamlit. Pages call the engine; the engine never calls pages.
- Write messages and captions in plain English for someone who is not a traffic engineer: what happened and what to
  do next.
- Comments say why, not what. Match the code around you.

## Honesty checklist for anything shown to users

- Does the page say what the number cannot tell you (blind spots, provisional calibration, unreviewed events)?
- Are rates shown next to raw counts, so busy and quiet clips are not compared by counts?
- Is anything presented as crash prediction? It must not be.
