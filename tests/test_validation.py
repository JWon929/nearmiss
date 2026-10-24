"""Validation bookkeeping: numbers come only from the human's labels."""
from engine import validation as v


def ev(kind, t, sev="serious"):
    return {"kind": kind, "t_peak": t, "severity": sev}


def test_precision_and_recall_bound():
    events = [ev("ped-veh", 5.0), ev("bike-veh", 9.0), ev("veh-veh", 20.0, "candidate"), ev("ped-veh", 30.0)]
    store = {}
    v.set_label(store, "ped-veh", 5.0, "tp")
    v.set_label(store, "bike-veh", 9.0, "tp")
    v.set_label(store, "veh-veh", 20.0, "fp", "queued cars")
    v.set_label(store, "ped-veh", 30.0, "unsure")
    store["missed"] = [{"t": 40.0, "note": "ped stepped out"}]
    s = v.summarize(events, store)
    assert (s["tp"], s["fp"], s["unsure"], s["unreviewed"], s["missed"]) == (2, 1, 1, 0, 1)
    assert s["precision"] == round(2 / 3, 3)
    assert s["recall_upper_bound"] == round(2 / 3, 3)


def test_labels_survive_small_time_shifts_after_reanalysis():
    store = {}
    v.set_label(store, "ped-veh", 5.0, "tp")
    events = [ev("ped-veh", 5.4)]
    v.apply_labels(events, store)
    assert events[0]["label"] == "tp"


def test_nothing_reviewed_gives_no_numbers():
    s = v.summarize([ev("ped-veh", 5.0)], {})
    assert s["precision"] is None and s["recall_upper_bound"] is None and s["unreviewed"] == 1


def test_clear_removes_label_and_bad_label_is_rejected():
    store = {}
    v.set_label(store, "ped-veh", 5.0, "fp")
    v.set_label(store, "ped-veh", 5.0, "clear")
    assert store["events"] == []
    try:
        v.set_label(store, "ped-veh", 5.0, "maybe")
        assert False, "should have raised"
    except ValueError:
        pass
