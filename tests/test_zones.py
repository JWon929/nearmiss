"""Named zones: events are tagged with the zone they happen in, and road users through each zone are counted."""
import numpy as np

from engine.calibration import Calibration
from engine.trajectories import Track
from engine.zones import tag_events, zone_summary

# straight-down camera, 10 px per metre: map metres are pixels / 10 (up to the map rotation, which keeps distances)
CAL = Calibration.from_two_points([0, 0], [100, 0], 10.0, (1000, 800))
CROSSWALK = {"name": "North crosswalk", "kind": "crosswalk", "points": [[100, 100], [300, 100], [300, 160], [100, 160]]}
BIKE_LANE = {"name": "Bike lane", "kind": "bike lane", "points": [[600, 0], [640, 0], [640, 800], [600, 800]]}


def to_map(px):
    return CAL.img_to_map([px])[0][0]


def ev(eid, px, sev="serious"):
    x, y = to_map(px)
    return {"id": eid, "x": float(x), "y": float(y), "severity": sev}


def track(tid, group, pixels):
    xy = CAL.img_to_map(pixels)[0]
    n = len(xy)
    return Track(tid, 0, {"veh": "car", "ped": "pedestrian", "bike": "bicycle"}[group], group, 0, np.arange(n) * 0.1,
                 xy, np.zeros((n, 2)), np.zeros(n), n)


def test_events_are_tagged_with_their_zone():
    events = [ev("E1", [200, 130]), ev("E2", [620, 400]), ev("E3", [900, 700])]
    tag_events(events, [CROSSWALK, BIKE_LANE], CAL)
    assert [e["zone"] for e in events] == ["North crosswalk", "Bike lane", ""]


def test_zone_summary_counts_road_users_through_each_zone_once():
    events = [ev("E1", [200, 130]), ev("E2", [210, 140], "candidate")]
    tag_events(events, [CROSSWALK, BIKE_LANE], CAL)
    tracks = [track(1, "ped", [[150, 50], [150, 130], [150, 200]]),       # walks across the crosswalk
              track(2, "veh", [[50, 130], [400, 130]]),                    # drives through it
              track(2, "veh", [[50, 135], [400, 135]]),                    # same car, second segment: counted once
              track(3, "bike", [[620, 0], [620, 800]]),                    # rides the bike lane
              track(4, "veh", [[800, 600], [900, 600]])]                   # nowhere near either zone
    rows = {r["zone"]: r for r in zone_summary(events, tracks, [CROSSWALK, BIKE_LANE], CAL)}
    cw = rows["North crosswalk"]
    assert (cw["pedestrians"], cw["vehicles"], cw["bicycles"], cw["events"]) == (1, 1, 0, 2)
    assert (cw["serious"], cw["candidate"], cw["events_per_100_users"]) == (1, 1, 100.0)
    assert rows["Bike lane"]["bicycles"] == 1 and rows["Bike lane"]["events_per_100_users"] == 0.0
