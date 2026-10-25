"""A bearing that has not been refreshed by a usable observation is not published (SIMULATED input)."""

import protocol as p
from track import Tracker
from track.scenarios import raw

DT = 100_000  # 10 Hz


def _confirm(tr, bearing=40.0, until=10):
    for i in range(until):
        out = tr.update(i * DT, [raw("s1", "speech", bearing, us=i * DT)], 0.0)
    return out, until * DT


def test_fresh_bearing_is_published():
    tr = Tracker()
    out, _ = _confirm(tr)
    assert out and out[0].bearing_deg is not None and out[0].bearing_conf > 0


def test_bearing_hidden_after_max_age_when_silent():
    tr = Tracker()
    out, now = _confirm(tr)
    age = tr.cfg.bearing_max_age_s
    for step in range(1, int((age + 3) * 10)):
        t = now + step * DT
        out = tr.update(t, [], 0.0)
        if step * DT / 1e6 < age - 0.2:
            assert out[0].bearing_deg is not None, step
    assert out[0].bearing_deg is None and out[0].bearing_conf == 0.0 and out[0].in_view is False
    assert out[0].bearing_source is None


def test_heard_but_directionless_slot_does_not_keep_an_old_bearing():
    # a different talker arrives in the same slot with no usable bearing: the old bearing must age out
    tr = Tracker()
    out, now = _confirm(tr)
    for step in range(1, 90):
        out = tr.update(now + step * DT, [raw("s1", "speech", None, us=now + step * DT)], 0.0)
    assert len(out) == 1 and out[0].bearing_deg is None and out[0].bearing_conf == 0.0


def test_reacquired_track_gets_its_bearing_back():
    tr = Tracker()
    out, now = _confirm(tr)
    for step in range(1, 100):
        out = tr.update(now + step * DT, [], 0.0)
    assert out[0].bearing_deg is None and out[0].state != p.TrackState.tracking
    t = now + 100 * DT
    tid = out[0].source_id
    for k in range(5):
        out = tr.update(t + k * DT, [raw("s1", "speech", 42.0, us=t + k * DT)], 0.0)
    assert out[0].source_id == tid and out[0].bearing_deg is not None
