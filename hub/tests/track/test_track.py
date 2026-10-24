"""Tracker tests on SIMULATED separator+locator output only (no audio, no device)."""

import pytest

import protocol as p
from track import Tracker
from track.scenarios import SCENES, raw


@pytest.mark.parametrize("scene", SCENES, ids=lambda s: s.__name__)
def test_scene(scene):
    for name, ok, val in scene():
        assert ok, f"{name} [{val}]"


def test_ids_are_never_reused_after_clock_restart():
    tr = Tracker()
    for k in range(5):
        tr.update(10_000_000 + k * 100_000, [raw("s1", "speech", 10.0)])
    for k in range(5):  # headset app relaunched: clock restarts from 0
        out = tr.update(k * 100_000, [raw("s1", "speech", 10.0)])
    assert [s.source_id for s in out] == ["t2"]


def test_non_finite_values_do_not_crash():
    tr = Tracker()
    for k in range(6):
        tr.update(k * 100_000, [raw("s1", "speech", 10.0), raw("s2", "speech", float("nan")),
                                raw("s3", "speech", 20.0, level=float("nan"))])
    assert [s.source_id for s in tr.sources(600_000)] == ["t1"]


def test_clock_restart_resets_ids():
    tr = Tracker()
    for k in range(5):
        tr.update(10_000_000 + k * 100_000, [raw("s1", "speech", 10.0)])
    assert [s.source_id for s in tr.sources(0)] == ["t1"]
    for k in range(5):  # headset app relaunched: device clock starts again from 0
        out = tr.update(k * 100_000, [raw("s1", "speech", 10.0)])
    assert len(out) == 1 and tr.tracks[out[0].source_id].first_us == 0


def test_event_is_valid_protocol_message():
    tr = Tracker()
    for k in range(4):
        tr.update(k * 100_000, [raw("s1", "speech", -10.0)], 0.0)
    ev = tr.event(400_000, 0.0)
    assert isinstance(ev, p.SourcesUpdated) and ev.sources[0].in_view
    p.SourcesUpdated.model_validate_json(ev.model_dump_json())
