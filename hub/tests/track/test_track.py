"""Tracker tests on SIMULATED separator+locator output only (no audio, no device)."""

import pytest

import protocol as p
from track import Tracker
from track.scenarios import SCENES, raw


@pytest.mark.parametrize("scene", SCENES, ids=lambda s: s.__name__)
def test_scene(scene):
    for name, ok, val in scene():
        assert ok, f"{name} [{val}]"


def test_clock_restart_resets_ids():
    tr = Tracker()
    for k in range(5):
        tr.update(10_000_000 + k * 100_000, [raw("s1", "speech", 10.0)])
    assert [s.source_id for s in tr.sources(0)] == ["t1"]
    for k in range(5):  # headset app relaunched: device clock starts again from 0
        out = tr.update(k * 100_000, [raw("s1", "speech", 10.0)])
    assert [s.source_id for s in out] == ["t1"] and tr.tracks["t1"].first_us == 0


def test_event_is_valid_protocol_message():
    tr = Tracker()
    for k in range(4):
        tr.update(k * 100_000, [raw("s1", "speech", -10.0)], 0.0)
    ev = tr.event(400_000, 0.0)
    assert isinstance(ev, p.SourcesUpdated) and ev.sources[0].in_view
    p.SourcesUpdated.model_validate_json(ev.model_dump_json())
