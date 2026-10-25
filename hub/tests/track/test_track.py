"""Tracker tests on SIMULATED separator+locator output only (no audio, no device)."""

import pytest

import protocol as p
from track import Tracker
from track.scenarios import PROBES, SCENES, raw


@pytest.mark.parametrize("scene", SCENES, ids=lambda s: s.__name__)
def test_scene(scene):
    for name, ok, val in scene():
        assert ok, f"{name} [{val}]"


# Confirmed tracker failures (workstream B). strict: a fix makes the test pass and the marker must go.
KNOWN_FAIL = {
    "near_talkers_8deg": "a second same-label source within merge_deg (12) of an announced one is "
                         "folded into it at _confirm (spec 2.2 / 13 'same direction, different speakers')",
    "two_bearingless_births": "two direction-less same-label tracks block each other's confirmation "
                              "(_confirm counts unconfirmed tracks as 'active'): neither is announced",
    "pause_while_crossing": "forced one-to-one assignment: a talker returning within gate_deg of an "
                            "active one takes its id, and the active one takes the returner's old id",
    "return_near_other": "same cause as pause_while_crossing: linear_sum_assignment must match every "
                         "heard stream, so it prefers a swap to leaving the returner unmatched",
    "low_conf_second_talker": "bearing_conf below min_bearing_conf (0.25) counts as no direction, and a "
                              "direction-less source is never announced while another same-label one is "
                              "active (track.offline base seed 301: the locator's 'ambiguous' cap held a "
                              "talker at conf 0.15 and it was never announced; 13 of 64 simulated talker "
                              "runs were unannounced, not all shown to have this cause)",
    "alternating_near_talkers": "bearing + label + slot cannot tell two talkers 10 deg apart taking "
                                "turns in one slot: needs voice identity (Phase 4 src_n)",
}


@pytest.mark.parametrize("scene", [pytest.param(s, marks=pytest.mark.xfail(reason=KNOWN_FAIL[s.__name__], strict=True))
                                   if s.__name__ in KNOWN_FAIL else s for s in PROBES], ids=lambda s: s.__name__)
def test_probe(scene):
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
