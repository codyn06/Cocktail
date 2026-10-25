"""Direction-less speech can be announced sooner when the deployment opts in (SIMULATED input)."""

from track import Tracker, TrackConfig
from track.scenarios import raw

HOP = 1_000_000  # one separator hop per second, as on the live hub


def _run(tr, slot, hops, start=0, label="speech"):
    out = []
    for i in range(hops):
        t = start + i * HOP
        out = tr.update(t, [raw(slot, label, None, us=t)], 0.0)
    return out, start + hops * HOP


def test_default_needs_the_long_bearingless_wait():
    out, _ = _run(Tracker(), "s1", 6)
    assert out == []


def test_opt_in_speech_is_announced_after_a_few_hops_without_a_bearing():
    out, _ = _run(Tracker(TrackConfig(bearingless_speech_hits=3)), "s1", 3)
    assert len(out) == 1 and out[0].label == "speech" and out[0].bearing_deg is None


def test_opt_in_does_not_speed_up_non_speech():
    out, _ = _run(Tracker(TrackConfig(bearingless_speech_hits=3)), "s3", 6, label="music")
    assert out == []


def test_opt_in_pause_and_resume_in_the_same_slot_keeps_the_id():
    tr = Tracker(TrackConfig(bearingless_speech_hits=3))
    out, now = _run(tr, "s1", 4)
    tid = out[0].source_id
    for step in range(1, 7):
        tr.update(now + step * HOP, [], 0.0)          # 6 s of silence: last_seen, not forgotten
    out, _ = _run(tr, "s1", 3, start=now + 7 * HOP)
    assert [s.source_id for s in out] == [tid]


def test_opt_in_second_speaker_gets_a_new_id_once_the_first_is_quiet():
    tr = Tracker(TrackConfig(bearingless_speech_hits=3))
    out, now = _run(tr, "s1", 4)
    first = out[0].source_id
    for step in range(1, 4):
        tr.update(now + step * HOP, [], 0.0)
    out, _ = _run(tr, "s2", 4, start=now + 4 * HOP)
    ids = {s.source_id for s in out}
    assert first in ids and len(ids) == 2
