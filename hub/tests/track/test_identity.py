"""Phase 5 identity in the tracker, the stream chunks and the locator. SIMULATED/mocked only: the
acoustic ids are scripted or come from fake speaker encoders, never from a real voice model."""

import asyncio
import threading

import numpy as np
import pytest

import protocol as p
import track.scenarios as sm
from events import EventsState
from ingest import IngestState
from locate.live import LiveLocator
from separate.live import LiveSeparator, StreamChunk
from track import Tracker
from track.scenarios import IDENTITY_PROBES, PROBES, SCENES, jit, raw

DT = 100_000


@pytest.mark.parametrize("scene", IDENTITY_PROBES, ids=lambda s: s.__name__)
def test_identity_probe(scene):
    for name, ok, val in scene():
        assert ok, f"{name} [{val}]"


def _run(frames, tr=None):
    tr = tr or Tracker()
    out = []
    for k, (srcs, ident) in enumerate(frames):
        out.append(tr.update(k * DT, srcs, 0.0, identity=ident))
    return tr, out


def test_none_or_empty_identity_equals_old_behaviour_on_all_scenes():
    orig = sm.simulate

    def with_empty(frame, seconds, head_yaw=lambda t: 0.0, tracker=None, identity=None):
        return orig(frame, seconds, head_yaw, tracker, identity=lambda t: {})

    for sc in [*SCENES, *PROBES]:
        a = sc()
        sm.simulate = with_empty
        try:
            b = sc()
        finally:
            sm.simulate = orig
        assert a == b, sc.__name__


def test_low_confidence_identity_is_ignored():
    rng = np.random.default_rng(1)
    tr, out = _run(([raw("s1", "speech", jit(rng, 30, 3)), raw("s2", "speech", jit(rng, 38, 3))],
                    {"s1": ("a", 0.5), "s2": ("b", 0.5)}) for _ in range(60))
    assert len(out[-1]) == 1 and not tr.id_map  # below identity_min_conf: today's fold applies


def test_different_confident_ids_never_merge_at_the_same_bearing():
    rng = np.random.default_rng(2)
    tr, out = _run(([raw("s1", "speech", jit(rng, 30, 1)), raw("s2", "speech", jit(rng, 31, 1))],
                    {"s1": ("a", 0.9), "s2": ("b", 0.9)}) for _ in range(80))
    assert len(out[-1]) == 2 and set(tr.id_map) == {"a", "b"}


def test_alternating_one_slot_pair_is_split_and_rejoins_after_bearing_moved():
    rng = np.random.default_rng(3)

    def frames():
        for k in range(200):
            t = k / 10
            aid, b = ("b", 28.0) if 5 <= t < 10 else ("a", 20.0)
            if t >= 14:
                b = 80.0  # a returns far beyond gate_deg: it still reattaches to its own track
            yield [raw("s1", "speech", jit(rng, b, 2))], {"s1": (aid, 0.9)}
    tr, out = _run(frames())
    ids = {s.source_id for o in out for s in o}
    assert ids == {"t1", "t2"} and tr.id_map == {"a": "t1", "b": "t2"}
    last = [s for s in out[-1] if s.state == p.TrackState.tracking]
    assert [s.source_id for s in last] == ["t1"] and abs(last[0].bearing_deg - 80) < 25


def test_mapping_cleared_on_forget_and_reset_but_ids_not_reused():
    tr = Tracker()
    for k in range(6):
        tr.update(k * DT, [raw("s1", "speech", 10.0)], identity={"s1": ("a", 0.9)})
    assert tr.id_map == {"a": "t1"}
    tr.update(70_000_000, [])  # silent for more than forget_s
    assert tr.tracks == {} and tr.id_map == {}
    for k in range(6):
        tr.update(80_000_000 + k * DT, [raw("s1", "speech", 10.0)], identity={"s1": ("a", 0.9)})
    assert tr.id_map == {"a": "t2"}
    tr.reset()
    assert tr.id_map == {}
    tr.update(0, [raw("s1", "speech", 10.0)], identity={"s1": ("a", 0.9)})
    assert list(tr.tracks) == ["t3"]


def test_low_confidence_direction_talker_announced_bearingless():
    rng = np.random.default_rng(4)

    def frames(ident):
        return (([raw("s1", "speech", jit(rng, -30, 2)), raw("s2", "speech", jit(rng, 40, 2), conf=0.1)], ident)
                for _ in range(30))
    _, out = _run(frames({"s1": ("a", 0.9), "s2": ("b", 0.9)}))
    srcs = out[-1]
    assert len(srcs) == 2 and sorted(s.bearing_deg is None for s in srcs) == [False, True]
    _, out2 = _run(frames(None))  # without identity the second talker is not announced (old behaviour)
    assert len(out2[-1]) == 1


# ---------------------------------------------------------------------------- StreamChunk / VoiceMemory
def _chunk(**kw):
    a = np.zeros(4, np.float32)
    return StreamChunk("s1", "speech", 1.0, 0, a, np.stack([a, a]), np.array([0]), np.zeros((1, 2, 5)),
                       np.ones(1), **kw)


def test_stream_chunk_identity_fields_default():
    c = _chunk()
    assert c.acoustic_id is None and c.identity_conf == 0.0


SR = 16000


class _ToneEncoder:
    def __call__(self, audios):
        out = []
        for a in audios:
            f = np.fft.rfftfreq(len(a), 1 / SR)
            out.append([1.0, 0.0] if f[np.argmax(np.abs(np.fft.rfft(a)))] < 500 else [0.0, 1.0])
        return np.asarray(out, np.float32)


class _Oracle:
    n_out = 2

    def __init__(self, a, b):
        self.a, self.b, self.pos = a, b, 0

    def __call__(self, _mix):
        i = self.pos
        self.pos += SR
        return np.stack([self.a[i:i + 2 * SR], self.b[i:i + 2 * SR]])


def _tone(f, n):
    return (np.sin(2 * np.pi * f * np.arange(n) / SR) * 0.1).astype(np.float32)


def test_live_separator_identity_comes_from_voice_memory():
    n = 9 * SR
    a, b = _tone(300, n), _tone(700, n)
    mix = np.stack([a + b, a + b], axis=1)
    off = LiveSeparator(_Oracle(a, b))
    on = LiveSeparator(_Oracle(a, b), voice_memory=True, speaker_encoder=_ToneEncoder())
    first_on = last_on = None
    for k in range(6):
        x = mix[k * SR:k * SR + on.W]
        ts = k * 1_000_000
        co, cf = on.step(x, ts), off.step(x, ts)
        assert all(c.acoustic_id is None and c.identity_conf == 0.0 for c in cf)
        first_on = first_on or co
        last_on = co
    assert [c.acoustic_id for c in first_on] == ["src_0", "src_1", None]
    assert all(c.identity_conf <= 0.3 for c in first_on)      # no profile yet
    assert all(c.identity_conf >= 0.8 for c in last_on[:2]) and last_on[2].identity_conf == 0.0


def test_live_separator_uncertain_return_is_low_confidence():
    n = 6 * SR
    a, b = _tone(300, n), _tone(300, n)
    ls = LiveSeparator(_Oracle(a, b), voice_memory=True, speaker_encoder=_ToneEncoder())
    ls.voice_memory.profiles = [np.array([1.0, 0.0], np.float32), np.array([0.8, 0.6], np.float32)]
    ls.voice_memory._silent_s = [2.0, 2.0]
    mix = np.stack([a + b, a + b], axis=1)
    chunks = ls.step(mix[:ls.W], 0)
    assert ls.last_gates["voice_memory"]["returning"]
    # slot 0 was matched confidently; the other input fell back to overlap order and is not trusted
    assert chunks[0].identity_conf >= 0.8 and chunks[1].identity_conf <= 0.3


# ---------------------------------------------------------------------------- locator + events
class _Ingest:
    pose_epoch = 0

    def __init__(self):
        q = type("Q", (), {"device_ts_us": 0, "yaw_deg": 0.0, "pitch_deg": 0.0})()
        self.poses = type("P", (), {"lock": threading.Lock(), "poses": [q]})()


def _lchunk(sid, aid, conf, ts):
    a = np.full(800, 0.1, np.float32)
    return StreamChunk(sid, "speech", 1.0, ts, a, np.stack([a, a]), np.array([ts + 50_000]),
                       np.full((1, 2, 5), -20.0), np.array([1.0]), acoustic_id=aid, identity_conf=conf)


def test_locator_identities_and_rekey_on_confident_change():
    loc = LiveLocator(_Ingest())
    seen = []
    orig = loc.est.add_levels
    loc.est.add_levels = lambda key, *a, **k: (seen.append(key), orig(key, *a, **k))[1]
    loc.on_chunks([_lchunk("s1", "src_0", 0.9, 1_000_000), _lchunk("s2", None, 0.0, 1_000_000)])
    assert loc.identities() == {"s1": ("src_0", 0.9), "s2": (None, 0.0)}
    loc.on_chunks([_lchunk("s1", "src_1", 0.3, 2_000_000)])  # unsure change: no re-key
    loc.on_chunks([_lchunk("s1", "src_0", 0.9, 3_000_000)])
    assert set(seen) == {"s1", "s2"}
    loc.on_chunks([_lchunk("s1", "src_1", 0.9, 4_000_000)])  # confident change: fresh direction state
    assert seen[-1] == "s1#1" and loc.identities()["s1"] == ("src_1", 0.9)
    assert [s.source_id for s in loc.sources()] == ["s1", "s2"]  # published ids stay the stream ids


def test_events_tick_passes_identity_from_the_snapshot_only_if_locator_has_identities():
    class Rec(Tracker):
        calls = []

        def update(self, now_us, raw_, head_yaw_deg=None, **kw):
            Rec.calls.append(kw)
            return super().update(now_us, raw_, head_yaw_deg, **kw)

    class Plain:  # a locator without identities(): update() is called exactly as before
        lock = threading.Lock()

    class WithIds(Plain):
        def identities(self):
            return {}

    async def go(locator):
        Rec.calls.clear()
        ev = EventsState(IngestState(), locator, Rec())
        ev._anchor_mono = 0.0  # tick(mono) counts from here
        fut = asyncio.get_running_loop().create_future()
        fut.set_result((ev.ingest.pose_epoch, [raw("s1", "speech", 0.0)], {"s1": ("src_0", 0.9)}))
        ev._snapshot_task = fut
        await ev.tick(0)
        return list(Rec.calls)

    assert asyncio.run(go(WithIds()))[0] == {"identity": {"s1": ("src_0", 0.9)}}
    assert asyncio.run(go(Plain()))[0] == {}
