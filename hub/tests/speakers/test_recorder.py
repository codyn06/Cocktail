"""Speaker recorder with synthetic chunks and a fake Tracker (no models, no device)."""

import io
import wave
from types import SimpleNamespace as NS

import numpy as np

from speakers.recorder import RING_S, SpeakerRecorder

SR = 16000
S = 1_000_000


class FakeTracker:
    def __init__(self):
        self.tracks = {}
        self._dup_of = {}

    def set(self, tid, stream, state="tracking", first=0, last=0, confirmed=True):
        self.tracks[tid] = NS(tid=tid, stream=stream, state=NS(value=state), first_us=first * S,
                              last_obs_us=last * S, confirmed=confirmed, bearing=0.0, bearing_conf=1.0)


def chunk(slot, t, val, active=True, label="speech"):
    ts = t * S + np.arange(20) * 50_000 + 25_000
    return NS(stream_id=slot, label=label, start_ts_us=int(t * S), audio=np.full(SR, val, np.float32),
              level_ts_us=ts, activity=np.full(20, 1.0 if active else 0.0))


def mk():
    tr = FakeTracker()
    return tr, SpeakerRecorder(lambda: tr)


def wav_samples(data):
    w = wave.open(io.BytesIO(data))
    assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR)
    return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def test_slot_swap_keeps_ids_and_audio():
    tr, rec = mk()
    tr.set("t1", "s1", first=0, last=0)
    tr.set("t2", "s2", first=0, last=0)
    for t in range(2):
        rec.on_chunks([chunk("s1", t, 0.1), chunk("s2", t, 0.5)])
    # the separator swaps: t1 now lives in s2, t2 in s1
    tr.tracks["t1"].stream, tr.tracks["t2"].stream = "s2", "s1"
    tr.tracks["t1"].last_obs_us = tr.tracks["t2"].last_obs_us = 2 * S
    for t in range(2, 4):
        rec.on_chunks([chunk("s1", t, 0.5), chunk("s2", t, 0.1)])
    for tid, val in (("t1", 0.1), ("t2", 0.5)):
        x = wav_samples(rec.wav_bytes(tid)) / 32767
        assert len(x) == 4 * SR
        assert np.allclose(x, val, atol=1e-3)
    assert rec.info("t1")["number"] == 1 and rec.info("t2")["number"] == 2


def test_resume_in_other_slot_no_leak():
    tr, rec = mk()
    tr.set("t1", "s1", last=0)
    tr.set("t2", "s2", last=0)
    rec.on_chunks([chunk("s1", 0, 0.1), chunk("s2", 0, 0.5)])
    # t1 pauses (silent in s1) while t2 takes over slot s1; t1 is still "tracking" with stale slot
    tr.tracks["t2"].stream = "s1"
    tr.tracks["t2"].last_obs_us = 2 * S
    rec.on_chunks([chunk("s1", 1, 0.5)])
    # t1 resumes in s2 (tracker decides)
    tr.tracks["t1"].stream = "s2"
    tr.tracks["t1"].last_obs_us = 3 * S
    rec.on_chunks([chunk("s1", 2, 0.5), chunk("s2", 2, 0.1)])
    a1 = wav_samples(rec.wav_bytes("t1")) / 32767
    assert np.allclose(a1[:SR], 0.1, atol=1e-3) and np.allclose(a1[-SR:], 0.1, atol=1e-3)
    assert np.allclose(a1[SR:2 * SR], 0.0)   # gap zero-filled, nothing leaked
    assert not np.any(np.isclose(a1, 0.5, atol=1e-2))


def test_overlap_both_have_intervals():
    tr, rec = mk()
    tr.set("t1", "s1")
    tr.set("t2", "s2")
    rec.on_chunks([chunk("s1", 0, 0.1), chunk("s2", 0, 0.5)])
    for tid in ("t1", "t2"):
        iv = rec.intervals(tid)
        assert len(iv) == 1 and iv[0]["start_us"] == 0 and iv[0]["end_us"] == S


def test_pending_flush_after_late_confirmation():
    tr, rec = mk()
    tr.set("t1", "s1", first=0, confirmed=False)
    for t in range(3):
        rec.on_chunks([chunk("s1", t, 0.3)])
    assert rec.status()["pending"] == 3 and rec.ids() == []
    tr.tracks["t1"].confirmed = True
    rec.on_chunks([chunk("s1", 3, 0.3)])
    assert len(wav_samples(rec.wav_bytes("t1"))) == 4 * SR
    assert rec.info("t1")["audio_start_ts_us"] == 0
    assert rec.status()["pending"] == 0


def test_pending_older_than_track_dropped():
    tr, rec = mk()
    tr.set("t1", "s1", first=5, confirmed=False)
    for t in range(3):
        rec.on_chunks([chunk("s1", t, 0.3)])
    tr.tracks["t1"].confirmed = True
    rec.on_chunks([chunk("s1", 5, 0.3)])
    assert rec.info("t1")["audio_start_ts_us"] == 5 * S
    assert rec.status()["hops_dropped"] == 3


def test_pending_not_flushed_to_different_track_in_slot():
    tr, rec = mk()
    tr.set("t1", "s1", first=0)
    rec.on_chunks([chunk("s1", 0, 0.1)])
    tr.tracks["t1"].state = NS(value="lost")
    rec.on_chunks([chunk("s1", 1, 0.9), chunk("s1", 2, 0.9)])   # unattributed
    tr.set("t2", "s1", first=1, last=3)
    rec.on_chunks([chunk("s1", 3, 0.9)])
    # slot previously resolved to t1, so the pending hops are ambiguous: dropped
    assert wav_samples(rec.wav_bytes("t2")).shape[0] == SR
    assert wav_samples(rec.wav_bytes("t1")).shape[0] == SR


def test_ambiguous_dropped_and_residual_ignored():
    tr, rec = mk()
    tr.set("t1", "s1")
    tr.set("t2", "s1")  # two tracks same slot, same last_obs: ambiguous
    rec.on_chunks([chunk("s1", 0, 0.1), chunk("s3", 0, 0.1), chunk("s2", 0, 0.1, label="music")])
    assert rec.ids() == []
    assert rec.status()["hops_seen"] == 1 and rec.status()["hops_dropped"] == 1


def test_intervals_match_audio_and_hysteresis():
    tr, rec = mk()
    tr.set("t1", "s1")
    # 1 s active, 1 s silent, 1 s active: gap 1 s > 300 ms -> two intervals
    for t, act in ((0, True), (1, False), (2, True)):
        rec.on_chunks([chunk("s1", t, 0.2, active=act)])
    iv = rec.intervals("t1")
    assert [(i["start_us"], i["end_us"]) for i in iv] == [(0, S), (2 * S, 3 * S)]
    sp = rec.speakers["t1"]
    n = len(wav_samples(rec.wav_bytes("t1")))
    assert n == 3 * SR and (sp.end_us - sp.audio_start_us()) == n * S // SR
    assert all(sp.audio_start_us() <= i["start_us"] and i["end_us"] <= sp.end_us for i in iv)


def test_short_island_dropped_and_small_gap_merged():
    tr, rec = mk()
    tr.set("t1", "s1")
    c = chunk("s1", 0, 0.2)
    c.activity[:] = 0
    c.activity[2:4] = 1          # 100 ms island -> dropped
    c.activity[10:14] = 1        # 200 ms
    c.activity[15:18] = 1        # gap 50 ms -> merged
    rec.on_chunks([c])
    iv = rec.intervals("t1")
    assert len(iv) == 1 and iv[0]["end_us"] - iv[0]["start_us"] == 8 * 50_000


def test_speaking_false_after_silence_and_wav_time_axis():
    tr, rec = mk()
    tr.set("t1", "s1")
    tr.set("t2", "s2")
    rec.on_chunks([chunk("s1", 0, 0.2), chunk("s2", 0, 0.2)])
    assert rec.info("t1")["speaking"]
    rec.on_chunks([chunk("s2", 1, 0.2, active=False)])
    rec.on_chunks([chunk("s2", 2, 0.2, active=False)])
    assert not rec.info("t1")["speaking"]
    assert rec.intervals("t1")[0]["open"] is False
    # gap in time axis zero-filled: t1 gets audio at t=5
    tr.tracks["t1"].last_obs_us = 5 * S
    rec.on_chunks([chunk("s1", 5, 0.4)])
    x = wav_samples(rec.wav_bytes("t1"))
    assert len(x) == 6 * SR and x[2 * SR] == 0 and x[-1] != 0


def test_bounded_memory_ten_minutes():
    tr, rec = mk()
    tr.set("t1", "s1")
    for t in range(600):
        rec.on_chunks([chunk("s1", t, 0.2)])
    info = rec.info("t1")
    assert info["audio_s"] <= RING_S
    assert len(wav_samples(rec.wav_bytes("t1"))) <= RING_S * SR
    assert info["audio_start_ts_us"] == (600 - RING_S) * S
    assert len(rec.intervals("t1")) <= 200


def test_epoch_reset():
    tr = FakeTracker()
    ep = [0]
    rec = SpeakerRecorder(lambda: tr, lambda: ep[0])
    tr.set("t1", "s1")
    rec.on_chunks([chunk("s1", 0, 0.2)])
    assert rec.ids() == ["t1"]
    ep[0] = 1
    tr.tracks.clear()
    rec.on_chunks([chunk("s1", 0, 0.2)])
    assert rec.ids() == []


def test_routes_lost_speaker_null_bearing():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from speakers.server import install_speakers

    tr = FakeTracker()
    tr.set("t1", "s1")
    tr.set("t2", "s2")

    def src(i, st):
        return NS(source_id=i, bearing_deg=30.0, bearing_conf=0.9, state=NS(value=st))

    events = NS(tracker=tr, ingest=NS(pose_epoch=0), selection=NS(chosen="t1"),
                latest=NS(sources=[src("t1", "tracking"), src("t2", "lost")]))
    worker = NS(listeners=[])
    app = FastAPI()
    rec = install_speakers(app, events, worker)
    assert worker.listeners == [rec.on_chunks]
    rec.on_chunks([chunk("s1", 0, 0.2), chunk("s2", 0, 0.2)])
    tr.tracks["t2"].state = NS(value="lost")
    c = TestClient(app)
    rows = {r["speaker_id"]: r for r in c.get("/speakers").json()}
    assert rows["t1"]["bearing_deg"] == 30.0 and rows["t1"]["selected"] and rows["t1"]["label"] == "Speaker 1"
    assert rows["t2"]["bearing_deg"] is None and rows["t2"]["state"] == "lost"
    assert c.get("/speakers/t1/audio.wav").headers["content-type"] == "audio/wav"
    assert c.get("/speakers/nope/audio.wav").status_code == 404
    assert c.get("/speakers/t1/intervals").json()[0]["start_us"] == 0
    assert c.get("/speakers/status").json()["hops_attributed"] == 2


def test_exception_safe():
    rec = SpeakerRecorder(lambda: None)
    rec.on_chunks([chunk("s1", 0, 0.2)])
    assert rec.status()["errors"] == 1
