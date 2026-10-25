"""Refinement on vs off: /audio/out frames, seq, sources.updated and mixer counters must be IDENTICAL,
whatever the (fake) runner does: slow, raising, refusing, accepting or never finishing."""

import asyncio
import threading
import time

import numpy as np

import protocol as p
from events import install_events
from ingest import IngestState, create_app
from mockclient.simulate import quat_yaw_pitch
from out import AudioOutConfig, ClientFormatter, install_audio_out
from refine.contracts import RefineConfig, RefineJob, RefineKey, RefineOutcome, RefineState
from refine.install import install_refinement
from separate.live import StreamChunk
from track import TrackConfig, Tracker

SR = 16000
FREQ = {"s1": 500, "s2": 1500}
CFG = RefineConfig(settle_s=0.0, refresh_s=0.0)
release = threading.Event()


class Worker:
    def __init__(self):
        self.listeners = []


def raw(slot, bearing, ts):
    return p.Source(source_id=slot, label="speech", label_conf=1, level_db=-20, bearing_deg=bearing,
                    bearing_conf=1, last_observed_us=ts)


def hop(k):
    t = (k * SR + np.arange(SR)) / SR
    return [StreamChunk(s, "speech", 1.0, 1_000_000 + k * 1_000_000, (0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32),
                        np.zeros((1, SR), np.float32), np.zeros(1), np.zeros((1, 1, 5)), np.ones(1))
            for s, f in FREQ.items()]


def fake_cut(live, job_id):
    return RefineJob(job_id=job_id, key=RefineKey(live.chosen, live.epoch, live.selection_version, live.audio_resets),
                     slots=live.slots, capture_start_us=live.now_us - 4_000_000, capture_end_us=live.now_us,
                     mix_16k=None, slot_audio={}, frames=[])


def slow(job):
    time.sleep(0.15)
    return RefineOutcome(RefineState.accepted, "ok", track_id="f0", score=9.0, assoc_conf=0.9)


def raising(job):
    raise RuntimeError("model exploded")


def refusing(job):
    return RefineOutcome(RefineState.refused, "no face")


def never(job):
    release.wait(30)
    return RefineOutcome(RefineState.accepted, "late", track_id="f0", score=9.0, assoc_conf=0.9)


def drive(runner):
    """Run the scripted session; return (frames, header seqs, sources payloads, mixer status)."""
    release.clear()

    async def go():
        ingest = IngestState()
        ingest.on_pose_channel(p.PoseSample(seq=0, device_ts_us=1_000_000, epoch=0, pos=(0, 0, 0),
                                            rot_xyzw=quat_yaw_pitch(0, 0)))
        app = create_app(ingest)
        events = install_events(app, None, Tracker(TrackConfig(confirm_hits=1)))
        out = install_audio_out(app, events, AudioOutConfig(settle_s=0.0))
        clock = {"t": 0.0}
        out.mixer.clock = lambda: clock["t"]    # injected clock: deterministic across runs
        st = None
        if runner is not None:
            st = install_refinement(app, events, ingest, Worker(), runner_factory=lambda: runner,
                                    config=CFG, executor="thread")
            st.bridge._cut_fn = fake_cut
            st._runner, st.models = runner, "ready"
        fmt = ClientFormatter("layers", out.config)
        frames, payloads, mono = [], [], 0.0
        for k in range(8):
            ts = 1_000_000 + int(mono * 1e6)
            fut = asyncio.get_running_loop().create_future()
            fut.set_result((0, [raw("s1", -60, ts), raw("s2", 50, ts)]))
            events._snapshot_task = fut
            await events.tick(mono)
            payloads.append(events.latest.model_dump_json())
            if k == 3:
                sid = next(s.source_id for s in events.latest.sources if round(s.bearing_deg) == 50)
                ingest.on_pose_channel(p.InputEvent(seq=1, device_ts_us=ts, epoch=0,
                                                    action=p.Action.select, source_id=sid))
            clock["t"] = mono
            out.on_chunks(hop(k))
            for _ in range(50):
                f = out.tick(now=mono + 100)
                if f is not None:
                    frames.append(fmt.format(f, events.inner_db, events.outer_db, events.head_yaw()))
            mono += 1
            await asyncio.sleep(0.05)    # let the refinement worker thread run alongside
        status = out.status()
        refine_hist = st.bridge.status_payload()["history"] if st else None
        release.set()
        if st:
            await asyncio.to_thread(st.bridge.close)
        return frames, payloads, status, refine_hist

    return asyncio.run(go())


def comparable(status):
    return {k: status[k] for k in ("chosen", "selection_version", "slots", "hops_in", "hops_dropped",
                                   "underruns", "frames_out", "queued_ms")}


def test_refinement_never_changes_audio_or_sources():
    base_frames, base_payloads, base_status, _ = drive(None)
    assert base_frames and base_status["chosen"] is not None
    for name, runner in [("slow", slow), ("raising", raising), ("refusing", refusing), ("never", never)]:
        frames, payloads, status, hist = drive(runner)
        assert frames == base_frames, name                  # every /audio/out frame byte-identical (incl. seq)
        assert payloads == base_payloads, name              # sources.updated untouched
        assert comparable(status) == comparable(base_status), name
        assert status["underruns"] == base_status["underruns"]
        assert status["hops_dropped"] == base_status["hops_dropped"]
        assert hist or runner is never, name  # refinement really ran (a never-finishing job records nothing)
