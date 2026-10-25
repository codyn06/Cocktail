"""install_refinement wiring with fakes: no models, no sockets (TestClient without lifespan unless stated)."""

import asyncio
import json

from fastapi.testclient import TestClient

import protocol as p
from events import install_events
from ingest import IngestState, create_app
from mockclient.simulate import quat_yaw_pitch
from refine.contracts import RefineConfig, RefineJob, RefineKey, RefineOutcome, RefineState
from refine.install import install_refinement
from track import TrackConfig, Tracker

CFG = RefineConfig(settle_s=0.0, refresh_s=1e9)


class Worker:
    def __init__(self):
        self.listeners = []


def pose(ingest, ts, epoch=0, yaw=0):
    ingest.on_pose_channel(p.PoseSample(
        seq=ingest.stats["pose"].messages, device_ts_us=ts, epoch=epoch,
        pos=(0, 0, 0), rot_xyzw=quat_yaw_pitch(yaw, 0)))


def raw(slot, bearing, ts):
    return p.Source(source_id=slot, label="speech", label_conf=1, level_db=-20, bearing_deg=bearing,
                    bearing_conf=1, last_observed_us=ts)


def fake_cut(live, job_id):
    return RefineJob(job_id=job_id, key=RefineKey(live.chosen, live.epoch, live.selection_version, live.audio_resets),
                     slots=live.slots, capture_start_us=live.now_us - 4_000_000, capture_end_us=live.now_us,
                     mix_16k=None, slot_audio={}, frames=[])


class Rig:
    def __init__(self, runner=None, models="ready"):
        self.ingest = IngestState()
        pose(self.ingest, 1_000_000)
        self.app = create_app(self.ingest)
        self.events = install_events(self.app, None, Tracker(TrackConfig(confirm_hits=1)))
        self.st = install_refinement(self.app, self.events, self.ingest, Worker(),
                                     runner_factory=lambda: runner, config=CFG, executor="inline")
        self.st.bridge._cut_fn = fake_cut
        self.st._runner, self.st.models = runner, models
        self.aux = asyncio.Queue(maxsize=64)
        self.events.aux_clients[object()] = self.aux
        self.mono = 0.0

    async def observe(self):
        ts = 1_000_000 + int(self.mono * 1e6)
        fut = asyncio.get_running_loop().create_future()
        fut.set_result((0, [raw("s1", -60, ts), raw("s2", 50, ts)]))
        self.events._snapshot_task = fut
        await self.events.tick(self.mono)
        self.mono += 1

    async def choose(self):
        for _ in range(3):
            await self.observe()
        sid = self.events.latest.sources[0].source_id
        self.ingest.on_pose_channel(p.InputEvent(
            seq=self.ingest.stats["pose"].messages, device_ts_us=1_000_000, epoch=0,
            action=p.Action.select, source_id=sid))
        assert self.events.selection.chosen == sid
        return sid

    def targets(self):
        out = []
        while not self.aux.empty():
            out.append(json.loads(self.aux.get_nowait()))
        return out


def run(coro):
    return asyncio.run(coro)


def test_status_route_shape_while_models_load():
    rig = Rig(runner=None, models="loading")
    body = TestClient(rig.app).get("/refinement/status").json()
    assert body["models"] == "loading" and body["audio_applied"] is False
    assert {"current", "history", "counts", "in_flight", "config"} <= set(body)


def test_accepted_refused_unavailable_reach_status_and_target_updated():
    async def go(outcome, state, kind, conf):
        rig = Rig(runner=lambda job: outcome)
        sid = await rig.choose()
        await rig.observe()
        msgs = rig.targets()
        last = p.TargetUpdated.model_validate(msgs[-1])
        assert (last.target_id, last.source_id, last.kind, last.assoc_conf) == (f"t_{sid}", sid, kind, conf)
        assert last.selection_version == rig.events.selection.version and last.voice_lock_conf is None
        assert last.bearing_deg is not None
        cur = TestClient(rig.app).get("/refinement/status").json()["current"]
        assert cur["state"] == state
    run(go(RefineOutcome(RefineState.accepted, "ok", track_id="f0", score=12.0, assoc_conf=0.6),
           "accepted", "person", 0.6))
    run(go(RefineOutcome(RefineState.refused, "no"), "refused", "speech", 0.0))
    run(go(RefineOutcome(RefineState.unavailable, "no video"), "unavailable", "speech", 0.0))


def test_models_loading_and_failed_are_unavailable():
    async def go(models, err, reason):
        rig = Rig(runner=None, models=models)
        rig.st.models_error = err
        await rig.choose()
        await rig.observe()
        body = TestClient(rig.app).get("/refinement/status").json()
        assert body["current"]["state"] == "unavailable" and body["current"]["reason"] == reason
        assert rig.targets()[-1]["kind"] == "speech"
    run(go("loading", None, "refinement models loading"))
    run(go("failed", "RuntimeError", "refinement models failed to load: RuntimeError"))


def test_lifespan_loader_failure_is_reported():
    def boom():
        raise RuntimeError("nope")
    ingest = IngestState()
    app = create_app(ingest)
    events = install_events(app)
    st = install_refinement(app, events, ingest, Worker(), runner_factory=boom, config=CFG)
    with TestClient(app) as client:
        st.loader.join(timeout=5)
        body = client.get("/refinement/status").json()
        assert body["models"] == "failed" and body["audio_applied"] is False


def test_stale_result_is_never_published_selection_change():
    async def go():
        holder = {}

        def runner(job):
            holder["rig"].events.selection.clear()      # user deselects while the job runs
            return RefineOutcome(RefineState.accepted, "ok", track_id="f0", score=9.0, assoc_conf=0.9)
        rig = holder["rig"] = Rig(runner=runner)
        await rig.choose()
        await rig.observe()
        msgs = rig.targets()
        assert msgs and all(m["kind"] == "speech" and m["assoc_conf"] == 0.0 for m in msgs)  # pending only
        hist = rig.st.bridge.status_payload()["history"]
        assert hist[0]["state"] == "stale" and hist[0]["reason"] == "selection changed"
        assert hist[0]["assoc_conf"] == 0.0
    run(go())


def test_new_pose_epoch_makes_the_old_job_stale():
    async def go():
        holder = {}

        def runner(job):
            pose(holder["rig"].ingest, 9_000_000, epoch=5)   # headset reconnect mid-job
            return RefineOutcome(RefineState.accepted, "ok", track_id="f0", score=9.0, assoc_conf=0.9)
        rig = holder["rig"] = Rig(runner=runner)
        await rig.choose()
        await rig.observe()
        assert all(m["kind"] == "speech" for m in rig.targets())
        hist = rig.st.bridge.status_payload()["history"]
        assert hist[0]["state"] == "stale" and hist[0]["reason"] == "epoch changed"
    run(go())
