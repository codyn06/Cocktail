"""Authoritative selection state on /events: state.changed + target.updated published on change only."""

import asyncio
import json

from fastapi.testclient import TestClient

import protocol as p
from events import install_events
from ingest import IngestState, create_app
from mockclient.simulate import quat_yaw_pitch
from track import TrackConfig, Tracker


def pose(ingest, ts, epoch=0, yaw=0):
    ingest.on_pose_channel(p.PoseSample(
        seq=ingest.stats["pose"].messages, device_ts_us=ts, epoch=epoch,
        pos=(0, 0, 0), rot_xyzw=quat_yaw_pitch(yaw, 0)))


def raw(slot, bearing, ts):
    return p.Source(source_id=slot, label="speech", label_conf=1, level_db=-20, bearing_deg=bearing,
                    bearing_conf=1, last_observed_us=ts)


def ev(action, source_id=None):
    return p.InputEvent(seq=0, device_ts_us=0, epoch=0, action=action, source_id=source_id)


class Rig:
    def __init__(self):
        self.ingest = IngestState()
        pose(self.ingest, 1_000_000)
        self.app = create_app(self.ingest)
        self.events = install_events(self.app, None, Tracker(TrackConfig(confirm_hits=1)))
        self.src_q, self.aux_q = asyncio.Queue(maxsize=1), asyncio.Queue(maxsize=64)
        self.events.clients.add(self.src_q)
        self.events.aux_clients[self.src_q] = self.aux_q
        self.k = 0

    async def tick(self, sources=None):
        self.k += 1
        ts = 1_000_000 + (self.k - 1) * 100_000
        if sources is not None:
            fut = asyncio.get_running_loop().create_future()
            fut.set_result((self.events._epoch if self.events._epoch is not None else self.ingest.pose_epoch,
                            [raw(slot, b, ts) for slot, b in sources]))
            self.events._snapshot_task = fut
        await self.events.tick(self.k)

    def drain(self):
        out = []
        while not self.aux_q.empty():
            out.append(json.loads(self.aux_q.get_nowait()))
        return out

    def select(self, sid):
        self.events.selection.handle(ev(p.Action.select, sid),
                                     self.events.latest.sources, None, self.events.tracker)

    def ids(self):
        return sorted(s.source_id for s in self.events.latest.sources)


def states(msgs):
    return [m for m in msgs if m["type"] == "state.changed"]


def run(coro):
    return asyncio.run(coro)


def test_select_publishes_playing_once_then_not_again():
    async def go():
        r = Rig()
        await r.tick([("s1", -60)])
        await r.tick([("s1", -60)])
        first = r.drain()
        assert [m["state"] for m in states(first)] == ["listening"]
        sid = r.ids()[0]
        r.select(sid)
        await r.tick([("s1", -60)])
        msgs = r.drain()
        st = states(msgs)
        assert len(st) == 1 and st[0]["state"] == "playing" and st[0]["target_id"] == f"t_{sid}"
        assert p.StateChanged.model_validate_json(json.dumps(st[0])).selection_version == r.events.selection.version
        tgt = [m for m in msgs if m["type"] == "target.updated"]
        assert len(tgt) == 1 and tgt[0]["source_id"] == sid and tgt[0]["assoc_conf"] == 0.0
        p.TargetUpdated.model_validate_json(json.dumps(tgt[0]))
        for _ in range(3):
            await r.tick([("s1", -60)])
        assert r.drain() == []
    run(go())


def test_switch_stop_and_lost():
    async def go():
        r = Rig()
        await r.tick([("s1", -60), ("s2", 60)])
        a, b = r.ids()
        r.select(a)
        await r.tick([("s1", -60), ("s2", 60)])
        r.drain()
        v = r.events.selection.version
        r.select(b)
        await r.tick([("s1", -60), ("s2", 60)])
        st = states(r.drain())
        assert len(st) == 1 and st[0]["target_id"] == f"t_{b}" and st[0]["selection_version"] > v
        r.k = 4  # advance the hub clock: unheard > hold_s
        await r.tick()
        st = states(r.drain())
        assert [m["state"] for m in st] == ["selected"]
        r.k = 14  # unheard > lost_s
        await r.tick()
        assert [m["state"] for m in states(r.drain())] == ["target_lost"]
        r.events.selection.handle(ev(p.Action.stop), [], None)
        await r.tick()
        st = states(r.drain())
        assert [m["state"] for m in st] == ["listening"] and st[0]["target_id"] is None
        assert "hub.target" not in r.events.aux_snapshot
    run(go())


def test_epoch_reset_publishes_listening():
    async def go():
        r = Rig()
        await r.tick([("s1", -60)])
        r.select(r.ids()[0])
        await r.tick([("s1", -60)])
        assert states(r.drain())[-1]["state"] == "playing"
        pose(r.ingest, 9_000_000, epoch=1)
        await r.tick()
        st = states(r.drain())
        assert [m["state"] for m in st] == ["listening"]
    run(go())


def test_late_client_gets_current_state_and_selection_endpoint():
    r = Rig()

    async def go():
        await r.tick([("s1", -60)])
        r.select(r.ids()[0])
        await r.tick([("s1", -60)])
    run(go())
    sid = r.events.selection.chosen
    assert sid is not None
    client = TestClient(r.app)  # no lifespan: the hub clock must not tick on
    if True:
        body = client.get("/selection").json()
        assert body["chosen"] == sid and body["version"] == r.events.selection.version
        assert body["state"] == "tracking" and body["slots"]
        with client.websocket_connect(p.WS_EVENTS) as ws:
            seen = [json.loads(ws.receive_text()) for _ in range(3)]
        st = [m for m in seen if m["type"] == "state.changed"]
        assert st and p.StateChanged.model_validate(st[0]).state == p.HubState.playing
        assert st[0]["target_id"] == f"t_{sid}"
