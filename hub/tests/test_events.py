"""Headset-free evidence: mocked processing, simulated audio, real socket replay."""

import asyncio
import socket
import threading
import time
from contextlib import contextmanager

import httpx
import numpy as np
import uvicorn
import websockets
from pydantic import TypeAdapter
from scipy.io import wavfile

import protocol as p
from events import EventsState, install_events
from events.server import _snapshot
from ingest import IngestState, create_app
from locate.estimator import BearingEstimate
from locate.live import LiveLocator
from mockclient import Session
from mockclient.replay import replay
from mockclient.simulate import quat_yaw_pitch, simulate
from separate.live import StreamChunk
from separate.worker import SeparationWorker
from track import TrackConfig, Tracker

EVENT = TypeAdapter(p.HubEvent)


def pose(state, ts, epoch=0, yaw=0):
    state.on_pose_channel(p.PoseSample(
        seq=state.stats['pose'].messages, device_ts_us=ts, epoch=epoch,
        pos=(0, 0, 0), rot_xyzw=quat_yaw_pitch(yaw, 0)))


def raw(slot, bearing, ts, label='speech'):
    return p.Source(source_id=slot, label=label, label_conf=1, level_db=-20,
                    bearing_deg=bearing, bearing_conf=1 if bearing is not None else 0,
                    last_observed_us=ts)


async def feed(events, mono, sources, epoch=0):
    future = asyncio.get_running_loop().create_future()
    future.set_result((epoch, sources))
    events._snapshot_task = future
    await events.tick(mono)
    return EVENT.validate_json(events.latest.model_dump_json()).sources


def test_slot_swap_pause_and_no_hops_age_tracks():
    async def go():
        ingest = IngestState()
        pose(ingest, 1_000_000, yaw=50)
        events = EventsState(ingest, None)
        for k in range(3):
            ts = (k + 1) * 1_000_000
            sources = await feed(events, k, [raw('s1', -60, ts), raw('s2', 50, ts)])
        ids = {s.bearing_deg: s.source_id for s in sources}
        assert len(ids) == 2
        assert {s.bearing_deg: s.in_view for s in sources} == {-60: False, 50: True}
        sources = await feed(events, 3, [raw('s2', -60, 4_000_000), raw('s1', 50, 4_000_000)])
        assert {s.bearing_deg: s.source_id for s in sources} == ids
        # A frozen loud snapshot must not extend observation time on repeated ticks.
        sources = await feed(events, 5, [raw('s2', -60, 4_000_000), raw('s1', 50, 6_000_000)])
        assert next(s for s in sources if s.bearing_deg == -60).state == p.TrackState.last_seen
        sources = await feed(events, 6, [raw('s1', -60, 7_000_000), raw('s2', 50, 7_000_000)])
        assert {s.bearing_deg: s.source_id for s in sources} == ids
        assert all(s.state == p.TrackState.tracking for s in sources)
        await events.tick(8)
        assert all(s.state == p.TrackState.last_seen for s in events.latest.sources)
        await events.tick(15)
        assert all(s.state == p.TrackState.lost for s in events.latest.sources)
        assert all(not s.in_view for s in events.latest.sources)
    asyncio.run(go())


def test_capture_latency_and_bearingless_confirmation_use_unique_observations():
    async def go():
        ingest = IngestState()
        pose(ingest, 3_000_000)
        events = EventsState(ingest, None)
        for k in range(15):
            source = raw('s1', None, 1_000_000 + k * 1_000_000)
            got = await feed(events, k, [source])  # 2 seconds of model latency
            if k < 14:
                assert got == []
            # Ten repeats of one hop cannot count as ten confirmation hits.
            for n in range(1, 10):
                await feed(events, k + n / 10, [source])
        assert len(events.latest.sources) == 1
        assert events.latest.sources[0].source_id == 't1'
    asyncio.run(go())


def test_pose_outage_delayed_resume_and_epoch_discard_pending_snapshot():
    async def go():
        ingest = IngestState()
        pose(ingest, 10_000_000)
        events = EventsState(ingest, None, Tracker(TrackConfig(confirm_hits=1)))
        got = await feed(events, 0, [raw('s1', 0, 10_000_000)])
        first = got[0].source_id
        await events.tick(5)
        pose(ingest, 10_100_000)  # delayed, but the device clock did not go backwards
        await events.tick(5.1)
        pose(ingest, 10_200_000)
        await events.tick(5.2)
        assert events.latest.sources[0].source_id == first
        pose(ingest, 16_000_000)
        got = await feed(events, 6, [raw('s1', 0, 16_000_000)])
        assert got[0].source_id == first
        pose(ingest, 100_000, epoch=1)
        assert await feed(events, 7, [raw('s1', 0, 16_000_000)], epoch=0) == []
        got = await feed(events, 7.1, [raw('s1', 0, 100_000)], epoch=1)
        assert got[0].source_id != first
    asyncio.run(go())


def test_busy_locator_snapshot_is_bounded():
    ingest = IngestState()
    locator = LiveLocator(ingest)
    with locator.lock:
        start = time.monotonic()
        assert _snapshot(locator) is None
        assert time.monotonic() - start < 0.5


def test_clock_rollback_future_snapshot_does_not_poison_slot():
    async def go():
        ingest = IngestState()
        pose(ingest, 16_000_000)
        events = EventsState(ingest, None, Tracker(TrackConfig(confirm_hits=1)))
        before = await feed(events, 0, [raw('s1', 0, 16_000_000)])
        pose(ingest, 100_000)  # clock rollback without an epoch increment
        assert await feed(events, 1, [raw('s1', 0, 16_000_000)]) == []
        after = await feed(events, 1.1, [raw('s1', 0, 150_000)])
        assert len(after) == 1
        assert after[0].source_id != before[0].source_id
    asyncio.run(go())


def test_blocked_sender_does_not_block_commands_ticks_or_shutdown():
    class BlockedSocket:
        def __init__(self):
            self.incoming = asyncio.Queue()
            self.started = asyncio.Event()

        async def accept(self):
            pass

        async def send_text(self, text):
            self.started.set()
            await asyncio.Future()  # transport backpressure forever

        async def receive(self):
            return await self.incoming.get()

    async def go():
        app = create_app()
        events = install_events(app)
        endpoint = next(r.endpoint for r in app.routes if r.path == p.WS_EVENTS)
        async with app.router.lifespan_context(app):
            ws = BlockedSocket()
            connection = asyncio.create_task(endpoint(ws))
            await asyncio.wait_for(ws.started.wait(), 1)
            await ws.incoming.put({'type': 'websocket.receive', 'text': p.SetMode(mode='fast').model_dump_json()})
            before = events.latest.device_ts_us
            await asyncio.sleep(0.35)
            assert events.latest.device_ts_us - before >= 200_000
            assert events.mode == p.Mode.fast
            assert len(events.clients) == 1
            queue = next(iter(events.clients))
            assert queue.qsize() == 1
            assert EVENT.validate_json(queue.get_nowait()) == events.latest
            await ws.incoming.put({'type': 'websocket.disconnect'})
            await asyncio.wait_for(connection, 1)
            assert not events.clients and not events._client_tasks
            # Also exercise lifespan cancellation while send remains blocked.
            ws2 = BlockedSocket()
            connection2 = asyncio.create_task(endpoint(ws2))
            await asyncio.wait_for(ws2.started.wait(), 1)
        await asyncio.gather(connection2, return_exceptions=True)
        assert events.task is None and not events.clients and not events._client_tasks
    asyncio.run(go())


@contextmanager
def running_hub(app):
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    url = f'ws://127.0.0.1:{sock.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        yield url
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


def test_websocket_commands_validation_immediate_snapshot_and_tracks():
    app = create_app()
    events = install_events(app)
    with running_hub(app) as url:
        async def go():
            async with websockets.connect(url + p.WS_EVENTS) as ws:
                assert isinstance(EVENT.validate_json(await asyncio.wait_for(ws.recv(), 1)), p.SourcesUpdated)
                for text in ['{', '{"type":"cmd.mode","mode":"bad"}', '{"type":"unknown"}',
                             '{"type":"cmd.volumes","inner_db":NaN}']:
                    await ws.send(text)
                await ws.send(b'not a text command')
                await ws.send(p.SetMode(mode='accurate').model_dump_json())
                await ws.send(p.SetVolumes(inner_db=-3, outer_db=-18).model_dump_json())
                await ws.send(p.AckAlert(alert_id='a1').model_dump_json())
                for _ in range(3):
                    EVENT.validate_json(await asyncio.wait_for(ws.recv(), 1))
                assert events.bad_commands == 5
                assert events.mode == p.Mode.accurate
                assert (events.inner_db, events.outer_db) == (-3, -18)
                assert events.acked_alert_ids == {'a1'}
                async with websockets.connect(url + p.WS_EVENTS) as second:
                    EVENT.validate_json(await asyncio.wait_for(second.recv(), 1))
            async with httpx.AsyncClient() as client:
                response = await client.get(url.replace('ws:', 'http:') + '/tracks')
                assert response.json() == []
        asyncio.run(go())
    assert not events.clients


class ScriptedEstimator:
    """Known world bearings isolate transport/tracking from acoustic accuracy."""
    def add_head_pose(self, *args):
        pass

    def add_levels(self, *args):
        pass

    def estimate(self, sid, t):
        bearings = [-60, 50, 150] if t < 4.5 else [50, -60, 150]
        bearing = bearings[int(sid[1:]) - 1]
        return BearingEstimate(sid, bearing, bearing, 1, 1, 'tracking', None, 30, 10)


class ScriptedLocator(LiveLocator):
    def _reset(self):
        super()._reset()
        self.est = ScriptedEstimator()


class ScriptedSeparator:
    """Mock model outputs: two voices plus music, one pause, then swapped slots.

    The real worker still requires replayed audio windows and delivers real
    StreamChunks to LiveLocator. This does not measure separation/localization.
    """
    W, H = 3200, 1600

    def reset(self):
        pass

    def step(self, x, ts):
        assert x.shape == (self.W, 2)
        t = ts / 1e6
        chunks = []
        for i in range(3):
            person = i if t < 4.5 else (1 - i if i < 2 else i)
            active = not (person == 0 and 3 <= t < 4.2)
            audio = np.full(self.H, 0.1 if active else 0, np.float32)
            chunks.append(StreamChunk(
                f's{i + 1}', 'speech' if i < 2 else 'music', 1, ts, audio,
                np.stack([audio, audio]), np.array([ts + 50_000]),
                np.full((1, 2, 5), -20.), np.array([float(active)])))
        return chunks


def test_simulated_two_voice_music_replay_events(tmp_path):
    t = np.arange(16000 * 6) / 16000
    specs = []
    for i, (frequency, bearing) in enumerate([(300, -60), (700, 50), (3000, 150)]):
        audio = 0.1 * np.sin(2 * np.pi * frequency * t)
        if i == 0:
            audio[(t >= 2) & (t < 3.2)] = 0
        path = tmp_path / f'stem{i}.wav'
        wavfile.write(path, 16000, audio.astype(np.float32))
        specs.append(f'{path}:{bearing}')
    simulate(tmp_path / 'session', specs)
    session = Session.load(tmp_path / 'session')
    ingest = IngestState()
    locator = ScriptedLocator(ingest)
    worker = SeparationWorker(ingest, ScriptedSeparator())
    worker.listeners.append(locator.on_chunks)
    app = create_app(ingest)
    events = install_events(app, locator, Tracker(TrackConfig(hold_s=0.5, lost_s=1.5)))
    worker.start()
    try:
        with running_hub(app) as url:
            async def go():
                async with (websockets.connect(url + p.WS_EVENTS) as ws,
                            websockets.connect(url + p.WS_EVENTS) as ws2):
                    first = EVENT.validate_json(await ws.recv())
                    assert first.sources == []
                    await ws2.recv()
                    replay_task = asyncio.create_task(replay(session, url))
                    messages, arrivals = [], []
                    for _ in range(75):
                        event = EVENT.validate_json(await asyncio.wait_for(ws.recv(), 1))
                        EVENT.validate_json(await asyncio.wait_for(ws2.recv(), 1))
                        arrivals.append(time.monotonic())
                        messages.append(event)
                    sent = await replay_task
                    rate = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
                    assert 8 <= rate <= 12, rate
                    ids = {}
                    states = {}
                    for event in messages:
                        for source in event.sources:
                            bearing = round(source.bearing_deg)
                            ids.setdefault(bearing, set()).add(source.source_id)
                            states.setdefault(bearing, set()).add(source.state)
                    assert set(ids) == {-60, 50, 150}
                    assert all(len(v) == 1 for v in ids.values()), ids
                    assert all(s.state == p.TrackState.lost for s in messages[-1].sources)
                    assert p.TrackState.last_seen in states[-60]
                    assert any(e.device_ts_us > 5_500_000 and any(
                        round(s.bearing_deg) == -60 and s.state == p.TrackState.tracking
                        for s in e.sources) for e in messages)
                    assert sent['audio'] == 300 and worker.listener_errors == 0
                    print(f'replay: {rate:.2f} Hz, {len(messages)} valid events/client, ids={ids}')
            asyncio.run(go())
    finally:
        worker.stop()
