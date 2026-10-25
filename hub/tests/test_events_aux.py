"""Auxiliary events channel: not overwritten by sources ticks, bounded, snapshot on connect, tick hooks."""

import asyncio
import json

from fastapi.testclient import TestClient

import protocol as p
from events import EventsState, install_events
from ingest import IngestState, create_app


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_aux_is_not_overwritten_by_a_sources_tick():
    async def go():
        events = EventsState(IngestState(), None)
        src_q, aux_q = asyncio.Queue(maxsize=1), asyncio.Queue(maxsize=16)
        events.clients.add(src_q)
        events.aux_clients[src_q] = aux_q
        events.publish_aux('{"type":"target.updated","n":1}', key="t_a")
        for k in range(3):
            await events.tick(k)
        assert json.loads(src_q.get_nowait())["type"] == "sources.updated"
        assert src_q.empty()                       # sources stays latest-only
        assert drain(aux_q) == ['{"type":"target.updated","n":1}']
    asyncio.run(go())


def test_aux_queue_is_bounded_and_drops_oldest():
    events = EventsState(IngestState(), None)
    q = asyncio.Queue(maxsize=events.aux_maxsize)
    events.aux_clients[object()] = q
    for i in range(events.aux_maxsize + 5):
        events.publish_aux(str(i))
    got = drain(q)
    assert len(got) == events.aux_maxsize
    assert got[0] == "5" and got[-1] == str(events.aux_maxsize + 4)


def test_snapshot_keeps_latest_per_key():
    events = EventsState(IngestState(), None)
    events.publish_aux("a1", key="a")
    events.publish_aux("b1", key="b")
    events.publish_aux("a2", key="a")
    assert events.aux_snapshot == {"a": "a2", "b": "b1"}


def test_tick_hooks_run_each_tick_and_exceptions_are_swallowed():
    async def go():
        events = EventsState(IngestState(), None)
        calls = []
        events.tick_hooks.append(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        events.tick_hooks.append(lambda: calls.append(1))
        await events.tick(0)
        await events.tick(1)
        assert calls == [1, 1]
    asyncio.run(go())


def test_new_client_gets_sources_then_aux_snapshot_and_live_aux():
    ingest = IngestState()
    app = create_app(ingest)
    events = install_events(app)
    events.aux_snapshot["t_x"] = '{"type":"target.updated","target_id":"t_x"}'
    with TestClient(app) as client, client.websocket_connect(p.WS_EVENTS) as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "sources.updated"
        second = json.loads(ws.receive_text())
        assert second["target_id"] == "t_x"
        assert len(events.aux_clients) == 1
    assert not events.aux_clients


def test_drop_aux_forgets_key():
    events = EventsState(IngestState(), None)
    events.publish_aux("a1", key="a")
    events.publish_aux("b1", key="b")
    events.drop_aux("a")
    events.drop_aux("missing")
    assert events.aux_snapshot == {"b": "b1"}
