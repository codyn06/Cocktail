"""Mock hub for Quest development: speaks the frozen protocol, does no processing.

Run from the repo root:
    cd hub && uv sync --python 3.12 && cd ..
    uv run --project hub python quest/tools/mock_hub.py [--port 8765]

Then point the headset app at ws://<laptop-ip>:8765. It accepts all ingest sockets and
prints what it receives each second; /events sends three fake sources (two people and
music) that drift in bearing and answers select/stop/replay/next with the matching state
events; /audio/out streams a 48 kHz stereo test signal (inner layer: 440 Hz tone,
outer layer: quiet noise) while a target is selected.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hub"))

import uvicorn  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

import protocol as p  # noqa: E402

app = FastAPI(title="Cocktail mock hub", version=p.PROTOCOL_VERSION)
T0 = time.monotonic()
counts = {"audio": 0, "video": 0, "pose": 0, "input": 0, "audio_bytes": 0, "video_bytes": 0}
last_pose: dict = {}
events: list[asyncio.Queue] = []

BASE = [("s_1", "speech", -35.0), ("s_2", "speech", 25.0), ("s_3", "music", 140.0)]
sel = {"target": None, "version": 0, "mode": p.Mode.balanced}


def now_us() -> int:
    return int((time.monotonic() - T0) * 1e6)


def broadcast(ev) -> None:
    for q in events:
        q.put_nowait(ev.model_dump_json())


def sources_msg() -> p.SourcesUpdated:
    t = time.monotonic() - T0
    yaw = last_pose.get("yaw", 0.0)
    out = []
    for sid, label, bearing in BASE:
        b = bearing + 6 * math.sin(t / 3 + bearing)
        rel = ((b - yaw + 180) % 360) - 180
        out.append(
            p.Source(
                source_id=sid, label=label, label_conf=0.9,
                bearing_deg=b, bearing_conf=0.8, bearing_source=p.BearingSource.fused,
                state=p.TrackState.tracking,
                level_db=-30 + 5 * math.sin(t + b),
                in_view=abs(rel) < 45,
                bbox=(0.4, 0.3, 0.6, 0.8) if abs(rel) < 20 else None,
                last_observed_us=now_us(),
            )
        )
    return p.SourcesUpdated(device_ts_us=now_us(), sources=out)


def yaw_from_quat(q) -> float:
    x, y, z, w = q  # Unity left-handed: yaw about +y
    return math.degrees(math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x)))


@app.get("/health")
def health() -> dict:
    return {"ok": True, "protocol": p.PROTOCOL_VERSION, **counts}


async def drain(ws: WebSocket, kind: str) -> None:
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            data = msg.get("bytes")
            if data is not None:
                counts[kind] += 1
                counts[f"{kind}_bytes"] += len(data)
                try:
                    hdr, _ = p.unpack(data, p.AudioChunk if kind == "audio" else p.VideoFrame)
                    if hdr.seq == 0:
                        print(f"[{kind}] first frame: {hdr.model_dump()}")
                except Exception as e:  # noqa: BLE001
                    print(f"[{kind}] bad frame: {e}")
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket(p.WS_AUDIO_IN)
async def ws_audio_in(ws: WebSocket) -> None:
    await ws.accept()
    await drain(ws, "audio")


@app.websocket(p.WS_VIDEO_IN)
async def ws_video_in(ws: WebSocket) -> None:
    await ws.accept()
    await drain(ws, "video")


@app.websocket(p.WS_POSE_IN)
async def ws_pose_in(ws: WebSocket) -> None:
    await ws.accept()
    adapter = TypeAdapter(p.PoseChannelMessage)
    try:
        while True:
            m = adapter.validate_json(await ws.receive_text())
            if isinstance(m, p.Hello):
                print(f"[hello] {m.model_dump()}")
            elif isinstance(m, p.PoseSample):
                counts["pose"] += 1
                last_pose["yaw"] = yaw_from_quat(m.rot_xyzw)
            elif isinstance(m, p.InputEvent):
                counts["input"] += 1
                print(f"[input] {m.action} via {m.modality} source={m.source_id}")
                handle_input(m)
    except (WebSocketDisconnect, RuntimeError):
        pass


def handle_input(m: p.InputEvent) -> None:
    ids = [s[0] for s in BASE]
    if m.action in (p.Action.select, p.Action.next, p.Action.prev):
        cur = sel["target"]
        if m.action == p.Action.select:
            nxt = m.source_id or ids[0]
        else:
            step = 1 if m.action == p.Action.next else -1
            nxt = ids[((ids.index(cur) if cur in ids else -1) + step) % len(ids)]
        sel["target"], sel["version"] = nxt, sel["version"] + 1
        label, bearing = next((l, b) for i, l, b in BASE if i == nxt)
        tid = f"t_{nxt}"
        broadcast(p.StateChanged(state=p.HubState.selected, target_id=tid, selection_version=sel["version"], mode=sel["mode"]))
        broadcast(p.TargetUpdated(target_id=tid, selection_version=sel["version"], source_id=nxt,
                                  kind="person" if label == "speech" else label, assoc_conf=0.8,
                                  state=p.TrackState.tracking, bearing_deg=bearing))
        broadcast(p.StateChanged(state=p.HubState.playing, target_id=tid, selection_version=sel["version"], mode=sel["mode"]))
    elif m.action == p.Action.stop:
        sel["target"] = None
        broadcast(p.StateChanged(state=p.HubState.listening, mode=sel["mode"]))
    elif m.action == p.Action.replay and sel["target"]:
        tid, ver = f"t_{sel['target']}", sel["version"]
        broadcast(p.StateChanged(state=p.HubState.replay_processing, target_id=tid, selection_version=ver))

        async def done() -> None:
            await asyncio.sleep(2)
            broadcast(p.ReplayReady(target_id=tid, selection_version=ver, duration_s=5.0))
            broadcast(p.StateChanged(state=p.HubState.replay_playing, target_id=tid, selection_version=ver))
            await asyncio.sleep(5)
            broadcast(p.StateChanged(state=p.HubState.playing, target_id=tid, selection_version=ver))

        asyncio.create_task(done())


@app.websocket(p.WS_EVENTS)
async def ws_events(ws: WebSocket) -> None:
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue()
    events.append(q)
    cmd_adapter = TypeAdapter(p.Command)

    async def reader() -> None:
        while True:
            try:
                c = cmd_adapter.validate_json(await ws.receive_text())
            except WebSocketDisconnect:
                return
            if isinstance(c, p.SetMode):
                sel["mode"] = c.mode
                print(f"[cmd] mode={c.mode}")

    rtask = asyncio.create_task(reader())
    try:
        broadcast(p.StateChanged(state=p.HubState.listening, mode=sel["mode"]))
        while not rtask.done():
            await ws.send_text(sources_msg().model_dump_json())
            deadline = time.monotonic() + 0.1
            while (left := deadline - time.monotonic()) > 0:
                try:
                    await ws.send_text(await asyncio.wait_for(q.get(), timeout=left))
                except TimeoutError:
                    break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        events.remove(q)
        rtask.cancel()


@app.websocket(p.WS_AUDIO_OUT)
async def ws_audio_out(ws: WebSocket) -> None:
    await ws.accept()
    rng = np.random.default_rng(0)
    n0 = 0
    try:
        seq = 0
        nxt = time.monotonic()
        while True:
            n = np.arange(n0, n0 + 960)
            n0 += 960
            bearing = next((b for i, _, b in BASE if i == sel["target"]), None)
            if sel["target"] is None:
                inner = np.zeros(960)
            else:
                inner = 0.25 * np.sin(2 * np.pi * 440 * n / 48000)
            outer = 0.02 * rng.standard_normal(960)
            pcm = (np.stack([inner, outer], axis=1) * 32767).astype("<i2")
            hdr = p.AudioOut(seq=seq, device_ts_us=now_us(), target_bearing_deg=bearing)
            await ws.send_bytes(p.pack(hdr, pcm.tobytes()))
            seq += 1
            nxt += 0.02
            await asyncio.sleep(max(0, nxt - time.monotonic()))
    except (WebSocketDisconnect, RuntimeError):
        pass


async def report() -> None:
    prev = dict(counts)
    while True:
        await asyncio.sleep(1)
        d = {k: counts[k] - prev[k] for k in counts}
        prev = dict(counts)
        if any(d.values()):
            print(
                f"[rx/s] audio={d['audio']} ({d['audio_bytes']//1024} KB) "
                f"video={d['video']} ({d['video_bytes']//1024} KB) pose={d['pose']} input={d['input']}"
            )


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=p.DEFAULT_PORT)
    a = ap.parse_args()
    print(f"mock hub on ws://{a.host}:{a.port}  (health: http://localhost:{a.port}/health)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
