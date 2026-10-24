"""Cocktail hub, P0 step 1: unprocessed audio loopback with per-stage timing.

Run: uvicorn hub.main:app --port 8000
"""

import asyncio
import logging
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from hub import protocol
from hub.audio import OUT_CHANNELS, OUT_RATE, LoopbackPipeline, OutFrame
from hub.stats import LoopbackStats

log = logging.getLogger("hub")

OUT_QUEUE_FRAMES = 5  # 100 ms per listener; older frames are dropped beyond this


class Hub:
    def __init__(self):
        self.pipeline = LoopbackPipeline()
        self.stats = LoopbackStats()
        self.listeners: set[asyncio.Queue] = set()
        self.ingest_gen = 0  # a new ingest connection takes over from the old one
        self.out_seq = 0

    def publish(self, frame: OutFrame) -> None:
        self.stats.frames_out += 1
        if not self.listeners:
            self.stats.frames_unheard += 1
            return
        header = protocol.AudioHeader(
            seq=self.out_seq,
            epoch=frame.ref_epoch,
            pts_us=frame.ref_pts_us,
            sample_rate=OUT_RATE,
            channels=OUT_CHANNELS,
            ref_seq=frame.ref_seq,
        )
        self.out_seq += 1
        item = (protocol.pack(header, frame.pcm), frame.ref_recv_ns)
        for q in self.listeners:
            if q.full():
                q.get_nowait()
                self.stats.frames_dropped += 1
            q.put_nowait(item)


async def _wait_disconnect(ws: WebSocket) -> None:
    while (await ws.receive())["type"] != "websocket.disconnect":
        pass


def create_app() -> FastAPI:
    app = FastAPI(title="Cocktail hub")
    hub = Hub()
    app.state.hub = hub

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/stats")
    def stats():
        return hub.stats.snapshot(listeners=len(hub.listeners))

    @app.websocket("/ingest/audio")
    async def ingest_audio(ws: WebSocket):
        await ws.accept()
        hub.ingest_gen += 1
        gen = hub.ingest_gen
        hub.pipeline.reset()
        hub.stats.on_connect()
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                if gen != hub.ingest_gen:
                    await ws.close(code=1008, reason="replaced by a newer ingest connection")
                    return
                recv_ns = time.monotonic_ns()
                data = msg.get("bytes")
                if data is None:
                    await ws.close(code=1003, reason="binary audio frames only")
                    return
                try:
                    header, payload = protocol.unpack(data)
                except protocol.FrameError as e:
                    hub.stats.bad_messages += 1
                    log.warning("dropped ingest message: %s", e)
                    continue
                hub.stats.on_chunk(header, len(payload) // header.frame_bytes, recv_ns)
                frames = hub.pipeline.process(header, payload, recv_ns)
                hub.stats.process_ms.append((time.monotonic_ns() - recv_ns) / 1e6)
                for frame in frames:
                    hub.publish(frame)
        except WebSocketDisconnect:
            pass

    @app.websocket("/audio/out")
    async def audio_out(ws: WebSocket):
        q: asyncio.Queue = asyncio.Queue(maxsize=OUT_QUEUE_FRAMES)
        hub.listeners.add(q)  # before accept, so frames are queued once the client is connected
        try:
            await ws.accept()
            closed = asyncio.create_task(_wait_disconnect(ws))
            while True:
                get = asyncio.create_task(q.get())
                done, _ = await asyncio.wait({get, closed}, return_when=asyncio.FIRST_COMPLETED)
                if closed in done:
                    get.cancel()
                    return
                data, ref_recv_ns = get.result()
                await ws.send_bytes(data)
                hub.stats.frames_sent += 1
                hub.stats.hub_latency_ms.append((time.monotonic_ns() - ref_recv_ns) / 1e6)
        except WebSocketDisconnect:
            pass
        finally:
            hub.listeners.discard(q)

    return app


app = create_app()
