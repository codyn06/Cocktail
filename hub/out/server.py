"""/audio/out: send the chosen source (inner) and the room (outer) to the headset."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import protocol as p

from .mixer import FRAME, SR_OUT, AudioOutConfig, ClientFormatter, LayerFrame, LayerMixer

log = logging.getLogger("out")
FRAME_S = FRAME / SR_OUT


@dataclass
class AudioOutState:
    events: object                      # events.EventsState: selection, tracker, volumes, head yaw
    config: AudioOutConfig = field(default_factory=AudioOutConfig)
    mixer: LayerMixer = None            # type: ignore[assignment]
    clients: set = field(default_factory=set)
    task: asyncio.Task | None = None
    last_inner_db: float | None = None
    last_outer_db: float | None = None
    bad_layouts: int = 0

    def __post_init__(self):
        self.mixer = self.mixer or LayerMixer(lambda: self.events.selection.route(self.events.tracker), self.config)

    def on_chunks(self, chunks) -> None:
        """SeparationWorker listener (worker thread)."""
        self.mixer.push(chunks)

    def tick(self, now: float | None = None) -> LayerFrame | None:
        """Pull one due frame and hand it to every client queue (latest few kept)."""
        frame = self.mixer.pull(now)
        if frame is None:
            return None
        rms = lambda x: float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9))
        self.last_inner_db, self.last_outer_db = rms(frame.inner), rms(frame.outer)
        for q in tuple(self.clients):
            if q.full():
                q.get_nowait()
            q.put_nowait(frame)
        return frame

    async def run(self) -> None:
        nxt = time.monotonic()
        while True:
            now = time.monotonic()
            if now - nxt > 0.2:           # fell behind (blocked loop): resync instead of bursting
                nxt = now
            while nxt <= now:
                self.tick(now)
                nxt += FRAME_S
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))

    def status(self) -> dict:
        sel, m = self.events.selection, self.mixer
        route = sel.route(self.events.tracker)
        return {
            "chosen": sel.chosen, "selection_version": sel.version,
            "slots": list(route.slots) if route else [],
            "inner_db": self.events.inner_db, "outer_db": self.events.outer_db,
            "queued_ms": round(m.queued_s * 1000), "settle_ms": round(self.config.settle_s * 1000),
            "hops_in": m.hops_in, "hops_dropped": m.hops_dropped, "underruns": m.underruns,
            "frames_out": m.frames_out, "clients": len(self.clients),
            "last_inner_rms_db": self.last_inner_db, "last_outer_rms_db": self.last_outer_db,
        }


def install_audio_out(app: FastAPI, events, config: AudioOutConfig | None = None) -> AudioOutState:
    """Attach /audio/out to the ingest+events app; preserves the existing lifespan.

    Feed it from the separation worker: `worker.listeners.append(state.on_chunks)`."""
    state = AudioOutState(events, config or AudioOutConfig())
    app.state.audio_out = state
    previous = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with previous(app) as value:
            state.task = asyncio.create_task(state.run(), name="hub-audio-out")
            try:
                yield value
            finally:
                state.task.cancel()
                await asyncio.gather(state.task, return_exceptions=True)
                state.task = None
                state.clients.clear()

    app.router.lifespan_context = lifespan

    @app.get("/audio/out/status")
    async def audio_out_status():
        return state.status()

    @app.websocket(p.WS_AUDIO_OUT)
    async def audio_out(ws: WebSocket):
        """?layout=layers (default, what the Quest expects) or ?layout=stereo (hub applies gains + pan)."""
        layout = ws.query_params.get("layout", state.config.default_layout)
        try:
            fmt = ClientFormatter(layout, state.config)
        except ValueError:
            state.bad_layouts += 1
            await ws.close(code=1008)
            return
        await ws.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        state.clients.add(queue)

        async def drain():           # the client never sends; notice a disconnect
            while True:
                if (await ws.receive())["type"] == "websocket.disconnect":
                    return

        reader = asyncio.create_task(drain())
        try:
            while True:
                get = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait({get, reader}, return_when=asyncio.FIRST_COMPLETED)
                if get not in done:
                    get.cancel()
                    break
                ev = state.events
                await ws.send_bytes(fmt.format(get.result(), ev.inner_db, ev.outer_db, ev.head_yaw()))
        except (WebSocketDisconnect, OSError, RuntimeError):
            log.debug("audio/out client disconnected")
        finally:
            state.clients.discard(queue)
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    return state
