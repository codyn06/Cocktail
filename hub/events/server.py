"""Latest-only 10 Hz events, driven by fresh locator observations."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

import protocol as p
from track import Tracker

log = logging.getLogger("events")
_command = TypeAdapter(p.Command)


def _snapshot(locator):
    """Copy immutable hop references, not audio arrays, without waiting on inference.

    LiveLocator.sources() takes its own lock and computes levels. Run it on a
    detached copy so neither that computation nor client I/O holds the live lock.
    A busy producer is retried next tick; only one snapshot task exists at a time.
    """
    if locator is None or not locator.lock.acquire(timeout=0.02):
        return None
    try:
        detached = copy.copy(locator)
        detached.latest = dict(locator.latest)
        detached.last_active_us = dict(locator.last_active_us)
        detached.lock = threading.Lock()
        epoch = locator._pose_epoch
    finally:
        locator.lock.release()
    return epoch, detached.sources()


@dataclass
class EventsState:
    ingest: object
    locator: object | None
    tracker: Tracker = field(default_factory=Tracker)
    mode: p.Mode = p.Mode.balanced
    inner_db: float = 0.0
    outer_db: float = -12.0
    acked_alert_ids: set[str] = field(default_factory=set)
    bad_commands: int = 0
    # Includes separator overlap, inference and scheduling latency; independent
    # of how long a track is held after its last delivered observation.
    max_observation_age_s: float = 5.0
    latest: p.SourcesUpdated = field(
        default_factory=lambda: p.SourcesUpdated(device_ts_us=0, sources=[]))
    clients: set = field(default_factory=set)
    _client_tasks: set = field(default_factory=set)
    task: asyncio.Task | None = None
    _snapshot_task: asyncio.Task | None = None
    _epoch: int | None = None
    _seen: dict[str, int] = field(default_factory=dict)
    _clock_key: tuple | None = None
    _last_device_us: int | None = None
    _anchor_us: int = 0
    _anchor_mono: float = field(default_factory=time.monotonic)

    def _clock(self, mono: float) -> tuple[int, float | None]:
        epoch = self.ingest.pose_epoch
        if epoch != self._epoch:
            self.tracker.reset()
            self._seen.clear()
            self._clock_key = None
            self._last_device_us = None
            self._anchor_us, self._anchor_mono = 0, mono
            self._epoch = epoch
        # Pose channels share the device clock; don't mix a previous connection's
        # audio or video timestamp into a newly restarted pose clock.
        stat = self.ingest.stats["pose"]
        key = (epoch, stat.messages)
        now = self._anchor_us + int((mono - self._anchor_mono) * 1e6)
        if key != self._clock_key and stat.messages:
            stamp = stat.last_device_ts_us
            if self._last_device_us is not None and stamp < self._last_device_us - 2_000_000:
                self.tracker.reset()
                self._seen.clear()
                now = stamp
            elif self._clock_key is None:
                now = stamp
            else:
                now = max(now, stamp)
            self._anchor_us, self._anchor_mono = now, mono
            self._clock_key = key
            self._last_device_us = stamp
        with self.ingest.poses.lock:
            yaw = self.ingest.poses.poses[-1].yaw_deg if self.ingest.poses.poses else None
        return now, yaw

    async def tick(self, mono: float | None = None) -> None:
        """Publish one tick; explicit monotonic seconds support deterministic replay."""
        now, yaw = self._clock(time.monotonic() if mono is None else mono)
        fresh = []
        task = self._snapshot_task
        if task is not None and task.done():
            self._snapshot_task = None
            try:
                snapshot = task.result()
            except Exception:
                log.exception("locator snapshot failed")
                snapshot = None
            if snapshot is not None and snapshot[0] == self._epoch:
                for source in snapshot[1]:
                    stamp = source.last_observed_us
                    if stamp <= self._seen.get(source.source_id, -1):
                        continue
                    # A delayed result must not resurrect a silent source, or
                    # import a future observation from a restarted device clock.
                    if 0 <= now - stamp <= int(self.max_observation_age_s * 1e6):
                        self._seen[source.source_id] = stamp
                        fresh.append(source)
        if self._snapshot_task is None and self.locator is not None:
            self._snapshot_task = asyncio.create_task(asyncio.to_thread(_snapshot, self.locator))
        self.latest = p.SourcesUpdated(
            device_ts_us=now, sources=self.tracker.update(now, fresh, head_yaw_deg=yaw))
        payload = self.latest.model_dump_json()
        for queue in tuple(self.clients):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(payload)

    async def run(self) -> None:
        try:
            while True:
                start = time.monotonic()
                await self.tick()
                await asyncio.sleep(max(0, 0.1 - (time.monotonic() - start)))
        finally:
            if self._snapshot_task is not None:
                # Snapshot lock waits are bounded. Await completion so no worker
                # remains attached to a closed lifespan or a subsequent startup.
                await asyncio.gather(self._snapshot_task, return_exceptions=True)
                self._snapshot_task = None

    def command(self, text: str) -> None:
        try:
            cmd = _command.validate_json(text)
            if isinstance(cmd, p.SetVolumes) and not all(
                    math.isfinite(v) for v in (cmd.inner_db, cmd.outer_db)):
                raise ValueError("non-finite volume")
        except (ValidationError, ValueError):
            self.bad_commands += 1
            # Don't log arbitrary client payloads or validation input values.
            log.warning("invalid events command rejected")
            return
        if isinstance(cmd, p.SetMode):
            self.mode = cmd.mode
        elif isinstance(cmd, p.SetVolumes):
            self.inner_db, self.outer_db = cmd.inner_db, cmd.outer_db
        elif isinstance(cmd, p.AckAlert):
            self.acked_alert_ids.add(cmd.alert_id)
        log.info("events command accepted: %s", cmd.type)


def install_events(app: FastAPI, locator=None, tracker: Tracker | None = None) -> EventsState:
    """Attach to the ingest app before startup; preserve its existing lifespan."""
    state = EventsState(app.state.ingest, locator, tracker or Tracker())
    app.state.events = state
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with previous_lifespan(app) as value:
            state.task = asyncio.create_task(state.run(), name="hub-events")
            try:
                yield value
            finally:
                state.task.cancel()
                await asyncio.gather(state.task, return_exceptions=True)
                state.task = None
                for task in tuple(state._client_tasks):
                    task.cancel()
                await asyncio.gather(*state._client_tasks, return_exceptions=True)
                state._client_tasks.clear()
                state.clients.clear()

    app.router.lifespan_context = lifespan

    @app.get("/tracks")
    async def tracks():
        return [source.model_dump(mode="json") for source in state.latest.sources]

    @app.websocket(p.WS_EVENTS)
    async def events(ws: WebSocket):
        await ws.accept()
        queue = asyncio.Queue(maxsize=1)
        state.clients.add(queue)
        queue.put_nowait(state.latest.model_dump_json())

        async def send():
            while True:
                await ws.send_text(await queue.get())

        async def receive():
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                text = message.get("text")
                if text is None:
                    state.bad_commands += 1
                    log.warning("non-text events command rejected")
                else:
                    state.command(text)

        tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
        state._client_tasks.update(tasks)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, OSError, RuntimeError):
            log.debug("events client disconnected")
        finally:
            state.clients.discard(queue)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            state._client_tasks.difference_update(tasks)

    return state
