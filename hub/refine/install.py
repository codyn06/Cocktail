"""Wire the refinement bridge into the live hub (ingest + events + separation worker).

Refinement is evidence only: it never changes selection, route or /audio/out. It publishes:

* GET /refinement/status: ``bridge.status_payload()`` plus ``{"models": "loading|ready|failed", "audio_applied": false}``.
* ``target.updated`` (protocol ``TargetUpdated``) on /events via ``EventsState.publish_aux`` for the CURRENT
  status only (stale results are never published). Keyed by target_id so a newly connected client gets the latest.

Mapping of the 5 refinement states onto TargetUpdated (``target_id = t_<source_id>``):

  accepted    -> kind "person", assoc_conf = status.assoc_conf (in (0, 1])
  pending     -> kind "speech", assoc_conf 0.0
  refused     -> kind "speech", assoc_conf 0.0
  unavailable -> kind "speech", assoc_conf 0.0
  stale       -> never published

pending/refused/unavailable are therefore indistinguishable on /events; only /refinement/status separates them
(the protocol has no field for it; the proposed field is NOT applied). ``voice_lock_conf`` is always None.
``state`` is the tracker's TrackState for the source, ``bearing_deg`` comes from the latest sources.updated.

The runner (models) loads in a daemon thread started by the lifespan. Until then the runner answers
unavailable("refinement models loading"); if loading raises, unavailable("refinement models failed to load: <type>").
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable

from fastapi import FastAPI

import protocol as p

from .bridge import RefinementBridge
from .contracts import LiveView, RefineConfig, RefineJob, RefineOutcome, RefineState, RefineStatus
from .snapshot import SlotRing, install_frame_tap, make_cut_fn

log = logging.getLogger("refine.install")


@dataclass
class RefinementState:
    bridge: RefinementBridge
    ring: SlotRing
    config: RefineConfig
    models: str = "loading"          # loading | ready | failed
    models_error: str | None = None
    loop: asyncio.AbstractEventLoop | None = None
    loader: threading.Thread | None = None
    _runner: Callable[[RefineJob], RefineOutcome] | None = field(default=None, repr=False)

    def run(self, job: RefineJob) -> RefineOutcome:
        """Runner given to the bridge: delegates to the loaded runner, else reports why not."""
        runner = self._runner
        if runner is not None:
            return runner(job)
        if self.models == "failed":
            return RefineOutcome(RefineState.unavailable, f"refinement models failed to load: {self.models_error}")
        return RefineOutcome(RefineState.unavailable, "refinement models loading")

    def post(self, fn: Callable[[], None]) -> None:
        loop = self.loop
        if loop is None:
            fn()
        else:
            loop.call_soon_threadsafe(fn)

    def status_payload(self) -> dict:
        out = self.bridge.status_payload()
        out.update(models=self.models, audio_applied=False)
        if self.models_error:
            out["models_error"] = self.models_error
        return out


def install_refinement(app: FastAPI, events, ingest, worker, *, runner_factory: Callable[[], Callable],
                       config: RefineConfig | None = None, label_fn: Callable[[str], str | None] | None = None,
                       executor: str = "thread") -> RefinementState:
    """Attach refinement to the hub app before startup; preserves the existing lifespan."""
    config = config or RefineConfig()
    install_frame_tap(ingest)
    ring = SlotRing(max(8.0, config.window_s))
    worker.listeners.append(ring.on_chunks)

    def live() -> LiveView:
        sel = events.selection
        chosen = sel.chosen
        route = sel.route(events.tracker) if chosen is not None else None
        audio = ingest.audio
        return LiveView(
            chosen=chosen, epoch=ingest.pose_epoch, selection_version=sel.version,
            audio_resets=(audio.resets if audio is not None else 0) or 0,
            slots=tuple(route.slots) if route else (),
            source_present=chosen is not None and chosen in events.tracker.tracks,
            now_us=events.latest.device_ts_us)

    def default_label(source_id: str) -> str | None:
        return next((s.label for s in events.latest.sources if s.source_id == source_id), None)

    cut = make_cut_fn(ingest, ring, label_fn or default_label, config)
    holder: dict = {}
    bridge = RefinementBridge(live_fn=live, cut_fn=cut, runner=lambda job: holder["st"].run(job), config=config,
                              post=lambda fn: holder["st"].post(fn), executor=executor)
    st = RefinementState(bridge, ring, config)
    holder["st"] = st
    events.tick_hooks.append(bridge.tick)

    def publish(status: RefineStatus) -> None:
        if status.state == RefineState.stale:
            return
        sid = status.key.source_id
        acc = status.state == RefineState.accepted
        src = next((s for s in events.latest.sources if s.source_id == sid), None)
        track = events.tracker.tracks.get(sid)
        tstate = src.state if src is not None else (track.state if track is not None else p.TrackState.lost)
        msg = p.TargetUpdated(
            target_id=f"t_{sid}", selection_version=status.key.selection_version, source_id=sid,
            kind="person" if acc else "speech", voice_lock_conf=None,
            assoc_conf=min(1.0, max(0.0, status.assoc_conf)) if acc else 0.0,
            state=tstate, bearing_deg=src.bearing_deg if src is not None else None)
        events.publish_aux(msg.model_dump_json(), key=msg.target_id)

    bridge.listeners.append(publish)

    def load() -> None:
        try:
            st._runner = runner_factory()
            st.models = "ready"
        except Exception as e:  # noqa: BLE001
            st.models, st.models_error = "failed", type(e).__name__
            log.warning("refinement models failed to load: %s", type(e).__name__)

    previous = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with previous(app) as value:
            st.loop = asyncio.get_running_loop()
            st.loader = threading.Thread(target=load, name="refine-load", daemon=True)
            st.loader.start()
            try:
                yield value
            finally:
                st.loop = None
                await asyncio.to_thread(bridge.close)

    app.router.lifespan_context = lifespan
    app.state.refinement = st

    @app.get("/refinement/status")
    async def refinement_status():
        return st.status_payload()

    return st
