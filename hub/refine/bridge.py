"""Refinement bridge: cuts one job at a time for the chosen source and admits results only if still live.

Never changes selection, route or audio. ``tick`` runs on the event loop and never blocks; the runner runs on
one worker thread (or inline in tests); ``_admit`` runs back on the loop via ``post``.

Listeners receive a ``RefineStatus`` when the current status changes, and ``None`` when there is no longer a
current status for the previous key (deselect, or the live key changed). A key change delivers ``None`` first,
then the new pending status.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import asdict
from typing import Callable

from .contracts import (CutUnavailable, LiveView, RefineConfig, RefineJob, RefineKey, RefineOutcome,
                        RefineState, RefineStatus, Runner)

log = logging.getLogger("refine.bridge")

CUT_RETRY_S = 1.0
FUTURE_SLACK_US = 2_000_000


class RefinementBridge:
    def __init__(self, *, live_fn: Callable[[], LiveView], cut_fn: Callable[[LiveView, int], RefineJob],
                 runner: Runner | None, config: RefineConfig | None = None,
                 post: Callable[[Callable[[], None]], None] | None = None,
                 executor: str = "thread", mono: Callable[[], float] = time.monotonic) -> None:
        self._live_fn, self._cut_fn, self._runner = live_fn, cut_fn, runner
        self.config = config or RefineConfig()
        self._post = post or (lambda fn: fn())
        self._inline = executor == "inline"
        self._mono = mono
        self.listeners: list[Callable[[RefineStatus | None], None]] = []
        self._lock = threading.Lock()
        self._current: RefineStatus | None = None
        self._history: list[RefineStatus] = []
        self._key: RefineKey | None = None
        self._first_seen = 0.0
        self._last_attempt: float | None = None
        self._last_cut = 0.0
        self._admitted = False
        self._inflight = False
        self._next_id = 1
        self._q: queue.Queue[RefineJob | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        if not self._inline:
            self._thread = threading.Thread(target=self._worker_loop, name="refine-worker", daemon=True)
            self._thread.start()

    # ---- loop side ----
    def tick(self) -> None:
        """Called ~10 Hz on the loop. Tracks the live key and cuts at most one job; never runs a model."""
        v = self._live_fn()
        now = self._mono()
        if v.chosen is None:
            with self._lock:
                had = self._current is not None
                self._key, self._admitted, self._current = None, False, None
            if had:
                self._notify(None, force=True)
            return
        key = RefineKey(v.chosen, v.epoch, v.selection_version, v.audio_resets)
        dropped = False
        with self._lock:
            if key != self._key:
                dropped = self._current is not None
                self._key, self._first_seen, self._last_attempt = key, now, None
                self._admitted = False
                changed = self._set_current(RefineStatus(0, key, RefineState.pending, "waiting for window"))
            else:
                changed = None
        if dropped:
            self._notify(None, force=True)
        self._notify(changed)
        if now - self._first_seen < self.config.settle_s:
            return
        if self._runner is None:
            with self._lock:
                changed = self._set_current(RefineStatus(0, key, RefineState.unavailable,
                                                         "refinement models not loaded"))
            self._notify(changed)
            return
        with self._lock:
            if self._inflight:
                return
            if self._admitted and now - self._last_cut < self.config.refresh_s:
                return
            if not self._admitted and self._last_attempt is not None and now - self._last_attempt < CUT_RETRY_S:
                return
            self._last_attempt = now
            job_id = self._next_id
            self._next_id += 1
        try:
            job = self._cut_fn(v, job_id)
        except CutUnavailable as e:
            self._cut_failed(key, str(e))
            return
        except Exception as e:  # noqa: BLE001
            log.warning("cut failed: %s", type(e).__name__)
            self._cut_failed(key, f"cut error: {type(e).__name__}")
            return
        if job.created_mono <= 0:
            job.created_mono = now
        with self._lock:
            self._inflight = True
            self._last_cut = now
        if self._inline:
            self._run(job)
        else:
            self._q.put(job)

    def current(self) -> RefineStatus | None:
        with self._lock:
            return self._current

    def statuses(self) -> list[RefineStatus]:
        with self._lock:
            return list(self._history)

    def status_payload(self) -> dict:
        with self._lock:
            counts = {s.value: 0 for s in RefineState}
            for st in self._history:
                counts[st.state.value] += 1
            return {"current": self._current.to_dict() if self._current else None,
                    "history": [s.to_dict() for s in self._history], "counts": counts,
                    "in_flight": self._inflight, "config": asdict(self.config)}

    def close(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=2.0)

    # ---- helpers ----
    def _set_current(self, st: RefineStatus | None) -> RefineStatus | None:
        """Under lock. Returns st if it differs from current (caller notifies after releasing the lock)."""
        changed = st != self._current
        self._current = st
        return st if changed else None

    def _notify(self, st: RefineStatus | None, force: bool = False) -> None:
        if st is None and not force:
            return
        for fn in list(self.listeners):
            try:
                fn(st)
            except Exception:  # noqa: BLE001
                log.exception("refine listener failed")

    def _cut_failed(self, key: RefineKey, reason: str) -> None:
        with self._lock:
            if self._key != key or self._admitted:
                return
            changed = self._set_current(RefineStatus(0, key, RefineState.unavailable, reason))
        self._notify(changed)

    # ---- worker side ----
    def _worker_loop(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            self._run(job)

    def _run(self, job: RefineJob) -> None:
        try:
            outcome = self._runner(job)  # type: ignore[misc]
        except Exception as e:  # noqa: BLE001
            log.warning("runner failed: %s", type(e).__name__)
            outcome = RefineOutcome(RefineState.unavailable, f"runner error: {type(e).__name__}: {str(e)[:160]}")
        try:
            self._post(lambda: self._admit(job, outcome))
        except Exception:  # noqa: BLE001
            log.exception("refine post failed")
            with self._lock:
                self._inflight = False

    def _stale_reason(self, job: RefineJob, v: LiveView) -> str | None:
        k = job.key
        if v.chosen != k.source_id:
            return "selection changed"
        if v.epoch != k.epoch:
            return "epoch changed"
        if v.selection_version != k.selection_version:
            return "selection version changed"
        if v.audio_resets != k.audio_resets:
            return "audio reset"
        if not v.source_present:
            return "source gone"
        if any(s not in v.slots for s in job.slots):
            return "route moved"
        if v.now_us - job.capture_end_us > self.config.max_age_s * 1e6:
            return "result too old"
        if job.capture_end_us > v.now_us + FUTURE_SLACK_US:
            return "capture in the future"
        if job.capture_start_us < v.window_floor_us:
            return "route or selection changed inside the window"
        return None

    def _admit(self, job: RefineJob, outcome: RefineOutcome) -> None:
        """Loop side: decide with a fresh live view whether the finished job still counts."""
        v = self._live_fn()
        stale = self._stale_reason(job, v)
        state, reason = outcome.state, outcome.reason
        track, conf = outcome.track_id, outcome.assoc_conf
        if stale:
            state, reason = RefineState.stale, stale
        elif state == RefineState.accepted and (conf <= 0 or not track):
            state, reason = RefineState.refused, "accepted without track or confidence"
        if state != RefineState.accepted:
            track, conf = None, 0.0
        st = RefineStatus(
            job_id=job.job_id, key=job.key, state=state, reason=reason, track_id=track,
            score=outcome.score if state == RefineState.accepted else None, assoc_conf=conf,
            capture_start_us=job.capture_start_us, capture_end_us=job.capture_end_us,
            latency_ms=(self._mono() - job.created_mono) * 1000 if job.created_mono > 0 else None,
            compute_ms=dict(outcome.compute_ms), evidence=dict(outcome.evidence), audio_applied=False)
        with self._lock:
            self._inflight = False
            self._history.insert(0, st)
            del self._history[self.config.history:]
            changed = None
            if not stale:
                self._admitted = True
                changed = self._set_current(st)
        self._notify(changed)
