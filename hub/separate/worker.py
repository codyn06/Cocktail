"""Background thread that runs LiveSeparator on the ingest audio ring (task C3)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from ingest import IngestState

from .live import LiveSeparator, StreamChunk

log = logging.getLogger("separate")


class SeparationWorker:
    def __init__(self, ingest: IngestState, live: LiveSeparator, keep: int = 10, max_lag_s: float = 3.0):
        self.ingest, self.live = ingest, live
        self.chunks: deque[list[StreamChunk]] = deque(maxlen=keep)  # newest hop last
        self.compute_ms: deque[float] = deque(maxlen=100)
        self.listeners: list = []  # callables(list[StreamChunk]); e.g. hub/locate
        self._next = None  # absolute ring index of the next window start
        self._ring = None
        self._resets = -1
        self.max_lag = int(max_lag_s * 16000)  # samples at 16 kHz
        self.lag_s = 0.0
        self.skips = 0
        self.listener_errors = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="separate", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        self.thread.join(5)

    def poll(self) -> bool:
        """Process one window if a full new one is available. Returns True if it did."""
        ring = self.ingest.audio
        if ring is None:
            return False
        if ring is not self._ring or ring.resets != self._resets:
            self._ring, self._resets, self._next = ring, ring.resets, None  # new ring or reset
            self.live.reset()
        W, H = self.live.W, self.live.H
        if self._next is None:
            if ring.written < W:
                return False
            self._next = ring.written - W
        self.lag_s = max(0, ring.written - (self._next + W)) / 16000
        if ring.written - (self._next + W) > self.max_lag:
            log.warning("separation %.1f s behind live; skipping ahead", self.lag_s)
            self.skips += 1
            self._next = ring.written - W
            self.live.reset()
        got = ring.read(self._next, W)
        if got is None:
            return False  # not enough new audio yet
        x, ts = got
        t = time.perf_counter()
        out = self.live.step(x, ts)
        self._next += H  # advance before anything else can fail: never re-run a window
        self.compute_ms.append((time.perf_counter() - t) * 1000)
        self.chunks.append(out)
        for f in self.listeners:
            try:
                f(out)
            except Exception:
                self.listener_errors += 1
                log.exception("separation listener failed")
        return True

    def _run(self):
        while not self._stop.is_set():
            try:
                if not self.poll():
                    time.sleep(0.01)
            except Exception:
                log.exception("separation step failed")
                time.sleep(0.1)
