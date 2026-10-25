"""Device-time snapshots for refinement jobs: a frame tap, a per-slot audio ring and the job cut.

Everything a job holds is copied, so later ring writes cannot change it. The cut runs on the event loop, so
it does only slicing and copies (no decoding, no models).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from ingest.buffers import HUB_SR, LatestFrame

from .contracts import CutUnavailable, LiveView, RefineConfig, RefineJob, RefineKey

_BACKWARDS_RESET_US = 2_000_000   # device time going back further than this = new epoch / clock restart
_CONTIG_TOL_US = 5_000            # hop gap/overlap tolerated before a slot stream restarts


class TappedFrames(LatestFrame):
    """LatestFrame that also keeps the last `seconds` of (device_ts_us, jpeg), addressable by device time."""

    def __init__(self, seconds: float = 8.0, max_frames: int = 400):
        super().__init__()
        self.window_us = int(seconds * 1e6)
        self.max_frames = max_frames
        self.ring: deque[tuple[int, bytes]] = deque()

    def put(self, header, payload: bytes) -> None:
        with self.lock:
            self.received += 1
            ts = header.device_ts_us
            if self.header is not None and ts < self.header.device_ts_us:
                if self.header.device_ts_us - ts > _BACKWARDS_RESET_US:
                    self.ring.clear()  # clock restart: accept as the new epoch
                else:
                    self.dropped += 1  # out of order: older than what we hold
                    return
            if self._unread:
                self.dropped += 1
            self.header, self.jpeg, self._unread = header, payload, True
            self.ring.append((ts, payload))
            while self.ring and (self.ring[0][0] < ts - self.window_us or len(self.ring) > self.max_frames):
                self.ring.popleft()

    def window(self, start_us: int, end_us: int) -> list[tuple[int, bytes]]:
        """Frames with start_us <= device_ts_us <= end_us, oldest first."""
        with self.lock:
            return sorted((f for f in self.ring if start_us <= f[0] <= end_us), key=lambda f: f[0])


def install_frame_tap(ingest, seconds: float = 8.0) -> TappedFrames:
    """Replace ingest.video with a TappedFrames, keeping counters and the current frame. Idempotent."""
    if isinstance(ingest.video, TappedFrames):
        return ingest.video
    old = ingest.video
    tap = TappedFrames(seconds)
    with old.lock:
        tap.received, tap.dropped = old.received, old.dropped
        tap.header, tap.jpeg, tap._unread = old.header, old.jpeg, old._unread
        if old.header is not None and old.jpeg is not None:
            tap.ring.append((old.header.device_ts_us, old.jpeg))
    ingest.video = tap
    return tap


class SlotRing:
    """Last `seconds` of contiguous 16 kHz mono audio per separator slot, fed by SeparationWorker listeners."""

    def __init__(self, seconds: float = 8.0, sr: int = HUB_SR):
        self.sr = sr
        self.cap = int(seconds * sr)
        self.lock = threading.Lock()
        self._clear()

    def _clear(self) -> None:
        self._start_us = 0
        self._audio: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        with self.lock:
            self._clear()

    def _len(self) -> int:
        return min((len(a) for a in self._audio.values()), default=0)

    def _end_us(self) -> int:
        return self._start_us + self._len() * 1_000_000 // self.sr

    def on_chunks(self, chunks) -> None:
        if not chunks:
            return
        with self.lock:
            first = min(c.start_ts_us for c in chunks)
            new = {c.stream_id: np.asarray(c.audio, np.float32) for c in chunks}
            if self._audio and (abs(first - self._end_us()) > _CONTIG_TOL_US or set(new) != set(self._audio)):
                self._clear()  # gap, overlap, backwards or slot set change: restart from this hop
            if not self._audio:
                self._start_us = first
            n = min(len(a) for a in new.values())
            have = self._len()
            for k, a in new.items():
                cur = self._audio.get(k)
                self._audio[k] = a[:n].copy() if cur is None else np.concatenate([cur[:have], a[:n]])
            over = self._len() - self.cap
            if over > 0:
                for k in self._audio:
                    self._audio[k] = self._audio[k][over:].copy()
                self._start_us += over * 1_000_000 // self.sr

    @property
    def end_us(self) -> int:
        with self.lock:
            return self._end_us()

    def slot_ids(self) -> list[str]:
        with self.lock:
            return sorted(self._audio)

    def latest_window(self, length_s: float) -> tuple[int, dict[str, np.ndarray]] | None:
        """(start_us, {slot: copy}) of the newest `length_s` (or less) ending at the last hop; None if empty."""
        with self.lock:
            total = self._len()
            n = min(int(length_s * self.sr), total)
            if n <= 0:
                return None
            start = self._start_us + (total - n) * 1_000_000 // self.sr
            return start, {k: a[total - n: total].copy() for k, a in self._audio.items()}


def _index_of(audio, ts_us: int) -> int:
    """Absolute sample index of a device time, from the latest chunk anchor at or before it."""
    best = None
    for i, ts in audio.anchors:
        if ts <= ts_us:
            best = (i, ts)
    if best is None:
        best = audio.anchors[0]
    i, ts = best
    return i + int(round((ts_us - ts) * audio.sr / 1_000_000))


def make_cut_fn(ingest, ring: SlotRing, label_fn: Callable[[str], str | None], config: RefineConfig,
                mono=time.monotonic) -> Callable[[LiveView, int], RefineJob]:
    def cut(live: LiveView, job_id: int) -> RefineJob:
        if not live.slots:
            raise CutUnavailable("not audible")
        label = label_fn(live.chosen)
        if label not in (None, "speech"):
            raise CutUnavailable("selected source is not speech")
        audio = ingest.audio
        if audio is None:
            raise CutUnavailable("no audio")
        if audio.resets != live.audio_resets:
            raise CutUnavailable("audio reset")
        win = ring.latest_window(config.window_s)
        if win is None or len(next(iter(win[1].values()))) < int(config.min_window_s * ring.sr):
            raise CutUnavailable("window too short")
        start_us, slot_audio = win
        n = len(next(iter(slot_audio.values())))
        if live.window_floor_us > start_us:  # drop audio from before the selection or the last route change
            off = -(-(live.window_floor_us - start_us) * ring.sr // 1_000_000)
            off = min(off, n)
            start_us += -(-off * 1_000_000 // ring.sr)
            slot_audio = {k: a[off:].copy() for k, a in slot_audio.items()}
            n -= off
            if n < int(config.min_window_s * ring.sr):
                raise CutUnavailable("window too short")
        end_us = start_us + n * 1_000_000 // ring.sr
        tap = ingest.video
        if not isinstance(tap, TappedFrames):
            raise CutUnavailable("no frame tap")
        frames = tap.window(start_us, end_us)
        fps = len(frames) / (n / ring.sr)
        if len(frames) < 2 or fps < config.min_video_fps:
            raise CutUnavailable(f"insufficient video ({fps:.1f} fps)")
        with audio.lock:
            if audio.resets != live.audio_resets or not audio.anchors:
                raise CutUnavailable("audio reset")
            got = audio.read(_index_of(audio, start_us), n)
        if got is None:
            raise CutUnavailable("audio no longer buffered")
        mix = got[0].mean(axis=1).astype(np.float32)
        return RefineJob(
            job_id=job_id,
            key=RefineKey(live.chosen, live.epoch, live.selection_version, live.audio_resets),
            slots=tuple(live.slots), capture_start_us=start_us, capture_end_us=end_us,
            mix_16k=mix, slot_audio=slot_audio, frames=list(frames),
            label=label or "speech", created_mono=mono())
    return cut
