"""Pure audio path for /audio/out: no sockets, no models, injectable clock.

    StreamChunk (16 kHz mono per separator slot, 1 s hops)
      -> Upsampler (3x, per slot, filter state kept across hops)
      -> LayerMixer: hop queue, settle delay, route lookup per 20 ms frame
           inner = sum of the chosen source's slots   (crossfaded 20 ms when the route changes)
           outer = everything else (all slots minus inner), so the chosen voice is not doubled
      -> ClientFormatter: 960-frame s16le frame in the layout the client asked for
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import signal

import protocol as p

SR_IN = 16000
SR_OUT = 48000
FRAME = 960            # 20 ms at 48 kHz
UP = SR_OUT // SR_IN   # 3


@dataclass
class AudioOutConfig:
    # Chunks are held this long before playing so the Tracker has consumed the same hop
    # (locator snapshot tick + delivery tick, about 2 x 100 ms) and the route for the hop is current.
    settle_s: float = 0.3
    # Extra hold for the first hop after idle or an underrun: the margin the next 1 s hop has to arrive
    # late by before playback starves. Steady state: the next hop arrives about 1 - (settle + jitter) s
    # before the current one ends.
    jitter_s: float = 0.2
    # Latency bound. Steady-state peak is about 1 s (hop) + settle + jitter, so keep this above ~1.6.
    # Past it the oldest audio is skipped down to trim_to_s with a 20 ms crossfade (no click).
    max_queue_s: float = 1.8
    trim_to_s: float = 1.2
    default_layout: str = "layers"
    # layers: the Quest applies inner/outer gain itself (AudioOutPlayer.SetVolumesDb), so the hub
    # sends unity-gain layers. Set True only for a client that does not apply them.
    apply_gains_in_layers: bool = False


class Upsampler:
    """16 kHz -> 48 kHz, linear-phase FIR with state kept across chunks (no seams at hop edges).

    Group delay is (len(taps)-1)/2 samples at 48 kHz (about 1 ms); it is not compensated."""

    def __init__(self):
        self.taps = signal.firwin(95, 7500 / (SR_OUT / 2), window=("kaiser", 8.0))
        self.zi = np.zeros(len(self.taps) - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        stuffed = np.zeros(len(x) * UP)
        stuffed[::UP] = np.asarray(x, np.float64) * UP
        y, self.zi = signal.lfilter(self.taps, 1.0, stuffed, zi=self.zi)
        return y.astype(np.float32)


@dataclass
class _Hop:
    ready: float
    start_ts_us: int
    audio: dict[str, np.ndarray]    # slot -> 48 kHz mono
    n: int
    pos: int = 0


@dataclass
class LayerFrame:
    seq: int
    device_ts_us: int               # capture time of the newest input sample in this frame
    inner: np.ndarray               # (960,) float32
    outer: np.ndarray
    bearing_deg: float | None
    target_id: str | None


class LayerMixer:
    """Hop queue -> 20 ms layer frames. `route_fn()` returns a selection.Route or None."""

    def __init__(self, route_fn, config: AudioOutConfig | None = None, clock=time.monotonic):
        self.cfg = config or AudioOutConfig()
        self.route_fn, self.clock = route_fn, clock
        self._hops: deque[_Hop] = deque()
        self._up: dict[str, Upsampler] = {}
        self._lock = threading.Lock()
        self._prev_slots: tuple[str, ...] = ()
        self._seq = 0
        self.started = False
        self.hops_in = 0
        self.hops_dropped = 0           # emergency whole-hop drops (queue far past the cap)
        self.trims = 0                  # times audio was skipped to get back under max_queue_s
        self.trimmed_s = 0.0
        self.underruns = 0              # episodes: starvation while a source is chosen, counted once
        self._starved = False
        self.frames_out = 0

    @property
    def queued_s(self) -> float:
        with self._lock:
            return sum(h.n - h.pos for h in self._hops) / SR_OUT

    # ---------------------------------------------------------------- producer side (worker thread)
    def push(self, chunks, now: float | None = None) -> None:
        """chunks: list[separate.StreamChunk] for one hop (one per slot, equal lengths)."""
        now = self.clock() if now is None else now
        if not chunks:
            return
        audio = {}
        for c in chunks:
            up = self._up.setdefault(c.stream_id, Upsampler())
            audio[c.stream_id] = up(np.asarray(c.audio, np.float32))
        # A slot absent from this hop would resume later with stale filter state: forget it.
        for slot in [k for k in self._up if k not in audio]:
            del self._up[slot]
        n = min(len(a) for a in audio.values())
        with self._lock:
            idle = not self._hops
            # Idle (start, or after an underrun): hold the hop a little longer so playback has margin.
            ready = now + self.cfg.settle_s + (self.cfg.jitter_s if idle else 0.0)
            self._hops.append(_Hop(ready, int(chunks[0].start_ts_us), {k: a[:n] for k, a in audio.items()}, n))
            self.hops_in += 1
            hard = int(4 * self.cfg.max_queue_s * SR_OUT)    # a stalled consumer must not grow memory
            while len(self._hops) > 1 and sum(h.n - h.pos for h in self._hops) > hard:
                self._hops.popleft()
                self.hops_dropped += 1

    def reset(self) -> None:
        with self._lock:
            self._hops.clear()
        self._up.clear()
        self._prev_slots = ()

    # ---------------------------------------------------------------- consumer side (event loop)
    def _read(self, n: int):
        """Consume n samples per slot (spanning hops), and the capture time of the last one. Lock held."""
        segs, need, ts = [], n, 0
        while need:
            h = self._hops[0]
            k = min(need, h.n - h.pos)
            segs.append((h, h.pos, k))
            h.pos += k
            need -= k
            ts = h.start_ts_us + h.pos * 1_000_000 // SR_OUT
            if h.pos >= h.n:
                self._hops.popleft()
        slots = {s for h, _, _ in segs for s in h.audio}
        out = {s: np.concatenate([h.audio[s][a:a + k] if s in h.audio else np.zeros(k, np.float32)
                                  for h, a, k in segs]) for s in slots}
        return out, ts

    def _skip(self, n: int) -> None:
        while n and self._hops:
            h = self._hops[0]
            k = min(n, h.n - h.pos)
            h.pos += k
            n -= k
            if h.pos >= h.n:
                self._hops.popleft()

    def _take(self, n: int, now: float):
        """Next n samples per slot and their capture time, or None if not ready. If the queue is over
        max_queue_s, skip the oldest audio down to trim_to_s, crossfading old into new over this frame."""
        with self._lock:
            ready = sum(h.n - h.pos for h in self._hops if h.ready <= now)
            if ready < n:
                return None
            queued = sum(h.n - h.pos for h in self._hops)
            drop = queued - int(self.cfg.trim_to_s * SR_OUT)
            if queued > int(self.cfg.max_queue_s * SR_OUT) and drop > 0 and ready - drop >= 2 * n:
                old, _ = self._read(n)
                self._skip(drop)
                new, ts = self._read(n)
                ramp = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
                zero = np.zeros(n, np.float32)
                out = {s: old.get(s, zero) * (1 - ramp) + new.get(s, zero) * ramp for s in set(old) | set(new)}
                self.trims += 1
                self.trimmed_s += (drop + n) / SR_OUT
                return out, ts
            return self._read(n)

    def pull(self, now: float | None = None) -> LayerFrame | None:
        now = self.clock() if now is None else now
        got = self._take(FRAME, now)
        if got is None:
            # Starvation only matters to someone who is listening: audio was flowing, a source is chosen.
            # Count the episode once (not each 20 ms pull) and not while idle with nothing chosen.
            if self.started and not self._starved and self.route_fn() is not None:
                self.underruns += 1
                self._starved = True
            return None
        audio, ts = got
        self.started = True
        self._starved = False
        total = np.zeros(FRAME, np.float32)
        for a in audio.values():
            total += a
        route = self.route_fn()
        slots = tuple(s for s in (route.slots if route else ()) if s in audio)

        def inner_of(sl):
            x = np.zeros(FRAME, np.float32)
            for s in sl:
                x += audio[s]
            return x
        inner = inner_of(slots)
        if slots != self._prev_slots:
            # Route changed (select / stop / next / slot swap): 20 ms crossfade, no click. Slots of
            # the previous route that this frame lacks contribute silence.
            old = inner_of(tuple(s for s in self._prev_slots if s in audio))
            ramp = np.linspace(0.0, 1.0, FRAME, endpoint=False, dtype=np.float32)
            inner = old * (1 - ramp) + inner * ramp
            self._prev_slots = slots
        frame = LayerFrame(self._seq, ts, inner, total - inner,
                           route.bearing_deg if route and slots else None,
                           route.target_id if route else None)
        self._seq += 1
        self.frames_out += 1
        return frame


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


class ClientFormatter:
    """Turns LayerFrames into AudioOut binary frames for one client; ramps gain changes per frame."""

    def __init__(self, layout: str, config: AudioOutConfig):
        if layout not in ("layers", "stereo"):
            raise ValueError(f"unknown layout {layout!r}")
        self.layout, self.cfg = layout, config
        self._gi = self._go = None   # last applied linear gains (for the ramp)

    def format(self, f: LayerFrame, inner_db: float, outer_db: float, head_yaw_deg: float | None) -> bytes:
        apply = self.layout == "stereo" or self.cfg.apply_gains_in_layers
        gi, go = (db_to_gain(inner_db), db_to_gain(outer_db)) if apply else (1.0, 1.0)
        ramp = np.linspace(0.0, 1.0, FRAME, endpoint=False, dtype=np.float32)
        pi = gi if self._gi is None else self._gi
        po = go if self._go is None else self._go
        inner = f.inner * (pi + (gi - pi) * ramp)
        outer = f.outer * (po + (go - po) * ramp)
        self._gi, self._go = gi, go
        if self.layout == "layers":
            pcm = np.stack([inner, outer], axis=1)
        else:
            if f.bearing_deg is not None and head_yaw_deg is not None:
                pan = math.sin(math.radians(f.bearing_deg - head_yaw_deg))   # -1 left .. +1 right
            else:
                pan = 0.0
            th = (pan + 1.0) * math.pi / 4.0                                  # constant power
            pcm = np.stack([inner * math.cos(th) + outer, inner * math.sin(th) + outer], axis=1)
        pcm = (np.clip(pcm, -1.0, 1.0) * 32767.0).round().astype("<i2")
        hdr = p.AudioOut(seq=f.seq, device_ts_us=f.device_ts_us, layout=self.layout,
                         target_bearing_deg=f.bearing_deg)
        return p.pack(hdr, pcm.tobytes())
