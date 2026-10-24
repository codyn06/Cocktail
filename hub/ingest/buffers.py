"""Rolling buffers for hub ingest (task B3): last N seconds of audio and pose, newest video frame.

All timestamps are the headset clock (device_ts_us). An AudioChunk's device_ts_us is taken as
the capture time of its first sample (the protocol does not say; confirm with B1).
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import signal

HUB_SR = 16000


class Resampler:
    """Stateful conversion of one stream to 16 kHz float32.

    Integer-factor downsampling (48 kHz, 32 kHz) keeps filter state across chunks, so chunk
    boundaries add no artifacts. Other rates fall back to per-chunk resample_poly."""

    def __init__(self, sr_in: int, channels: int):
        self.sr_in, self.channels = sr_in, channels
        self.factor = sr_in // HUB_SR if sr_in % HUB_SR == 0 else 0
        self.delay = 0  # output samples by which the output lags the input (linear-phase FIR)
        if self.factor > 1:
            self.taps = signal.firwin(24 * self.factor + 1, 0.9 / self.factor)
            self.zi = np.zeros((len(self.taps) - 1, channels))
            self.phase = 0  # index of the next kept sample within the next chunk
            self.delay = (len(self.taps) - 1) // 2 // self.factor

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.factor == 1:
            return x
        if self.factor > 1:
            y, self.zi = signal.lfilter(self.taps, 1.0, x, axis=0, zi=self.zi)
            out = y[self.phase :: self.factor]
            self.phase = (self.phase - len(x)) % self.factor
            return out.astype(np.float32)
        g = math.gcd(HUB_SR, self.sr_in)
        return signal.resample_poly(x, HUB_SR // g, self.sr_in // g, axis=0).astype(np.float32)


class AudioRing:
    """Last `seconds` of multichannel 16 kHz audio, addressable by device time."""

    def __init__(self, channels: int, seconds: float = 10.0, sr: int = HUB_SR):
        self.sr, self.channels = sr, channels
        self.buf = np.zeros((int(seconds * sr), channels), np.float32)
        self.written = 0  # total samples ever written (absolute index of the next sample)
        self.anchors: deque[tuple[int, int]] = deque(maxlen=4096)  # (abs index, device_ts_us)
        self.contiguous_since_us: int | None = None  # start of the current gap-free stretch
        self.resets = 0  # consumers compare this to notice a reset
        self.lock = threading.RLock()

    def write(self, x: np.ndarray, device_ts_us: int) -> None:
        n, cap = len(x), len(self.buf)
        with self.lock:
            if self.contiguous_since_us is None:
                self.contiguous_since_us = device_ts_us
            self.anchors.append((self.written, device_ts_us))
            skip = max(0, n - cap)  # a write longer than the ring keeps only its tail
            x = x[skip:]
            start = (self.written + skip) % cap
            k = min(len(x), cap - start)
            self.buf[start : start + k] = x[:k]
            self.buf[: len(x) - k] = x[k:]
            self.written += n

    def ts_of(self, index: int) -> int:
        """Device time of an absolute sample index, from the nearest earlier chunk anchor."""
        with self.lock:
            for i, ts in reversed(self.anchors):
                if i <= index:
                    return ts + (index - i) * 1_000_000 // self.sr
            i, ts = self.anchors[0]
            return ts + (index - i) * 1_000_000 // self.sr

    def reset(self) -> None:
        """Forget everything (reconnect, clock jump, or a gap too long to zero-fill)."""
        with self.lock:
            self.buf[:] = 0
            self.written = 0
            self.anchors.clear()
            self.contiguous_since_us = None
            self.resets += 1

    def last(self, seconds: float) -> tuple[np.ndarray, int]:
        """Most recent audio, shape (n, channels), and the device time of its first sample."""
        with self.lock:
            cap = len(self.buf)
            n = min(int(seconds * self.sr), self.written, cap)
            first = self.written - n
            idx = np.arange(first, self.written) % cap
            return self.buf[idx].copy(), (self.ts_of(first) if n else 0)

    def read(self, start: int, n: int) -> tuple[np.ndarray, int] | None:
        """Samples [start, start + n) by absolute index and the device time of the first one;
        None if any of them is not (or no longer) in the ring."""
        with self.lock:
            cap = len(self.buf)
            if start < max(0, self.written - cap) or start + n > self.written:
                return None
            idx = np.arange(start, start + n) % cap
            return self.buf[idx].copy(), self.ts_of(start)

    @property
    def seconds_held(self) -> float:
        return min(self.written, len(self.buf)) / self.sr


@dataclass
class Pose:
    device_ts_us: int
    pos: tuple[float, float, float]
    rot_xyzw: tuple[float, float, float, float]
    yaw_deg: float    # + = right turn, 0 = tracking-space forward
    pitch_deg: float  # + = looking up


def yaw_pitch(rot_xyzw) -> tuple[float, float]:
    """Yaw/pitch of the head's forward (+z) axis in Unity tracking space (+x right, +y up)."""
    x, y, z, w = rot_xyzw
    fx = 2 * (x * z + w * y)
    fy = 2 * (y * z - w * x)
    fz = 1 - 2 * (x * x + y * y)
    return math.degrees(math.atan2(fx, fz)), math.degrees(math.asin(max(-1.0, min(1.0, fy))))


class PoseBuffer:
    """Last `seconds` of head poses (by device time) with interpolated lookup."""

    def __init__(self, seconds: float = 10.0):
        self.window_us = int(seconds * 1e6)
        self.poses: deque[Pose] = deque()
        self.out_of_order = 0
        self.lock = threading.Lock()

    def reset(self) -> None:
        with self.lock:
            self.poses.clear()

    def add(self, device_ts_us: int, pos, rot_xyzw) -> Pose | None:
        """Append a pose; poses not newer than the last one are dropped (returns None)."""
        yaw, pitch = yaw_pitch(rot_xyzw)
        p = Pose(device_ts_us, tuple(pos), tuple(rot_xyzw), yaw, pitch)
        with self.lock:
            if self.poses and device_ts_us <= self.poses[-1].device_ts_us:
                self.out_of_order += 1
                return None
            self.poses.append(p)
            while self.poses and self.poses[0].device_ts_us < device_ts_us - self.window_us:
                self.poses.popleft()
        return p

    def at(self, device_ts_us: int, max_extrapolate_us: int = 50_000) -> tuple[float, float] | None:
        """(yaw, pitch) at a device time, linearly interpolated (yaw unwrapped).

        None if the time is outside the buffer by more than max_extrapolate_us."""
        with self.lock:
            ps = list(self.poses)
        if not ps or not (ps[0].device_ts_us - max_extrapolate_us <= device_ts_us
                          <= ps[-1].device_ts_us + max_extrapolate_us):
            return None
        if device_ts_us <= ps[0].device_ts_us:
            return ps[0].yaw_deg, ps[0].pitch_deg
        for a, b in zip(ps, ps[1:]):
            if a.device_ts_us <= device_ts_us <= b.device_ts_us:
                f = (device_ts_us - a.device_ts_us) / max(1, b.device_ts_us - a.device_ts_us)
                dyaw = (b.yaw_deg - a.yaw_deg + 180) % 360 - 180
                return a.yaw_deg + f * dyaw, a.pitch_deg + f * (b.pitch_deg - a.pitch_deg)
        return ps[-1].yaw_deg, ps[-1].pitch_deg

    def __len__(self) -> int:
        return len(self.poses)


class LatestFrame:
    """Keeps only the newest video frame; anything it replaces unread counts as dropped."""

    def __init__(self):
        self.header = None
        self.jpeg: bytes | None = None
        self.received = self.dropped = 0
        self._unread = False
        self.lock = threading.Lock()

    def put(self, header, payload: bytes) -> None:
        with self.lock:
            self.received += 1
            if self.header is not None and header.device_ts_us < self.header.device_ts_us:
                self.dropped += 1  # out of order: older than what we hold
                return
            if self._unread:
                self.dropped += 1
            self.header, self.jpeg, self._unread = header, payload, True

    def take(self):
        """Newest (header, jpeg) for a consumer such as SAM 3; marks it read."""
        with self.lock:
            self._unread = False
            return self.header, self.jpeg
