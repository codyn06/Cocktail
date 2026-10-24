"""Loopback audio path: decode, map to stereo, resample to 48 kHz, cut 20 ms frames."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from hub.protocol import FMT_INT16, AudioHeader

OUT_RATE = 48_000
OUT_CHANNELS = 2
FRAME_SAMPLES = OUT_RATE * 20 // 1000  # 960 samples = 20 ms


def decode(header: AudioHeader, payload) -> np.ndarray:
    """PCM payload -> float32 array of shape (frames, channels) in [-1, 1]."""
    if header.sample_format == FMT_INT16:
        x = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    else:
        x = np.nan_to_num(np.frombuffer(payload, dtype="<f4"), nan=0.0, posinf=1.0, neginf=-1.0)
    return x.reshape(-1, header.channels)


def to_stereo(x: np.ndarray) -> np.ndarray:
    """Mono is duplicated; beyond two channels only the first two are kept."""
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


class StreamingResampler:
    """Linear-interpolation resampler that keeps its phase across chunks.

    Adds one input sample of delay. Good enough for an unprocessed loopback; swap
    for a polyphase filter once audio quality matters.
    """

    def __init__(self, src_rate: int, dst_rate: int = OUT_RATE):
        self.src_rate = src_rate
        self.step = src_rate / dst_rate
        self._prev: np.ndarray | None = None  # last input sample, shape (1, ch)
        self._pos = 0.0  # next output position, in input samples from buf[0]

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.step == 1.0:
            return x
        buf = x if self._prev is None else np.concatenate([self._prev, x])
        n = len(buf)
        if n < 2:
            self._prev = buf[-1:] if n else self._prev
            return np.zeros((0, x.shape[1]), np.float32)
        idx = self._pos + self.step * np.arange(int(np.ceil((n - 1 - self._pos) / self.step)))
        idx = idx[idx < n - 1]
        i0 = idx.astype(np.int64)
        frac = (idx - i0)[:, None].astype(np.float32)
        out = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac
        self._pos = self._pos + self.step * len(idx) - (n - 1)
        self._prev = buf[-1:]
        return out.astype(np.float32)


@dataclass(frozen=True)
class OutFrame:
    pcm: bytes  # FRAME_SAMPLES x 2 int16, little-endian, interleaved
    pts_us: int  # glasses timestamp of this frame's first sample
    epoch: int
    ref_seq: int  # newest input chunk whose samples are in this frame
    newest_recv_ns: int  # hub monotonic time that chunk arrived (frame became complete)
    oldest_recv_ns: int  # hub monotonic time the chunk holding the first sample arrived


class Framer:
    """Collects 48 kHz stereo samples and emits fixed 20 ms int16 frames.

    Sample timestamps map from the chunk that produced them; the resampler's one
    input sample of delay (<= 125 us) is ignored.
    """

    def __init__(self):
        self._buf = np.zeros((0, OUT_CHANNELS), np.float32)
        self._base = 0  # absolute sample index of _buf[0]
        self._tags: deque[tuple[int, AudioHeader, int]] = deque()  # (start, header, recv_ns)

    def push(self, samples: np.ndarray, header: AudioHeader, recv_ns: int) -> None:
        if len(samples):
            self._tags.append((self._base + len(self._buf), header, recv_ns))
            self._buf = np.concatenate([self._buf, samples])

    def pop_frames(self) -> list[OutFrame]:
        frames = []
        while len(self._buf) >= FRAME_SAMPLES:
            end = self._base + FRAME_SAMPLES
            while len(self._tags) >= 2 and self._tags[1][0] <= self._base:
                self._tags.popleft()
            first_start, first, first_recv_ns = self._tags[0]
            newest, newest_recv_ns = first, first_recv_ns
            for start, header, recv_ns in self._tags:
                if start >= end:
                    break
                newest, newest_recv_ns = header, recv_ns
            pcm = np.round(np.clip(self._buf[:FRAME_SAMPLES], -1.0, 1.0) * 32767).astype("<i2")
            frames.append(
                OutFrame(
                    pcm.tobytes(),
                    pts_us=first.pts_us + (self._base - first_start) * 1_000_000 // OUT_RATE,
                    epoch=newest.epoch,
                    ref_seq=newest.seq,
                    newest_recv_ns=newest_recv_ns,
                    oldest_recv_ns=first_recv_ns,
                )
            )
            self._buf = self._buf[FRAME_SAMPLES:]
            self._base = end
        return frames


class LoopbackPipeline:
    """Unprocessed path from one ingest stream to 48 kHz stereo out frames.

    Resets (dropping any partial frame) when the epoch, sample rate or channel
    count changes, or when reset() is called for a new connection.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._key = None
        self._resampler: StreamingResampler | None = None
        self._framer = Framer()

    def process(self, header: AudioHeader, payload, recv_ns: int) -> list[OutFrame]:
        key = (header.epoch, header.sample_rate, header.channels)
        if key != self._key:
            self._key = key
            self._resampler = StreamingResampler(header.sample_rate)
            self._framer = Framer()
        stereo = to_stereo(decode(header, payload))
        self._framer.push(self._resampler.process(stereo), header, recv_ns)
        return self._framer.pop_frames()
