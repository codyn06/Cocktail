"""Binary audio frame format for /ingest/audio and /audio/out (proposed contract, v1).

Every WebSocket binary message is one 32-byte little-endian header followed by
interleaved PCM samples:

    offset size field
    0      4    magic          b"CKAU"
    4      1    version        1
    5      1    sample_format  1 = int16, 2 = float32
    6      2    channels       >= 1
    8      4    sample_rate    Hz
    12     4    seq            per-stream counter, +1 per message
    16     4    epoch          bumped by the sender on every reconnect
    20     8    pts_us         glasses presentation timestamp (microseconds)
    28     4    ref_seq        out frames: seq of the newest input chunk inside; in: 0

Out frames reuse the epoch and pts_us of the input chunk named by ref_seq, so the
phone can relate what it plays back to the glasses timestamp it came from.
"""

import struct
from dataclasses import dataclass

MAGIC = b"CKAU"
VERSION = 1
FMT_INT16 = 1
FMT_FLOAT32 = 2
SAMPLE_BYTES = {FMT_INT16: 2, FMT_FLOAT32: 4}
MIN_RATE, MAX_RATE = 8_000, 96_000

_HEADER = struct.Struct("<4sBBHIIIqI")
HEADER_SIZE = _HEADER.size  # 32


class FrameError(ValueError):
    """A message that does not follow the frame format."""


@dataclass(frozen=True)
class AudioHeader:
    seq: int
    epoch: int
    pts_us: int
    sample_rate: int
    channels: int
    sample_format: int = FMT_INT16
    ref_seq: int = 0

    @property
    def frame_bytes(self) -> int:
        """Bytes per multichannel sample frame."""
        return SAMPLE_BYTES[self.sample_format] * self.channels


def pack(header: AudioHeader, payload: bytes) -> bytes:
    return (
        _HEADER.pack(
            MAGIC,
            VERSION,
            header.sample_format,
            header.channels,
            header.sample_rate,
            header.seq,
            header.epoch,
            header.pts_us,
            header.ref_seq,
        )
        + payload
    )


def unpack(data: bytes) -> tuple[AudioHeader, memoryview]:
    if len(data) < HEADER_SIZE:
        raise FrameError(f"message is {len(data)} bytes, shorter than the {HEADER_SIZE}-byte header")
    magic, version, fmt, channels, rate, seq, epoch, pts_us, ref_seq = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise FrameError(f"bad magic {magic!r}")
    if version != VERSION:
        raise FrameError(f"unsupported version {version}")
    if fmt not in SAMPLE_BYTES:
        raise FrameError(f"unknown sample_format {fmt}")
    if channels < 1:
        raise FrameError("channels must be >= 1")
    if not MIN_RATE <= rate <= MAX_RATE:
        raise FrameError(f"sample_rate {rate} outside {MIN_RATE}..{MAX_RATE}")
    header = AudioHeader(seq, epoch, pts_us, rate, channels, fmt, ref_seq)
    payload = memoryview(data)[HEADER_SIZE:]
    if len(payload) % header.frame_bytes:
        raise FrameError(f"payload of {len(payload)} bytes is not whole {channels}-channel frames")
    return header, payload
