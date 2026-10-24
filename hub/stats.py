"""Loopback counters and rolling timing windows, served at GET /stats."""

from collections import deque

import numpy as np

from hub.protocol import AudioHeader

WINDOW = 1000  # samples kept per timing window


def _summary(values) -> dict:
    if not values:
        return {"n": 0}
    a = np.asarray(values)
    return {
        "n": len(a),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "max_ms": round(float(a.max()), 3),
    }


class LoopbackStats:
    def __init__(self):
        self.chunks_in = 0
        self.bad_messages = 0
        self.seq_gaps = 0  # input chunks missing between consecutive seqs
        self.out_of_order = 0
        self.epochs = 0
        self.ingest_connections = 0
        self.format: dict | None = None
        self.frames_out = 0
        self.frames_sent = 0
        self.frames_unheard = 0  # produced with no /audio/out listener
        self.frames_dropped = 0  # listener queue full; oldest frame dropped
        # |arrival gap - previous chunk duration|: network/phone jitter seen by the hub
        self.arrival_jitter_ms: deque[float] = deque(maxlen=WINDOW)
        # chunk received -> processing done
        self.process_ms: deque[float] = deque(maxlen=WINDOW)
        # newest input chunk in a frame received -> frame handed to the socket
        self.hub_latency_ms: deque[float] = deque(maxlen=WINDOW)
        self._prev: tuple[AudioHeader, int, float] | None = None  # header, recv_ns, duration_ms

    def on_connect(self) -> None:
        """A new ingest connection starts a fresh sequence, even with the same epoch."""
        self.ingest_connections += 1
        self._prev = None

    def on_chunk(self, header: AudioHeader, n_frames: int, recv_ns: int) -> None:
        self.chunks_in += 1
        self.format = {
            "sample_rate": header.sample_rate,
            "channels": header.channels,
            "sample_format": header.sample_format,
            "chunk_ms": round(1000 * n_frames / header.sample_rate, 3),
        }
        prev = self._prev
        if prev is None or prev[0].epoch != header.epoch:
            self.epochs += 1
        else:
            prev_header, prev_recv_ns, prev_ms = prev
            if header.seq > prev_header.seq + 1:
                self.seq_gaps += header.seq - prev_header.seq - 1
            elif header.seq <= prev_header.seq:
                self.out_of_order += 1
            self.arrival_jitter_ms.append(abs((recv_ns - prev_recv_ns) / 1e6 - prev_ms))
        self._prev = (header, recv_ns, 1000 * n_frames / header.sample_rate)

    def snapshot(self, listeners: int) -> dict:
        return {
            "ingest_connections": self.ingest_connections,
            "listeners": listeners,
            "format": self.format,
            "chunks_in": self.chunks_in,
            "bad_messages": self.bad_messages,
            "seq_gaps": self.seq_gaps,
            "out_of_order": self.out_of_order,
            "epochs": self.epochs,
            "frames_out": self.frames_out,
            "frames_sent": self.frames_sent,
            "frames_unheard": self.frames_unheard,
            "frames_dropped": self.frames_dropped,
            "arrival_jitter": _summary(self.arrival_jitter_ms),
            "process": _summary(self.process_ms),
            "hub_latency": _summary(self.hub_latency_ms),
        }
