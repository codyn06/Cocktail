"""Sliding-window pseudo-streaming wrapper for offline (non-causal) separators.

A separator is any callable ``fn(window: Tensor[T]) -> Tensor[n_spk, T]`` at 16 kHz.
Every ``hop`` samples the last ``window`` samples are re-separated and the slice
``[end - look - hop, end - look)`` is emitted. ``look`` is the lookahead in samples
(0 = causal output: the emitted slice is the right edge of the window, where
non-causal models are weakest). Permutation across windows is resolved by matching
the model's re-estimate of already-emitted audio against what was emitted.

Algorithmic latency = (hop + look) / sr; add measured compute time per hop for
end-to-end latency.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import torch

SR = 16000

MODES = {
    # name: (window_s, hop_ms, look_ms)
    "fast": (1.5, 20, 0),
    "balanced": (2.0, 50, 100),
    "accurate": (6.0, 1000, 2000),
}


@dataclass
class StreamStats:
    hop_ms: list[float] = field(default_factory=list)  # compute time per hop
    peak_vram_mb: float = 0.0


class WindowedStreamer:
    def __init__(self, fn, n_spk: int, window_s: float, hop_ms: float, look_ms: float,
                 device: str = "cuda", hist_ms: float = 300.0):
        self.fn, self.n_spk, self.device = fn, n_spk, device
        self.W = int(window_s * SR)
        self.hop = int(hop_ms * SR / 1000)
        self.look = int(look_ms * SR / 1000)
        self.H = int(hist_ms * SR / 1000)
        assert self.look + self.hop <= self.W
        self.buf = torch.zeros(self.W, device=device)
        self.hist = torch.zeros(n_spk, self.H, device=device)
        self.perms = list(itertools.permutations(range(n_spk)))
        self.stats = StreamStats()
        self._sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") else (lambda: None)

    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        """chunk: [hop] float32 on device. Returns emitted [n_spk, hop] (delayed by look)."""
        assert chunk.numel() == self.hop
        self.buf = torch.cat([self.buf[self.hop:], chunk])
        self._sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            o = self.fn(self.buf)  # [n_spk, W]
        e = self.W - self.look
        seg = o[:, e - self.hop:e]
        # permutation tracking on the overlap region (already-emitted audio)
        h = min(self.H, e - self.hop)
        if h > 0:
            cur = o[:, e - self.hop - h:e - self.hop]
            ref = self.hist[:, -h:]
            if ref.abs().sum() > 0:
                best, bp = -1e30, None
                for p in self.perms:
                    s = (cur[list(p)] * ref).sum()
                    if s > best:
                        best, bp = s, p
                seg = seg[list(bp)]
        self._sync()
        self.stats.hop_ms.append((time.perf_counter() - t0) * 1000)
        self.hist = torch.cat([self.hist[:, self.hop:], seg], dim=1) if self.hop < self.H else seg[:, -self.H:]
        return seg

    def run(self, mix: np.ndarray) -> np.ndarray:
        """Stream a whole mixture; returns [n_spk, N - look] aligned to input time."""
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        x = torch.from_numpy(mix.astype(np.float32)).to(self.device)
        n = (len(x) // self.hop) * self.hop
        outs = [self.push(x[i:i + self.hop]) for i in range(0, n, self.hop)]
        y = torch.cat(outs, dim=1).cpu().numpy()
        if self.device.startswith("cuda"):
            self.stats.peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20
        # output sample k corresponds to input time k - look
        return y[:, self.look:]
