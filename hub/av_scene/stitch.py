"""Chunk stitching: finalize each chunk's owned audio, crossfade only across resolved identities.

Per source (persistent ``src_<n>`` / ``tse:<track>``) and per chunk k the writer emits one FINAL span

    first chunk : [0, own_end - h)            h = crossfade_samples // 2
    middle      : [own_start - h, own_end - h)
    last chunk  : [own_start - h, n)

and keeps a TAIL ``[own_end - h, own_end + h)`` (= the next chunk's blend region) for the next chunk. The first
``m`` samples of a non-first final span are the crossfade between the previous chunk's tail and this chunk's
window stream. Spans therefore tile ``[0, n)`` per source without gap or duplicate, and no sample is emitted
twice. Slot permutation is resolved BEFORE this module is called: callers pass streams keyed by persistent
source id, never by slot.

Guards (never blend two different speakers):
  * a source missing on one side (ambiguous/unresolved identity, or simply silent there) fades from/into
    silence instead of borrowing another stream;
  * if both sides exist but disagree in waveform (normalized correlation below ``min_corr`` while both are
    energetic) the blend is replaced by a hard switch at the boundary and flagged ``boundary_mismatch``.
"""

from __future__ import annotations

import numpy as np

from .contracts import ChunkSpec

MIN_CORR = 0.5
ENERGETIC_RMS = 1e-3


def crossfade_samples(crossfade_ms: float, sr: int) -> int:
    f = int(round(crossfade_ms * sr / 1000.0))
    return f - (f % 2)


def final_region(spec: ChunkSpec, n_samples: int, f: int) -> tuple[int, int]:
    h = f // 2
    start = 0 if spec.is_first else spec.own_start - h
    end = n_samples if spec.is_last else spec.own_end - h
    return start, end


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def blend(prev: np.ndarray | None, head: np.ndarray | None, min_corr: float = MIN_CORR
          ) -> tuple[np.ndarray, str | None]:
    """Crossfade ``prev`` (previous chunk's tail) into ``head`` (this chunk), equal length ``m``."""
    m = len(prev) if prev is not None else len(head)
    ramp = np.linspace(0.0, 1.0, m, endpoint=False, dtype=np.float32) + np.float32(0.5 / max(m, 1))
    if prev is None:
        return (head * ramp).astype(np.float32), None
    if head is None:
        return (prev * (1.0 - ramp)).astype(np.float32), None
    if _rms(prev) > ENERGETIC_RMS and _rms(head) > ENERGETIC_RMS:
        denom = float(np.linalg.norm(prev) * np.linalg.norm(head))
        corr = float(np.dot(prev.astype(np.float64), head.astype(np.float64)) / denom) if denom > 0 else 0.0
        if corr < min_corr:
            out = np.concatenate([prev[: m // 2], head[m // 2:]]).astype(np.float32)
            return out, f"boundary_mismatch(corr={corr:.2f})"
    return (prev * (1.0 - ramp) + head * ramp).astype(np.float32), None


def emit_chunk(spec: ChunkSpec, n_samples: int, f: int, streams: dict[str, np.ndarray],
               prev_tails: dict[str, tuple[int, np.ndarray]], min_corr: float = MIN_CORR):
    """-> (finals, tails, flags).

    streams     source id -> window audio covering [spec.win_start, spec.win_end)
    prev_tails  source id -> (absolute start sample, array) from the previous chunk
    finals      source id -> (absolute start sample, array)
    tails       source id -> (absolute start sample, array) for the next chunk (empty for the last chunk)
    flags       source id -> reason string (e.g. boundary_mismatch)
    """
    h = f // 2
    r0, r1 = final_region(spec, n_samples, f)
    off = spec.win_start
    finals: dict[str, tuple[int, np.ndarray]] = {}
    flags: dict[str, str] = {}
    for sid in sorted(set(streams) | set(prev_tails)):
        cur = streams.get(sid)
        pieces: list[np.ndarray] = []
        start = r0
        m = 0
        if not spec.is_first:
            ts, tail = prev_tails.get(sid, (None, None))
            m = min(len(tail) if tail is not None else f, r1 - r0, f)
            if m > 0:
                if tail is not None:
                    assert ts == r0, f"tail for {sid} starts at {ts}, chunk {spec.index} region starts at {r0}"
                head = cur[r0 - off: r0 - off + m] if cur is not None else None
                if tail is not None or head is not None:
                    seg, why = blend(tail[:m] if tail is not None else None, head, min_corr)
                    pieces.append(seg)
                    if why:
                        flags[sid] = why
        if cur is not None and r1 > r0 + m:
            pieces.append(cur[r0 - off + m: r1 - off].astype(np.float32))
        if pieces:
            finals[sid] = (start, np.concatenate(pieces).astype(np.float32))
    tails: dict[str, tuple[int, np.ndarray]] = {}
    if not spec.is_last:
        for sid, cur in streams.items():
            a, b = spec.own_end - h, spec.own_end + h
            tails[sid] = (a, cur[a - off: b - off].astype(np.float32))
    return finals, tails, flags
