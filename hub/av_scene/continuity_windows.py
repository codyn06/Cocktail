"""Identity continuity across pauses: persistent source IDs for separated streams.

Slide windows over a run, embed each separated stream per window with an injected
``embed(audio16k) -> np.ndarray`` and assign stable source IDs by cosine similarity
to running per-ID centroids (Hungarian one-to-one, threshold, new ID otherwise).

``associate.continuity()`` is NOT reused: its contract is Association-level
(track<->stream mapping downgrade using a (prev_stream, curr_stream) similarity
dict), not embedding-to-identity assignment. This module sits underneath it: its
output can feed ``stream_embed_sim`` there if wanted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

SR = 16000
WINDOW_S = 4.0
HOP_S = 2.0
MATCH_THRESHOLD = 0.6      # min cosine to join an existing ID
MIN_RMS = 0.005            # below this a window-stream is treated as silent


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def window_starts(n_samples: int, sr: int = SR, window_s: float = WINDOW_S,
                  hop_s: float = HOP_S) -> list[int]:
    """Sample start of each full window (at least one if the run is shorter)."""
    w, h = int(window_s * sr), int(hop_s * sr)
    if n_samples <= w:
        return [0]
    return list(range(0, n_samples - w + 1, h))


class EmbeddingCache:
    """Memoize embeddings keyed by (stream_id, window_start); call as cache(sid, start, audio)."""

    def __init__(self, embed: Callable[[np.ndarray], np.ndarray]):
        self._embed = embed
        self._store: dict[tuple, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, stream_id, start: int, audio: np.ndarray) -> np.ndarray:
        key = (stream_id, start)
        if key in self._store:
            self.hits += 1
        else:
            self.misses += 1
            self._store[key] = _unit(self._embed(audio))
        return self._store[key]


@dataclass
class WindowAssignment:
    start: int                                      # window start sample
    mapping: dict = field(default_factory=dict)     # stream_id -> source_id
    confidence: dict = field(default_factory=dict)  # stream_id -> [0,1]
    new_ids: list = field(default_factory=list)
    skipped: list = field(default_factory=list)     # silent stream_ids


class IdentityTracker:
    """Running centroids + per-window one-to-one assignment."""

    def __init__(self, threshold: float = MATCH_THRESHOLD):
        self.threshold = threshold
        self.centroids: dict[int, np.ndarray] = {}
        self.counts: dict[int, int] = {}
        self._next = 0

    def assign(self, embs: dict) -> tuple[dict, dict, list]:
        """embs: stream_id -> unit embedding. Returns (mapping, confidence, new_ids)."""
        sids = list(embs)
        ids = list(self.centroids)
        mapping: dict = {}
        conf: dict = {}
        matched: set[int] = set()
        if sids and ids:
            sim = np.array([[float(embs[s] @ self.centroids[i]) for i in ids] for s in sids])
            rows, cols = linear_sum_assignment(-sim)
            for r, c in zip(rows, cols):
                if sim[r, c] >= self.threshold:
                    mapping[sids[r]] = ids[c]
                    conf[sids[r]] = float(np.clip(sim[r, c], 0.0, 1.0))
                    matched.add(r)
        new_ids = []
        for r, s in enumerate(sids):
            if r in matched:
                continue
            best = max((float(embs[s] @ c) for c in self.centroids.values()), default=-1.0)
            nid = self._next
            self._next += 1
            self.centroids[nid] = embs[s].copy()
            self.counts[nid] = 1
            mapping[s] = nid
            conf[s] = float(np.clip(1.0 - max(best, 0.0), 0.0, 1.0))
            new_ids.append(nid)
        for r in matched:  # running-mean update for re-identified IDs only
            s = sids[r]
            i = mapping[s]
            n = self.counts[i]
            self.centroids[i] = _unit((self.centroids[i] * n + embs[s]) / (n + 1))
            self.counts[i] = n + 1
        return mapping, conf, new_ids


def track_identities(streams: dict, embed, sr: int = SR, window_s: float = WINDOW_S,
                     hop_s: float = HOP_S, threshold: float = MATCH_THRESHOLD,
                     min_rms: float = MIN_RMS) -> list[WindowAssignment]:
    """streams: stream_id -> 1-D float audio at ``sr``.

    ``embed`` is a plain ``audio -> vector`` callable or an EmbeddingCache
    (called as ``embed(stream_id, start, audio)``).
    """
    n = max((len(a) for a in streams.values()), default=0)
    tracker = IdentityTracker(threshold)
    w = int(window_s * sr)
    out: list[WindowAssignment] = []
    for start in window_starts(n, sr, window_s, hop_s):
        wa = WindowAssignment(start=start)
        embs = {}
        for sid, a in streams.items():
            seg = np.asarray(a[start:start + w], dtype=np.float32)
            if seg.size == 0 or float(np.sqrt(np.mean(seg ** 2))) < min_rms:
                wa.skipped.append(sid)
                continue
            embs[sid] = embed(sid, start, seg) if isinstance(embed, EmbeddingCache) \
                else _unit(embed(seg))
        wa.mapping, wa.confidence, wa.new_ids = tracker.assign(embs)
        out.append(wa)
    return out
