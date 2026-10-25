"""Identity continuity across pauses: persistent source IDs for separated streams.

Slide windows over a run, embed each separated stream per window with an injected
``embed(audio16k) -> np.ndarray`` and assign stable source IDs by cosine similarity
to running per-ID centroids (Hungarian one-to-one, threshold, new ID otherwise).

``associate.continuity()`` is NOT reused: its contract is Association-level
(track<->stream mapping downgrade using a (prev_stream, curr_stream) similarity
dict), not embedding-to-identity assignment. This module sits underneath it:
``IdentityStore.assign_chunk`` yields a per-chunk ``ChunkAssignment`` (slot -> persistent
``src_<n>``); ``stream_sims_for_continuity(prev, curr, store)`` turns two consecutive
ChunkAssignments into the ``{(prev_stream, curr_stream): cosine}`` dict that
``associate.continuity()`` takes as ``stream_embed_sim`` (1.0 when both slots resolved to the
same source_id, centroid cosine capped at 0.99 for different ids, absent for ambiguous/none
slots). Keys are slot indices unless a ``name`` callable maps slots to stream ids.

``IdentityStore`` (chunked/long-clip path): slot order is arbitrary per chunk, so slots are
matched one-to-one to identities by centroid cosine. Ambiguous slots are never mapped, never
enrolled, never mint ids. Enrollment uses only the excerpt ``own & eligible & voiced`` so
overlap audio is never double counted and off-axis audio never changes a centroid. At most
``max_active`` centroids stay resident; the rest spill to ``spill_dir`` as content-addressed
npz files (resume-safe). A new id is minted only when its enrollable excerpt is at least
``min_enroll_s`` long (otherwise status "none", reason "insufficient_eligible_audio" when the eligibility mask
removed voiced owned audio, or "short_utterance" when the speech is just too short; both abstain, neither mints an id).
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
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


# ---------------------------------------------------------------------------------- IdentityStore
FRAME = 400          # 25 ms voiced-activity frames at 16 kHz
MIN_MATCH_S = 0.25   # minimum voiced owned audio for a slot to take part in matching


def _sid(n: int) -> str:
    from .contracts import source_id_for
    return source_id_for(n)


def _num(sid: str) -> int:
    return int(sid.rsplit("_", 1)[1])


@dataclass
class SlotInput:
    audio: np.ndarray
    own: tuple[int, int]
    eligible: list[tuple[int, int]] | None = None


@dataclass
class ChunkAssignment:
    mapping: dict[int, str] = field(default_factory=dict)
    confidence: dict[int, float] = field(default_factory=dict)
    status: dict[int, str] = field(default_factory=dict)
    reasons: dict[int, str] = field(default_factory=dict)
    new_ids: list[str] = field(default_factory=list)
    enrolled_s: dict[str, float] = field(default_factory=dict)
    active_s: dict[int, float] = field(default_factory=dict)


def _digest(vec: np.ndarray, weight: float, last: int) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vec, dtype=np.float64).tobytes())
    h.update(struct.pack("<dq", float(weight), int(last)))
    return h.hexdigest()


class IdentityStore:
    """Persistent acoustic identities across chunks with bounded resident memory."""

    def __init__(self, embed, *, threshold=0.6, ambiguous_band=0.1, margin=0.08, max_active=8,
                 min_rms=MIN_RMS, min_enroll_s=1.0, spill_dir=None, sr=SR):
        self.embed = embed
        self.threshold, self.ambiguous_band, self.margin = threshold, ambiguous_band, margin
        self.max_active, self.min_rms, self.min_enroll_s, self.sr = max_active, min_rms, min_enroll_s, sr
        self.spill_dir = Path(spill_dir) if spill_dir is not None else None
        self._active: dict[int, dict] = {}     # id -> {"sum": ndarray, "weight": float, "last": int}
        self._spilled: dict[int, dict] = {}    # id -> {"file": str, "sha256": str, "last": int}
        self._next = 0
        self._last_chunk = -1
        self.peak_resident = 0                 # max centroids in memory, incl. one transient load

    # ------------------------------------------------------------------ residency
    @property
    def resident_count(self) -> int:
        return len(self._active)

    def _note(self, extra: int = 0) -> None:
        self.peak_resident = max(self.peak_resident, len(self._active) + extra)

    def _spill(self, i: int) -> None:
        if self.spill_dir is None:
            raise RuntimeError(f"more than max_active={self.max_active} identities; pass spill_dir "
                               "to IdentityStore so least-recently-matched centroids can be spilled")
        st = self._active[i]
        sha = _digest(st["sum"], st["weight"], st["last"])
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        name = f"{_sid(i)}.{sha[:16]}.npz"
        path = self.spill_dir / name
        if not path.exists():
            tmp = self.spill_dir / (name + ".tmp")
            with open(tmp, "wb") as f:
                np.savez(f, sum=st["sum"], weight=np.float64(st["weight"]), last=np.int64(st["last"]))
            tmp.replace(path)
        self._spilled[i] = {"file": name, "sha256": sha, "last": st["last"]}
        del self._active[i]

    def _read_spilled(self, i: int) -> dict:
        meta = self._spilled[i]
        with np.load(self.spill_dir / meta["file"]) as z:
            st = {"sum": np.array(z["sum"], dtype=np.float64), "weight": float(z["weight"]),
                  "last": int(z["last"])}
        if _digest(st["sum"], st["weight"], st["last"]) != meta["sha256"]:
            raise RuntimeError(f"spilled identity {_sid(i)} failed checksum ({meta['file']})")
        return st

    def _make_room(self, keep: int | None) -> None:
        while len(self._active) >= self.max_active:
            cands = [(st["last"], i) for i, st in self._active.items() if i != keep]
            if not cands:
                raise RuntimeError("max_active too small for this operation")
            self._spill(min(cands)[1])

    def _ensure_resident(self, i: int) -> dict:
        if i in self._active:
            return self._active[i]
        self._make_room(keep=i)
        st = self._read_spilled(i)
        del self._spilled[i]
        self._active[i] = st
        self._note()
        return st

    def _all_ids(self) -> list[int]:
        return sorted(set(self._active) | set(self._spilled))

    def _centroid(self, i: int) -> np.ndarray:
        if i in self._active:
            return _unit(self._active[i]["sum"])
        self._note(1)                          # transient: loaded, used, dropped
        return _unit(self._read_spilled(i)["sum"])

    def id_similarity(self, a: str, b: str) -> float:
        return float(self._centroid(_num(a)) @ self._centroid(_num(b)))

    # ------------------------------------------------------------------ audio helpers
    def _voiced(self, audio: np.ndarray) -> np.ndarray:
        m = np.zeros(len(audio), dtype=bool)
        for s in range(0, len(audio), FRAME):
            seg = audio[s:s + FRAME].astype(np.float64)
            if float(np.sqrt(np.mean(seg ** 2))) >= self.min_rms:
                m[s:s + FRAME] = True
        return m

    @staticmethod
    def _range_mask(n: int, ranges) -> np.ndarray:
        m = np.zeros(n, dtype=bool)
        for a, b in ranges:
            m[max(0, int(a)):max(0, min(n, int(b)))] = True
        return m

    # ------------------------------------------------------------------ assignment
    def assign_chunk(self, chunk_index: int, slots: dict[int, SlotInput]) -> ChunkAssignment:
        res = ChunkAssignment()
        info: dict[int, dict] = {}
        for k in sorted(slots):
            si = slots[k]
            audio = np.asarray(si.audio, dtype=np.float32).ravel()
            voiced = self._voiced(audio)
            own = self._range_mask(len(audio), [si.own])
            elig = (np.ones(len(audio), bool) if si.eligible is None
                    else self._range_mask(len(audio), si.eligible))
            res.active_s[k] = float((voiced & own).sum()) / self.sr
            if res.active_s[k] < MIN_MATCH_S:
                res.status[k], res.reasons[k] = "none", "insufficient_audio"
                continue
            info[k] = {"emb": _unit(self.embed(audio)), "excerpt": audio[voiced & own & elig],
                       "cut": int((voiced & own).sum() - (voiced & own & elig).sum())}   # samples eligibility removed
        rows = sorted(info)
        ids = self._all_ids()
        S = np.zeros((len(rows), len(ids)))
        assigned: dict[int, int] = {}
        if rows and ids:
            for c, i in enumerate(ids):
                cen = self._centroid(i)
                for r, k in enumerate(rows):
                    S[r, c] = float(info[k]["emb"] @ cen)
            ra, ca = linear_sum_assignment(-S)
            assigned = dict(zip(ra.tolist(), ca.tolist()))
        lo = self.threshold - self.ambiguous_band
        decision: dict[int, tuple] = {}
        for r in range(len(rows)):
            if not ids:
                decision[r] = ("new",)
            elif r not in assigned:
                decision[r] = ("amb", "contested") if float(S[r].max()) >= lo else ("new",)
            else:
                c = assigned[r]
                sim = float(S[r, c])
                if sim < lo:
                    decision[r] = ("new",)
                elif sim < self.threshold:
                    decision[r] = ("amb", "below_threshold_band")
                else:
                    others = [float(S[r, j]) for j in range(len(ids)) if j != c]
                    if others and sim - max(others) < self.margin:
                        decision[r] = ("amb", f"margin<{self.margin}")
                    else:
                        decision[r] = ("match", c, sim)
        for r, c in assigned.items():          # two slots claiming one id with near-equal scores
            for r2 in range(len(rows)):
                if (r2 != r and float(S[r2, c]) >= lo
                        and abs(float(S[r, c]) - float(S[r2, c])) < self.margin):
                    for x in (r, r2):
                        if decision[x][0] != "new":
                            decision[x] = ("amb", "contested")
        for r, k in enumerate(rows):
            d = decision[r]
            exc = info[k]["excerpt"]
            dur = len(exc) / self.sr
            if d[0] == "amb":
                res.status[k], res.reasons[k] = "ambiguous", d[1]
            elif d[0] == "match":
                i = ids[d[1]]
                res.mapping[k], res.status[k] = _sid(i), "resolved"
                res.confidence[k] = float(np.clip(d[2], 0.0, 1.0))
                st = self._ensure_resident(i)
                st["last"] = chunk_index
                if dur >= self.min_enroll_s:
                    st["sum"] = st["sum"] + _unit(self.embed(exc)) * dur
                    st["weight"] += dur
                    res.enrolled_s[_sid(i)] = res.enrolled_s.get(_sid(i), 0.0) + dur
            else:
                if dur < self.min_enroll_s:
                    # say WHY: eligibility removed audio, or the utterance is simply too short to enroll a new speaker
                    res.status[k], res.reasons[k] = "none", ("insufficient_eligible_audio" if info[k]["cut"] > 0
                                                              else "short_utterance")
                    continue
                i = self._next
                self._next += 1
                self._make_room(keep=None)
                self._active[i] = {"sum": _unit(self.embed(exc)) * dur, "weight": dur, "last": chunk_index}
                self._note()
                res.mapping[k], res.status[k] = _sid(i), "new"
                best = float(S[r].max()) if len(ids) else 0.0
                res.confidence[k] = float(np.clip(1.0 - max(best, 0.0), 0.0, 1.0))
                res.new_ids.append(_sid(i))
                res.enrolled_s[_sid(i)] = dur
        self._last_chunk = chunk_index
        return res

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "params": {"threshold": self.threshold, "ambiguous_band": self.ambiguous_band,
                       "margin": self.margin, "max_active": self.max_active, "min_rms": self.min_rms,
                       "min_enroll_s": self.min_enroll_s, "sr": self.sr},
            "next": self._next, "last_chunk": self._last_chunk,
            "active": {_sid(i): {"sum": [float(x) for x in st["sum"]], "weight": float(st["weight"]),
                                 "last": int(st["last"])} for i, st in sorted(self._active.items())},
            "spilled": {_sid(i): dict(m) for i, m in sorted(self._spilled.items())},
        }

    @classmethod
    def from_dict(cls, d, embed, spill_dir=None) -> "IdentityStore":
        if d.get("version") != 1:
            raise ValueError(f"unsupported IdentityStore version {d.get('version')!r}")
        self = cls(embed, spill_dir=spill_dir, **d["params"])
        self._next, self._last_chunk = int(d["next"]), int(d["last_chunk"])
        for name, st in d["active"].items():
            self._active[_num(name)] = {"sum": np.array(st["sum"], dtype=np.float64),
                                        "weight": float(st["weight"]), "last": int(st["last"])}
        for name, m in d["spilled"].items():
            if self.spill_dir is None or not (self.spill_dir / m["file"]).exists():
                raise RuntimeError(f"spilled identity file {m['file']} missing; pass the original spill_dir")
            self._spilled[_num(name)] = dict(m)
        return self


def stream_sims_for_continuity(prev: ChunkAssignment, curr: ChunkAssignment, store: IdentityStore,
                               name: Callable[[int], object] | None = None) -> dict:
    """Build ``associate.continuity``'s ``stream_embed_sim`` from two consecutive ChunkAssignments."""
    nm = name or (lambda s: s)
    out: dict = {}
    for ps, pid in prev.mapping.items():
        for cs, cid in curr.mapping.items():
            sim = 1.0 if pid == cid else min(0.99, max(0.0, store.id_similarity(pid, cid)))
            out[(nm(ps), nm(cs))] = float(sim)
    return out
