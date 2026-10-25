"""Event-scoped non-speech processing (Phase 4): bounded windows, bounded candidates, sparse results.

Memory must not grow with clip length, so each AST event is handled in its own padded window of at most
`event_max_window_s`; a sustained event is split into contiguous windows. Inside a window:

  1. SAM 3 tracks the visual noun(s)                       (injected `tracker`)
  2. SAM Audio generates at most `max_candidates` stems in batches of `candidate_batch`, with RECORDED
     seeds `seed_base + i`                                  (injected `sam_audio`)
  3. each batch is ranked by the judge, else CLAP, else left "unranked" (seed order, status "degraded");
     candidate arrays outside the current batch are released, only the best per ranker tier is kept
  4. the absent-target gate runs on the chosen candidate
  5. object-side evidence (sound envelope, mask motion, TDOA) is resampled onto ONE timeline, the
     window's video-frame pts (integer us), with explicit validity; missing series stay None/NaN
  6. `attribute_nonspeech` picks the OBJECT from that evidence. Ranking picks the audio; extraction
     quality, rank score and text similarity never enter attribution.

The result is SPARSE: window-length 48 kHz stem plus its start sample on the master timeline. Callers
zero-fill inactive spans; nothing here allocates a full-clip array. Real-model behaviour is UNTESTED here:
every model is an injected callable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .contracts import MemoryCaps
from .nonspeech_attr import (ABSENT_FLOOR_DB, ABSENT_REL_DB, EVENT_PAD_S, MERGE_GAP_S, Attribution,
                             _rms_db, absent_target_gate, attribute_nonspeech, padded_event_windows)
from .schemas import Extraction

SAM_RATE = 48_000
_TIER = {"judge": 2, "clap": 1, "none": 0}


# ------------------------------------------------------------------------------------- windows
class EventWindow(tuple):
    """`(lo_s, hi_s)` (compares equal to a plain 2-tuple) plus continuation flags: ``continues_prev`` means this
    window starts where the previous split window of the same merged span ended, ``continues_next`` the same
    toward the following window."""

    continues_prev: bool
    continues_next: bool

    def __new__(cls, lo: float, hi: float, continues_prev: bool = False, continues_next: bool = False):
        self = super().__new__(cls, (float(lo), float(hi)))
        self.continues_prev, self.continues_next = continues_prev, continues_next
        return self

    @property
    def lo(self) -> float:
        return self[0]

    @property
    def hi(self) -> float:
        return self[1]

    def __getnewargs__(self):
        return (self[0], self[1], self.continues_prev, self.continues_next)


def _as_span(e) -> tuple[float, float]:
    if hasattr(e, "start_us"):
        return e.start_us / 1e6, e.end_us / 1e6
    return float(e[0]), float(e[1])


def bounded_event_windows(events: Sequence, duration_s: float, pad_s: float = EVENT_PAD_S,
                          merge_gap_s: float = MERGE_GAP_S, max_window_s: float = MemoryCaps.event_max_window_s
                          ) -> list[EventWindow]:
    """Pad + merge like `padded_event_windows`, then split any window longer than `max_window_s` into
    contiguous equal pieces (each <= max_window_s). Pieces share their edge value exactly, so the windows of
    one merged span tile it with no gap and no duplicate. `events`: (start_s, end_s) pairs or objects with
    start_us/end_us."""
    if max_window_s <= 0:
        raise ValueError("max_window_s must be positive")
    merged = padded_event_windows([_as_span(e) for e in events], pad_s, duration_s, merge_gap_s)
    out: list[EventWindow] = []
    for lo, hi in merged:
        n = max(1, math.ceil((hi - lo) / max_window_s - 1e-12))
        edges = [lo + (hi - lo) * i / n for i in range(n)] + [hi]
        for i in range(n):
            out.append(EventWindow(edges[i], edges[i + 1], i > 0, i < n - 1))
    return out


# ------------------------------------------------------------------------------------- config
@dataclass
class EventConfig:
    max_candidates: int = MemoryCaps.max_candidates      # SAM Audio candidates per window (hard cap)
    candidate_batch: int = MemoryCaps.candidate_batch    # candidates resident / ranked at once
    seed_base: int = 1000                                # candidate i uses seed seed_base + i (recorded)
    ranker: str = "judge"                                # "judge" | "clap" | "none"; falls back down this order
    min_judge_score: float | None = None                 # gate floor applied to the rank score (judge tier only)
    pad_s: float = EVENT_PAD_S
    merge_gap_s: float = MERGE_GAP_S
    max_window_s: float = MemoryCaps.event_max_window_s
    floor_db: float = ABSENT_FLOOR_DB
    rel_db: float = ABSENT_REL_DB
    tdoa_sigma_s: float = 1e-4
    envelope_hop_ms: float = 10.0
    envelope_win_ms: float = 40.0

    def validate(self) -> None:
        if self.ranker not in _TIER:
            raise ValueError(f"ranker must be one of {sorted(_TIER)}")
        if self.max_candidates < 1 or self.candidate_batch < 1:
            raise ValueError("max_candidates and candidate_batch must be >= 1")


@dataclass
class EventAdapters:
    """All injected; any may be None (that capability is then reported missing, never faked).

    sam_audio(audio_48k, prompt, seeds) -> Sequence[np.ndarray]
        mono 48 kHz candidate stems, one per seed, same order, each the window length. Called once per batch.
    judge(candidates, prompt, sample_rate, mix=None) -> Sequence[float]
        higher = better; `JudgeRanker.rank` fits (it needs `mix`, the window mixture, to score).
    clap(candidates, prompt, sample_rate, mix=None) -> Sequence[float]
        same shape; text/audio similarity. Fallback when the judge is missing or raises.
    tracker(frames, concept) -> {track_id: [mask|None per frame]}
        SAM 3 noun tracking over the window's frames (bool masks, or None where the track is absent).
    """
    sam_audio: Callable | None = None
    judge: Callable | None = None
    clap: Callable | None = None
    tracker: Callable | None = None


# ------------------------------------------------------------------------------------- evidence
@dataclass
class AlignedEvidence:
    """Everything on the window's video pts timeline. ``None`` = the series is MISSING; NaN = that frame is
    not observed. ``*_valid`` are explicit bool masks (True = observed)."""
    pts_us: list[int]
    envelope: np.ndarray | None = None
    envelope_valid: np.ndarray | None = None
    track_area: dict = field(default_factory=dict)       # track -> float[F] (NaN where no mask)
    track_motion: dict = field(default_factory=dict)     # track -> float[F] (NaN where either frame lacks a mask)
    track_valid: dict = field(default_factory=dict)      # track -> bool[F] (mask present)
    spatial: np.ndarray | None = None                    # measured TDOA (s) per frame
    spatial_valid: np.ndarray | None = None


def _resample_to_pts(t_us, values, pts_us) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of (t_us, values) at pts_us. Outside the source span or at non-finite source
    values the frame is invalid (NaN), never zero."""
    t = np.asarray(t_us, np.float64)
    v = np.asarray(values, np.float64)
    p = np.asarray(pts_us, np.float64)
    out = np.full(p.shape, np.nan)
    if t.size == 0:
        return out, np.zeros(p.shape, bool)
    inside = (p >= t[0]) & (p <= t[-1])
    good = np.isfinite(v)
    if good.sum() >= 1 and inside.any():
        out[inside] = np.interp(p[inside], t[good], v[good])
        # a frame whose neighbouring source samples are not finite is not observed either
        idx = np.clip(np.searchsorted(t, p[inside]), 0, t.size - 1)
        lo = np.clip(idx - 1, 0, t.size - 1)
        bad = ~(good[idx] & good[lo])
        tmp = out[inside]
        tmp[bad] = np.nan
        out[inside] = tmp
    return out, np.isfinite(out)


def mask_area_motion(masks: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(area, motion, valid) per frame. Frames without a mask are NaN/False, never 0; motion is the XOR/union
    fraction of consecutive masks and is observed only when both frames have masks (frame 0 is not)."""
    f = len(masks)
    area, motion, valid = np.full(f, np.nan), np.full(f, np.nan), np.zeros(f, bool)
    for i, m in enumerate(masks):
        if m is None:
            continue
        a = np.asarray(m, bool)
        valid[i] = True
        area[i] = float(a.sum())
        if i and masks[i - 1] is not None:
            b = np.asarray(masks[i - 1], bool)
            motion[i] = np.logical_xor(a, b).sum() / max(1, np.logical_or(a, b).sum())
    return area, motion, valid


def align_evidence(pts_us: Sequence[int], *, envelope: tuple | None = None,
                   track_masks: Mapping[str, Sequence] | None = None,
                   spatial: tuple | None = None) -> AlignedEvidence:
    """Resample onto ``pts_us`` (integer us, the window's video-frame timeline).

    envelope / spatial: ``(t_us, values)`` on their own clocks (or None = missing). track_masks: per track one
    entry (bool mask | None) per frame, already one per pts. Missing inputs stay missing."""
    pts = [int(p) for p in pts_us]
    ev = AlignedEvidence(pts_us=pts)
    if envelope is not None:
        ev.envelope, ev.envelope_valid = _resample_to_pts(envelope[0], envelope[1], pts)
    if spatial is not None:
        ev.spatial, ev.spatial_valid = _resample_to_pts(spatial[0], spatial[1], pts)
    for tid, masks in (track_masks or {}).items():
        if len(masks) != len(pts):
            raise ValueError(f"track {tid}: {len(masks)} masks for {len(pts)} frames")
        ev.track_area[tid], ev.track_motion[tid], ev.track_valid[tid] = mask_area_motion(masks)
    return ev


def rms_envelope(x: np.ndarray, rate_hz: int, start_us: int = 0, hop_ms: float = 10.0,
                 win_ms: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """(t_us, linear RMS) with a centred window, on the common clock (``start_us`` = time of sample 0)."""
    x = np.asarray(x, np.float64)
    hop, win = max(1, int(rate_hz * hop_ms / 1000)), max(1, int(rate_hz * win_ms / 1000))
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    centres = np.arange(0, x.size, hop)
    lo, hi = np.clip(centres - win // 2, 0, x.size), np.clip(centres + win // 2 + 1, 0, x.size)
    rms = np.sqrt((c[hi] - c[lo]) / np.maximum(hi - lo, 1))
    return start_us + np.round(centres * 1e6 / rate_hz), rms


def _tdoa_from_evidence(ev: AlignedEvidence, active: np.ndarray | None) -> float | None:
    """One TDOA per window: median of observed frames, preferring frames where the sound is active."""
    if ev.spatial is None or ev.spatial_valid is None or not ev.spatial_valid.any():
        return None
    ok = ev.spatial_valid & (active if active is not None else True)
    if not np.any(ok):
        ok = ev.spatial_valid
    return float(np.median(ev.spatial[ok]))


# ------------------------------------------------------------------------------------- result
@dataclass
class CandidateRecord:
    """Compact per-candidate record; the waveform itself is NOT kept (except the chosen one)."""
    seed: int
    rank_score: float | None
    ranker: str                  # "judge" | "clap" | "none"
    rms_db: float
    gate: str | None             # absent-target gate outcome for this candidate (None = passes)


@dataclass
class EventWindowResult:
    window_start_sample: int     # master timeline, at master_rate_hz
    window_end_sample: int
    master_rate_hz: int
    stem_start_sample: int       # chosen stem's start on the master timeline (== window_start_sample)
    stem: np.ndarray | None      # window-length mono 48 kHz, or None (absent target / unavailable)
    stem_rate_hz: int
    chosen_seed: int | None
    candidates: list[CandidateRecord]
    ranker_used: str             # tier of the winning candidate
    gate: str | None             # None = passed; else rejection reason
    attribution: Attribution | None
    extraction: Extraction
    evidence: AlignedEvidence | None = None
    continues_prev: bool = False
    continues_next: bool = False
    seeds: list[int] = field(default_factory=list)
    prompt: str = ""
    nouns: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return self.extraction.status


# ------------------------------------------------------------------------------------- run
def _union_mask(masks):
    u = None
    for m in masks:
        if m is None:
            continue
        m = np.asarray(m, bool)
        u = m.copy() if u is None else (u | m if u.shape == m.shape else u)
    return u


def match_control_tracks(event_tracks: dict, control_tracks: dict, min_iou: float = 0.3) -> dict:
    """control track key -> event track key, greedily by union-mask IoU (one-to-one). Tracks with no overlap are
    unmatched: the caller then has no control for that event track."""
    ev = {k: _union_mask(v) for k, v in event_tracks.items()}
    ct = {k: _union_mask(v) for k, v in control_tracks.items()}
    pairs = []
    for ck, cm in ct.items():
        for ek, em in ev.items():
            if cm is None or em is None or cm.shape != em.shape:
                continue
            inter, uni = int((cm & em).sum()), int((cm | em).sum())
            if uni and inter / uni >= min_iou:
                pairs.append((inter / uni, ck, ek))
    out, used_c, used_e = {}, set(), set()
    for _, ck, ek in sorted(pairs, reverse=True):
        if ck not in used_c and ek not in used_e:
            out[ck] = ek
            used_c.add(ck)
            used_e.add(ek)
    return out


def _rank_batch(batch: list[np.ndarray], prompt: str, cfg: EventConfig, ad: EventAdapters, mix,
                dead: set, notes: list[str]) -> tuple[list[float | None], str]:
    """Fallback order judge -> clap -> none, starting at cfg.ranker. A ranker that is missing, raises, or
    returns the wrong length is marked dead for the rest of the window."""
    order = [r for r in ("judge", "clap") if _TIER[r] <= _TIER[cfg.ranker]]
    for name in order:
        fn = getattr(ad, name)
        if name in dead:
            continue
        if fn is None:
            dead.add(name)
            notes.append(f"{name}_missing")
            continue
        try:
            scores = [float(s) for s in fn(batch, prompt, SAM_RATE, mix=mix)]
            if len(scores) != len(batch):
                raise ValueError(f"{name} returned {len(scores)} scores for {len(batch)} candidates")
            return scores, name
        except Exception as exc:   # noqa: BLE001 - any adapter failure degrades, never aborts the window
            dead.add(name)
            notes.append(f"{name}_failed({type(exc).__name__}: {exc})")
    return [None] * len(batch), "none"


def run_event_window(window: tuple, *, mix_48k: np.ndarray, prompt: str, nouns: Sequence[str],
                     frames, frame_pts_us: Sequence[int], adapters: EventAdapters,
                     config: EventConfig | None = None, master_rate_hz: int = SAM_RATE,
                     other_sources: Sequence[np.ndarray] = (), spatial: tuple | None = None,
                     tdoa_expected: Mapping[str, float] | None = None,
                     control_frames=None, control_pts_us: Sequence[int] | None = None,
                     window_start_us: int | None = None) -> EventWindowResult:
    """Process ONE event window. ``window`` = (lo_s, hi_s) (an `EventWindow` keeps its continuation flags);
    ``mix_48k`` is only that window's mono 48 kHz mixture; ``other_sources`` are already-explained sources
    (e.g. speech stems) at 48 kHz over the same window, used to reject leaked audio; ``spatial`` is
    ``(t_us, measured_tdoa_s)`` or None; ``tdoa_expected`` maps track -> calibrated expected delay or None;
    ``control_frames``/``control_pts_us`` are frames of a SILENT window for the counterfactual (None =
    missing control, which blocks "matched"). ``frame_pts_us`` are integer us on the common clock and
    ``window_start_us`` is the common-clock time of mix sample 0 (default ``round(lo*1e6)``)."""
    cfg = config or EventConfig()
    cfg.validate()
    lo, hi = float(window[0]), float(window[1])
    w0 = int(round(lo * master_rate_hz))
    w1 = int(round(hi * master_rate_hz))
    start_us = int(round(lo * 1e6)) if window_start_us is None else int(window_start_us)
    base = dict(window_start_sample=w0, window_end_sample=w1, master_rate_hz=master_rate_hz,
                stem_start_sample=w0, stem_rate_hz=SAM_RATE,
                continues_prev=bool(getattr(window, "continues_prev", False)),
                continues_next=bool(getattr(window, "continues_next", False)),
                prompt=prompt, nouns=list(nouns))
    if adapters.sam_audio is None:
        return EventWindowResult(stem=None, chosen_seed=None, candidates=[], ranker_used="none", gate=None,
                                 attribution=None, extraction=Extraction(
                                     status="unavailable", method="sam_audio", reason="sam_audio_adapter_missing"),
                                 **base)

    # ---- SAM Audio: bounded candidates, batches, recorded seeds, release outside the batch
    n_total = cfg.max_candidates
    seeds_all = [cfg.seed_base + i for i in range(n_total)]
    records: list[CandidateRecord] = []
    best: dict[str, tuple[float, int, np.ndarray]] = {}      # tier -> (score, seed, stem); one array per tier
    best_ok: dict[str, tuple[float, int, np.ndarray]] = {}   # same, restricted to candidates that pass the gate
    dead: set = set()
    notes: list[str] = []
    mix = np.asarray(mix_48k, np.float32)
    for b0 in range(0, n_total, cfg.candidate_batch):
        seeds = seeds_all[b0:b0 + cfg.candidate_batch]
        batch = [np.asarray(c, np.float32) for c in adapters.sam_audio(mix, prompt, list(seeds))]
        if len(batch) != len(seeds):
            raise ValueError(f"sam_audio returned {len(batch)} candidates for {len(seeds)} seeds")
        scores, ranker = _rank_batch(batch, prompt, cfg, adapters, mix, dead, notes)
        for seed, cand, sc in zip(seeds, batch, scores):
            floor = cfg.min_judge_score if ranker == "judge" else None
            gate = absent_target_gate(cand, mix, cfg.floor_db, cfg.rel_db, other_sources=other_sources,
                                      judge_score=sc if ranker == "judge" else None, judge_floor=floor)
            records.append(CandidateRecord(seed, sc, ranker, float(_rms_db(cand)), gate))
            # unranked: keep the lowest seed (candidate order); ranked: highest score
            key = sc if sc is not None else -float(seed)
            cur = best.get(ranker)
            if cur is None or key > cur[0]:
                best[ranker] = (key, seed, cand.copy())
            if gate is None:                                  # best candidate that actually PASSES the absent-target gate
                curp = best_ok.get(ranker)
                if curp is None or key > curp[0]:
                    best_ok[ranker] = (key, seed, cand.copy())
        del batch                                             # arrays beyond this batch are released

    # a gated top-ranked candidate must not hide a passing one: prefer the best passing candidate
    pool = best_ok if best_ok else best
    tier = max(pool, key=lambda r: _TIER[r])
    _, chosen_seed, stem = pool[tier]
    best.clear()
    best_ok.clear()
    chosen = next(r for r in records if r.seed == chosen_seed)
    degraded_reason = ""
    if tier == "none":
        degraded_reason = "unranked: " + (", ".join(dict.fromkeys(notes)) or "ranker=none") + "; candidates in seed order"
    elif _TIER[tier] < _TIER[cfg.ranker]:
        degraded_reason = f"ranker fell back to {tier}: " + ", ".join(dict.fromkeys(notes))

    if chosen.gate is not None:
        return EventWindowResult(
            stem=None, chosen_seed=chosen_seed, candidates=records, ranker_used=tier, gate=chosen.gate,
            attribution=Attribution("unmatched", "unknown", None, None, "target_absent"),
            extraction=Extraction(status="unavailable", method="sam_audio", reason=f"absent_target: {chosen.gate}"),
            seeds=seeds_all, **base)

    # ---- visual tracking + evidence on one timeline
    pts = [int(p) for p in frame_pts_us]
    track_masks: dict[str, list] = {}
    missing_notes: list[str] = []
    if adapters.tracker is None:
        missing_notes.append("tracker_missing")
    else:
        for noun in nouns:
            for tid, masks in adapters.tracker(frames, noun).items():
                track_masks[f"{noun}:{tid}"] = list(masks)
    env = rms_envelope(stem, SAM_RATE, start_us, cfg.envelope_hop_ms, cfg.envelope_win_ms)
    ev = align_evidence(pts, envelope=env, track_masks=track_masks, spatial=spatial)

    control_motion: dict[str, np.ndarray] | None = None
    if control_frames is not None and adapters.tracker is not None:
        control_motion = {}
        for noun in nouns:
            ctrl = {f"{noun}:{tid}": list(masks) for tid, masks in adapters.tracker(control_frames, noun).items()}
            # Track ids are only meaningful inside one tracker call, so the control run's ids cannot be compared with
            # the event window's. Pair them by where the object IS (union-mask IoU); an unpaired track keeps NO
            # control (explicit missing evidence), it is never given another object's motion.
            for ckey, ekey in match_control_tracks(
                    {k: v for k, v in track_masks.items() if k.startswith(noun + ":")}, ctrl).items():
                _, mot, _ = mask_area_motion(ctrl[ckey])
                # the control window has its own frames: compare on frame index with the same length
                fixed = np.full(len(pts), np.nan)
                fixed[:min(len(pts), mot.size)] = mot[:len(pts)]
                control_motion[ekey] = fixed
    attribution: Attribution | None = None
    if ev.envelope is not None and track_masks:
        env_series = np.where(ev.envelope_valid, ev.envelope, np.nan)
        thr = np.nanmax(env_series) * 0.5 if np.isfinite(env_series).any() else None
        active = None if thr is None else (np.nan_to_num(env_series, nan=0.0) >= thr)
        attribution = attribute_nonspeech(env_series, ev.track_motion, _tdoa_from_evidence(ev, active),
                                          dict(tdoa_expected or {}), cfg.tdoa_sigma_s, control_motion)
    elif not track_masks:
        attribution = Attribution("unmatched", "unknown", None, None, "no_tracks")

    reasons = [r for r in (degraded_reason, "; ".join(missing_notes)) if r]
    status = "degraded" if reasons else "ok"
    extraction = Extraction(status=status, method=f"sam_audio+{tier}", reason="; ".join(reasons))
    return EventWindowResult(stem=stem, chosen_seed=chosen_seed, candidates=records, ranker_used=tier,
                             gate=None, attribution=attribution, extraction=extraction, evidence=ev,
                             seeds=seeds_all, **base)
