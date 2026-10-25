"""Non-speech sound attribution and gates (slice 6a/6b). Pure functions; no model loads.

6a: padded AST-event windows for SAM Audio, cached-judge candidate ranking, absent-target gate.
6b: attribute a non-speech event to a SAM 3 track only when mask motion tracks the sound
envelope AND the calibrated TDOA direction agrees, and counterfactuals (competing track,
silent-window control) do not explain it as well. Statuses/visibility follow schemas.py.

Real-model behaviour is UNTESTED: all inputs here are synthetic/injected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Sequence

import numpy as np

# 6a
EVENT_PAD_S = 0.25
MERGE_GAP_S = 0.2
ABSENT_FLOOR_DB = -50.0        # candidate RMS below this (dBFS) => silent
ABSENT_REL_DB = -30.0          # candidate RMS this far below the mix window => absent
# 6b
MIN_MOTION_CORR = 0.5          # mask-motion vs envelope Pearson r
MIN_MARGIN = 0.2               # best must beat competitor / control by this
TDOA_SIGMA_MULT = 3.0          # |measured-expected| <= mult * sigma
MIN_FRAMES = 5


def padded_event_windows(events: Sequence[tuple[float, float]], pad_s: float, duration_s: float,
                         merge_gap_s: float = MERGE_GAP_S) -> list[tuple[float, float]]:
    """Pad each (start,end) event, clamp to [0,duration], merge windows closer than merge_gap_s."""
    wins = []
    for a, b in events:
        if b < a:
            a, b = b, a
        lo, hi = max(0.0, a - pad_s), min(duration_s, b + pad_s)
        if hi > lo:
            wins.append((lo, hi))
    wins.sort()
    out: list[list[float]] = []
    for lo, hi in wins:
        if out and lo - out[-1][1] <= merge_gap_s:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


@dataclass(frozen=True)
class Candidate:
    key: Hashable
    audio: np.ndarray
    prompt: str


def rank_candidates(candidates: Sequence[Candidate], judge: Callable[[np.ndarray, str], float],
                    cache: dict | None = None) -> list[tuple[Candidate, float]]:
    """Rank by judge(audio, prompt) descending; judge called once per (key, prompt) (cached)."""
    cache = {} if cache is None else cache
    scored = []
    for c in candidates:
        k = (c.key, c.prompt)
        if k not in cache:
            cache[k] = float(judge(c.audio, c.prompt))
        scored.append((c, cache[k]))
    return sorted(scored, key=lambda t: -t[1])  # stable: ties keep input order


def _rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return -np.inf
    r = float(np.sqrt(np.mean(x ** 2)))
    return 20 * np.log10(r) if r > 0 else -np.inf


def absent_target_gate(candidate: np.ndarray, mix_window: np.ndarray,
                       floor_db: float = ABSENT_FLOOR_DB, rel_db: float = ABSENT_REL_DB) -> str | None:
    """None if the candidate passes; else a rejection reason (prompted target absent)."""
    c = _rms_db(candidate)
    if c < floor_db:
        return f"silent_output({c:.1f}dB<{floor_db}dB)"
    m = _rms_db(mix_window)
    if np.isfinite(m) and c - m < rel_db:
        return f"tiny_vs_mix({c - m:.1f}dB<{rel_db}dB)"
    return None


def mask_motion_series(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Per-frame motion = XOR/union of consecutive masks (len = len(masks)); frame 0 = 0."""
    out = np.zeros(len(masks))
    for i in range(1, len(masks)):
        a, b = np.asarray(masks[i - 1], bool), np.asarray(masks[i], bool)
        out[i] = np.logical_xor(a, b).sum() / max(1, np.logical_or(a, b).sum())
    return out


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(a.size, b.size)
    if n < MIN_FRAMES:
        return 0.0
    a, b = a[:n], b[:n]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class Attribution:
    status: str                       # matched | ambiguous | unmatched (schemas.AssociationStatus)
    visibility: str                   # visible | outside_view | unknown
    track_id: str | None
    score: float | None
    reason: str
    runner_up_margin: float | None = None
    scores: dict = field(default_factory=dict)
    control_scores: dict = field(default_factory=dict)
    tdoa_ok: dict = field(default_factory=dict)


def attribute_nonspeech(sound_env: np.ndarray, track_motion: dict[str, np.ndarray],
                        tdoa_measured_s: float | None, tdoa_expected: dict[str, float],
                        tdoa_sigma_s: float, control_motion: dict[str, np.ndarray] | None = None,
                        min_corr: float = MIN_MOTION_CORR, margin: float = MIN_MARGIN,
                        sigma_mult: float = TDOA_SIGMA_MULT) -> Attribution:
    """sound_env and every motion array share one frame rate and window. control_motion[track] is the
    same track's motion in a SILENT window, correlated against the same envelope: a track that
    moves 'with the sound' equally in silence is not evidence."""
    control_motion = control_motion or {}
    scores = {t: _corr(sound_env, m) for t, m in track_motion.items()}
    ctrl = {t: _corr(sound_env, control_motion[t]) for t in track_motion if t in control_motion}
    tdoa_ok = {t: (tdoa_measured_s is not None and t in tdoa_expected
                   and abs(tdoa_measured_s - tdoa_expected[t]) <= sigma_mult * tdoa_sigma_s)
               for t in track_motion}
    base = dict(scores=scores, control_scores=ctrl, tdoa_ok=tdoa_ok)

    cand = [t for t in track_motion if scores[t] >= min_corr]
    if not cand:
        off = tdoa_measured_s is not None and not any(tdoa_ok.values())
        return Attribution("unmatched", "outside_view" if off else "unknown", None, None,
                           "no_motion_correlation", **base)
    cand = [t for t in cand if scores[t] - ctrl.get(t, 0.0) >= margin]
    if not cand:
        return Attribution("unmatched", "unknown", None, None, "silent_control_explains", **base)
    dir_ok = [t for t in cand if tdoa_ok[t]]
    if not dir_ok:
        off = tdoa_measured_s is not None
        return Attribution("unmatched", "outside_view" if off else "unknown", None, None,
                           "tdoa_inconsistent" if off else "no_tdoa", **base)
    dir_ok.sort(key=lambda t: -scores[t])
    best = dir_ok[0]
    others = [scores[t] for t in dir_ok[1:]]
    gap = scores[best] - max(others) if others else scores[best]
    if others and gap < margin:
        return Attribution("ambiguous", "visible", None, scores[best],
                           "competing_track_explains_equally", runner_up_margin=gap, **base)
    return Attribution("matched", "visible", best, scores[best], "ok", runner_up_margin=gap, **base)
