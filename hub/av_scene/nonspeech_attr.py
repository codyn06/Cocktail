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


LEAK_SISDR_DB = 10.0           # candidate this similar to an already-explained source => leaked, not the target


def _si_sdr_db(est: np.ndarray, ref: np.ndarray) -> float:
    """SI-SDR of `est` against `ref` (higher = est is more a scaled copy of ref). -inf when undefined."""
    n = min(est.size, ref.size)
    if n == 0:
        return -np.inf
    e, r = np.asarray(est[:n], float), np.asarray(ref[:n], float)
    rr = float(r @ r)
    if rr < 1e-20 or float(e @ e) < 1e-20:
        return -np.inf
    proj = (float(e @ r) / rr) * r
    noise = e - proj
    nn = float(noise @ noise)
    return 10 * np.log10(float(proj @ proj) / nn) if nn > 1e-20 else np.inf


def absent_target_gate(candidate: np.ndarray, mix_window: np.ndarray,
                       floor_db: float = ABSENT_FLOOR_DB, rel_db: float = ABSENT_REL_DB, *,
                       other_sources: Sequence[np.ndarray] = (), judge_score: float | None = None,
                       judge_floor: float | None = None, leak_db: float = LEAK_SISDR_DB) -> str | None:
    """None if the candidate passes; else a rejection reason (prompted target absent).

    SAM Audio returns sound even for an absent target, so quietness alone is not enough: a LOUD
    candidate that is mostly a scaled copy of an already-explained source (`other_sources`, e.g. speech
    stems at the same rate and window) is leakage ("leaked_from_other_source(<SI-SDR dB>)"), and a judge
    score under `judge_floor` is rejected ("judge_below_floor"). Both extras are opt-in."""
    c = _rms_db(candidate)
    if c < floor_db:
        return f"silent_output({c:.1f}dB<{floor_db}dB)"
    m = _rms_db(mix_window)
    if np.isfinite(m) and c - m < rel_db:
        return f"tiny_vs_mix({c - m:.1f}dB<{rel_db}dB)"
    cand = np.asarray(candidate, float)
    for other in other_sources:
        sd = _si_sdr_db(cand, np.asarray(other, float))
        if sd >= leak_db:
            return f"leaked_from_other_source({sd:.1f}dB)"
    if judge_floor is not None and judge_score is not None and judge_score < judge_floor:
        return f"judge_below_floor({judge_score:.3g}<{judge_floor:.3g})"
    return None


def mask_motion_series(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Per-frame motion = XOR/union of consecutive masks (len = len(masks)); frame 0 = 0."""
    out = np.zeros(len(masks))
    for i in range(1, len(masks)):
        a, b = np.asarray(masks[i - 1], bool), np.asarray(masks[i], bool)
        out[i] = np.logical_xor(a, b).sum() / max(1, np.logical_or(a, b).sum())
    return out


def _corr(a, b) -> float:
    """Pearson r over frames where BOTH series are observed (NaN = missing, never zero-filled)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < MIN_FRAMES:
        return 0.0
    a, b = a[ok], b[ok]
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
    control_scores: dict = field(default_factory=dict)   # track -> r in the silent window, None = MISSING
    tdoa_ok: dict = field(default_factory=dict)          # track -> True/False, None = no spatial evidence
    missing_evidence: dict = field(default_factory=dict)  # {"control": [tracks], "spatial": [tracks]}


def attribute_nonspeech(sound_env: np.ndarray, track_motion: dict[str, np.ndarray],
                        tdoa_measured_s: float | None, tdoa_expected: dict[str, float],
                        tdoa_sigma_s: float, control_motion: dict[str, np.ndarray] | None = None,
                        min_corr: float = MIN_MOTION_CORR, margin: float = MIN_MARGIN,
                        sigma_mult: float = TDOA_SIGMA_MULT) -> Attribution:
    """sound_env and every motion array share one frame rate and window (NaN = not observed).

    control_motion[track] is the same track's motion in a SILENT window, correlated against the same
    envelope: a track that moves 'with the sound' equally in silence is not evidence. A MISSING control is
    not a passed control: that track cannot reach "matched" (reason `control_missing`) but still counts as
    a competitor. Likewise a missing/uncalibrated spatial measurement (`tdoa_measured_s is None` or no
    expected delay for the track) cannot support "matched" (reason `missing_spatial_evidence`).

    Inputs are the object-side evidence only. Extraction quality, judge/CLAP rank score and text
    similarity pick the AUDIO (see event_scope); they are deliberately not parameters here."""
    control_motion = control_motion or {}
    scores = {t: _corr(sound_env, m) for t, m in track_motion.items()}
    ctrl = {t: (_corr(sound_env, control_motion[t]) if t in control_motion else None) for t in track_motion}
    tdoa_ok: dict = {}
    for t in track_motion:
        if tdoa_measured_s is None or t not in tdoa_expected:
            tdoa_ok[t] = None
        else:
            tdoa_ok[t] = abs(tdoa_measured_s - tdoa_expected[t]) <= sigma_mult * tdoa_sigma_s
    base = dict(scores=scores, control_scores=ctrl, tdoa_ok=tdoa_ok)

    cand = [t for t in track_motion if scores[t] >= min_corr]
    if not cand:
        off = tdoa_measured_s is not None and not any(v for v in tdoa_ok.values())
        return Attribution("unmatched", "outside_view" if off else "unknown", None, None,
                           "no_motion_correlation", **base)
    no_ctrl = [t for t in cand if ctrl[t] is None]
    with_ctrl = [t for t in cand if ctrl[t] is not None and scores[t] - ctrl[t] >= margin]
    miss = {"control": list(no_ctrl), "spatial": []}
    if not with_ctrl:
        reason = "control_missing" if no_ctrl else "silent_control_explains"
        return Attribution("unmatched", "unknown", None, None, reason, missing_evidence=miss, **base)
    no_sp = [t for t in with_ctrl if tdoa_ok[t] is None]
    miss["spatial"] = list(no_sp)
    dir_ok = [t for t in with_ctrl if tdoa_ok[t]]
    if not dir_ok:
        if no_sp:
            return Attribution("unmatched", "unknown", None, None, "missing_spatial_evidence",
                               missing_evidence=miss, **base)
        return Attribution("unmatched", "outside_view", None, None, "tdoa_inconsistent",
                           missing_evidence=miss, **base)
    dir_ok.sort(key=lambda t: -scores[t])
    best = dir_ok[0]
    # tracks whose evidence is incomplete are not excluded: they may explain the sound just as well
    rivals = [scores[t] for t in dir_ok[1:] + no_sp + no_ctrl]
    gap = scores[best] - max(rivals) if rivals else scores[best]
    if rivals and gap < margin:
        return Attribution("ambiguous", "visible", None, scores[best],
                           "competing_track_explains_equally", runner_up_margin=gap,
                           missing_evidence=miss, **base)
    return Attribution("matched", "visible", best, scores[best], "ok", runner_up_margin=gap,
                       missing_evidence=miss, **base)
