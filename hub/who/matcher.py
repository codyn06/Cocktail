"""Interpretable audio-visual association (spec Section 7) and off-axis learning eligibility.

Scoring: every candidate source/track pair gets named `Evidence` items. An item whose value is
`None` is UNAVAILABLE (no face, no calibrated array, off-axis lips, ...) and is excluded from the
normalisation; it is never turned into a numeric 0.0 that would look like a negative observation.
Remaining items are combined as a quality-weighted mean, and the components are kept so the
viewer can show why a link was made.

Assignment: gated bipartite matching with explicit "unmatched" dummy options, then an absolute
acceptance threshold plus a margin over the runner-up. A source with no convincing track stays
unmatched; that is a statement about evidence, not proof that the source is off screen, so
association status and visibility are separate enums.

Timing: `lag = t_audio_onset - t_visual_onset` (positive = audio late). Acoustic time of flight
makes audio arrive 0-35 ms late at room distances; that range is free. Audio that LEADS the
picture beyond clock/frame uncertainty is penalised, as is lag far beyond the ToF range.

Status semantics: `Decision.status` is about the SOURCE (matched / ambiguous / unmatched). For an
unmatched source `Decision.score` is the best LOSING candidate's score, not a match score. `rejected`
is a per-candidate label (`Decision.candidates[track]["status"]`, evidence below `reject_threshold`);
it is never the status of a whole source.

Learning eligibility: lip-sync ASD is `unavailable` and voice-profile learning is frozen while
the observed face is turned more than 60 deg from the camera or the mouth is occluded. The gate
is bound to the audio interval and logical slot, so a later clear face cannot approve audio that
was already contaminated. `EligibilityGatedVoiceMemory` wraps `VoiceMemory` without changing its
signatures: it lets recognition run but restores every learning field for blocked slots.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

STATUS_MATCHED, STATUS_AMBIGUOUS, STATUS_UNMATCHED, STATUS_REJECTED = "matched", "ambiguous", "unmatched", "rejected"
OFF_AXIS_YAW_LIMIT_DEG = 60.0
BIG = 1e6


# ------------------------------------------------------------------ evidence
@dataclass
class Evidence:
    name: str
    value: float | None            # 0..1, 1 = supports the link; None = unavailable
    quality: float = 1.0           # 0..1 reliability of this observation
    weight: float = 1.0            # prior importance of this evidence type
    note: str = ""


@dataclass
class MatcherConfig:
    accept_threshold: float = 0.55
    reject_threshold: float = 0.25   # candidate links below this (with enough evidence) are 'rejected'
    margin: float = 0.10             # required lead over the runner-up
    min_evidence_weight: float = 0.5  # sum of weight*quality that must be available to score at all
    continuity_bonus: float = 0.05
    tof_min_ms: float = 0.0
    tof_max_ms: float = 35.0
    clock_uncertainty_ms: float = 5.0
    lead_penalty_span_ms: float = 20.0   # audio lead beyond uncertainty reaches score 0 over this span
    lag_decay_span_ms: float = 40.0      # lag beyond the ToF range decays to 0 over this span


@dataclass
class ScoreResult:
    score: float | None
    components: list[Evidence]
    available_weight: float


def causal_lag_score(lag_ms: float, cfg: MatcherConfig, quantization_ms: float = 0.0) -> float:
    """1.0 for lag in [tof_min - u, tof_max], decaying outside. lag = t_audio_onset - t_visual_onset."""
    u = cfg.clock_uncertainty_ms + quantization_ms
    if lag_ms < cfg.tof_min_ms - u:
        return max(0.0, 1.0 - (cfg.tof_min_ms - u - lag_ms) / cfg.lead_penalty_span_ms)
    if lag_ms > cfg.tof_max_ms:
        return max(0.0, 1.0 - (lag_ms - cfg.tof_max_ms) / cfg.lag_decay_span_ms)
    return 1.0


def lag_evidence(lags_ms: list[float], cfg: MatcherConfig, quantization_ms: float = 0.0,
                 weight: float = 1.0, name: str = "onset_synchrony") -> Evidence:
    """Onset-synchrony evidence from paired audio/visual onsets; unavailable without pairs."""
    if not lags_ms:
        return Evidence(name, None, 0.0, weight, "no paired onsets")
    scores = [causal_lag_score(l, cfg, quantization_ms) for l in lags_ms]
    q = min(1.0, len(lags_ms) / 3.0)
    return Evidence(name, float(np.mean(scores)), q, weight, f"{len(lags_ms)} onset pair(s), mean lag "
                    f"{np.mean(lags_ms):.1f} ms")


def combine(evidence: list[Evidence], cfg: MatcherConfig) -> ScoreResult:
    """Quality-weighted mean over AVAILABLE evidence; None when too little is available."""
    avail = [e for e in evidence if e.value is not None and e.quality > 0 and e.weight > 0]
    wsum = sum(e.weight * e.quality for e in avail)
    if not avail or wsum < cfg.min_evidence_weight:
        return ScoreResult(None, evidence, wsum)
    return ScoreResult(float(sum(e.weight * e.quality * float(e.value) for e in avail) / wsum), evidence, wsum)


# ------------------------------------------------------------------ assignment
@dataclass
class Decision:
    source_id: str
    track_id: str | None
    status: Literal["matched", "ambiguous", "unmatched", "rejected"]
    score: float | None
    runner_up_id: str | None = None
    runner_up_score: float | None = None
    margin: float | None = None
    components: list[Evidence] = field(default_factory=list)
    candidates: dict[str, dict] = field(default_factory=dict)   # track -> {score, status}
    reason: str = ""


def assign(sources: list[str], tracks: list[str], scores: dict[tuple[str, str], ScoreResult],
           cfg: MatcherConfig | None = None, previous: dict[str, str] | None = None,
           shared_emitter_tracks: set[str] | None = None) -> dict[str, Decision]:
    """One-to-one gated assignment with unmatched dummies; tracks in `shared_emitter_tracks`
    (e.g. a TV) may carry several sources."""
    cfg = cfg or MatcherConfig()
    previous = previous or {}
    shared = shared_emitter_tracks or set()
    S = len(sources)
    cols: list[str | None] = []
    for t in tracks:
        cols.extend([t] * (max(S, 1) if t in shared else 1))
    nT = len(cols)

    def eff(s: str, t: str) -> float | None:
        r = scores.get((s, t))
        if r is None or r.score is None:
            return None
        return min(1.0, r.score + (cfg.continuity_bonus if previous.get(s) == t else 0.0))

    cost = np.full((S, nT + S), BIG)
    for i, s in enumerate(sources):
        for j, t in enumerate(cols):
            e = eff(s, t)
            if e is not None and e >= cfg.reject_threshold:        # gate: weak links never compete
                cost[i, j] = -e
        cost[i, nT + i] = -cfg.accept_threshold + 1e-6              # staying unmatched is worth the threshold
    out: dict[str, Decision] = {}
    if S == 0:
        return out
    rows, chosen = linear_sum_assignment(cost)
    for i, j in zip(rows, chosen):
        s = sources[i]
        t = cols[j] if j < nT else None
        cands = {}
        for tt in tracks:
            r = scores.get((s, tt))
            e = eff(s, tt)
            st = "unmatched" if e is None else (STATUS_REJECTED if e < cfg.reject_threshold else "candidate")
            cands[tt] = {"score": e, "status": st}
        ranked = sorted(((v["score"], k) for k, v in cands.items() if v["score"] is not None), reverse=True)
        if t is None or cost[i, j] >= BIG / 2:
            best = ranked[0] if ranked else None
            reason = ("no available evidence for any track" if not ranked else
                      f"best candidate {best[1]} scored {best[0]:.2f} below acceptance {cfg.accept_threshold:.2f}"
                      if best[0] < cfg.accept_threshold else f"{best[1]} lost the assignment")
            out[s] = Decision(s, None, STATUS_UNMATCHED, best[0] if best else None,
                              components=scores[(s, best[1])].components if best else [], candidates=cands, reason=reason)
            continue
        sc = eff(s, t)
        others = [(v, k) for v, k in ranked if k != t]
        ru = others[0] if others else None
        margin = None if ru is None else sc - ru[0]
        status = STATUS_MATCHED
        reason = "score above threshold with margin"
        if sc < cfg.accept_threshold:
            status, t, reason = STATUS_UNMATCHED, None, "below acceptance threshold"
        elif margin is not None and margin < cfg.margin:
            status, reason = STATUS_AMBIGUOUS, f"margin {margin:.2f} over {ru[1]} below {cfg.margin:.2f}"
        out[s] = Decision(s, t if status != STATUS_UNMATCHED else None, status, sc, ru[1] if ru else None,
                          ru[0] if ru else None, margin,
                          scores[(s, cols[j])].components if status != STATUS_UNMATCHED else [], cands, reason)
    return out


# ------------------------------------------------------------------ off-axis learning eligibility
@dataclass
class FaceObservation:
    start_us: int
    end_us: int
    yaw_deg: float | None = None   # None = pose unknown (distinct from observed off-axis)
    occluded: bool = False


def lip_asd_available(face: FaceObservation | None, yaw_limit_deg: float = OFF_AXIS_YAW_LIMIT_DEG) -> bool:
    """Lip-sync ASD needs an observed, forward-ish, unoccluded face."""
    return bool(face and not face.occluded and face.yaw_deg is not None and abs(face.yaw_deg) <= yaw_limit_deg)


@dataclass
class _Interval:
    slot: int
    start_us: int
    end_us: int
    eligible: bool
    reason: str


class LearningEligibility:
    """Per logical slot, per audio interval permission to LEARN a voice profile.

    `unknown_pose` is the explicit policy for faces whose pose was never observed: 'allow' keeps
    the pre-existing behaviour, 'block' is stricter. Observed off-axis (|yaw| > 60 deg) or occluded
    always blocks. Overlap rule: audio is eligible only if NO overlapping interval blocks it, so a
    later clear observation cannot approve previously contaminated audio."""

    def __init__(self, unknown_pose: Literal["allow", "block"] = "allow", yaw_limit_deg: float = OFF_AXIS_YAW_LIMIT_DEG):
        self.unknown_pose = unknown_pose
        self.yaw_limit_deg = yaw_limit_deg
        self.intervals: list[_Interval] = []

    def observe(self, slot: int, face: FaceObservation | None) -> _Interval | None:
        if face is None:
            return None
        if face.occluded:
            eligible, why = False, "mouth_occluded"
        elif face.yaw_deg is None:
            eligible, why = self.unknown_pose == "allow", f"pose_unknown_policy_{self.unknown_pose}"
        elif abs(face.yaw_deg) > self.yaw_limit_deg:
            eligible, why = False, f"off_axis_yaw_{face.yaw_deg:.0f}deg"
        else:
            eligible, why = True, "face_forward"
        iv = _Interval(slot, int(face.start_us), int(face.end_us), eligible, why)
        self.intervals.append(iv)
        return iv

    def blocked(self, slot: int, start_us: int, end_us: int) -> tuple[bool, str | None]:
        for iv in self.intervals:
            if iv.slot == slot and not iv.eligible and iv.start_us < end_us and iv.end_us > start_us:
                return True, iv.reason
        return False, None

    def to_records(self) -> list[dict]:
        return [dict(slot=i.slot, start_us=i.start_us, end_us=i.end_us, eligible=i.eligible, reason=i.reason)
                for i in self.intervals]


class EligibilityGatedVoiceMemory:
    """Wraps `separate.voice_memory.VoiceMemory` (unchanged) with a learning gate.

    `reorder` runs normally so recognition against existing clean profiles continues. For each
    logical slot blocked over the emitted hop, all learning state touched by that call is
    restored: enrollment, return-window profile creation, history appends, active-duration
    counters, update scheduling and EMA adaptation."""

    def __init__(self, memory, eligibility: LearningEligibility):
        self.memory = memory
        self.eligibility = eligibility

    def _snapshot(self, slot: int):
        m = self.memory
        return (m.profiles[slot], deque(m._history[slot]), m._history_samples[slot], m._profile_active_s[slot],
                m._next_update_s[slot])

    def _restore(self, slot: int, snap) -> None:
        m = self.memory
        m.profiles[slot], m._history[slot], m._history_samples[slot], m._profile_active_s[slot], \
            m._next_update_s[slot] = snap

    def reorder(self, voices: np.ndarray, start_ts_us: int):
        from separate.voice_memory import SR

        m = self.memory
        snaps = {s: self._snapshot(s) for s in range(2)}
        result, diag = m.reorder(voices, start_ts_us)
        hop_end = int(start_ts_us + m.hop_samples * 1_000_000 // SR)
        blocked = {}
        for slot in range(2):
            is_blocked, why = self.eligibility.blocked(slot, int(start_ts_us), hop_end)
            if is_blocked:
                self._restore(slot, snaps[slot])
                blocked[slot] = why
        diag["learning_blocked"] = {f"s{k + 1}": v for k, v in blocked.items()}
        diag["profile_pair_similarity"] = (None if any(p is None for p in m.profiles)
                                           else round(m._cosine(m.profiles[0], m.profiles[1]), 4))
        m.last = diag
        if m.trace:
            m.trace[-1] = diag.copy()
        return result, diag
