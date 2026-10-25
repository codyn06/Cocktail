"""Phase 2 speech branch: separate voices, then decide which visible person (if any) each voice is.

Separation and attribution are different steps:
  1. MossFormer2_SS splits the mixture into 2 voice streams (the checkpoint is a 2-speaker model).
  2. TalkNet scores whether each visible face is speaking, per 25 fps frame.
  3. ``associate.associate_streams`` correlates stream activity with each face's speaking score.
     Weak or tied evidence stays ambiguous/unmatched; an active stream while every face is
     silent is ``outside_view``.
  4. Optionally AV_MossFormer2_TSE extracts a stream per face track, conditioned on the lips.

Face boxes come from ``vision.face_detect`` (S3FD faces assigned to SAM 3 person masks); a frame
without a detected face carries no speaking evidence.

Models are injected so the logic can be tested without weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .associate import FPS, associate_streams, select_for_track, track_activity
from .schemas import Association, Extraction

SR16 = 16000


def frame_envelope(x: np.ndarray, n_frames: int, sr: int = SR16, fps: int = FPS) -> np.ndarray:
    """RMS per video frame at `fps`."""
    hop = sr // fps
    x = np.pad(x, (0, max(0, n_frames * hop - len(x))))[: n_frames * hop]
    return np.sqrt(np.mean(np.square(x.reshape(n_frames, hop), dtype=np.float64), axis=1))


def logit_to_prob(s: np.ndarray) -> np.ndarray:
    """TalkNet scores are raw logits (>0 == speaking); associate.py thresholds expect [0, 1]."""
    return 1.0 / (1.0 + np.exp(-np.asarray(s, dtype=np.float64)))


@dataclass
class SpeechResult:
    streams: dict[str, np.ndarray]                       # stream id -> 16 kHz waveform
    track_scores: dict[str, np.ndarray]                  # track id -> speaking prob per 25 fps frame (NaN = no face)
    associations: list[Association]
    activity_us: dict[str, list[tuple[int, int]]]
    selection: dict[str, tuple[str | None, Extraction]]  # track id -> (stream id | None, Extraction)
    tse: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def run_speech_branch(audio_16k: np.ndarray, frames: np.ndarray, fps: float,
                      boxes_by_track: dict[str, Sequence], *,
                      separator: Callable[[np.ndarray], np.ndarray],
                      scorer=None, tse=None, n_suspected_talkers: int | None = None) -> SpeechResult:
    """frames: uint8 RGB [N,H,W,3]; boxes_by_track: per-frame head box or None for each track."""
    n25 = int(len(audio_16k) / SR16 * FPS)
    sep = separator(audio_16k)
    streams = {f"voice_{k}": sep[k].astype(np.float32) for k in range(sep.shape[0])}
    env = {sid: frame_envelope(s, n25) for sid, s in streams.items()}

    scores: dict[str, np.ndarray] = {}
    info: dict = {"tracks": {}}
    for tid, boxes in boxes_by_track.items():
        valid = [b is not None for b in boxes]
        if not any(valid):
            scores[tid] = np.full(n25, np.nan)
            info["tracks"][tid] = {"face_coverage": 0.0}
            continue
        if scorer is None:
            scores[tid] = np.full(n25, np.nan)
            continue
        raw, _, sinfo = scorer.score_track(audio_16k, boxes, frames, fps)
        p = np.full(n25, np.nan)
        p[: min(n25, len(raw))] = logit_to_prob(raw[:n25])
        # frames with no detected head carry no speaking evidence (not "silent")
        src = np.minimum(np.rint(np.arange(n25) / FPS * fps).astype(int), len(valid) - 1)   # as resample_indices
        p[~np.asarray(valid)[src]] = np.nan
        scores[tid] = p
        info["tracks"][tid] = {"face_coverage": float(np.mean(valid)), **{k: v for k, v in sinfo.items()
                                                                           if isinstance(v, (int, float))}}

    activity = track_activity(scores)
    active_s = {t: sum(b - a for a, b in v) / 1e6 for t, v in activity.items()}
    active = {t: s >= MIN_ACTIVE_S for t, s in active_s.items()}   # absolute: short windows refuse

    if tse is None:
        assoc = associate_streams(env, scores)
        selection = {tid: select_for_track(tid, assoc, streams, target_active=active[tid],
                                           n_suspected_talkers=n_suspected_talkers)
                     for tid in boxes_by_track}
        return SpeechResult(streams, scores, assoc, activity, selection, {}, info)

    tse_out: dict[str, np.ndarray] = {}
    for tid, boxes in boxes_by_track.items():
        if active[tid] and any(b is not None for b in boxes):
            tse_out[tid], _ = tse.extract_target(audio_16k, boxes, frames, fps)
    info["tse_similarity_db"] = {t: {s: round(_si_sdr(x, streams[s]), 1) for s in streams} for t, x in tse_out.items()}
    assoc, selection = _tse_attribution(streams, env, tse_out, active, active_s, scores, n25)
    streams.update({f"tse:{tid}": x for tid, x in tse_out.items()})
    return SpeechResult(streams, scores, assoc, activity, selection, tse_out, info)


MIN_ACTIVE_S = 1.0       # TalkNet activity needed before a person gets a stream (B_alone: silent A had 0.52 s)
MIN_AGREE = 0.5          # fraction of a person's speaking frames during which the matched voice must be active
CONFLICT_RATIO = 2.0     # a contested voice goes to one claimant only if it has 2x the activity ...
CONFLICT_LEAD_S = 1.0    # ... and at least 1 s more; otherwise every claimant is refused
OFFSCREEN_OVERLAP = 0.3  # an unmatched voice is outside_view if under 30% of its frames overlap visible speech
TSE_MATCH_DB = 3.0       # TSE output must resemble one separated voice by this SI-SDR ...
TSE_MARGIN_DB = 3.0      # ... and beat the other voice by this much


def _si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    if n == 0 or not (np.isfinite(est).all() and np.isfinite(ref).all()):
        return float("-inf")
    a = np.dot(est, ref) / max(np.dot(ref, ref), 1e-12)
    e = est - a * ref
    return float(10 * np.log10(max(np.dot(a * ref, a * ref), 1e-12) / max(np.dot(e, e), 1e-12)))


def _voice_active(env: np.ndarray) -> np.ndarray:
    peak = float(env.max()) if env.size else 0.0
    return env > max(0.1 * peak, 1e-3) if peak > 1e-3 else np.zeros_like(env, bool)


def _tse_attribution(streams, env, tse_out, active, active_s, scores, n25):
    """Each active person's AV-TSE output is matched to the separated voice it resembles, and that voice
    must be active while the person's face is speaking. A voice claimed by several people goes to one
    only with a clear activity lead; otherwise all claimants are refused. The listening stream is the
    TSE output (AV-conditioned), keyed ``tse:<track>`` by the caller."""
    from .schemas import ComponentScore

    speaking = {t: np.nan_to_num(scores[t], nan=0.0) >= 0.5 for t in scores}
    vact = {sid: _voice_active(e) for sid, e in env.items()}
    claims: dict[str, list[tuple[str, float, float | None, float]]] = {}
    no_match: dict[str, str] = {}
    for tid, x in tse_out.items():
        sims = sorted(((_si_sdr(x, s), sid) for sid, s in streams.items()), reverse=True)
        best, sid = sims[0]
        margin = best - sims[1][0] if len(sims) > 1 else None
        agree = float(vact[sid][speaking[tid]].mean()) if speaking[tid].any() else 0.0
        if best < TSE_MATCH_DB or (margin is not None and margin < TSE_MARGIN_DB):
            no_match[tid] = "TSE output matches no separated voice"
        elif agree < MIN_AGREE:
            no_match[tid] = f"matched voice inactive while face speaks ({agree:.0%})"
        else:
            claims.setdefault(sid, []).append((tid, best, margin, agree))

    owner: dict[str, tuple[str, float, float | None, float]] = {}
    conflicted: dict[str, str] = {}
    for sid, cs in claims.items():
        cs.sort(key=lambda c: -active_s[c[0]])
        lead = len(cs) == 1 or (active_s[cs[0][0]] >= CONFLICT_RATIO * active_s[cs[1][0]]
                                and active_s[cs[0][0]] - active_s[cs[1][0]] >= CONFLICT_LEAD_S)
        if lead:
            owner[sid] = cs[0]
            losers = cs[1:]
        else:
            losers = cs
        for c in losers:
            conflicted[c[0]] = ", ".join(o[0] for o in cs if o[0] != c[0])

    dur_us = int(n25 / FPS * 1e6)
    vis_speaking = np.zeros(n25, bool)
    for t, sp in speaking.items():
        vis_speaking |= sp
    assoc: list[Association] = []
    for sid in streams:
        if sid in owner:
            tid, sim, margin, agree = owner[sid]
            assoc.append(Association(
                association_id=f"assoc_{sid}", source_id=sid, track_id=tid, valid_start_us=0, valid_end_us=dur_us,
                status="matched", visibility="visible", score=round(sim, 2), runner_up_margin=margin,
                components=[ComponentScore(name="tse_stream_si_sdr_db", value=round(sim, 2)),
                            ComponentScore(name="voice_active_while_face_speaks", value=round(agree, 3), weight=0.0),
                            ComponentScore(name="talknet_active_s", value=round(active_s[tid], 2), weight=0.0)]))
            continue
        va = vact[sid]
        if va.mean() < 0.05:
            vis, note = "unknown", "voice stream near-silent"
        else:
            overlap = float(vis_speaking[va].mean())
            vis = "outside_view" if overlap < OFFSCREEN_OVERLAP else "unknown"
            note = f"{overlap:.0%} of its active frames overlap a visible face speaking"
        status = "unmatched"
        if sid in claims and sid not in owner:
            status, note = "ambiguous", "claimed by " + ", ".join(c[0] for c in claims[sid])
        assoc.append(Association(
            association_id=f"assoc_{sid}", source_id=sid, track_id=None, valid_start_us=0, valid_end_us=dur_us,
            status=status, visibility=vis,
            components=[ComponentScore(name="tse_stream_si_sdr_db", value=None, note=note)]))

    owned = {o[0]: sid for sid, o in owner.items()}
    selection = {}
    for tid in active:
        if not active[tid]:
            selection[tid] = (None, Extraction(status="unavailable", method="av_tse", reason="target inactive"))
        elif tid in conflicted:
            selection[tid] = (None, Extraction(status="unavailable", method="av_tse",
                                               reason=f"ambiguous: same voice as {conflicted[tid]}"))
        elif tid in owned:
            selection[tid] = (f"tse:{tid}", Extraction(status="ok", method="av_tse", stem_source_id=f"tse:{tid}",
                                                       reason=f"matches separated {owned[tid]}"))
        else:
            selection[tid] = (None, Extraction(status="unavailable", method="av_tse",
                                               reason="ambiguous: " + no_match.get(tid, "no TSE output")))
    return assoc, selection
