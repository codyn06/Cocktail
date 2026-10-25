"""Phase 2 speech branch: separate voices, then decide which visible person (if any) each voice is.

Separation and attribution are different steps:
  1. MossFormer2_SS splits the mixture into 2 voice streams (the checkpoint is a 2-speaker model).
  2. TalkNet scores whether each visible face is speaking, per 25 fps frame.
  3. ``associate.associate_streams`` correlates stream activity with each face's speaking score.
     Weak or tied evidence stays ambiguous/unmatched; an active stream while every face is
     silent is ``outside_view``.
  4. Optionally AV_MossFormer2_TSE extracts a stream per face track, conditioned on the lips.

A SAM 3 "person" mask is not a face box. ``head_box`` takes the top of the person mask as a
heuristic face crop; it is wrong for back views, lying people and heavy occlusion.

Models are injected so the logic can be tested without weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .associate import FPS, associate_streams, select_for_track, track_activity
from .schemas import Association, Extraction

SR16 = 16000


def head_box(mask: np.ndarray, top_frac: float = 0.35) -> tuple[float, float, float, float] | None:
    """Square-ish head box from the top of a person mask, in pixels; None if the mask is empty."""
    rows = np.flatnonzero(mask.any(1))
    if rows.size == 0:
        return None
    y0, y1 = rows[0], rows[-1] + 1
    band = mask[y0:y0 + max(1, int((y1 - y0) * top_frac))]
    cols = np.flatnonzero(band.any(0))
    x0, x1 = cols[0], cols[-1] + 1
    w = x1 - x0
    return float(x0), float(y0), float(x1), float(min(y1, y0 + 1.1 * w))


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
        src = np.minimum((np.arange(n25) / FPS * fps).astype(int), len(valid) - 1)
        p[~np.asarray(valid)[src]] = np.nan
        scores[tid] = p
        info["tracks"][tid] = {"face_coverage": float(np.mean(valid)), **{k: v for k, v in sinfo.items()
                                                                           if isinstance(v, (int, float))}}

    assoc = associate_streams(env, scores)
    activity = track_activity(scores)
    selection = {tid: select_for_track(tid, assoc, streams, target_active=bool(activity[tid]),
                                       n_suspected_talkers=n_suspected_talkers)
                 for tid in boxes_by_track}

    tse_out: dict[str, np.ndarray] = {}
    if tse is not None:
        for tid, boxes in boxes_by_track.items():
            if activity.get(tid) and any(b is not None for b in boxes):
                tse_out[tid], _ = tse.extract_target(audio_16k, boxes, frames, fps)
    return SpeechResult(streams, scores, assoc, activity, selection, tse_out, info)
