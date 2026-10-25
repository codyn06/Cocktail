"""Shared types for the live refinement bridge. No I/O, no models.

A refinement job is cut from the live hub when a source is chosen. It is slow (seconds, on the GPU), so
by the time it finishes the hub may have moved on. The rule that keeps wrong results out:

  A result is ADMITTED only if, at publish time on the event loop, the live hub still reports the same
  (source_id, epoch, selection_version, audio_resets), the job's slots still carry that source, the source
  still exists, and the result is not older than ``max_age_s``. Anything else is ``stale``.

Refinement never changes the selection, the route or the audio sent on /audio/out. It only publishes a
verdict about the chosen source. ``audio_applied`` is therefore always False in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

import numpy as np


class CutUnavailable(Exception):
    """A job cannot be cut right now (not audible, not speech, window too short, no video ...).
    ``str(e)`` is the reason published with state ``unavailable``."""


class RefineState(StrEnum):
    pending = "pending"          # a job exists for the live key and has not finished (or has not started yet)
    accepted = "accepted"        # the chosen voice matched a visible face: evidence only
    refused = "refused"          # analysed, and the voice was NOT matched to a face (stays unmatched)
    stale = "stale"              # finished, but the source/epoch/selection/audio changed or the result was too old
    unavailable = "unavailable"  # could not be analysed (no models, no video, not speech, not audible, error)


@dataclass(frozen=True)
class RefineKey:
    """Identity of what a job was cut for. All four fields are snapshotted at job creation."""

    source_id: str        # public tracker id (Source.source_id, e.g. "t3"); never a separator slot
    epoch: int            # ingest.pose_epoch
    selection_version: int
    audio_resets: int     # ingest.audio.resets: a gap resets LiveSeparator and the slots without a pose epoch change


@dataclass(frozen=True)
class LiveView:
    """What the hub says is current right now. Built on the event loop by ``key_fn``."""

    chosen: str | None
    epoch: int
    selection_version: int
    audio_resets: int
    slots: tuple[str, ...]        # slots carrying the chosen source now (selection.route)
    source_present: bool          # chosen id still in the tracker
    now_us: int                   # device-clock now (EventsState clock)


@dataclass
class RefineJob:
    """Everything a worker needs, snapshotted by device time at creation. Workers never re-read live state."""

    job_id: int
    key: RefineKey
    slots: tuple[str, ...]                 # route slots at creation, e.g. ("s1",)
    capture_start_us: int
    capture_end_us: int
    mix_16k: np.ndarray                    # (n,) float32 mono mixture over the window
    slot_audio: dict[str, np.ndarray]      # slot id -> (n,) float32, same window, gain-fitted hub streams
    frames: list[tuple[int, bytes]]        # (device_ts_us, jpeg) inside the window, oldest first
    label: str = "speech"                  # tracker label of the chosen source at creation
    created_mono: float = 0.0


@dataclass
class RefineOutcome:
    """Worker result before admission. ``state`` is accepted/refused/unavailable only."""

    state: RefineState
    reason: str
    track_id: str | None = None            # per-job visual track id; meaningless across jobs
    score: float | None = None             # accepted: TSE-vs-slot SI-SDR in dB
    margin_db: float | None = None
    assoc_conf: float = 0.0                # 0 unless accepted; in [0, 1]
    refined_audio: np.ndarray | None = None  # inspection only, never routed
    compute_ms: dict[str, float] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)   # frames, fps, face coverage, activity_s, thresholds


@dataclass
class RefineStatus:
    """What is published. One per (key, job); the newest for the live key is "current"."""

    job_id: int
    key: RefineKey
    state: RefineState
    reason: str
    track_id: str | None = None
    score: float | None = None
    assoc_conf: float = 0.0
    capture_start_us: int = 0
    capture_end_us: int = 0
    latency_ms: float | None = None        # job creation -> admission decision
    compute_ms: dict[str, float] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    audio_applied: bool = False            # always False in Phase 5

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "source_id": self.key.source_id, "epoch": self.key.epoch,
            "selection_version": self.key.selection_version, "audio_resets": self.key.audio_resets,
            "state": self.state.value, "reason": self.reason, "track_id": self.track_id,
            "score_db": self.score, "assoc_conf": self.assoc_conf,
            "capture_start_us": self.capture_start_us, "capture_end_us": self.capture_end_us,
            "latency_ms": self.latency_ms, "compute_ms": self.compute_ms, "evidence": self.evidence,
            "audio_applied": self.audio_applied,
        }


@dataclass
class RefineConfig:
    window_s: float = 4.0          # audio/video window per job
    min_window_s: float = 2.0      # shorter than this: unavailable("window too short")
    settle_s: float = 1.0          # wait after a selection before the first job, so the window holds the chosen voice
    refresh_s: float = 4.0         # re-run for the same key after this long (one job in flight; none queued)
    max_age_s: float = 12.0        # result older than this (live now - capture end) is stale
    min_video_fps: float = 8.0     # fewer frames per second than this: unavailable("insufficient video")
    min_face_coverage: float = 0.3  # fraction of window frames with a detected face
    history: int = 32              # statuses kept for diagnostics


# Worker: pure function of the job. Real one wraps run_speech_branch; tests pass fakes.
Runner = Callable[[RefineJob], RefineOutcome]
