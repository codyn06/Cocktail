"""Internal, schema-versioned contracts for the AV scene pipeline (spec Section 9).

These are PRIVATE artifacts. They are not wire messages: nothing here is added to
`protocol/messages.py`, and `PairCalibration` / `VoiceMemory` keep their own types. Waveforms
and dense masks never appear in these JSON documents; they are referenced by relative path.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1
EVIDENCE_TIER_SYNTHETIC = "mocked/synthetic"
EvidenceTier = Literal["mocked/synthetic", "replayed", "hardware-tested", "untested"]

AssociationStatus = Literal["matched", "ambiguous", "unmatched", "rejected"]
Visibility = Literal["visible", "occluded", "outside_view", "unknown"]
RemainderKind = Literal["model", "raw_waveform_diagnostic", "phase_projected_stft", "unexplained_power"]
GapKind = Literal["gap", "padding", "missing_frames"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = SCHEMA_VERSION

    def to_json(self, **kw) -> str:
        return self.model_dump_json(**kw)

    @classmethod
    def from_json(cls, text: str):
        return cls.model_validate_json(text)


# ---------------------------------------------------------------- media index
class StreamTransform(_Model):
    """`t_common = a * t_stream + b` (microseconds); one offset is enough only without drift."""

    a: float = 1.0
    b_us: float = 0.0
    confidence: float = 1.0
    method: str = "identity"


class VideoStreamInfo(_Model):
    stream_id: str
    width: int
    height: int
    nominal_fps: float | None = None   # informational only; timing always comes from pts_us
    pts_us: list[int]                  # presentation timestamps on the stream clock
    transform: StreamTransform = Field(default_factory=StreamTransform)
    frames_dir: str | None = None
    video_path: str | None = None


class AudioViewInfo(_Model):
    rate_hz: int
    path: str
    n_samples: int
    channels: int
    derived_from: str                  # "master" or another view name


class AudioStreamInfo(_Model):
    stream_id: str
    sample_rate: int
    channels: int
    n_samples: int
    start_us: int = 0                  # stream clock time of sample 0
    transform: StreamTransform = Field(default_factory=StreamTransform)
    master_path: str
    layout: Literal["mono", "program_stereo", "calibrated_array"] = "mono"
    channel_order: list[str] = Field(default_factory=list)
    views: dict[str, AudioViewInfo] = Field(default_factory=dict)


class GapInterval(_Model):
    """Missing-data bookkeeping. Padding/gaps are NOT evidence that the room was silent."""

    stream_id: str
    start_us: int
    end_us: int
    kind: GapKind
    note: str = ""


class MediaIndex(_Model):
    run_id: str
    input_hashes: dict[str, str] = Field(default_factory=dict)
    origin_us: int = 0
    epoch: int = 0
    video: list[VideoStreamInfo] = Field(default_factory=list)
    audio: list[AudioStreamInfo] = Field(default_factory=list)
    gaps: list[GapInterval] = Field(default_factory=list)
    calibration_id: str | None = None
    ambient_temperature_c: float = 20.0
    temperature_provenance: Literal["assumed", "measured"] = "assumed"
    r_min_m: float = 0.5
    evidence_tier: EvidenceTier = EVIDENCE_TIER_SYNTHETIC


# ---------------------------------------------------------------- calibration
class CameraModel(_Model):
    """Pinhole intrinsics on UNDISTORTED pixels. Axes: +x right, +y down (image), rays are
    returned in the array frame (+x right, +y up, +z forward) by `locate.av_spatial`."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: list[float] = Field(default_factory=list)
    width: int | None = None
    height: int | None = None


class FittedPair(_Model):
    """Mirror of `locate.pair.PairCalibration` (empirical sine fit), kept distinct from geometry."""

    delay_amp_s: float
    axis_deg: float
    offset_s: float
    noise_s: float
    device: str = "quest3s"


class ArrayCalibration(_Model):
    calibration_id: str
    mic_positions_m: list[list[float]]          # n x 3 in the array frame (+x right, +y up, +z forward)
    channel_order: list[str]
    units: Literal["m"] = "m"
    axes: str = "x_right_y_up_z_forward"
    camera: CameraModel | None = None
    r_cam_to_array: list[list[float]] = Field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t_cam_to_array_m: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    prediction_model: Literal["geometric", "fitted_pair"] = "geometric"
    model_version: str = "1"
    channel_offsets_s: list[float] = Field(default_factory=list)
    channel_noise_s: float = 2e-6
    baffle_scale: float = 1.0                   # beta >= 1, geometric branch only
    bandwise_uncertainty: dict[str, float] = Field(default_factory=dict)
    fitted_pair: FittedPair | None = None

    @field_validator("baffle_scale")
    @classmethod
    def _baffle(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError("baffle_scale must be >= 1.0")
        return v


# ---------------------------------------------------------------- discovery / identity
class LabelScore(_Model):
    label: str
    score: float


class AcousticEvent(_Model):
    event_id: str
    labels: list[LabelScore]
    start_us: int
    end_us: int
    activity_score: float
    audio_prompt: str | None = None
    visual_noun_candidates: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)


class SpeakerTrack(_Model):
    speaker_id: str
    turns_us: list[tuple[int, int]]
    profile_ref: str | None = None
    count_evidence: dict = Field(default_factory=dict)
    anchors: dict = Field(default_factory=dict)  # AnchorSet.to_dict() incl. empty_reason


class MaskletTrack(_Model):
    track_id: str                      # namespaced: "<backend>:<session>:<obj>"
    concept: str
    mask_file: str
    image_height: int
    image_width: int
    n_frames: int
    first_pts_us: int | None = None
    last_pts_us: int | None = None
    mean_confidence: float = 1.0
    visibility_summary: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------- association
class ComponentScore(_Model):
    name: str
    value: float | None                # None == unavailable (never a pretend 0.0)
    quality: float = 1.0
    weight: float = 1.0
    note: str = ""


class SpatialEvidence(_Model):
    peaks_s: list[float] = Field(default_factory=list)
    onset_flux_weight: float | None = None
    aliasing_multimodal_flag: bool = False
    aliasing_pairs: list[str] = Field(default_factory=list)
    parallax_sigma_rad: float | None = None
    bandwidth_hz: float | None = None
    drr_confidence: float | None = None
    rt60_s: float | None = None
    calibration_quality: float | None = None


class EligibilityInterval(_Model):
    slot: int
    start_us: int
    end_us: int
    eligible: bool
    reason: str


class Association(_Model):
    association_id: str
    source_id: str
    track_id: str | None
    valid_start_us: int
    valid_end_us: int
    status: AssociationStatus
    visibility: Visibility
    score: float | None = None
    components: list[ComponentScore] = Field(default_factory=list)
    runner_up_margin: float | None = None
    evidence_dependencies: list[str] = Field(default_factory=list)
    spatial: SpatialEvidence | None = None
    causal_lag_ms: float | None = None
    face_quality: float | None = None
    learning_eligibility: list[EligibilityInterval] = Field(default_factory=list)


class SceneSource(_Model):
    source_id: str
    kind: Literal["speech", "non_speech", "unknown"] = "unknown"
    label: str = ""
    event_ids: list[str] = Field(default_factory=list)
    speaker_id: str | None = None
    emitter_id: str | None = None
    lifecycle: Literal["candidate", "confirmed", "retired"] = "candidate"
    visibility: Visibility = "unknown"
    association_revision: int = 0
    current_association_id: str | None = None


# ---------------------------------------------------------------- audio results
class StemResult(_Model):
    source_id: str
    model_revision: str
    prompt_hash: str | None = None
    anchor_hash: str | None = None
    sample_rate: int
    n_samples: int
    offset_samples: int = 0
    channels: int = 1
    raw_path: str
    normalized_path: str | None = None
    arrival_alignment: Literal["preserved"] = "preserved"
    errors: list[str] = Field(default_factory=list)
    linked_remainder_ids: list[str] = Field(default_factory=list)


class RemainderResult(_Model):
    remainder_id: str
    kind: RemainderKind
    source_stem_ids: list[str]
    downmix: str = "mean_over_channels"
    sample_rate: int
    n_samples: int
    offset_samples: int = 0
    stft: dict = Field(default_factory=dict)
    gain_bounds: tuple[float, float] | None = None
    projection_error_db: float | None = None
    artifact_path: str | None = None
    use: Literal["diagnostic", "discovery", "playback_proxy"]
    proxy_waveform_provenance: str | None = None


# ---------------------------------------------------------------- manifest
class StageRecord(_Model):
    name: str
    key: str
    status: Literal["complete", "skipped", "failed"] = "complete"
    seconds: float = 0.0
    detail: str = ""


class RunManifest(_Model):
    run_id: str
    evidence_tier: EvidenceTier = EVIDENCE_TIER_SYNTHETIC
    evidence_note: str = ("Synthetic fixture with oracle-derived stems. Not learned perception, "
                          "not room or device validation.")
    config_hash: str
    code_revision: str | None = None
    model_revisions: dict[str, str] = Field(default_factory=dict)
    input_hashes: dict[str, str] = Field(default_factory=dict)
    stages: list[StageRecord] = Field(default_factory=list)
    assets: dict[str, str] = Field(default_factory=dict)
    timing: dict = Field(default_factory=dict)
    envelope: dict = Field(default_factory=dict)
    hardware: dict = Field(default_factory=dict)
    capabilities: dict = Field(default_factory=dict)
    metrics_path: str | None = None
    media: MediaIndex | None = None
    calibration: ArrayCalibration | None = None
    events: list[AcousticEvent] = Field(default_factory=list)
    speakers: list[SpeakerTrack] = Field(default_factory=list)
    tracks: list[MaskletTrack] = Field(default_factory=list)
    sources: list[SceneSource] = Field(default_factory=list)
    associations: list[Association] = Field(default_factory=list)
    stems: list[StemResult] = Field(default_factory=list)
    remainders: list[RemainderResult] = Field(default_factory=list)


def canonical_json(obj) -> str:
    """Stable JSON used for hashing (sorted keys, no whitespace, floats via repr)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o):
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    from pathlib import Path
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")
