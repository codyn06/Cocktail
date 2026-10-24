"""Cocktail wire protocol (task A2). FROZEN: change only by team agreement.

Quest (or any capture client) <-> laptop hub over WebSockets on a private hotspot or
USB (adb reverse). Python is the source of truth; `python -m protocol.export` writes
`protocol/schema/*.json` and `quest/Assets/Cocktail/Protocol/Messages.cs`.

Framing
- JSON text frames for everything except audio and video.
- Binary frames (audio, video): 4-byte little-endian header length N, N bytes of UTF-8
  JSON header, then the payload bytes.
Every client message carries seq (per channel), device_ts_us (headset clock) and epoch
(increments on every reconnect). Every target result carries target_id and
selection_version.
"""

from __future__ import annotations

import struct
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

PROTOCOL_VERSION = "1.0.0"
DEFAULT_PORT = 8765

# Endpoints (all WebSocket on ws://<hub>:DEFAULT_PORT)
WS_AUDIO_IN = "/ingest/audio"      # client -> hub, binary AudioChunk
WS_VIDEO_IN = "/ingest/video"      # client -> hub, binary VideoFrame
WS_POSE_IN = "/ingest/pose"        # client -> hub, JSON PoseSample | InputEvent | Hello
WS_EVENTS = "/events"              # hub -> client, JSON HubEvent; client -> hub, JSON Command
WS_AUDIO_OUT = "/audio/out"        # hub -> client, binary AudioOut


class Msg(BaseModel):
    seq: int = Field(ge=0)
    device_ts_us: int = Field(description="Headset monotonic clock, microseconds")
    epoch: int = Field(ge=0, description="Increments on every reconnect")


# ---------------- client -> hub ----------------


class Hello(Msg):
    """First message on /ingest/pose after connecting."""

    type: Literal["hello"] = "hello"
    device: str = Field(description='"quest3s", "quest2", "mock", "glasses"')
    protocol_version: str = PROTOCOL_VERSION
    mic_channels: int = Field(ge=1)
    mic_raw: bool = Field(description="True if the mic path is unprocessed")
    camera: bool
    camera_hfov_deg: float | None = None


class AudioChunk(Msg):
    """Binary on /ingest/audio. Payload: interleaved PCM."""

    sample_rate: int = Field(description="Hub resamples to 16 kHz at ingest")
    channels: int = Field(ge=1)
    sample_format: Literal["s16le", "f32le"] = "s16le"
    frames: int = Field(ge=1, description="Samples per channel")


class VideoFrame(Msg):
    """Binary on /ingest/video. Payload: JPEG. Hub keeps only the newest frame.

    cam_pose is the camera pose in the tracking space at capture time."""

    width: int
    height: int
    codec: Literal["jpeg", "h264"] = "jpeg"
    cam_pos: tuple[float, float, float] | None = None
    cam_rot_xyzw: tuple[float, float, float, float] | None = None
    hfov_deg: float | None = None


class PoseSample(Msg):
    """Head pose in Unity tracking space: +x right, +y up, +z forward (left-handed).

    Hub derives yaw (deg, + = right turn) from rot."""

    type: Literal["pose"] = "pose"
    pos: tuple[float, float, float]
    rot_xyzw: tuple[float, float, float, float]


class Action(StrEnum):
    select = "select"
    stop = "stop"
    replay = "replay"
    next = "next"
    prev = "prev"


class InputEvent(Msg):
    """A user action. For select, give a ray (head gaze or hand/controller) in
    tracking space, and/or a source_id if the user picked a labeled source."""

    type: Literal["input"] = "input"
    action: Action
    source_id: str | None = None
    ray_origin: tuple[float, float, float] | None = None
    ray_dir: tuple[float, float, float] | None = None
    modality: Literal["head", "hand", "controller", "ui"] = "ui"


PoseChannelMessage = Annotated[Union[Hello, PoseSample, InputEvent], Field(discriminator="type")]


class Mode(StrEnum):
    fast = "fast"
    balanced = "balanced"
    accurate = "accurate"


class SetMode(BaseModel):
    type: Literal["cmd.mode"] = "cmd.mode"
    mode: Mode


class SetVolumes(BaseModel):
    type: Literal["cmd.volumes"] = "cmd.volumes"
    inner_db: float = 0.0
    outer_db: float = -12.0


class AckAlert(BaseModel):
    type: Literal["cmd.ack_alert"] = "cmd.ack_alert"
    alert_id: str


Command = Annotated[Union[SetMode, SetVolumes, AckAlert], Field(discriminator="type")]


# ---------------- hub -> client ----------------


class TrackState(StrEnum):
    tracking = "tracking"
    last_seen = "last_seen"
    lost = "lost"


class BearingSource(StrEnum):
    head_motion = "head_motion"
    mics = "mics"
    camera = "camera"
    fused = "fused"


class Source(BaseModel):
    """One separated, tracked sound source (also the public part of the target record)."""

    source_id: str
    label: str = Field(description='"speech", "music", "alarm", ...')
    label_conf: float = Field(ge=0, le=1)
    bearing_deg: float | None = Field(default=None, description="World yaw, 0 = initial forward, + = right")
    bearing_conf: float = Field(default=0, ge=0, le=1)
    bearing_source: BearingSource | None = None
    elevation_deg: float | None = None
    distance_m: float | None = None
    state: TrackState = TrackState.tracking
    level_db: float = Field(description="Current loudness of the separated source")
    in_view: bool = False
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="Normalized x0,y0,x1,y1 in the latest camera frame"
    )
    mask_rle: str | None = None
    last_observed_us: int


class SourcesUpdated(BaseModel):
    """Sent at ~10 Hz. Stable source_ids; clients must not re-announce on every update."""

    type: Literal["sources.updated"] = "sources.updated"
    device_ts_us: int
    sources: list[Source]


class HubState(StrEnum):
    listening = "listening"
    selected = "selected"
    playing = "playing"
    target_lost = "target_lost"
    replay_processing = "replay_processing"
    replay_playing = "replay_playing"


class StateChanged(BaseModel):
    type: Literal["state.changed"] = "state.changed"
    state: HubState
    target_id: str | None = None
    selection_version: int = 0
    mode: Mode = Mode.balanced


class TargetUpdated(BaseModel):
    """Target record as sent to clients."""

    type: Literal["target.updated"] = "target.updated"
    target_id: str
    selection_version: int
    source_id: str
    kind: str = Field(description='"person" or a sound label')
    voice_lock_conf: float | None = None
    assoc_conf: float = Field(ge=0, le=1, description="How sure we are this object makes this sound")
    state: TrackState
    bearing_deg: float | None = None


class AlertFired(BaseModel):
    type: Literal["alert.fired"] = "alert.fired"
    alert_id: str
    label: str
    critical: bool
    spoken_text: str
    bearing_deg: float | None = None


class ReplayReady(BaseModel):
    type: Literal["replay.ready"] = "replay.ready"
    target_id: str
    selection_version: int
    duration_s: float


class Metrics(BaseModel):
    """Per-stage latency for the live dashboard, ~1 Hz."""

    type: Literal["metrics"] = "metrics"
    stage_ms: dict[str, float]
    gpu_util: float | None = None


HubEvent = Annotated[
    Union[SourcesUpdated, StateChanged, TargetUpdated, AlertFired, ReplayReady, Metrics],
    Field(discriminator="type"),
]


class AudioOut(BaseModel):
    """Binary on /audio/out. Payload: s16le, 48 kHz, 20 ms (960 frames).

    layout "layers": channel 0 = inner layer (mono), channel 1 = outer layer (mono);
    the Quest spatializes the inner layer at target_bearing_deg (Meta XR Audio).
    layout "stereo": already-panned stereo (glasses / simple clients)."""

    seq: int
    device_ts_us: int = Field(description="Capture time of the newest input sample used")
    layout: Literal["layers", "stereo"] = "layers"
    sample_rate: Literal[48000] = 48000
    channels: Literal[2] = 2
    frames: int = 960
    target_bearing_deg: float | None = None


# ---------------- binary framing ----------------


def pack(header: BaseModel, payload: bytes) -> bytes:
    h = header.model_dump_json().encode()
    return struct.pack("<I", len(h)) + h + payload


def unpack[T: BaseModel](data: bytes, model: type[T]) -> tuple[T, bytes]:
    (n,) = struct.unpack_from("<I", data)
    return model.model_validate_json(data[4 : 4 + n]), data[4 + n :]
