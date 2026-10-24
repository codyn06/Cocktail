"""Hub ingest server (task B3): the client -> hub WebSocket channels of the frozen protocol."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter

import protocol as p

from .buffers import AudioRing, LatestFrame, PoseBuffer, Resampler

log = logging.getLogger("ingest")
_pose_channel = TypeAdapter(p.PoseChannelMessage)


@dataclass
class ChannelStats:
    messages: int = 0
    seq_gaps: int = 0
    last_seq: int = -1
    epoch: int = -1
    last_device_ts_us: int = 0
    # Hub monotonic minus device clock; the minimum over time approximates the clock offset
    # plus the smallest transport delay.
    min_offset_us: int | None = None

    def update(self, seq: int, epoch: int, device_ts_us: int) -> None:
        if epoch != self.epoch:
            self.epoch, self.last_seq = epoch, seq - 1
        if seq != self.last_seq + 1:
            self.seq_gaps += 1
        self.last_seq, self.last_device_ts_us = seq, device_ts_us
        self.messages += 1
        off = time.monotonic_ns() // 1000 - device_ts_us
        self.min_offset_us = off if self.min_offset_us is None else min(self.min_offset_us, off)


@dataclass
class IngestState:
    seconds: float = 10.0
    hello: p.Hello | None = None
    audio: AudioRing | None = None
    poses: PoseBuffer = None  # type: ignore[assignment]
    video: LatestFrame = field(default_factory=LatestFrame)
    inputs: list[p.InputEvent] = field(default_factory=list)
    stats: dict[str, ChannelStats] = field(
        default_factory=lambda: {k: ChannelStats() for k in ("audio", "video", "pose")}
    )
    _resampler: Resampler | None = None

    def __post_init__(self):
        self.poses = self.poses or PoseBuffer(self.seconds)

    def on_audio(self, h: p.AudioChunk, payload: bytes) -> None:
        self.stats["audio"].update(h.seq, h.epoch, h.device_ts_us)
        dtype = np.int16 if h.sample_format == "s16le" else np.float32
        x = np.frombuffer(payload, dtype).reshape(h.frames, h.channels).astype(np.float32)
        if dtype == np.int16:
            x /= 32768.0
        r = self._resampler
        if r is None or r.sr_in != h.sample_rate or r.channels != h.channels:
            self._resampler = r = Resampler(h.sample_rate, h.channels)
        if self.audio is None or self.audio.channels != h.channels:
            self.audio = AudioRing(h.channels, self.seconds)
        self.audio.write(r(x), h.device_ts_us)

    def on_video(self, h: p.VideoFrame, payload: bytes) -> None:
        self.stats["video"].update(h.seq, h.epoch, h.device_ts_us)
        self.video.put(h, payload)

    def on_pose_channel(self, msg) -> None:
        self.stats["pose"].update(msg.seq, msg.epoch, msg.device_ts_us)
        if isinstance(msg, p.Hello):
            self.hello = msg
        elif isinstance(msg, p.PoseSample):
            self.poses.add(msg.device_ts_us, msg.pos, msg.rot_xyzw)
        else:
            self.inputs.append(msg)

    def status(self) -> dict:
        a = self.audio
        return {
            "device": self.hello.device if self.hello else None,
            "audio_seconds_held": round(a.seconds_held, 3) if a else 0,
            "audio_channels": a.channels if a else 0,
            "poses_held": len(self.poses),
            "video_received": self.video.received,
            "video_dropped": self.video.dropped,
            "latest_frame_ts_us": self.video.header.device_ts_us if self.video.header else None,
            "inputs": len(self.inputs),
            "channels": {k: vars(v) for k, v in self.stats.items()},
        }


def create_app(state: IngestState | None = None) -> FastAPI:
    state = state or IngestState()
    app = FastAPI(title="Cocktail hub ingest")
    app.state.ingest = state

    async def binary_loop(ws: WebSocket, model, handler, name: str):
        await ws.accept()
        try:
            while True:
                h, payload = p.unpack(await ws.receive_bytes(), model)
                handler(h, payload)
        except WebSocketDisconnect:
            log.info("%s client disconnected", name)

    @app.websocket(p.WS_AUDIO_IN)
    async def audio_in(ws: WebSocket):
        await binary_loop(ws, p.AudioChunk, state.on_audio, "audio")

    @app.websocket(p.WS_VIDEO_IN)
    async def video_in(ws: WebSocket):
        await binary_loop(ws, p.VideoFrame, state.on_video, "video")

    @app.websocket(p.WS_POSE_IN)
    async def pose_in(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                state.on_pose_channel(_pose_channel.validate_json(await ws.receive_text()))
        except WebSocketDisconnect:
            log.info("pose client disconnected")

    @app.get("/ingest/status")
    def status():
        return state.status()

    return app
