"""Hub ingest server (task B3): the client -> hub WebSocket channels of the frozen protocol."""

from __future__ import annotations

import logging
import time
from collections import deque
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
    inputs: deque = field(default_factory=lambda: deque(maxlen=1000))
    bad_messages: int = 0
    stats: dict[str, ChannelStats] = field(
        default_factory=lambda: {k: ChannelStats() for k in ("audio", "video", "pose")}
    )
    _resampler: Resampler | None = None
    _audio_epoch: int = -1
    _audio_next_ts_us: int | None = None  # expected device time of the next chunk
    _pose_epoch: int = -1
    max_gap_fill_us: int = 2_000_000

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
        # Keep ring time honest: zero-fill short forward gaps (lost chunks); start over on a
        # reconnect, a backward clock jump or a long gap.
        gap = None if self._audio_next_ts_us is None else h.device_ts_us - self._audio_next_ts_us
        if h.epoch != self._audio_epoch or (gap is not None and not -2_000 <= gap <= self.max_gap_fill_us):
            if self._audio_epoch != -1:
                log.warning("audio discontinuity (epoch %s->%s, gap %s us): reset", self._audio_epoch, h.epoch, gap)
            self.audio.reset()
            self._resampler = r = Resampler(h.sample_rate, h.channels)
        elif gap is not None and gap > 2_000:
            n = int(gap * self.audio.sr / 1e6)
            self.audio.write(np.zeros((n, h.channels), np.float32), self._audio_next_ts_us)
            self._resampler = r = Resampler(h.sample_rate, h.channels)
            log.warning("audio gap %d us: zero-filled", gap)
        self._audio_epoch = h.epoch
        self._audio_next_ts_us = h.device_ts_us + h.frames * 1_000_000 // h.sample_rate
        self.audio.write(r(x), h.device_ts_us)

    def on_video(self, h: p.VideoFrame, payload: bytes) -> None:
        self.stats["video"].update(h.seq, h.epoch, h.device_ts_us)
        self.video.put(h, payload)

    def on_pose_channel(self, msg) -> None:
        self.stats["pose"].update(msg.seq, msg.epoch, msg.device_ts_us)
        if msg.epoch != self._pose_epoch:
            self.poses.reset()  # new connection: the device clock may have restarted
            self._pose_epoch = msg.epoch
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
            "poses_out_of_order": self.poses.out_of_order,
            "audio_contiguous_since_us": a.contiguous_since_us if a else None,
            "bad_messages": self.bad_messages,
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
                data = await ws.receive_bytes()
                try:
                    handler(*p.unpack(data, model))
                except Exception:  # a bad message must not kill the stream
                    state.bad_messages += 1
                    log.exception("bad %s message dropped", name)
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
                text = await ws.receive_text()
                try:
                    state.on_pose_channel(_pose_channel.validate_json(text))
                except Exception:
                    state.bad_messages += 1
                    log.exception("bad pose-channel message dropped")
        except WebSocketDisconnect:
            log.info("pose client disconnected")

    @app.get("/ingest/status")
    def status():
        return state.status()

    return app
