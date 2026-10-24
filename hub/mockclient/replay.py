"""Mock headset client (task B4): stream a session into the hub in real time, like the Quest.

  uv run python -m mockclient.replay sessions/synth [--speed 1] [--loop] [--fake-video-fps 15]
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import time
from pathlib import Path

import numpy as np
import websockets

import protocol as p

from .session import Session

CHUNK_MS = 20
# Not a decodable image: placeholder bytes so the video channel can be exercised without a camera.
FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


def events(s: Session, fake_video_fps: float = 0.0):
    """Yield (t_rel_s, kind, payload) in time order."""
    sr, ch = s.sample_rate, s.audio.shape[1]
    step = sr * CHUNK_MS // 1000
    out = [(i / sr, "audio", i) for i in range(0, len(s.audio) - step + 1, step)]
    out += [((q["device_ts_us"] - s.start_ts_us) / 1e6, "pose", q) for q in s.poses]
    if s.frames:
        out += [((ts - s.start_ts_us) / 1e6, "video", f) for ts, f in s.frames]
    elif fake_video_fps:
        dur = len(s.audio) / sr
        out += [(k / fake_video_fps, "video", None) for k in range(int(dur * fake_video_fps))]
    heapq.heapify(out := [(t, k, i, x) for i, (t, k, x) in enumerate(out)])
    while out:
        t, k, _, x = heapq.heappop(out)
        yield t, k, x


async def replay(s: Session, url: str, speed: float = 1.0, epoch: int = 0, fake_video_fps: float = 0.0,
                 ts_offset_us: int = 0) -> dict:
    sr, ch = s.sample_rate, s.audio.shape[1]
    step = sr * CHUNK_MS // 1000
    seq = {"audio": 0, "video": 0, "pose": 0}
    sent = dict.fromkeys(seq, 0)
    async with (websockets.connect(url + p.WS_AUDIO_IN) as wa,
                websockets.connect(url + p.WS_VIDEO_IN) as wv,
                websockets.connect(url + p.WS_POSE_IN) as wp):

        def nxt(k):
            seq[k] += 1
            return seq[k] - 1

        base = s.start_ts_us + ts_offset_us
        hello = p.Hello(seq=nxt("pose"), device_ts_us=base, epoch=epoch, device=s.meta.get("device", "mock"),
                        mic_channels=ch, mic_raw=True, camera=bool(s.frames or fake_video_fps),
                        camera_hfov_deg=s.meta.get("camera_hfov_deg"))
        await wp.send(hello.model_dump_json())
        t0 = time.perf_counter()
        for t, kind, x in events(s, fake_video_fps):
            if speed > 0:
                delay = t / speed - (time.perf_counter() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
            ts = base + int(t * 1e6)
            if kind == "audio":
                pcm = (np.clip(s.audio[x : x + step], -1, 1) * 32767).astype("<i2").tobytes()
                h = p.AudioChunk(seq=nxt("audio"), device_ts_us=ts, epoch=epoch, sample_rate=sr,
                                 channels=ch, frames=step)
                await wa.send(p.pack(h, pcm))
            elif kind == "pose":
                m = p.PoseSample(seq=nxt("pose"), device_ts_us=x["device_ts_us"] + ts_offset_us, epoch=epoch,
                                 pos=tuple(x["pos"]), rot_xyzw=tuple(x["rot_xyzw"]))
                await wp.send(m.model_dump_json())
            else:
                data = FAKE_JPEG if x is None else Path(x).read_bytes()
                h = p.VideoFrame(seq=nxt("video"), device_ts_us=ts, epoch=epoch, width=1280, height=960,
                                 hfov_deg=s.meta.get("camera_hfov_deg"))
                await wv.send(p.pack(h, data))
            sent[kind] += 1
    return sent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--hub", default=f"ws://127.0.0.1:{p.DEFAULT_PORT}")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--fake-video-fps", type=float, default=0.0)
    a = ap.parse_args()
    s = Session.load(a.session)
    dur_us, epoch = int(len(s.audio) / s.sample_rate * 1e6), 0
    while True:
        # Each loop is a new epoch and continues the device clock, like a reconnect.
        sent = asyncio.run(replay(s, a.hub, a.speed, epoch, a.fake_video_fps, ts_offset_us=epoch * dur_us))
        print(json.dumps({"epoch": epoch, "sent": sent}))
        if not a.loop:
            break
        epoch += 1


if __name__ == "__main__":
    main()
