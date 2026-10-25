"""Record a headset session (mic audio + head pose) to disk, for real-world tests.

Acts like the hub from the headset's point of view (all five sockets connect), but only
saves data. Stop with Ctrl+C or use --seconds.

Run from the repo root (hub venv):
    uv run --project hub python quest/tools/record_session.py --out hub/recordings/take1 --truth 90 --seconds 25

Writes OUT/audio.wav (int16, interleaved), OUT/pose.csv (t_us,qx,qy,qz,qw), OUT/meta.json,
and OUT/audio_chunks.csv (seq,device_ts_us,frames,epoch of every audio chunk, so
eval/dataset_check.py can look for dropped chunks and clock jumps; audio.wav is unchanged).
`--truth` is the angle you placed the sound source at, in degrees from where your head
faced when you started (+ = right, 180 = behind); it is stored for the scorer.
Optional labels stored in meta.json for the eval coverage grid (eval/dataset_check.py):
`--distance-m` (source distance in meters), `--motion small|normal|large`,
`--stimulus pink|speech`. They default to null.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hub"))

import uvicorn  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

import protocol as p  # noqa: E402

app = FastAPI(title="Cocktail session recorder")
S = {
    "audio": [], "audio_rate": None, "audio_ch": None, "first_audio_ts_us": None,
    "audio_frames": 0, "pose": [], "hello": None, "video_frames": 0, "last_audio_ts_us": None,
    "t_first_wall": None, "cue": None,
    "chunks": [], "clock_restarts": 0,
}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "audio": len(S["audio"]), "pose": len(S["pose"]), "video": S["video_frames"],
            "audio_frames": S["audio_frames"]}


@app.websocket(p.WS_AUDIO_IN)
async def ws_audio(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            data = msg.get("bytes")
            if data is None:
                continue
            hdr, pcm = p.unpack(data, p.AudioChunk)
            if hdr.sample_format != "s16le":
                continue
            if S["last_audio_ts_us"] is not None and hdr.device_ts_us < S["last_audio_ts_us"] - 2_000_000:
                # the app restarted (its clock restarted from 0): keep only the new session
                print("[audio] headset clock restarted; discarding the earlier session")
                S["audio"], S["audio_frames"], S["audio_rate"] = [], 0, None
                S["pose"] = []
                S["chunks"] = []
                S["clock_restarts"] += 1
            if S["audio_rate"] is None:
                S["audio_rate"], S["audio_ch"], S["first_audio_ts_us"] = hdr.sample_rate, hdr.channels, hdr.device_ts_us
                S["t_first_wall"] = time.monotonic()
                print(f"[audio] first chunk: {hdr.sample_rate} Hz, {hdr.channels} ch, ts={hdr.device_ts_us}")
            S["audio"].append(pcm)
            S["audio_frames"] += hdr.frames
            S["chunks"].append((hdr.seq, hdr.device_ts_us, hdr.frames, hdr.epoch))
            S["last_audio_ts_us"] = hdr.device_ts_us
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket(p.WS_VIDEO_IN)
async def ws_video(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                S["video_frames"] += 1
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket(p.WS_POSE_IN)
async def ws_pose(ws: WebSocket) -> None:
    await ws.accept()
    ad = TypeAdapter(p.PoseChannelMessage)
    try:
        while True:
            m = ad.validate_json(await ws.receive_text())
            if isinstance(m, p.Hello):
                S["hello"] = m.model_dump()
            elif isinstance(m, p.PoseSample):
                q = m.rot_xyzw
                if S["pose"] and m.device_ts_us < S["pose"][-1][0] - 2_000_000:
                    S["pose"] = []  # app restarted: new clock
                    S["pose_restarts"] = S.get("pose_restarts", 0) + 1
                S["pose"].append((m.device_ts_us, q[0], q[1], q[2], q[3]))
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket(p.WS_EVENTS)
async def ws_events(ws: WebSocket) -> None:
    await ws.accept()
    try:
        await ws.send_text(p.StateChanged(state=p.HubState.listening).model_dump_json())
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket(p.WS_AUDIO_OUT)
async def ws_audio_out(ws: WebSocket) -> None:
    """Silence, plus cue beeps in the headset: 2 beeps = start turning, 3 beeps = stop."""
    await ws.accept()
    import numpy as np

    n0 = 0
    try:
        seq, nxt = 0, time.monotonic()
        while True:
            pcm = np.zeros((960, 2), dtype="<i2")
            cue, t0w = S["cue"], S["t_first_wall"]
            if cue and t0w is not None:
                el = time.monotonic() - t0w
                beeps = [(cue["go"] + 0.3 * i, 0.15) for i in range(2)] + [(cue["stop"] + 0.3 * i, 0.15) for i in range(3)]
                if any(b <= el < b + d for b, d in beeps):
                    tone = 0.3 * np.sin(2 * np.pi * 1000 * (n0 + np.arange(960)) / 48000)
                    pcm[:] = (tone * 32767).astype("<i2")[:, None]
            n0 += 960
            await ws.send_bytes(p.pack(p.AudioOut(seq=seq, device_ts_us=0, layout="stereo"), pcm.tobytes()))
            seq += 1
            nxt += 0.02
            await asyncio.sleep(max(0, nxt - time.monotonic()))
    except (WebSocketDisconnect, RuntimeError):
        pass


def save(out: Path, truth: float | None, note: str, distance_m: float | None = None,
         motion: str | None = None, stimulus: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if S["audio_rate"] is None:
        print("No audio received: nothing saved. Is the headset app running and pointed at this laptop?")
        return
    with wave.open(str(out / "audio.wav"), "wb") as w:
        w.setnchannels(S["audio_ch"]); w.setsampwidth(2); w.setframerate(S["audio_rate"])
        w.writeframes(b"".join(S["audio"]))
    with open(out / "pose.csv", "w") as f:
        f.write("t_us,qx,qy,qz,qw\n")
        for row in S["pose"]:
            f.write(",".join(str(v) for v in row) + "\n")
    with open(out / "audio_chunks.csv", "w") as f:
        f.write("seq,device_ts_us,frames,epoch\n")
        for row in S["chunks"]:
            f.write(",".join(str(v) for v in row) + "\n")
    meta = {
        "truth_deg": truth, "note": note, "sample_rate": S["audio_rate"], "channels": S["audio_ch"],
        "first_audio_ts_us": S["first_audio_ts_us"], "audio_frames": S["audio_frames"],
        "seconds": S["audio_frames"] / S["audio_rate"], "pose_samples": len(S["pose"]),
        "video_frames": S["video_frames"], "hello": S["hello"], "cue": S["cue"], "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "distance_m": distance_m, "motion": motion, "stimulus": stimulus,
        "audio_chunks": len(S["chunks"]), "clock_restarts": S["clock_restarts"],
        "pose_restarts": S.get("pose_restarts", 0),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {meta['seconds']:.1f} s of audio ({S['audio_ch']} ch @ {S['audio_rate']} Hz), "
          f"{len(S['pose'])} pose samples -> {out}")


async def main_async(a) -> None:
    cfg = uvicorn.Config(app, host="0.0.0.0", port=a.port, log_level="warning")
    server = uvicorn.Server(cfg)
    task = asyncio.create_task(server.serve())
    print(f"recorder on ws://0.0.0.0:{a.port}. Start the headset app (hub_host pointing here). "
          f"{'Auto-stops after %d s of audio.' % a.seconds if a.seconds else 'Ctrl+C to stop.'}")
    try:
        while not task.done():
            await asyncio.sleep(0.5)
            if a.seconds and S["audio_rate"] and S["audio_frames"] / S["audio_rate"] >= a.seconds:
                break
    except asyncio.CancelledError:
        pass
    server.should_exit = True
    await asyncio.sleep(0.5)
    save(Path(a.out), a.truth, a.note, a.distance_m, a.motion, a.stimulus)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", type=float, default=None, help="source angle in degrees from initial facing (+ right)")
    ap.add_argument("--seconds", type=float, default=0, help="auto-stop after this many seconds of audio")
    ap.add_argument("--noise-until", type=float, default=0, help="seconds of ambient-only audio at the start (source off)")
    ap.add_argument("--go-at", type=float, default=0, help="second to start turning the head (2 beeps)")
    ap.add_argument("--stop-at", type=float, default=0, help="second to stop (3 beeps); recording ends 1.5 s later")
    ap.add_argument("--note", default="")
    ap.add_argument("--distance-m", type=float, default=None, help="source distance in meters (coverage grid label)")
    ap.add_argument("--motion", choices=["small", "normal", "large"], default=None, help="head-movement size label")
    ap.add_argument("--stimulus", choices=["pink", "speech"], default=None, help="what the source played")
    ap.add_argument("--port", type=int, default=p.DEFAULT_PORT)
    args = ap.parse_args()
    if args.stop_at:
        S["cue"] = {"noise_until": args.noise_until, "go": args.go_at, "stop": args.stop_at}
        args.seconds = args.stop_at + 1.5
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        save(Path(args.out), args.truth, args.note, args.distance_m, args.motion, args.stimulus)
