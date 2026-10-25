"""quest/tools/record_session.py camera-frame saving, and Session.load / replay of what it writes.

Mocked: synthetic AudioChunk / VideoFrame messages through the recorder's real websocket handlers
(no headset, no camera); the JPEG payloads are opaque bytes, the recorder never decodes them.
"""

import asyncio
import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

import protocol as p
from ingest import IngestState, create_app
from mockclient import Session, active_at, frame_truth, load_talkers
from mockclient.replay import events, replay

spec = importlib.util.spec_from_file_location(
    "record_session", Path(__file__).resolve().parents[2] / "quest" / "tools" / "record_session.py"
)
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)

T0 = 5_000_000  # headset clock at the first audio chunk, us
RATE, CH = 48000, 2


@pytest.fixture(autouse=True)
def fresh_recorder():
    rec.reset_state()
    rec.VIDEO.update(enabled=True, min_interval_us=0, max_frames=0, flipped=False)
    yield
    rec.reset_state()


def jpeg(k: int) -> bytes:  # not decodable; unique per frame so mixups show
    return b"\xff\xd8" + bytes([k % 256]) * (50 + k) + b"\xff\xd9"


def video_msg(ts: int, k: int = 0, seq: int | None = None, **kw) -> bytes:
    kw.setdefault("width", 640); kw.setdefault("height", 480)
    kw.setdefault("cam_pos", (0.01 * k, 1.6, -0.02)); kw.setdefault("cam_rot_xyzw", (0.0, 0.1 * k, 0.0, 0.99))
    kw.setdefault("hfov_deg", 78.5)
    return p.pack(p.VideoFrame(seq=k if seq is None else seq, device_ts_us=ts, epoch=0, **kw), jpeg(k))


def audio_msg(seq: int, ts: int, frames: int = 4800) -> bytes:
    pcm = (np.sin(np.arange(frames * CH) / 20.0) * 8000).astype("<i2").tobytes()
    return p.pack(p.AudioChunk(seq=seq, device_ts_us=ts, epoch=0, sample_rate=RATE, channels=CH, frames=frames), pcm)


def stream(c: TestClient, n_frames: int, fps: int = 10) -> None:
    """1 s of audio (10 x 100 ms) and n_frames camera frames spread over it, with a hello and poses."""
    with c.websocket_connect(p.WS_POSE_IN) as pose, c.websocket_connect(p.WS_AUDIO_IN) as aud, \
            c.websocket_connect(p.WS_VIDEO_IN) as vid:
        pose.send_text(p.Hello(seq=0, device_ts_us=T0, epoch=0, device="quest3s", mic_channels=CH, mic_raw=True,
                               camera=True, camera_hfov_deg=78.5).model_dump_json())
        for k in range(10):
            pose.send_text(p.PoseSample(seq=k + 1, device_ts_us=T0 + k * 100_000, epoch=0, pos=(0, 1.6, 0),
                                        rot_xyzw=(0, 0.01 * k, 0, 1)).model_dump_json())
            aud.send_bytes(audio_msg(k, T0 + k * 100_000))
        for k in range(n_frames):
            vid.send_bytes(video_msg(T0 + k * 1_000_000 // fps, k))
        for _ in range(200):  # websocket handlers run in the TestClient's portal thread
            if rec.S["video_frames"] == n_frames and rec.S["audio_frames"] == 10 * 4800 and len(rec.S["pose"]) == 10:
                break
            time.sleep(0.01)


def test_frames_saved_and_session_round_trips(tmp_path):
    with TestClient(rec.app) as c:
        stream(c, 10)
    rec.save(tmp_path / "take", truth=90.0, note="synthetic")
    take = tmp_path / "take"

    names = sorted(f.name for f in (take / "video").iterdir())
    assert names == sorted([f"{T0 + k * 100_000}.jpg" for k in range(10)] + ["frames.jsonl"])  # replay layout
    rows = [json.loads(l) for l in (take / "video" / "frames.jsonl").read_text().splitlines()]
    assert [r["device_ts_us"] for r in rows] == [T0 + k * 100_000 for k in range(10)]
    assert rows[3]["width"] == 640 and rows[3]["height"] == 480 and rows[3]["hfov_deg"] == 78.5
    assert rows[3]["cam_pos"] == pytest.approx([0.03, 1.6, -0.02]) and rows[3]["cam_rot_xyzw"][1] == pytest.approx(0.3)
    assert (take / "video" / f"{T0 + 300_000}.jpg").read_bytes() == jpeg(3)
    meta = json.loads((take / "meta.json").read_text())
    assert meta["video_frames"] == 10 and meta["video_saved"] == 10 and meta["video_camera_flipped"] is False

    s = Session.load(take)  # a recorder take loads without a session.json
    assert s.start_ts_us == T0 and s.sample_rate == RATE and s.audio.shape == (48000, CH)
    assert s.meta["device"] == "quest3s" and s.meta["camera_hfov_deg"] == 78.5 and s.meta["truth_deg"] == 90.0
    assert len(s.poses) == 10 and s.poses[2]["device_ts_us"] == T0 + 200_000
    assert s.poses[2]["rot_xyzw"] == pytest.approx([0, 0.02, 0, 1])
    assert [ts for ts, _ in s.frames] == [T0 + k * 100_000 for k in range(10)]
    assert s.frames[7][1].read_bytes() == jpeg(7) and s.frame_meta[T0 + 700_000]["cam_pos"][0] == pytest.approx(0.07)
    ev = [e for e in events(s) if e[1] == "video"]
    assert len(ev) == 10 and ev[4][0] == pytest.approx(0.4)  # one video event per frame, on the audio timeline


def test_limits_and_disable(tmp_path):
    rec.VIDEO.update(min_interval_us=200_000, max_frames=3)  # 20 fps offered, ask for 5 fps, at most 3 frames
    with TestClient(rec.app) as c:
        stream(c, 20, fps=20)
    rec.save(tmp_path / "limited", None, "")
    ts = sorted(int(f.stem) for f in (tmp_path / "limited" / "video").glob("*.jpg"))
    assert ts == [T0, T0 + 200_000, T0 + 400_000]
    assert json.loads((tmp_path / "limited" / "meta.json").read_text())["video_frames"] == 20

    rec.reset_state()
    rec.VIDEO.update(enabled=False, min_interval_us=0, max_frames=0)
    with TestClient(rec.app) as c:
        stream(c, 5)
    rec.save(tmp_path / "off", None, "")
    meta = json.loads((tmp_path / "off" / "meta.json").read_text())
    assert not (tmp_path / "off" / "video").exists() and meta["video_frames"] == 5 and meta["video_saved"] == 0
    assert (tmp_path / "off" / "audio.wav").exists()
    assert Session.load(tmp_path / "off").frames == []  # still loadable without video


def test_bad_duplicate_and_restarted_frames():
    assert rec.on_video(b"\x01\x02") is False                       # truncated: no header
    assert rec.on_video(b"\x04\x00\x00\x00nope" + b"xx") is False   # header is not JSON
    assert rec.on_video(video_msg(10_000_000)) is True
    assert rec.on_video(video_msg(10_000_000, k=1, seq=1)) is False  # same timestamp: file name would clash
    assert rec.on_video(video_msg(10_100_000, k=2, seq=2, codec="h264")) is False  # not a per-frame jpeg
    assert rec.on_video(video_msg(10_200_000, k=3, seq=3)) is True and len(rec.S["video"]) == 2
    assert rec.on_video(video_msg(10_000, k=4, seq=0)) is True      # clock restarted from 0: earlier frames dropped
    assert [h.device_ts_us for h, _ in rec.S["video"]] == [10_000] and rec.S["video_frames"] == 7


def test_audio_restart_discards_earlier_frames():
    with TestClient(rec.app) as c:
        stream(c, 5)
        assert len(rec.S["video"]) == 5
        with c.websocket_connect(p.WS_AUDIO_IN) as aud:
            aud.send_bytes(audio_msg(0, 1_000))  # headset clock went back to ~0: new session
            for _ in range(200):
                if rec.S["audio_frames"] == 4800:
                    break
                time.sleep(0.01)
    assert rec.S["video"] == [] and rec.S["audio_frames"] == 4800


@pytest.fixture
def hub():
    state = IngestState()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(state), port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True); th.start()
    while not server.started:
        time.sleep(0.01)
    yield state, f"ws://127.0.0.1:{port}"
    server.should_exit = True; th.join(5)


def test_recorded_take_replays_with_its_frame_headers(tmp_path, hub):
    state, url = hub
    with TestClient(rec.app) as c:
        stream(c, 10)
    rec.save(tmp_path / "take", None, "")
    s = Session.load(tmp_path / "take")
    sent = asyncio.run(replay(s, url, speed=0))
    assert sent["video"] == 10
    for _ in range(200):
        if state.video.received == 10:
            break
        time.sleep(0.02)
    hdr, data = state.video.take()  # the newest frame the hub holds is the last recorded one
    assert data == jpeg(9) and (hdr.width, hdr.height, hdr.hfov_deg) == (640, 480, 78.5)
    assert hdr.cam_pos == pytest.approx((0.09, 1.6, -0.02)) and hdr.cam_rot_xyzw[1] == pytest.approx(0.9)
    assert hdr.device_ts_us == T0 + 900_000


def test_talker_labels(tmp_path):
    with TestClient(rec.app) as c:
        stream(c, 10)
    rec.save(tmp_path / "take", None, "")
    (tmp_path / "take" / "talkers.csv").write_text(
        "# who is talking\nstart_s,end_s,talker_id,bearing_deg,note\n"
        "0.0,0.35,alice,-30,left\n0.30,0.7,bob,40,\n")
    spans = load_talkers(tmp_path / "take")
    assert [(x.talker_id, x.bearing_deg) for x in spans] == [("alice", -30.0), ("bob", 40.0)]
    s = Session.load(tmp_path / "take")
    truth = {(ts - T0) // 1000: [x.talker_id for x in who] for ts, who in frame_truth(s, spans)}
    assert truth == {0: ["alice"], 100: ["alice"], 200: ["alice"], 300: ["alice", "bob"], 400: ["bob"],
                     500: ["bob"], 600: ["bob"], 700: [], 800: [], 900: []}
    assert active_at(spans, 0.35) == [spans[1]]  # end is exclusive
    (tmp_path / "bad.csv").write_text("start_s,end_s,talker_id\n2,1,x\n")
    with pytest.raises(ValueError, match="end_s > start_s"):
        load_talkers(tmp_path / "bad.csv")


def test_resaving_into_an_existing_take_does_not_leave_the_old_frames(tmp_path):
    take = tmp_path / "take"
    with TestClient(rec.app) as c:
        stream(c, 10)
    rec.save(take, truth=None, note="first")
    assert len(list((take / "video").glob("*.jpg"))) == 10
    rec.reset_state()
    rec.VIDEO.update(enabled=False)                      # second take records no video at all
    with TestClient(rec.app) as c:
        stream(c, 10)
    rec.save(take, truth=None, note="second")
    assert not list((take / "video").glob("*.jpg")) and not (take / "video" / "frames.jsonl").exists()
    assert Session.load(take).frames == []
    rec.reset_state()
    rec.VIDEO.update(enabled=True, min_interval_us=0, max_frames=3, flipped=False)   # a limited third take
    with TestClient(rec.app) as c:
        stream(c, 10)
    rec.save(take, truth=None, note="third")
    s = Session.load(take)
    assert len(s.frames) == 3 and set(s.frame_meta) == {ts for ts, _ in s.frames}


def test_fps_throttle_at_the_camera_rate_keeps_jittery_frames():
    rec.VIDEO.update(min_interval_us=int(1e6 / 15))      # asked for the camera's own 15 fps
    kept = 0
    for k in range(60):
        ts = T0 + int(k * 66_667 + (5_000 if k % 2 else -5_000))      # +-5 ms of frame-timing jitter
        kept += rec.on_video(video_msg(ts, k))
    assert kept >= 58                                    # the old delta-from-last-kept rule lost about every other frame
    rec.reset_state()
    rec.VIDEO.update(min_interval_us=200_000)            # 5 fps asked of a 15 fps camera: still about 5 fps
    kept5 = sum(rec.on_video(video_msg(T0 + int(k * 66_667 + (3_000 if k % 2 else -3_000)), k)) for k in range(150))
    assert 48 <= kept5 <= 52


def test_load_survives_a_truncated_frames_file_and_replay_falls_back_for_a_null_hfov(tmp_path):
    with TestClient(rec.app) as c:
        stream(c, 6)
    take = tmp_path / "take"
    rec.save(take, truth=None, note="")
    fj = take / "video" / "frames.jsonl"
    rows = fj.read_text().splitlines()
    nulled = json.loads(rows[0])
    nulled["hfov_deg"] = None
    rows[0] = json.dumps(nulled)
    fj.write_text("\n".join(rows) + '\n{"seq": 7}\n{"device_ts_us": 99, "wi')      # no timestamp; truncated tail
    s = Session.load(take)
    assert len(s.frame_meta) == 6 and s.frame_meta[T0]["hfov_deg"] is None
    assert (s.frame_meta[T0].get("hfov_deg") or s.meta.get("camera_hfov_deg")) == 78.5   # replay's null fallback


def test_talker_csv_bad_rows_are_reported_as_value_errors(tmp_path):
    f = tmp_path / "talkers.csv"
    for body in ("start_s,end_s\n1,2\n", "start_s,end_s,talker_id\n1,nan,a\n", "start_s,end_s,talker_id\n1,inf,a\n"):
        f.write_text(body)
        with pytest.raises(ValueError):
            load_talkers(f)
    f.write_bytes("﻿start_s,end_s,talker_id,bearing_deg,note\n1,2,ann,30,é\n".encode("utf-8"))   # Excel BOM + non-ASCII
    assert load_talkers(f)[0].talker_id == "ann" and load_talkers(f)[0].note == "é"
