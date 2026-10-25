import asyncio
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn
from scipy import signal
from scipy.io import wavfile

import protocol as p
from ingest import AudioRing, IngestState, LatestFrame, PoseBuffer, Resampler, create_app, yaw_pitch
from mockclient import Session
from mockclient.replay import replay
from mockclient.simulate import quat_yaw_pitch, simulate


def test_resampler_chunked_matches_whole():
    x = np.random.default_rng(0).standard_normal((48000, 2)).astype(np.float32)
    whole = Resampler(48000, 2)(x)
    r = Resampler(48000, 2)
    chunked = np.concatenate([r(x[i : i + 961]) for i in range(0, len(x), 961)])  # odd chunk size
    assert whole.shape == (16000, 2) and np.allclose(whole, chunked, atol=1e-5)


def test_resampler_keeps_a_tone():
    t = np.arange(48000) / 48000
    y = Resampler(48000, 1)(np.sin(2 * np.pi * 1000 * t)[:, None].astype(np.float32))[:, 0]
    f, pxx = signal.periodogram(y[2000:], fs=16000)
    assert abs(f[np.argmax(pxx)] - 1000) < 5


def test_audio_ring_keeps_last_seconds_and_times():
    ring = AudioRing(channels=1, seconds=1.0)
    for k in range(30):  # 3 s of 100 ms chunks, value = chunk index
        ring.write(np.full((1600, 1), k, np.float32), device_ts_us=1_000_000 + k * 100_000)
    x, ts0 = ring.last(10.0)
    assert len(x) == 16000 and x[0, 0] == 20 and x[-1, 0] == 29
    assert ts0 == 1_000_000 + 20 * 100_000
    big = AudioRing(channels=1, seconds=1.0)
    big.write(np.arange(20000, dtype=np.float32)[:, None], 0)  # single write longer than the ring
    assert big.last(1.0)[0][0, 0] == 4000 and big.last(1.0)[1] == 4000 * 1_000_000 // 16000


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (30, 0), (-120, 10), (170, -20)])
def test_quaternion_yaw_pitch_roundtrip(yaw, pitch):
    y, pt = yaw_pitch(quat_yaw_pitch(yaw, pitch))
    assert abs((y - yaw + 180) % 360 - 180) < 1e-6 and abs(pt - pitch) < 1e-6


def test_pose_buffer_window_and_interp():
    b = PoseBuffer(seconds=1.0)
    for k in range(30):
        b.add(k * 100_000, (0, 0, 0), quat_yaw_pitch(170 + k, 0))  # crosses +180
    assert len(b) == 11  # last 1 s inclusive
    yaw, _ = b.at(2_850_000)
    assert abs((yaw - (170 + 28.5) + 180) % 360 - 180) < 1e-6


def test_latest_frame_drops_old():
    v = LatestFrame()
    mk = lambda ts: p.VideoFrame(seq=0, device_ts_us=ts, epoch=0, width=1, height=1)
    v.put(mk(1), b"a"); v.put(mk(2), b"b"); v.put(mk(3), b"c")
    assert v.take()[1] == b"c" and v.dropped == 2
    v.put(mk(2), b"late")
    assert v.take()[1] == b"c" and v.dropped == 3


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


def test_simulated_session_replays_into_hub(hub, tmp_path):
    state, url = hub
    t = np.arange(16000 * 12) / 16000
    for name, f in [("a", 300), ("b", 700)]:
        wavfile.write(tmp_path / f"{name}.wav", 16000, (0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32))
    simulate(tmp_path / "sess", [f"{tmp_path / 'a.wav'}:-60", f"{tmp_path / 'b.wav'}:60"])
    s = Session.load(tmp_path / "sess")
    assert s.audio.shape == (48000 * 12, 2) and len(s.poses) == 600

    sent = asyncio.run(replay(s, url, speed=0, fake_video_fps=15))
    for _ in range(200):  # let the server drain its sockets
        if state.stats["audio"].messages == sent["audio"] and len(state.inputs) == 0 and \
                state.stats["pose"].messages == sent["pose"] + 1 and state.video.received == sent["video"]:
            break
        time.sleep(0.02)
    st = state.status()
    assert st["device"] == "mock" and st["audio_channels"] == 2
    assert st["audio_seconds_held"] == 10.0                  # 12 s sent, last 10 s kept
    assert st["poses_held"] == 501                            # last 10 s of 50 Hz poses
    assert st["video_received"] == 180 and st["video_dropped"] == 179  # nobody consumed frames
    assert all(c["seq_gaps"] == 0 for c in st["channels"].values())
    audio, ts0 = state.audio.last(10.0)
    assert ts0 == s.start_ts_us + 2_000_000 - 750  # minus the 12-sample resampler lag
    assert np.abs(audio).max() > 0.01


@pytest.mark.parametrize("bearing", [-60, 60])
def test_render_head_shadow_and_itd(bearing):
    from mockclient.simulate import render
    x = np.random.default_rng(1).standard_normal(48000).astype(np.float32)
    y = render([x], [bearing], 48000, np.zeros(48000))
    left, right = y[:, 0], y[:, 1]
    ild = 20 * np.log10(np.std(right) / np.std(left))
    lag = np.argmax(signal.correlate(left, right, "full")) - (len(right) - 1)  # >0: left hears it later
    assert np.sign(ild) == np.sign(bearing) and abs(ild) > 2
    assert np.sign(lag) == np.sign(bearing) and 10 < abs(lag) < 40  # ~0.4 ms ITD at 48 kHz


def _chunk(state, seq, ts, epoch=0, value=0.1, frames=320):
    h = p.AudioChunk(seq=seq, device_ts_us=ts, epoch=epoch, sample_rate=16000, channels=1, frames=frames)
    state.on_audio(h, (np.full(frames, value * 32767)).astype("<i2").tobytes())


def test_audio_gap_is_zero_filled_and_epoch_resets():
    st = IngestState()
    _chunk(st, 0, 0); _chunk(st, 1, 20_000)
    _chunk(st, 3, 60_000)                       # chunk 2 lost: 20 ms gap
    x, ts0 = st.audio.last(10)
    assert len(x) == 4 * 320 and ts0 == 0
    assert np.all(x[640:960] == 0) and x[-1, 0] > 0.09   # gap is silence, timing preserved
    assert st.audio.ts_of(960) == 60_000
    _chunk(st, 0, 5_000, epoch=1)               # reconnect: clock restarted
    x, ts0 = st.audio.last(10)
    assert len(x) == 320 and ts0 == 5_000 and st.audio.contiguous_since_us == 5_000


def test_pose_out_of_order_and_no_silent_extrapolation():
    st = IngestState()
    pose = lambda seq, ts, ep=0: st.on_pose_channel(
        p.PoseSample(seq=seq, device_ts_us=ts, epoch=ep, pos=(0, 0, 0), rot_xyzw=quat_yaw_pitch(10, 0)))
    pose(0, 100_000); pose(1, 200_000); pose(2, 150_000)
    assert len(st.poses) == 2 and st.poses.out_of_order == 1
    assert st.poses.at(160_000) is not None and st.poses.at(900_000) is None
    pose(0, 1_000, ep=1)
    assert len(st.poses) == 1


def test_bad_message_does_not_kill_stream(hub):
    state, url = hub
    import websockets

    async def go():
        async with websockets.connect(url + p.WS_AUDIO_IN) as ws:
            h = p.AudioChunk(seq=0, device_ts_us=0, epoch=0, sample_rate=16000, channels=1, frames=320)
            await ws.send(p.pack(h, b"\x00" * 10))       # payload too short
            await ws.send(p.pack(h.model_copy(update={"seq": 1}), b"\x00" * 640))
            await asyncio.sleep(0.2)
    asyncio.run(go())
    assert state.bad_messages == 1 and state.stats["audio"].messages == 2
    assert state.audio.seconds_held == 0.02


# ---- read_png filters (wrap-around) and invalid filter
def _png_with_filters(rows, w):
    import struct
    import zlib

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    raw = b"".join(bytes([f]) + bytes(r) for f, r in rows)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, len(rows), 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def test_read_png_all_filters_wrap_around(tmp_path):
    import numpy as np
    from ingest.media_io import read_png
    w, bpp = 3, 3
    rng = np.random.default_rng(1)
    want = rng.integers(0, 256, (5, w * bpp), dtype=np.uint8)
    want[0] = 250
    rows = []
    for y, f in enumerate([1, 2, 3, 4, 0]):
        prev = want[y - 1].astype(int) if y else np.zeros(w * bpp, int)
        cur = want[y].astype(int)
        enc = []
        for x in range(w * bpp):
            a = cur[x - bpp] if x >= bpp else 0
            c = prev[x - bpp] if x >= bpp else 0
            pred = {0: 0, 1: a, 2: prev[x], 3: (a + prev[x]) // 2, 4: _paeth(a, prev[x], c)}[f]
            enc.append(int((cur[x] - pred) & 0xFF))
        rows.append((f, enc))
    p = tmp_path / "f.png"
    p.write_bytes(_png_with_filters(rows, w))
    assert np.array_equal(read_png(p).reshape(5, -1), want)


def test_read_png_rejects_filter_above_4(tmp_path):
    import pytest
    from ingest.media_io import read_png
    p = tmp_path / "bad.png"
    p.write_bytes(_png_with_filters([(5, [1] * 6)], 2))
    with pytest.raises(ValueError, match="filter 5"):
        read_png(p)
