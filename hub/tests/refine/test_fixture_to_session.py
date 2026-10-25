"""fixture_to_session on a tiny synthetic mp4 (1 s, 25 fps colour frames + 440 Hz sine)."""

import numpy as np
import pytest

from mockclient.session import Session
from refine.tools import fixture_to_session as F


def _make_mp4(path, fps=25, secs=1.0, sr=48000):
    av = pytest.importorskip("av")
    try:
        with av.open(str(path), "w") as c:
            v = c.add_stream("mpeg4", rate=fps)
            v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
            a = c.add_stream("aac", rate=sr)
            for i in range(int(fps * secs)):
                img = np.full((48, 64, 3), (i * 9 % 256, 80, 200), np.uint8)
                f = av.VideoFrame.from_ndarray(img, format="rgb24")
                f.pts = i
                f.time_base = v.codec_context.time_base
                c.mux(v.encode(f))
            c.mux(v.encode())
            n = int(sr * secs)
            x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / sr) * 32767).astype(np.int16)
            for s in range(0, n - 1023, 1024):
                af = av.AudioFrame.from_ndarray(x[None, s:s + 1024], format="s16p", layout="mono")
                af.sample_rate, af.pts = sr, s
                c.mux(a.encode(af))
            c.mux(a.encode())
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no mpeg4/aac encoder available: {e}")


def test_mp4_to_session(tmp_path):
    mp4 = tmp_path / "t.mp4"
    _make_mp4(mp4)
    out = tmp_path / "s"
    r = F.convert_mp4(mp4, out, fps=25.0)
    s = Session.load(out)
    assert s.sample_rate == 16000 and s.audio.shape[1] == 1
    assert s.meta["device"] == "mock" and s.start_ts_us == 1_000_000 and "source" in s.meta
    assert abs(len(s.audio) / 16000 - 1.0) < 0.1
    ts = [t for t, _ in s.frames]
    assert len(ts) == r["frames"] and 23 <= len(ts) <= 26
    d = np.diff(ts)
    assert (d > 0).all() and abs(d.mean() - 40_000) < 200
    assert ts[0] == 1_000_000 and set(ts) == set(s.frame_meta)
    assert s.frame_meta[ts[0]]["hfov_deg"] == 90.0 and s.frame_meta[ts[0]]["cam_pos"] is None
    pts = [p["device_ts_us"] for p in s.poses]
    assert abs(np.diff(pts).mean() - 100_000) < 1 and s.poses[0]["rot_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert pts[-1] - pts[0] >= 0.85e6


def test_seconds_cut(tmp_path):
    mp4 = tmp_path / "t.mp4"
    _make_mp4(mp4)
    r = F.convert_mp4(mp4, tmp_path / "s", fps=10.0, seconds=0.5)
    assert 4 <= r["frames"] <= 6 and r["duration_s"] <= 0.55
