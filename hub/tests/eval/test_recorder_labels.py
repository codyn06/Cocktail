"""The recorder keeps both the camera-frame recording (PR #44) and the take labels / chunk timestamps.

Drives the real /ingest/audio and /ingest/video handlers with fake data (evidence: mocked, no headset).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "quest" / "tools"))
import record_session as rec  # noqa: E402
import protocol as p  # noqa: E402


def test_meta_has_video_keys_and_label_keys_and_chunks_csv(tmp_path):
    rec.reset_state()
    rec.VIDEO.update(enabled=True, min_interval_us=0, max_frames=0, flipped=False)
    c = TestClient(rec.app)
    pcm = np.zeros((960, 2), dtype="<i2").tobytes()
    with c.websocket_connect(p.WS_AUDIO_IN) as ws:
        for i in range(4):
            ws.send_bytes(p.pack(p.AudioChunk(seq=i, device_ts_us=5_000_000 + i * 20000, epoch=0, sample_rate=48000,
                                              channels=2, frames=960), pcm))
    with c.websocket_connect(p.WS_VIDEO_IN) as ws:
        ws.send_bytes(p.pack(p.VideoFrame(seq=0, device_ts_us=5_010_000, epoch=0, width=2, height=2), b"\xff\xd8jpeg"))
    import time
    time.sleep(0.3)
    out = tmp_path / "take"
    rec.save(out, 90.0, "n", 1.0, "large", "speech")
    meta = json.loads((out / "meta.json").read_text())
    # #44's keys and the labels / clock keys, together
    for k in ("video_frames", "video_saved", "video_enabled", "video_camera_flipped"):
        assert k in meta, k
    assert meta["video_saved"] == 1 and (out / "video" / "5010000.jpg").exists()
    assert (meta["distance_m"], meta["motion"], meta["stimulus"]) == (1.0, "large", "speech")
    assert (meta["audio_chunks"], meta["clock_restarts"], meta["pose_restarts"]) == (4, 0, 0)
    rows = list(csv.reader((out / "audio_chunks.csv").open()))
    assert rows[0] == ["seq", "device_ts_us", "frames", "epoch"] and len(rows) == 5
    assert (out / "audio.wav").exists() and (out / "pose.csv").exists()


def test_clock_restart_clears_chunks_and_counts(tmp_path):
    rec.reset_state()
    c = TestClient(rec.app)
    pcm = np.zeros((960, 2), dtype="<i2").tobytes()

    def chunk(i, ts):
        return p.pack(p.AudioChunk(seq=i, device_ts_us=ts, epoch=0, sample_rate=48000, channels=2, frames=960), pcm)
    with c.websocket_connect(p.WS_AUDIO_IN) as ws:
        ws.send_bytes(chunk(0, 20_000_000))
        ws.send_bytes(chunk(1, 20_020_000))
        ws.send_bytes(chunk(0, 1_000_000))   # the headset app restarted: its clock starts again
        ws.send_bytes(chunk(1, 1_020_000))
    import time
    time.sleep(0.3)
    rec.save(tmp_path / "t", None, "")
    meta = json.loads((tmp_path / "t" / "meta.json").read_text())
    assert meta["clock_restarts"] == 1 and meta["audio_chunks"] == 2 and meta["first_audio_ts_us"] == 1_000_000
    rec.reset_state()
