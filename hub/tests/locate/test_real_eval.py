"""Smoke test: the real-recording scorer runs end to end on a synthetic take.

It checks the plumbing (WAV + pose.csv + meta.json -> verdict), not real-world accuracy.
"""

import json
import math
import wave

import numpy as np

from locate import real_eval


def _make_take(tmp_path, truth=90.0, source_gain=1.0, cue=None, room=0.0):
    rate, secs = 48000, 25
    rng = np.random.default_rng(1)
    n = rate * secs
    t = np.arange(n) / rate
    yaw = 40 * np.sin(2 * math.pi * t / 10)  # head sweeps +-40 deg
    # crude directional cue: source at `truth` is louder/brighter when the head faces it
    off = np.radians(yaw - truth)
    gain = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(off))
    noise = rng.standard_normal(n) * 0.05
    x = (noise * gain) * source_gain
    if cue:  # source off until noise_until; a steady room noise throughout
        x = np.where(t < cue["noise_until"], 0.0, x) + rng.standard_normal(n) * room
    pcm = np.stack([x, x], axis=1)
    pcm = np.clip(pcm * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(tmp_path / "audio.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    base = 1_000_000
    pt = np.arange(0, secs, 0.02)
    py = 40 * np.sin(2 * math.pi * pt / 10)
    rows = []
    for ti, yi in zip(pt, py):
        h = math.radians(yi) / 2
        rows.append((base + int(ti * 1e6), 0.0, math.sin(h), 0.0, math.cos(h)))
    np.savetxt(tmp_path / "pose.csv", rows, delimiter=",", header="t_us,qx,qy,qz,qw", comments="")
    (tmp_path / "meta.json").write_text(json.dumps({
        "truth_deg": truth, "sample_rate": rate, "channels": 2, "first_audio_ts_us": base,
        "audio_frames": n, "pose_samples": len(rows), "cue": cue}))


def test_scorer_runs_and_returns_verdict(tmp_path):
    _make_take(tmp_path)
    res = real_eval.run(tmp_path, 90.0, 12.0, verbose=False)
    assert res["yaw_std"] > 20
    assert 0.5 < res["active_frac"] <= 1.0
    assert "pass" in res and "world_deg" in res


CUE = {"noise_until": 4.0, "go": 8.0, "stop": 24.0}


def test_loud_source_is_a_valid_take(tmp_path):
    _make_take(tmp_path, cue=CUE, room=0.005)
    res = real_eval.run(tmp_path, 90.0, 12.0, verbose=False)
    assert res["snr_db"] > 10 and res["valid"]


def test_source_barely_above_room_is_invalid_not_a_locator_failure(tmp_path):
    """Live take 2026-10-24: the laptop was ~1.3 dB above the room and the locator 'found' the room."""
    _make_take(tmp_path, cue=CUE, source_gain=0.1, room=0.01)
    res = real_eval.run(tmp_path, 90.0, 12.0, verbose=False)
    assert res["snr_db"] < real_eval.MIN_SNR_DB
    assert not res["valid"] and not res["pass"]
