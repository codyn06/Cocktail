"""pairsim scripted scenes: moving talkers, scripted silences, head-pose gaps (SIMULATED)."""

import json

import numpy as np
from scipy.io import wavfile

from locate import pairsim


def test_scripted_session_metadata_and_audio(tmp_path):
    pairsim.write_session(tmp_path, [-60.0, 30.0], seed=1, secs=6.0,
                          paths=[[[0, -60], [6, 60]], None], silent=[None, [[2.0, 4.0]]], pose_gaps=[[3.0, 4.0]])
    meta = json.loads((tmp_path / "session.json").read_text())
    a, b = meta["sources"]
    assert a["moving"] and a["bearing_deg"] == -60.0 and a["path"] == [[0.0, -60.0], [6.0, 60.0]]
    assert "moving" not in b and b["silent"] == [[2.0, 4.0]] and meta["pose_gaps"] == [[3.0, 4.0]]
    stem = wavfile.read(tmp_path / "stems" / "talker2.wav")[1].astype(float)
    rms = lambda t0, t1: np.sqrt(np.mean(stem[int(t0 * 16000):int(t1 * 16000)] ** 2))
    assert rms(2.7, 3.9) < 0.01 * max(rms(0, 2), rms(4, 6))   # silenced span is silent (after the 0.4 s reverb tail)
    ts = [json.loads(line)["device_ts_us"] for line in (tmp_path / "pose.jsonl").read_text().splitlines()]
    assert not any(1_000_000 + 3_000_000 <= t < 1_000_000 + 4_000_000 for t in ts)  # no poses in the gap
    assert len(ts) > 300


def test_default_session_has_no_scripted_fields(tmp_path):
    pairsim.write_session(tmp_path, [10.0], seed=1, secs=3.0)
    meta = json.loads((tmp_path / "session.json").read_text())
    assert "pose_gaps" not in meta and set(meta["sources"][0]) == {"name", "bearing_deg", "stem"}
