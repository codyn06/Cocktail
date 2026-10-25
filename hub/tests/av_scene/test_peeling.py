"""Residual peeling tests: fake separators + synthetic noise talkers (mocked; real separator untested)."""

from __future__ import annotations

import numpy as np

from av_scene.peeling import dedup_streams, peel_residual_speech, si_sdr_db

N = 16000 * 3
THIRD_MIN_SI_SDR_DB = 15.0


def _talkers(k: int):
    rng = np.random.default_rng(0)
    return [rng.standard_normal(N) * 0.1 for _ in range(k)]


def _fake(a, b):
    """First call (the mixture) returns only a,b; later calls (residual) return it as one stream."""
    calls = {"n": 0}

    def sep(x):
        calls["n"] += 1
        if calls["n"] == 1:
            return np.stack([a, b])
        return np.stack([x, np.zeros_like(x)])

    sep.calls = calls
    return sep


def test_three_talkers_third_recovered():
    a, b, c = _talkers(3)
    sep = _fake(a, b)
    streams, diag = peel_residual_speech(a + b + c, sep, max_rounds=1)
    assert len(streams) == 3
    assert si_sdr_db(streams[2], c) > THIRD_MIN_SI_SDR_DB
    assert diag["separator_calls"] == 2


def test_two_talkers_no_third():
    a, b = _talkers(2)
    sep = _fake(a, b)
    streams, diag = peel_residual_speech(a + b, sep, max_rounds=3)
    assert len(streams) == 2
    assert diag["stop_reason"] == "residual_below_floor"
    assert sep.calls["n"] == 1


def test_duplicates_dropped():
    a, b = _talkers(2)
    out = dedup_streams([a, a * 0.5, b, a + 0.01 * b], thr_db=10.0)
    assert len(out) == 2


def test_max_rounds_respected():
    a, b, c = _talkers(3)
    calls = {"n": 0}

    def sep(x):  # never converges: always claims a new (noise) voice
        calls["n"] += 1
        if calls["n"] == 1:
            return np.stack([a, b])
        return np.stack([x * 0.5, np.zeros_like(x)])

    mix = a + b + c
    _, d0 = peel_residual_speech(mix, sep, max_rounds=0)
    assert d0["separator_calls"] == 1
    calls["n"] = 0
    _, d2 = peel_residual_speech(mix, sep, max_rounds=2)
    assert d2["separator_calls"] <= 3


def test_first_round_drops_empty_separator_channel():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(16000 * 3).astype(np.float32) * 0.1
    leak = (0.001 * a).astype(np.float32)                    # -40 dB channel: bleed only, not a voice
    out, _ = peel_residual_speech(a, lambda x: np.stack([x * 0 + a, leak]), max_rounds=0)
    assert len(out) == 1
