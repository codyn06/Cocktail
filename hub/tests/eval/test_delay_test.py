"""Tests for eval/delay_test.py (round-trip delay test).

Lives under hub/tests because that is where pytest runs (hub/pyproject.toml testpaths =
["tests"]). Evidence label: MOCKED. The end-to-end test runs the real hub sockets against an
in-process mock headset with a known simulated delay; nothing here touches a Quest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
import delay_test as dt  # noqa: E402
import protocol as p  # noqa: E402

RATE = 48000


def test_find_click_locates_the_chirp_exactly():
    rng = np.random.default_rng(0)
    tpl = dt.click_template(RATE)
    x = rng.standard_normal(RATE // 2) * 0.02
    x[12345:12345 + len(tpl)] += tpl * 0.2
    idx, ncc = dt.find_click(x, tpl)
    assert idx == 12345 and ncc > 0.7


def test_find_click_says_not_heard_for_noise_only():
    rng = np.random.default_rng(1)
    assert dt.find_click(rng.standard_normal(RATE) * 0.05, dt.click_template(RATE)) is None


def test_find_click_handles_short_input():
    assert dt.find_click(np.zeros(10), dt.click_template(RATE)) is None


def test_summary_passes_with_30_good_trials():
    s = dt.summarize([100.0] * 29 + [200.0], 30)
    assert s["pass"] and s["median_ms"] == 100.0 and s["worst_ms"] == 200.0 and s["n_missed"] == 0
    assert s["p90_ms"] == pytest.approx(100.0)


def test_summary_fails_with_fewer_than_30_valid_trials():
    s = dt.summarize([100.0] * 29, 40)
    assert not s["pass"] and s["n_missed"] == 11 and "only 29 valid" in s["fail_reasons"][0]


def test_summary_fails_on_median_and_on_worst_case_separately():
    slow = dt.summarize([180.0] * 30, 30)
    assert not slow["pass"] and any("median" in r for r in slow["fail_reasons"])
    spike = dt.summarize([100.0] * 29 + [450.0], 30)
    assert not spike["pass"] and [r for r in spike["fail_reasons"] if "worst" in r] and not any("median" in r for r in spike["fail_reasons"])
    assert dt.summarize([180.0] * 30, 30, budget_median_ms=200.0)["pass"]


def test_summary_with_nothing_heard():
    s = dt.summarize([], 30)
    assert not s["pass"] and "median_ms" not in s


def _chunk(seq: int, x: np.ndarray) -> bytes:
    pcm = (np.stack([x, x], axis=1) * 32767).astype("<i2")
    hdr = p.AudioChunk(seq=seq, device_ts_us=seq * 20000, epoch=0, sample_rate=RATE, channels=2, frames=len(x))
    return p.pack(hdr, pcm.tobytes())


def test_timing_uses_chunk_arrival_minus_offset_from_chunk_end():
    """Chirp starting at sample 100 of a 960-sample chunk that arrived at t=10.000 s, sent at t=9.850 s."""
    hub = dt.DelayHub()
    tpl = dt.click_template(RATE)
    quiet = lambda: np.random.default_rng(3).standard_normal(960) * 0.001  # noqa: E731
    hub._on_audio(9.980, _chunk(0, quiet()))
    x = quiet()
    x[100:100 + len(tpl)] += tpl * 0.4
    hub._on_audio(10.000, _chunk(1, x))
    hub._on_audio(10.020, _chunk(2, quiet()))
    delay, ncc = hub.analyze(9.850)
    # the sample 100 of the chunk is 859 samples before the chunk's last one
    assert delay == pytest.approx(10.000 - 859 / RATE - 9.850, abs=1e-6)
    assert ncc > 0.9


def test_chirp_split_over_two_chunks_is_still_found():
    hub = dt.DelayHub()
    tpl = dt.click_template(RATE)
    full = np.zeros(1920)
    full[900:900 + len(tpl)] = tpl * 0.4
    hub._on_audio(5.000, _chunk(0, full[:960]))
    hub._on_audio(5.020, _chunk(1, full[960:]))
    delay, _ = hub.analyze(4.9)
    # onset is sample 900 of chunk 0, which arrived at 5.000
    assert delay == pytest.approx(5.000 - 59 / RATE - 4.9, abs=1e-6)


def test_chunks_that_arrived_before_the_send_are_ignored():
    hub = dt.DelayHub()
    tpl = dt.click_template(RATE)
    x = np.zeros(960)
    x[:len(tpl)] = tpl * 0.4
    hub._on_audio(1.0, _chunk(0, x))
    assert hub.analyze(2.0) is None


def _args(*extra: str):
    return ["--mock", "--timeout-s", "0.5", "--gap-min-s", "0.2", "--gap-max-s", "0.3", "--quiet", *extra]


def test_mock_headset_end_to_end_recovers_the_simulated_delay(tmp_path, capsys):
    out = tmp_path / "delay.json"
    rc = dt.main(_args("--trials", "6", "--json", str(out)))
    text = capsys.readouterr().out
    s = json.loads(out.read_text())
    assert s["mode"] == "mock" and s["n_valid"] == 6 and s["n_missed"] == 0
    # simulated delay: 15 down + 3 x 20 prefill + 30 device + 15 up + ~2 mean jitter = 122 ms
    assert s["simulated_expected_ms"] == pytest.approx(122.0)
    assert abs(s["median_ms"] - s["simulated_expected_ms"]) < 20, s
    assert s["worst_ms"] < 200
    assert rc == 1 and "RESULT: FAIL (only 6 valid trials" in text  # < 30 trials can never pass
    assert "MOCKED" in text and "assumed, not a team target" in text


def test_mock_headset_changes_in_the_simulated_path_show_up_in_the_number(tmp_path, capsys):
    out = tmp_path / "slow.json"
    dt.main(_args("--trials", "4", "--sim-device-ms", "130", "--json", str(out)))
    s = json.loads(out.read_text())
    assert abs(s["median_ms"] - 222.0) < 20, s


def test_mic_that_cannot_hear_the_speaker_gives_not_heard_and_fail(tmp_path, capsys):
    out = tmp_path / "deaf.json"
    rc = dt.main(_args("--trials", "3", "--sim-coupling", "0", "--json", str(out)))
    text = capsys.readouterr().out
    s = json.loads(out.read_text())
    assert rc == 1 and s["n_valid"] == 0 and s["pass"] is False
    assert "no trial was heard" in text
