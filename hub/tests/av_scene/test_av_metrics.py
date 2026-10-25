"""Unit tests for eval/av_scene/metrics.py (pure numpy): delay compensation and silent references."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("t_av_metrics", REPO / "eval" / "av_scene" / "metrics.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

SR = 16000


def _speechlike(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * np.arange(n) / SR) ** 2
    return (0.1 * x * env).astype(np.float32)


def test_si_sdr_scale_invariant_and_noise_sensitive():
    ref = _speechlike(SR, 0)
    assert M.si_sdr(0.3 * ref, ref) > 60
    noisy = ref + 0.1 * _speechlike(SR, 1)
    assert 5 < M.si_sdr(noisy, ref) < 30


@pytest.mark.parametrize("delay", [0, 37, -52, 400])
def test_delay_compensation_recovers_delayed_estimate(delay):
    ref = _speechlike(2 * SR, 2)
    est = np.roll(ref, delay)
    est[:delay] = 0 if delay > 0 else est[:delay]
    if delay < 0:
        est[delay:] = 0
    naive = M.si_sdr(est, ref)
    comp, lag = M.si_sdr_delay(est, ref, SR, max_lag_s=0.05)
    assert lag == delay
    assert comp > 40
    if delay != 0:
        assert naive < comp - 20


def test_delay_search_is_bounded():
    ref = _speechlike(2 * SR, 3)
    est = np.roll(ref, 2000)  # 125 ms, outside the 50 ms search window
    _, lag = M.si_sdr_delay(est, ref, SR, max_lag_s=0.05)
    assert abs(lag) <= int(0.05 * SR)


def test_si_sdri_vs_mixture():
    a, b = _speechlike(2 * SR, 4), _speechlike(2 * SR, 5)
    mix = a + b
    good = M.si_sdri(a + 0.05 * b, a, mix, SR)
    assert good["si_sdri"] > 10
    useless = M.si_sdri(mix, a, mix, SR)
    assert abs(useless["si_sdri"]) < 1e-6


def test_si_sdri_returns_none_for_silent_reference():
    mix = _speechlike(SR, 6)
    silent = np.zeros(SR, np.float32)
    assert M.is_silent(silent)
    assert M.si_sdri(mix, silent, mix, SR) is None
    assert M.si_sdri(mix, silent + 1e-6, mix, SR) is None  # -120 dBFS: still silent
    assert M.selection_correct(mix, silent, mix, SR) is None


def test_output_dbfs_for_absent_target():
    assert M.output_dbfs(np.zeros(SR)) == pytest.approx(-180.0)  # floor of the 1e-9 clamp
    assert M.output_dbfs(0.1 * np.ones(SR)) == pytest.approx(-20.0, abs=1e-6)


def test_selection_correct_picks_the_prompted_source():
    a, b = _speechlike(2 * SR, 7), _speechlike(2 * SR, 8)
    ok = M.selection_correct(a + 0.1 * b, a, b, SR)
    assert ok["correct"] and ok["margin_db"] > 10
    wrong = M.selection_correct(b + 0.1 * a, a, b, SR)
    assert not wrong["correct"] and wrong["margin_db"] < -10


def test_label_accuracy():
    assert M.label_accuracy(["speech", "music"], ["speech", "speech"]) == 0.5
    assert M.label_accuracy([], []) is None


def test_active_spans_and_overlap():
    x = np.zeros(4 * SR, np.float32)
    x[int(0.5 * SR):int(1.5 * SR)] = _speechlike(SR, 9)
    x[int(2.5 * SR):int(3.0 * SR)] = _speechlike(SR // 2, 10)
    spans = M.active_spans(x, SR)
    assert len(spans) == 2
    assert spans[0][0] == pytest.approx(0.5, abs=0.06) and spans[0][1] == pytest.approx(1.5, abs=0.06)
    assert M.spans_overlap_s([(0.0, 1.0)], [(0.5, 2.0)]) == pytest.approx(0.5)
    assert M.active_spans(np.zeros(SR, np.float32), SR) == []
