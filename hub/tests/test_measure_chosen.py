"""measure_chosen on synthetic streams with a known target/other ratio (mocked, no models)."""

import numpy as np
import pytest

from separate.measure_chosen import SR, choose_stream, measure, reach_times, window_margins

rng = np.random.default_rng(3)
N = SR * 10
A = rng.standard_normal(N).astype(np.float32) * 0.1  # target voice stem
B = rng.standard_normal(N).astype(np.float32) * 0.1  # other voice
C = rng.standard_normal(N).astype(np.float32) * 0.1  # third voice
REFS = {"a": A, "b": B, "c": C}


def gain(db):
    return 10 ** (db / 20)


def test_known_ratio_is_reported():
    for db in (3.0, 10.0, 17.0):
        m = window_margins(A + gain(-db) * B, REFS, "a", t0_s=1.0)
        assert np.nanmedian(m.margin_db) == pytest.approx(db, abs=0.5)
        assert m.active.all()


def test_others_are_summed_power():
    x = A + gain(-12) * B + gain(-12) * C  # two equal interferers: 3 dB worse than one
    m = window_margins(x, REFS, "a", t0_s=0.0)
    assert np.nanmedian(m.margin_db) == pytest.approx(12 - 3.0, abs=0.5)


def test_scale_of_stream_does_not_matter_and_target_can_be_any_stem():
    x = 5.0 * (B + gain(-14) * A)
    assert np.nanmedian(window_margins(x, REFS, "b").margin_db) == pytest.approx(14, abs=0.5)
    assert np.nanmedian(window_margins(x, REFS, "a").margin_db) == pytest.approx(-14, abs=0.5)


def test_unexplained_sound_is_not_counted_as_other_voice():
    noise = rng.standard_normal(N).astype(np.float32) * 0.1
    x = A + gain(-20) * B + gain(-6) * noise
    m = window_margins(x, REFS, "a")
    assert np.nanmedian(m.margin_db) == pytest.approx(20, abs=0.5)
    assert np.nanmedian(m.unexplained_db) == pytest.approx(-6, abs=0.7)


def test_time_to_threshold_after_selection():
    # 0 dB until 4 s, then 15 dB.
    g = np.where(np.arange(N) < 4 * SR, 1.0, gain(-15)).astype(np.float32)
    x = A + g * B
    r = measure({"s1": x}, x, REFS, "a", t0_s=3.0, threshold_db=10.0)
    assert r["first_reach_s"] == pytest.approx(1.25)  # end of the first window after the 1.0 s mark
    assert r["stable_s"] == pytest.approx(1.25)
    assert r["met_within_budget"] is False  # budget 1.0 s
    assert r["steady_margin_db"] == pytest.approx(15, abs=0.5)
    r = measure({"s1": x}, x, REFS, "a", t0_s=4.0, threshold_db=10.0)
    assert r["stable_s"] == pytest.approx(0.25) and r["met_within_budget"] is True


def test_breakthrough_later_delays_stable_but_not_first_reach():
    g = np.full(N, gain(-15), np.float32)
    g[6 * SR : 6 * SR + SR // 2] = 1.0  # a 0.5 s breakthrough at t = 6 s
    x = A + g * B
    r = measure({"s1": x}, x, REFS, "a", t0_s=4.0)
    assert r["first_reach_s"] == pytest.approx(0.25)
    assert r["stable_s"] == pytest.approx(2.75)  # first clean window ends at 6.75 s; t0 = 4 s
    assert r["met_within_budget"] is False


def test_never_reaching_is_reported_as_none():
    x = A + gain(-4) * B
    r = measure({"s1": x}, x, REFS, "a", t0_s=2.0, threshold_db=10.0)
    assert r["first_reach_s"] is None and r["stable_s"] is None and r["met_within_budget"] is False
    assert r["steady_margin_db"] == pytest.approx(4, abs=0.5)


def test_target_pauses_are_skipped_not_failed():
    half = SR // 2
    gate = np.repeat(np.arange(N // half) % 2 == 0, half).astype(np.float32)  # on/off every 0.5 s
    ref = {"a": A * gate, "b": B}
    m = window_margins(A * gate + gain(-15) * B, ref, "a", t0_s=0.0)
    assert 0.4 < m.active.mean() < 0.6
    assert np.isnan(m.margin_db[~m.active]).all()
    assert np.nanmin(m.margin_db) > 12


def test_chooses_stream_carrying_the_target():
    est = {"s1": B + gain(-20) * A, "s2": A + gain(-20) * B, "s3": C}
    assert choose_stream(est, A, 2.0) == "s2"
    r = measure(est, A + B + C, REFS, "a", t0_s=2.0)
    assert r["stream"] == "s2" and r["steady_margin_db"] == pytest.approx(20, abs=0.5)
    assert r["mix_margin_db"] == pytest.approx(-3.0, abs=0.7)  # a against b + c in the raw mix


def test_reports_when_the_voice_moves_to_another_stream():
    # After 6 s the voice a swaps from s1 to s2: the chosen stream (s1) then carries b instead.
    sw = np.arange(N) >= 6 * SR
    est = {"s1": np.where(sw, B, A + gain(-20) * B), "s2": np.where(sw, A, B)}
    r = measure(est, A + B, {"a": A, "b": B}, "a", t0_s=4.0, stream="s1")
    assert r["carried_by"] == {"s1": pytest.approx(0.333, abs=0.01), "s2": pytest.approx(0.667, abs=0.01)}
    assert r["stable_s"] is None and r["steady_min_db"] < -10


def test_no_active_windows_never_passes():
    m = window_margins(B, {"a": np.zeros(N, np.float32), "b": B}, "a")
    assert not m.active.any()
    assert reach_times(m, 10.0)["stable_s"] is None
