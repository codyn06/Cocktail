"""Tracker smoothing and the jitter metrics (track.jitter), on scripted locator output (simulated)."""

import numpy as np

from track import jitter
from track.scenarios import jit, raw, simulate
from track.tracker import TrackConfig, Tracker


def _hops(bearing_of, seconds=40, conf=0.7):
    """Locator output as locate.scenes --raw-json writes it: one snapshot per 1 s hop."""
    return [{"t_us": int((jitter.START_S + k) * 1e6),
             "sources": [raw("s1", "speech", bearing_of(k), conf=conf, us=int(k * 1e6)).model_dump(mode="json")]}
            for k in range(1, seconds)]


def test_a_steady_locator_gives_a_steady_arrow():
    ticks = jitter.replay(_hops(lambda k: 30.0))
    m = jitter.stationary(ticks, {"s1": 30.0})["s1"]
    assert m["wander"] < 0.05 and m["sd"] < 0.1 and m["err_med"] < 0.5


def test_a_20_degree_move_is_followed_within_a_second_without_overshoot():
    hops = _hops(lambda k: 30.0)
    r = jitter.step_response(hops, {"s1": 30.0})["s1"]
    assert r["t80"] is not None and r["t80"] <= 1.2, r
    assert r["overshoot"] < 1.0


def test_walking_source_lags_by_less_than_a_second():
    r = jitter.ramp_response(_hops(lambda k: 30.0), {"s1": 30.0})["s1"]
    assert 0.0 <= r["lag_s"] < 1.0, r


def test_white_noise_on_every_reading_is_smoothed_more_than_before():
    # before this change the per-tick gain (0.3 * (0.5 + conf)) left ~2.8 deg of spread on 6 deg noise
    assert jitter.noisy_stationary(TrackConfig())["sd"] < 2.3
    slow = jitter.noisy_stationary(TrackConfig(smooth_tau_s=2.0))["sd"]
    assert slow < jitter.noisy_stationary(TrackConfig(smooth_tau_s=0.3))["sd"]


def test_smoothing_does_not_depend_on_the_update_rate():
    def settled(dt_us):
        tr, rng = Tracker(), np.random.default_rng(0)
        out = []
        for k in range(int(30e6 // dt_us)):
            s = tr.update(k * dt_us, [raw("s1", "speech", jit(rng, 40.0, 6.0))])
            out.append(s[0].bearing_deg if s else None)
        b = np.array([x for x in out[len(out) // 3:] if x is not None])
        return b.std()
    assert abs(settled(50_000) - settled(100_000)) < 0.7  # twice the rate: similar steadiness, not half the smoothing


def test_a_real_move_beyond_the_gate_is_not_smoothed_into_a_crawl():
    tr = Tracker()
    for k in range(30):
        tr.update(k * 100_000, [raw("s1", "speech", 0.0)])
    for k in range(30, 40):  # the source jumps 25 deg: inside the 30 deg gate, outside the noise band
        out = tr.update(k * 100_000, [raw("s1", "speech", 25.0)])
    assert out[0].bearing_deg > 20.0  # 1 s later it is nearly there
