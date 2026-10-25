"""PairLocator: moving-talker forward filter (drift_dps) and the calibration sanity gate.

MOCKED delays (calibration model + noise), not audio.
"""
import numpy as np

from locate.pair import PairCalibration, PairConfig, PairLocator

CAL = PairCalibration(delay_amp_s=15e-6, axis_deg=-110.0, offset_s=-1e-6, noise_s=3e-6)


def _run(cfg, truth_at, T=40.0, rate=20.0, seed=0, offset_extra=0.0, amp=45.0):
    rng = np.random.default_rng(seed)
    loc = PairLocator(CAL, cfg)
    errs = []
    for i in range(int(T * rate)):
        t = i / rate
        yaw = amp * np.sin(2 * np.pi * 0.12 * t) + 20 * np.sin(2 * np.pi * 0.31 * t + 1)
        loc.add_head_pose(t, yaw)
        loc.add_delay("s", t, CAL.predict(truth_at(t), yaw) + offset_extra + rng.normal(0, 3e-6))
        if t >= 15 and i % 20 == 0:
            e = loc.estimate("s", t)
            errs.append(abs((e.world_deg - truth_at(t) + 180) % 360 - 180))
    return np.array(errs), loc.estimate("s")


def test_drift_filter_follows_a_walking_talker():
    walk = lambda t: -60.0 + 6.0 * t            # 6 deg/s sweep across 240 deg in 40 s
    still, _ = _run(PairConfig(tau_s=20.0), walk)
    drift, _ = _run(PairConfig(tau_s=60.0, drift_dps=8.0), walk)
    print(f"[mocked] walking talker, err t>=15s: static filter median {np.median(still):.0f} deg, "
          f"drift filter median {np.median(drift):.0f} deg")
    assert np.median(drift) < 20
    assert np.median(drift) < 0.6 * np.median(still)


def test_drift_filter_keeps_a_static_talker_accurate():
    stay = lambda t: 70.0
    drift, e = _run(PairConfig(tau_s=60.0, drift_dps=8.0), stay)
    base, _ = _run(PairConfig(), stay)
    print(f"[mocked] static talker: drift median {np.median(drift):.1f} deg, default {np.median(base):.1f} deg")
    assert np.median(drift) < 12
    assert e.state in ("tracking", "locked", "ambiguous")


def test_drift_zero_is_unchanged():
    stay = lambda t: 70.0
    a, _ = _run(PairConfig(drift_dps=0.0), stay)
    b, _ = _run(PairConfig(), stay)
    assert np.array_equal(a, b)


def test_calibration_gate_flags_a_shifted_channel_offset():
    stay = lambda t: 70.0
    gate = PairConfig(max_offset_dev_ratio=2.0)
    _, ok = _run(gate, stay)
    _, bad = _run(gate, stay, offset_extra=150e-6)        # 4-channel mode: ~150 us inter-channel lag
    _, off = _run(PairConfig(), stay, offset_extra=150e-6)  # gate off: no flag (old behaviour)
    print(f"[mocked] gate on, calibrated: {ok.state}; gate on, +150 us: {bad.state}; gate off, +150 us: {off.state}")
    assert ok.state in ("tracking", "locked")
    assert bad.state == "calibration_mismatch"
    assert off.state != "calibration_mismatch"
