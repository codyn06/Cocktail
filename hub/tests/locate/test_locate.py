"""Tests on SIMULATED data only (mocked world, no real device/audio).

Numbers are printed (run with `pytest -s`) and asserted with margins.
"""
import time

import numpy as np

from locate import BearingEstimator, LocateConfig, gcc_phat, wrap180
from locate.sim import SimWorld, SimSource, run_scenario

AZS = [-150, -90, -30, 20, 70, 130]


def _final_errs(amp, seeds, T=30.0, **kw):
    out = []
    for s in seeds:
        r = run_scenario(AZS, amp, T=T, seed=s, **kw)
        out.append(r["err"][:, -1])
    return np.concatenate(out)


def test_error_vs_head_motion():
    rows = {}
    for amp in (0, 15, 30, 45, 60):
        e = _final_errs(amp, range(6))
        rows[amp] = (np.median(e), np.percentile(e, 90), (e < 20).mean())
        print(f"[sim] yaw amp +-{amp:2d} deg: median {rows[amp][0]:5.1f}  p90 {rows[amp][1]:6.1f}  frac<20deg {rows[amp][2]:.2f}")
    assert rows[30][0] < 10 and rows[30][2] > 0.8
    assert rows[60][0] < 10 and rows[60][2] > 0.9
    # more motion must not be worse than none
    assert rows[45][2] > rows[0][2]


def test_time_to_stable_bearing():
    times = []
    for s in range(6):
        r = run_scenario(AZS, 40, T=30, seed=100 + s)
        t = r["t"]
        for i in range(len(AZS)):
            ok = (r["err"][i] < 15) & (r["conf"][i] > 0.5)
            stable = np.nan
            for k in range(len(t)):
                if ok[k:].all():
                    stable = t[k]
                    break
            times.append(stable)
    times = np.array(times)
    got = times[~np.isnan(times)]
    print(f"[sim] time until err<15deg AND conf>0.5 and stays so (+-40deg motion): n={len(times)} reached={len(got)} median {np.median(got):.1f}s p90 {np.percentile(got, 90):.1f}s")
    assert len(got) / len(times) > 0.8
    assert np.median(got) < 12


def test_front_back_resolution():
    mirror_pairs = [(30, 150), (-30, -150), (60, 120), (-60, -120)]
    res = {}
    for amp in (0, 40):
        ok = n = 0
        for s, (f, b) in enumerate(mirror_pairs):
            for az in (f, b):
                for seed in range(3):
                    r = run_scenario([az], amp, T=15, seed=seed + 10 * s)
                    est = r["est_obj"].estimate("s0")
                    truth = float(wrap180(az - r["yaw"][-1]))
                    n += 1
                    ok += (np.cos(np.deg2rad(est.bearing_deg)) > 0) == (np.cos(np.deg2rad(truth)) > 0)
        res[amp] = ok / n
        print(f"[sim] front/back correct, yaw amp +-{amp} deg, 15 s: {ok}/{n} = {ok / n:.2f}")
    assert res[40] >= 0.9
    assert res[40] > res[0]


def test_no_motion_reports_ambiguity_not_false_confidence():
    r = run_scenario([130], 0, T=20, seed=3)
    e = r["est_obj"].estimate("s0")
    print(f"[sim] no motion: state={e.state} conf={e.confidence:.2f} alt={e.alt_bearing_deg}")
    assert e.state in ("ambiguous", "insufficient_motion")


def test_camera_fusion_helps_low_motion():
    front = [-40, 0, 35]   # always inside the 100 deg camera FOV
    def errs(**kw):
        return np.concatenate([run_scenario(front, 8, T=15, seed=s, **kw)["err"][:, -1] for s in range(6)])
    base = errs()
    fused = errs(camera=dict(fov_deg=100, conf=0.8, sigma_deg=4, rate_hz=4))
    print(f"[sim] amp +-8 deg, 15 s, in-view sources: level-only mean err {base.mean():.1f}  with camera {fused.mean():.1f}")
    assert fused.mean() < base.mean()


def test_tdoa_fusion_helps_low_motion():
    base = _final_errs(8, range(5), T=15)
    fused = _final_errs(8, range(5), T=15, tdoa_sigma_s=40e-6)
    print(f"[sim] amp +-8 deg, 15 s: level-only mean err {base.mean():.1f}  with TDOA {fused.mean():.1f}")
    assert fused.mean() < base.mean()


def test_gcc_phat_sign_and_delay():
    fs = 16000
    rng = np.random.default_rng(0)
    s = rng.standard_normal(4000)
    d = 7  # samples: x_i delayed vs x_j
    xi = np.concatenate([np.zeros(d), s])[:4000]
    xj = s
    lags, cc = gcc_phat(xi, xj, fs, max_tau=0.001)
    tau = lags[np.argmax(cc)]
    assert abs(tau - d / fs) < 1.5 / fs


def test_tdoa_gcc_curve_consistent_with_geometry():
    """GCC-PHAT delay from synthesized two-mic audio -> add_tdoa -> bearing (mirror-resolved by yaw)."""
    fs = 48000
    rng = np.random.default_rng(1)
    world = SimWorld([SimSource(50.0)], seed=1)
    est = BearingEstimator()
    sig = rng.standard_normal(fs * 1)
    errs = []
    for k in range(40):
        t = k * 0.25
        yaw = 40 * np.sin(2 * np.pi * 0.2 * t)
        est.add_head_pose(t, yaw, 0.0)
        tau = world.true_tau(0, 0, 1, yaw, 0.0)
        n = int(round(tau * fs))
        xi = np.roll(sig, n)[:4800] + 0.1 * rng.standard_normal(4800)
        xj = sig[:4800] + 0.1 * rng.standard_normal(4800)
        lags, cc = gcc_phat(xi, xj, fs, max_tau=0.0008)
        est.add_gcc("a", t, 0, 1, lags, cc)
    e = est.estimate("a", 39 * 0.25)
    truth = float(wrap180(50.0 - 40 * np.sin(2 * np.pi * 0.2 * 39 * 0.25)))
    err = abs(float(wrap180(e.bearing_deg - truth)))
    print(f"[sim] GCC-PHAT synthetic audio -> bearing err {err:.1f} deg (state {e.state})")
    assert err < 10


def test_update_cost():
    est = BearingEstimator()
    est.add_head_pose(0.0, 0.0)
    L = np.zeros((2, 5))
    t0 = time.perf_counter()
    n = 400
    for k in range(n):
        est.add_head_pose(k * 0.02, 20 * np.sin(k * 0.05))
        est.add_levels("a", k * 0.02, L + np.random.randn(2, 5))
    dt_add = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(100):
        est.estimate("a")
    dt_est = (time.perf_counter() - t0) / 100
    print(f"[measured, this laptop CPU] add_levels+pose {dt_add * 1e3:.2f} ms/call, estimate() {dt_est * 1e3:.2f} ms/call (1 source)")
    assert dt_add < 0.01 and dt_est < 0.01
