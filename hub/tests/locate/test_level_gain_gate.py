"""Level locator vs the measured Quest 3S channel imbalance (SIMULATED data with the measured gaps).

The real 4-channel take (take_4ch_voice3, not in git) showed ch0 15-29 dB below ch1 in the speech
bands and nearly co-located mics. These tests pin: the equal-gain default fails with that gap,
profile_diff / mic_gain_db fix it, and the fit gates withhold bearings where the cue cannot work.
"""
import numpy as np
import pytest

from locate import BearingEstimator, LocateConfig
from locate.sim import run_scenario

AZS = [-150, -90, -30, 20, 70, 130]
OFF = np.zeros((2, 5))
OFF[0] = [-18.6, -23.2, -29.0, -28.7, -20.2]   # measured ch0-ch1 gap per band (dB), real take
GATE = dict(max_resid_chi2=3.0, min_fit_gain=0.05)


def _finals(cfg, seeds=(0, 1), **wk):
    errs, states = [], []
    for s in seeds:
        r = run_scenario(AZS, 40, T=25, seed=s, cfg=cfg, world_kw=wk or None)
        errs += list(r["err"][:, -1])
        states += [r["est_obj"].estimate(f"s{i}").state for i in range(len(AZS))]
    return np.array(errs), states


def test_equal_gain_default_fails_with_measured_imbalance_and_fixes_work():
    e_def, _ = _finals(None, mic_band_offset_db=OFF)
    e_prof, _ = _finals(LocateConfig(profile_diff=True), mic_band_offset_db=OFF)
    e_cal, _ = _finals(LocateConfig(mic_gain_db=tuple(map(tuple, OFF))), mic_band_offset_db=OFF)
    print(f"[sim] frac<25deg: default {(e_def < 25).mean():.2f}, profile_diff {(e_prof < 25).mean():.2f}, "
          f"mic_gain_db {(e_cal < 25).mean():.2f}")
    assert (e_def < 25).mean() < 0.4
    assert (e_prof < 25).mean() > 0.9
    assert (e_cal < 25).mean() > 0.9


def test_gate_withholds_unreliable_and_keeps_good_bearings():
    cfg = LocateConfig(profile_diff=True, **GATE)
    # ear-like geometry, imbalance: most sources shown, and none of the shown are wrong
    e, st = _finals(cfg, mic_band_offset_db=OFF)
    shown = np.array([s != "model_mismatch" for s in st])
    assert shown.mean() > 0.5
    assert np.median(e[shown]) < 10 and (e[shown] < 45).all()
    # equal-gain model without profiling: the residual gate catches the imbalance
    e, st = _finals(LocateConfig(**GATE), mic_band_offset_db=OFF)
    assert all(s == "model_mismatch" for s in st)
    # nearly co-located mics: no head-shadow cue; almost nothing is shown, and no wrong bearing at that
    e, st = _finals(cfg, mic_az=(-5.0, 5.0), mic_band_offset_db=OFF)
    shown = np.array([s != "model_mismatch" for s in st])
    print(f"[sim] co-located: shown {shown.sum()}/{len(st)}, shown errors {e[shown].round(0).tolist()}")
    assert shown.mean() < 0.25
    assert (e[shown] < 30).all()


def test_gate_is_off_by_default():
    e, st = _finals(None, seeds=(0,), mic_az=(-5.0, 5.0))
    assert "model_mismatch" not in st


def test_mic_gain_shape_checked():
    with pytest.raises(ValueError):
        BearingEstimator(LocateConfig(mic_gain_db=(0.0, 0.0, 0.0)))
    BearingEstimator(LocateConfig(mic_gain_db=(0.0, -20.0)))          # (M,)
    BearingEstimator(LocateConfig(mic_gain_db=tuple(map(tuple, OFF))))  # (M, B)
