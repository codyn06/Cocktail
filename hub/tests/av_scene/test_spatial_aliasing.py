"""Narrowband multi-peak retention, per-pair aliasing and post-PHAT WDO (Phase 1 contract 2)."""

import numpy as np
import pytest
from scipy import signal

from locate.av_spatial import (analyze_array, analyze_pair, c_sound, direction_from_az_el, enumerate_delay_aliases,
                               pair_alias_freq, stft_frames, wdo_weights, weighted_gcc_phat)

FS = 48000
N = FS // 2
T = np.arange(N) / FS
ENV = np.hanning(N)
MICS = np.array([[0.0, 0, 0], [0.05, 0, 0], [0.15, 0, 0]])   # pair spacings 0.05, 0.15, 0.10 m
R = direction_from_az_el(30.0, 0.0)


def arrival_times(temp_c=20.0):
    # wavefront reaches mics with larger p.r first: t_m = -p_m.r / c
    return -(MICS @ R) / c_sound(temp_c)


def tone_array(freq, noise=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    tm = arrival_times()
    return np.stack([ENV * np.sin(2 * np.pi * freq * (T - t)) for t in tm], 1) + noise * rng.standard_normal((N, 3))


def noise_array(seed=1):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(N + 2000)
    tm = arrival_times()
    X = np.fft.rfft(s)
    f = np.fft.rfftfreq(len(s), 1 / FS)
    cols = [np.fft.irfft(X * np.exp(-2j * np.pi * f * (t - tm.min())), len(s))[1000:1000 + N] for t in tm]
    return np.stack(cols, 1)


def test_pair_cutoffs_depend_on_each_pairs_spacing():
    assert pair_alias_freq(0.05) == pytest.approx(c_sound() / 0.1)
    assert pair_alias_freq(0.15) < pair_alias_freq(0.10) < pair_alias_freq(0.05)


def test_tone_above_pair_cutoff_keeps_exactly_the_analytic_aliases():
    f, d = 2000.0, 0.15                                   # above 1144 Hz cutoff of this pair
    x = tone_array(f)
    a = analyze_pair(x[:, 0], x[:, 2], FS, d, pair=(0, 2))
    tau0 = d * np.sin(np.radians(30.0)) / c_sound()       # channel 0 later (positive)
    expected = sorted(enumerate_delay_aliases(tau0, f, d))
    assert len(expected) == 2                             # tau0 and tau0 - 1/f
    got = sorted(t for t, _ in a.peaks)
    assert len(got) == len(expected)
    assert got == pytest.approx(expected, abs=3e-6)
    assert a.aliasing_multimodal_flag and a.narrowband
    strengths = [s for _, s in a.peaks]
    assert max(strengths) / min(strengths) < 1.1          # a tie: do not force an identity from it
    assert a.best_tau_s is None


def test_flag_is_per_pair_longer_baselines_flag_first():
    flags = {a.pair: a.aliasing_multimodal_flag for a in analyze_array(tone_array(2000.0), FS, MICS)}
    assert flags == {(0, 1): False, (0, 2): True, (1, 2): True}   # cutoffs 3434 / 1144 / 1717 Hz
    flags_hi = {a.pair: a.aliasing_multimodal_flag for a in analyze_array(tone_array(4000.0), FS, MICS)}
    assert all(flags_hi.values())


def test_broadband_control_keeps_a_single_peak_and_no_flag():
    x = noise_array()
    a = analyze_pair(x[:, 0], x[:, 2], FS, 0.15, pair=(0, 2))
    tau0 = 0.15 * np.sin(np.radians(30.0)) / c_sound()
    assert not a.narrowband and not a.aliasing_multimodal_flag
    assert len(a.peaks) == 1 and a.peaks[0][0] == pytest.approx(tau0, abs=3e-6)
    assert a.best_tau_s == pytest.approx(tau0, abs=3e-6)


def test_temperature_changes_the_admissible_alias_set():
    cold = enumerate_delay_aliases(115e-6, 3000.0, 0.15, temp_c=0.0)
    hot = enumerate_delay_aliases(115e-6, 3000.0, 0.15, temp_c=40.0)
    assert len(cold) == 3 and len(hot) == 2
    assert pair_alias_freq(0.15, 40.0) > pair_alias_freq(0.15, 0.0)


# ---------------------------------------------------------------- post-PHAT WDO
def _two_band_sources(seed=3):
    rng = np.random.default_rng(seed)
    n = FS
    lo = signal.sosfilt(signal.butter(6, [200, 2500], "bandpass", fs=FS, output="sos"), rng.standard_normal(n))
    hi = signal.sosfilt(signal.butter(6, [3000, 8000], "bandpass", fs=FS, output="sos"), rng.standard_normal(n))
    floor = 0.01                                            # -40 dB broadband floor keeps masks soft, not 0/1
    return lo + floor * rng.standard_normal(n), hi + floor * rng.standard_normal(n)


def _frac_delay(x, tau):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * tau), len(x))


def test_soft_wdo_dominance_survives_only_when_applied_after_phat():
    s1, s2 = _two_band_sources()
    t1, t2 = 150e-6, -150e-6                                 # channel 0 later for source 1, earlier for source 2
    x0 = _frac_delay(s1, t1) + _frac_delay(s2, t2)
    x1 = s1 + s2
    nfft, hop, max_tau = 1024, 512, 0.3e-3
    X0, X1 = stft_frames(x0, nfft, hop), stft_frames(x1, nfft, hop)
    W1, W2 = wdo_weights([stft_frames(s1, nfft, hop), stft_frames(s2, nfft, hop)], min_weight=0.0)

    def at(lags, cc, tau):
        return cc[np.argmin(np.abs(lags - tau))]

    # Control: identical soft mask on both channels BEFORE PHAT -> cross-spectrum scaled by W^2, PHAT divides it out.
    lags, pre = weighted_gcc_phat(W1 * X0, W1 * X1, FS, nfft, max_tau, W=None, min_power_db=-300)
    ratio_pre = at(lags, pre, t1) / at(lags, pre, t2)
    assert 0.4 < ratio_pre < 2.5                            # dominance lost: both sources count about equally

    # Weights applied AFTER PHAT keep the dominance and recover source 1's delay.
    lags, post = weighted_gcc_phat(X0, X1, FS, nfft, max_tau, W=W1)
    assert at(lags, post, t1) > 5 * abs(at(lags, post, t2))
    assert lags[np.argmax(post)] == pytest.approx(t1, abs=3e-6)
    lags, post2 = weighted_gcc_phat(X0, X1, FS, nfft, max_tau, W=W2)
    assert lags[np.argmax(post2)] == pytest.approx(t2, abs=3e-6)


def test_wdo_min_weight_floors_poorly_dominated_bins():
    s1, s2 = _two_band_sources()
    mags = [np.abs(stft_frames(s, 1024, 512)) for s in (s1, s2)]
    soft = wdo_weights(mags, min_weight=0.0)[0]
    floored = wdo_weights(mags, min_weight=0.1)[0]
    assert (soft > 0).all() and (floored == 0).any()
    assert (floored[floored > 0] >= 0.1).all()
    assert np.allclose(floored[floored > 0], soft[floored > 0])      # soft where kept, never binarised
