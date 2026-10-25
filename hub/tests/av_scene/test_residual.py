"""Phase-rotated residual vs genuinely missing source (Phase 1 contract 3). Seeded, 16 kHz."""

import numpy as np
import pytest
from scipy import signal

from replay.residual import (ResidualConfig, downmix, find_unexplained_events, power_db, project_remainder,
                             raw_waveform_remainder)

FS = 16000
N = 2 * FS
CFG = ResidualConfig(n_fft=512, hop=128)


def clean_source(seed=0):
    """Harmonic voiced-like source with syllable AM and a little breath noise, band-limited < 2.2 kHz."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / FS
    x = sum(np.sin(h * phase) / h for h in range(1, 14))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.0 * t) ** 2
    x = x * env + 0.01 * rng.standard_normal(N)
    sos = signal.butter(8, 2200, "low", fs=FS, output="sos")
    x = signal.sosfilt(sos, x)
    return (0.3 * x / np.max(np.abs(x))).astype(np.float64)


def rotate_phase(x, phi_of_f):
    """Unit-magnitude, conjugate-symmetric phase rotation (real output). phi(DC) = phi(Nyquist) = 0."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    phi = np.asarray(phi_of_f(f), float)
    phi[0] = 0.0
    phi[-1] = 0.0
    return np.fft.irfft(X * np.exp(1j * phi), len(x))


ROTATIONS = {
    "quadrature": (lambda f: np.full_like(f, np.pi / 2), 3.01),
    "polarity_inversion": (lambda f: np.full_like(f, np.pi), 6.02),
    # about 137 deg plus a smooth ripple (group delay < 0.6 ms, far below the 32 ms STFT window)
    "smooth_frequency_dependent": (lambda f: 2.4 + 0.5 * np.sin(2 * np.pi * f / 6000.0), None),
}


@pytest.mark.parametrize("name", list(ROTATIONS))
def test_phase_rotation_inflates_raw_remainder_but_not_the_spectral_ones(name):
    phi, expected_db = ROTATIONS[name]
    s = clean_source()
    stem = rotate_phase(s, phi)
    assert power_db(stem) == pytest.approx(power_db(s), abs=0.05)            # same power, different phase

    r_wave, meta = raw_waveform_remainder(s, [stem])
    inflation = power_db(r_wave) - power_db(s)
    assert meta["use"] == "diagnostic" and meta["downmix"] == "mean_over_channels"
    if expected_db is not None:
        assert inflation == pytest.approx(expected_db, abs=0.4)             # +3 dB / +6 dB vs source power
    assert inflation > 2.0                                                  # raw difference is large

    events, proj = find_unexplained_events(s, [stem], FS, CFG)
    assert events == []                                                     # zero false unexplained events
    assert proj.mask_sum_max <= 1.0 + 1e-9
    assert proj.projection_error_db < -25.0                                 # projected remainder is tiny


def test_zero_mismatch_gives_zero_raw_remainder():
    s = clean_source()
    r, _ = raw_waveform_remainder(s, [s.copy()])
    assert np.max(np.abs(r)) == 0.0


def test_genuinely_omitted_second_source_is_still_detected():
    s = clean_source()
    rng = np.random.default_rng(5)
    burst = np.zeros(N)
    a, b = int(1.2 * FS), int(1.5 * FS)
    sos = signal.butter(6, [3000, 5500], "bandpass", fs=FS, output="sos")
    burst[a:b] = signal.sosfilt(sos, rng.standard_normal(b - a)) * np.hanning(b - a)
    burst *= 0.25 / np.max(np.abs(burst))                                   # about -6 dB below the main source peak
    mixture = s + burst
    stem = rotate_phase(s, ROTATIONS["quadrature"][0])                      # phase-rotated AND incomplete inventory

    events, proj = find_unexplained_events(mixture, [stem], FS, CFG)
    assert len(events) == 1
    ev = events[0]
    assert ev["start_s"] == pytest.approx(1.2, abs=0.08) and ev["end_s"] == pytest.approx(1.5, abs=0.1)
    # the projection does not simply zero every bin: mass remains where the missing source is
    assert proj.projection_error_db > -20.0
    # control: same mixture, with the omitted source supplied as a (phase-rotated) stem -> no events
    events2, _ = find_unexplained_events(mixture, [stem, rotate_phase(burst, ROTATIONS["polarity_inversion"][0])],
                                         FS, CFG)
    assert events2 == []


def test_mono_shapes_unchanged_stems_and_multichannel_downmix():
    s = clean_source()
    stem = rotate_phase(s, ROTATIONS["quadrature"][0])
    before = stem.tobytes()
    stereo = np.stack([s, s * 0.5], 1)
    proj = project_remainder(stereo, [stem], FS, CFG)
    assert proj.proxy_waveform.shape == (N,)
    assert np.array_equal(downmix(stereo), stereo.mean(axis=1))
    assert stem.tobytes() == before                                         # raw stems never modified
    with pytest.raises(ValueError):
        raw_waveform_remainder(s, [stem[:-1]])
    r, _ = raw_waveform_remainder(stereo, [stem])
    assert r.shape == (N,)                                                  # explicit mono, no channel broadcast


@pytest.mark.parametrize("basis", ["projection", "power_subtraction"])
def test_both_bases_agree_on_phase_rotation_and_omitted_source(basis):
    s = clean_source()
    stem = rotate_phase(s, ROTATIONS["quadrature"][0])
    assert find_unexplained_events(s, [stem], FS, CFG, basis=basis)[0] == []
    rng = np.random.default_rng(5)
    burst = np.zeros(N)
    a, b = int(1.2 * FS), int(1.5 * FS)
    burst[a:b] = signal.sosfilt(signal.butter(6, [3000, 5500], "bandpass", fs=FS, output="sos"),
                                rng.standard_normal(b - a)) * np.hanning(b - a)
    burst *= 0.25 / np.max(np.abs(burst))
    assert len(find_unexplained_events(s + burst, [stem], FS, CFG, basis=basis)[0]) == 1
    with pytest.raises(ValueError):
        find_unexplained_events(s, [stem], FS, CFG, basis="nope")


def test_coherent_mask_partitions_do_not_create_unexplained_events():
    """Stems that are soft-masked copies of the mixture add coherently (|S_k| = M_k |X|)."""
    rng = np.random.default_rng(8)
    a = clean_source()
    b = signal.sosfilt(signal.butter(4, [150, 2000], "bandpass", fs=FS, output="sos"), rng.standard_normal(N)) * 0.1
    mix = a + b
    from replay.residual import istft, stft
    X, A, B = stft(mix, CFG), np.abs(stft(a, CFG)) ** 2, np.abs(stft(b, CFG)) ** 2
    Ma = A / (A + B + 1e-12)
    sa, sb = istft(X * Ma, CFG, N), istft(X * (1 - Ma), CFG, N)
    events, proj = find_unexplained_events(mix, [sa, sb], FS, CFG)
    assert events == [] and proj.projection_error_db < -25.0


def test_omitted_source_overlapping_an_included_stems_bins_is_detected_when_clearly_louder():
    """alpha_max bounds the gain a candidate stem may take, so a missing source that shares the SAME
    time-frequency bins is only detectable once it exceeds what alpha_max * |S| can explain (about
    +3.5 dB over the stem for amplitude masks at alpha_max = 1.5). Quieter overlapped sources can be
    partly absorbed: a documented limit of the projection, not a claim of full coverage."""
    s = clean_source()
    stem = rotate_phase(s, ROTATIONS["quadrature"][0])
    rng = np.random.default_rng(6)
    a, b = int(1.2 * FS), int(1.5 * FS)
    loud = np.zeros(N)
    loud[a:b] = signal.sosfilt(signal.butter(6, [200, 1800], "bandpass", fs=FS, output="sos"),
                               rng.standard_normal(b - a)) * np.hanning(b - a)
    loud *= 3.0 * np.max(np.abs(s)) / np.max(np.abs(loud))              # about +10 dB over the included stem
    events, _ = find_unexplained_events(s + loud, [stem], FS, CFG)
    assert len(events) == 1 and events[0]["start_s"] < 1.4 and events[0]["end_s"] > 1.3
