"""K-weighted perceptual envelope and reverberant-decay hysteresis (Phase 1 contract 4).

Documented levels (all RMS over the sounding part, unweighted dBFS):
  rumble 40 / 60 Hz sines and 20-80 Hz band noise : -34 dBFS
  4 kHz-region chirp (3 -> 5 kHz sweep)           : -34 dBFS, and -46 dBFS for the low-level case
  sustained 440 Hz tone                           : -30 dBFS
Gate settings are EnvelopeConfig defaults: activate -45 dBFS_K, sustain -51 dBFS_K, tail hold 150 ms,
peak drop 12 dB, reopen lockout 600 ms / fresh-onset rise 6 dB, LF fraction limit 0.7, 25 fps frames.
"""

import numpy as np
import pytest
from scipy import signal

from ingest.media_io import read_wav, write_wav
from render.envelope import (EnvelopeConfig, compute_envelope, k_response_db, k_response_reference_db, k_weight,
                             k_weighting_sos)

DUR = 3.0


def frames(dur=DUR, fps=25):
    return (np.arange(0, dur, 1.0 / fps) * 1e6).astype(np.int64)


def scale_to(x, dbfs):
    seg = x[np.abs(x) > 0]
    return x * (10 ** (dbfs / 20) / np.sqrt(np.mean(seg ** 2)))


def place(sig, fs, t0=1.0, dur=DUR):
    out = np.zeros(int(dur * fs))
    a = int(t0 * fs)
    out[a:a + len(sig)] = sig
    return out


def chirp(fs, dbfs, secs=1.0):
    t = np.arange(int(secs * fs)) / fs
    return scale_to(signal.chirp(t, 3000, secs, 5000) * np.hanning(len(t)) ** 0.2, dbfs)


def sine(fs, f, dbfs, secs=1.0):
    t = np.arange(int(secs * fs)) / fs
    return scale_to(np.sin(2 * np.pi * f * t), dbfs)


def rumble_noise(fs, dbfs, secs=1.0, seed=2):
    rng = np.random.default_rng(seed)
    sos = signal.butter(4, [20, 80], "bandpass", fs=fs, output="sos")
    return scale_to(signal.sosfilt(sos, rng.standard_normal(int(secs * fs))), dbfs)


# ------------------------------------------------------------------ filter response
def test_published_48k_coefficients():
    sos = k_weighting_sos(48000)
    assert sos[0, :3] == pytest.approx([1.53512485958697, -2.69169618940638, 1.19839281085285], abs=1e-9)
    assert sos[0, 4:] == pytest.approx([-1.69065929318241, 0.73248077421585], abs=1e-9)
    assert sos[1, 4:] == pytest.approx([-1.99004745483398, 0.99007225036621], abs=1e-9)


@pytest.mark.parametrize("fs", [16000, 48000])
def test_frequency_response_matches_the_analog_prototype(fs):
    f = np.array([20, 40, 60, 100, 300, 1000, 2000, 4000, 6000.0])
    assert k_response_db(fs, f) == pytest.approx(k_response_reference_db(fs, f), abs=0.02)
    resp = dict(zip(f, k_response_db(fs, f)))
    assert resp[1000.0] == pytest.approx(0.69, abs=0.1)         # BS.1770's -0.691 LKFS offset
    assert 2.8 < resp[6000.0] < 4.25                             # shelf boost toward +4 dB
    assert resp[40.0] < -4.5 and resp[20.0] < resp[40.0]          # high-pass, but nowhere near brick-wall


@pytest.mark.parametrize("fs", [8000, 500000, 44100.5])
def test_unsupported_sample_rates_are_rejected(fs):
    with pytest.raises(ValueError):
        k_weighting_sos(fs)


# ------------------------------------------------------------------ rumble versus chirp
@pytest.mark.parametrize("fs", [16000, 48000])
def test_rumble_rejected_and_chirp_detected_at_equal_rms(fs):
    ft = frames()
    cfg = EnvelopeConfig()
    rumbles = {"40Hz": sine(fs, 40, -34), "60Hz": sine(fs, 60, -34), "noise20-80": rumble_noise(fs, -34)}
    c = place(chirp(fs, -34), fs)
    env_c = compute_envelope(c, fs, ft, cfg)
    assert len(env_c.segments_us) == 1
    onset_s = env_c.segments_us[0][0] / 1e6
    assert onset_s == pytest.approx(1.0, abs=0.02)
    for name, r in rumbles.items():
        x = place(r, fs)
        env = compute_envelope(x, fs, ft, cfg)
        assert env.segments_us == [], name                        # never lights a silent person
        assert not env.active.any(), name
        # K-weighting is doing its part: lower weighted level than the chirp at equal RMS...
        assert env_c.level_dbfs_k.max() - env.level_dbfs_k.max() > (8.0 if name != "noise20-80" else 3.0), name
        # ...but the weighting alone is not the guarantee: with the LF-dominance test off, the level gate opens
        loose = compute_envelope(x, fs, ft, EnvelopeConfig(lf_fraction_max=1.01))
        assert loose.segments_us != [], name


@pytest.mark.parametrize("fs", [16000, 48000])
def test_low_level_chirp_and_sustained_tone_are_kept(fs):
    ft = frames()
    low = compute_envelope(place(chirp(fs, -46), fs), fs, ft)
    assert len(low.segments_us) == 1
    tone = compute_envelope(place(sine(fs, 440, -30, secs=2.0), fs), fs, ft)
    assert len(tone.segments_us) == 1                              # no repeated broadband onsets needed
    a, b = tone.segments_us[0]
    assert a / 1e6 == pytest.approx(1.0, abs=0.02) and b / 1e6 >= 2.9


# ------------------------------------------------------------------ reverberation
def reverberant_burst(fs, rt60=0.7, level=-12.0):
    rng = np.random.default_rng(7)
    burst = scale_to(signal.sosfilt(signal.butter(4, [500, 3000], "bandpass", fs=fs, output="sos"),
                                    rng.standard_normal(int(0.4 * fs))) * np.hanning(int(0.4 * fs)) ** 0.1, level)
    n_ir = int(1.5 * fs)
    t = np.arange(n_ir) / fs
    ir = rng.standard_normal(n_ir) * np.exp(-6.9078 * t / rt60)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir ** 2))
    return signal.fftconvolve(burst, ir)[: int(1.9 * fs)]


def test_reverberant_tail_cannot_reopen_the_active_flag():
    fs = 48000
    x = place(reverberant_burst(fs), fs, t0=0.5, dur=4.0)
    ft = frames(4.0)
    env = compute_envelope(x, fs, ft)
    assert len(env.segments_us) == 1
    off = env.segments_us[0][1] / 1e6
    assert off < 0.5 + 0.4 + 0.6                                   # closes soon after the direct sound ends
    assert not env.gate_active[ft / 1e6 > off + 0.05].any()        # and stays closed through the decaying tail
    # tail energy really is above the activation level when the flag closes (this is what must not reopen it)
    tail_i = np.argmin(np.abs(ft / 1e6 - off))
    assert env.level_dbfs_k[tail_i] > env.config["activate_dbfs_k"]
    # control: without lockout / fresh-onset rule a plain threshold gate chatters open again
    naive = compute_envelope(x, fs, ft, EnvelopeConfig(reopen_lockout_ms=0.0, onset_rise_db=-1e9))
    assert len(naive.segments_us) > 1


def test_trusted_offset_bounds_the_tail_hold_and_validity_masks_gate_the_flag():
    fs = 48000
    x = place(sine(fs, 440, -30, secs=2.0), fs)
    ft = frames()
    env = compute_envelope(x, fs, ft, trusted_offsets_us=[1_500_000])
    a, b = env.segments_us[0]
    assert b / 1e6 == pytest.approx(1.5, abs=0.01)                 # source offset is trusted: no extra prolonging
    valid = ft < 1_800_000
    env2 = compute_envelope(x, fs, ft, assoc_valid=valid, mask_valid=np.ones(len(ft), bool))
    assert env2.gate_active[ft > 1_900_000].any() and not env2.active[ft > 1_900_000].any()


# ------------------------------------------------------------------ timing / immutability
def test_attack_smoothing_adds_no_onset_delay_and_wavs_are_untouched(tmp_path):
    fs = 48000
    x = place(sine(fs, 1000, -30, secs=1.5), fs, t0=1.013)         # onset between frames
    ft = frames()
    path = tmp_path / "stem.wav"
    write_wav(path, x, fs, "float32")
    before_bytes, before_x = path.read_bytes(), x.copy()
    env = compute_envelope(x, fs, ft)
    assert np.array_equal(x, before_x) and path.read_bytes() == before_bytes
    assert np.array_equal(read_wav(path)[0][:, 0], before_x.astype(np.float32))
    assert k_weight(x, fs).shape == x.shape                        # weighting is a separate copy
    gate_onset = env.segments_us[0][0] / 1e6
    assert gate_onset == pytest.approx(1.013, abs=0.005)           # back-tracked, not smoothed
    q = 20.0                                                       # 1000 / (2 * 25 fps) ms
    half = 0.5 * env.glow.max()
    t_glow = ft[np.argmax(env.glow >= half)] / 1e6
    assert abs(t_glow - 1.013) * 1000 <= q + 1.0                   # compensated: 50% glow within one quantum
    # an uncompensated 30 ms one-pole fed the gate flag would reach 50% roughly 21 ms late
    assert env.glow[ft / 1e6 < 0.95].max() == 0.0                  # no glow before the onset window
    assert not env.active[ft / 1e6 < 1.0].any()
    # never claims activity long after the source stopped
    assert not env.active[ft / 1e6 > 2.7].any()
