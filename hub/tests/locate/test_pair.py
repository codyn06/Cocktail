"""Mic-pair delay locator (pair.py) on synthetic signals with known answers."""

import numpy as np
import pytest

import protocol as p
from ingest import IngestState
from locate.live import LiveLocator
from locate.pair import PairCalibration, PairLocator, fit_calibration, frame_delays
from mockclient.simulate import quat_yaw_pitch
from separate.live import BAND_CENTERS_HZ, StreamChunk

CAL = PairCalibration(delay_amp_s=15e-6, axis_deg=-110.0, offset_s=-1e-6, noise_s=3e-6)


def sweep(n=400, amp=45.0):
    return amp * np.sin(np.linspace(0, 6 * np.pi, n))


def frac_delay(x, d_samples):
    """Delay x by a fractional number of samples (circular, fine for noise)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x))
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * d_samples), len(x))


def test_calibration_recovers_device_constants():
    rng = np.random.default_rng(0)
    takes = []
    for truth in (0, 90, -90, 180):
        y = sweep()
        tau = CAL.predict(truth, y) + rng.normal(0, 3e-6, len(y))
        takes.append((y, tau, truth))
    cal = fit_calibration(takes)
    assert abs(cal.delay_amp_s - CAL.delay_amp_s) < 1e-6
    assert abs((cal.axis_deg - CAL.axis_deg + 180) % 360 - 180) < 3
    assert abs(cal.offset_s - CAL.offset_s) < 1e-6
    assert 2e-6 < cal.noise_s < 4e-6


@pytest.mark.parametrize("truth", [0, 90, -90, 180, 45])
def test_locator_finds_the_source_with_head_motion(truth):
    rng = np.random.default_rng(1)
    loc = PairLocator(CAL)
    y = sweep()
    for i, yi in enumerate(y):
        t = i * 0.05
        loc.add_head_pose(t, yi)
        loc.add_delay("s", t, CAL.predict(truth, yi) + rng.normal(0, 3e-6))
    e = loc.estimate("s")
    assert abs((e.world_deg - truth + 180) % 360 - 180) <= 10, e
    assert e.state in ("tracking", "locked")


def test_still_head_is_insufficient_motion():
    loc = PairLocator(CAL)
    for i in range(200):
        loc.add_head_pose(i * 0.05, 0.0)
        loc.add_delay("s", i * 0.05, CAL.predict(60, 0.0))
    assert loc.estimate("s").state in ("insufficient_motion", "ambiguous")


def test_frame_delays_measures_a_sub_sample_delay():
    rng = np.random.default_rng(2)
    rate = 48000
    a = rng.standard_normal(rate)
    d = 15e-6 * rate  # 0.72 samples
    x = np.stack([frac_delay(a, d), a], 1)  # channel 0 later by 15 us
    _, tau, _ = frame_delays(x, rate, 4096, 2400)
    assert abs(np.median(tau) - 15e-6) < 2e-6


def test_live_locator_tdoa_mode_finds_a_stream(monkeypatch):
    rng = np.random.default_rng(3)
    st = IngestState()
    from dataclasses import replace
    loc = LiveLocator(st, method="tdoa", calib=replace(CAL, rate_hz=16000, win=1024))  # a live-rate calibration
    truth, sr, hop_s = 70.0, 16000, 1.0
    n_hops = 24
    yaw_t = lambda t: 45 * np.sin(2 * np.pi * t / 8)
    for k in range(int(n_hops / 0.02)):
        t = k * 0.02
        st.on_pose_channel(p.PoseSample(seq=k, device_ts_us=int(t * 1e6), epoch=0, pos=(0, 1.6, 0),
                                        rot_xyzw=quat_yaw_pitch(float(yaw_t(t)), 0)))
    for h in range(n_hops):
        t0 = h * hop_s
        # a hop of noise whose two mic images are offset by the model delay at this hop's yaw
        a = rng.standard_normal(sr)
        # delay channel 0 block by block (800 samples = 50 ms) by the model delay at that time
        ch0 = np.concatenate([frac_delay(a[f0:f0 + 800], CAL.predict(truth, yaw_t(t0 + (f0 + 400) / sr)) * sr)
                              for f0 in range(0, sr, 800)])
        mic = np.stack([ch0, a]) * 0.1
        lv_ts = int(t0 * 1e6) + 50_000 + np.arange(19) * 50_000
        chunk = StreamChunk("s1", "speech", 0.9, int(t0 * 1e6), mic.mean(0), mic, lv_ts,
                            np.zeros((19, 2, len(BAND_CENTERS_HZ))), np.ones(19))
        loc.on_chunks([chunk])
    e = loc.estimates()["s1"]
    assert abs((e.world_deg - truth + 180) % 360 - 180) <= 25, e
    assert loc.sources()[0].bearing_deg is not None


def _make_take_for_tdoa(tmp_path):
    import json
    import math
    import wave
    rate, secs = 48000, 24
    rng = np.random.default_rng(4)
    n = rate * secs
    x = rng.standard_normal(n) * 0.05
    pcm = np.clip(np.stack([x, x], 1) * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(tmp_path / "audio.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
    base = 1_000_000
    rows = []
    for ti in np.arange(0, secs, 0.02):
        h = math.radians(40 * math.sin(2 * math.pi * ti / 10)) / 2
        rows.append((base + int(ti * 1e6), 0.0, math.sin(h), 0.0, math.cos(h)))
    np.savetxt(tmp_path / "pose.csv", rows, delimiter=",", header="t_us,qx,qy,qz,qw", comments="")
    (tmp_path / "meta.json").write_text(json.dumps({
        "truth_deg": 0.0, "sample_rate": rate, "channels": 2, "first_audio_ts_us": base,
        "audio_frames": n, "pose_samples": len(rows), "cue": {"noise_until": 0.0, "go": 4.0, "stop": 22.0}}))


def test_real_eval_runs_with_tdoa_method(tmp_path):
    from locate import real_eval
    _make_take_for_tdoa(tmp_path)
    r = real_eval.run(tmp_path, 0.0, 12.0, verbose=False, method="tdoa", calib=CAL)
    assert r["method"] == "tdoa" and "world_deg" in r and "pass" in r


def test_phase_slope_delay_and_its_sigma_are_honest():
    from locate.pair import frame_delay_phase
    rng = np.random.default_rng(5)
    rate, n, tau = 48000, 4096, 10e-6
    err, z = [], []
    for _ in range(100):
        a = rng.standard_normal(n)
        x = np.stack([frac_delay(a, tau * rate), a], 1) + rng.standard_normal((n, 2)) * 0.3
        d, sig = frame_delay_phase(x, rate)
        err.append(d - tau)
        z.append((d - tau) / sig)
    assert abs(np.mean(err)) < 0.3e-6 and np.std(err) < 1e-6
    assert 0.7 < np.std(z) < 1.4  # the reported sigma matches the actual spread


def test_weighted_calibration_recovers_constants_and_model_error():
    rng = np.random.default_rng(6)
    takes = []
    for truth in (0, 90, -90, 180):
        y = sweep()
        sig = rng.uniform(1e-6, 4e-6, len(y))
        tau = CAL.predict(truth, y) + rng.normal(0, 1, len(y)) * np.hypot(sig, 2e-6)
        takes.append((y, tau, truth, sig))
    cal = fit_calibration(takes)
    assert cal.delay_method == "phase"
    assert abs(cal.delay_amp_s - CAL.delay_amp_s) < 1e-6
    assert abs((cal.axis_deg - CAL.axis_deg + 180) % 360 - 180) < 3
    assert 1.4e-6 < cal.model_s < 2.6e-6
    assert cal.frame_sigma(3e-6) > cal.frame_sigma(1e-6) > cal.model_s


def test_old_calibration_files_still_load_as_gcc():
    cal = PairCalibration.default()
    assert cal.delay_method == "gcc" and cal.model_s is None and cal.frame_sigma(1e-6) == cal.noise_s


def test_live_locator_rejects_a_phase_calibration():
    from dataclasses import replace
    with pytest.raises(ValueError):
        LiveLocator(IngestState(), method="tdoa", calib=replace(CAL, delay_method="phase", model_s=2e-6))


def test_adapt_tracks_a_source_whose_swing_and_offset_differ_from_calibration():
    rng = np.random.default_rng(7)
    errs = {False: [], True: []}
    for truth in (0, 40, 90, 135, 180, -45, -90, -150):
        for adapt in (False, True):
            from locate.pair import PairConfig
            loc = PairLocator(CAL, PairConfig(adapt=adapt))
            for i, yi in enumerate(sweep(400, 40.0)):
                loc.add_head_pose(i * 0.05, yi)
                tau = 0.7 * CAL.delay_amp_s * np.sin(np.radians(truth - CAL.axis_deg - yi)) + CAL.offset_s + 2e-6
                loc.add_delay("s", i * 0.05, tau + rng.normal(0, 3e-6))
            e = loc.estimate("s")
            errs[adapt].append(abs((e.world_deg - truth + 180) % 360 - 180))
    assert np.median(errs[True]) < np.median(errs[False]) and max(errs[True]) <= 20, errs
