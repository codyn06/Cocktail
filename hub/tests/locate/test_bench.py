"""Scoreboard (locate.bench): metric definitions, and an end-to-end run on synthetic takes.

The synthetic takes carry the exact pair-delay model, so they check the plumbing and the
metrics, not real-world accuracy.
"""

import json
import math
import wave

import numpy as np

from locate import bench
from locate.pair import PairCalibration

CAL = PairCalibration(delay_amp_s=15e-6, axis_deg=-110.0, offset_s=-1e-6, noise_s=3e-6, takes=["t0"])
CUE = {"noise_until": 2.0, "go": 4.0, "stop": 18.0}


def frac_delay(x, d_samples):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x))
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * d_samples), len(x))


def yaw_at(t):
    return np.where(t < CUE["go"], 0.0, 40 * np.sin(2 * np.pi * (t - CUE["go"]) / 8))


def make_take(d, truth, seed):
    d.mkdir()
    rate, secs, blk = 48000, 20, 800
    rng = np.random.default_rng(seed)
    n = rate * secs
    a = rng.standard_normal(n) * 0.05
    a[: int(CUE["noise_until"] * rate)] = 0.0  # source off: room only
    ch0 = np.concatenate([frac_delay(a[s:s + blk], CAL.predict(truth, yaw_at((s + blk / 2) / rate)) * rate)
                          for s in range(0, n, blk)])
    x = np.stack([ch0, a], 1) + rng.standard_normal((n, 2)) * 0.002
    with wave.open(str(d / "audio.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(np.clip(x * 32767, -32768, 32767).astype("<i2").tobytes())
    base = 1_000_000
    pt = np.arange(0, secs, 0.02)
    rows = [(base + int(ti * 1e6), 0.0, math.sin(math.radians(yi) / 2), 0.0, math.cos(math.radians(yi) / 2))
            for ti, yi in zip(pt, yaw_at(pt))]
    np.savetxt(d / "pose.csv", rows, delimiter=",", header="t_us,qx,qy,qz,qw", comments="")
    (d / "meta.json").write_text(json.dumps({
        "truth_deg": truth, "sample_rate": rate, "channels": 2, "first_audio_ts_us": base,
        "audio_frames": n, "pose_samples": len(rows), "cue": CUE}))


def test_region_and_front_back():
    assert [bench.region(t) for t in (0, 60, 90, -100, 150, 180)] == ["front", "front", "side", "side", "behind", "behind"]
    assert bench.front_back_right(10, 30) and not bench.front_back_right(170, 30)
    assert bench.front_back_right(-170, 180) and bench.front_back_right(95, 90) is None


def test_seconds_to_lock_needs_the_bearing_to_stay_close():
    tl = [(t, w, 0.5, "tracking") for t, w in [(4, 90), (5, 40), (6, 5), (7, 60), (8, 10), (9, 0), (10, -5)]]
    assert bench.seconds_to_lock(tl, 0.0, 4.0) == 4.0  # within 25 deg from t=8 on, not at t=6
    assert bench.seconds_to_lock(tl[:4], 0.0, 4.0) is None
    assert bench.seconds_to_lock([(5, 1, 0.9, "locked")], 0.0, 4.0) == 1.0


def test_calibration_table_and_ece():
    pts = [(0.9, True)] * 9 + [(0.9, False)] + [(0.1, False)] * 9 + [(0.1, True)]
    rows, ece = bench.calibration_table(pts)
    assert [(lo, n) for lo, _, _, n in rows] == [(0.0, 10), (0.8, 10)]
    assert ece < 1e-9  # says 90% and is right 90%; says 10% and is right 10%
    _, ece = bench.calibration_table([(0.9, False)] * 10)
    assert abs(ece - 0.9) < 1e-9


def test_bench_end_to_end_on_synthetic_takes(tmp_path, capsys):
    rec = tmp_path / "recordings"
    rec.mkdir()
    for name, truth, seed in (("t0", 0.0, 1), ("t90", 90.0, 2), ("t180", 180.0, 3), ("t-45", -45.0, 4)):
        make_take(rec / name, truth, seed)
    make_take(rec / "unlabeled", 30.0, 5)  # no truth_deg: skipped, not scored
    meta = json.loads((rec / "unlabeled" / "meta.json").read_text())
    (rec / "unlabeled" / "meta.json").write_text(json.dumps({**meta, "truth_deg": None}))
    cal_path, out = tmp_path / "cal.json", tmp_path / "bench.json"
    CAL.save(cal_path)
    bench.main([str(rec), "--calib", str(cal_path), "--json", str(out)])
    text = capsys.readouterr().out
    assert all(f"== {m}" in text for m in bench.METHODS) and "*in-calib" in text
    res = json.loads(out.read_text())
    assert len(res["rows"]) == 4 * len(bench.METHODS)
    s = res["summaries"]["tdoa"]
    assert s["n"] == 4 and s["median_err"] <= 10 and s["fb_right_rate"] == 1.0
    assert s["ece"] is not None and 0.0 <= s["ece"] <= 1.0
    assert res["summaries"]["tdoa-loo"]["n"] == 4
    assert res["summaries"]["phase-loo"]["n"] == 4 and res["summaries"]["phase-loo"]["median_err"] <= 10


def test_an_ambiguous_estimate_is_not_a_lock():
    tl = [(4, 2, 0.5, "ambiguous"), (5, 1, 0.6, "ambiguous"), (6, 0, 0.8, "tracking"), (7, 1, 0.9, "locked")]
    assert bench.seconds_to_lock(tl, 0.0, 4.0) == 2.0


def test_leave_one_out_never_fits_on_the_scored_take(tmp_path):
    from locate import pairsim
    takes = []
    for i, ang in enumerate((0.0, 90.0, 180.0)):
        pairsim.write_take(tmp_path / f"t{i}", ang, seed=i)
        takes.append(tmp_path / f"t{i}")
    loo = bench.loo_calibrations(takes, "quest3s")
    assert set(loo) == set(takes)
    for tk, cal in loo.items():
        assert tk.name not in cal.takes and len(cal.takes) == 2
