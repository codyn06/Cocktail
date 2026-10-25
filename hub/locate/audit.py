"""Channel and cue audit of a recorded take (works on takes without a truth angle).

    cd hub
    uv run python -m locate.audit C:/path/to/take_dir [more takes]

Answers "can the level and delay cues work on this hardware's channels?" before any locator runs:

* channel health: rms, sub-100 Hz rumble share, which channels carry sound at all;
* ch0 vs ch1 response: per-band level gap, how that gap moves with the loudness of the sound
  (a channel near its noise floor does), speech-band coherence, group-delay offset;
* head-motion coupling: how much the per-band level difference (ILD) and the inter-mic delay
  change with head yaw (no coupling = the cue has nothing to measure, whatever the model says);
* the level locator's own gates (resid chi-square, fit gain) on the mixture, default vs recommended.

Measured on the take as recorded (the mixture, no separation), so it checks the hardware, not
the separator. Offline CPU only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal

from .estimator import BearingEstimator, LocateConfig
from .pair import PairCalibration
from .real_eval import circ_diff, load, yaw_deg  # noqa: F401  (circ_diff re-exported for callers)

BANDS = (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
RECOMMENDED = dict(profile_diff=True, max_resid_chi2=3.0, min_fit_gain=0.05)


def band_levels(x: np.ndarray, rate: int, hz: int = 20) -> np.ndarray:
    """(n,) -> (frames, bands) dB levels in 100 ms windows every 50 ms (as separate.live.band_levels)."""
    hop = rate // hz
    f, _, Z = signal.stft(x, rate, nperseg=2 * hop, noverlap=hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    edges = [(c / np.sqrt(2), min(c * np.sqrt(2), 8000.0)) for c in BANDS]
    out = np.stack([P[(f >= lo) & (f < hi)].mean(axis=0) for lo, hi in edges], axis=-1)
    return 10 * np.log10(out + 1e-10)


def band_delay(x2: np.ndarray, rate: int, fmin=300.0, fmax=3500.0, n=4096, hop=2400, up=16):
    """GCC-PHAT delay (s, + = ch0 later) per frame from the fmin..fmax band at `up`x interpolation,
    plus ch1 level (dB) of each frame. Returns t, tau, level."""
    t, tau, lev = [], [], []
    fr = np.fft.rfftfreq(n, 1 / rate)
    m = (fr > fmin) & (fr < fmax)
    L = int(0.0003 * rate * up)
    for s in range(0, len(x2) - n, hop):
        a, b = x2[s:s + n, 0] * np.hanning(n), x2[s:s + n, 1] * np.hanning(n)
        G = np.fft.rfft(a) * np.conj(np.fft.rfft(b))
        G = G / (np.abs(G) + 1e-30)
        G[~m] = 0
        cc = np.fft.irfft(G, n * up)
        cc = np.concatenate([cc[-L:], cc[:L + 1]])
        t.append((s + n / 2) / rate)
        tau.append((int(np.argmax(cc)) - L) / (rate * up))
        lev.append(10 * np.log10((b ** 2).mean() + 1e-12))
    return np.array(t), np.array(tau), np.array(lev)


def audit(take: Path, verbose: bool = True) -> dict:
    meta, x, rate, pose = load(take)
    out: dict = {"take": str(take), "channels": x.shape[1], "rate": rate, "seconds": len(x) / rate,
                 "truth": meta.get("truth_deg"), "flags": []}
    pt = (pose[:, 0] - meta["first_audio_ts_us"]) / 1e6
    yaw = np.degrees(np.unwrap(np.radians(yaw_deg(pose[:, 1:]))))
    out["yaw_std"] = float(yaw.std())
    sos = signal.butter(4, 100, "hp", fs=rate, output="sos")
    rms = [float(20 * np.log10(x[:, c].std() + 1e-12)) for c in range(x.shape[1])]
    hp = [float(20 * np.log10(signal.sosfilt(sos, x[:, c]).std() + 1e-12)) for c in range(x.shape[1])]
    out["rms_db"], out["rms_hp100_db"] = rms, hp
    live = [c for c in range(x.shape[1]) if rms[c] > -100]
    out["live_channels"] = live
    if x.shape[1] >= 2 and len(live) < x.shape[1]:
        out["flags"].append(f"only channels {live} carry sound ({x.shape[1]} delivered)")
    if len(live) < 2:
        out["flags"].append("fewer than 2 live channels: no inter-mic cue is possible")
        return _report(out, verbose)
    if max(rms[0] - hp[0], rms[1] - hp[1]) > 8:
        out["flags"].append("a channel is dominated by sub-100 Hz energy (rumble/handling): broadband level is not speech")

    x2 = x[:, :2]
    L0, L1 = band_levels(x2[:, 0], rate), band_levels(x2[:, 1], rate)
    t = (np.arange(len(L0)) * (rate // 20) + rate // 20) / rate
    ild = L0 - L1
    bb1 = 10 * np.log10((10 ** (L1 / 10)).sum(1))
    act = bb1 > np.percentile(bb1, 60)
    y = np.interp(t, pt, yaw)
    out["ild_mean_db"] = ild[act].mean(0).round(1).tolist()
    out["ild_std_db"] = ild[act].std(0).round(1).tolist()
    # does the gap move with loudness? slope of ch0 level on ch1 level (1.0 = linear, matched channels)
    out["ch0_on_ch1_slope"] = [round(float(np.polyfit(L1[:, b], L0[:, b], 1)[0]), 2) for b in range(len(BANDS))]
    out["ild_yaw_corr"] = [round(float(np.corrcoef(ild[act, b], y[act])[0, 1]), 2) for b in range(len(BANDS))]
    out["ild_yaw_slope_db_per_deg"] = [round(float(np.polyfit(y[act], ild[act, b], 1)[0]), 3) for b in range(len(BANDS))]
    f, C = signal.coherence(x2[:, 0], x2[:, 1], rate, nperseg=4096)
    out["coherence"] = [round(float(C[(f >= c / 1.414) & (f < c * 1.414)].mean()), 2) for c in BANDS]

    gap = np.array(out["ild_mean_db"])
    if np.abs(gap).max() > 6:
        out["flags"].append(f"ch0 vs ch1 level gap up to {np.abs(gap).max():.0f} dB (equal-gain model invalid)")
    if np.mean(out["ch0_on_ch1_slope"]) < 0.7:
        out["flags"].append("ch0 rises much less than ch1 with loudness (near its noise floor or compressed): "
                            "the gap is level-dependent, not a fixed gain")
    if np.mean(out["coherence"][:3]) < 0.5:
        out["flags"].append("low ch0/ch1 coherence in speech bands: the channels are not a matched mic pair")
    swing = np.abs(np.array(out["ild_yaw_slope_db_per_deg"])).max() * max(float(yaw.std()), 1.0)
    if swing < 1.5 * min(out["ild_std_db"]) / 3:
        out["flags"].append("level difference does not follow head yaw (no head-shadow signal to fit)")

    # delay cue
    td, tau, lev = band_delay(x2, rate)
    loud = lev > np.percentile(lev, 70)
    yd = np.interp(td, pt, yaw)
    cal = PairCalibration.default()
    A = np.stack([np.ones(loud.sum()), np.sin(np.radians(yd[loud])), np.cos(np.radians(yd[loud]))], 1)
    coef = np.linalg.lstsq(A, tau[loud], rcond=None)[0]
    r2 = 1 - ((tau[loud] - A @ coef) ** 2).sum() / ((tau[loud] - tau[loud].mean()) ** 2).sum()
    out["delay"] = {"median_us": float(np.median(tau[loud]) * 1e6), "iqr_us": [float(v * 1e6) for v in np.percentile(tau[loud], [25, 75])],
                    "yaw_fit_amp_us": float(np.hypot(coef[1], coef[2]) * 1e6), "yaw_fit_r2": float(r2),
                    "calib_offset_us": cal.offset_s * 1e6, "calib_amp_us": cal.delay_amp_s * 1e6}
    dev = abs(np.median(tau[loud]) - cal.offset_s)
    if dev > 2 * cal.delay_amp_s:
        out["flags"].append(f"delay offset {np.median(tau[loud]) * 1e6:.0f} us vs calibrated {cal.offset_s * 1e6:.1f} us "
                            f"(physical swing is only +-{cal.delay_amp_s * 1e6:.0f} us): the tdoa calibration does not apply "
                            "to this channel mode")

    # level locator gates on the mixture as one source
    gates = {}
    for name, kw in (("default", {}), ("recommended", RECOMMENDED)):
        est = BearingEstimator(LocateConfig(band_freqs_hz=BANDS, **kw))
        j = 0
        for i in range(len(t)):
            while j < len(pt) and pt[j] <= t[i]:
                est.add_head_pose(float(pt[j]), float(yaw[j])); j += 1
            if j == 0:
                continue
            est.add_levels("mix", float(t[i]), np.stack([L0[i], L1[i]]), float(act[i]))
        e = est.estimate("mix")
        gates[name] = {"state": e.state, "world_deg": round(e.world_deg, 1), "conf": round(e.confidence, 2),
                       "resid_chi2": None if e.resid_chi2 is None else round(e.resid_chi2, 2),
                       "fit_gain": None if e.fit_gain is None else round(e.fit_gain, 3)}
    out["level_locator"] = gates
    return _report(out, verbose)


def _report(o: dict, verbose: bool) -> dict:
    if verbose:
        print(f"== {o['take']}: {o['channels']} ch @ {o['rate']} Hz, {o['seconds']:.1f} s, truth {o['truth']}")
        print(f"   rms dBFS {np.round(o['rms_db'], 1).tolist()}  after 100 Hz high-pass {np.round(o['rms_hp100_db'], 1).tolist()}")
        for k in ("ild_mean_db", "ild_std_db", "ch0_on_ch1_slope", "ild_yaw_corr", "ild_yaw_slope_db_per_deg", "coherence"):
            if k in o:
                print(f"   {k} (bands {', '.join(str(int(b)) for b in BANDS)} Hz): {o[k]}")
        if "delay" in o:
            print("   delay:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in o["delay"].items()})
        for k, v in o.get("level_locator", {}).items():
            print(f"   level locator [{k}]: {v}")
        for fl in o["flags"]:
            print("   FLAG:", fl)
        if not o["flags"]:
            print("   no flags")
    return o


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("takes", nargs="+", type=Path)
    a = ap.parse_args(argv)
    for t in a.takes:
        audit(t)


if __name__ == "__main__":
    main()
