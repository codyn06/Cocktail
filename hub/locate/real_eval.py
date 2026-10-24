"""Score the head-movement locator on a real recording (quest/tools/record_session.py).

    cd hub
    uv run python -m locate.real_eval recordings/take1            # truth from meta.json
    uv run python -m locate.real_eval recordings/take1 --truth 90

Easiest: run quest/tools/run_locate_take.ps1, which records and scores a take in one go.

A take has: a few seconds of ambient room sound with the source OFF (used as a noise
profile and subtracted, so a noisy room is fine), then the source ON while you face it
at 0 deg, then a turn-your-head phase. Beeps in the headset mark the phases and are
excluded from scoring. The whole mic signal is treated as one source (no separation).
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

from .estimator import BearingEstimator, LocateConfig

BANDS = (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
WIN, HOP = 4096, 2400  # 85 ms window, 50 ms hop at 48 kHz


def load(take: Path):
    meta = json.loads((take / "meta.json").read_text())
    with wave.open(str(take / "audio.wav")) as w:
        ch, rate = w.getnchannels(), w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, ch).astype(np.float64) / 32768.0
    pose = np.loadtxt(take / "pose.csv", delimiter=",", skiprows=1)
    return meta, x, rate, pose


def yaw_deg(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]  # Unity: yaw about +y, + = turn right
    return np.degrees(np.arctan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x)))


def band_power(frame: np.ndarray, rate: int) -> np.ndarray:
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) ** 2
    f = np.fft.rfftfreq(len(frame), 1 / rate)
    out = []
    for c in BANDS:
        m = (f >= c / math.sqrt(2)) & (f < c * math.sqrt(2))
        out.append(spec[m].sum())
    return np.array(out)


def circ_diff(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


def run(take: Path, truth: float | None, within_db: float, verbose: bool = True) -> dict:
    meta, x, rate, pose = load(take)
    cue = meta.get("cue") or {}
    nu, go, stop = cue.get("noise_until", 0.0), cue.get("go"), cue.get("stop")
    mono = x.mean(axis=1)
    base = meta["first_audio_ts_us"]
    pt = (pose[:, 0] - base) / 1e6
    keep = pt >= 0
    pt, yaw = pt[keep], np.degrees(np.unwrap(np.radians(yaw_deg(pose[keep, 1:]))))
    base_win = (pt >= nu) & (pt < (go - 0.5)) if go else (pt < pt[0] + 1.0)
    if base_win.sum() < 5:
        base_win = pt < pt[0] + 1.0
    yaw = yaw - np.median(yaw[base_win])  # 0 deg = how the head faced before turning

    starts = np.arange(0, len(mono) - WIN, HOP)
    pwr = np.array([band_power(mono[s:s + WIN], rate) for s in starts])
    t_audio = (starts + WIN / 2) / rate
    if nu:
        noise = pwr[t_audio < nu - 0.2].mean(axis=0)
        pwr = np.maximum(pwr - noise, 0.1 * pwr)  # spectral subtraction of the room
    lev = 10 * np.log10(pwr + 1e-12)
    bb = 10 * np.log10(pwr.sum(axis=1) + 1e-12)

    in_win = t_audio >= (nu + 0.5 if nu else 0.0)
    if stop:
        in_win &= t_audio <= stop - 0.3
    if go and stop:  # drop frames that contain the cue beeps
        for b0, n in ((go, 2), (stop, 3)):
            in_win &= ~((t_audio >= b0 - 0.2) & (t_audio <= b0 + 0.3 * n + 0.4))
    cut = np.percentile(bb[in_win], 95) - within_db if in_win.any() else np.inf
    weight = (in_win & (bb > cut)).astype(float)

    est = BearingEstimator(LocateConfig(mic_az_deg=(0.0,), band_freqs_hz=BANDS))
    j, timeline = 0, []
    for i, t in enumerate(t_audio):
        while j < len(pt) and pt[j] <= t:
            est.add_head_pose(float(pt[j]), float(yaw[j]))
            j += 1
        if j == 0:  # audio can start before the first pose sample
            continue
        est.add_levels("src", float(t), lev[i], float(weight[i]))
        if i % int(5 / (HOP / rate)) == 0 and i > 0 and weight[i] > 0:
            e = est.estimate("src", float(t))
            timeline.append((t, e.world_deg, e.confidence, e.state))
    e = est.estimate("src")

    motion = (pt >= go) & (pt <= stop) if (go and stop) else np.ones_like(pt, bool)
    res = {
        "seconds": len(mono) / rate, "yaw_min": float(yaw[motion].min()), "yaw_max": float(yaw[motion].max()),
        "yaw_std": float(yaw[motion].std()), "active_frac": float(weight[in_win].mean()) if in_win.any() else 0.0,
        "world_deg": e.world_deg, "confidence": e.confidence, "state": e.state,
        "alt_head_deg": e.alt_bearing_deg, "truth": truth,
    }
    if verbose:
        print(f"recording: {res['seconds']:.1f} s, {meta['channels']} ch @ {rate} Hz, {meta['pose_samples']} pose samples"
              + (f"; room noise profile from first {nu:.1f} s subtracted" if nu else ""))
        print(f"head yaw during the turn phase: {res['yaw_min']:+.0f} .. {res['yaw_max']:+.0f} deg (std {res['yaw_std']:.1f}); "
              f"source active {100 * res['active_frac']:.0f}% of scored frames")
        for t, w_, c, s in timeline:
            print(f"  t={t:5.1f}s  world bearing {w_:+7.1f} deg  conf {c:.2f}  {s}")
        print(f"FINAL: world bearing {e.world_deg:+.1f} deg, confidence {e.confidence:.2f}, state {e.state}, "
              f"sigma {e.sigma_deg:.1f} deg")
    if truth is not None:
        err = abs(circ_diff(e.world_deg, truth))
        checks = [
            ("enough head motion (std >= 8 deg)", res["yaw_std"] >= 8, f"{res['yaw_std']:.1f}"),
            ("source heard over the room (active >= 30% of frames)", res["active_frac"] >= 0.30, f"{100 * res['active_frac']:.0f}%"),
            ("correct hemisphere (front/back and left/right)", err < 90, f"err {err:.0f} deg"),
            ("bearing error <= 25 deg", err <= 25, f"err {err:.0f} deg"),
            ("state is tracking/locked (not ambiguous)", e.state in ("tracking", "locked"), e.state),
        ]
        print(f"\ntruth {truth:+.0f} deg")
        for name, ok, val in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {val}")
        res["pass"] = all(ok for _, ok, _ in checks)
        print("RESULT:", "PASS" if res["pass"] else "FAIL")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("--truth", type=float, default=None)
    ap.add_argument("--active-within-db", type=float, default=12.0,
                    help="frames within this many dB of the loudest count as 'source active'")
    a = ap.parse_args()
    take = Path(a.take)
    truth = a.truth
    if truth is None:
        truth = json.loads((take / "meta.json").read_text()).get("truth_deg")
    r = run(take, truth, a.active_within_db)
    raise SystemExit(0 if r.get("pass", True) else 1)


if __name__ == "__main__":
    main()
