"""Score the head-movement locator on a real recording (quest/tools/record_session.py).

    cd hub
    uv run python -m locate.real_eval recordings/take1            # uses truth from meta.json
    uv run python -m locate.real_eval recordings/take1 --truth 90

Setup for a take: one steady sound source (a phone or speaker playing pink noise, music or
a podcast) at a known angle from where you face when you start, 1-2 m away; sit still for
~3 s facing it at 0 deg, then turn your head smoothly left and right (about +-40 deg) for
20 s. The scorer prints PASS/FAIL against explicit thresholds.

The recording has no separation step: the whole mic signal is treated as one source, so
use a source that dominates the room.
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


def band_levels(frame: np.ndarray, rate: int) -> np.ndarray:
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) ** 2
    f = np.fft.rfftfreq(len(frame), 1 / rate)
    out = []
    for c in BANDS:
        m = (f >= c / math.sqrt(2)) & (f < c * math.sqrt(2))
        out.append(10 * math.log10(spec[m].sum() + 1e-12))
    return np.array(out)


def circ_diff(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


def run(take: Path, truth: float | None, within_db: float, verbose: bool = True) -> dict:
    meta, x, rate, pose = load(take)
    mono = x.mean(axis=1)
    base = meta["first_audio_ts_us"]
    pt = (pose[:, 0] - base) / 1e6
    keep = pt >= 0
    pt, yaw = pt[keep], np.unwrap(np.radians(yaw_deg(pose[keep, 1:])), axis=0)
    yaw = np.degrees(yaw)
    yaw = yaw - np.median(yaw[pt < pt[0] + 1.0])  # 0 deg = how the head faced at the start

    starts = np.arange(0, len(mono) - WIN, HOP)
    lev = np.array([band_levels(mono[s:s + WIN], rate) for s in starts])
    t_audio = (starts + WIN / 2) / rate
    bb = 10 * np.log10(np.array([np.mean(mono[s:s + WIN] ** 2) for s in starts]) + 1e-15)
    # a steady source barely varies, so 'active' = within `within_db` of the loudest frames (drops silence)
    weight = (bb > np.percentile(bb, 95) - within_db).astype(float)

    est = BearingEstimator(LocateConfig(mic_az_deg=(0.0,), band_freqs_hz=BANDS))
    j, timeline = 0, []
    for i, t in enumerate(t_audio):
        while j < len(pt) and pt[j] <= t:
            est.add_head_pose(float(pt[j]), float(yaw[j]))
            j += 1
        if j == 0:  # audio can start before the first pose sample
            continue
        est.add_levels("src", float(t), lev[i], float(weight[i]))
        if i % int(5 / (HOP / rate)) == 0 and i > 0:
            e = est.estimate("src", float(t))
            timeline.append((t, e.world_deg, e.confidence, e.state))
    e = est.estimate("src")

    res = {
        "seconds": len(mono) / rate, "yaw_min": float(yaw.min()), "yaw_max": float(yaw.max()),
        "yaw_std": float(yaw.std()), "active_frac": float(weight.mean()),
        "world_deg": e.world_deg, "confidence": e.confidence, "state": e.state,
        "alt_head_deg": e.alt_bearing_deg, "truth": truth,
    }
    if verbose:
        print(f"recording: {res['seconds']:.1f} s, {meta['channels']} ch @ {rate} Hz, {meta['pose_samples']} pose samples")
        print(f"head yaw: {res['yaw_min']:+.0f} .. {res['yaw_max']:+.0f} deg (std {res['yaw_std']:.1f}); "
              f"source active {100 * res['active_frac']:.0f}% of frames")
        for t, w_, c, s in timeline:
            print(f"  t={t:5.1f}s  world bearing {w_:+7.1f} deg  conf {c:.2f}  {s}")
        print(f"FINAL: world bearing {e.world_deg:+.1f} deg, confidence {e.confidence:.2f}, state {e.state}, "
              f"sigma {e.sigma_deg:.1f} deg")
    if truth is not None:
        err = abs(circ_diff(e.world_deg, truth))
        checks = [
            ("enough head motion (std >= 8 deg)", res["yaw_std"] >= 8, f"{res['yaw_std']:.1f}"),
            ("source heard (active >= 30% of frames)", res["active_frac"] >= 0.30, f"{100 * res['active_frac']:.0f}%"),
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
    ap.add_argument("--active-within-db", type=float, default=12.0, help="frames within this many dB of the loudest count as 'source active'")
    a = ap.parse_args()
    take = Path(a.take)
    truth = a.truth
    if truth is None:
        truth = json.loads((take / "meta.json").read_text()).get("truth_deg")
    r = run(take, truth, a.active_within_db)
    raise SystemExit(0 if r.get("pass", True) else 1)


if __name__ == "__main__":
    main()
