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
from scipy import signal

from .estimator import BearingEstimator, LocateConfig
from .pair import PairCalibration, PairConfig, PairLocator, fit_calibration, frame_delay_phase
from .tdoa import gcc_phat

BANDS = (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
MIN_SNR_DB = 6.0  # the source must be at least this much louder than the room, or the take is invalid
WIN, HOP = 4096, 2400  # 85 ms window, 50 ms hop at 48 kHz
LIVE_RATE, LIVE_WIN = 16000, 1024  # how the live hub measures delays (locate/live.py TDOA_WIN)


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


def frame_delay(x2: np.ndarray, rate: int, max_tau_s: float) -> tuple[float, float]:
    """Inter-mic delay (s, + = channel 0 later) of one frame and its GCC peak sharpness."""
    lags, cc = gcc_phat(x2[:, 0], x2[:, 1], rate, max_tau=max_tau_s, interp=32)
    k = int(np.argmax(cc))
    return float(lags[k]), float(cc[k] / (np.median(np.abs(cc)) + 1e-12))


def delay_audio(x: np.ndarray, rate: int, rate_hz: int | None = None, win: int | None = None):
    """The two mic channels at the rate delays are measured at, and a frame cutter there:
    returns (frame(t) -> (win, 2) samples centred on t seconds, rate)."""
    x2, r, n = x[:, :2], rate, win or WIN
    if rate_hz and rate_hz != rate:
        x2, r = signal.resample_poly(x2, rate_hz, rate, axis=0), rate_hz  # zero-phase: timing kept
        n = win or round(WIN * rate_hz / rate)
    return (lambda t: x2[max(int(t * r - n / 2), 0):int(t * r + n / 2)]), r


def delay_observations(take: Path, within_db: float = 12.0, max_tau_s: float = 0.3e-3,
                       rate_hz: int | None = None, win: int | None = None):
    """Frames of a take where the source is on and the GCC peak is in the sharper half:
    returns (frame times s, head yaw deg at those times, delays s, meta). For calibration.
    rate_hz/win: measure the delays at that rate and frame length (default: the take's, WIN)."""
    t, w, x, rate, pt, yaw, meta = _active_frames(take, within_db)
    frame, r = delay_audio(x, rate, rate_hz, win)
    obs = [(ti,) + frame_delay(frame(ti), r, max_tau_s) for ti, wi in zip(t, w) if wi > 0]
    if not obs:
        return np.array([]), np.array([]), np.array([]), meta
    ot, tau, sharp = np.array(obs).T
    keep = sharp >= np.median(sharp)
    return ot[keep], np.interp(ot[keep], pt, yaw), tau[keep], meta


def phase_observations(take: Path, within_db: float = 12.0, rate_hz: int | None = None, win: int | None = None):
    """Like delay_observations, with phase-slope delays (pair.frame_delay_phase) and their own
    standard errors, for every frame where the source is on: (times, yaw, delays, sigmas, meta)."""
    t, w, x, rate, pt, yaw, meta = _active_frames(take, within_db)
    frame, r = delay_audio(x, rate, rate_hz, win)
    sub = min(512, (win or WIN) // 4)
    obs = [(ti,) + frame_delay_phase(frame(ti), r, sub=sub) for ti, wi in zip(t, w) if wi > 0]
    obs = [o for o in obs if np.isfinite(o[2])]
    if not obs:
        return np.array([]), np.array([]), np.array([]), np.array([]), meta
    ot, tau, sig = np.array(obs).T
    return ot, np.interp(ot, pt, yaw), tau, sig, meta


def is_valid(take: Path, within_db: float = 12.0) -> bool:
    """real_eval's validity rule (head motion, source above the room) without running a locator."""
    return run(take, None, within_db, verbose=False, method="frames")["valid"]


def _active_frames(take: Path, within_db: float):
    f = run(take, None, within_db, verbose=False, method="frames", keep_frames=True)["_frames"]
    return tuple(f[k] for k in ("t", "weight", "x", "rate", "pt", "yaw", "meta"))


def run(take: Path, truth: float | None, within_db: float, verbose: bool = True, method: str = "level",
        calib: PairCalibration | None = None, keep_frames: bool = False, timeline_s: float = 5.0,
        pair_cfg: PairConfig | None = None) -> dict:
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
    # How far the source stands above the room: room-only frames (before noise_until) vs source-on
    # frames while the head is still (before the turn cue), measured before noise subtraction.
    # A source barely above the room makes the level-vs-yaw cue track the room's own noise instead.
    snr_db = None
    if nu and go:
        room = pwr[t_audio < nu - 0.2].sum(axis=1).mean()
        on = pwr[(t_audio > nu + 0.5) & (t_audio < go - 0.5)].sum(axis=1).mean()
        snr_db = float(10 * np.log10(max(on - room, 1e-12) / (room + 1e-12)))
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
            in_win &= ~((t_audio >= b0 - 0.5) & (t_audio <= b0 + 0.3 * n + 3.5))  # beeps + spoken cue
    cut = np.percentile(bb[in_win], 95) - within_db if in_win.any() else np.inf
    weight = (in_win & (bb > cut)).astype(float)
    if method == "frames":  # only which frames count (for calibration), no locator
        motion = (pt >= go) & (pt <= stop) if (go and stop) else np.ones_like(pt, bool)
        valid = (yaw[motion] if motion.any() else yaw).std() >= 8 and (snr_db is None or snr_db >= MIN_SNR_DB)
        return {"_frames": {"t": t_audio, "weight": weight, "x": x, "rate": rate, "pt": pt, "yaw": yaw, "meta": meta},
                "valid": bool(valid)}

    if method == "tdoa":
        # Inter-mic delay + head yaw (pair.py): for closely spaced mics such as the Quest 3S.
        calib = calib or PairCalibration.default()
        est = PairLocator(calib, pair_cfg)
        phase = calib.delay_method == "phase"
        frame, r = delay_audio(x, rate, calib.rate_hz, calib.win)  # as the calibration was fitted
        sub = min(512, calib.win // 4)
        delay = ((lambda f: frame_delay_phase(f, r, sub=sub)) if phase
                 else (lambda f: frame_delay(f, r, calib.max_tau_s)))
        d = [delay(frame(t)) if weight[i] > 0 else (0.0, np.inf if phase else 0.0) for i, t in enumerate(t_audio)]
        sharp = np.array([v[1] for v in d])
        on = weight > 0
        if phase:  # the more precise half (lower sigma), each weighted by its own sigma
            use = on & np.isfinite(sharp) & (sharp <= (np.median(sharp[on]) if on.any() else -np.inf))
        else:  # sharper half of the GCC peaks
            use = (weight > 0) & (sharp >= (np.median(sharp[weight > 0]) if (weight > 0).any() else np.inf))
    else:
        est = BearingEstimator(LocateConfig(mic_az_deg=(0.0,), band_freqs_hz=BANDS))
    j, timeline = 0, []
    for i, t in enumerate(t_audio):
        while j < len(pt) and pt[j] <= t:
            est.add_head_pose(float(pt[j]), float(yaw[j]))
            j += 1
        if j == 0:  # audio can start before the first pose sample
            continue
        if method == "tdoa":
            if use[i]:
                est.add_delay("src", float(t), d[i][0], 1.0, sigma_s=d[i][1] if phase else None)
        else:
            est.add_levels("src", float(t), lev[i], float(weight[i]))
        if i % max(1, round(timeline_s / (HOP / rate))) == 0 and i > 0 and weight[i] > 0:
            e = est.estimate("src", float(t))
            timeline.append((t, e.world_deg, e.confidence, e.state))
    e = est.estimate("src")

    motion = (pt >= go) & (pt <= stop) if (go and stop) else np.ones_like(pt, bool)
    if not motion.any():  # no head pose during the turn phase (clock/recording problem)
        motion = np.ones_like(pt, bool)
        print("WARNING: no head-pose samples inside the turn phase; recording timing looks wrong")
    res = {
        "seconds": len(mono) / rate, "yaw_min": float(yaw[motion].min()), "yaw_max": float(yaw[motion].max()),
        "yaw_std": float(yaw[motion].std()), "active_frac": float(weight[in_win].mean()) if in_win.any() else 0.0,
        "world_deg": e.world_deg, "confidence": e.confidence, "state": e.state,
        "alt_head_deg": e.alt_bearing_deg, "truth": truth, "snr_db": snr_db, "method": method,
        "go": go, "stop": stop, "timeline": timeline,
    }
    if keep_frames:
        res["_frames"] = {"t": t_audio, "weight": weight, "x": x, "rate": rate, "pt": pt, "yaw": yaw, "meta": meta}
    if verbose:
        print(f"recording: {res['seconds']:.1f} s, {meta['channels']} ch @ {rate} Hz, {meta['pose_samples']} pose samples"
              + (f"; room noise profile from first {nu:.1f} s subtracted" if nu else ""))
        print(f"head yaw during the turn phase: {res['yaw_min']:+.0f} .. {res['yaw_max']:+.0f} deg (std {res['yaw_std']:.1f}); "
              f"source active {100 * res['active_frac']:.0f}% of scored frames"
              + (f"; source {snr_db:+.1f} dB above the room" if snr_db is not None else ""))
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
        if snr_db is not None:
            checks.insert(1, (f"source clearly above the room (>= {MIN_SNR_DB:.0f} dB)", snr_db >= MIN_SNR_DB,
                              f"{snr_db:+.1f} dB"))
        if verbose:
            print(f"\ntruth {truth:+.0f} deg")
            for name, ok, val in checks:
                print(f"  {'PASS' if ok else 'FAIL'}  {name}: {val}")
        res["pass"] = all(ok for _, ok, _ in checks)
        # A take with no head motion or a source lost in the room says nothing about the locator.
        res["valid"] = res["yaw_std"] >= 8 and (snr_db is None or snr_db >= MIN_SNR_DB)
        if not verbose:
            pass
        elif not res["valid"]:
            print("RESULT: INVALID TAKE (not a locator result: "
                  + ("source too quiet over the room; move it closer or turn it up" if snr_db is not None
                     and snr_db < MIN_SNR_DB else "not enough head motion") + ")")
        else:
            print("RESULT:", "PASS" if res["pass"] else "FAIL")
    return res


def observations(take: Path, delay: str = "gcc", rate_hz: int | None = None, win: int | None = None):
    """(yaw, delays, [sigmas,] meta) of a labeled take for fit_calibration."""
    if delay == "phase":
        _, y, tau, sig, meta = phase_observations(take, rate_hz=rate_hz, win=win)
        return (y, tau, sig), meta
    _, y, tau, meta = delay_observations(take, rate_hz=rate_hz, win=win)
    return (y, tau), meta


def framing(rate_hz: int | None) -> tuple[int, int]:
    """(rate, frame length) delays are measured at for a requested rate: the live hub's framing
    at 16 kHz, the take's own (48 kHz, WIN) otherwise."""
    if rate_hz == LIVE_RATE:
        return LIVE_RATE, LIVE_WIN
    return (rate_hz, round(WIN * rate_hz / 48000)) if rate_hz else (48000, WIN)


def fit_at(obs, device, names, rate_hz: int | None) -> PairCalibration:
    cal = fit_calibration(obs, device, names)
    cal.rate_hz, cal.win = framing(rate_hz)
    return cal


def calibrate(takes: list[Path], out: Path | None, device: str = "quest3s", delay: str = "gcc",
              rate_hz: int | None = None) -> PairCalibration:
    """Fit the mic-pair constants from labeled takes and report a leave-one-out check.
    rate_hz: measure delays at this rate (16000 = as the live hub does)."""
    rate_hz, win = framing(rate_hz)
    obs = []
    for tk in takes:
        (y, tau, *sig), meta = observations(tk, delay, rate_hz, win)
        if len(tau) < 20:
            print(f"skip {tk.name}: too few usable frames ({len(tau)})")
            continue
        obs.append((tk, y, tau, float(meta["truth_deg"]), *sig))
    cal = fit_at([o[1:] for o in obs], device, [o[0].name for o in obs], rate_hz)
    print(f"calibration from {len(obs)} takes: delay amplitude {cal.delay_amp_s * 1e6:.1f} us "
          f"(spacing ~{cal.delay_amp_s * 343 * 100:.2f} cm), axis {cal.axis_deg:+.1f} deg, "
          f"offset {cal.offset_s * 1e6:+.1f} us, residual {cal.noise_s * 1e6:.1f} us"
          + (f", model error {cal.model_s * 1e6:.1f} us" if cal.model_s is not None else ""))
    if len(obs) >= 3:
        print("leave-one-out (calibrate on the others, locate the held-out take):")
        errs = []
        for i, (tk, _, _, truth, *_) in enumerate(obs):
            sub = fit_at([o[1:] for k, o in enumerate(obs) if k != i], device, None, rate_hz)
            r = run(tk, truth, 12.0, verbose=False, method="tdoa", calib=sub)
            err = abs(circ_diff(r["world_deg"], truth))
            errs.append(err)
            print(f"  {tk.name:22s} truth {truth:+5.0f} -> {r['world_deg']:+5.0f}  error {err:5.1f} deg  "
                  f"conf {r['confidence']:.2f} {r['state']}")
        print(f"  median error {np.median(errs):.0f} deg; within 25 deg: {sum(e <= 25 for e in errs)}/{len(errs)}")
    if out:
        cal.save(out)
        print(f"saved {out}")
    return cal


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        ap = argparse.ArgumentParser(prog="real_eval calibrate",
                                     description="fit mic-pair (tdoa) constants from labeled takes")
        ap.add_argument("takes", nargs="+")
        ap.add_argument("--out", type=Path, default=None, help="e.g. locate/calib/quest3s.json")
        ap.add_argument("--device", default="quest3s")
        ap.add_argument("--delay", choices=["gcc", "phase"], default="gcc",
                        help="gcc: GCC-PHAT peak per frame; phase: cross-spectrum phase slope with its own sigma")
        ap.add_argument("--rate", type=int, default=None,
                        help="measure delays at this sample rate; 16000 = as the live hub does (use for live)")
        a = ap.parse_args(sys.argv[2:])
        calibrate([Path(t) for t in a.takes], a.out, a.device, a.delay, a.rate)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("--truth", type=float, default=None)
    ap.add_argument("--active-within-db", type=float, default=12.0,
                    help="frames within this many dB of the loudest count as 'source active'")
    ap.add_argument("--method", choices=["level", "tdoa"], default="level",
                    help="level: head-shadow level model (estimator.py); tdoa: inter-mic delay (pair.py)")
    ap.add_argument("--calib", type=Path, default=None, help="tdoa calibration JSON (default: quest3s)")
    ap.add_argument("--adapt", action="store_true", help="tdoa: per-source delay swing and offset (PairConfig.adapt)")
    a = ap.parse_args()
    take = Path(a.take)
    truth = a.truth
    if truth is None:
        truth = json.loads((take / "meta.json").read_text()).get("truth_deg")
    r = run(take, truth, a.active_within_db, method=a.method,
            calib=PairCalibration.load(a.calib) if a.calib else None, pair_cfg=PairConfig(adapt=a.adapt))
    raise SystemExit(0 if r.get("pass", True) else 1)


if __name__ == "__main__":
    main()
