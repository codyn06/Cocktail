"""Why does the mic-pair delay method miss on a take? Checks the things the model assumes.

    cd hub
    uv run python -m locate.diagnose recordings/t0-a recordings/t90-b ...

On the real 3S (2026-10-24) the delays fit the model with a residual of 13.7 us against a
swing of only 14.6 us, while per-frame delay noise is ~1 us (simulated) and even a very
reverberant simulated room leaves ~2 us. So something the model leaves out dominates; the
candidates this checks are timing (audio and head pose out of step, so delays are matched
to the wrong yaw), device constants that differ between takes, and the mic channels
themselves. Per take this prints:

* fit: the take's own sinusoid (truth known, D, axis and offset free) and its residual;
* timing: the pose-vs-audio shift that minimises the residual (+ = pose timestamps late),
  and the residual there. A large drop at a non-zero shift means a timing problem;
* clock: audio seconds (sample count) vs pose time span, a sign of dropped audio chunks;
* channels: median inter-channel coherence and the whole-take integer-sample lag.
With several takes it also compares one shared calibration against the per-take fits:
much worse shared than per-take means the device constants differ between takes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pair import fit_calibration, frame_delay_phase
from .real_eval import WIN, _active_frames

SHIFTS_S = np.round(np.arange(-1.5, 1.5001, 0.02), 3)


def take_delays(take: Path):
    """(frame times s, pose times s, yaw deg, delays s, sigmas s, x, rate, meta) for active frames."""
    t, w, x, rate, pt, yaw, meta = _active_frames(take, 12.0)
    obs = [(ti,) + frame_delay_phase(x[int(ti * rate - WIN / 2):int(ti * rate + WIN / 2), :2], rate)
           for ti, wi in zip(t, w) if wi > 0]
    obs = np.array([o for o in obs if np.isfinite(o[2])]).reshape(-1, 3)
    return obs[:, 0], pt, yaw, obs[:, 1], obs[:, 2], x, rate, meta


def own_fit(y, tau, truth):
    """The take's own sinusoid with D, axis and offset free (truth known)."""
    cal = fit_calibration([(y, tau, truth)])
    return cal, residual_at(cal, y, tau, truth)


def residual_at(cal, y, tau, truth) -> float:
    r = tau - cal.predict(truth, np.asarray(y))
    return 1.4826 * float(np.median(np.abs(r - np.median(r))))


def best_shift(t, pt, yaw, tau, truth):
    """Pose-vs-audio shift (s, + = pose timestamps late) minimising the own-fit residual."""
    res = np.array([own_fit(np.interp(t + s, pt, yaw), tau, truth)[1] for s in SHIFTS_S])
    k = int(np.argmin(res))
    return float(SHIFTS_S[k]), float(res[k]), res


def channel_stats(x, rate):
    a, b = x[:, 0] - x[:, 0].mean(), x[:, 1] - x[:, 1].mean()
    n = 1 << (len(a) * 2 - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    lags = np.r_[0:6, -5:0]
    lag = int(lags[np.argmax(cc[lags])])
    seg = 4096
    k = len(a) // seg
    A = np.fft.rfft(a[: k * seg].reshape(k, seg) * np.hanning(seg), axis=1)
    B = np.fft.rfft(b[: k * seg].reshape(k, seg) * np.hanning(seg), axis=1)
    coh = np.abs((A * np.conj(B)).mean(0)) ** 2 / ((np.abs(A) ** 2).mean(0) * (np.abs(B) ** 2).mean(0) + 1e-30)
    f = np.fft.rfftfreq(seg, 1 / rate)
    return lag, float(np.median(coh[(f > 300) & (f < 4000)])), float(np.median(coh[(f > 4000) & (f < 12000)]))


def diagnose(takes: list[Path]) -> list[dict]:
    out = []
    for tk in takes:
        t, pt, yaw, tau, sig, x, rate, meta = take_delays(tk)
        if meta.get("truth_deg") is None:
            print(f"{tk.name}: no truth_deg in meta.json, skipped")
            continue
        truth = float(meta["truth_deg"])
        if len(tau) < 20:
            print(f"{tk.name}: too few active frames ({len(tau)})")
            continue
        y0 = np.interp(t, pt, yaw)
        cal, res0 = own_fit(y0, tau, truth)
        shift, res_s, _ = best_shift(t, pt, yaw, tau, truth)
        lag, coh_lo, coh_hi = channel_stats(x, rate)
        d = {"take": tk.name, "truth": truth, "n": len(tau), "D_us": cal.delay_amp_s * 1e6, "axis": cal.axis_deg,
             "offset_us": cal.offset_s * 1e6, "res_us": res0 * 1e6, "sigma_us": float(np.median(sig)) * 1e6,
             "shift_s": shift, "res_shift_us": res_s * 1e6, "audio_s": len(x) / rate,
             "pose_span_s": float(pt[-1] - pt[0]), "lag": lag, "coh_lo": coh_lo, "coh_hi": coh_hi,
             "y": y0, "tau": tau}
        out.append(d)
        print(f"\n{tk.name}  truth {truth:+.0f}  ({len(tau)} active frames)")
        print(f"  fit       D {d['D_us']:5.1f} us  axis {d['axis']:+6.1f}  offset {d['offset_us']:+5.1f} us  "
              f"residual {d['res_us']:5.1f} us  (median frame sigma {d['sigma_us']:.1f} us)")
        print(f"  timing    best pose shift {shift:+.2f} s -> residual {d['res_shift_us']:5.1f} us"
              + ("   <-- timing problem?" if res_s < 0.6 * res0 and abs(shift) > 0.05 else ""))
        print(f"  clock     audio {d['audio_s']:.2f} s by sample count, pose span {d['pose_span_s']:.2f} s")
        print(f"  channels  lag {lag:+d} samples  coherence 0.3-4 kHz {coh_lo:.2f}, 4-12 kHz {coh_hi:.2f}")
    if len(out) >= 2:
        shared = fit_calibration([(d["y"], d["tau"], d["truth"]) for d in out])
        print(f"\nshared calibration: D {shared.delay_amp_s * 1e6:.1f} us, axis {shared.axis_deg:+.1f}, "
              f"offset {shared.offset_s * 1e6:+.1f} us")
        for d in out:
            d["res_shared_us"] = residual_at(shared, d["y"], d["tau"], d["truth"]) * 1e6
            print(f"  {d['take']:26s} residual shared {d['res_shared_us']:5.1f} us vs own fit {d['res_us']:5.1f} us")
    for d in out:
        del d["y"], d["tau"]
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m locate.diagnose", description=__doc__.split("\n")[0])
    ap.add_argument("takes", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    res = diagnose([t for t in a.takes if (t / "meta.json").exists()])
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
