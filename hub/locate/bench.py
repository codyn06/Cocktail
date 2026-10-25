"""Scoreboard: every direction method on the same labeled real takes, side by side.

    cd hub
    uv run python -m locate.bench recordings                      # every take under recordings/
    uv run python -m locate.bench recordings/t0-a recordings/t90-b --json bench.json

A take is a folder from quest/tools/record_session.py (audio.wav, pose.csv, meta.json with
truth_deg); a folder of takes is searched one level deep. Each take is scored by
real_eval.run, so a take that real_eval calls INVALID (no head motion, source lost in the
room) is listed but left out of the summary.

Methods: `level` (head-shadow levels, estimator.py), `tdoa` (inter-mic delay with the stored
calibration, pair.py) and `tdoa-loo` (delay, calibration refitted without the take being
scored). Takes inside the stored calibration are marked `*`: their `tdoa` score is
in-sample and flatters the method; `tdoa-loo` is the honest number.

Per method it prints:
* median / 90th-percentile final bearing error, overall and in front / at the side / behind;
* front/back right: share of takes whose final bearing is on the true side of the ear axis
  (side takes, within 15 deg of +-90, are not counted);
* seconds to lock: from the first estimate in the turn phase (real_eval drops the ~4 s of cue
  beeps and voice after the turn cue, so the clock starts when sound reaches the locator with
  the head moving) until the bearing comes within 25 deg of the truth and stays there;
* honest confidence: confidence is the posterior mass within 20 deg of the peak, so among
  estimates that say c, a share c should be within 20 deg of the truth. Reported per
  confidence bin over the timeline (consecutive estimates are correlated, so read the counts
  as optimistic), with the expected calibration error (ECE, 0 = honest).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pair import PairCalibration, fit_calibration
from .real_eval import circ_diff, delay_observations, run

METHODS = ("level", "tdoa", "tdoa-loo")
LOCK_TOL_DEG = 25.0
HIT_DEG = 20.0          # matches the +-20 deg window that defines BearingEstimate.confidence
SIDE_BAND_DEG = 15.0
TIMELINE_S = 0.25
CONF_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)
TARGETS = "targets: <=15 deg in front, <=25 deg behind, front/back right >=95%, locked within 3 s"


def find_takes(paths: list[Path]) -> list[Path]:
    takes = []
    for p in paths:
        if (p / "meta.json").exists():
            takes.append(p)
        elif p.is_dir():
            takes += sorted(d for d in p.iterdir() if (d / "meta.json").exists())
    return takes


def region(truth: float) -> str:
    a = abs(circ_diff(truth, 0.0))
    return "side" if abs(a - 90) < SIDE_BAND_DEG else ("front" if a < 90 else "behind")


def front_back_right(est: float, truth: float) -> bool | None:
    """True if est is on the same side of the ear axis as truth; None for side sources."""
    if region(truth) == "side":
        return None
    return bool((np.cos(np.radians(est)) > 0) == (np.cos(np.radians(truth)) > 0))


def seconds_to_lock(timeline, truth: float, t0: float, tol: float = LOCK_TOL_DEG) -> float | None:
    """Time from t0 until the bearing is within tol of truth and stays there; None if never."""
    pts = [(t, abs(circ_diff(w, truth))) for t, w, *_ in timeline if t >= t0]
    k = len(pts)
    while k > 0 and pts[k - 1][1] <= tol:
        k -= 1
    if k == len(pts):
        return None
    return max(pts[k][0] - t0, 0.0)


def calibration_table(points) -> tuple[list[tuple[float, float, float, int]], float | None]:
    """points: (confidence, hit). Returns per bin (lo, mean conf, hit rate, n) and the ECE."""
    if not points:
        return [], None
    c, h = np.array(points, float).T
    rows, ece = [], 0.0
    for lo, hi in zip(CONF_BINS[:-1], CONF_BINS[1:]):
        m = (c >= lo) & (c < hi)
        if m.any():
            rows.append((lo, float(c[m].mean()), float(h[m].mean()), int(m.sum())))
            ece += m.sum() / len(c) * abs(c[m].mean() - h[m].mean())
    return rows, float(ece)


def score_take(take: Path, method: str, calib: PairCalibration | None) -> dict:
    meta = json.loads((take / "meta.json").read_text())
    truth = float(meta["truth_deg"])
    r = run(take, truth, 12.0, verbose=False, method="tdoa" if method.startswith("tdoa") else "level",
            calib=calib, timeline_s=TIMELINE_S)
    tl = [(t, w, c, s) for t, w, c, s in r["timeline"] if r["go"] is None or t >= r["go"]]
    t0 = tl[0][0] if tl else 0.0
    return {
        "take": take.name, "method": method, "truth": truth, "region": region(truth),
        "est": r["world_deg"], "err": abs(circ_diff(r["world_deg"], truth)), "conf": r["confidence"],
        "state": r["state"], "valid": r["valid"], "snr_db": r["snr_db"], "yaw_std": r["yaw_std"],
        "fb_right": front_back_right(r["world_deg"], truth),
        "lock_s": seconds_to_lock(r["timeline"], truth, t0),
        "conf_points": [(c, abs(circ_diff(w, truth)) <= HIT_DEG) for _, w, c, _ in tl],
        "in_calib": method == "tdoa" and calib is not None and take.name in calib.takes,
    }


def loo_calibrations(takes: list[Path], device: str) -> dict[str, PairCalibration]:
    """For each take, the delay calibration fitted on all the other labeled takes."""
    obs = {}
    for tk in takes:
        _, y, tau, meta = delay_observations(tk)
        if len(tau) >= 20:
            obs[tk.name] = (y, tau, float(meta["truth_deg"]))
    out = {}
    for tk in takes:
        others = [o for name, o in obs.items() if name != tk.name]
        if len(others) >= 2:
            out[tk.name] = fit_calibration(others, device, [n for n in obs if n != tk.name])
    return out


def _q(xs, q):
    return f"{np.percentile(xs, q):5.1f}" if len(xs) else "    -"


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["valid"]]
    errs = [r["err"] for r in ok]
    fb = [r["fb_right"] for r in ok if r["fb_right"] is not None]
    locks = [r["lock_s"] for r in ok]
    table, ece = calibration_table([p for r in ok for p in r["conf_points"]])
    sure = [h for r in ok for c, h in r["conf_points"] if c >= 0.9]
    return {
        "n": len(ok), "n_invalid": len(rows) - len(ok),
        "median_err": float(np.median(errs)) if errs else None,
        "p90_err": float(np.percentile(errs, 90)) if errs else None,
        "by_region": {g: [r["err"] for r in ok if r["region"] == g] for g in ("front", "side", "behind")},
        "fb_right_rate": float(np.mean(fb)) if fb else None, "n_fb": len(fb),
        "lock_s": locks, "locked_3s_rate": float(np.mean([s is not None and s <= 3 for s in locks])) if locks else None,
        "calibration": table, "ece": ece,
        "hit_rate_when_90": float(np.mean(sure)) if sure else None, "n_90": len(sure),
        "final_mean_conf": float(np.mean([r["conf"] for r in ok])) if ok else None,
        "final_hit_rate": float(np.mean([r["err"] <= HIT_DEG for r in ok])) if ok else None,
    }


def print_report(rows: list[dict], summaries: dict[str, dict]) -> None:
    print(f"{'take':26s} {'method':9s} {'truth':>6s} {'est':>6s} {'err':>6s} {'conf':>5s} "
          f"{'state':19s} {'lock s':>6s}  note")
    for r in rows:
        note = "INVALID" if not r["valid"] else ""
        if r["in_calib"]:
            note = (note + " *in-calib").strip()
        lock = "never" if r["lock_s"] is None else f"{r['lock_s']:.1f}"
        print(f"{r['take'][:26]:26s} {r['method']:9s} {r['truth']:+6.0f} {r['est']:+6.0f} {r['err']:6.1f} "
              f"{r['conf']:5.2f} {r['state']:19s} {lock:>6s}  {note}")
    print(f"\n{TARGETS}")
    for m, s in summaries.items():
        print(f"\n== {m}: {s['n']} valid takes" + (f" ({s['n_invalid']} invalid, not scored)" if s["n_invalid"] else ""))
        if not s["n"]:
            continue
        print(f"  error (deg)          median {s['median_err']:5.1f}   p90 {s['p90_err']:5.1f}")
        for g, e in s["by_region"].items():
            if e:
                print(f"    {g:7s} n={len(e):<3d}      median {_q(e, 50)}   p90 {_q(e, 90)}")
        if s["fb_right_rate"] is not None:
            print(f"  front/back right     {100 * s['fb_right_rate']:5.1f}%  ({s['n_fb']} takes, side takes not counted)")
        locks = [x for x in s["lock_s"] if x is not None]
        print(f"  seconds to lock      median {_q(locks, 50)}   locked within 3 s: {100 * s['locked_3s_rate']:.0f}%  "
              f"never: {sum(x is None for x in s['lock_s'])}/{len(s['lock_s'])}")
        if s["ece"] is not None:
            print(f"  confidence honesty   ECE {s['ece']:.2f}   final: says {s['final_mean_conf']:.2f}, "
                  f"right {s['final_hit_rate']:.2f}"
                  + (f"   when >=0.9 sure: right {100 * s['hit_rate_when_90']:.0f}% of {s['n_90']}"
                     if s["n_90"] else "   never >=0.9 sure"))
            for lo, mc, hr, n in s["calibration"]:
                print(f"    conf {lo:.1f}-{min(lo + 0.2, 1.0):.1f}  says {mc:.2f}  right {hr:.2f}  (n={n})")


def bench(takes: list[Path], methods=METHODS, calib: PairCalibration | None = None):
    calib = calib or PairCalibration.default()
    loo = loo_calibrations(takes, calib.device) if "tdoa-loo" in methods else {}
    rows = []
    for tk in takes:
        for m in methods:
            if m == "tdoa-loo" and tk.name not in loo:
                continue
            rows.append(score_take(tk, m, loo.get(tk.name) if m == "tdoa-loo" else calib))
    summaries = {m: summarize([r for r in rows if r["method"] == m]) for m in methods}
    return rows, summaries


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m locate.bench", description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="take folders, or folders of takes")
    ap.add_argument("--methods", default=",".join(METHODS), help=f"comma list from {','.join(METHODS)}")
    ap.add_argument("--calib", type=Path, default=None, help="tdoa calibration JSON (default: quest3s)")
    ap.add_argument("--json", type=Path, default=None, help="also write all rows and summaries here")
    a = ap.parse_args(argv)
    methods = [m for m in a.methods.split(",") if m]
    if bad := [m for m in methods if m not in METHODS]:
        ap.error(f"unknown method(s): {', '.join(bad)}")
    takes = []
    for tk in find_takes(a.paths):
        if json.loads((tk / "meta.json").read_text()).get("truth_deg") is None:
            print(f"skip {tk}: no truth_deg in meta.json")
        else:
            takes.append(tk)
    if not takes:
        raise SystemExit("no labeled takes found")
    rows, summaries = bench(takes, methods, PairCalibration.load(a.calib) if a.calib else None)
    print_report(rows, summaries)
    if a.json:
        a.json.write_text(json.dumps({"rows": rows, "summaries": summaries}, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
