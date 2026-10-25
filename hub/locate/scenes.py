"""Scoreboard for the live two-talker path on SIMULATED Quest 3S scenes (no GPU, no recordings).

    cd hub
    uv run python -m locate.scenes sessions/scenes --calib sessions/pairsim-calib16.json
    uv run python -m locate.scenes sessions/scenes --calib ... --adapt --json scenes.json
    uv run python -m locate.scenes sessions/held --calib ... --seed0 500 --n 6   # held-out seeds

Each scene is a mockclient session (locate.pairsim.write_session) with two speech-like talkers at
fixed world bearings, written into <dir>/sceneNN on first use. Every scene is run through the
live code (LiveSeparator masks, per-stream mic images, LiveLocator, PairLocator) with the TRUE
stems as the separator, so separation error is taken out: this scores locating each separated
voice, not the separation model. Calibrate at the live rate first
(`python -m locate.real_eval calibrate sessions/pairsim/sim-* --rate 16000 --out ...`).

Same metrics as locate.bench, per talker:
* final bearing error: median / p90 overall and in front / side / behind;
* front/back right (talkers within 15 deg of the ear axis are not counted);
* both talkers within 25 deg, counted per scene (the "two speakers within 25 deg" target);
* seconds to lock: from the start of head motion (meta head_go_s, 1 s) until the bearing is within
  25 deg in a committed state (tracking or locked) and stays so; share committed (tracking or locked) within 3 s;
* honest confidence: every estimate on the timeline is a (confidence, hit) point, hit = error
  <= 20 deg (the window that defines the confidence), as a reliability table plus the ECE.
  Consecutive estimates are correlated, so read the counts as optimistic.

Everything here is simulated: it says nothing about the real headset.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from mockclient import Session
from separate.live import LiveSeparator

from . import bench, pairsim, score_live
from .estimator import wrap180
from .pair import PairCalibration, PairConfig, sim_tuned

# Two-talker scenes covering front, side and behind, same-side and opposite-side pairs.
PAIRS = [(-30, 40), (0, 120), (-90, 60), (150, -120), (20, -160), (90, -60),
         (-45, 135), (170, 60), (-120, -20), (75, -15), (110, -100), (0, 180)]
SCENE_SECONDS = 40.0
HEAD_GO_S = 1.0       # pairsim starts the head turning at 1 s
BOTH_TOL_DEG = 25.0


def ensure_scene(root: Path, k: int, pair, seed: int, secs: float = SCENE_SECONDS) -> Path:
    """Write scene k under root unless the same scene (angles, seed, length) is already there."""
    d = root / f"scene{k:02d}"
    want = [float(a) for a in pair]
    if (d / "session.json").exists():
        meta = Session.load(d).meta
        if ([q["bearing_deg"] for q in meta["sources"]] == want and meta.get("seed") == seed
                and abs(meta.get("duration_s", 0) - secs) < 1e-6):
            return d
    pairsim.write_session(d, want, seed=seed, secs=secs)
    return d


def score_scene(d: Path, calib: PairCalibration, pair_cfg: PairConfig | None = None,
                raw: list | None = None) -> list[dict]:
    """One row per talker: truth, final estimate, and the whole timeline of (t, world, conf, state).
    raw: if a list, appended after every hop with the device time and loc.sources() as JSON dicts,
    the locator output the hub's tracker would see (see track.jitter)."""
    s = Session.load(d)
    live = LiveSeparator(score_live.oracle_separator(s), None)
    step = None if raw is None else (lambda loc: raw.append(
        {"t_us": loc.now_us, "sources": [x.model_dump(mode="json") for x in loc.sources()]}))
    est, trace, loc = score_live.run(s, live, "tdoa", calib, pair_cfg, on_step=step)
    streams = score_live.assign_streams(s, est)
    final = loc.estimates()
    go = float(s.meta.get("head_go_s", HEAD_GO_S))
    rows = []
    for sid, q in zip(streams, s.meta["sources"]):
        truth, e = float(q["bearing_deg"]), final[sid]
        tl = [(t, w, c, st) for t, w, st, c, _alt in trace[sid]]
        rows.append({
            "scene": d.name, "talker": q["name"], "stream": sid, "truth": truth, "region": bench.region(truth),
            "est": e.world_deg, "err": abs(float(wrap180(e.world_deg - truth))), "conf": e.confidence,
            "state": e.state, "fb_right": bench.front_back_right(e.world_deg, truth),
            "lock_s": bench.seconds_to_lock(tl, truth, go),
            "conf_points": [(c, abs(float(wrap180(w - truth))) <= bench.HIT_DEG)
                            for _, w, c, st in tl if st != "unobserved"],
            "timeline": [(round(t, 3), round(w, 2), round(c, 4), st) for t, w, c, st in tl],
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    errs = [r["err"] for r in rows]
    fb = [r["fb_right"] for r in rows if r["fb_right"] is not None]
    locks = [r["lock_s"] for r in rows]
    table, ece = bench.calibration_table([p for r in rows for p in r["conf_points"]])
    sure = [h for r in rows for c, h in r["conf_points"] if c >= 0.9]
    scenes: dict[str, list[bool]] = {}
    for r in rows:
        scenes.setdefault(r["scene"], []).append(r["err"] <= BOTH_TOL_DEG)
    states: dict[str, int] = {}
    for r in rows:
        states[r["state"]] = states.get(r["state"], 0) + 1
    return {
        "n": len(rows), "n_scenes": len(scenes),
        "median_err": float(np.median(errs)), "p90_err": float(np.percentile(errs, 90)),
        "by_region": {g: [r["err"] for r in rows if r["region"] == g] for g in ("front", "side", "behind")},
        "fb_right_rate": float(np.mean(fb)) if fb else None, "n_fb": len(fb),
        "both_within_25": sum(all(v) for v in scenes.values()),
        "lock_s": locks, "locked_3s_rate": float(np.mean([s is not None and s <= 3 for s in locks])),
        "calibration": table, "ece": ece,
        "hit_rate_when_90": float(np.mean(sure)) if sure else None, "n_90": len(sure),
        "final_mean_conf": float(np.mean([r["conf"] for r in rows])),
        "final_hit_rate": float(np.mean([r["err"] <= bench.HIT_DEG for r in rows])),
        "final_states": states,
    }


def run_suite(root: Path, calib: PairCalibration, pair_cfg: PairConfig | None = None, seed0: int = 100,
              n: int | None = None, secs: float = SCENE_SECONDS, verbose: bool = True,
              raw_out: dict | None = None) -> tuple[list[dict], dict]:
    """raw_out: if a dict, filled with scene name -> {"raw": [...], "streams": {stream: truth deg}}."""
    rows = []
    for k, pair in enumerate(PAIRS[:n]):
        raw = None if raw_out is None else []
        r = score_scene(ensure_scene(root, k, pair, seed0 + k, secs), calib, pair_cfg, raw)
        if raw_out is not None:
            raw_out[r[0]["scene"]] = {"raw": raw, "streams": {x["stream"]: x["truth"] for x in r}}
        rows += r
        if verbose:
            print(f"scene {k:2d} {str(pair):12s} " + "  ".join(
                f"{x['truth']:+5.0f}->{x['est']:+6.0f} ({x['err']:5.1f}, {x['conf']:.2f} {x['state']})" for x in r),
                flush=True)
    return rows, summarize(rows)


def _q(xs, q):
    return f"{np.percentile(xs, q):5.1f}" if len(xs) else "    -"


def print_summary(s: dict, title: str = "") -> None:
    print(f"\n{bench.TARGETS}; two speakers within 25 deg")
    print(f"== {title}{s['n']} talkers in {s['n_scenes']} scenes (simulated, oracle separation)")
    print(f"  error (deg)          median {s['median_err']:5.1f}   p90 {s['p90_err']:5.1f}")
    for g, e in s["by_region"].items():
        if e:
            print(f"    {g:7s} n={len(e):<3d}      median {_q(e, 50)}   p90 {_q(e, 90)}")
    if s["fb_right_rate"] is not None:
        print(f"  front/back right     {100 * s['fb_right_rate']:5.1f}%  ({s['n_fb']} talkers, side not counted)")
    print(f"  both within 25 deg   {s['both_within_25']}/{s['n_scenes']} scenes")
    locks = [x for x in s["lock_s"] if x is not None]
    print(f"  seconds to lock      median {_q(locks, 50)}   committed within 3 s: {100 * s['locked_3s_rate']:.0f}%  "
          f"never: {sum(x is None for x in s['lock_s'])}/{len(s['lock_s'])}")
    print(f"  final states         {s['final_states']}")
    print(f"  confidence honesty   ECE {s['ece']:.3f}   final: says {s['final_mean_conf']:.2f}, "
          f"right {s['final_hit_rate']:.2f}"
          + (f"   when >=0.9 sure: right {100 * s['hit_rate_when_90']:.0f}% of {s['n_90']}"
             if s["n_90"] else "   never >=0.9 sure"))
    for lo, mc, hr, n in s["calibration"]:
        print(f"    conf {lo:.1f}-{min(lo + 0.2, 1.0):.1f}  says {mc:.2f}  right {hr:.2f}  (n={n})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m locate.scenes", description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path, help="folder for the scenes (written on first use)")
    ap.add_argument("--calib", type=Path, default=None, help="tdoa calibration JSON fitted at 16 kHz")
    ap.add_argument("--adapt", action="store_true", help="PairConfig(adapt=True)")
    ap.add_argument("--sim-tuned", action="store_true",
                    help="pair.sim_tuned(): less tempering and a looser ambiguity rule, fitted on simulation only")
    ap.add_argument("--seed0", type=int, default=100, help="scene k uses seed seed0+k (other seeds = held out)")
    ap.add_argument("--n", type=int, default=None, help="only the first n scenes")
    ap.add_argument("--seconds", type=float, default=SCENE_SECONDS)
    ap.add_argument("--json", type=Path, default=None, help="also write rows and summary here")
    ap.add_argument("--raw-json", type=Path, default=None,
                    help="also write the locator's per-hop sources (input for python -m track.jitter)")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    calib = PairCalibration.load(a.calib) if a.calib else PairCalibration.default()
    raw_out = {} if a.raw_json else None
    rows, summ = run_suite(a.root, calib, sim_tuned(adapt=a.adapt) if a.sim_tuned else PairConfig(adapt=a.adapt), a.seed0, a.n, a.seconds, raw_out=raw_out)
    print_summary(summ, "adapt: " if a.adapt else "fixed: ")
    if a.json:
        a.json.write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))
        print(f"\nwrote {a.json}")
    if a.raw_json:
        a.raw_json.write_text(json.dumps(raw_out))
        print(f"wrote {a.raw_json}")


if __name__ == "__main__":
    main()
