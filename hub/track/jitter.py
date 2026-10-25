"""How much does the arrow jitter, and how late does it follow? (tracker smoothing, SIMULATED input)

    cd hub
    uv run python -m locate.scenes sessions/scenes --calib sessions/pairsim-calib16.json --raw-json raw.json
    uv run python -m track.jitter raw.json                      # default TrackConfig
    uv run python -m track.jitter raw.json --set smooth_tau_s=2.0 --set follow_tau_s=0.4

The input is what the live locator publishes for the 12 two-talker scenes of locate.scenes (oracle
separation, pairsim rooms): the locator's sources after every separator hop (1 per second), replayed
into the Tracker at 10 Hz exactly as the hub's events loop does. The talkers stand still, so any
movement of a tracked bearing after it has settled is jitter. Metrics, per talker:

* wander: total bearing travel per second after settling (deg/s; 0 = a steady arrow);
* step p95: 95th percentile of the tick-to-tick bearing change (deg per 0.1 s);
* sd: spread of the bearing around its own median (deg);
* error: median / p90 distance of the tracked bearing from the true bearing (smoothing must not
  cost accuracy).

Following a moving source cannot be measured from standing talkers, so it is SEMI-SYNTHETIC: the
locator's own error trace (raw bearing minus truth) is kept and added to a scripted trajectory,
so the tracker sees realistic noise around a talker who moves.
* step: the talker jumps +STEP_DEG at STEP_AT_S. t80 = seconds until the tracked bearing has
  covered 80% of the jump; overshoot = degrees past the new settled bearing.
* ramp: the talker walks at RAMP_DEG_S for RAMP_S. lag = mean of (true - tracked) / speed in
  seconds over the walk after its first second.
It scores the tracker only: the locator itself needs seconds to follow a talker who really moves.
white: a synthetic locator that is not steady (6 deg of white noise on every 10 Hz reading): the
spread of the tracked bearing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

import protocol as p

from .tracker import TrackConfig, Tracker, wrap180

DT_US = 100_000          # the hub ticks the tracker at 10 Hz
START_S = 1.0            # pairsim sessions start at device time 1 s
SETTLE_S = 12.0          # the locator needs this long to converge; earlier ticks are not jitter
STEP_DEG, STEP_AT_S = 20.0, 20.0
RAMP_DEG_S, RAMP_S, RAMP_AT_S = 15.0, 2.0, 20.0


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def replay(raw: list[dict], cfg: TrackConfig | None = None, shift=None, dt_us: int = DT_US) -> list[dict]:
    """Run the locator's per-hop sources through a Tracker at 10 Hz. shift(stream, t_s, bearing)
    -> bearing lets a caller move a talker. Returns one dict per tick: t_s and {stream: bearing}
    of the announced track that currently carries that stream."""
    tr, out = Tracker(cfg), []
    snaps = [(r["t_us"], [p.Source.model_validate(s) for s in r["sources"]]) for r in raw]
    if not snaps:
        return out
    k = 0
    for now in range(snaps[0][0], snaps[-1][0] + 1, dt_us):
        while k + 1 < len(snaps) and snaps[k + 1][0] <= now:
            k += 1
        t_s = now / 1e6 - START_S
        src = snaps[k][1]
        if shift is not None:
            src = [s.model_copy(update={"bearing_deg": float(wrap180(shift(s.source_id, t_s, s.bearing_deg)))})
                   if s.bearing_deg is not None else s for s in src]
        tr.update(now, src, None)
        by = {}
        for t in tr.tracks.values():
            if t.confirmed and t.bearing is not None and (t.stream not in by or t.last_obs_us > by[t.stream][1]):
                by[t.stream] = (t.bearing, t.last_obs_us)
        out.append({"t_s": t_s, "by_stream": {s: b for s, (b, _) in by.items()}})
    return out


def _series(ticks, sid, t0=None, t1=None):
    pts = [(x["t_s"], x["by_stream"][sid]) for x in ticks
           if sid in x["by_stream"] and (t0 is None or x["t_s"] >= t0) and (t1 is None or x["t_s"] < t1)]
    return np.array([t for t, _ in pts]), np.array([b for _, b in pts])


def stationary(ticks, truth: dict, settle_s: float = SETTLE_S) -> dict[str, dict]:
    res = {}
    for sid, tru in truth.items():
        t, b = _series(ticks, sid, settle_s)
        if len(b) < 20:
            continue
        d = np.abs(wrap180(np.diff(b)))
        med = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(b))))))
        err = np.abs(wrap180(b - tru))
        res[sid] = {"wander": float(d.sum() / (t[-1] - t[0])), "step_p95": float(np.percentile(d, 95)),
                    "sd": float(np.std(wrap180(b - med))), "err_med": float(np.median(err)),
                    "err_p90": float(np.percentile(err, 90))}
    return res


def _noise_shift(raw, truth, traj):
    """shift() that keeps each reading's error against the true bearing and moves the truth along traj(t)."""
    return lambda sid, t_s, b: traj(t_s) + wrap180(b - truth[sid]) + truth[sid] if sid in truth else b


def step_response(raw, truth, cfg=None) -> dict[str, dict]:
    traj = lambda t: STEP_DEG if t >= STEP_AT_S else 0.0
    ticks = replay(raw, cfg, _noise_shift(raw, truth, traj))
    res = {}
    for sid, tru in truth.items():
        t, b = _series(ticks, sid)
        if len(b) < 50:
            continue
        rel = wrap180(b - tru)  # tracked bearing relative to the unmoved truth
        pre = np.median(rel[(t >= STEP_AT_S - 5) & (t < STEP_AT_S)])
        post = np.median(rel[t >= t[-1] - 5])
        jump = post - pre
        if abs(jump) < 0.5 * STEP_DEG:
            res[sid] = {"t80": None, "overshoot": 0.0}
            continue
        after = (t >= STEP_AT_S)
        done = after & ((rel - pre) / jump >= 0.8)
        t80 = float(t[done][0] - STEP_AT_S) if done.any() else None
        res[sid] = {"t80": t80, "overshoot": float(max(0.0, np.max(np.sign(jump) * (rel[after] - post))))}
    return res


def ramp_response(raw, truth, cfg=None) -> dict[str, dict]:
    def traj(t):
        return RAMP_DEG_S * float(np.clip(t - RAMP_AT_S, 0.0, RAMP_S))
    ticks = replay(raw, cfg, _noise_shift(raw, truth, traj))
    res = {}
    for sid, tru in truth.items():
        t, b = _series(ticks, sid, RAMP_AT_S + 1.0, RAMP_AT_S + RAMP_S)
        if len(b) < 5:
            continue
        gap = np.array([traj(x) for x in t]) - wrap180(b - tru)
        res[sid] = {"lag_s": float(np.mean(gap) / RAMP_DEG_S)}
    return res


def noisy_stationary(cfg: TrackConfig | None = None, sd_deg: float = 6.0, seconds: float = 30.0) -> dict:
    """A locator that is NOT steady: a standing talker whose reading is off by N(0, sd_deg) at
    every 10 Hz tick (scenarios.jit). Synthetic. Returns the tracked bearing's sd and p95 tick step."""
    from .scenarios import jit, raw, simulate
    run = simulate(lambda t, rng: [raw("s1", "speech", jit(rng, 40.0, sd_deg))], seconds, tracker=Tracker(cfg))
    b = np.array([o[0].bearing_deg for o in run.out[100:] if o])
    return {"sd": float(np.std(b)), "step_p95": float(np.percentile(np.abs(np.diff(b)), 95))}


def evaluate(scenes: dict, cfg: TrackConfig | None = None, follow: bool = True) -> dict:
    """Aggregate over all talkers of all scenes: medians of each per-talker metric."""
    st, sp, rp = [], [], []
    for sc in scenes.values():
        st += list(stationary(replay(sc["raw"], cfg), sc["streams"]).values())
        if follow:
            sp += list(step_response(sc["raw"], sc["streams"], cfg).values())
            rp += list(ramp_response(sc["raw"], sc["streams"], cfg).values())
    med = lambda rows, k: float(np.median([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None
    out = {"n": len(st), **{k: med(st, k) for k in ("wander", "step_p95", "sd", "err_med", "err_p90")}}
    if follow:
        out.update({"t80": med(sp, "t80"), "t80_missed": sum(r["t80"] is None for r in sp), "n_step": len(sp),
                    "overshoot": med(sp, "overshoot"), "lag_s": med(rp, "lag_s")})
    out["noisy_sd"] = noisy_stationary(cfg)["sd"]
    return out


def line(name: str, m: dict) -> str:
    f = lambda v, fmt: "-" if v is None else format(v, fmt)
    s = (f"{name:34s} wander {f(m['wander'], '5.1f')} deg/s  step p95 {f(m['step_p95'], '5.2f')}  sd {f(m['sd'], '4.1f')}  "
         f"err med {f(m['err_med'], '4.1f')} p90 {f(m['err_p90'], '4.1f')}")
    if "t80" in m:
        s += (f"  | white 6deg sd {f(m['noisy_sd'], '4.2f')}  | step t80 {f(m['t80'], '4.2f')} s (missed {m['t80_missed']}/{m['n_step']}) overshoot "
              f"{f(m['overshoot'], '4.1f')} deg  ramp lag {f(m['lag_s'], '4.2f')} s")
    return s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m track.jitter", description=__doc__.split("\n")[0])
    ap.add_argument("raw", type=Path, nargs="+", help="--raw-json files from locate.scenes")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="TrackConfig field to change; also scores the default for comparison")
    a = ap.parse_args(argv)
    scenes = {}
    for f in a.raw:
        scenes.update({f"{f.stem}/{k}": v for k, v in load(f).items()})
    fields = {f.name: f.type for f in dataclasses.fields(TrackConfig)}
    kw = {}
    for item in a.set:
        k, v = item.split("=")
        if k not in fields:
            ap.error(f"unknown TrackConfig field {k}")
        kw[k] = float(v)
    print(f"{len(scenes)} scenes (simulated, oracle separation); medians over talkers; "
          f"settled from {SETTLE_S:.0f} s; follow metrics are semi-synthetic")
    print(line("default TrackConfig", evaluate(scenes)))
    if kw:
        print(line(" ".join(f"{k}={v:g}" for k, v in kw.items()), evaluate(scenes, TrackConfig(**kw))))


if __name__ == "__main__":
    main()
