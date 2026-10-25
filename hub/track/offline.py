"""Offline position and identity check: the live locator + tracker on SIMULATED scenes with truth.

    cd hub
    uv run python -m track.offline sessions/offline --calib sessions/pairsim-calib16.json
    uv run python -m track.offline sessions/offline --calib ... --only pause cross --json offline.json

Each case is a pairsim session (locate.pairsim.write_session: talkers that can move, pause, and a
head-pose stream with gaps), written under <dir>/<case> on first use. It runs through the live code
exactly as locate.scenes does (oracle separation: the TRUE stems as the separator, LiveSeparator
masks and mic images, LiveLocator, PairLocator), then the locator's per-hop sources are replayed
into the Tracker at 10 Hz, as the hub's events loop does. Optionally the first two stream slots are
swapped from given times on, after LiveSeparator (a slot swap its own order matching missed).

Truth: with oracle stems, stream s<k> carries talker k (s1/s2 exchanged after each swap), so a
track's identity is scored by provenance: at every tick a true talker is CARRIED by the confirmed,
tracking track whose current stream carries it (no bearing gate, which would blur exactly the
nearby and crossing cases). A talker is AUDIBLE at a tick if its stem has energy (within 30 dB of
its own loud ticks) in the last 2 s, the locator's window: talker()'s short phrase gaps and the
scripted pauses both come from the stem, not the script. Per talker:

* ids: distinct track ids that ever carried it (1 = no fragmentation);
* switches: times the carrying id changed, not counting a return after more than forget_s of
  silence (a new id is then by design);
* coverage: share of audible ticks, from 3 s after it is first audible, when some track carries it;
* first id: seconds from first audible to first carried;
* bearing error of the carrying track against the true bearing (median / p90, ticks after
  --settle s), and the locator's own raw error on the same ticks (if the raw error is already
  large the locator failed, not the tracker).

Per case: confirmed ids that never carried a talker (false), ids that carried two talkers (merged).
Everything here is simulated (pairsim room, oracle separation): it says nothing about the real
headset.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import protocol as p

from .tracker import TrackConfig, Tracker, wrap180

DT_US = 100_000          # the hub ticks the tracker at 10 Hz
TICK = 1600              # 100 ms at 16 kHz (stems are 16 kHz)
WINDOW_S = 2.0           # LiveSeparator window: a talker heard in it can be observed
AUDIBLE_DB = -30.0       # stem tick energy within this of the stem's own p90 = talking
WARMUP_S = 3.0           # coverage is counted from this long after a talker is first audible
SETTLE_S = 12.0          # bearing errors from here on (the locator needs ~12 s to converge)

# Cases: pairsim.write_session arguments plus harness options. Bearings in deg, + = right.
CASES = {
    "base":       dict(angles=[-30, 40]),
    "behind":     dict(angles=[150, -160]),
    "near10":     dict(angles=[20, 30]),
    "near20":     dict(angles=[20, 40]),
    "near30":     dict(angles=[20, 50]),
    "pause10":    dict(angles=[-30, 40], silent=[None, [[14, 24]]]),
    "pause30":    dict(angles=[-30, 40], silent=[None, [[14, 44]]], secs=60.0),
    "pause70":    dict(angles=[-30, 40], silent=[None, [[14, 84]]], secs=100.0),
    "swap":       dict(angles=[-30, 40], swaps=[20.0]),
    "pause_swap": dict(angles=[-30, 40], silent=[None, [[14, 24]]], swaps=[19.0]),
    "cross_slow": dict(angles=[-60, 0], paths=[[[8, -60], [38, 60]], None]),
    "cross_fast": dict(angles=[-60, 0], paths=[[[16, -60], [24, 60]], None]),
    "cross_pause": dict(angles=[-60, 0], paths=[[[8, -60], [38, 60]], None], silent=[[[19, 27]], None]),
    "no_head":    dict(angles=[-30, 40], head_amp_deg=0.0),
    "pose_gap":   dict(angles=[-30, 40], pose_gaps=[[15, 25]]),
    "calib_off":  dict(angles=[-30, 40], calib_offset_us=5.0),
}
SESSION_KEYS = ("angles", "paths", "silent", "pose_gaps", "secs", "head_amp_deg")


def ensure_case(root: Path, name: str, spec: dict, seed: int) -> Path:
    """Write the case's session under root/<name>-s<seed> unless the same spec is already there."""
    from locate import pairsim

    d = root / f"{name}-s{seed}"
    want = {k: spec[k] for k in SESSION_KEYS if k in spec} | {"seed": seed}
    if (d / "spec.json").exists() and json.loads((d / "spec.json").read_text()) == want:
        return d
    room = pairsim.Room(head_amp_deg=spec["head_amp_deg"]) if "head_amp_deg" in spec else None
    pairsim.write_session(d, [float(a) for a in spec["angles"]], seed=seed, room=room,
                          secs=spec.get("secs", 40.0), paths=spec.get("paths"), silent=spec.get("silent"),
                          pose_gaps=spec.get("pose_gaps"))
    (d / "spec.json").write_text(json.dumps(want))
    return d


class SwapSlots:
    """A LiveSeparator whose first two stream ids are exchanged from each time in swaps_s on
    (seconds from the session start)."""

    def __init__(self, live, swaps_s: list[float], start_us: int):
        self.live, self.swaps_us = live, [start_us + int(t * 1e6) for t in swaps_s]

    def __getattr__(self, k):
        return getattr(self.live, k)

    def step(self, x, start_ts_us):
        chunks = self.live.step(x, start_ts_us)
        if sum(start_ts_us >= t for t in self.swaps_us) % 2:
            ren = {"s1": "s2", "s2": "s1"}
            chunks = [dataclasses.replace(c, stream_id=ren.get(c.stream_id, c.stream_id)) for c in chunks]
        return chunks


def carried_by(n_src: int, swaps_us: list[int], t_us: int) -> dict[str, int]:
    """stream id -> index of the true talker it carries at device time t_us."""
    m = {f"s{k + 1}": k for k in range(n_src)}
    if sum(t_us >= t for t in swaps_us) % 2:
        m = {{"s1": "s2", "s2": "s1"}.get(s, s): k for s, k in m.items()}
    return m


def audible(stem: np.ndarray, n_ticks: int) -> np.ndarray:
    """(n_ticks,) bool: the talker had energy in the WINDOW_S before each tick end."""
    k = len(stem) // TICK
    e = np.sqrt(np.mean(stem[: k * TICK].reshape(k, TICK) ** 2, axis=1))
    on = e > np.percentile(e, 90) * 10 ** (AUDIBLE_DB / 20)
    w = int(WINDOW_S * 1e6 / DT_US)
    out = np.zeros(n_ticks, bool)
    for i in range(n_ticks):
        out[i] = on[max(0, i - w + 1): i + 1].any() if i < k else False
    return out


def true_bearing(q: dict, t_s: float) -> float:
    if q.get("path"):
        kt, kd = np.array(q["path"], float).T
        return float(wrap180(np.interp(t_s, kt, kd)))
    return float(q["bearing_deg"])


def run_case(d: Path, calib, swaps_s=(), cfg: TrackConfig | None = None, pair_cfg=None) -> dict:
    """Locator + tracker on one session. Returns raw per-hop sources and per-tick tracker state."""
    from locate import score_live
    from mockclient import Session
    from separate.live import LiveSeparator

    s = Session.load(d)
    live = LiveSeparator(score_live.oracle_separator(s), None)
    if swaps_s:
        live = SwapSlots(live, list(swaps_s), s.start_ts_us)
    raw = []
    score_live.run(s, live, "tdoa", calib, pair_cfg, on_step=lambda loc: raw.append(
        (loc.now_us, [x.model_copy() for x in loc.sources()])))
    tr, ticks, k = Tracker(cfg), [], 0
    raw = [r for r in raw if r[0] > 0]
    for now in range(raw[0][0], raw[-1][0] + 1, DT_US):
        while k + 1 < len(raw) and raw[k + 1][0] <= now:
            k += 1
        tr.update(now, raw[k][1], None)
        ticks.append((now, {t.tid: (t.stream, t.bearing, t.state, t.last_obs_us) for t in tr.tracks.values() if t.confirmed},
                      {x.source_id: x.bearing_deg for x in raw[k][1]}))
    return {"session": s, "ticks": ticks, "swaps_us": [s.start_ts_us + int(t * 1e6) for t in swaps_s]}


def score_case(res: dict, forget_s: float = TrackConfig.forget_s, settle_s: float = SETTLE_S) -> dict:
    s, ticks, swaps_us = res["session"], res["ticks"], res["swaps_us"]
    srcs = s.meta["sources"]
    t0 = ticks[0][0]
    idx0 = (t0 - s.start_ts_us) // DT_US  # stem tick index of the first tracker tick
    stems = [wavfile.read(s.path / q["stem"])[1].astype(np.float32) for q in srcs]
    aud = [audible(st, idx0 + len(ticks))[idx0:] for st in stems]
    carry = [[] for _ in srcs]          # per talker: (tick index, tid or None)
    err = [[] for _ in srcs]
    raw_err = [[] for _ in srcs]
    tids_seen: dict[str, set] = {}
    all_tids = set()
    for i, (now, tracks, rawb) in enumerate(ticks):
        t_s = (now - s.start_ts_us) / 1e6
        m = carried_by(len(srcs), swaps_us, now)
        all_tids |= set(tracks)
        best: dict[int, tuple] = {}
        for tid, (stream, b, state, last) in tracks.items():
            k = m.get(stream)
            if k is None or state != p.TrackState.tracking:
                continue
            if k not in best or last > best[k][1]:
                best[k] = (tid, last, b)
        for k, q in enumerate(srcs):
            tid = best.get(k, (None,))[0]
            carry[k].append((i, tid))
            if tid is not None:
                tids_seen.setdefault(tid, set()).add(k)
                tb = true_bearing(q, t_s)
                if t_s >= settle_s and best[k][2] is not None:
                    err[k].append(abs(float(wrap180(best[k][2] - tb))))
                    sid = next((sd for sd, kk in m.items() if kk == k), None)
                    if rawb.get(sid) is not None:
                        raw_err[k].append(abs(float(wrap180(rawb[sid] - tb))))
    out = {"talkers": [], "false_ids": sorted(t for t in all_tids if t not in tids_seen),
           "merged_ids": sorted(t for t, ks in tids_seen.items() if len(ks) > 1)}
    for k, q in enumerate(srcs):
        a = aud[k]
        first = int(np.argmax(a)) if a.any() else None
        seq = [(i, tid) for i, tid in carry[k] if tid is not None]
        sw, prev = 0, None
        for i, tid in seq:
            if prev is not None and tid != prev[1] and (i - prev[0]) * DT_US / 1e6 <= forget_s:
                sw += 1
            prev = (i, tid)
        start = None if first is None else first + int(WARMUP_S * 1e6 / DT_US)
        cov_ticks = [i for i in range(len(ticks)) if a[i] and start is not None and i >= start]
        covered = sum(carry[k][i][1] is not None for i in cov_ticks)
        first_id = next((i for i, _ in seq if first is not None and i >= first), None)
        q50 = lambda v, pct: round(float(np.percentile(v, pct)), 1) if v else None
        out["talkers"].append({
            "name": q["name"], "truth": q.get("path") or q["bearing_deg"],
            "ids": sorted({tid for _, tid in seq}), "switches": sw,
            "coverage": round(covered / len(cov_ticks), 3) if cov_ticks else None,
            "first_id_s": None if first_id is None else round((first_id - first) * DT_US / 1e6, 1),
            "err_med": q50(err[k], 50), "err_p90": q50(err[k], 90),
            "raw_err_med": q50(raw_err[k], 50), "raw_err_p90": q50(raw_err[k], 90)})
    return out


def run_suite(root: Path, calib, names, seeds, cfg: TrackConfig | None = None) -> dict:
    results = {}
    for name in names:
        spec = CASES[name]
        c = calib
        if spec.get("calib_offset_us"):
            c = dataclasses.replace(calib, offset_s=calib.offset_s + spec["calib_offset_us"] * 1e-6)
        for seed in seeds:
            d = ensure_case(root, name, spec, seed)
            sc = score_case(run_case(d, c, spec.get("swaps", ()), cfg), cfg.forget_s if cfg else TrackConfig.forget_s)
            results[f"{name}-s{seed}"] = sc
            print_case(f"{name}-s{seed}", sc)
    return results


def print_case(key: str, sc: dict) -> None:
    flags = []
    if sc["false_ids"]:
        flags.append(f"false ids {sc['false_ids']}")
    if sc["merged_ids"]:
        flags.append(f"MERGED {sc['merged_ids']}")
    print(f"\n{key}" + (f"   [{', '.join(flags)}]" if flags else ""))
    for r in sc["talkers"]:
        print(f"  {r['name']:8s} ids {','.join(r['ids']) or '-':10s} switches {r['switches']}  "
              f"coverage {r['coverage']}  first id {r['first_id_s']} s  "
              f"err {r['err_med']}/{r['err_p90']}  (locator {r['raw_err_med']}/{r['raw_err_p90']})")


def main(argv=None) -> int:
    from locate.pair import PairCalibration

    ap = argparse.ArgumentParser(prog="python -m track.offline", description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path, help="folder for the case sessions (written on first use)")
    ap.add_argument("--calib", type=Path, required=True, help="tdoa calibration JSON fitted at 16 kHz")
    ap.add_argument("--only", nargs="*", default=None, help="cases whose name contains any of these")
    ap.add_argument("--seeds", type=int, default=1, help="seeds per case (seed0, seed0+1, ...)")
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--set", action="append", default=[], help="TrackConfig override, e.g. merge_deg=8")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg = TrackConfig()
    for kv in a.set:
        k, v = kv.split("=")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    names = [n for n in CASES if not a.only or any(w in n for w in a.only)]
    res = run_suite(a.root, PairCalibration.load(a.calib), names, range(a.seed0, a.seed0 + a.seeds), cfg)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
