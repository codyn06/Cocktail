"""Scoring test for live direction finding (task C4) on a simulated session with ground truth.

  uv run --group gpu python -m locate.score_live sessions/synth_2talk_music [--no-labels]

Steps the session through LiveSeparator + LiveLocator exactly as live (poses arrive up to each
window's end), then reports, per true source, the matched stream's final world bearing, error,
confidence and state, plus how long until the bearing first stayed within 20 degrees.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from scipy.io import wavfile

import protocol as p
from ingest import IngestState, Resampler
from mockclient import Session
from separate.live import SR, LiveSeparator
from separate.score import sisdr

from .estimator import wrap180
from .live import LiveLocator


def run(s: Session, live: LiveSeparator):
    st = IngestState()
    loc = LiveLocator(st)
    r = Resampler(s.sample_rate, s.audio.shape[1])
    x = r(s.audio)[r.delay :]
    poses = iter(s.poses)
    nxt = next(poses, None)
    est = {sid: [] for sid in live.ids}
    trace = {sid: [] for sid in live.ids}  # (t_s, world_deg, state)
    seq = 0
    for start in range(0, len(x) - live.W + 1, live.H):
        t0 = s.start_ts_us + start * 1_000_000 // SR
        end_us = t0 + live.W * 1_000_000 // SR
        while nxt is not None and nxt["device_ts_us"] <= end_us:
            st.on_pose_channel(p.PoseSample(seq=seq, device_ts_us=nxt["device_ts_us"], epoch=0,
                                            pos=tuple(nxt["pos"]), rot_xyzw=tuple(nxt["rot_xyzw"])))
            seq += 1
            nxt = next(poses, None)
        chunks = live.step(x[start : start + live.W], t0)
        loc.on_chunks(chunks)
        for c in chunks:
            est[c.stream_id].append(c.audio)
        for sid, e in loc.estimates().items():
            trace[sid].append(((end_us - s.start_ts_us) / 1e6, e.world_deg, e.state, e.confidence, e.alt_bearing_deg))
    return {k: np.concatenate(v) for k, v in est.items()}, trace, loc


def score(s: Session, live: LiveSeparator, tol_deg: float = 20.0) -> dict:
    est, trace, loc = run(s, live)
    srcs = s.meta["sources"]
    refs = {q["name"]: wavfile.read(s.path / q["stem"])[1].astype(np.float32) for q in srcs}
    n = min(len(next(iter(est.values()))), *(len(v) for v in refs.values()))
    names = list(refs)
    # Oracle assignment of streams to true sources by separation quality (as in separate.score).
    best = max(itertools.permutations(list(est), len(names)),
               key=lambda pm: sum(sisdr(est[sid][:n], refs[nm][:n]) for sid, nm in zip(pm, names)))
    final = loc.estimates()
    out = {}
    for sid, q in zip(best, srcs):
        e, truth = final[sid], q["bearing_deg"]
        err = abs(float(wrap180(e.world_deg - truth)))
        settle = next((t for i, (t, w, *_r) in enumerate(trace[sid])
                       if all(abs(float(wrap180(w2 - truth))) <= tol_deg for _, w2, *_r2 in trace[sid][i:])), None)
        out[q["name"]] = {"stream": sid, "truth_deg": truth, "world_deg": round(e.world_deg, 1),
                          "error_deg": round(err, 1), "confidence": round(e.confidence, 2),
                          "state": e.state, "alt_deg": e.alt_bearing_deg, "stable_after_s": settle}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--tol", type=float, default=20.0)
    a = ap.parse_args()
    from separate.models import SoundLabeler, VoiceSeparator

    res = score(Session.load(a.session), LiveSeparator(VoiceSeparator(), None if a.no_labels else SoundLabeler()),
                a.tol)
    for nm, r in res.items():
        ok = "PASS" if r["error_deg"] <= a.tol else "FAIL"
        print(f"{nm:14s} -> {r['stream']}  truth {r['truth_deg']:7.1f}  got {r['world_deg']:7.1f}  "
              f"error {r['error_deg']:5.1f} deg [{ok}]  conf {r['confidence']:.2f}  {r['state']}  "
              f"stable after {r['stable_after_s']} s")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
