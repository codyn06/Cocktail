"""Scoring test for live direction finding (task C4) on a simulated session with ground truth.

  uv run --group gpu python -m locate.score_live sessions/synth_2talk_music [--no-labels]
  uv run python -m locate.score_live sessions/two-talkers --oracle --locate-method tdoa \
      --calib sessions/pairsim-calib.json        # no GPU: pairsim scene, true sources as the separator

Steps the session through LiveSeparator + LiveLocator exactly as live (poses arrive up to each
window's end), then reports, per true source, the matched stream's final world bearing, error,
confidence and state, plus how long until the bearing first stayed within --tol degrees.

--oracle replaces the separation model with the session's ground-truth stems (simulated sessions
only): the masks, per-stream mic images and locator are the live code, so it scores locating
each separated stream with separation taken out of the error (an upper bound for the live hub).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import protocol as p
from ingest import IngestState, Resampler
from mockclient import Session
from separate.live import SR, LiveSeparator
from separate.score import sisdr

from .estimator import wrap180
from .live import LiveLocator
from .pair import PairCalibration, PairConfig


class OracleSeparator:
    """Stands in for the voice separator: returns the true stems for each window, in order.
    LiveSeparator calls it once per window, stepping by its hop from the start of the session.
    Like the voice separator it has at least two outputs (a silent one if there is one stem)."""

    def __init__(self, stems: np.ndarray, hop: int, win: int):
        if len(stems) < 2:
            stems = np.concatenate([stems, np.zeros((2 - len(stems), stems.shape[1]), stems.dtype)])
        self.stems, self.hop, self.win, self.k = stems, hop, win, 0
        self.n_out = len(stems)

    def __call__(self, mono: np.ndarray) -> np.ndarray:
        # one call per window, in order: a different window length or a reused instance would shift every stem
        assert len(mono) == self.win, f"oracle built for {self.win}-sample windows, got {len(mono)}"
        a = self.k * self.hop
        self.k += 1
        out = np.zeros((self.n_out, self.win), np.float32)
        seg = self.stems[:, a:a + self.win]
        out[:, :seg.shape[1]] = seg
        return out


def oracle_separator(s: Session, win_s: float = 2.0, hop_s: float = 1.0) -> OracleSeparator:
    refs = [wavfile.read(s.path / q["stem"])[1].astype(np.float32) for q in s.meta["sources"]]
    n = max(len(r) for r in refs)
    return OracleSeparator(np.stack([np.pad(r, (0, n - len(r))) for r in refs]), int(hop_s * SR), int(win_s * SR))


def run(s: Session, live: LiveSeparator, method: str = "level", calib=None, pair_cfg=None, on_step=None):
    """on_step(loc): called after every hop with the LiveLocator (e.g. to record loc.sources())."""
    st = IngestState()
    loc = LiveLocator(st, method=method, calib=calib, pair_cfg=pair_cfg)
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
        if on_step is not None:
            on_step(loc)
        for c in chunks:
            est[c.stream_id].append(c.audio)
        for sid, e in loc.estimates().items():
            trace[sid].append(((end_us - s.start_ts_us) / 1e6, e.world_deg, e.state, e.confidence, e.alt_bearing_deg))
    return {k: np.concatenate(v) for k, v in est.items()}, trace, loc


def assign_streams(s: Session, est: dict) -> list[str]:
    """Stream id matched to each true source, in s.meta["sources"] order. Oracle assignment by
    separation quality (as in separate.score). est: stream id -> separated audio."""
    srcs = s.meta["sources"]
    refs = {q["name"]: wavfile.read(s.path / q["stem"])[1].astype(np.float32) for q in srcs}
    n = min(len(next(iter(est.values()))), *(len(v) for v in refs.values()))
    names = list(refs)
    return list(max(itertools.permutations(list(est), len(names)),
                    key=lambda pm: sum(sisdr(est[sid][:n], refs[nm][:n]) for sid, nm in zip(pm, names))))


def score(s: Session, live: LiveSeparator, tol_deg: float = 20.0, method: str = "level", calib=None,
          pair_cfg=None) -> dict:
    est, trace, loc = run(s, live, method, calib, pair_cfg)
    srcs = s.meta["sources"]
    best = assign_streams(s, est)
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
    ap.add_argument("--locate-method", choices=["level", "tdoa"], default="level")
    ap.add_argument("--calib", type=Path, default=None, help="tdoa calibration JSON (default: quest3s)")
    ap.add_argument("--oracle", action="store_true", help="ground-truth stems instead of the separation model")
    ap.add_argument("--adapt", action="store_true", help="tdoa: per-source delay swing and offset (PairConfig.adapt)")
    a = ap.parse_args()
    s = Session.load(a.session)
    calib = PairCalibration.load(a.calib) if a.calib else None
    if a.oracle:
        live = LiveSeparator(oracle_separator(s), None)
    else:
        from separate.models import SoundLabeler, VoiceSeparator
        live = LiveSeparator(VoiceSeparator(), None if a.no_labels else SoundLabeler())
    res = score(s, live, a.tol, a.locate_method, calib, PairConfig(adapt=a.adapt))
    for nm, r in res.items():
        ok = "PASS" if r["error_deg"] <= a.tol else "FAIL"
        print(f"{nm:14s} -> {r['stream']}  truth {r['truth_deg']:7.1f}  got {r['world_deg']:7.1f}  "
              f"error {r['error_deg']:5.1f} deg [{ok}]  conf {r['confidence']:.2f}  {r['state']}  "
              f"stable after {r['stable_after_s']} s")
    ok = all(r["error_deg"] <= a.tol for r in res.values())
    print(f"all {len(res)} sources within {a.tol:.0f} deg: {'PASS' if ok else 'FAIL'}")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
