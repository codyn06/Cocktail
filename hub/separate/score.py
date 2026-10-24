"""Scoring test for separation (task A6): run the live separator over a session that has
ground-truth stems and report SI-SDR improvement and labels per source.

  uv run --group gpu python -m separate.score sessions/synth_2talk_music [--no-labels]

The session is stepped window by window exactly as live, without the network.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
from scipy.io import wavfile

from ingest import Resampler
from mockclient import Session

from .live import SR, LiveSeparator


def sisdr(est: np.ndarray, ref: np.ndarray) -> float:
    est, ref = est - est.mean(), ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    return float(10 * np.log10(np.sum((a * ref) ** 2) / (np.sum((a * ref - est) ** 2) + 1e-12)))


def run_session(s: Session, live: LiveSeparator):
    """Returns (mono mix at 16 kHz, {stream_id: estimate}, {stream_id: [labels per hop]}, first sample)."""
    r = Resampler(s.sample_rate, s.audio.shape[1])
    x = r(s.audio)[r.delay :]  # drop the resampler lag so sample 0 lines up with the references
    est = {sid: [] for sid in live.ids}
    labels = {sid: [] for sid in live.ids}
    for st in range(0, len(x) - live.W + 1, live.H):
        for c in live.step(x[st : st + live.W], s.start_ts_us + st * 1_000_000 // SR):
            est[c.stream_id].append(c.audio)
            labels[c.stream_id].append(c.label)
    return x.mean(axis=1), {k: np.concatenate(v) for k, v in est.items()}, labels


def score(s: Session, live: LiveSeparator) -> dict:
    mix, est, labels = run_session(s, live)
    refs = {}
    for src in s.meta["sources"]:
        sr, r = wavfile.read(s.path / src["stem"])
        assert sr == SR
        refs[src["name"]] = r.astype(np.float32)
    n = min(len(next(iter(est.values()))), len(mix), *(len(r) for r in refs.values()))
    names, sids = list(refs), list(est)
    # Best one-to-one assignment of streams to references by total SI-SDR.
    best = max(itertools.permutations(sids, len(names)) if len(sids) >= len(names) else [],
               key=lambda p: sum(sisdr(est[sid][:n], refs[nm][:n]) for sid, nm in zip(p, names)))
    out = {}
    for sid, nm in zip(best, names):
        e, r = est[sid][:n], refs[nm][:n]
        base, got = sisdr(mix[:n], r), sisdr(e, r)
        common = max(set(labels[sid]), key=labels[sid].count)
        out[nm] = {"stream": sid, "si_sdr_db": round(got, 2), "mix_si_sdr_db": round(base, 2),
                   "improvement_db": round(got - base, 2), "label": common,
                   "label_hops": labels[sid]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--target-db", type=float, default=8.0)
    a = ap.parse_args()
    from .models import SoundLabeler, VoiceSeparator

    live = LiveSeparator(VoiceSeparator(), None if a.no_labels else SoundLabeler())
    res = score(Session.load(a.session), live)
    for nm, r in res.items():
        ok = "PASS" if r["improvement_db"] >= a.target_db else "FAIL"
        print(f"{nm:14s} -> {r['stream']}  SI-SDR {r['si_sdr_db']:6.2f} dB  (mix {r['mix_si_sdr_db']:6.2f})  "
              f"improvement {r['improvement_db']:+6.2f} dB [{ok}]  label {r['label']}")
    print(json.dumps({k: {kk: v for kk, v in r.items() if kk != "label_hops"} for k, r in res.items()}))


if __name__ == "__main__":
    main()
