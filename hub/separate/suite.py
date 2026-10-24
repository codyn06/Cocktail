"""Scenario suite for separation (task A6): builds simulated scenes from librosa's example clips,
runs the live separator on each exactly as live, and scores every source.

  uv run --group gpu --with librosa python -m separate.suite [--only 1,6] [--no-labels]

Per source: best-matching stream (sources may share one when there are more sources than
streams), SI-SDR improvement, and that stream's majority label. Per scene: level of streams
no source claimed (should be quiet), and how often the voices swapped streams between hops.
Clips: LibriSpeech readers libri1/libri2 (men) and libri3 (woman), music (brahms, pistachio,
choice), a robin. All simulated (two ear-level mics, head turns, no reverb).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from mockclient import Session
from mockclient.simulate import simulate

from .live import SR, LiveSeparator
from .score import run_session, sisdr

DUR = 12.0
SPEECH_RMS = 0.05


@dataclass
class Src:
    name: str
    clip: str             # librosa example key
    bearing: float
    gain_db: float = 0.0  # relative to speech level
    start: float = 0.0    # seconds into the scene
    length: float = DUR
    clip_offset: float = 0.0
    repeat_every: float = 0.0  # >0: repeat the clip with this period (short clips like the robin)


SCENES = {
    1: ("2 voices (man + woman), no music", [Src("man", "libri2", -50), Src("woman", "libri3", 40)]),
    2: ("2 voices + music", [Src("man", "libri2", -50), Src("woman", "libri3", 40),
                             Src("music", "brahms", 150, -5, clip_offset=5)]),
    3: ("2 voices, same gender (two men)", [Src("man1", "libri1", -50), Src("man2", "libri2", 40)]),
    4: ("1 voice + music", [Src("woman", "libri3", 40), Src("music", "brahms", 150, -5, clip_offset=5)]),
    5: ("1 voice alone", [Src("woman", "libri3", 40)]),
    6: ("turn-taking: man 0-5 s, pause, woman 7-12 s",
        [Src("man", "libri2", -50, length=5), Src("woman", "libri3", 40, start=7, length=5)]),
    7: ("3 voices, no music", [Src("man1", "libri1", -60), Src("man2", "libri2", 20),
                               Src("woman", "libri3", 100)]),
    8: ("3 voices + music", [Src("man1", "libri1", -60), Src("man2", "libri2", 20),
                             Src("woman", "libri3", 100), Src("music", "pistachio", 170, -5)]),
    9: ("2 voices + loud music (+5 dB)", [Src("man", "libri2", -50), Src("woman", "libri3", 40),
                                          Src("music", "choice", 150, 5)]),
    10: ("1 voice + bird song", [Src("woman", "libri3", 40), Src("bird", "robin", -120, -3, repeat_every=4.0)]),
    11: ("music only", [Src("music", "brahms", 150, 0, clip_offset=5)]),
}


def build_stem(s: Src, n: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(librosa.ex(s.clip), sr=SR, offset=s.clip_offset)
    y = y / (np.sqrt(np.mean(y[np.abs(y) > 1e-4] ** 2)) + 1e-9) * SPEECH_RMS * 10 ** (s.gain_db / 20)
    out = np.zeros(n, np.float32)
    a, b = int(s.start * SR), min(n, int((s.start + s.length) * SR))
    if s.repeat_every:
        for t in np.arange(a, b, int(s.repeat_every * SR)):
            k = min(len(y), b - t)
            out[t : t + k] = y[:k]
    else:
        k = min(len(y), b - a)
        out[a : a + k] = y[:k]
    return out


def make_session(root: Path, sid: int) -> Session:
    _, srcs = SCENES[sid]
    d = root / f"scene{sid:02d}"
    (d / "in").mkdir(parents=True, exist_ok=True)
    n = int(DUR * SR)
    specs = []
    for s in srcs:
        p = d / "in" / f"{s.name}.wav"
        wavfile.write(p, SR, build_stem(s, n))
        specs.append(f"{p}:{s.bearing}")
    return simulate(d / "session", specs, duration_s=DUR)


def score_scene(sess: Session, live: LiveSeparator) -> dict:
    live.reset()
    mix, est, labels = run_session(sess, live)
    n = min(len(mix), *(len(v) for v in est.values()))
    lab = {sid: max(set(v), key=v.count) for sid, v in labels.items()}
    od = sess.path.parent / "out"  # what to listen to: the mix and every output stream
    od.mkdir(exist_ok=True)
    peak = max(np.abs(mix[:n]).max(), 1e-9)
    wavfile.write(od / "mix.wav", SR, (mix[:n] / peak * 0.9).astype(np.float32))
    for sid, e in est.items():  # same scale as the mix, so a quiet stream really is quiet
        wavfile.write(od / f"{sid}_{lab[sid]}.wav", SR, np.clip(e[:n] / peak * 0.9, -1, 1).astype(np.float32))
    refs = {}
    for q in sess.meta["sources"]:
        r = wavfile.read(sess.path / q["stem"])[1].astype(np.float32)[:n]
        if np.sqrt(np.mean(r**2)) > 1e-5:
            refs[q["name"]] = r
    H = live.H
    out, claimed = {}, {}
    for nm, r in refs.items():
        sc = {sid: sisdr(e[:n], r) for sid, e in est.items()}
        sid = max(sc, key=sc.get)
        # Label of the stream during hops where this source is actually present.
        act = [k for k in range(len(labels[sid])) if np.sqrt(np.mean(r[k * H : (k + 1) * H] ** 2)) > 0.2 * SPEECH_RMS]
        labs = [labels[sid][k] for k in act] or labels[sid]
        out[nm] = {"stream": sid, "improvement_db": round(sc[sid] - sisdr(mix[:n], r), 1),
                   "si_sdr_db": round(sc[sid], 1), "label": max(set(labs), key=labs.count)}
        claimed.setdefault(sid, []).append(nm)
    mix_db = 10 * np.log10(np.mean(mix[:n] ** 2) + 1e-12)
    unclaimed = {sid: round(10 * np.log10(np.mean(e[:n] ** 2) + 1e-12) - mix_db, 1)
                 for sid, e in est.items() if sid not in claimed}
    # Voice swaps: per hop, which stream correlates best with each voice; count changes.
    swaps = {}
    for nm, r in refs.items():
        seq = []
        for k in range(n // H):
            seg = r[k * H : (k + 1) * H]
            if np.sqrt(np.mean(seg**2)) < 0.2 * SPEECH_RMS:
                continue
            seq.append(max(est, key=lambda s: abs(np.dot(est[s][k * H : (k + 1) * H], seg))))
        swaps[nm] = int(sum(a != b for a, b in zip(seq, seq[1:])))
    shared = {sid: v for sid, v in claimed.items() if len(v) > 1}
    return {"sources": out, "unclaimed_stream_db": unclaimed, "shared_streams": shared, "swaps": swaps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated scene numbers")
    ap.add_argument("--out", default="sessions/suite", type=Path)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--win", type=float, default=4.0, help="window seconds (hop = half)")
    ap.add_argument("--fp32", action="store_true", help="separator in full precision (default bf16)")
    a = ap.parse_args()
    from .models import SoundLabeler, VoiceSeparator

    live = LiveSeparator(VoiceSeparator(bf16=not a.fp32), None if a.no_labels else SoundLabeler(), win_s=a.win)
    ids = [int(x) for x in a.only.split(",")] if a.only else list(SCENES)
    results = {}
    for sid in ids:
        title = SCENES[sid][0]
        t0 = time.perf_counter()
        r = score_scene(make_session(a.out, sid), live)
        r["compute_s_per_audio_s"] = round((time.perf_counter() - t0) / DUR, 2)
        results[sid] = {"title": title, **r}
        print(f"\n## {sid}. {title}", flush=True)
        for nm, s in r["sources"].items():
            print(f"  {nm:7s} -> {s['stream']}  {s['improvement_db']:+6.1f} dB  label {s['label']:7s} "
                  f"swaps {r['swaps'].get(nm, 0)}", flush=True)
        if r["shared_streams"]:
            print(f"  shared streams: {r['shared_streams']}", flush=True)
        print(f"  compute: {r['compute_s_per_audio_s']} s per second of audio", flush=True)
        if r["unclaimed_stream_db"]:
            print(f"  unused streams (dB vs mix, lower = quieter): {r['unclaimed_stream_db']}", flush=True)
    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {a.out / 'results.json'}")


if __name__ == "__main__":
    main()
