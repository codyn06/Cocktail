"""Run live separation over a saved headset recording and save audio/identity metrics.

  uv run --group gpu python -m separate.replay sessions/live1 --out sessions/live1/out_voicememory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from mockclient import Session

from .live import SR, LiveSeparator
from .score import run_session


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, help="output folder (default: <session>/out_voicememory)")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--no-voice-memory", action="store_true")
    args = ap.parse_args()

    from .models import SoundLabeler, VoiceSeparator

    sess = Session.load(args.session)
    out_dir = args.out or (args.session / "out_voicememory")
    out_dir.mkdir(parents=True, exist_ok=True)
    live = LiveSeparator(VoiceSeparator(), None if args.no_labels else SoundLabeler(),
                         voice_memory=not args.no_voice_memory)
    mix, estimates, _ = run_session(sess, live)
    n = min([len(mix), *(len(x) for x in estimates.values())])
    peak = max(float(np.max(np.abs(mix[:n]))), 1e-9)
    wavfile.write(out_dir / "mix.wav", SR, (mix[:n] / peak * 0.9).astype(np.float32))
    energy = {}
    for stream, audio in estimates.items():
        audio = audio[:n]
        wavfile.write(out_dir / f"{stream}.wav", SR,
                      np.clip(audio / peak * 0.9, -1, 1).astype(np.float32))
        energy[stream] = round(float(10 * np.log10(np.mean(audio ** 2) + 1e-12)), 2)

    steps = np.asarray(live.compute_ms, dtype=np.float32)
    metrics = {
        "evidence": "replayed recording",
        "session": str(args.session),
        "output_samples": n,
        "sample_rate": SR,
        "stream_energy_dbfs": energy,
        "step_compute_ms": {
            "mean": round(float(steps.mean()), 2) if len(steps) else None,
            "p95": round(float(np.percentile(steps, 95)), 2) if len(steps) else None,
            "max": round(float(steps.max()), 2) if len(steps) else None,
            "hop_budget_ms": live.H / SR * 1000,
        },
        "voice_memory_trace": list(live.voice_memory.trace) if live.voice_memory else [],
    }
    if live.voice_memory:
        embed = np.asarray([x["embedding_ms"] for x in live.voice_memory.trace], dtype=np.float32)
        metrics["embedding_compute_ms"] = {
            "mean": round(float(embed.mean()), 2) if len(embed) else 0.0,
            "p95": round(float(np.percentile(embed, 95)), 2) if len(embed) else 0.0,
            "max": round(float(embed.max()), 2) if len(embed) else 0.0,
        }
    try:
        import torch
        if torch.cuda.is_available():
            metrics["cuda_memory_mb"] = {
                "allocated": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "peak_allocated": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "reserved": round(torch.cuda.memory_reserved() / 1024**2, 1),
            }
    except ImportError:
        pass
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "voice_memory_trace"}, indent=2))
    print(f"Wrote stream WAVs and metrics to {out_dir}")


if __name__ == "__main__":
    main()
