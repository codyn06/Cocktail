"""Streaming separation benchmark: SI-SDRi, RTF, per-hop latency, peak VRAM.

Run from hub/:  uv run python ../eval/bench.py --models convtasnet-libri2mix-16k sepformer-whamr16k
Evidence label: replayed (offline mixtures streamed chunk-by-chunk on the local GPU).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hub"))
from separate.models import REGISTRY  # noqa: E402
from separate.streaming import MODES, WindowedStreamer, SR  # noqa: E402


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    a = (est @ ref) / (ref @ ref + 1e-12)
    t = a * ref
    return float(10 * np.log10((t @ t + 1e-12) / (((est - t) ** 2).sum() + 1e-12)))


def score(est: np.ndarray, stems: np.ndarray, mix: np.ndarray, skip: int) -> tuple[float, float]:
    """Best permutation over the whole stream. Returns (SI-SDR, SI-SDRi) mean over refs."""
    n = est.shape[1]
    est, stems, mix = est[:, skip:], stems[:, skip:n], mix[skip:n]
    best = None
    for p in itertools.permutations(range(est.shape[0]), stems.shape[0]):
        s = [si_sdr(est[j], stems[i]) for i, j in enumerate(p)]
        if best is None or np.mean(s) > np.mean(best):
            best = s
    base = np.mean([si_sdr(mix, stems[i]) for i in range(stems.shape[0])])
    return float(np.mean(best)), float(np.mean(best) - base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["convtasnet-libri2mix-16k"])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--max-cases", type=int, default=99)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()
    files = sorted(Path(a.data).glob("*.npz"))[: a.max_cases]
    assert files, "run eval/prepare_data.py first"
    results = []
    for mname in a.models:
        fn, nspk, info = REGISTRY[mname]("cuda")
        # warmup (cuDNN autotune/allocator) so first-hop compile cost is not counted
        for _ in range(3):
            fn(torch.randn(SR * 2, device="cuda") * 0.1)
        for mode in a.modes:
            W, hop, look = MODES[mode]
            rows, hop_ms, vram = [], [], 0.0
            for f in files:
                d = np.load(f)
                st = WindowedStreamer(fn, nspk, W, hop, look)
                est = st.run(d["mix"])
                sd, sdi = score(est, d["stems"], d["mix"], skip=int(W * SR))
                rows.append((f.stem, sd, sdi))
                hop_ms += st.stats.hop_ms[5:]
                vram = max(vram, st.stats.peak_vram_mb)
            hm = np.array(hop_ms)
            r = dict(model=mname, mode=mode, window_s=W, hop_ms=hop, look_ms=look,
                     sisdri_mean=float(np.mean([x[2] for x in rows])),
                     sisdr_mean=float(np.mean([x[1] for x in rows])),
                     per_case={x[0]: round(x[2], 2) for x in rows},
                     compute_ms_mean=float(hm.mean()), compute_ms_p95=float(np.percentile(hm, 95)),
                     compute_ms_max=float(hm.max()), rtf=float(hm.mean() / hop),
                     algo_latency_ms=hop + look,
                     e2e_latency_ms_p95=float(hop + look + np.percentile(hm, 95)),
                     peak_vram_mb=vram, **info)
            results.append(r)
            print(f"{mname:28s} {mode:9s} SI-SDRi {r['sisdri_mean']:+6.2f} dB  RTF {r['rtf']:.2f}  "
                  f"compute p95 {r['compute_ms_p95']:.1f} ms  e2e p95 {r['e2e_latency_ms_p95']:.0f} ms  "
                  f"VRAM {vram:.0f} MB", flush=True)
        del fn
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
