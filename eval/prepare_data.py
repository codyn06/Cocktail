"""Download small public (ungated) test audio into eval/data/ (gitignored) and build
controlled mixtures at 16 kHz.

Sources (both ungated HF datasets):
  - hf-internal-testing/librispeech_asr_dummy  (LibriSpeech dev-clean excerpts, CC-BY-4.0)
  - lewtun/music_genres_small                  (GTZAN-derived music clips)

Usage: uv run python ../eval/prepare_data.py   (from hub/)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly
import pandas as pd

SR = 16000
DATA = Path(__file__).parent / "data"
DUR = 20.0  # seconds per mixture


def _decode(cell) -> tuple[np.ndarray, int]:
    x, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x, SR


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return np.tile(x, int(np.ceil(n / len(x))))[:n] if len(x) < n else x[:n]


def scale_to_snr(sig: np.ndarray, ref: np.ndarray, snr_db: float) -> np.ndarray:
    ps, pr = (sig**2).mean() + 1e-12, (ref**2).mean() + 1e-12
    return sig * np.sqrt(pr / ps * 10 ** (-snr_db / 10))


def main() -> None:
    DATA.mkdir(exist_ok=True)
    import tarfile
    tgz = DATA / "raw" / "dev-clean.tar.gz"  # https://www.openslr.org/resources/12/dev-clean.tar.gz
    by_spk: dict[int, list[np.ndarray]] = {}
    with tarfile.open(tgz) as tf:
        for m in tf:
            if not m.name.endswith(".flac"):
                continue
            spk = int(m.name.split("/")[-1].split("-")[0])
            if len(by_spk.get(spk, [])) >= 8 or (len(by_spk) >= 6 and spk not in by_spk):
                continue
            x, sr = sf.read(tf.extractfile(m), dtype="float32")
            by_spk.setdefault(spk, []).append(x)
    spk_audio = {k: np.concatenate(v) for k, v in by_spk.items()}
    spk_audio = {k: v for k, v in spk_audio.items() if len(v) > 8 * SR}
    print("speakers with >8 s:", {k: round(len(v) / SR, 1) for k, v in spk_audio.items()})
    ids = sorted(spk_audio)

    mp = hf_hub_download("lewtun/music_genres_small",
                         "data/train-00000-of-00001-63d68663287b1638.parquet", repo_type="dataset")
    mdf = pd.read_parquet(mp)
    music = [_decode(c)[0] for c in mdf["audio"].iloc[:6]]
    music = np.concatenate(music)

    n = int(DUR * SR)
    rng = np.random.default_rng(0)
    cases = []  # (name, spkA, spkB, snr_B_rel_A, music_snr or None)
    pairs = [(ids[i], ids[(i + 1) % len(ids)]) for i in range(min(3, len(ids)))]
    for ai, (a, b) in enumerate(pairs):
        for snr in (0.0, 5.0):
            cases.append((f"2spk_p{ai}_snr{snr:+.0f}", a, b, snr, None))
        cases.append((f"2spk+music_p{ai}_snr0_m5", a, b, 0.0, 5.0))
    for name, a, b, snr, msnr in cases:
        sa = _fit(spk_audio[a], n)
        sb = scale_to_snr(_fit(spk_audio[b][len(spk_audio[b]) // 3:] if len(spk_audio[b]) > 1.5 * n else spk_audio[b], n), sa, snr)
        stems = [sa, sb]
        mix = sa + sb
        mus = None
        if msnr is not None:
            off = int(rng.integers(0, len(music) - n))
            mus = scale_to_snr(music[off:off + n], sa + sb, msnr)
            mix = mix + mus
        peak = np.abs(mix).max() + 1e-9
        k = 0.9 / peak
        np.savez_compressed(DATA / f"{name}.npz", mix=mix * k, stems=np.stack(stems) * k,
                            music=(mus * k if mus is not None else np.zeros(0, np.float32)))
        print("wrote", name)


if __name__ == "__main__":
    main()
