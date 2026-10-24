"""Loaders for candidate separators. Each returns (fn, n_spk, info) with
fn(window[T] at 16 kHz, on cuda) -> [n_spk, T]. Weights go to hub/models/ (gitignored)."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torchaudio.functional as AF

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _fit_len(o: torch.Tensor, T: int) -> torch.Tensor:
    if o.shape[-1] < T:
        o = torch.nn.functional.pad(o, (0, T - o.shape[-1]))
    return o[..., :T]


def load_sepformer(source: str, native_sr: int, device="cuda"):
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy
    m = SepformerSeparation.from_hparams(
        source=source, savedir=str(MODELS_DIR / source.replace("/", "__")),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    m.mods.eval()

    def fn(w: torch.Tensor) -> torch.Tensor:
        x = w[None]
        if native_sr != 16000:
            x = AF.resample(x, 16000, native_sr)
        est = m.separate_batch(x)[0].T  # [n_spk, T']
        if native_sr != 16000:
            est = AF.resample(est, native_sr, 16000)
        return _fit_len(est, w.numel())
    n_params = sum(p.numel() for p in m.mods.parameters())
    return fn, 2, {"params_M": n_params / 1e6, "native_sr": native_sr}


def load_convtasnet_libri2mix(device="cuda"):
    from asteroid.models import ConvTasNet
    m = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k").to(device).eval()

    def fn(w: torch.Tensor) -> torch.Tensor:
        return _fit_len(m(w[None, None])[0], w.numel())
    return fn, 2, {"params_M": sum(p.numel() for p in m.parameters()) / 1e6, "native_sr": 16000}


REGISTRY = {
    "sepformer-whamr16k": lambda d: load_sepformer("speechbrain/sepformer-whamr16k", 16000, d),
    "sepformer-libri2mix-8k": lambda d: load_sepformer("speechbrain/sepformer-libri2mix", 8000, d),
    "convtasnet-libri2mix-16k": load_convtasnet_libri2mix,
}
