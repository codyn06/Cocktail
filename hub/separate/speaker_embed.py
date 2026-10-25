"""Import-safe WavLM speaker-verification embedder adapter (microsoft/wavlm-base-plus-sv).

Never downloads: weights are loaded with ``local_files_only=True`` and never touches the network.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Callable

import numpy as np

MODEL_ID = "microsoft/wavlm-base-plus-sv"
SR = 16000


def _hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    base = Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def _weights_cached() -> bool:
    snaps = _hub_cache() / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return False
    for s in snaps.iterdir():
        if (s / "config.json").exists() and ((s / "model.safetensors").exists()
                                             or (s / "pytorch_model.bin").exists()):
            return True
    return False


def wavlm_sv_status() -> dict:
    mods = {m: importlib.util.find_spec(m) is not None for m in ("torch", "transformers")}
    cached = _weights_cached()
    return {"model": MODEL_ID, "packages": mods, "importable": all(mods.values()),
            "weights_cached": cached, "ready": all(mods.values()) and cached}


def load_wavlm_sv(device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    st = wavlm_sv_status()
    if not st["importable"]:
        missing = [m for m, ok in st["packages"].items() if not ok]
        raise RuntimeError(f"WavLM SV needs packages {missing}; install them in the project env.")
    if not st["weights_cached"]:
        raise RuntimeError(f"{MODEL_ID} is not in the local Hugging Face cache ({_hub_cache()}). "
                           "This adapter never downloads; fetch it yourself with approval, then retry.")
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID, local_files_only=True)
    model = WavLMForXVector.from_pretrained(MODEL_ID, local_files_only=True).to(device).eval()

    def embed(audio: np.ndarray) -> np.ndarray:
        a = np.asarray(audio, dtype=np.float32).ravel()
        inputs = fe(a, sampling_rate=SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            return model(**inputs).embeddings[0].detach().cpu().numpy().astype(np.float64).ravel()

    return embed
