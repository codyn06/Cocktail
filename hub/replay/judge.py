"""Candidate ranker for SAM Audio stems: SAM-Audio judge model with a CLAP fallback.

Import-safe: nothing heavy (torch, sam_audio, laion_clap) is imported until `rank` is first called, and
`availability()` only inspects installed packages and the local Hugging Face cache; it never loads a model,
downloads, or reads credentials. Real-model behaviour is UNTESTED here (interfaces were read from the
installed sam_audio 0.1.0 source: `sam_audio.model.judge.SAMAudioJudgeModel`,
`sam_audio.processor.SAMAudioJudgeProcessor`, `sam_audio.ranking.clap`).

`rank(candidates, prompt, sample_rate, mix=None) -> list[float]` (higher = better). The judge scores a
candidate against the MIXTURE it came from, so it needs `mix`; without it the judge backend raises
`RankerUnavailable` and the caller falls back (judge -> CLAP -> unranked).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Sequence

import numpy as np

JUDGE_REPO = "facebook/sam-audio-judge"
CLAP_REPO = "lukewys/laion_clap"
CLAP_FILE = "630k-best.pt"
RANK_RATE = 48_000


class RankerUnavailable(RuntimeError):
    """The requested ranking backend cannot run (package/cache missing, or inputs it needs were not given)."""


def _hf_cache_dir() -> Path:
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var])
    home = os.environ.get("HF_HOME")
    return (Path(home) / "hub") if home else Path.home() / ".cache" / "huggingface" / "hub"


def _cached(repo: str, filename: str | None = None) -> bool:
    root = _hf_cache_dir() / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not root.is_dir():
        return False
    snaps = [d for d in root.iterdir() if d.is_dir()]
    return any((d / filename).exists() if filename else any(d.iterdir()) for d in snaps)


def _has(pkg: str) -> bool:
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError):
        return False


def availability() -> dict:
    """{backend: {"installed": bool, "cached": bool, "ok": bool}}; nothing is loaded, no network, no tokens."""
    judge_pkg = _has("sam_audio") and _has("torch")
    clap_pkg = _has("laion_clap") and _has("torch")
    j_cache, c_cache = _cached(JUDGE_REPO), _cached(CLAP_REPO, CLAP_FILE)
    return {"judge": {"installed": judge_pkg, "cached": j_cache, "ok": judge_pkg and j_cache, "repo": JUDGE_REPO},
            "clap": {"installed": clap_pkg, "cached": c_cache, "ok": clap_pkg and c_cache, "repo": CLAP_REPO}}


def _to_rate(x: np.ndarray, rate: int):
    import torch

    t = torch.as_tensor(np.asarray(x, np.float32))
    if rate != RANK_RATE:
        import torchaudio

        t = torchaudio.functional.resample(t, rate, RANK_RATE)
    return t


class JudgeRanker:
    """Lazy judge/CLAP scorer. ``backend``: "judge", "clap", or "auto" (judge if it is available, else CLAP).
    Models load on the first `rank` call (an explicit action; weights must already be in the HF cache)."""

    def __init__(self, backend: str = "auto", device: str = "cpu"):
        if backend not in ("auto", "judge", "clap"):
            raise ValueError("backend must be auto|judge|clap")
        self.backend, self.device = backend, device
        self._judge = None      # (model, processor)
        self._clap = None

    def availability(self) -> dict:
        return availability()

    def resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        av = availability()
        for name in ("judge", "clap"):
            if av[name]["ok"]:
                return name
        raise RankerUnavailable("no ranking backend installed and cached (see availability())")

    def rank(self, candidates: Sequence[np.ndarray], prompt: str, sample_rate: int,
             mix: np.ndarray | None = None) -> list[float]:
        if not len(candidates):
            return []
        name = self.resolve_backend()
        av = availability()[name]
        if not av["ok"]:
            raise RankerUnavailable(f"{name} backend unavailable (installed={av['installed']}, cached={av['cached']})")
        if name == "judge":
            if mix is None:
                raise RankerUnavailable("judge scores a candidate against its mixture; pass mix=")
            return self._rank_judge(candidates, prompt, sample_rate, mix)
        return self._rank_clap(candidates, prompt, sample_rate)

    # ------------------------------------------------------------------ backends (UNTESTED here)
    def _rank_judge(self, candidates, prompt, sample_rate, mix) -> list[float]:
        import torch

        if self._judge is None:
            from sam_audio.model.judge import SAMAudioJudgeModel
            from sam_audio.processor import SAMAudioJudgeProcessor

            self._judge = (SAMAudioJudgeModel.from_pretrained(JUDGE_REPO).eval().to(self.device),
                           SAMAudioJudgeProcessor.from_pretrained(JUDGE_REPO))
        model, proc = self._judge
        n = len(candidates)
        m = _to_rate(mix, sample_rate)
        with torch.inference_mode():
            processed = proc(text=[prompt] * n, input_audio=[m[None] for _ in range(n)],
                             separated_audio=[_to_rate(c, sample_rate)[None] for c in candidates],
                             return_tensors="pt", padding=True, sampling_rate=RANK_RATE)
            res = model(**processed.to(self.device))
        return [float(v) for v in res.overall.reshape(-1).float().cpu()]

    def _rank_clap(self, candidates, prompt, sample_rate) -> list[float]:
        import torch

        if self._clap is None:
            from sam_audio.model.config import ClapRankerConfig
            from sam_audio.ranking.clap import ClapRanker

            self._clap = ClapRanker(ClapRankerConfig()).to(self.device)
        batch = torch.stack([_to_rate(c, sample_rate) for c in _pad_equal(candidates)])[None]  # (1, N, T)
        scores = self._clap(extracted_audio=[batch[0]], descriptions=[prompt], sample_rate=RANK_RATE)
        return [float(v) for v in scores.reshape(-1).float().cpu()]


def _pad_equal(xs: Sequence[np.ndarray]) -> list[np.ndarray]:
    n = max(len(x) for x in xs)
    return [np.pad(np.asarray(x, np.float32), (0, n - len(x))) for x in xs]
