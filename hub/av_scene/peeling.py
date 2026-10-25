"""Bounded residual peeling for >2 simultaneous talkers (pure numpy, separator injected).

The only available separator is 2-speaker (MossFormer2_SS_16K). For a third voice we run it on the
mixture, subtract the projection-fitted estimates (per-STFT-bin complex least squares) from the
mixture, and, if the residual still holds speech-like energy, run the separator again on the
residual. Streams that duplicate (SI-SDR above a threshold) an already-kept stream are dropped.

Quality with a real separator is UNTESTED; tests use fake separators and synthetic noise talkers.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

SR = 16000
N_FFT = 512
HOP = 128
MAX_ROUNDS = 1                 # extra separator passes on the residual (compute bound)
MIN_RESIDUAL_DB = -12.0        # residual energy relative to mixture; below this, stop peeling
DEDUP_SI_SDR_DB = 10.0         # stream is a duplicate of a kept one above this SI-SDR
MIN_STREAM_REL_DB = -20.0      # peeled stream this far below the residual is leakage/noise, drop
RIDGE = 1e-6
EPS = 1e-12

Separator = Callable[[np.ndarray], np.ndarray]


def si_sdr_db(est: np.ndarray, ref: np.ndarray) -> float:
    est = np.asarray(est, np.float64)
    ref = np.asarray(ref, np.float64)
    n = min(len(est), len(ref))
    est, ref = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    a = (est @ ref) / (ref @ ref + EPS)
    t = a * ref
    return float(10 * np.log10((t @ t + EPS) / (((est - t) ** 2).sum() + EPS)))


def _energy(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, np.float64) ** 2))


def _stft(x: np.ndarray, n: int) -> np.ndarray:
    pad = N_FFT
    xp = np.pad(x, (pad, pad + N_FFT))
    win = np.hanning(N_FFT)
    nfr = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(nfr)[:, None]
    return np.fft.rfft(xp[idx] * win, axis=1)  # (frames, bins)


def _istft(S: np.ndarray, n: int) -> np.ndarray:
    win = np.hanning(N_FFT)
    frames = np.fft.irfft(S, N_FFT, axis=1) * win
    out = np.zeros(HOP * (len(frames) - 1) + N_FFT)
    norm = np.zeros_like(out)
    for i, f in enumerate(frames):
        out[i * HOP:i * HOP + N_FFT] += f
        norm[i * HOP:i * HOP + N_FFT] += win ** 2
    out = out / np.maximum(norm, 1e-8)
    return out[N_FFT:N_FFT + n]


def subtract_fitted(mix: np.ndarray, sources: list[np.ndarray]) -> np.ndarray:
    """mix minus its per-STFT-bin complex least-squares projection onto `sources`."""
    mix = np.asarray(mix, np.float64)
    n = len(mix)
    if not sources:
        return mix.copy()
    M = _stft(mix, n)
    S = np.stack([_stft(np.asarray(s, np.float64)[:n], n) for s in sources], axis=-1)  # (T, F, K)
    R = np.empty_like(M)
    k = S.shape[-1]
    for f in range(M.shape[1]):
        A = S[:, f, :]
        G = A.conj().T @ A
        G = G + RIDGE * (np.trace(G).real / k + EPS) * np.eye(k)
        g = np.linalg.solve(G, A.conj().T @ M[:, f])
        R[:, f] = M[:, f] - A @ g
    return _istft(R, n)


def dedup_streams(streams: list[np.ndarray], thr_db: float = DEDUP_SI_SDR_DB) -> list[np.ndarray]:
    """Keep streams in order, dropping any whose SI-SDR against an already-kept stream exceeds thr_db."""
    kept: list[np.ndarray] = []
    for s in streams:
        if _energy(s) <= 0.0:
            continue
        if any(si_sdr_db(s, k) > thr_db for k in kept):
            continue
        kept.append(s)
    return kept


def peel_residual_speech(
    mix: np.ndarray,
    separator: Separator,
    max_rounds: int = MAX_ROUNDS,
    min_residual_db: float = MIN_RESIDUAL_DB,
    dedup_si_sdr_db: float = DEDUP_SI_SDR_DB,
) -> tuple[list[np.ndarray], dict]:
    """Run `separator` on `mix` (16 kHz mono), then up to `max_rounds` more times on the residual.

    Returns (streams, diagnostics). Separator calls <= 1 + max_rounds.
    """
    mix = np.asarray(mix, np.float64)
    n = len(mix)
    mix_e = _energy(mix)
    diag: dict = {"separator_calls": 0, "rounds": [], "stop_reason": "max_rounds"}
    if mix_e <= 0.0 or n < N_FFT:
        diag["stop_reason"] = "empty_mix"
        return [], diag

    def run(x: np.ndarray) -> list[np.ndarray]:
        diag["separator_calls"] += 1
        out = np.asarray(separator(x.astype(np.float32)), np.float64)
        return [o[:n] for o in out.reshape(-1, out.shape[-1])]

    first = dedup_streams(run(mix), dedup_si_sdr_db)
    loud = [f for f in first if 10 * np.log10((_energy(f) + EPS) / (mix_e + EPS)) >= MIN_STREAM_REL_DB]
    first = loud or first[:1]          # an empty separator channel (bleed only) is not a voice
    streams = list(first)
    for r in range(max(0, int(max_rounds))):
        resid = subtract_fitted(mix, streams)
        rel_db = float(10 * np.log10((_energy(resid) + EPS) / (mix_e + EPS)))
        info = {"round": r + 1, "residual_db": rel_db}
        diag["rounds"].append(info)
        if rel_db < min_residual_db:
            diag["stop_reason"] = "residual_below_floor"
            break
        resid_e = _energy(resid)
        cand = run(resid)
        added = 0
        for c in cand:
            if 10 * np.log10((_energy(c) + EPS) / (resid_e + EPS)) < MIN_STREAM_REL_DB:
                continue
            if any(si_sdr_db(c, k) > dedup_si_sdr_db for k in streams):
                continue
            streams.append(c)
            added += 1
        info["added"] = added
        if added == 0:
            diag["stop_reason"] = "nothing_new"
            break
    diag["n_streams"] = len(streams)
    return streams, diag
