"""Residual accounting (spec Section 8). Four different things, never conflated:

  * raw waveform remainder  r_wave = x - sum_k s_k   (diagnostic ONLY; phase/polarity sensitive)
  * phase-projected STFT remainder  R = X - sum_k M_k X   with  sum_k M_k <= 1
  * non-negative unexplained power  P_unexplained = max(0, |X|^2 - sum_k alpha_k |S_k|^2)
  * unexplained acoustic EVENTS discovered from the spectral remainder (never from r_wave)

Generated stems are not sample-phase-locked to the mixture, so the raw difference can be large
even for a perfect separation: equal-power copies at phase offset phi leave 2*P*(1-cos(phi)),
i.e. +3 dB (quadrature) or +6 dB (polarity inversion) relative to P. The spectral remainders
depend on STFT magnitudes only, so a unit-magnitude phase rotation does not create residual,
while a source that is missing from the inventory still leaves unexplained mass.

Raw stems are inputs only and are never modified. All signals are mono (a declared mean
downmix); nothing is broadcast across array channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.optimize import lsq_linear


@dataclass
class ResidualConfig:
    n_fft: int = 1024
    hop: int = 256
    alpha_max: float = 1.5          # bounded non-negative gain per (block, band, source)
    block_frames: int = 8
    n_bands: int = 16
    smooth_blocks: int = 3          # moving-average length over blocks (smoothness regularisation)
    event_threshold_db: float = -20.0   # unexplained frame energy vs. the loudest mixture frame
    event_min_s: float = 0.08
    event_merge_gap_s: float = 0.05
    floor_power: float = 1e-12


def downmix(x: np.ndarray) -> np.ndarray:
    """Declared mono downmix: mean over channels (the SAM Audio processor rule)."""
    a = np.asarray(x, np.float64)
    return a if a.ndim == 1 else a.mean(axis=1)


def power_db(x: np.ndarray, ref: float = 1.0) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) / ref ** 2 + 1e-30))


def raw_waveform_remainder(x: np.ndarray, stems: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """r_wave(t) = x(t) - sum_k s_k(t) in a declared mono downmix. Diagnostic artifact only."""
    xm = downmix(x)
    total = np.zeros_like(xm)
    for s in stems:
        sm = downmix(s)
        if len(sm) != len(xm):
            raise ValueError("stems must be aligned and the same length as the mixture")
        total += sm
    return xm - total, {"kind": "raw_waveform_diagnostic", "downmix": "mean_over_channels",
                        "gain": 1.0, "alignment_samples": 0, "use": "diagnostic"}


def stft(x: np.ndarray, cfg: ResidualConfig) -> np.ndarray:
    _, _, z = signal.stft(x, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop, boundary="zeros", padded=True,
                          window="hann")
    return z


def istft(z: np.ndarray, cfg: ResidualConfig, n: int) -> np.ndarray:
    _, y = signal.istft(z, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop, boundary=True, window="hann")
    y = y[:n]
    return np.pad(y, (0, n - len(y))) if len(y) < n else y


@dataclass
class Projection:
    unexplained_power: np.ndarray            # (bins, frames)
    residual_stft: np.ndarray                # (bins, frames) complex: X - sum M_k X
    mask_sum_max: float                      # max over bins of sum_k M_k (<= 1 by construction)
    alpha: np.ndarray                        # (K, bands, frames) bounded gains
    projection_error_db: float               # energy of R relative to energy of X
    proxy_waveform: np.ndarray               # istft of R, labelled a projection proxy
    sample_rate: int
    meta: dict = field(default_factory=dict)


def _band_edges(n_bins: int, n_bands: int) -> np.ndarray:
    # roughly log spaced, at least one bin per band
    e = np.unique(np.round(np.geomspace(1, n_bins, n_bands + 1)).astype(int))
    e[0] = 0
    return e


def fit_gains(Px: np.ndarray, Ps: np.ndarray, cfg: ResidualConfig) -> tuple[np.ndarray, np.ndarray]:
    """Bounded non-negative gains alpha in [0, alpha_max] per (source, band, frame).

    Least squares of |X|^2 on sum_k alpha_k |S_k|^2 per (frame block, band), then smoothed over
    blocks. The bound and the smoothing stop the fit from explaining away a missing source."""
    K, n_bins, n_frames = Ps.shape
    edges = _band_edges(n_bins, cfg.n_bands)
    bands = len(edges) - 1
    n_blocks = int(np.ceil(n_frames / cfg.block_frames))
    a = np.ones((K, bands, n_blocks))
    for b in range(n_blocks):
        f0, f1 = b * cfg.block_frames, min((b + 1) * cfg.block_frames, n_frames)
        for j in range(bands):
            k0, k1 = edges[j], max(edges[j + 1], edges[j] + 1)
            A = Ps[:, k0:k1, f0:f1].reshape(K, -1).T
            y = Px[k0:k1, f0:f1].reshape(-1)
            scale = max(float(y.mean()), cfg.floor_power)
            if float(A.sum()) <= cfg.floor_power * A.size:
                a[:, j, b] = 1.0       # no candidate energy here: gain is moot
                continue
            res = lsq_linear(A / scale, y / scale, bounds=(0.0, cfg.alpha_max))
            a[:, j, b] = res.x
    if cfg.smooth_blocks > 1 and n_blocks > 1:
        ker = np.ones(cfg.smooth_blocks) / cfg.smooth_blocks
        pad = cfg.smooth_blocks // 2
        for k in range(K):
            for j in range(bands):
                row = np.pad(a[k, j], pad, mode="edge")
                a[k, j] = np.convolve(row, ker, mode="valid")[:n_blocks]
    per_frame = np.repeat(a, cfg.block_frames, axis=2)[:, :, :n_frames]
    return per_frame, edges


def project_remainder(x: np.ndarray, stems: list[np.ndarray], rate: int,
                      cfg: ResidualConfig | None = None) -> Projection:
    """Phase-projected STFT remainder and non-negative unexplained power for mono stems.

    Wiener-style masks M_k = alpha_k |S_k|^2 / max(|X|^2, sum_j alpha_j |S_j|^2) so that
    sum_k M_k <= 1; known-source masks are NOT normalised to sum to one, because the source
    inventory may be incomplete."""
    cfg = cfg or ResidualConfig()
    xm = downmix(x)
    sm = [downmix(s) for s in stems]
    if any(len(s) != len(xm) for s in sm):
        raise ValueError("stems must be aligned and the same length as the mixture")
    X = stft(xm, cfg)
    Px = np.abs(X) ** 2
    if not sm:
        return Projection(Px, X, 0.0, np.zeros((0, 0, X.shape[1])), 0.0, xm.copy(), rate, {"stems": 0})
    S = np.stack([np.abs(stft(s, cfg)) ** 2 for s in sm])
    alpha, edges = fit_gains(Px, S, cfg)
    A_bins = np.zeros((len(sm), Px.shape[0], Px.shape[1]))
    for j in range(len(edges) - 1):
        k0, k1 = edges[j], max(edges[j + 1], edges[j] + 1)
        A_bins[:, k0:k1, :] = alpha[:, j:j + 1, :]
    A_bins[:, edges[-1]:, :] = alpha[:, -1:, :]
    explained = A_bins * S
    total = explained.sum(axis=0)
    denom = np.maximum(Px, total) + cfg.floor_power
    M = explained / denom
    R = X * (1.0 - M.sum(axis=0))
    unexplained = np.maximum(0.0, Px - total)
    err = 10 * np.log10((np.abs(R) ** 2).sum() / (Px.sum() + cfg.floor_power) + 1e-30)
    return Projection(unexplained, R, float(M.sum(axis=0).max()), alpha, float(err), istft(R, cfg, len(xm)), rate,
                      {"kind": "phase_projected_stft", "stems": len(sm), "n_fft": cfg.n_fft, "hop": cfg.hop,
                       "alpha_max": cfg.alpha_max, "block_frames": cfg.block_frames, "n_bands": cfg.n_bands,
                       "downmix": "mean_over_channels", "proxy_waveform": "istft(R) projection proxy"})


def _spans(e_un: np.ndarray, mixture_peak: float, cfg: ResidualConfig, rate: int) -> list[dict]:
    """Spans where unexplained frame energy exceeds `event_threshold_db` below the loudest mixture
    frame. A span is unexplained ACTIVITY only; it is not by itself a new emitter."""
    peak = max(mixture_peak, 1e-30)
    thr = peak * 10 ** (cfg.event_threshold_db / 10.0)
    on = e_un > thr
    hop_s = cfg.hop / rate
    spans = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            spans.append([i * hop_s, (j + 1) * hop_s, float(10 * np.log10(e_un[i:j + 1].max() / peak + 1e-30))])
            i = j + 1
        else:
            i += 1
    merged: list[list[float]] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= cfg.event_merge_gap_s:
            merged[-1][1] = s[1]
            merged[-1][2] = max(merged[-1][2], s[2])
        else:
            merged.append(s)
    return [{"start_s": a, "end_s": b, "peak_db_rel_mixture": c} for a, b, c in merged
            if b - a >= cfg.event_min_s]


def find_unexplained_events(x: np.ndarray, stems: list[np.ndarray], rate: int,
                            cfg: ResidualConfig | None = None) -> tuple[list[dict], Projection]:
    """Convenience: project, then detect events against the loudest mixture frame."""
    cfg = cfg or ResidualConfig()
    p = project_remainder(x, stems, rate, cfg)
    X = stft(downmix(x), cfg)
    peak = float((np.abs(X) ** 2).sum(axis=0).max())
    p.meta["mixture_frame_peak"] = peak
    return _spans(p.unexplained_power.sum(axis=0), peak, cfg, rate), p


def save_unexplained_power(path, p: Projection, cfg: ResidualConfig) -> None:
    """Optional spectral discovery artifact (float32 to keep bundles small)."""
    np.savez_compressed(path, unexplained_power=p.unexplained_power.astype(np.float32),
                        sample_rate=p.sample_rate, n_fft=cfg.n_fft, hop=cfg.hop)
