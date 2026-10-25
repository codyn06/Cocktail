"""Separation metrics for the AV multi-source diagnostics (pure numpy; no model imports).

- `si_sdr` / `si_sdr_delay`: scale-invariant SDR; the delay-compensated form first removes a bulk
  lag (cross-correlation, within +-`max_lag_s`) between estimate and reference, so a codec/model
  delay is not scored as separation error.
- `si_sdri`: improvement over the mixture, `None` when the reference is silent (SI-SDR against an
  all-zero reference is undefined; silent targets are scored with `output_dbfs` instead).
- `selection_correct`: did the output follow the prompted source rather than the other one.
- `label_accuracy`, `active_spans`: small helpers shared by the matrix runner.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
SILENT_DBFS = -60.0           # a reference quieter than this (RMS over the clip) counts as silent


def rms_dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(20 * np.log10(max(float(np.sqrt(np.mean(x * x))) if x.size else 0.0, 1e-9)))


def is_silent(ref: np.ndarray, thresh_dbfs: float = SILENT_DBFS) -> bool:
    return rms_dbfs(ref) < thresh_dbfs


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = np.asarray(est, np.float64)
    ref = np.asarray(ref, np.float64)
    est, ref = est - est.mean(), ref - ref.mean()
    a = (est @ ref) / (ref @ ref + EPS)
    t = a * ref
    return float(10 * np.log10((t @ t + EPS) / (((est - t) ** 2).sum() + EPS)))


def best_lag(est: np.ndarray, ref: np.ndarray, max_lag: int) -> int:
    """Lag L (samples) maximising correlation of est[n + L] with ref[n]; L > 0 means est is delayed."""
    n = min(len(est), len(ref))
    e, r = np.asarray(est[:n], np.float64), np.asarray(ref[:n], np.float64)
    if not (np.any(e) and np.any(r)):
        return 0
    size = 1 << (2 * n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(e, size) * np.conj(np.fft.rfft(r, size)), size)
    lags = np.arange(-max_lag, max_lag + 1)
    return int(lags[np.argmax(cc[lags % size])])


def align(est: np.ndarray, ref: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Shift `est` by its best lag against `ref` and trim both to the common span."""
    n = min(len(est), len(ref))
    est, ref = np.asarray(est[:n]), np.asarray(ref[:n])
    lag = best_lag(est, ref, max_lag)
    if lag > 0:
        return est[lag:], ref[: n - lag], lag
    if lag < 0:
        return est[: n + lag], ref[-lag:], lag
    return est, ref, 0


def si_sdr_delay(est: np.ndarray, ref: np.ndarray, sr: int, max_lag_s: float = 0.05) -> tuple[float, int]:
    """(SI-SDR after delay compensation, lag in samples). Reference must not be silent."""
    e, r, lag = align(est, ref, int(round(max_lag_s * sr)))
    return si_sdr(e, r), lag


def si_sdri(est: np.ndarray, ref: np.ndarray, mix: np.ndarray, sr: int, max_lag_s: float = 0.05) -> dict | None:
    """SI-SDR improvement of `est` over `mix` against `ref`, both delay-compensated.
    None when `ref` is silent."""
    if is_silent(ref):
        return None
    got, lag = si_sdr_delay(est, ref, sr, max_lag_s)
    base, mix_lag = si_sdr_delay(mix, ref, sr, max_lag_s)
    return {"si_sdr": got, "mix_si_sdr": base, "si_sdri": got - base, "lag_ms": 1000.0 * lag / sr,
            "mix_lag_ms": 1000.0 * mix_lag / sr}


def output_dbfs(est: np.ndarray) -> float:
    """RMS level of an output that should be silent (target-absent cases); lower is better."""
    return rms_dbfs(est)


def selection_correct(est: np.ndarray, ref_target: np.ndarray, ref_other: np.ndarray, sr: int,
                      max_lag_s: float = 0.05) -> dict | None:
    """True when the output is closer (SI-SDR) to the prompted source than to the other one.
    None unless both references are non-silent."""
    if is_silent(ref_target) or is_silent(ref_other):
        return None
    t, _ = si_sdr_delay(est, ref_target, sr, max_lag_s)
    o, _ = si_sdr_delay(est, ref_other, sr, max_lag_s)
    return {"sdr_vs_target": t, "sdr_vs_other": o, "correct": bool(t > o), "margin_db": t - o}


def label_accuracy(pred: list[str], truth: list[str]) -> float | None:
    if not truth or len(pred) != len(truth):
        return None
    return float(np.mean([p == t for p, t in zip(pred, truth)]))


def active_spans(x: np.ndarray, sr: int, *, win_s: float = 0.025, rel_db: float = 28.0, floor_dbfs: float = -55.0,
                 min_gap_s: float = 0.25, min_len_s: float = 0.08) -> list[tuple[float, float]]:
    """Energy VAD on a CLEAN reference: windows within `rel_db` of the loudest window and above
    `floor_dbfs`; gaps shorter than `min_gap_s` are bridged, spans shorter than `min_len_s` dropped."""
    x = np.asarray(x, np.float64)
    w = max(1, int(round(win_s * sr)))
    n = len(x) // w
    if n == 0:
        return []
    e = 10 * np.log10(np.mean(x[: n * w].reshape(n, w) ** 2, axis=1) + 1e-12)
    on = (e > max(e.max() - rel_db, floor_dbfs)) & (e > floor_dbfs)
    spans: list[list[float]] = []
    for i in np.flatnonzero(on):
        s, t = i * win_s, (i + 1) * win_s
        if spans and s - spans[-1][1] <= min_gap_s:
            spans[-1][1] = t
        else:
            spans.append([s, t])
    return [(round(a, 3), round(b, 3)) for a, b in spans if b - a >= min_len_s]


def spans_overlap_s(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return float(sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b))
