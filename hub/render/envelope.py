"""Perceptual visual-envelope branch (spec Section 10).

ITU-R BS.1770 K-weighting (high shelf + RLB-style high-pass biquads) is applied ONLY inside this
branch, to a copy of the signal. Exported stems, the raw-array spatial branch and unweighted dBFS
diagnostics are never touched. The weighted short-window RMS here is NOT LUFS (no gating or
integration per BS.1770).

K-weighting is not a brick-wall low-cut: at 40 Hz it removes only ~6 dB. Rumble/handling rejection
therefore comes from the combination of
  1. the weighted level threshold,
  2. a low-frequency-dominance test (most energy below `lf_cutoff_hz` means "rumble", not a source),
  3. a hysteresis gate whose reopening needs a fresh onset (a decaying reverberant tail cannot
     reopen the flag), and
  4. the association / mask validity masks supplied by the caller.

Timing: the active flag and its onset come from the gate, not from the display smoothing. The
~30 ms attack / ~180 ms release smoothing only shapes the glow amplitude, and its attack lag is
compensated by evaluating the input `attack_ms * ln 2` ahead (file mode), so smoothing adds no
onset delay. The trailing-window gate crossing is refined by walking back to where a short window
first exceeded `activate - backtrack_db`. Captured audio timing is never shifted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

# BS.1770-5 Annex 1 filter prototypes (stage 1: shelving; stage 2: high-pass).
_SHELF = dict(f0=1681.974450955533, gain_db=3.999843853973347, q=0.7071752369554196)
_HP = dict(f0=38.13547087602444, q=0.5003270373238773)
MIN_RATE_HZ = 16_000
MAX_RATE_HZ = 384_000


def _validate_rate(fs: int) -> int:
    if int(fs) != fs or not (MIN_RATE_HZ <= fs <= MAX_RATE_HZ):
        raise ValueError(f"K-weighting is validated for integer rates {MIN_RATE_HZ}-{MAX_RATE_HZ} Hz, got {fs}")
    return int(fs)


def k_weighting_sos(fs: int) -> np.ndarray:
    """Two biquads (rows: b0 b1 b2 1 a1 a2) designed for `fs` from the analog prototypes."""
    fs = _validate_rate(fs)
    # stage 1: high shelf
    k = math.tan(math.pi * _SHELF["f0"] / fs)
    vh = 10 ** (_SHELF["gain_db"] / 20.0)
    vb = vh ** 0.4996667741545416
    q = _SHELF["q"]
    a0 = 1.0 + k / q + k * k
    shelf = [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0,
             1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    # stage 2: high-pass
    k = math.tan(math.pi * _HP["f0"] / fs)
    q = _HP["q"]
    a0 = 1.0 + k / q + k * k
    hp = [1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    sos = np.array([shelf, hp], dtype=np.float64)
    for row in sos:  # stability self-check
        if np.any(np.abs(np.roots([1.0, row[4], row[5]])) >= 1.0):
            raise ValueError(f"unstable K-weighting section at fs={fs}")
    return sos


def k_weight(x: np.ndarray, fs: int) -> np.ndarray:
    """Return a K-weighted COPY of x (1-D)."""
    return signal.sosfilt(k_weighting_sos(fs), np.asarray(x, np.float64))


def k_response_db(fs: int, freqs_hz) -> np.ndarray:
    _, h = signal.sosfreqz(k_weighting_sos(fs), worN=np.asarray(freqs_hz, float), fs=fs)
    return 20 * np.log10(np.abs(h))


def k_response_reference_db(fs: int, freqs_hz) -> np.ndarray:
    """Analog-prototype reference with bilinear frequency pre-warping, computed independently."""
    f = np.asarray(freqs_hz, float)
    out = np.zeros_like(f)
    vh = 10 ** (_SHELF["gain_db"] / 20.0)
    vb = vh ** 0.4996667741545416
    for name in ("shelf", "hp"):
        f0 = _SHELF["f0"] if name == "shelf" else _HP["f0"]
        q = _SHELF["q"] if name == "shelf" else _HP["q"]
        w = 1j * np.tan(np.pi * f / fs) / np.tan(np.pi * f0 / fs)
        # shelf: unity at DC, +gain_db above f0; high-pass: unity far above f0
        h = (1 + vb * w / q + vh * w * w) / (1 + w / q + w * w) if name == "shelf" else (w * w) / (1 + w / q + w * w)
        if name == "hp":  # BS.1770 specifies b = [1, -2, 1] unnormalised, so HF gain is a0 = 1 + K/Q + K^2
            k0 = math.tan(math.pi * f0 / fs)
            h = h * (1 + k0 / q + k0 * k0)
        out += 20 * np.log10(np.abs(h))
    return out


# --------------------------------------------------------------------------- configuration
@dataclass
class EnvelopeConfig:
    weighting: str = "k"                # "k" (BS.1770) or "none"; A-weighting needs its own verified response
    rms_window_ms: float = 40.0         # display/gate window (one frame only at 25 fps)
    fine_window_ms: float = 4.0         # onset back-tracking window
    fine_hop_ms: float = 1.0
    attack_ms: float = 30.0
    release_ms: float = 180.0
    floor_dbfs: float = -100.0
    activate_dbfs_k: float = -45.0
    sustain_dbfs_k: float = -51.0
    tail_hold_ms: float = 150.0         # continuous time below sustain before the flag drops
    peak_drop_db: float = 12.0          # also "below" when this far under the recent peak (reverb tail)
    peak_decay_db_per_s: float = 30.0
    reopen_lockout_ms: float = 600.0    # fallback when RT60 is unknown
    onset_rise_db: float = 6.0          # fresh-onset rise needed to reopen during the lockout
    onset_lookback_ms: float = 60.0
    backtrack_db: float = 12.0
    backtrack_max_ms: float = 60.0
    lf_cutoff_hz: float = 100.0
    lf_fraction_max: float = 0.7        # more LF energy than this = rumble/handling, may not open
    lf_window_ms: float = 200.0         # long enough for a stable fraction of band-limited rumble noise
    display_range_db: float = 18.0      # glow ramps from activate-6 dB up over this range

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnvelopeResult:
    frame_times_us: np.ndarray          # common-clock PTS of each video frame
    level_dbfs_k: np.ndarray            # weighted short-window RMS per frame (centred window)
    level_dbfs: np.ndarray              # unweighted diagnostic
    gate_active: np.ndarray             # bool per frame from the hysteresis gate
    active: np.ndarray                  # gate AND association AND mask validity
    glow: np.ndarray                    # 0..1 smoothed, onset-compensated display amplitude
    segments_us: list[tuple[int, int]]  # half-open gate intervals (onset back-tracked)
    config: dict
    k_weighting: dict


# --------------------------------------------------------------------------- helpers
def _trailing_ms(x2: np.ndarray, win: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([[0.0], x2]))
    idx = np.arange(1, len(x2) + 1)
    lo = np.maximum(idx - win, 0)
    return (c[idx] - c[lo]) / win


def _forward_ms(x2: np.ndarray, win: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([[0.0], x2]))
    idx = np.arange(len(x2))
    hi = np.minimum(idx + win, len(x2))
    return (c[hi] - c[idx]) / win


def _centred_ms(x2: np.ndarray, win: int) -> np.ndarray:
    pad = win // 2
    c = np.cumsum(np.concatenate([[0.0], x2, np.zeros(win)]))
    idx = np.arange(len(x2))
    lo = np.clip(idx - pad, 0, len(x2))
    hi = np.clip(idx - pad + win, 0, len(x2))
    return (c[hi] - c[lo]) / win


def _db(ms: np.ndarray, floor_dbfs: float) -> np.ndarray:
    return np.maximum(10 * np.log10(np.maximum(ms, 1e-20)), floor_dbfs)


def run_gate(level_k: np.ndarray, level_short: np.ndarray, lf_frac: np.ndarray, hop_s: float,
             cfg: EnvelopeConfig, trusted_offsets_s: list[float] | None = None) -> list[tuple[float, float]]:
    """Hysteresis gate on a fine-hop weighted level series. Returns (onset_s, offset_s) segments."""
    n = len(level_k)
    hold = max(1, int(round(cfg.tail_hold_ms / 1000.0 / hop_s)))
    look = max(1, int(round(cfg.onset_lookback_ms / 1000.0 / hop_s)))
    lock = int(round(cfg.reopen_lockout_ms / 1000.0 / hop_s))
    back_max = int(round(cfg.backtrack_max_ms / 1000.0 / hop_s))
    trusted = sorted(trusted_offsets_s or [])
    segs: list[tuple[float, float]] = []
    active = False
    onset_i = 0
    below = 0
    below_start = 0
    peak = -1e9
    lock_until = -1
    for i in range(n):
        lv = level_k[i]
        if not active:
            if lv >= cfg.activate_dbfs_k and lf_frac[i] <= cfg.lf_fraction_max:
                fresh = lv - level_k[max(0, i - look):i + 1].min() >= cfg.onset_rise_db
                if i >= lock_until or fresh:
                    j = i
                    thr = cfg.activate_dbfs_k - cfg.backtrack_db
                    while j > 0 and i - j < back_max and level_short[j - 1] >= thr:
                        j -= 1
                    active, onset_i, peak, below = True, j, lv, 0
        else:
            peak = max(lv, peak - cfg.peak_decay_db_per_s * hop_s)
            is_below = lv < max(cfg.sustain_dbfs_k, peak - cfg.peak_drop_db)
            if is_below:
                if below == 0:
                    below_start = i
                below += 1
            else:
                below = 0
            forced = None
            for t_o in trusted:
                if t_o >= onset_i * hop_s and i * hop_s >= t_o + cfg.tail_hold_ms / 1000.0:
                    forced = t_o
                    break
            if forced is not None or below >= hold:
                if forced is None:
                    # the trailing window lags the true end: walk back to the last sample where the short window was still up
                    j, thr_end = below_start, cfg.activate_dbfs_k - cfg.backtrack_db
                    while j > onset_i + 1 and below_start - j < back_max and level_short[j - 1] < thr_end:
                        j -= 1
                    below_start = j
                off_s = forced if forced is not None else below_start * hop_s
                segs.append((onset_i * hop_s, max(off_s, (onset_i + 1) * hop_s)))
                active, lock_until, below = False, i + lock, 0
    if active:
        segs.append((onset_i * hop_s, n * hop_s))
    return segs


def compute_envelope(x: np.ndarray, fs: int, frame_times_us, cfg: EnvelopeConfig | None = None, *,
                     start_us: int = 0, assoc_valid=None, mask_valid=None,
                     trusted_offsets_us: list[int] | None = None) -> EnvelopeResult:
    """Envelope, hysteresis gate and glow for a mono signal, sampled at video-frame timestamps.

    `x` is read-only: weighting runs on a copy. `start_us` is the common-clock time of sample 0."""
    cfg = cfg or EnvelopeConfig()
    x = np.asarray(x, np.float64)
    if x.ndim != 1:
        raise ValueError("envelope expects a mono signal; declare the downmix first")
    fs = _validate_rate(fs)
    if cfg.weighting == "k":
        xw = k_weight(x, fs)
        kinfo = {"standard": "ITU-R BS.1770-5 Annex 1", "sections": k_weighting_sos(fs).tolist(),
                 "sample_rate": fs, "note": "weighted short-window RMS, not LUFS"}
    elif cfg.weighting == "none":
        xw, kinfo = x.copy(), {"standard": None, "sample_rate": fs}
    else:
        raise NotImplementedError("only 'k' and 'none' are implemented; A-weighting needs its own verified response")

    w_gate = max(1, int(round(cfg.rms_window_ms * fs / 1000.0)))
    w_fine = max(1, int(round(cfg.fine_window_ms * fs / 1000.0)))
    hop = max(1, int(round(cfg.fine_hop_ms * fs / 1000.0)))
    hop_s = hop / fs

    # fine, trailing series for the gate
    lvl_k_full = _db(_trailing_ms(xw * xw, w_gate), cfg.floor_dbfs)
    lvl_short_full = _db(_trailing_ms(xw * xw, w_fine), cfg.floor_dbfs)
    # LF-dominance uses a zero-phase low-pass and a FORWARD window (file mode) so a sine onset is
    # judged on its first 40 ms rather than on the low-pass filter's start-up transient.
    sos_lf = signal.butter(2, cfg.lf_cutoff_hz, "low", fs=fs, output="sos")
    lf2 = signal.sosfiltfilt(sos_lf, x) ** 2
    w_lf = max(w_gate, int(round(cfg.lf_window_ms * fs / 1000.0)))

    def _frac(window_fn):
        tot = window_fn(x * x, w_lf)
        return np.where(tot > 1e-12, window_fn(lf2, w_lf) / np.maximum(tot, 1e-20), 0.0)

    # either window calling it rumble is enough (the trailing one covers the tail after a source ends)
    lf_frac_full = np.maximum(_frac(_forward_ms), _frac(_trailing_ms))
    sel = np.arange(0, len(x), hop)           # index i <-> sample i*hop <-> time i*hop_s (trailing window end)
    trusted_s = [(t - start_us) / 1e6 for t in (trusted_offsets_us or [])]
    segs_s = run_gate(lvl_k_full[sel], lvl_short_full[sel], lf_frac_full[sel], hop_s, cfg, trusted_s)
    segs_us = [(int(round(start_us + a * 1e6)), int(round(start_us + b * 1e6))) for a, b in segs_s]

    # per-frame levels with a CENTRED window at each frame PTS
    ft = np.asarray(frame_times_us, np.int64)
    s_idx = np.clip(np.round((ft - start_us) * fs / 1e6).astype(int), 0, len(x) - 1)
    lv_k = _db(_centred_ms(xw * xw, w_gate)[s_idx], cfg.floor_dbfs)
    lv_u = _db(_centred_ms(x * x, w_gate)[s_idx], cfg.floor_dbfs)

    gate = np.zeros(len(ft), bool)
    for a, b in segs_us:
        gate |= (ft >= a) & (ft < b)
    active = gate.copy()
    if assoc_valid is not None:
        active &= np.asarray(assoc_valid, bool)
    if mask_valid is not None:
        active &= np.asarray(mask_valid, bool)

    # glow: onset-compensated lookahead, then attack/release one-pole at the actual frame spacing
    comp_us = int(round(cfg.attack_ms * math.log(2.0) * 1000.0))
    ahead = np.zeros(len(ft), bool)
    for a, b in segs_us:
        ahead |= (ft + comp_us >= a) & (ft + comp_us < b)
    if assoc_valid is not None:
        ahead &= np.asarray(assoc_valid, bool)
    if mask_valid is not None:
        ahead &= np.asarray(mask_valid, bool)
    lv_ahead = _db(_centred_ms(xw * xw, w_gate)[np.clip(s_idx + int(comp_us * fs / 1e6), 0, len(x) - 1)],
                   cfg.floor_dbfs)
    drive = np.clip((lv_ahead - (cfg.activate_dbfs_k - 6.0)) / cfg.display_range_db, 0.0, 1.0) * ahead
    glow = np.zeros(len(ft))
    state = 0.0
    for i in range(len(ft)):
        dt = (ft[i] - ft[i - 1]) / 1e6 if i else 0.04
        tau = cfg.attack_ms / 1000.0 if drive[i] > state else cfg.release_ms / 1000.0
        state += (1.0 - math.exp(-dt / tau)) * (drive[i] - state)
        glow[i] = state
    return EnvelopeResult(ft, lv_k, lv_u, gate, active, glow, segs_us, cfg.to_dict(), kinfo)
