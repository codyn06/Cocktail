"""Per-track activity gate combining TalkNet with AV-TSE output energy (pure numpy, no I/O)."""
from __future__ import annotations

import numpy as np

TALKNET_THRESHOLD = 0.5
# TSE energy relative to the mixture frame energy (dB). A visible SILENT bystander's TSE output still
# bleeds the other talker at ~-18.7 dB (measured); two simultaneous equal talkers each sit near -3 dB
# and a quieter one around -6..-9 dB. -11 dB leaves ~7.7 dB of headroom above the measured bleed while
# still admitting a talker ~8 dB below the mixture.
TSE_ACTIVE_REL_MIX_DB = -11.0
# Required lead over the strongest OTHER track's TSE output (dB). Bleed of a shared talker appears in
# every silent track roughly equally, so a real talker must stand clearly above its peers.
TSE_LEAD_DB = 6.0
# Mouth-sync (TalkNet score) must not be strongly negative. Scale-agnostic: only scores below this
# veto TSE-only activation. NaN (no face evidence) counts as 0.
MIN_SYNC_SCORE = -0.5
# Absolute dBFS floor so near-silent frames never activate through relative comparisons.
TSE_ABS_FLOOR_DBFS = -50.0
SMOOTH_FRAMES = 3  # majority vote over this many frames (odd) to drop single-frame flicker


def _frame_db(x, n: int, spf: float) -> np.ndarray:
    x = np.asarray(x, np.float64)
    out = np.full(n, -120.0)
    for i in range(n):
        seg = x[int(i * spf):int((i + 1) * spf)]
        if seg.size:
            out[i] = 10 * np.log10(float(np.mean(seg ** 2)) + 1e-12)
    return out


def combine_activity(talknet_scores: dict, tse_out: dict, mix, fps: float = 25.0, *,
                     rel_mix_db: float = TSE_ACTIVE_REL_MIX_DB, lead_db: float = TSE_LEAD_DB,
                     min_sync: float = MIN_SYNC_SCORE, sr: int = 16000):
    """Return ({track: bool[n_frames]}, diagnostics).

    Active = TalkNet >= 0.5, OR (TSE energy >= rel_mix_db re. mixture AND >= abs floor AND lead_db above
    every other track's TSE energy AND mouth-sync score >= min_sync)."""
    n = max((len(s) for s in talknet_scores.values()), default=0)
    spf = sr / fps
    mix_db = _frame_db(mix, n, spf)
    tse_db = {t: _frame_db(x, n, spf) for t, x in tse_out.items()}
    active, diag = {}, {}
    for t, sc in talknet_scores.items():
        s = np.zeros(n)
        s[:len(sc)] = np.nan_to_num(np.asarray(sc, float), nan=0.0)
        tk = s >= TALKNET_THRESHOLD
        if t in tse_db:
            d = tse_db[t]
            others = [v for k, v in tse_db.items() if k != t]
            lead = d - np.max(others, axis=0) if others else np.full(n, np.inf)
            tse_ok = (d - mix_db >= rel_mix_db) & (d >= TSE_ABS_FLOOR_DBFS) & (lead >= lead_db) & (s >= min_sync)
            if SMOOTH_FRAMES > 1:
                tse_ok = np.convolve(tse_ok.astype(float), np.ones(SMOOTH_FRAMES), "same") > SMOOTH_FRAMES / 2
        else:
            tse_ok = np.zeros(n, bool)
        active[t] = tk | tse_ok
        diag[t] = {"talknet_frames": int(tk.sum()), "tse_rescued_frames": int((tse_ok & ~tk).sum()),
                   "median_rel_mix_db": float(np.median(tse_db[t] - mix_db)) if t in tse_db else None}
    return active, diag
