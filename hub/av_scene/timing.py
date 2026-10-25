"""Signed sync gates (spec Section 10). Positive always means LATE.

    delta_audio_ms = t_audio_onset - t_visual_truth   in [-15 - q, +45 + q]
    delta_mask_ms  = t_mask_onset  - t_visual_truth   in [-15 - q, +45 + q]
    delta_skew_ms  = t_mask_onset  - t_audio_onset    abs(.) <= 20 + q

q = 1000 / (2 * FPS), or half the local bracketing frame interval for variable frame rate.
The natural 0-35 ms acoustic time of flight is part of the +45 ms budget; it is never added
again and exported stems are never advanced to remove it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SyncResult:
    delta_audio_ms: float
    delta_mask_ms: float
    delta_skew_ms: float
    q_ms: float
    audio_ok: bool
    mask_ok: bool
    skew_ok: bool

    @property
    def ok(self) -> bool:
        return self.audio_ok and self.mask_ok and self.skew_ok

    def to_dict(self) -> dict:
        return {"delta_audio_ms": round(self.delta_audio_ms, 3), "delta_mask_ms": round(self.delta_mask_ms, 3),
                "delta_skew_ms": round(self.delta_skew_ms, 3), "q_ms": round(self.q_ms, 3),
                "audio_ok": self.audio_ok, "mask_ok": self.mask_ok, "skew_ok": self.skew_ok, "ok": self.ok}


def check_sync(t_audio_onset_us: float, t_mask_onset_us: float, t_visual_truth_us: float, q_ms: float,
               early_ms: float = 15.0, late_ms: float = 45.0, skew_base_ms: float = 20.0) -> SyncResult:
    da = (t_audio_onset_us - t_visual_truth_us) / 1000.0
    dm = (t_mask_onset_us - t_visual_truth_us) / 1000.0
    ds = (t_mask_onset_us - t_audio_onset_us) / 1000.0
    return SyncResult(da, dm, ds, q_ms, -early_ms - q_ms <= da <= late_ms + q_ms,
                      -early_ms - q_ms <= dm <= late_ms + q_ms, abs(ds) <= skew_base_ms + q_ms)
