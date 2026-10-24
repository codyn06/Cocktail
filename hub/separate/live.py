"""Live separation on sliding windows (task C3).

Every hop (default 1 s) the separator sees the last `win` seconds (default 2 s) of multichannel
16 kHz audio and emits the finished hop for each stream:

  stream 0, 1: the two voices from the voice separator (order kept by matching the overlap)
  stream 2:    the residual (mix minus the gain-fitted voices): music and other sounds

The voice separator always outputs two voices. With one talker it splits that person across
both, so two gates run on every window:
  merge:  if the two voice streams arrive from the same direction (their left/right level
          difference, 1-8 kHz, differs by less than merge_ild_db) they are one source: summed
          into the louder slot, the other slot goes silent. Needs >= 2 mics.
  speech: a voice stream the labeler does not hear as speech (music only, birds) moves into
          the residual. Needs a labeler.

For each stream it also gives the stream as heard at each mic (ratio masks applied to every
mic's spectrogram) and per-mic band levels at 20 Hz for hub/locate (A8, PR #11).
Output latency: a hop is final one hop after it ends (window end), plus compute time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

SR = 16000
N_FFT, HOP_FFT = 512, 128
LEVEL_HZ = 20
BAND_CENTERS_HZ = (500.0, 1000.0, 2000.0, 4000.0, 8000.0)  # matches hub/locate LocateConfig


@dataclass
class StreamChunk:
    stream_id: str
    label: str
    label_conf: float
    start_ts_us: int                 # device time of audio[0]
    audio: np.ndarray                # (n,) mono estimate at 16 kHz
    mic_audio: np.ndarray            # (mics, n) the stream as heard at each mic
    level_ts_us: np.ndarray          # (frames,) device time of each level frame centre
    levels_db: np.ndarray            # (frames, mics, bands) per-mic band levels
    activity: np.ndarray             # (frames,) 0..1, use as the A8 weight
    scores: dict[str, float] = field(default_factory=dict)


def _band_edges():
    return [(c / np.sqrt(2), min(c * np.sqrt(2), SR / 2)) for c in BAND_CENTERS_HZ]


def band_levels(x: np.ndarray) -> np.ndarray:
    """(mics, n) -> (frames, mics, bands) dB levels in 50 ms frames."""
    hop = SR // LEVEL_HZ
    f, _, Z = signal.stft(x, SR, nperseg=2 * hop, noverlap=hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2  # (mics, freqs, frames)
    out = np.stack([P[:, (f >= lo) & (f < hi), :].mean(axis=1) for lo, hi in _band_edges()], axis=-1)
    return 10 * np.log10(out.transpose(1, 0, 2) + 1e-10)


class LiveSeparator:
    def __init__(self, separator, labeler=None, win_s: float = 2.0, hop_s: float = 1.0,
                 label_every: int = 1, label_smooth: float = 0.6, swap_margin: float = 0.1,
                 floor_db: float = -60.0, merge_ild_db: float = 4.5, speech_min: float = 0.2):
        self.sep, self.labeler = separator, labeler
        self.W, self.H = int(win_s * SR), int(hop_s * SR)
        assert self.W == 2 * self.H, "overlap-add assumes 50 % overlap"
        self.fade = signal.windows.hann(self.W, sym=False)  # halves sum to 1 at 50 % overlap
        self.n_streams = separator.n_out + 1
        self.ids = [f"s{i + 1}" for i in range(self.n_streams)]
        self.label_every, self.label_smooth = label_every, label_smooth
        self.swap_margin, self.floor_db = swap_margin, floor_db
        self.merge_ild_db, self.speech_min = merge_ild_db, speech_min
        self.last_gates: dict = {}  # what the gates did on the latest window (for status pages)
        self.reset()

    def reset(self):
        self._prev = None        # (streams, mics+1, H) faded tail of the previous window
        self._prev_mono = None   # (voices, H) unfaded tail, for order matching
        self._scores: list[dict[str, float]] = [{} for _ in range(self.n_streams)]
        self._steps = 0
        self.last_gates = {}
        self._voice_slot = None  # voice slot that last held a merged single source

    def _separate_window(self, x: np.ndarray) -> np.ndarray:
        """x: (W, mics) -> (streams, 1 + mics, W): mono estimate then per-mic images."""
        mono = x.mean(axis=1)
        v = self.sep(mono)  # (voices, W)
        if self._prev_mono is not None:  # keep voice order consistent with the previous window
            a, b = v[:, : self.H], self._prev_mono

            def cos(u, w):
                return abs(np.dot(u, w)) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9)
            same = cos(a[0], b[0]) + cos(a[1], b[1])
            swap = cos(a[0], b[1]) + cos(a[1], b[0])
            if swap > same + self.swap_margin:  # weak evidence keeps the previous order
                v = v[::-1]
        # Fit the separator's arbitrary per-output scale (ridge keeps it sane when a voice is absent).
        A = v @ v.T
        lam = 1e-6 * np.trace(A) / len(A) + 1e-12
        g = np.linalg.solve(A + lam * np.eye(len(A)), v @ mono)
        v = v * g[:, None]
        est = np.concatenate([v, (mono - v.sum(0))[None]])  # (streams, W)
        self._prev_mono = v[:, self.H :].copy()

        # Ratio masks from the mono estimates, applied to every mic.
        f, _, S = signal.stft(est, SR, nperseg=N_FFT, noverlap=N_FFT - HOP_FFT)
        P = np.abs(S) ** 2
        M = P / (P.sum(0, keepdims=True) + 1e-10)
        _, _, X = signal.stft(x.T, SR, nperseg=N_FFT, noverlap=N_FFT - HOP_FFT)  # (mics, F, T)
        TF = M[:, None] * X[None]                                                # (streams, mics, F, T)
        gates = {"ild_diff_db": None, "merged": False, "to_residual": []}
        if x.shape[1] >= 2 and self.merge_ild_db > 0:
            d = self._ild_diff(np.abs(TF[:2, :2]) ** 2, f)
            gates["ild_diff_db"] = round(d, 2)
            if d < self.merge_ild_db:  # same direction: one source split across both voice slots
                # Continuity, not identity: stay in the slot the voice was in. Whether a voice after
                # a pause is the same person is the tracker's job (D1: world bearing, voice).
                if self._voice_slot is None:
                    self._voice_slot = int(np.sum(est[1] ** 2) > np.sum(est[0] ** 2))
                keep = self._voice_slot
                self._prev_mono[keep], self._prev_mono[1 - keep] = self._prev_mono.sum(0), 0.0
                est[keep], est[1 - keep] = est[0] + est[1], 0.0
                TF[keep], TF[1 - keep] = TF[0] + TF[1], 0.0
                gates["merged"] = True
        _, imgs = signal.istft(TF, SR, nperseg=N_FFT, noverlap=N_FFT - HOP_FFT)
        out = np.concatenate([est[:, None], imgs[..., : self.W]], axis=1)
        self.last_gates = gates
        return out

    @staticmethod
    def _ild_diff(E: np.ndarray, f: np.ndarray) -> float:
        """E: (2 voices, 2 mics, F, T) power. Mean |ILD(v1) - ILD(v2)| in dB over 1-8 kHz,
        weighted toward STFT frames where both voices are present."""
        band = (f >= 1000) & (f <= 8000)
        e = E[:, :, band].sum(2)                                   # (2, mics, T)
        ild = 10 * np.log10((e[:, 1] + 1e-10) / (e[:, 0] + 1e-10))
        w = np.minimum(e[0].sum(0), e[1].sum(0))
        return float(np.sum(w * np.abs(ild[0] - ild[1])) / (w.sum() + 1e-12))

    def _speech_gate(self, out: np.ndarray, p: np.ndarray) -> None:
        """Move voice streams that don't sound like speech into the residual (in place)."""
        for k in range(self.n_streams - 1):
            sc = self.labeler.coarse(p[k])
            if np.sum(out[k, 0] ** 2) > 0 and sc.get("speech", 0.0) < self.speech_min                     and max(sc.values()) > sc.get("speech", 0.0):
                out[-1] += out[k]
                out[k] = 0.0
                p[-1] = np.maximum(p[-1], p[k])
                self._prev_mono[k] = 0.0
                self.last_gates.setdefault("to_residual", []).append(self.ids[k])

    def _update_labels(self, p: np.ndarray):
        # Labels look at the whole 2 s window and are smoothed across hops, so they describe the
        # recent past of the stream rather than exactly the emitted hop.
        a = self.label_smooth
        for k in range(self.n_streams):
            new = self.labeler.coarse(p[k])
            old = self._scores[k]
            self._scores[k] = {c: a * old.get(c, s) + (1 - a) * s for c, s in new.items()}

    def step(self, x: np.ndarray, start_ts_us: int) -> list[StreamChunk]:
        """x: (W, mics) window whose first sample is at start_ts_us; windows must advance by
        exactly one hop. Returns the finished first hop of this window for every stream
        (the very first window's first hop is emitted unblended)."""
        assert x.shape[0] == self.W
        out = self._separate_window(x)
        if self.labeler is not None:
            p = self.labeler.probs([out[k, 0].astype(np.float32) for k in range(self.n_streams)])
            if self.speech_min > 0:
                self._speech_gate(out, p)
            if self._steps % self.label_every == 0:
                self._update_labels(p)
        self._steps += 1
        faded = out * self.fade
        head = faded[..., : self.H] + self._prev if self._prev is not None else out[..., : self.H]
        self._prev = faded[..., self.H :]

        lv_all = band_levels(head[:, 1:].reshape(-1, self.H))  # (frames, streams*mics, bands)
        frames = lv_all.shape[0]
        lv_all = lv_all.reshape(frames, self.n_streams, -1, len(BAND_CENTERS_HZ))
        hop_us = 1_000_000 // LEVEL_HZ
        level_ts = start_ts_us + hop_us + np.arange(frames) * hop_us  # centres of 100 ms windows, 50 ms apart
        pow_bb = np.mean(10 ** (lv_all / 10), axis=(2, 3))  # (frames, streams) broadband power
        mix_db = 10 * np.log10(pow_bb.sum(axis=1) + 1e-10)

        chunks = []
        for k in range(self.n_streams):
            s = self._scores[k]
            label, conf = max(s.items(), key=lambda kv: kv[1]) if s else ("speech" if k < 2 else "other", 0.0)
            bb = 10 * np.log10(pow_bb[:, k] + 1e-10)
            # Active = loud in absolute terms (above floor_db, power re full scale) and not buried
            # in the other streams (0 at 25 dB below the mix, 1 within 5 dB of it).
            activity = np.clip((bb - self.floor_db) / 10, 0, 1) * np.clip((bb - (mix_db - 25)) / 20, 0, 1)
            chunks.append(StreamChunk(self.ids[k], label, float(conf), start_ts_us, head[k, 0].copy(),
                                      head[k, 1:].copy(), level_ts, lv_all[:, k], activity,
                                      scores=dict(sorted(s.items(), key=lambda kv: -kv[1])[:3])))
        return chunks
