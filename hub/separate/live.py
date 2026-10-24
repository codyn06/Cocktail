"""Live separation on sliding windows (tasks A6, C3).

Every hop (default 2 s) the separator sees the last `win` seconds (default 4 s) of multichannel
16 kHz audio and emits the finished hop for every stream. Streams are voice slots plus a residual
(music and other sounds). Per window:

  1. The voice separator splits two voices from the mix; the residual is the mix minus them.
  2. Direction gate: if the two voices arrive from the same direction (their left/right level
     difference, 1-8 kHz, differs by less than merge_ild_db) they are one source split in two
     and are merged. Needs >= 2 mics.
  3. Second pass (default "auto", needs a labeler): if the residual still sounds like speech, it
     is separated again; voices found there get their own slots (direction-gated the same way),
     anything else stays in the residual. Recovers a third and fourth talker.
  4. Speech gate: a voice slot that doesn't sound like speech moves into the residual.
  5. Order matching: streams are permuted (Hungarian assignment on overlap similarity) so each
     continues the previous window's stream. Silent or ambiguous streams keep their slot.

Longer windows separate 3+ voices much better (4 s vs 2 s, scenario suite) at the cost of delay:
a hop is final one hop after it ends, plus compute (~2 s at the default). For each stream it
also gives the stream as heard at each mic (ratio masks on every mic's spectrogram) and per-mic
band levels at 20 Hz for hub/locate (A8, PR #11).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment

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
    def __init__(self, separator, labeler=None, win_s: float = 4.0, hop_s: float | None = None,
                 label_every: int = 1, label_smooth: float = 0.6, swap_margin: float = 0.05,
                 floor_db: float = -60.0, merge_ild_db: float = 4.5, speech_min: float = 0.2,
                 second_pass: str = "auto", residual_speech_min: float = 0.3,
                 second_speech_min: float = 0.5):
        self.sep, self.labeler = separator, labeler
        self.W = int(win_s * SR)
        self.H = int(hop_s * SR) if hop_s else self.W // 2
        assert self.W == 2 * self.H, "overlap-add assumes 50 % overlap"
        self.fade = signal.windows.hann(self.W, sym=False)  # halves sum to 1 at 50 % overlap
        assert second_pass in ("auto", "always", "never")
        self.second_pass = second_pass if (labeler is not None or second_pass != "auto") else "never"
        self.n_voices = separator.n_out * (2 if self.second_pass != "never" else 1)
        self.n_streams = self.n_voices + 1  # voice slots, then the residual
        self.ids = [f"s{i + 1}" for i in range(self.n_streams)]
        self.label_every, self.label_smooth = label_every, label_smooth
        self.swap_margin, self.floor_db = swap_margin, floor_db
        self.merge_ild_db, self.speech_min = merge_ild_db, speech_min
        self.residual_speech_min, self.second_speech_min = residual_speech_min, second_speech_min
        self.last_gates: dict = {}  # what the gates did on the latest window (for status pages)
        self.reset()

    def reset(self):
        self._prev = None       # (streams, mics+1, H) faded tail of the previous window
        self._prev_tail = None  # (streams, H) unfaded mono tail, for order matching
        self._scores: list[dict[str, float]] = [{} for _ in range(self.n_streams)]
        self._steps = 0
        self.last_gates = {}

    # ------------------------------------------------------------------ one window
    def _fit(self, v: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Scale the separator's outputs to the target (ridge keeps it sane when one is absent)."""
        A = v @ v.T
        g = np.linalg.solve(A + (1e-6 * np.trace(A) / len(A) + 1e-12) * np.eye(len(A)), v @ target)
        return v * g[:, None]

    def _images(self, est: np.ndarray, x: np.ndarray):
        """Ratio masks from the mono estimates applied to every mic: (streams, mics, F, T)."""
        f, _, S = signal.stft(est, SR, nperseg=N_FFT, noverlap=N_FFT - HOP_FFT)
        P = np.abs(S) ** 2
        M = P / (P.sum(0, keepdims=True) + 1e-10)
        _, _, X = signal.stft(x.T, SR, nperseg=N_FFT, noverlap=N_FFT - HOP_FFT)
        return f, M[:, None] * X[None]

    def _same_direction(self, a: np.ndarray, b: np.ndarray, x: np.ndarray) -> tuple[bool, float | None]:
        """Do two estimates come from the same direction? Uses their per-mic images."""
        if x.shape[1] < 2 or self.merge_ild_db <= 0:
            return False, None
        rest = x.mean(axis=1) - a - b
        f, TF = self._images(np.stack([a, b, rest]), x)
        d = self._ild_diff(np.abs(TF[:2, :2]) ** 2, f)
        return d < self.merge_ild_db, round(d, 2)

    def _is_speech(self, p: np.ndarray, min_speech: float, strict: bool = False) -> bool:
        """Speech score above min_speech, or speech is the top label. strict: both must hold."""
        sc = self.labeler.coarse(p)
        hi, top = sc.get("speech", 0.0) >= min_speech, max(sc, key=sc.get) == "speech"
        return (hi and top) if strict else (hi or top)

    def _separate_window(self, x: np.ndarray):
        """x: (W, mics) -> (out (streams, 1 + mics, W), label probs or None)."""
        mono = x.mean(axis=1)
        gates = {"merged": [], "second_pass": False, "to_residual": [], "ild_diff_db": None}
        voices = np.zeros((self.n_voices, self.W))
        v = self._fit(self.sep(mono), mono)
        same, d = self._same_direction(v[0], v[1], x)
        gates["ild_diff_db"] = d
        if same:  # one source split across both outputs
            v = np.stack([v[0] + v[1], np.zeros(self.W)])
            gates["merged"].append("pass1")
        voices[:2] = v
        residual = mono - v.sum(0)

        # Label scores per voice slot, computed once and reused (AST pads every clip to ~10 s,
        # so each call is expensive). The residual is re-scored only if it changes.
        vp = [None] * self.n_voices
        rp = None
        if self.labeler is not None:
            p1 = self.labeler.probs([voices[0].astype(np.float32), voices[1].astype(np.float32),
                                     residual.astype(np.float32)])
            vp[0], vp[1], rp = p1[0], p1[1], p1[2]
        run2 = self.second_pass == "always" or (
            self.second_pass == "auto" and self._is_speech(rp, self.residual_speech_min))
        if run2:  # the residual still holds a voice: separate it again
            gates["second_pass"] = True
            w = self._fit(self.sep(residual), residual)
            same2, _ = self._same_direction(w[0], w[1], x)
            if same2:
                w = np.stack([w[0] + w[1], np.zeros(self.W)])
                gates["merged"].append("pass2")
            if self.labeler is not None:
                pw = self.labeler.probs([w[0].astype(np.float32), w[1].astype(np.float32)])
                # Stricter than pass 1: only clear voices leave the residual (birds and music
                # fragments otherwise get voice slots).
                keep = [k for k in range(2)
                        if np.any(w[k]) and self._is_speech(pw[k], self.second_speech_min, strict=True)]
            else:
                keep = [k for k in range(2) if np.any(w[k])]
            for k in keep:  # only voices leave the residual; music etc. stays in it
                voices[2 + k] = w[k]
                residual = residual - w[k]
                if self.labeler is not None:
                    vp[2 + k] = pw[k]
            if keep:
                rp = None  # residual changed

        # Speech gate: a voice slot that doesn't sound like speech belongs to the residual.
        if self.labeler is not None and self.speech_min > 0:
            for k in range(self.n_voices):
                if np.any(voices[k]) and vp[k] is not None and not self._is_speech(vp[k], self.speech_min):
                    residual = residual + voices[k]
                    voices[k] = 0.0
                    vp[k] = None
                    rp = None
                    gates["to_residual"].append(k)

        est = np.concatenate([voices, residual[None]])
        f, TF = self._images(est, x)
        _, imgs = signal.istft(TF, SR, nperseg=N_FFT, noverlap=N_FFT - HOP_FFT)
        out = np.concatenate([est[:, None], imgs[..., : self.W]], axis=1)
        self.last_gates = gates
        p = None
        if self.labeler is not None:
            if rp is None:
                rp = self.labeler.probs([residual.astype(np.float32)])[0]
            silent = np.zeros_like(p1[0])  # empty slot: no evidence for any label
            p = np.stack([vp[k] if vp[k] is not None and np.any(voices[k]) else silent
                          for k in range(self.n_voices)] + [rp])
        return out, p

    def _match_order(self, out: np.ndarray, p):
        """Permute streams so each continues the previous window's stream it overlaps most.
        Ties and silent streams keep their slot (swap_margin)."""
        if self._prev_tail is None:
            return out, p
        head = out[:, 0, : self.H]

        def cos(u, w):
            return abs(np.dot(u, w)) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9)
        # A stream more than 30 dB below the total is silent: a faint scaled copy of a voice must
        # not "match" that voice just because cosine similarity ignores level.
        def live_mask(z):
            e = np.sum(z**2, axis=1)
            return e > 1e-3 * (e.sum() + 1e-12)
        on_new, on_old = live_mask(head), live_mask(self._prev_tail)
        C = np.array([[cos(head[j], self._prev_tail[i]) if on_new[j] and on_old[i] else 0.0
                       for j in range(self.n_streams)] for i in range(self.n_streams)])
        C += self.swap_margin * np.eye(self.n_streams)  # prefer staying put
        rows, cols = linear_sum_assignment(-C)
        order = cols[np.argsort(rows)]
        return out[order], (p[order] if p is not None else None)

    @staticmethod
    def _ild_diff(E: np.ndarray, f: np.ndarray) -> float:
        """E: (2 voices, 2 mics, F, T) power. Mean |ILD(v1) - ILD(v2)| in dB over 1-8 kHz,
        weighted toward STFT frames where both voices are present."""
        band = (f >= 1000) & (f <= 8000)
        e = E[:, :, band].sum(2)                                   # (2, mics, T)
        ild = 10 * np.log10((e[:, 1] + 1e-10) / (e[:, 0] + 1e-10))
        w = np.minimum(e[0].sum(0), e[1].sum(0))
        return float(np.sum(w * np.abs(ild[0] - ild[1])) / (w.sum() + 1e-12))

    def _update_labels(self, p: np.ndarray):
        # Labels look at the whole window and are smoothed across hops, so they describe the
        # recent past of the stream rather than exactly the emitted hop.
        a = self.label_smooth
        for k in range(self.n_streams):
            new = self.labeler.coarse(p[k])
            old = self._scores[k]
            self._scores[k] = {c: a * old.get(c, s) + (1 - a) * s for c, s in new.items()}

    # ------------------------------------------------------------------ public
    def step(self, x: np.ndarray, start_ts_us: int) -> list[StreamChunk]:
        """x: (W, mics) window whose first sample is at start_ts_us; windows must advance by
        exactly one hop. Returns the finished first hop of this window for every stream
        (the very first window's first hop is emitted unblended)."""
        assert x.shape[0] == self.W
        out, p = self._separate_window(x)
        out, p = self._match_order(out, p)
        if p is not None and self._steps % self.label_every == 0:
            self._update_labels(p)
        self._steps += 1
        self._prev_tail = out[:, 0, self.H :].copy()
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
            default = "speech" if k < self.n_voices else "other"
            label, conf = max(s.items(), key=lambda kv: kv[1]) if s else (default, 0.0)
            bb = 10 * np.log10(pow_bb[:, k] + 1e-10)
            # Active = loud in absolute terms (above floor_db, power re full scale) and not buried
            # in the other streams (0 at 25 dB below the mix, 1 within 5 dB of it).
            activity = np.clip((bb - self.floor_db) / 10, 0, 1) * np.clip((bb - (mix_db - 25)) / 20, 0, 1)
            chunks.append(StreamChunk(self.ids[k], label, float(conf), start_ts_us, head[k, 0].copy(),
                                      head[k, 1:].copy(), level_ts, lv_all[:, k], activity,
                                      scores=dict(sorted(s.items(), key=lambda kv: -kv[1])[:3])))
        return chunks
