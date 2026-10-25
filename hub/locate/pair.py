"""Bearing from the inter-mic delay of a closely spaced mic pair plus head movement (Quest 3S).

The 3S has two raw mics about 0.5-1 cm apart, so level differences between them carry almost
no direction information and the level/head-shadow estimator (estimator.py) failed on real
takes. The arrival-time difference still does: for a far-field source at world bearing theta,

    tau(t) = D * sin(theta - phi - yaw(t)) + c

where D is the largest possible delay (spacing / speed of sound), phi the direction of the mic
pair's axis in the head frame, and c a fixed channel offset of the device. As the head turns,
tau traces a sinusoid whose phase gives theta; head motion also resolves the front/back mirror
of a single pair. D, phi and c are device constants fitted from labeled takes
(`python -m locate.real_eval calibrate ...`, saved in locate/calib/<device>.json).

Measured on the 3S (2026-10-24, 6 takes, leave-one-out): 14-23 deg on clean front/side takes
with strong head motion, 57-76 deg on behind/weaker takes. Caveat: the headset may still apply
some processing in hardware even on the UNPROCESSED source, which would blur the delays.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .estimator import BearingEstimate, wrap180
from .tdoa import gcc_phat

CALIB_DIR = Path(__file__).parent / "calib"


@dataclass
class PairCalibration:
    delay_amp_s: float      # D: largest delay (source along the pair axis)
    axis_deg: float         # phi: pair axis direction in the head frame (+ = right of forward)
    offset_s: float         # c: fixed channel offset of the device
    noise_s: float          # robust spread of delay residuals on the calibration takes
    device: str = "quest3s"
    max_tau_s: float = 0.3e-3
    takes: list = field(default_factory=list)
    delay_method: str = "gcc"   # how frame delays are measured: "gcc" (frame_delays) or "phase"
    model_s: float | None = None  # phase only: model error added to each frame's own sigma
    # Sample rate and frame length the delays were measured at. They are part of the calibration:
    # mics whose responses differ a little show a different channel offset in a narrower band
    # (pairsim: ~5 us more at 16 kHz than at 48 kHz), so fit at the rate the locator runs at
    # (live hub: 16 kHz, 1024-sample frames; `real_eval calibrate --rate 16000`).
    rate_hz: int = 48000
    win: int = 4096

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> PairCalibration:
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def default(cls, device: str = "quest3s") -> PairCalibration:
        return cls.load(CALIB_DIR / f"{device}.json")

    def predict(self, theta_deg, yaw_deg):
        return self.delay_amp_s * np.sin(np.radians(theta_deg - self.axis_deg - yaw_deg)) + self.offset_s

    def frame_sigma(self, sigma_s):
        """Delay uncertainty of one frame for the likelihood: the fixed residual spread for "gcc",
        the frame's own sigma plus the fitted model error for "phase"."""
        if sigma_s is None or self.model_s is None:
            return self.noise_s
        return float(np.hypot(sigma_s, self.model_s))


def frame_delays(x: np.ndarray, rate: int, win: int, hop: int, max_tau_s: float = 0.3e-3, interp: int = 32):
    """x: (n, 2). Returns frame-centre times (s), delays tau (s, + = channel 0 later), and peak
    sharpness (GCC peak over its median magnitude; higher = cleaner)."""
    t, tau, sharp = [], [], []
    for s in range(0, len(x) - win + 1, hop):
        lags, cc = gcc_phat(x[s:s + win, 0], x[s:s + win, 1], rate, max_tau=max_tau_s, interp=interp)
        k = int(np.argmax(cc))
        t.append((s + win / 2) / rate)
        tau.append(lags[k])
        sharp.append(cc[k] / (np.median(np.abs(cc)) + 1e-12))
    return np.array(t), np.array(tau), np.array(sharp)


def frame_delay_phase(x2: np.ndarray, rate: int, sub: int = 512, fmin: float = 200.0,
                      fmax: float = 8000.0) -> tuple[float, float]:
    """Inter-mic delay of one frame (s, + = channel 0 later) and its standard error, from the
    slope of the cross-spectrum phase: phase(f) = -2 pi f tau.

    The phase must not wrap inside the band: |tau| < 1 / (2 fmax), 62 us for the default 8 kHz,
    which also covers the ~2-sample (42 us) lag once seen between the 3S mics; the line is then
    fitted directly. The cross-spectrum and coherence come from Welch averaging
    over sub-blocks of the frame; each frequency is weighted by its phase precision
    2K g2 / (1 - g2) (K blocks, g2 coherence), which is the maximum-likelihood delay for a
    single source in uncorrelated noise (Hannan & Thomson 1973; Carter 1987). Coherent
    zero-delay sound (diffuse reverb and room noise at closely spaced mics) still biases it
    toward 0; the calibration's model error covers that. Sample coherence is biased up by
    ~1/K where there is no signal, so it is debiased before weighting."""
    hop = sub // 2
    n = (len(x2) - sub) // hop + 1
    if n < 2:
        return 0.0, float("inf")
    w = np.hanning(sub)
    idx = np.arange(sub)[None, :] + hop * np.arange(n)[:, None]
    X0 = np.fft.rfft(x2[idx, 0] * w, axis=1)
    X1 = np.fft.rfft(x2[idx, 1] * w, axis=1)
    sxy = (X0 * np.conj(X1)).mean(0)
    sxx, syy = (np.abs(X0) ** 2).mean(0), (np.abs(X1) ** 2).mean(0)
    f = np.fft.rfftfreq(sub, 1 / rate)
    m = (f >= fmin) & (f <= min(fmax, 0.45 * rate))
    k_eff = n * 0.5  # 50%-overlapped Hann blocks are not independent
    g2 = np.abs(sxy[m]) ** 2 / (sxx[m] * syy[m] + 1e-30)
    g2 = np.clip((g2 - 1 / k_eff) / (1 - 1 / k_eff), 0.0, 0.99)
    wt = 2 * k_eff * g2 / (1 - g2)
    om = 2 * np.pi * f[m]
    den = float((wt * om * om).sum())
    if den <= 0:
        return 0.0, float("inf")
    return float(-(wt * om * np.angle(sxy[m])).sum() / den), float(1 / np.sqrt(den))


def fit_calibration(takes, device: str = "quest3s", names=None) -> PairCalibration:
    """takes: list of (yaw_deg array, tau_s array, truth_deg) for labeled takes, or
    (yaw, tau, truth, sigma_s array) for phase delays: then the fit is weighted by each frame's
    sigma plus a model error model_s, chosen so normalised residuals have unit robust spread."""
    A, b, sig = [], [], []
    for y, tau, truth, *rest in takes:
        rel = np.radians(truth - np.asarray(y))
        A.append(np.stack([np.sin(rel), np.cos(rel), np.ones_like(rel)], 1))
        b.append(np.asarray(tau))
        sig.append(np.asarray(rest[0]) if rest else np.full(len(tau), np.nan))
    A, b, sig = np.concatenate(A), np.concatenate(b), np.concatenate(sig)
    phase = not np.isnan(sig).any()
    wt = np.ones_like(b)
    model = None
    for _ in range(4 if phase else 1):
        sw = np.sqrt(wt)
        (u, v, c), *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        # D sin(rel - phi) = D cos(phi) sin(rel) - D sin(phi) cos(rel)
        res = b - A @ np.array([u, v, c])
        if phase:
            model = _model_error(res, sig)
            wt = 1 / (sig ** 2 + model ** 2)
    noise = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    return PairCalibration(float(np.hypot(u, v)), float(np.degrees(np.arctan2(-v, u))), float(c),
                           max(noise, 1e-6), device, takes=list(names or []),
                           delay_method="phase" if phase else "gcc", model_s=model)


def _model_error(res, sig) -> float:
    """Smallest m >= 0 with median(|res| / sqrt(sig^2 + m^2)) <= 0.6745 (unit normal spread)."""
    z = lambda m: np.median(np.abs(res) / np.sqrt(sig ** 2 + m * m))
    lo, hi = 0.0, 1e-3
    if z(lo) <= 0.6745:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if z(mid) > 0.6745 else (lo, mid)
    return float(hi)


@dataclass
class PairConfig:
    n_az: int = 360
    tau_s: float = 20.0           # forgetting time constant of the evidence
    p_inlier: float = 0.7         # robust likelihood: share of frames that follow the model
    neff_ratio: float = 0.1       # consecutive frames are correlated: temper the evidence
    min_excite_deg: float = 8.0   # yaw std below this = "insufficient_motion"
    lock_conf: float = 0.7
    lock_sigma_deg: float = 15.0
    ambiguous_ratio: float = 0.35


class _Src:
    def __init__(self, n):
        self.loglik = np.zeros(n)
        self.last_t = None
        self.n = 0
        self.yw = self.ym = self.ymm = 0.0


class PairLocator:
    """Online grid posterior over world bearing from (yaw, delay) observations, per source.
    Same estimate type as BearingEstimator, so callers can switch between the two."""

    def __init__(self, calib: PairCalibration, cfg: PairConfig | None = None):
        self.calib, self.cfg = calib, cfg or PairConfig()
        self.az = np.arange(self.cfg.n_az) * 360.0 / self.cfg.n_az
        self._poses: deque = deque(maxlen=4000)  # (t, unwrapped yaw)
        self._src: dict[str, _Src] = {}

    def add_head_pose(self, t, yaw_deg, pitch_deg=0.0):
        if self._poses:
            yaw_deg = self._poses[-1][1] + float(wrap180(yaw_deg - self._poses[-1][1]))
        self._poses.append((float(t), float(yaw_deg)))

    def _yaw_at(self, t):
        ts = np.fromiter((p[0] for p in self._poses), float, len(self._poses))
        ys = np.fromiter((p[1] for p in self._poses), float, len(self._poses))
        return float(np.interp(t, ts, ys))

    def add_delay(self, source_id, t, tau_s, weight=1.0, sigma_s=None):
        """sigma_s: this frame's own delay uncertainty (phase delays); None = calibration noise."""
        if not self._poses or weight <= 0:
            return
        s = self._src.setdefault(source_id, _Src(self.cfg.n_az))
        if s.last_t is not None:
            lam = float(np.exp(-max(t - s.last_t, 0.0) / self.cfg.tau_s))
            s.loglik *= lam
            s.yw *= lam; s.ym *= lam; s.ymm *= lam
        s.last_t = t
        yaw = self._yaw_at(t)
        c = self.calib
        r = tau_s - c.predict(self.az, yaw)
        sig = c.frame_sigma(sigma_s)
        inl = self.cfg.p_inlier * np.exp(-0.5 * (r / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        out = (1 - self.cfg.p_inlier) / (2 * c.max_tau_s)
        s.loglik += weight * self.cfg.neff_ratio * np.log(inl + out)
        s.yw += weight; s.ym += weight * yaw; s.ymm += weight * yaw * yaw
        s.n += 1

    def estimate(self, source_id, t=None) -> BearingEstimate:
        s = self._src.get(source_id)
        if s is None or s.n == 0 or not self._poses:
            return BearingEstimate(source_id, 0.0, 0.0, 0.0, 180.0, "unobserved", None, 0.0, 0)
        yaw = self._yaw_at(t if t is not None else self._poses[-1][0])
        lp = s.loglik - s.loglik.max()
        p = np.exp(lp)
        p /= p.sum()
        N, step = self.cfg.n_az, 360.0 / self.cfg.n_az
        k = int(np.argmax(p))
        w20 = int(round(20 / step))
        idx = (np.arange(-w20, w20 + 1) + k) % N
        conf = float(p[idx].sum())
        dd = np.radians(wrap180(self.az[idx] - self.az[k]))
        wl = p[idx] / p[idx].sum()
        sig = float(np.degrees(np.sqrt((wl * dd ** 2).sum())))
        mask = np.ones(N, bool)
        mask[(np.arange(-int(40 / step), int(40 / step) + 1) + k) % N] = False
        k2 = int(np.argmax(np.where(mask, p, -1)))
        mass2 = p[(np.arange(-w20, w20 + 1) + k2) % N].sum()
        alt = float(wrap180(self.az[k2] - yaw)) if mass2 >= self.cfg.ambiguous_ratio * conf and mass2 > 0.05 else None
        mu = s.ym / s.yw
        exc = float(np.sqrt(max(s.ymm / s.yw - mu * mu, 0.0)))
        if alt is not None:
            state = "ambiguous"
        elif exc < self.cfg.min_excite_deg:
            state = "insufficient_motion"
        elif conf >= self.cfg.lock_conf and sig <= self.cfg.lock_sigma_deg:
            state = "locked"
        else:
            state = "tracking"
        world = float(wrap180(self.az[k]))
        return BearingEstimate(source_id, float(wrap180(self.az[k] - yaw)), world, conf, sig, state, alt, exc, s.n)
