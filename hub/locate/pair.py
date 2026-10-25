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


def fit_calibration(takes, device: str = "quest3s", names=None) -> PairCalibration:
    """takes: list of (yaw_deg array, tau_s array, truth_deg) for labeled takes."""
    A, b = [], []
    for y, tau, truth in takes:
        rel = np.radians(truth - np.asarray(y))
        A.append(np.stack([np.sin(rel), np.cos(rel), np.ones_like(rel)], 1))
        b.append(np.asarray(tau))
    A, b = np.concatenate(A), np.concatenate(b)
    (u, v, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    # D sin(rel - phi) = D cos(phi) sin(rel) - D sin(phi) cos(rel)
    res = b - A @ np.array([u, v, c])
    noise = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    return PairCalibration(float(np.hypot(u, v)), float(np.degrees(np.arctan2(-v, u))), float(c),
                           max(noise, 1e-6), device, takes=list(names or []))


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

    def add_delay(self, source_id, t, tau_s, weight=1.0):
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
        sig = c.noise_s
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
