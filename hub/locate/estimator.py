"""Online per-source bearing estimator from natural head motion.

Formulation (Bayesian grid filter on the circle, per source):

* State: static world-frame azimuth theta in a 360-bin grid (exact, handles
  multimodality -- needed for front/back ambiguity; a unimodal Kalman/von Mises
  filter cannot represent two mirror hypotheses). Slow exponential forgetting
  (tau_s) lets the posterior follow slowly moving sources.
* Level measurement (Brown & Duda 1998 spherical-head shadow): per-mic band
  level L[m,b](t) = a_b + g(t) + G_b(inc_m(theta, yaw(t), pitch(t))) + noise,
  where a_b is the (unknown) source spectrum and g(t) an unknown common gain.
  Nuisances are removed by channel transforms (inter-mic difference removes a_b
  and g; spectral tilt removes g) and a profiled per-band offset (least-squares
  elimination of a_b from exponentially weighted sufficient statistics).
  Under Gaussian noise the log-likelihood of theta is the weighted residual SSE.
* Front/back (Wallach 1940): no special rule. Mirror hypotheses predict
  different level traces as yaw changes, so the accumulated likelihood
  separates them when the head moves.
* External evidence (camera bearing as a von Mises + uniform mixture; TDOA as a
  Gaussian+uniform mixture over the far-field delay; GCC-PHAT curves SRP-style)
  adds log-likelihood to the same grid, so fusion is exact Bayes product.

Conventions: bearing deg, 0 = ahead, + = right (clockwise from above), range
(-180, 180]. Yaw increases when the head turns right. Elevation of sources is
assumed 0 (pitch is used only to correct the geometry).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .headmodel import C_SOUND, GainTable, mic_incidence_deg, source_dir_head_frame


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


@dataclass
class LocateConfig:
    n_az: int = 360
    band_freqs_hz: tuple = (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    mic_az_deg: tuple = (-90.0, 90.0)     # head-frame mic axis azimuths (ear-like default)
    head_radius_m: float = 0.0875
    tau_s: float = 8.0                    # forgetting time constant of level statistics
    tau_ext_s: float = 20.0               # forgetting of camera/TDOA evidence
    sigma_diff_db: float = 2.0            # noise std, inter-mic difference channel
    sigma_tilt_db: float = 2.5            # noise std, spectral tilt channel
    sigma_bb_db: float = 6.0              # noise std, broadband level channel
    neff_ratio: float = 0.25              # <1: features are temporally correlated
    profile_diff: bool = False            # True: also profile out static inter-mic offset
    min_excite_deg: float = 5.0           # yaw std below this = "insufficient_motion"
    lock_conf: float = 0.7
    lock_sigma_deg: float = 15.0
    ambiguous_ratio: float = 0.35


@dataclass
class BearingEstimate:
    source_id: str
    bearing_deg: float            # relative to head yaw at t (0 ahead, + right)
    world_deg: float              # in the yaw=0 world frame
    confidence: float             # posterior mass within +-20 deg of the peak, 0..1
    sigma_deg: float              # local circular std around the peak
    state: str                    # unobserved | insufficient_motion | ambiguous | tracking | locked
    alt_bearing_deg: float | None  # second hypothesis (head-relative) if any
    excitation_deg: float         # weighted yaw std in the window
    n_updates: int


class _Src:
    def __init__(self, n_az, C):
        self.W = 0.0
        self.Sy = np.zeros(C)
        self.Syy = np.zeros(C)
        self.SG = np.zeros((n_az, C))
        self.SGG = np.zeros((n_az, C))
        self.SyG = np.zeros((n_az, C))
        self.ext = np.zeros(n_az)
        self.last_t = None
        self.last_ext_t = None
        self.n = 0
        self.ym = 0.0   # EW yaw mean / second moment for excitation
        self.ymm = 0.0
        self.yw = 0.0


class BearingEstimator:
    def __init__(self, cfg: LocateConfig | None = None):
        self.cfg = cfg or LocateConfig()
        c = self.cfg
        self.az = np.arange(c.n_az) * 360.0 / c.n_az
        self.M = len(c.mic_az_deg)
        self.B = len(c.band_freqs_hz)
        self.gain = GainTable(c.band_freqs_hz, radius_m=c.head_radius_m)
        # channel layout: diff (B per M-1 rows), tilt (B), broadband (1)
        nd = (self.M - 1) * self.B if self.M >= 2 else 0
        self.C = nd + self.B + 1
        sig = np.concatenate([np.full(nd, c.sigma_diff_db), np.full(self.B, c.sigma_tilt_db), [c.sigma_bb_db]])
        self.inv_var = 1.0 / sig ** 2
        prof = np.ones(self.C, bool)
        prof[:nd] = c.profile_diff
        self.profiled = prof
        r = c.head_radius_m
        m = np.deg2rad(c.mic_az_deg)
        self.mic_pos = np.stack([r * np.cos(m), r * np.sin(m), np.zeros_like(m)], axis=1)
        self._poses = deque(maxlen=2000)   # (t, unwrapped yaw, pitch)
        self._src: dict[str, _Src] = {}
        self._gcache = (None, None)

    # ----------------------------------------------------------------- pose
    def add_head_pose(self, t, yaw_deg, pitch_deg=0.0):
        """Head pose sample (timestamps seconds, monotonic, hub clock)."""
        if self._poses:
            _, y0, _ = self._poses[-1]
            yaw_deg = y0 + float(wrap180(yaw_deg - y0))   # unwrap
        self._poses.append((float(t), float(yaw_deg), float(pitch_deg)))

    def _pose_at(self, t):
        if not self._poses:
            raise RuntimeError("no head pose yet")
        ts = np.fromiter((p[0] for p in self._poses), float, len(self._poses))
        i = np.searchsorted(ts, t)
        if i == 0:
            return self._poses[0][1], self._poses[0][2]
        if i >= len(ts):
            return self._poses[-1][1], self._poses[-1][2]
        (t0, y0, p0), (t1, y1, p1) = self._poses[i - 1], self._poses[i]
        f = (t - t0) / max(t1 - t0, 1e-9)
        return y0 + f * (y1 - y0), p0 + f * (p1 - p0)

    # ------------------------------------------------------------ channels
    def _channels(self, x):
        """(M,B) levels or gains -> (C,) channel vector; leading dims allowed."""
        x = np.asarray(x, float)
        common = x.mean(axis=-2)                       # (...,B)
        tilt = common - common.mean(axis=-1, keepdims=True)
        bb = common.mean(axis=-1, keepdims=True)
        parts = []
        if self.M >= 2:
            d = x[..., :-1, :] - common[..., None, :]
            parts.append(d.reshape(*d.shape[:-2], -1))
        parts += [tilt, bb]
        return np.concatenate(parts, axis=-1)

    def _pred(self, yaw, pitch):
        key = (round(yaw, 4), round(pitch, 4))
        if self._gcache[0] == key:
            return self._gcache[1]
        u = source_dir_head_frame(self.az, yaw, pitch)               # (N,3)
        inc = mic_incidence_deg(u, self.cfg.mic_az_deg)              # (N,M)
        g = self.gain(inc)                                           # (N,M,B)
        out = self._channels(np.moveaxis(g, 1, 1))                   # (N,C)
        self._gcache = (key, out)
        return out

    def _get(self, sid):
        s = self._src.get(sid)
        if s is None:
            s = self._src[sid] = _Src(self.cfg.n_az, self.C)
        return s

    # -------------------------------------------------------------- levels
    def add_levels(self, source_id, t, levels_db, weight=1.0):
        """Per-source band levels (dB) at time t. levels_db: (n_bands,) for one
        mic, or (n_mics, n_bands). weight in [0,1]: source activity (0 = silent;
        use e.g. a VAD or normalized level) so silence does not corrupt stats."""
        L = np.asarray(levels_db, float)
        if L.ndim == 1:
            L = L[None, :]
        if L.shape != (self.M, self.B):
            raise ValueError(f"levels shape {L.shape} != ({self.M},{self.B})")
        s = self._get(source_id)
        yaw, pitch = self._pose_at(t)
        if s.last_t is not None:
            lam = float(np.exp(-max(t - s.last_t, 0.0) / self.cfg.tau_s))
            s.W *= lam; s.Sy *= lam; s.Syy *= lam
            s.SG *= lam; s.SGG *= lam; s.SyG *= lam
            s.yw *= lam; s.ym *= lam; s.ymm *= lam
        s.last_t = t
        if weight <= 0:
            return
        y = self._channels(L)
        G = self._pred(yaw, pitch)
        w = float(weight)
        s.W += w
        s.Sy += w * y
        s.Syy += w * y * y
        s.SG += w * G
        s.SGG += w * G * G
        s.SyG += w * G * y
        s.yw += w; s.ym += w * yaw; s.ymm += w * yaw * yaw
        s.n += 1

    def _level_loglik(self, s):
        if s.W < 1e-6:
            return np.zeros(self.cfg.n_az)
        W = s.W
        Syy_c = np.where(self.profiled, s.Syy - s.Sy ** 2 / W, s.Syy)            # (C,)
        SyG_c = np.where(self.profiled, s.SyG - s.Sy * s.SG / W, s.SyG)          # (N,C)
        SGG_c = np.where(self.profiled, s.SGG - s.SG ** 2 / W, s.SGG)
        sse = Syy_c - 2 * SyG_c + SGG_c
        return -0.5 * self.cfg.neff_ratio * (sse * self.inv_var).sum(axis=1)

    # ------------------------------------------------------ external evidence
    def _add_ext(self, source_id, t, loglik):
        s = self._get(source_id)
        if s.last_ext_t is not None:
            s.ext *= np.exp(-max(t - s.last_ext_t, 0.0) / self.cfg.tau_ext_s)
        s.last_ext_t = t
        s.ext += loglik
        s.ext -= s.ext.max()
        np.maximum(s.ext, -60.0, out=s.ext)

    def add_camera_bearing(self, source_id, t, bearing_deg, conf, sigma_deg=5.0):
        """Camera bearing (head-relative, deg) measured at time t, with
        confidence conf in [0,1] = probability the detection is the right
        source. Likelihood: conf*VonMises(kappa=1/sigma^2) + (1-conf)*Uniform.
        Suggested rate <= ~5 Hz (successive frames are not independent)."""
        yaw, _ = self._pose_at(t)
        world = bearing_deg + yaw
        kappa = 1.0 / np.deg2rad(sigma_deg) ** 2
        d = np.deg2rad(self.az - world)
        # von Mises density exp(k cos d)/(2 pi I0(k)); for large k ~ N(0, 1/k)
        vm = np.exp(kappa * (np.cos(d) - 1)) / np.sqrt(2 * np.pi / kappa)
        conf = float(np.clip(conf, 0.0, 1.0))
        p = conf * vm + (1 - conf) / (2 * np.pi)
        self._add_ext(source_id, t, np.log(p))

    def _tau_pred(self, yaw, pitch, i, j):
        u = source_dir_head_frame(self.az, yaw, pitch)               # (N,3)
        return (self.mic_pos[j] - self.mic_pos[i]) @ u.T / C_SOUND   # tau = t_i - t_j

    def add_tdoa(self, source_id, t, mic_i, mic_j, tau_s, sigma_s=30e-6, p_inlier=0.8):
        """TDOA tau = t_i - t_j (s) for mic pair at time t (far-field model,
        gaussian+uniform mixture). Resolves azimuth up to the pair's mirror
        symmetry; head yaw then breaks the symmetry in the accumulated grid."""
        yaw, pitch = self._pose_at(t)
        tp = self._tau_pred(yaw, pitch, mic_i, mic_j)
        tmax = np.linalg.norm(self.mic_pos[mic_i] - self.mic_pos[mic_j]) / C_SOUND
        g = np.exp(-0.5 * ((tau_s - tp) / sigma_s) ** 2) / (sigma_s * np.sqrt(2 * np.pi))
        p = p_inlier * g + (1 - p_inlier) / (2 * tmax)
        self._add_ext(source_id, t, np.log(p))

    def add_gcc(self, source_id, t, mic_i, mic_j, lags_s, cc, kappa=20.0):
        """SRP-PHAT-style evidence from a full GCC-PHAT curve (see tdoa.gcc_phat):
        loglik(theta) = kappa * cc_normalised(tau_ij(theta)). kappa is a heuristic
        sharpness, not a calibrated likelihood."""
        yaw, pitch = self._pose_at(t)
        tp = self._tau_pred(yaw, pitch, mic_i, mic_j)
        cc = np.asarray(cc, float)
        cc = (cc - np.median(cc)) / max(np.ptp(cc), 1e-9)
        self._add_ext(source_id, t, kappa * np.interp(tp, lags_s, cc))

    # ------------------------------------------------------------ estimates
    def posterior(self, source_id):
        s = self._src[source_id]
        lp = self._level_loglik(s) + s.ext
        lp -= lp.max()
        p = np.exp(lp)
        return p / p.sum()

    def estimate(self, source_id, t=None) -> BearingEstimate:
        s = self._get(source_id)
        if t is None:
            t = self._poses[-1][0]
        yaw, _ = self._pose_at(t)
        N = self.cfg.n_az
        step = 360.0 / N
        have_ext = s.ext.any()
        if s.n == 0 and not have_ext:
            return BearingEstimate(source_id, 0.0, 0.0, 0.0, 180.0, "unobserved", None, 0.0, 0)
        p = self.posterior(source_id)
        k = int(np.argmax(p))
        w20 = int(round(20.0 / step))
        idx = (np.arange(-w20, w20 + 1) + k) % N
        conf = float(p[idx].sum())
        dd = np.deg2rad(wrap180(self.az[idx] - self.az[k]))
        wl = p[idx] / p[idx].sum()
        sig = float(np.rad2deg(np.sqrt((wl * dd ** 2).sum())))
        # second hypothesis: highest peak outside +-40 deg
        mask = np.ones(N, bool)
        mask[(np.arange(-int(40 / step), int(40 / step) + 1) + k) % N] = False
        alt = None
        if mask.any():
            k2 = int(np.argmax(np.where(mask, p, -1)))
            mass2 = p[(np.arange(-w20, w20 + 1) + k2) % N].sum()
            if mass2 >= self.cfg.ambiguous_ratio * conf and mass2 > 0.05:
                alt = float(wrap180(self.az[k2] - yaw))
        if s.yw > 1e-6:
            mu = s.ym / s.yw
            exc = float(np.sqrt(max(s.ymm / s.yw - mu * mu, 0.0)))
        else:
            exc = 0.0
        if alt is not None:
            state = "ambiguous"
        elif not have_ext and exc < self.cfg.min_excite_deg:
            state = "insufficient_motion"
        elif conf >= self.cfg.lock_conf and sig <= self.cfg.lock_sigma_deg:
            state = "locked"
        else:
            state = "tracking"
        world = float(wrap180(self.az[k]))
        return BearingEstimate(source_id, float(wrap180(self.az[k] - yaw)), world, conf, sig,
                               state, alt, exc, s.n)

    def estimates(self, t=None):
        return {sid: self.estimate(sid, t) for sid in self._src}
