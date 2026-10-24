"""Simulator for head-motion bearing estimation (simulated/mocked data only).

Deliberately mismatched from the estimator's model (different head radius,
shadow constants, mic angles, uncorrected mic gain offsets) to avoid an
"inverse crime".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimator import BearingEstimator, LocateConfig, wrap180
from .headmodel import C_SOUND, GainTable, mic_incidence_deg, source_dir_head_frame

FEAT_HZ = 20.0
POSE_HZ = 50.0


def head_trajectory(T, amp_deg, rng, pitch_amp_deg=4.0, center_deg=0.0):
    """Natural-ish yaw track: 3 slow sinusoids (0.07-0.35 Hz) scaled so the
    peak excursion ~ amp_deg, plus small jitter. Returns t, yaw, pitch."""
    t = np.arange(0, T, 1 / POSE_HZ)
    f = rng.uniform(0.07, 0.35, 3)
    ph = rng.uniform(0, 2 * np.pi, 3)
    w = np.array([1.0, 0.6, 0.3])
    yaw = sum(wi * np.sin(2 * np.pi * fi * t + pi) for wi, fi, pi in zip(w, f, ph))
    yaw = yaw / max(np.abs(yaw).max(), 1e-9) * amp_deg
    yaw += center_deg + np.cumsum(rng.normal(0, 0.05, len(t)))  # small drift/jitter
    fp = rng.uniform(0.1, 0.3)
    pitch = pitch_amp_deg * np.sin(2 * np.pi * fp * t + rng.uniform(0, 6.28))
    return t, yaw, pitch


@dataclass
class SimSource:
    az_deg: float
    tilt_db_per_oct: float = -3.0
    gate: bool = True   # speech-like on/off gating


class SimWorld:
    def __init__(self, sources, freqs=(500.0, 1000.0, 2000.0, 4000.0, 8000.0), seed=0,
                 mic_az=(-95.0, 95.0), radius=0.10, alpha_min=0.15, theta_min=140.0,
                 noise_db=1.5, spec_var_db=3.0, mic_offset_db=1.0):
        self.rng = np.random.default_rng(seed)
        self.sources = sources
        self.freqs = np.asarray(freqs)
        self.mic_az = np.asarray(mic_az, float)
        self.gt = GainTable(freqs, radius_m=radius, alpha_min=alpha_min, theta_min=theta_min)
        self.noise_db = noise_db
        self.spec_var_db = spec_var_db
        self.mic_off = self.rng.normal(0, mic_offset_db, len(mic_az))
        self.radius = radius
        self.mic_pos = np.array([[radius * np.cos(np.deg2rad(a)), radius * np.sin(np.deg2rad(a)), 0.0]
                                 for a in mic_az])
        n = len(sources)
        B = len(freqs)
        self._gain = np.zeros(n)                     # slow common gain (dB)
        self._spec = np.zeros((n, B))                # AR spectral fluctuation (common to mics)
        self._noise = np.zeros((n, len(mic_az), B))  # AR per-mic noise
        self._on = [True] * n

    def step_levels(self, yaw, pitch):
        """One 20 Hz frame -> list of (levels (M,B), activity)."""
        rng = self.rng
        out = []
        for i, s in enumerate(self.sources):
            self._gain[i] = 0.995 * self._gain[i] + rng.normal(0, 0.6)
            self._spec[i] = 0.95 * self._spec[i] + rng.normal(0, self.spec_var_db * np.sqrt(1 - 0.95 ** 2), self._spec[i].shape)
            self._noise[i] = 0.9 * self._noise[i] + rng.normal(0, self.noise_db * np.sqrt(1 - 0.81), self._noise[i].shape)
            if s.gate:
                if rng.random() < (0.02 if self._on[i] else 0.04):
                    self._on[i] = not self._on[i]
            else:
                self._on[i] = True
            u = source_dir_head_frame(s.az_deg, yaw, pitch)
            inc = mic_incidence_deg(u, self.mic_az)               # (M,)
            g = self.gt(inc)                                       # (M,B)
            spec = s.tilt_db_per_oct * np.log2(self.freqs / 1000.0)
            L = g + spec + self._gain[i] + self._spec[i] + self._noise[i] + self.mic_off[:, None]
            out.append((L, 1.0 if self._on[i] else 0.0))
        return out

    def true_tau(self, i, mic_i, mic_j, yaw, pitch):
        u = source_dir_head_frame(self.sources[i].az_deg, yaw, pitch)
        return float((self.mic_pos[mic_j] - self.mic_pos[mic_i]) @ u / C_SOUND)


def run_scenario(az_list, amp_deg, T=30.0, seed=0, camera=None, tdoa_sigma_s=None,
                 cfg=None, eval_every=0.5):
    """Simulate and run the estimator. camera: None or dict(fov_deg=100, conf=0.8,
    sigma_deg=4, rate_hz=4) -> adds camera bearings when the source is in view.
    tdoa_sigma_s: if set, adds noisy TDOA for the mic pair at 5 Hz.
    Returns dict(t, err[n_src,len(t)], state, conf, est, truth_rel)."""
    rng = np.random.default_rng(seed + 12345)
    world = SimWorld([SimSource(a, tilt_db_per_oct=rng.uniform(-6, 0)) for a in az_list], seed=seed)
    est = BearingEstimator(cfg or LocateConfig())
    tp, yaw, pitch = head_trajectory(T, amp_deg, rng)
    n = len(az_list)
    times, errs, states, confs, ests = [], [], [], [], []
    next_feat = 0.0
    next_cam = 0.0
    next_tdoa = 0.0
    next_eval = 0.0
    for k, t in enumerate(tp):
        est.add_head_pose(t, yaw[k], pitch[k])
        if t >= next_feat:
            next_feat += 1 / FEAT_HZ
            for i, (L, act) in enumerate(world.step_levels(yaw[k], pitch[k])):
                est.add_levels(f"s{i}", t, L, weight=act)
        if camera and t >= next_cam:
            next_cam += 1 / camera.get("rate_hz", 4)
            for i, a in enumerate(az_list):
                rel = float(wrap180(a - yaw[k]))
                if abs(rel) <= camera.get("fov_deg", 100) / 2:
                    if rng.random() < camera.get("conf", 0.8):
                        est.add_camera_bearing(f"s{i}", t, rel + rng.normal(0, camera.get("sigma_deg", 4)),
                                               camera.get("conf", 0.8))
        if tdoa_sigma_s is not None and t >= next_tdoa:
            next_tdoa += 0.2
            for i in range(n):
                tau = world.true_tau(i, 0, 1, yaw[k], pitch[k]) + rng.normal(0, tdoa_sigma_s)
                est.add_tdoa(f"s{i}", t, 0, 1, tau, sigma_s=max(tdoa_sigma_s, 20e-6))
        if t >= next_eval:
            next_eval += eval_every
            row_e, row_s, row_c, row_b = [], [], [], []
            for i, a in enumerate(az_list):
                e = est.estimate(f"s{i}", t)
                truth = float(wrap180(a - yaw[k]))
                row_e.append(abs(float(wrap180(e.bearing_deg - truth))))
                row_s.append(e.state); row_c.append(e.confidence); row_b.append(e.bearing_deg)
            times.append(t); errs.append(row_e); states.append(row_s); confs.append(row_c); ests.append(row_b)
    return dict(t=np.array(times), err=np.array(errs).T, state=np.array(states).T,
                conf=np.array(confs).T, est=np.array(ests).T, est_obj=est, yaw=yaw)
