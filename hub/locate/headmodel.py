"""Head-shadow (spherical head) level model.

Brown & Duda (1998), "A structural model for binaural sound synthesis", IEEE
Trans. Speech Audio Process. 6(5):476-488 -- Rayleigh spherical-head
approximation as a single pole/zero:

    H(w, th) = (1 + j*alpha(th)*w/(2*w0)) / (1 + j*w/(2*w0)),   w0 = c/a
    alpha(th) = 1 + a_min/2 + (1 - a_min/2) * cos(th/th_min * 180deg)
    a_min = 0.1, th_min = 150 deg

th is the angle of incidence between the source direction and the ear/mic
axis. Constants are recalled from the paper and not re-verified against the
full text (see report).
"""
from __future__ import annotations

import numpy as np

C_SOUND = 343.0


def head_shadow_db(freqs_hz, inc_deg, radius_m=0.0875, alpha_min=0.1, theta_min=150.0):
    """Gain in dB, shape inc_deg.shape + (len(freqs),)."""
    f = np.asarray(freqs_hz, float)
    th = np.asarray(inc_deg, float)[..., None]
    w0 = C_SOUND / radius_m
    x = 2 * np.pi * f / (2 * w0)
    alpha = 1 + alpha_min / 2 + (1 - alpha_min / 2) * np.cos(np.deg2rad(th * 180.0 / theta_min))
    mag2 = (1 + (alpha * x) ** 2) / (1 + x ** 2)
    return 10 * np.log10(mag2)


class GainTable:
    """Lookup of head-shadow dB over incidence angle 0..180 deg (1 deg steps)."""

    def __init__(self, freqs_hz, **kw):
        self.angles = np.arange(0, 181, 1.0)
        self.table = head_shadow_db(freqs_hz, self.angles, **kw)  # (181, B)

    def __call__(self, inc_deg):
        inc = np.clip(np.asarray(inc_deg, float), 0, 180)
        i0 = np.minimum(inc.astype(int), 179)
        fr = (inc - i0)[..., None]
        return self.table[i0] * (1 - fr) + self.table[i0 + 1] * fr


def source_dir_head_frame(world_az_deg, yaw_deg, pitch_deg=0.0):
    """Unit vector (x fwd, y right, z up) in the head frame of a horizontal
    (elevation 0) source at world azimuth, for head yaw (right+) and pitch (up+).
    Broadcasts world_az_deg."""
    rel = np.deg2rad(np.asarray(world_az_deg, float) - yaw_deg)
    p = np.deg2rad(pitch_deg)
    a, b = np.cos(rel), np.sin(rel)
    return np.stack([a * np.cos(p), b, -a * np.sin(p)], axis=-1)


def mic_incidence_deg(u_head, mic_az_deg):
    """Angle between head-frame source direction(s) and mic axes (azimuth, horizontal).
    u_head (...,3) -> (..., M)."""
    m = np.deg2rad(np.asarray(mic_az_deg, float))
    axes = np.stack([np.cos(m), np.sin(m), np.zeros_like(m)], axis=-1)  # (M,3)
    cosang = np.clip(u_head @ axes.T, -1, 1)
    return np.rad2deg(np.arccos(cosang))
