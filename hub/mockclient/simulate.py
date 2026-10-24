"""Render a simulated session: mono stems at world bearings, heard by two ear-level mics on a
turning head (task B4). Evidence label for anything tested this way: simulated.

Acoustics: spherical head, no room reverb. Woodworth ITD; head shadow per frequency band from
the Brown & Duda (1998) one-pole form, with constants (alpha_min 0.2, theta_min 140 deg, radius
9 cm) that deliberately differ from hub/locate's estimator so tests don't feed it its own model.

  uv run python -m mockclient.simulate --out sessions/synth --stem v1.wav:-50 --stem v2.wav:40 \
      --stem music.wav:150
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .session import Session

C_SOUND = 343.0
HEAD_R = 0.0875
SHADOW_R, ALPHA_MIN, THETA_MIN = 0.09, 0.2, 140.0
BAND_EDGES = (700.0, 1400.0, 2800.0, 5600.0)  # crossovers between the 5 shadow bands


def shadow_gain(f_hz: float, inc_rad: np.ndarray) -> np.ndarray:
    """Linear magnitude of the spherical-head shadow at frequency f for incidence inc (0 = facing)."""
    th = np.degrees(inc_rad)
    alpha = 1 + ALPHA_MIN / 2 + (1 - ALPHA_MIN / 2) * np.cos(np.radians(np.minimum(th, THETA_MIN) * 180 / THETA_MIN))
    x = 2 * np.pi * f_hz * SHADOW_R / C_SOUND / 2  # w / (2 w0)
    return np.sqrt((1 + (alpha * x) ** 2) / (1 + x ** 2))
POSE_HZ = 50


def head_yaw(t: np.ndarray, amp_deg: float) -> np.ndarray:
    """Slow natural-looking head turns: two incommensurate sinusoids, peak ~amp_deg."""
    y = np.sin(2 * np.pi * 0.13 * t) + 0.45 * np.sin(2 * np.pi * 0.31 * t + 1.1)
    return y / 1.45 * amp_deg


def quat_yaw_pitch(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float, float]:
    """Unity rotation (x, y, z, w): yaw about +y (+ = right), then pitch (+ = look up)."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(-pitch_deg) / 2), math.sin(math.radians(-pitch_deg) / 2)
    # q_yaw (0, sy, 0, cy) * q_pitch (sp, 0, 0, cp)
    return (cy * sp, sy * cp, -sy * sp, cy * cp)


def load_mono(path: str, sr: int) -> np.ndarray:
    s, x = wavfile.read(path)
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    x = x.mean(axis=1) if x.ndim > 1 else x
    g = math.gcd(sr, s)
    return signal.resample_poly(x, sr // g, s // g).astype(np.float32) if s != sr else x


def render(stems, bearings_deg, sr: int, yaw_deg: np.ndarray, mic_az_deg=(-90.0, 90.0)) -> np.ndarray:
    """stems: list of mono arrays at sr; yaw_deg: per-sample head yaw. Returns (n, mics)."""
    n = len(yaw_deg)
    out = np.zeros((n, len(mic_az_deg)), np.float32)
    edges = (0.0, *BAND_EDGES, sr / 2)
    centers = [np.sqrt(max(lo, 250.0) * hi) for lo, hi in zip(edges, edges[1:])]
    idx = np.arange(n, dtype=np.float64)
    for x, b in zip(stems, bearings_deg):
        x = np.pad(x[:n], (0, max(0, n - len(x))))
        # Split into bands (zero-phase so the bands sum back to x), shadow each band separately.
        bands, rest = [], x.astype(np.float64)
        for e in BAND_EDGES:
            fb, fa = signal.butter(4, e, fs=sr)
            low = signal.filtfilt(fb, fa, rest)
            bands.append(low)
            rest = rest - low
        bands.append(rest)
        rel = np.radians(b - yaw_deg)  # source azimuth in the head frame, + = right
        for m, maz in enumerate(np.radians(mic_az_deg)):
            inc = np.arccos(np.clip(np.cos(rel - maz), -1, 1))  # angle between source and mic axis
            # Woodworth path length to a mic on a sphere: direct if facing, wraps around if shadowed.
            path = np.where(inc < np.pi / 2, -HEAD_R * np.cos(inc), HEAD_R * (inc - np.pi / 2))
            d = path / C_SOUND * sr
            src = idx - d - HEAD_R / C_SOUND * sr   # keep delays causal
            for band, fc in zip(bands, centers):
                out[:, m] += np.interp(src, idx, band, left=0) * shadow_gain(fc, inc)
    return out


def simulate(out: Path, stem_specs: list[str], duration_s: float | None = None, sr: int = 48000,
             yaw_amp_deg: float = 35.0, start_ts_us: int = 1_000_000) -> Session:
    paths, bearings = zip(*[(s.rsplit(":", 1)[0], float(s.rsplit(":", 1)[1])) for s in stem_specs])
    stems = [load_mono(p, sr) for p in paths]
    n = int(duration_s * sr) if duration_s else max(len(s) for s in stems)
    t = np.arange(n) / sr
    yaw = head_yaw(t, yaw_amp_deg)
    images = [render([s], [b], sr, yaw) for s, b in zip(stems, bearings)]  # each source alone
    audio = sum(images)
    peak = np.abs(audio).max()
    scale = 0.7 / peak if peak > 0.7 else 1.0
    audio *= scale

    tp = np.arange(0, n / sr, 1 / POSE_HZ)
    yaw_p, pitch_p = head_yaw(tp, yaw_amp_deg), 3.0 * np.sin(2 * np.pi * 0.2 * tp)
    poses = [{"device_ts_us": start_ts_us + int(ti * 1e6), "pos": [0.0, 1.6, 0.0],
              "rot_xyzw": list(quat_yaw_pitch(y, p))} for ti, y, p in zip(tp, yaw_p, pitch_p)]

    (out / "stems").mkdir(parents=True, exist_ok=True)
    sources = []
    for path, b, img in zip(paths, bearings, images):
        name = Path(path).stem
        # Reference = this source as the mics hear it (mean over mics), 16 kHz, same scale as the mix.
        ref = signal.resample_poly(img.mean(axis=1) * scale, 1, sr // 16000)
        wavfile.write(out / "stems" / f"{name}.wav", 16000, ref.astype(np.float32))
        sources.append({"name": name, "bearing_deg": b, "stem": f"stems/{name}.wav"})
    meta = {"device": "mock", "simulated": True, "sample_rate": sr, "mic_channels": 2,
            "mic_az_deg": [-90.0, 90.0], "camera": False, "start_ts_us": start_ts_us,
            "duration_s": n / sr, "yaw_amp_deg": yaw_amp_deg, "sources": sources}
    sess = Session(out, meta, audio, sr, poses)
    sess.save()
    return sess


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stem", action="append", required=True, help="path.wav:world_bearing_deg (+ = right)")
    ap.add_argument("--duration", type=float)
    ap.add_argument("--yaw-amp", type=float, default=35.0)
    a = ap.parse_args()
    s = simulate(a.out, a.stem, a.duration, yaw_amp_deg=a.yaw_amp)
    print(f"wrote {a.out}: {s.meta['duration_s']:.1f} s, {len(s.poses)} poses, sources {s.meta['sources']}")


if __name__ == "__main__":
    main()
