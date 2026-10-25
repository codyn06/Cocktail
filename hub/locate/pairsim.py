"""Simulated Quest 3S takes for the mic-pair delay method (simulated data only).

Writes take folders in the recorder's format (audio.wav, pose.csv, meta.json), so
`locate.real_eval` and `locate.bench` score them exactly like real takes:

    cd hub
    uv run python -m locate.pairsim sessions/pairsim --angles 0,90,180,-90,45,135,-45,-135
    uv run python -m locate.bench sessions/pairsim

With --scene it instead writes one mockclient session (session.json, audio.wav, pose.jsonl,
stems/) with a speech-like talker at each angle, for the live path (`locate.score_live`):

    uv run python -m locate.pairsim sessions/two-talkers --scene --angles -30,40

Same protocol as quest/tools/run_locate_take.ps1: room only until 4 s, pink noise from the
source, head still until the turn cue at 8 s, then slow left/right turns (~+-40 deg) until 28 s.

The room is deliberately richer than pair.py's model (no "inverse crime"): early reflections
from fixed world directions, each with its own inter-mic delay; a late reverberant tail and a
room noise that are near-identical at both mics (5 mm apart, so diffuse sound is almost fully
coherent there and pulls delays toward zero); slight source elevation; a channel offset and a
phase mismatch between the two mics; independent sensor noise. It cannot reproduce the real
headset's own processing, so treat sim results as a check of the method, not of the 3S.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

C_SOUND = 343.0
CUE = {"noise_until": 4.0, "go": 8.0, "stop": 28.0}


@dataclass
class Room:
    spacing_m: float = 0.0052          # mic spacing (the fitted 3S calibration says ~5 mm)
    axis_deg: float = -112.0           # pair axis in the head frame (+ = right of forward)
    offset_s: float = -0.8e-6          # fixed channel offset
    elev_deg: float = 8.0              # source a little above the mics
    n_reflections: int = 6
    reflection_gain: tuple = (0.25, 0.55)
    reflection_delay_ms: tuple = (2.0, 12.0)
    reverb_db: float = -6.0            # late tail relative to the direct sound
    rt60_s: float = 0.4
    room_db: float = -22.0             # room noise relative to the source
    sensor_db: float = -45.0           # independent noise per mic
    mismatch_hz: float = 9000.0        # one-pole lowpass on channel 0 only: phase mismatch; it also
                                       # delays channel 0 by ~9 us at low frequencies, so the
                                       # effective offset is ~+8 us and frequency-dependent
    head_amp_deg: float = 40.0


def pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    X[1:] /= np.sqrt(f[1:])
    X[0] = 0
    x = np.fft.irfft(X, n)
    return x / x.std()


def talker(n, rng, rate=48000):
    """Speech-like source: speech-shaped noise with a syllable-rate envelope, in phrases of
    1.5-4 s separated by 0.3-1.5 s pauses (unit RMS while talking)."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / rate)
    X *= (f > 100) / (1 + (np.maximum(f, 1) / 500.0) ** 0.5 * (1 + (f / 4000.0) ** 2))
    x = np.fft.irfft(X, n)
    E = np.fft.rfft(rng.standard_normal(n))
    env = np.fft.irfft(E * (f < 6.0), n)  # syllable-rate (< 6 Hz) envelope
    env = np.maximum(env / (env.std() + 1e-12) + 0.8, 0.0)
    on = np.zeros(n, bool)
    i = int(rng.uniform(0, 1.0) * rate)
    while i < n:
        k = int(rng.uniform(1.5, 4.0) * rate)
        on[i:i + k] = True
        i += k + int(rng.uniform(0.3, 1.5) * rate)
    y = x * env * on
    return y / (y[on].std() + 1e-12)


def head_yaw(t, rng, amp, go=None, stop=None):
    """Still before the turn cue and after the stop cue; slow irregular turns in between."""
    go = CUE["go"] if go is None else go
    stop = CUE["stop"] if stop is None else stop
    f = rng.uniform(0.12, 0.3, 2)
    ph = rng.uniform(0, 2 * np.pi, 2)
    s = np.clip((t - go) / 0.8, 0, 1) * np.clip((stop - t) / 0.8, 0, 1)
    y = np.sin(2 * np.pi * f[0] * (t - go) + ph[0]) + 0.35 * np.sin(2 * np.pi * f[1] * (t - go) + ph[1])
    return amp / 1.2 * s * y + rng.normal(0, 0.3, len(t))


def _stft(x, N, H):
    w = np.hanning(N + 1)[:N]
    idx = np.arange(N)[None, :] + H * np.arange((len(x) - N) // H + 1)[:, None]
    return np.fft.rfft(x[idx] * w, axis=1), w, idx


def _istft(X, w, idx, n):
    y = np.zeros(n)
    norm = np.zeros(n)
    np.add.at(y, idx, np.fft.irfft(X, idx.shape[1], axis=1) * w)
    np.add.at(norm, idx, np.broadcast_to(w * w, idx.shape))
    return y / np.maximum(norm, 1e-9)


def simulate(truth_deg, seed=0, room: Room | None = None, secs=30.0, rate=48000):
    """Returns (audio (n, 2) float, pose times s, yaw deg). Channel 0 hears the source tau later
    than channel 1, with tau = D sin(bearing - axis) + offset, as pair.py assumes."""
    r = room or Room()
    rng = np.random.default_rng(seed)
    n = int(secs * rate)
    t = np.arange(n) / rate
    src = pink(n, rng) * (t >= CUE["noise_until"])
    pt = np.arange(0, secs, 1 / 72)  # the 3S sends head pose at ~72 Hz
    yaw = head_yaw(pt, rng, r.head_amp_deg)
    ch0, ch1 = _render(src, truth_deg, pt, yaw, r, rng, rate)
    x = _finish(ch0, ch1, r, rng, rate)
    return 0.1 * x / np.abs(x).max() * 3, pt, yaw


def _render(src, az_deg, pt, yaw, r: Room, rng, rate):
    """One source at world bearing az_deg as the two mics hear it: direct path, early
    reflections from fixed world directions, and a late reverberant tail (coherent, zero delay)."""
    n = len(src)
    N, H = 1024, 256
    tf = (np.arange((n - N) // H + 1) * H + N / 2) / rate
    yf = np.interp(tf, pt, yaw)
    D = r.spacing_m / C_SOUND
    f = np.fft.rfftfreq(N, 1 / rate)

    # direct path + early reflections: (world azimuth, elevation, gain, extra delay in samples)
    paths = [(az_deg, r.elev_deg, 1.0, 0)]
    for _ in range(r.n_reflections):
        paths.append((rng.uniform(-180, 180), rng.uniform(-30, 40), rng.uniform(*r.reflection_gain),
                      int(rng.uniform(*r.reflection_delay_ms) * rate / 1000)))
    ch0 = np.zeros(n)
    ch1 = np.zeros(n)
    for az, el, g, lag in paths:
        s = g * np.roll(src, lag)
        tau = D * math.cos(math.radians(el)) * np.sin(np.radians(az - r.axis_deg - yf)) + r.offset_s
        S, w, idx = _stft(s, N, H)
        ch0 += _istft(S * np.exp(-2j * np.pi * f[None, :] * tau[:, None]), w, idx, n)
        ch1 += s
    # late reverb and room noise: diffuse, so near-identical at 5 mm (coherent, zero delay)
    k = int(r.rt60_s * rate)
    tail = rng.standard_normal(k) * np.exp(-6.9 * np.arange(k) / k)
    late = np.fft.irfft(np.fft.rfft(src, n + k) * np.fft.rfft(tail, n + k), n + k)[:n]
    late *= 10 ** (r.reverb_db / 20) * src.std() / (late.std() + 1e-12)
    return ch0 + late, ch1 + late


def _finish(ch0, ch1, r: Room, rng, rate):
    """Room noise (diffuse: the same at both mics), the channel-0 phase mismatch, sensor noise."""
    n = len(ch0)
    room_n = pink(n, rng) * 10 ** (r.room_db / 20)
    ch0 = ch0 + room_n
    ch1 = ch1 + room_n
    a = math.exp(-2 * math.pi * r.mismatch_hz / rate)  # one-pole lowpass: frequency-dependent phase
    ch0 = np.fft.irfft(np.fft.rfft(ch0) * (1 - a) / (1 - a * np.exp(-2j * np.pi * np.fft.rfftfreq(n))), n)
    x = np.stack([ch0, ch1], 1)
    x += rng.standard_normal(x.shape) * 10 ** (r.sensor_db / 20)
    return x


def write_session(out: Path, angles, seed=0, room: Room | None = None, secs=40.0, rate=48000,
                  start_ts_us: int = 1_000_000):
    """A mockclient session (see mockclient/session.py) with a speech-like talker at each world
    bearing, the head turning slowly from 1 s on. stems/ hold each talker alone as the mics
    hear it (mean over mics), 16 kHz, same scale as audio.wav, for scoring and oracle masks."""
    from mockclient import Session
    from mockclient.simulate import quat_yaw_pitch

    r = room or Room()
    rng = np.random.default_rng(seed)
    n = int(secs * rate)
    pt = np.arange(0, secs, 1 / 72)
    yaw = head_yaw(pt, rng, r.head_amp_deg, go=1.0, stop=secs - 0.5)
    images = [_render(talker(n, rng, rate), float(a), pt, yaw, r, rng, rate) for a in angles]
    x = _finish(sum(i[0] for i in images), sum(i[1] for i in images), r, rng, rate)
    scale = 0.3 / np.abs(x).max()
    out.mkdir(parents=True, exist_ok=True)
    (out / "stems").mkdir(exist_ok=True)
    sources = []
    for k, (a, (c0, c1)) in enumerate(zip(angles, images)):
        name = f"talker{k + 1}"
        ref = signal.resample_poly((c0 + c1) / 2 * scale, 1, rate // 16000).astype(np.float32)
        wavfile.write(out / "stems" / f"{name}.wav", 16000, ref)
        sources.append({"name": name, "bearing_deg": float(a), "stem": f"stems/{name}.wav"})
    poses = [{"device_ts_us": start_ts_us + int(ti * 1e6), "pos": [0.0, 1.6, 0.0],
              "rot_xyzw": list(quat_yaw_pitch(float(yi), 0.0))} for ti, yi in zip(pt, yaw)]
    meta = {"device": "pairsim", "simulated": True, "sample_rate": rate, "mic_channels": 2,
            "camera": False, "start_ts_us": start_ts_us, "duration_s": secs,
            "yaw_amp_deg": r.head_amp_deg, "seed": seed, "sources": sources}
    sess = Session(out, meta, (x * scale).astype(np.float32), rate, poses)
    sess.save()
    return sess


def write_take(out: Path, truth_deg: float, seed=0, room: Room | None = None, secs=30.0, rate=48000,
               pose_lag_s: float = 0.0):
    """pose_lag_s: stamp head poses this much later than they happened (a timing fault to find)."""
    x, pt, yaw = simulate(truth_deg, seed, room, secs, rate)
    out.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out / "audio.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(np.clip(x * 32767, -32768, 32767).astype("<i2").tobytes())
    base = 1_000_000
    h = np.radians(yaw) / 2  # Unity: yaw about +y, + = right
    rows = np.stack([base + ((pt + pose_lag_s) * 1e6).astype(np.int64), np.zeros_like(h), np.sin(h), np.zeros_like(h), np.cos(h)], 1)
    np.savetxt(out / "pose.csv", rows, delimiter=",", header="t_us,qx,qy,qz,qw", comments="",
               fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])
    (out / "meta.json").write_text(json.dumps({
        "truth_deg": truth_deg, "note": f"pairsim seed {seed}", "sample_rate": rate, "channels": 2,
        "first_audio_ts_us": base, "audio_frames": len(x), "pose_samples": len(rows), "cue": CUE,
        "simulated": True, "pose_lag_s": pose_lag_s}, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m locate.pairsim", description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--angles", default="0,90,180,-90", help="source bearings, deg (+ = right)")
    ap.add_argument("--seeds", type=int, default=1, help="takes per angle (different rooms and head motion)")
    ap.add_argument("--scene", action="store_true",
                    help="one mockclient session with a talker at every angle (for locate.score_live)")
    ap.add_argument("--seconds", type=float, default=40.0, help="--scene length")
    a = ap.parse_args(argv)
    if a.scene:
        write_session(a.out, [float(v) for v in a.angles.split(",")], seed=a.seeds - 1, secs=a.seconds)
        print(f"wrote {a.out}")
        return
    for ang in (float(v) for v in a.angles.split(",")):
        for s in range(a.seeds):
            name = f"sim-t{ang:+.0f}-s{s}".replace("+", "")
            write_take(a.out / name, ang, seed=1000 * s + int(ang) % 360)
            print(f"wrote {a.out / name}")


if __name__ == "__main__":
    main()
