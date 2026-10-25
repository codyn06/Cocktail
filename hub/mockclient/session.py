"""Recorded/simulated headset session on disk (task B4).

A session is a folder:
  session.json   device info, sample rate, mic layout, start time; for simulated sessions also
                 the ground-truth sources (name, world bearing, stem file)
  audio.wav      multichannel mic audio (int16), first sample at start_ts_us
  pose.jsonl     one PoseSample-like object per line: device_ts_us, pos, rot_xyzw
  video/         optional JPEG frames named <device_ts_us>.jpg, plus optional frames.jsonl: one
                 VideoFrame header per line (device_ts_us, width, height, cam_pos, cam_rot_xyzw, hfov_deg, ...)
  stems/         optional ground truth at 16 kHz: each source alone as the mics hear it (mean
                 over mics), same scale as audio.wav (simulated sessions only)

A take written by quest/tools/record_session.py (meta.json, pose.csv instead of session.json,
pose.jsonl; the same audio.wav and video/) also loads: start_ts_us is the first audio timestamp, the
device and camera_hfov_deg come from the recorded Hello, and poses carry the rotation only (pos is 0,0,0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile


@dataclass
class Session:
    path: Path
    meta: dict
    audio: np.ndarray            # (n, channels) float32 in [-1, 1]
    sample_rate: int
    poses: list[dict] = field(default_factory=list)
    frames: list[tuple[int, Path]] = field(default_factory=list)  # (device_ts_us, jpeg path)
    frame_meta: dict[int, dict] = field(default_factory=dict)  # device_ts_us -> VideoFrame header fields

    @property
    def start_ts_us(self) -> int:
        return int(self.meta.get("start_ts_us", 0))

    @classmethod
    def load(cls, path: str | Path) -> Session:
        path = Path(path)
        if (path / "session.json").exists():
            meta = json.loads((path / "session.json").read_text())
        else:
            meta = _meta_from_recording(path)
        sr, a = wavfile.read(path / "audio.wav")
        a = a.astype(np.float32) / 32768.0 if a.dtype == np.int16 else a.astype(np.float32)
        a = a[:, None] if a.ndim == 1 else a
        poses = []
        if (path / "pose.jsonl").exists():
            poses = [json.loads(l) for l in (path / "pose.jsonl").read_text().splitlines() if l.strip()]
        elif (path / "pose.csv").exists():
            poses = _poses_from_csv(path / "pose.csv")
        frames = sorted((int(f.stem), f) for f in (path / "video").glob("*.jpg")) if (path / "video").exists() else []
        frame_meta = {}
        if (path / "video" / "frames.jsonl").exists():
            rows = [json.loads(l) for l in (path / "video" / "frames.jsonl").read_text().splitlines() if l.strip()]
            frame_meta = {int(r["device_ts_us"]): r for r in rows}
        return cls(path, meta, a, sr, poses, frames, frame_meta)

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "session.json").write_text(json.dumps(self.meta, indent=1))
        wavfile.write(self.path / "audio.wav", self.sample_rate,
                      (np.clip(self.audio, -1, 1) * 32767).astype(np.int16))
        (self.path / "pose.jsonl").write_text("".join(json.dumps(p) + "\n" for p in self.poses))


def _meta_from_recording(path: Path) -> dict:
    """session.json equivalent for a take written by quest/tools/record_session.py."""
    rec = json.loads((path / "meta.json").read_text())
    hello = rec.get("hello") or {}
    meta = {k: rec[k] for k in ("truth_deg", "note", "video_camera_flipped") if k in rec}
    meta.update(start_ts_us=rec.get("first_audio_ts_us") or 0, device=hello.get("device", "quest"),
                camera_hfov_deg=hello.get("camera_hfov_deg"), recorded=True)
    return meta


def _poses_from_csv(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()[1:]  # header: t_us,qx,qy,qz,qw
    rows = [[float(v) for v in l.split(",")] for l in lines if l.strip()]
    return [{"device_ts_us": int(r[0]), "pos": [0.0, 0.0, 0.0], "rot_xyzw": r[1:5]} for r in rows]
