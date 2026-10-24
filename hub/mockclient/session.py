"""Recorded/simulated headset session on disk (task B4).

A session is a folder:
  session.json   device info, sample rate, mic layout, start time; for simulated sessions also
                 the ground-truth sources (name, world bearing, stem file)
  audio.wav      multichannel mic audio (int16), first sample at start_ts_us
  pose.jsonl     one PoseSample-like object per line: device_ts_us, pos, rot_xyzw
  video/         optional JPEG frames named <device_ts_us>.jpg
  stems/         optional dry ground-truth stems at 16 kHz (simulated sessions only)
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

    @property
    def start_ts_us(self) -> int:
        return int(self.meta.get("start_ts_us", 0))

    @classmethod
    def load(cls, path: str | Path) -> Session:
        path = Path(path)
        meta = json.loads((path / "session.json").read_text())
        sr, a = wavfile.read(path / "audio.wav")
        a = a.astype(np.float32) / 32768.0 if a.dtype == np.int16 else a.astype(np.float32)
        a = a[:, None] if a.ndim == 1 else a
        poses = []
        if (path / "pose.jsonl").exists():
            poses = [json.loads(l) for l in (path / "pose.jsonl").read_text().splitlines() if l.strip()]
        frames = sorted((int(f.stem), f) for f in (path / "video").glob("*.jpg")) if (path / "video").exists() else []
        return cls(path, meta, a, sr, poses, frames)

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "session.json").write_text(json.dumps(self.meta, indent=1))
        wavfile.write(self.path / "audio.wav", self.sample_rate,
                      (np.clip(self.audio, -1, 1) * 32767).astype(np.int16))
        (self.path / "pose.jsonl").write_text("".join(json.dumps(p) + "\n" for p in self.poses))
