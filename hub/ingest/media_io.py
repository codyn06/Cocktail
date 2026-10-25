"""Timestamped media index and file helpers for recorded AV scenes.

Time model: every stream has its own clock; `t_common_us = round(a * t_stream_us + b_us)`.
Video frames carry their own presentation timestamps (integer microseconds, possibly
variable-rate). Nothing here ever derives a timestamp as `frame_idx / nominal_fps`.
Missing data (gaps, padding, dropped frames) is recorded explicitly and is never evidence of
silence.

Audio: the multichannel master is preserved at its capture rate. 16 kHz (hub models) and
48 kHz (SAM Audio) views are derived and recorded; upsampling the 16 kHz hub ring would not
restore discarded bandwidth, so views are always derived from the master.

ffmpeg/ffprobe are optional. `locate_ffmpeg` looks at an explicit path, then the environment
(`COCKTAIL_FFMPEG`, `FFMPEG_BINARY`, `IMAGEIO_FFMPEG_EXE`), then PATH, then the optional
`imageio_ffmpeg` package, and reports what it found instead of assuming.
"""

from __future__ import annotations

import bisect
import math
import os
import shutil
import subprocess
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from av_scene.schemas import (AudioStreamInfo, AudioViewInfo, GapInterval, MediaIndex, StreamTransform,
                              VideoStreamInfo)

VIEW_RATES_HZ = (16_000, 48_000)
PNG_SIGNATURE = bytes([0x89]) + b"PNG" + bytes([0x0D, 0x0A, 0x1A, 0x0A])


# ------------------------------------------------------------------ timeline
def to_common_us(t_stream_us: float, tf: StreamTransform) -> int:
    return int(round(tf.a * t_stream_us + tf.b_us))


def from_common_us(t_common_us: float, tf: StreamTransform) -> float:
    return (t_common_us - tf.b_us) / tf.a


class MediaTimeline:
    """Query layer over a `MediaIndex` (all inputs/outputs in common-clock microseconds)."""

    def __init__(self, index: MediaIndex, video_id: str | None = None, audio_id: str | None = None):
        self.index = index
        self.video = next((v for v in index.video if video_id in (None, v.stream_id)), None)
        self.audio = next((a for a in index.audio if audio_id in (None, a.stream_id)), None)
        self._vpts = [to_common_us(p, self.video.transform) for p in self.video.pts_us] if self.video else []

    # ---- video
    @property
    def frame_pts_us(self) -> list[int]:
        return list(self._vpts)

    def frame_at(self, t_us: int) -> int:
        """Index of the frame being presented at `t_us` (-1 before the first frame)."""
        return bisect.bisect_right(self._vpts, t_us) - 1

    def quantization_ms_at(self, t_us: int) -> float:
        """q: half of the local bracketing frame interval (equals 1000/(2*FPS) at constant rate)."""
        v = self._vpts
        if len(v) < 2:
            return 0.0
        i = min(max(bisect.bisect_right(v, t_us) - 1, 0), len(v) - 2)
        return (v[i + 1] - v[i]) / 2000.0

    def first_frame_at_or_after(self, t_us: int) -> int:
        i = bisect.bisect_left(self._vpts, t_us)
        return i if i < len(self._vpts) else -1

    # ---- audio
    def sample_to_common_us(self, sample_idx: float) -> int:
        a = self.audio
        return to_common_us(a.start_us + sample_idx * 1e6 / a.sample_rate, a.transform)

    def common_us_to_sample(self, t_us: float) -> float:
        a = self.audio
        return (from_common_us(t_us, a.transform) - a.start_us) * a.sample_rate / 1e6

    # ---- gaps
    def gaps(self, start_us: int, end_us: int, stream_id: str | None = None) -> list[GapInterval]:
        return [g for g in self.index.gaps
                if g.start_us < end_us and g.end_us > start_us and stream_id in (None, g.stream_id)]

    def has_missing_data(self, start_us: int, end_us: int, stream_id: str | None = None) -> bool:
        return bool(self.gaps(start_us, end_us, stream_id))


def detect_dropped_frames(pts_us: list[int], tolerance: float = 1.6) -> list[GapInterval]:
    """Mark intervals where consecutive frames are much farther apart than the median spacing."""
    if len(pts_us) < 3:
        return []
    d = np.diff(pts_us)
    med = float(np.median(d))
    return [GapInterval(stream_id="video", start_us=int(pts_us[i]), end_us=int(pts_us[i + 1]),
                        kind="missing_frames", note=f"interval {d[i] / 1000:.1f} ms vs median {med / 1000:.1f} ms")
            for i in np.flatnonzero(d > tolerance * med)]


# ------------------------------------------------------------------ audio files
def write_wav(path: str | Path, data: np.ndarray, rate: int, subtype: str = "float32") -> None:
    """Write (n,) or (n, channels) audio. `float32` keeps samples exactly; `pcm16` clips to [-1, 1]."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(data)
    if subtype == "float32":
        wavfile.write(p, int(rate), x.astype(np.float32))
    elif subtype == "pcm16":
        wavfile.write(p, int(rate), (np.clip(x, -1.0, 1.0) * 32767.0).round().astype(np.int16))
    else:
        raise ValueError(f"unsupported subtype {subtype!r}")


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Return float32 (n, channels) in [-1, 1] and the sample rate."""
    rate, a = wavfile.read(path)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    elif a.dtype == np.int32:
        a = a.astype(np.float32) / 2147483648.0
    else:
        a = a.astype(np.float32)
    return (a[:, None] if a.ndim == 1 else a), int(rate)


def resample_view(x: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Polyphase resample (n, channels) -> (ceil(n*rate_out/rate_in), channels)."""
    if rate_in == rate_out:
        return x.astype(np.float32, copy=True)
    g = math.gcd(rate_in, rate_out)
    return signal.resample_poly(x, rate_out // g, rate_in // g, axis=0).astype(np.float32)


def write_audio_stream(out_dir: str | Path, stream_id: str, master: np.ndarray, rate: int, *,
                       layout: str = "calibrated_array", channel_order: list[str] | None = None,
                       start_us: int = 0, transform: StreamTransform | None = None,
                       view_rates=VIEW_RATES_HZ, rel_to: str | Path | None = None) -> AudioStreamInfo:
    """Persist the multichannel master untouched plus derived views; return the index entry."""
    out_dir = Path(out_dir)
    base = Path(rel_to) if rel_to else out_dir
    master = master[:, None] if master.ndim == 1 else master
    mp = out_dir / f"{stream_id}_master.wav"
    write_wav(mp, master, rate, "float32")
    views: dict[str, AudioViewInfo] = {}
    for vr in view_rates:
        v = resample_view(master, rate, vr)
        vp = out_dir / f"{stream_id}_{vr // 1000}k.wav"
        write_wav(vp, v, vr, "float32")
        views[f"{vr // 1000}k"] = AudioViewInfo(rate_hz=vr, path=vp.relative_to(base).as_posix(),
                                                 n_samples=len(v), channels=v.shape[1], derived_from="master")
    return AudioStreamInfo(stream_id=stream_id, sample_rate=rate, channels=master.shape[1], n_samples=len(master),
                           start_us=start_us, transform=transform or StreamTransform(),
                           master_path=mp.relative_to(base).as_posix(), layout=layout,
                           channel_order=channel_order or [f"ch{i}" for i in range(master.shape[1])], views=views)


def build_media_index(run_id: str, *, video_pts_us: list[int], width: int, height: int,
                      audio: list[AudioStreamInfo], nominal_fps: float | None = None,
                      frames_dir: str | None = None, video_path: str | None = None,
                      video_transform: StreamTransform | None = None, input_hashes: dict | None = None,
                      extra_gaps: list[GapInterval] | None = None, calibration_id: str | None = None,
                      ambient_temperature_c: float = 20.0, temperature_provenance: str = "assumed",
                      r_min_m: float = 0.5) -> MediaIndex:
    pts = [int(p) for p in video_pts_us]
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("video presentation timestamps must be strictly increasing")
    vs = VideoStreamInfo(stream_id="video", width=width, height=height, nominal_fps=nominal_fps, pts_us=pts,
                         transform=video_transform or StreamTransform(), frames_dir=frames_dir, video_path=video_path)
    gaps = detect_dropped_frames(pts) + list(extra_gaps or [])
    return MediaIndex(run_id=run_id, input_hashes=input_hashes or {}, video=[vs], audio=audio, gaps=gaps,
                      calibration_id=calibration_id, ambient_temperature_c=ambient_temperature_c,
                      temperature_provenance=temperature_provenance, r_min_m=r_min_m)


# ------------------------------------------------------------------ minimal PNG writer
def write_png(path: str | Path, rgb: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 image as PNG with only the standard library + numpy."""
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("expected (H, W, 3) uint8")
    h, w, _ = a.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(tag: bytes, body: bytes) -> bytes:
        c = struct.pack(">I", len(body)) + tag + body
        return c + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(png)


def read_png(path: str | Path) -> np.ndarray:
    """Decode an 8-bit RGB, non-interlaced PNG (all five row filters) into (H, W, 3) uint8."""
    b = Path(path).read_bytes()
    if b[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG file")
    pos, idat, w = 8, [], None
    while pos < len(b):
        n, tag = struct.unpack(">I4s", b[pos:pos + 8])
        body = b[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if (depth, ctype, interlace) != (8, 2, 0):
                raise ValueError("only 8-bit RGB non-interlaced PNG is supported")
        elif tag == b"IDAT":
            idat.append(body)
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, w * 3 + 1)
    out = np.zeros((h, w * 3), np.uint8)
    bpp = 3
    for y in range(h):
        f, row = raw[y, 0], raw[y, 1:].astype(np.int32)
        prev = out[y - 1].astype(np.int32) if y else np.zeros(w * 3, np.int32)
        if f == 0:
            cur = row
        elif f == 2:
            cur = row + prev
        else:
            cur = np.zeros(w * 3, np.int32)
            for x in range(w * 3):
                a = cur[x - bpp] if x >= bpp else 0
                c = prev[x - bpp] if x >= bpp else 0
                if f == 1:
                    pred = a
                elif f == 3:
                    pred = (a + prev[x]) // 2
                else:
                    p0 = a + prev[x] - c
                    pa, pb, pc = abs(p0 - a), abs(p0 - prev[x]), abs(p0 - c)
                    pred = a if pa <= pb and pa <= pc else (prev[x] if pb <= pc else c)
                cur[x] = row[x] + pred
        out[y] = (cur & 0xFF).astype(np.uint8)
    return out.reshape(h, w, 3)


# ------------------------------------------------------------------ ffmpeg discovery
class FfmpegUnavailable(RuntimeError):
    pass


@dataclass
class FfmpegInfo:
    ffmpeg: str | None = None
    ffprobe: str | None = None
    version: str | None = None
    source: str = "none"          # config | env:<NAME> | path | imageio_ffmpeg | none
    can_mux_mp4: bool = False
    video_encoder: str | None = None

    def to_dict(self) -> dict:
        return {"ffmpeg": self.ffmpeg, "ffprobe": self.ffprobe, "version": self.version, "source": self.source,
                "can_mux_mp4": self.can_mux_mp4, "video_encoder": self.video_encoder}


def _run(args: list[str], timeout: float = 20.0) -> str:
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")


def _probe_encoders(exe: str) -> tuple[str | None, str | None]:
    try:
        version = (_run([exe, "-version"]).splitlines() or [""])[0].strip() or None
        enc = _run([exe, "-hide_banner", "-encoders"])
    except (OSError, subprocess.SubprocessError):
        return None, None
    for name in ("libx264", "mpeg4"):
        if f" {name} " in enc:
            return version, name
    return version, None


def locate_ffmpeg(config_path: str | os.PathLike | None = None) -> FfmpegInfo:
    candidates: list[tuple[str, str]] = []
    if config_path:
        candidates.append((str(config_path), "config"))
    for env in ("COCKTAIL_FFMPEG", "FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        if os.environ.get(env):
            candidates.append((os.environ[env], f"env:{env}"))
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append((on_path, "path"))
    try:  # optional package; never a dependency
        import imageio_ffmpeg  # type: ignore

        candidates.append((imageio_ffmpeg.get_ffmpeg_exe(), "imageio_ffmpeg"))
    except Exception:
        pass
    for exe, source in candidates:
        if not Path(exe).is_file():
            continue
        version, encoder = _probe_encoders(exe)
        if version is None:
            continue
        probe = shutil.which("ffprobe") or None
        sib = Path(exe).with_name("ffprobe" + Path(exe).suffix)
        if sib.is_file():
            probe = str(sib)
        return FfmpegInfo(exe, probe, version, source, encoder is not None, encoder)
    return FfmpegInfo()


def probe_video_pts(info: FfmpegInfo, path: str | Path) -> list[int]:
    """Presentation timestamps (integer microseconds) of every video packet, via ffprobe."""
    if not info.ffprobe:
        raise FfmpegUnavailable("ffprobe not found; set COCKTAIL_FFMPEG or add it to PATH")
    out = _run([info.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                "-of", "csv=p=0", str(path)], timeout=120)
    pts = sorted(int(round(float(line.split(",")[0]) * 1e6)) for line in out.splitlines()
                 if line.strip() and line.split(",")[0].replace(".", "", 1).replace("-", "", 1).isdigit())
    if not pts:
        raise FfmpegUnavailable(f"no video packets found in {path}")
    return pts


def encode_mp4(info: FfmpegInfo, frame_paths: list[Path], pts_us: list[int], out_path: str | Path,
               audio_path: str | Path | None = None, tail_us: int = 40_000) -> None:
    """Mux a PNG sequence (variable frame rate via per-frame durations) and optional audio to MP4."""
    if not (info.ffmpeg and info.can_mux_mp4):
        raise FfmpegUnavailable("no ffmpeg with an MP4-capable video encoder was found")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".concat.txt")
    lines = []
    for i, f in enumerate(frame_paths):
        dur = (pts_us[i + 1] - pts_us[i]) if i + 1 < len(pts_us) else tail_us
        lines.append(f"file '{Path(f).resolve().as_posix()}'\nduration {dur / 1e6:.6f}")
    lines.append(f"file '{Path(frame_paths[-1]).resolve().as_posix()}'")
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [info.ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if audio_path:
        cmd += ["-i", str(audio_path)]
    cmd += ["-fps_mode", "vfr", "-c:v", info.video_encoder or "mpeg4"]
    if info.video_encoder == "libx264":
        cmd += ["-pix_fmt", "yuv420p", "-crf", "20"]
    else:
        cmd += ["-pix_fmt", "yuv420p", "-q:v", "3"]
    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd.append(str(out))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    lst.unlink(missing_ok=True)
    if r.returncode != 0:
        raise FfmpegUnavailable(f"ffmpeg failed: {r.stderr.strip()[-400:]}")

