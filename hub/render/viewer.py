"""Self-contained session bundle and local server for the multi-stem inspection player.

Bundle layout (spec Section 10), rooted at `hub/sessions/<run_id>/av/`:

    manifest.json  metrics.json  mixture.wav  stems/source_<id>.wav
    residual.wav                          phase-projected playback PROXY (method in manifest)
    diagnostics/residual_waveform.wav     raw difference; never a discovery input
    residuals/unexplained_power.npz       spectral discovery artifact
    masks/<track>.jsonl  activity/<source>.json  activity/track_<track>.json
    frames/NNNNNN.png + frames.json       source video as variable-rate frames
    annotated/NNNNNN.png                  composited overlays (also feeds annotated.mp4)
    source_video.mp4 annotated.mp4        ONLY when an ffmpeg with an MP4 encoder is discovered
    media/                                multichannel master + 16/48 kHz views
    viewer.html  assets/viewer.js  assets/viewer.css

The folder has no CDN or network dependency. Serve it on localhost (`python -m av_scene serve`):
browsers restrict `file://` fetches and `<video>` seeking needs HTTP Range support.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ingest.media_io import FfmpegInfo, FfmpegUnavailable, encode_mp4, write_wav
from render.video import OverlayItem, render_sequence
from replay.residual import Projection, ResidualConfig, save_unexplained_power
from vision.mask_store import MaskRecord, write_track_jsonl

ASSET_DIR = Path(__file__).parent / "assets"


@dataclass
class BundleInput:
    out_dir: Path
    manifest: object                                # av_scene.schemas.RunManifest
    mixture: np.ndarray
    sample_rate: int
    stems: dict[str, np.ndarray]
    residual_proxy: np.ndarray
    residual_waveform: np.ndarray
    projection: Projection
    residual_cfg: ResidualConfig
    frame_paths: list[Path]
    pts_us: list[int]
    tracks: dict[str, list[MaskRecord]]
    activity: dict[str, dict]
    track_states: dict[str, list[str]]
    overlays: list[tuple[list[OverlayItem], str | None]]
    ffmpeg: FfmpegInfo = field(default_factory=FfmpegInfo)
    write_annotated: bool = True


def build_bundle(b: BundleInput):
    out = b.out_dir
    for sub in ("stems", "diagnostics", "residuals", "masks", "activity", "frames", "assets"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    assets: dict[str, str] = {"viewer": "viewer.html", "manifest": "manifest.json", "frames": "frames.json"}

    write_wav(out / "mixture.wav", b.mixture, b.sample_rate, "float32")
    assets["mixture"] = "mixture.wav"
    for sid, st in b.stems.items():
        write_wav(out / "stems" / f"source_{sid}.wav", st, b.sample_rate, "float32")
        assets[f"stem:{sid}"] = f"stems/source_{sid}.wav"
    write_wav(out / "residual.wav", b.residual_proxy, b.sample_rate, "float32")
    write_wav(out / "diagnostics" / "residual_waveform.wav", b.residual_waveform, b.sample_rate, "float32")
    save_unexplained_power(out / "residuals" / "unexplained_power.npz", b.projection, b.residual_cfg)
    assets.update({"residual": "residual.wav", "residual_waveform": "diagnostics/residual_waveform.wav",
                   "unexplained_power": "residuals/unexplained_power.npz"})

    for tid, recs in b.tracks.items():
        write_track_jsonl(out / "masks" / f"{tid}.jsonl", recs)
        (out / "activity" / f"track_{tid}.json").write_text(
            json.dumps({"track_id": tid, "state": b.track_states[tid], "evidence_tier": "mocked/synthetic"}))
        assets[f"masks:{tid}"] = f"masks/{tid}.jsonl"
        assets[f"track_state:{tid}"] = f"activity/track_{tid}.json"
    for sid, act in b.activity.items():
        (out / "activity" / f"{sid}.json").write_text(json.dumps(act))
        assets[f"activity:{sid}"] = f"activity/{sid}.json"

    for p in b.frame_paths:
        shutil.copyfile(p, out / "frames" / Path(p).name)
    names = [Path(p).name for p in b.frame_paths]
    annotated_ok = False
    if b.write_annotated:
        render_sequence([out / "frames" / n for n in names], b.overlays, out / "annotated")
        annotated_ok = True
    first = b.manifest.media.video[0]
    (out / "frames.json").write_text(json.dumps({
        "pts_us": b.pts_us, "files": [f"frames/{n}" for n in names], "width": first.width, "height": first.height,
        "annotated": [f"annotated/{n}" for n in names] if annotated_ok else None}))

    mp4 = {}
    if b.ffmpeg.can_mux_mp4:
        try:
            encode_mp4(b.ffmpeg, [out / "frames" / n for n in names], b.pts_us, out / "source_video.mp4", out / "mixture.wav")
            assets["source_video"] = "source_video.mp4"
            if annotated_ok:
                encode_mp4(b.ffmpeg, [out / "annotated" / n for n in names], b.pts_us, out / "annotated.mp4", out / "mixture.wav")
                assets["annotated_video"] = "annotated.mp4"
            mp4 = {"status": "written", "encoder": b.ffmpeg.video_encoder}
        except FfmpegUnavailable as exc:
            mp4 = {"status": "failed", "detail": str(exc)}
    else:
        mp4 = {"status": "skipped", "detail": "no ffmpeg with an MP4-capable encoder discovered; "
                                              "the viewer plays the PNG frame sequence"}
    b.manifest.capabilities["mp4"] = mp4

    for name, dest in (("viewer.html", "viewer.html"), ("viewer.js", "assets/viewer.js"), ("viewer.css", "assets/viewer.css")):
        shutil.copyfile(ASSET_DIR / name, out / dest)
    b.manifest.assets = assets
    b.manifest.metrics_path = "metrics.json"
    (out / "manifest.json").write_text(b.manifest.to_json(indent=1), encoding="utf-8")
    return b.manifest


# ---------------------------------------------------------------- local server
class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler with HTTP Range (needed for <video> seeking) and no caching."""

    _range_left: int | None = None

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the CLI quiet
        pass

    def send_head(self):
        self._range_left = None
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        size = os.path.getsize(path)
        if not m or not (m.group(1) or m.group(2)):
            return super().send_head()
        if m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        else:
            start, end = max(size - int(m.group(2)), 0), size - 1
        end = min(end, size - 1)
        if start >= size or start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None
        f = open(path, "rb")
        f.seek(start)
        self._range_left = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._range_left))
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        if self._range_left is None:
            return super().copyfile(source, outputfile)
        left = self._range_left
        while left > 0:
            chunk = source.read(min(65536, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)


def make_server(directory: str | Path, host: str = "127.0.0.1", port: int = 0) -> http.server.ThreadingHTTPServer:
    directory = str(Path(directory).resolve())

    class Handler(RangeHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

    return http.server.ThreadingHTTPServer((host, port), Handler)


def serve_in_thread(directory: str | Path, host: str = "127.0.0.1", port: int = 0):
    srv = make_server(directory, host, port)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, th
