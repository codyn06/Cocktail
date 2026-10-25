"""Phase 4 offline pipeline: visual -> speech -> (diarization) -> events -> residual -> export, all chunked.

This module is the integration layer. Stage logic lives in ``replay.chunked`` (speech/visual chunks),
``av_scene.event_scope`` (event windows), ``replay.residual`` (projection) and ``render.long_bundle`` (export).
Memory rule: nothing here holds an array whose size grows with clip length; per-chunk artifacts go to the
stage cache on disk and are streamed back one at a time.

Evidence labelling is the caller's job (manifest ``evidence_tier``); this module never claims real-time
operation or device validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from replay import residual as rs
from replay.chunked import ChunkedSpeechRunner, SpeechRunResult, read_arrays, resample_to
from .cache import StageCache, code_fingerprint, hash_config
from .config import AvConfig
from .contracts import SPEECH_SR, ChunkSpec


# ------------------------------------------------------------------------------------- bandwidth
def stem_nyquist_hz(origin_rate_hz: int, master_rate_hz: int) -> float | None:
    """Real bandwidth limit of a stem that is processed at the master rate but originated at
    ``origin_rate_hz``. A 16 kHz speech stem has an 8 kHz Nyquist limit: upsampling to 48 kHz adds no evidence
    above it, so the residual must not be evaluated there while that stem is active. ``None`` = full band."""
    if origin_rate_hz >= master_rate_hz:
        return None
    return origin_rate_hz / 2.0


@dataclass
class SparseStem:
    """A stem that exists only for a span (event windows). Everything outside is silence by definition."""

    source_id: str
    start_master_sample: int
    audio: np.ndarray                 # mono float32 at ``origin_rate_hz``
    origin_rate_hz: int


# ------------------------------------------------------------------------------------- residual
class ResidualStage:
    """Per-chunk phase-projected remainder. The proxy is computed on the chunk WINDOW (context on both sides) and
    only the owned region is kept, so output is nonduplicated and memory is one window regardless of clip length."""

    def __init__(self, runner: ChunkedSpeechRunner, cfg: AvConfig):
        self.r, self.cfg = runner, cfg
        self.fp = code_fingerprint(rs, __import__(__name__, fromlist=["x"]))

    def _stems_for_window(self, spec: ChunkSpec, sentry: Path, events: list[SparseStem], m0: int, m1: int,
                          master_rate: int):
        stems, nyq, ids = [], [], []
        for sid, (start16, a) in sorted(read_arrays(sentry, "win", "win.json").items()):
            up = resample_to(a[None], SPEECH_SR, master_rate)[0]
            n = m1 - m0
            up = up[:n] if len(up) >= n else np.pad(up, (0, n - len(up)))
            stems.append(up.astype(np.float32))
            nyq.append(stem_nyquist_hz(SPEECH_SR, master_rate))
            ids.append(sid)
        for ev in events:
            rate = ev.origin_rate_hz
            e0 = ev.start_master_sample
            e1 = e0 + int(len(ev.audio) * master_rate / rate)
            if e1 <= m0 or e0 >= m1:
                continue
            up = resample_to(ev.audio[None].astype(np.float32), rate, master_rate)[0]
            buf = np.zeros(m1 - m0, np.float32)
            a0, a1 = max(e0, m0), min(e0 + len(up), m1)
            buf[a0 - m0:a1 - m0] = up[a0 - e0:a1 - e0]
            stems.append(buf)
            nyq.append(stem_nyquist_hz(rate, master_rate))
            ids.append(ev.source_id)
        return stems, nyq, ids

    def run(self, res: SpeechRunResult, events: list[SparseStem], event_keys: dict[str, str] | None = None
            ) -> list[tuple[ChunkSpec, Path]]:
        src = self.r.src
        master_rate = src.audio_rate_hz
        out = []
        for rec in res.records:
            spec = rec.spec
            m0, m1 = self.r._master(spec.win_start), self.r._master(spec.win_end)
            o0, o1 = self.r._master(spec.own_start), self.r._master(spec.own_end)
            overlapping = sorted(k for k, v in (event_keys or {}).items() if v)
            key = self.r.cache.key(
                "residual_chunk", inputs={"media": self.r.media_id, "chunk": str(spec.to_dict())},
                config={"residual": self.cfg.residual.__dict__, "master_rate": master_rate},
                upstream={"speech_chunk": rec.speech_key, **{f"event:{k}": event_keys[k] for k in overlapping}},
                fingerprint=self.fp)

            def produce(d: Path, meta: dict, spec=spec, rec=rec, m0=m0, m1=m1, o0=o0, o1=o1) -> None:
                self.r.fault("residual:mid", spec.index)
                x = src.read_audio(m0, m1, master_rate).mean(axis=0).astype(np.float32)
                stems, nyq, ids = self._stems_for_window(spec, rec.speech_dir, events, m0, m1, master_rate)
                x = x[: m1 - m0] if len(x) >= m1 - m0 else np.pad(x, (0, m1 - m0 - len(x)))
                spans, proj = rs.find_unexplained_events(x, stems, master_rate, self.cfg.residual, stem_nyquist_hz=nyq)
                proxy = proj.proxy_waveform
                proxy = proxy[: m1 - m0] if len(proxy) >= m1 - m0 else np.pad(proxy, (0, m1 - m0 - len(proxy)))
                np.save(d / "proxy.npy", proxy[o0 - m0:o1 - m0].astype(np.float32))
                own_s0 = (o0 - m0) / master_rate
                own_s1 = (o1 - m0) / master_rate
                keep = [dict(s, start_s=s["start_s"] + m0 / master_rate, end_s=s["end_s"] + m0 / master_rate)
                        for s in spans if s["end_s"] > own_s0 and s["start_s"] < own_s1]
                meta.update({"stems": ids, "stem_nyquist_hz": nyq, "master_rate": master_rate,
                             "owned_master": [o0, o1], "unexplained_spans": keep,
                             "projection_error_db": proj.projection_error_db})
                (d / "residual.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
                self.r.counters.chunks_computed.append(("residual", spec.index))

            entry, computed = self.r.cache.get_or_compute("residual_chunk", key, produce,
                                                           required=("proxy.npy", "residual.json"))
            if not computed:
                self.r.counters.chunks_restored.append(("residual", spec.index))
            self.r.fault("residual:after", spec.index)
            out.append((spec, entry))
        return out
