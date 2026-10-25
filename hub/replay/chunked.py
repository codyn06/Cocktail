"""Chunked offline speech pipeline: bounded decode -> visual -> speech -> identity -> stitch, resumable.

Per chunk (4 s window / 2 s overlap by default, see ``av_scene.contracts``):
  1. read ONLY that window of audio (master -> 16 kHz) and frames (pts-selected), never the whole clip;
  2. visual stage   (cache ``visual_chunk``): persistent track ids/boxes/masks; state chained chunk to chunk;
  3. speech stage   (cache ``speech_chunk``): S3FD boxes -> TalkNet -> MossFormer2 -> optional AV-TSE with the
     Phase 3 switches -> persistent identities (IdentityStore, slot permutation resolved) -> stitch the
     chunk's OWNED region (crossfade only across resolved identities) -> timeline segments;
  4. publish atomically; the next chunk's key includes the hash of this chunk's output state.

Everything a chunk needs from its predecessor is explicit: ``PipelineState`` (identity, enrollment seconds,
output progress) plus the previous entry's crossfade tails. A valid cached entry is restored WITHOUT invoking a
model; an interrupted run only recomputes unfinished or invalidated chunks, and a resumed run is bit-identical
to an uninterrupted one because every stage is deterministic given (inputs, config, state_in).

Not real time. Nothing here is hardware-validated unless a run report says so.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

from av_scene import speech as speech_mod
from av_scene import stitch as stitch_mod
from av_scene.cache import StageCache, code_fingerprint, hash_config
from av_scene.config import AvConfig
from av_scene.contracts import (SPEECH_SR, AvMap, ChunkSpec, PipelineState, SpeechModels, VisualFrontEnd,
                                crossfade_check, plan_chunks, sample_to_us, source_id_for, video_index_range)
from av_scene.schemas import TimelineSegment

FPS25 = 25


class InjectedCrash(RuntimeError):
    """Raised by a fault hook to simulate a killed process at a stage boundary (tests)."""


FaultHook = Callable[[str, int], None]      # (boundary name, chunk index)


# ----------------------------------------------------------------------------------------- sources
class ArraySource:
    """In-memory ``ChunkSource`` for tests/fixtures. ``audio`` is [channels, n] at ``audio_rate_hz``;
    ``frames`` an array [T,H,W,3] or a callable ``(i0, i1) -> uint8 array`` (so long scenes need not be stored)."""

    def __init__(self, audio, audio_rate_hz: int, pts_us: list[int], frames, av_map: AvMap | None = None,
                 n_audio_samples: int | None = None):
        self._audio = audio
        self.audio_rate_hz = audio_rate_hz
        self.n_audio_samples = n_audio_samples if n_audio_samples is not None else int(audio.shape[-1])
        self.av_map = av_map or AvMap(audio_rate_hz)
        self._pts = list(pts_us)
        self._frames = frames
        self.reads: list[tuple[str, int, int]] = []              # audit trail for boundedness tests

    def video_pts_us(self) -> list[int]:
        return self._pts

    def read_audio(self, start_sample: int, end_sample: int, rate_hz: int) -> np.ndarray:
        self.reads.append(("audio", start_sample, end_sample))
        x = self._audio(start_sample, end_sample) if callable(self._audio) else self._audio[..., start_sample:end_sample]
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        return x if rate_hz == self.audio_rate_hz else resample_to(x, self.audio_rate_hz, rate_hz)

    def read_frames(self, i0: int, i1: int) -> np.ndarray:
        self.reads.append(("frames", i0, i1))
        return self._frames(i0, i1) if callable(self._frames) else self._frames[i0:i1]


def resample_to(x: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Polyphase resample [C, n] (derived from the MASTER; never chained through another view)."""
    if rate_in == rate_out:
        return x
    g = int(np.gcd(rate_in, rate_out))
    return resample_poly(x, rate_out // g, rate_in // g, axis=-1).astype(np.float32)


class TorchcodecSource:
    """Incremental decode with TorchCodec (UNTESTED here: libtorchcodec cannot load in the dev shell; see
    docs/STATUS.md). Decodes only the requested sample/frame range per call; never the full clip.
    The audio master keeps its capture rate; frames are resized so max(h, w) <= ``max_side``."""

    def __init__(self, path: str | Path, *, max_side: int = 640, ffmpeg_info=None):
        from av_scene import preflight
        preflight.configure_ffmpeg_dlls()
        from torchcodec.decoders import AudioDecoder, VideoDecoder
        self.path = str(path)
        self.max_side = max_side
        self._ad = AudioDecoder(self.path)
        self._vd = VideoDecoder(self.path, dimension_order="NCHW")
        md = self._ad.metadata
        self.audio_rate_hz = int(md.sample_rate)
        dur = float(md.duration_seconds_from_header or 0.0)
        self.n_audio_samples = int(round(dur * self.audio_rate_hz))
        if self.n_audio_samples <= 0:
            raise ValueError(f"{self.path}: the container header gives no audio duration; refusing to run on an "
                             "empty plan (re-mux the clip or pass a source with a known length)")
        self.av_map = AvMap(self.audio_rate_hz, audio_start_us=0, video_transform="container_pts")
        self._pts: list[int] | None = None
        self._ffmpeg = ffmpeg_info

    def video_pts_us(self) -> list[int]:
        if self._pts is None:
            pts = None
            if self._ffmpeg is not None and getattr(self._ffmpeg, "ffprobe", None):
                from ingest.media_io import probe_video_pts
                pts = probe_video_pts(self._ffmpeg, self.path)
            if pts is None:                                        # bounded streaming pass: pts only, frames dropped
                pts, n, step = [], len(self._vd), 64
                for i in range(0, n, step):
                    b = self._vd.get_frames_in_range(i, min(i + step, n))
                    pts.extend(int(round(float(p) * 1e6)) for p in b.pts_seconds)
                    del b
            self._pts = pts
        return self._pts

    def read_audio(self, start_sample: int, end_sample: int, rate_hz: int) -> np.ndarray:
        s = self._ad.get_samples_played_in_range(start_sample / self.audio_rate_hz, end_sample / self.audio_rate_hz)
        x = s.data.float().numpy()
        want = end_sample - start_sample
        x = x[:, :want] if x.shape[1] >= want else np.pad(x, ((0, 0), (0, want - x.shape[1])))
        return resample_to(x, self.audio_rate_hz, rate_hz)

    def read_frames(self, i0: int, i1: int) -> np.ndarray:
        import torch
        b = self._vd.get_frames_in_range(i0, i1).data                  # uint8 [N,C,H,W]
        h, w = b.shape[-2:]
        if max(h, w) > self.max_side:
            s = self.max_side / max(h, w)
            b = torch.nn.functional.interpolate(b.float(), size=(round(h * s / 2) * 2, round(w * s / 2) * 2),
                                                mode="area").round().clamp(0, 255).to(torch.uint8)
        return b.permute(0, 2, 3, 1).contiguous().numpy()


class LoopedSource:
    """Repeat a short recording up to ``target_seconds`` (duration-matched workloads when no long footage exists).

    Content repeats exactly, so results from a looped source match workloads in DURATION, resolution and source
    count only; they cannot show content-dependent drift and must be labelled 'looped'. Reads stay bounded: every
    call maps its span onto at most a few reads of the base source."""

    def __init__(self, base, target_seconds: float):
        self.base = base
        self.audio_rate_hz = base.audio_rate_hz
        self.period = int(base.n_audio_samples)
        if self.period <= 0:
            raise ValueError("cannot loop an empty source")
        self.n_audio_samples = int(round(target_seconds * self.audio_rate_hz))
        self.av_map = base.av_map
        t0 = base.av_map.audio_start_us
        period_us = base.av_map.sample_to_common_us(self.period) - t0
        bp = [int(p) for p in base.video_pts_us() if int(p) - t0 < period_us]
        self._nb = len(bp)
        end_us = t0 + int(round(self.n_audio_samples * 1e6 / self.audio_rate_hz))
        pts, k = [], 0
        while True:
            chunk = [p + k * period_us for p in bp]
            pts += [p for p in chunk if p < end_us]
            if not chunk or chunk[-1] >= end_us:
                break
            k += 1
        self._pts = pts

    def video_pts_us(self) -> list[int]:
        return self._pts

    def read_audio(self, start_sample: int, end_sample: int, rate_hz: int) -> np.ndarray:
        parts, s = [], int(start_sample)
        end_sample = min(int(end_sample), self.n_audio_samples)
        while s < end_sample:
            off = s % self.period
            e = min(end_sample, s + (self.period - off))
            parts.append(self.base.read_audio(off, off + (e - s), rate_hz))
            s = e
        return np.concatenate(parts, axis=-1) if parts else np.zeros((1, 0), np.float32)

    def read_frames(self, i0: int, i1: int) -> np.ndarray:
        parts, i = [], int(i0)
        while i < i1:
            j = i % self._nb
            n = min(i1 - i, self._nb - j)
            parts.append(self.base.read_frames(j, j + n))
            i += n
        return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------------------- bookkeeping
@dataclass
class Counters:
    """Everything the acceptance gates assert on. Model calls are counted by wrapping the injected models."""

    calls: dict = field(default_factory=dict)
    chunks_computed: list = field(default_factory=list)       # (stage, chunk index) actually produced
    chunks_restored: list = field(default_factory=list)       # (stage, chunk index) served from cache
    max_frames_resident: int = 0
    max_queued_chunks: int = 0

    def bump(self, name: str, n: int = 1) -> None:
        self.calls[name] = self.calls.get(name, 0) + n


class _Counting:
    def __init__(self, inner, name: str, counters: Counters, method: str | None = None):
        self._inner, self._name, self._c, self._m = inner, name, counters, method

    def __call__(self, *a, **kw):
        self._c.bump(self._name)
        return (self._inner if self._m is None else getattr(self._inner, self._m))(*a, **kw)

    def __getattr__(self, item):                                  # score_track / extract_target pass-through
        attr = getattr(self._inner, item)
        if callable(attr) and item in ("score_track", "extract_target"):
            def wrapped(*a, **kw):
                self._c.bump(item)
                return attr(*a, **kw)
            return wrapped
        return attr


def count_models(models: SpeechModels, counters: Counters) -> SpeechModels:
    return SpeechModels(
        separator=_Counting(models.separator, "separator", counters),
        scorer=None if models.scorer is None else _Counting(models.scorer, "scorer", counters),
        tse=None if models.tse is None else _Counting(models.tse, "tse", counters),
        embed=_Counting(models.embed, "embed", counters), revisions=dict(models.revisions))


def _safe(sid: str, i: int) -> str:
    return f"{i:03d}_" + re.sub(r"[^A-Za-z0-9_.-]", "_", sid)


def _write_arrays(d: Path, arrays: dict[str, tuple[int, np.ndarray]], index_name: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    index = {}
    for i, (sid, (start, a)) in enumerate(sorted(arrays.items())):
        fn = _safe(sid, i) + ".npy"
        np.save(d / fn, np.ascontiguousarray(a, dtype=np.float32))
        index[sid] = {"start": int(start), "n": int(len(a)), "file": fn}
    (d / index_name).write_text(json.dumps(index, sort_keys=True), encoding="utf-8")


def read_arrays(entry: Path, sub: str, index_name: str) -> dict[str, tuple[int, np.ndarray]]:
    p = entry / sub / index_name
    if not p.is_file():
        return {}
    idx = json.loads(p.read_text(encoding="utf-8"))
    return {sid: (r["start"], np.load(entry / sub / r["file"])) for sid, r in idx.items()}


# --------------------------------------------------------------------------------------------- runner
@dataclass
class ChunkRecord:
    spec: ChunkSpec
    visual_key: str
    speech_key: str
    visual_dir: Path
    speech_dir: Path


@dataclass
class SpeechRunResult:
    records: list[ChunkRecord]
    state: PipelineState
    counters: Counters
    n_samples: int


class ChunkedSpeechRunner:
    """Drives the per-chunk visual+speech stages over a ``ChunkSource``. Models and the visual front end are
    injected, so the same code runs on CPU fakes (mocked gates) and on the real adapters."""

    def __init__(self, source, models: SpeechModels | None, visual: VisualFrontEnd | None, cache: StageCache,
                 cfg: AvConfig, run_dir: str | Path, *, media_id: str, identity_factory=None, yaw_provider=None,
                 counters: Counters | None = None, fault: FaultHook | None = None,
                 visual_revisions: dict | None = None, progress: Callable[[str, int, int], None] | None = None):
        self.src, self.cache, self.cfg = source, cache, cfg
        self.run_dir = Path(run_dir)
        self.media_id = media_id
        self.counters = counters or Counters()
        self.models = None if models is None else count_models(models, self.counters)
        self.visual = visual
        self.fault = fault or (lambda name, k: None)
        self.progress = progress or (lambda stage, i, n: None)
        self.visual_revisions = visual_revisions or {}
        self.yaw_provider = yaw_provider
        self._identity_factory = identity_factory
        self._pts = source.video_pts_us()
        self.n16 = int(source.n_audio_samples * SPEECH_SR // source.audio_rate_hz)
        self.specs = plan_chunks(self.n16, cfg.chunking.window_s, cfg.chunking.overlap_s)
        self.f = stitch_mod.crossfade_samples(cfg.chunking.crossfade_ms, SPEECH_SR)
        crossfade_check(self.f, cfg.chunking.window_s, cfg.chunking.overlap_s, SPEECH_SR)
        self.spill_dir = cache.root / "identity_spill"
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        import av_scene.associate as _assoc, av_scene.activity_gate as _gate, av_scene.peeling as _peel
        import av_scene.continuity_windows as _cont
        import av_scene.contracts as _contracts
        import separate.speaker_embed as _emb
        import who.matcher as _matcher
        self._fp = code_fingerprint(__import__(__name__, fromlist=["x"]), speech_mod, stitch_mod, _assoc, _gate,
                                    _peel, _cont, _contracts, _matcher, _emb)
        self._ingest_key = cache.key("ingest", inputs={"media": media_id},
                                     transform={"rate": source.audio_rate_hz, "av_map": repr(source.av_map)})

    # ---- per-pass model hand-over (pipeline loads each pass's models under a lifecycle lease)
    def set_models(self, models: SpeechModels) -> None:
        self.models = count_models(models, self.counters)

    def set_visual(self, visual: VisualFrontEnd, revisions: dict | None = None) -> None:
        self.visual = visual
        if revisions is not None:
            self.visual_revisions = revisions

    # ---- helpers
    def _master(self, s16: int) -> int:
        return (int(s16) * self.src.audio_rate_hz + SPEECH_SR // 2) // SPEECH_SR

    def _window_frames(self, spec: ChunkSpec) -> tuple[np.ndarray, list[int]]:
        t0 = self.src.av_map.sample_to_common_us(self._master(spec.win_start))
        t1 = self.src.av_map.sample_to_common_us(self._master(spec.win_end))
        i0, i1 = video_index_range(self._pts, t0, t1)
        if i1 - i0 > self.cfg.memory.max_frames_per_window if hasattr(self.cfg.memory, "max_frames_per_window") \
                else i1 - i0 > self.cfg.memory.max_decoded_frames:
            raise RuntimeError(f"chunk {spec.index}: {i1 - i0} frames exceed max_decoded_frames="
                               f"{self.cfg.memory.max_decoded_frames}")
        frames = self.src.read_frames(i0, i1) if i1 > i0 else np.zeros((0, 2, 2, 3), np.uint8)
        self.counters.max_frames_resident = max(self.counters.max_frames_resident, int(len(frames)))
        return frames, self._pts[i0:i1]

    def _window_audio(self, spec: ChunkSpec) -> np.ndarray:
        x = self.src.read_audio(self._master(spec.win_start), self._master(spec.win_end), SPEECH_SR)
        mono = x.mean(axis=0).astype(np.float32)
        n = spec.win_len
        return mono[:n] if len(mono) >= n else np.pad(mono, (0, n - len(mono)))

    def _vfp(self) -> str:
        """Visual stage fingerprint: the chunk machinery plus the actual front-end implementation in use."""
        import av_scene.contracts as _contracts
        import vision.chunk_tracker as _ct
        import vision.face_detect as _fd
        import vision.sam3_tracker as _s3
        # the real front end may sit behind a lazy proxy, so fingerprint its implementation modules explicitly
        objs = [__import__(__name__, fromlist=["x"]), _contracts, _ct, _fd, _s3]
        try:
            objs.append(type(self.visual))
        except TypeError:
            pass
        return code_fingerprint(*objs)

    def _speech_cfg(self) -> dict:
        s = self.cfg.switches
        yp, idf = self.yaw_provider, self._identity_factory
        return {"chunking": self.cfg.chunking.__dict__, "off_axis": self.cfg.off_axis.__dict__,
                "memory": {"max_active_identities": self.cfg.memory.max_active_identities},
                "yaw_provider": None if yp is None else getattr(yp, "__qualname__", repr(type(yp))),
                "identity_factory": None if idf is None else getattr(idf, "__qualname__", repr(type(idf))),
                "switches": {k: getattr(s, k) for k in ("use_tse", "activity_gate", "offscreen_upgrade", "peel",
                                                       "identity")}}

    # ---- stages
    def _visual_stage(self, spec: ChunkSpec, vstate: dict):
        key = self.cache.key("visual_chunk", inputs={"media": self.media_id, "chunk": str(spec.to_dict())},
                             config={"chunking": self.cfg.chunking.__dict__},
                             upstream={"ingest": self._ingest_key}, fingerprint=self._vfp(),
                             models=self.visual_revisions, state_in=hash_config(vstate))

        def produce(d: Path, meta: dict) -> None:
            self.fault("visual:mid", spec.index)
            frames, pts = self._window_frames(spec)
            vc = self.visual.step(frames, pts, vstate)
            (d / "visual.json").write_text(json.dumps({
                "pts_us": [int(p) for p in pts], "n_frames": int(len(frames)),
                "boxes": {t: [None if b is None else [v.item() if hasattr(v, "item") else v for v in b] for b in bs]
                          for t, bs in vc.boxes_by_track.items()}}, sort_keys=True), encoding="utf-8")
            (d / "state_out.json").write_text(json.dumps(vc.state, sort_keys=True), encoding="utf-8")
            masks = {t: [(_mask_to_dict(m)) for m in ms] for t, ms in vc.masks_by_track.items()}
            (d / "masks.json").write_text(json.dumps(masks, sort_keys=True), encoding="utf-8")
            self.counters.chunks_computed.append(("visual", spec.index))

        self.fault("visual:before", spec.index)
        entry, computed = self.cache.get_or_compute("visual_chunk", key, produce, required=("visual.json", "state_out.json"))
        if not computed:
            self.counters.chunks_restored.append(("visual", spec.index))
        self.fault("visual:after", spec.index)
        self.progress("visual", spec.index + 1, len(self.specs))
        return key, entry

    def _eligibility(self, track_id, spec: ChunkSpec):
        """Index ranges (into the window) eligible for enrollment, from the associated track's pose policy."""
        from who.matcher import FaceObservation, LearningEligibility
        o = self.cfg.off_axis
        el = LearningEligibility(o.unknown_pose, o.yaw_limit_deg)
        t0 = self.src.av_map.sample_to_common_us(self._master(spec.own_start))
        t1 = self.src.av_map.sample_to_common_us(self._master(spec.own_end))
        yaw = None if self.yaw_provider is None or track_id is None else self.yaw_provider(track_id, t0, t1)
        el.observe(0, FaceObservation(t0, t1, yaw))
        blocked, _ = el.blocked(0, t0, t1)
        if blocked:
            return [], "off_axis_or_pose_policy"
        return None, None

    def _speech_stage(self, spec: ChunkSpec, vkey: str, ventry: Path, state_in: PipelineState,
                      prev_entry: Path | None, prev_key: str | None = None):
        key = self.cache.key("speech_chunk", inputs={"media": self.media_id, "chunk": str(spec.to_dict())},
                             config=self._speech_cfg(),
                             upstream={"ingest": self._ingest_key, "visual_chunk": vkey,
                                       # the predecessor's tails feed this chunk's stitched head, so its identity (not just
                                       # the identity state) must re-key this chunk when it is recomputed differently
                                       "prev_speech": prev_key or "none"},
                             fingerprint=self._fp, models=self.models.revisions, state_in=state_in.hash())

        def produce(d: Path, meta: dict) -> None:
            self.fault("speech:mid", spec.index)
            self._produce_speech(d, meta, spec, ventry, state_in, prev_entry)
            self.counters.chunks_computed.append(("speech", spec.index))

        self.fault("speech:before", spec.index)
        entry, computed = self.cache.get_or_compute("speech_chunk", key, produce,
                                                    required=("meta.json", "state_out.json", "final/final.json"))
        if not computed:
            self.counters.chunks_restored.append(("speech", spec.index))
        self.fault("speech:after", spec.index)
        self.progress("speech", spec.index + 1, len(self.specs))
        return key, entry

    def _produce_speech(self, d: Path, meta: dict, spec: ChunkSpec, ventry: Path, state_in: PipelineState,
                        prev_entry: Path | None) -> None:
        sw = self.cfg.switches
        vis = json.loads((ventry / "visual.json").read_text(encoding="utf-8"))
        boxes = {t: [None if b is None else tuple(b) for b in bs] for t, bs in vis["boxes"].items()}
        frames, _ = self._window_frames(spec)
        audio = self._window_audio(spec)
        pts = vis["pts_us"]
        fps = (len(pts) - 1) / ((pts[-1] - pts[0]) / 1e6) if len(pts) > 1 and pts[-1] > pts[0] else float(FPS25)
        o0 = int(round((spec.own_start - spec.win_start) / SPEECH_SR * FPS25))
        o1 = int(round((spec.own_end - spec.win_start) / SPEECH_SR * FPS25))
        own_s = spec.own_len / SPEECH_SR
        res = speech_mod.run_speech_branch(
            audio, frames, fps, boxes, separator=self.models.separator, scorer=self.models.scorer,
            tse=self.models.tse if sw.use_tse else None, activity_gate=sw.activity_gate,
            offscreen_upgrade=sw.offscreen_upgrade, peel=sw.peel, own_frames=(o0, o1),
            min_active_s=min(speech_mod.MIN_ACTIVE_S, 0.5 * own_s))

        voice_ids = sorted(s for s in res.streams if s.startswith("voice_"))
        assoc_by_voice = {a.source_id: a for a in res.associations}
        state = PipelineState.from_dict(state_in.to_dict())
        state.next_chunk = spec.index + 1

        # ---- identity: slot -> persistent source id (permutation resolved here, before any crossfade)
        slots_status: dict[int, str] = {}
        slot_src: dict[int, str | None] = {}
        reasons: dict[int, str] = {}
        enrolled: dict[str, float] = {}
        new_ids: list[str] = []
        if sw.identity and voice_ids:
            from av_scene.continuity_windows import IdentityStore, SlotInput
            spill = self.spill_dir
            store = (IdentityStore.from_dict(state.identity, self.models.embed, spill) if state.identity
                     else (self._identity_factory() if self._identity_factory else
                           IdentityStore(self.models.embed, max_active=self.cfg.memory.max_active_identities,
                                         spill_dir=spill)))
            ins = {}
            for k, vid in enumerate(voice_ids):
                a = assoc_by_voice.get(vid)
                elig, _ = self._eligibility(a.track_id if a else None, spec)
                w = np.asarray(res.streams[vid], np.float32)
                w = w[: spec.win_len] if len(w) >= spec.win_len else np.pad(w, (0, spec.win_len - len(w)))
                ins[k] = SlotInput(audio=w, own=(spec.own_start - spec.win_start, spec.own_end - spec.win_start),
                                   eligible=elig)
            ca = store.assign_chunk(spec.index, ins)
            state.identity = store.to_dict()
            for k in range(len(voice_ids)):
                slots_status[k] = ca.status.get(k, "none")
                slot_src[k] = ca.mapping.get(k) if slots_status[k] in ("resolved", "new") else None
                reasons[k] = ca.reasons.get(k, "")
            enrolled, new_ids = ca.enrolled_s, list(ca.new_ids)
            for sid, sec in enrolled.items():
                state.enrolled_s[sid] = round(state.enrolled_s.get(sid, 0.0) + sec, 6)
            self.counters.bump("identity_chunks")
        else:   # identity switch off: baseline behaviour, slot order is NOT persistent across chunks
            for k, vid in enumerate(voice_ids):
                slot_src[k] = f"slot_{k}"
                slots_status[k] = "resolved"

        # ---- streams to stitch (window audio keyed by persistent id)
        streams: dict[str, np.ndarray] = {}
        for k, vid in enumerate(voice_ids):
            if slot_src[k] is not None:
                w = np.asarray(res.streams[vid], np.float32)
                streams[slot_src[k]] = w[: spec.win_len] if len(w) >= spec.win_len else np.pad(w, (0, spec.win_len - len(w)))
        resolved_win = dict(streams)
        for tid, x in res.tse.items():
            w = np.asarray(x, np.float32)
            streams[f"tse:{tid}"] = w[: spec.win_len] if len(w) >= spec.win_len else np.pad(w, (0, spec.win_len - len(w)))

        prev_tails = read_arrays(prev_entry, "tails", "tails.json") if prev_entry is not None else {}
        finals, tails, flags = stitch_mod.emit_chunk(spec, self.n16, self.f, streams, prev_tails)
        _write_arrays(d / "final", finals, "final.json")
        _write_arrays(d / "tails", tails, "tails.json")
        _write_arrays(d / "win", {sid: (spec.win_start, a) for sid, a in resolved_win.items()}, "win.json")

        # ---- timeline segments for the OWNED region (four independent status fields)
        us = lambda s: self.src.av_map.audio_start_us + sample_to_us(s, SPEECH_SR)           # noqa: E731
        segs: list[dict] = []

        def seg(sid, a0=None, a1=None, **kw):
            a0 = spec.own_start if a0 is None else a0
            a1 = spec.own_end if a1 is None else a1
            segs.append(TimelineSegment(source_id=sid, start_sample=a0, end_sample=a1,
                                        start_us=us(a0), end_us=us(a1),
                                        chunk_index=spec.index, **kw).model_dump(mode="json"))

        def pieces(tid):
            """Owned region split by the track's speaking activity -> [(a, b, active)]. A person is only claimed
            where they are actually speaking; the rest of the chunk is NOT attributed to them (sub-chunk
            resolution, otherwise one spoken word would highlight a whole 2 s region)."""
            out, pos = [], spec.own_start
            for lo, hi in sorted(res.activity_us.get(tid, [])):
                a = max(spec.own_start, spec.win_start + int(round(lo * SPEECH_SR / 1e6)))
                b = min(spec.own_end, spec.win_start + int(round(hi * SPEECH_SR / 1e6)))
                a = max(a, pos)                       # clip (not drop) an interval overlapping the previous one
                if b <= a:
                    continue
                if a > pos:
                    out.append((pos, a, False))
                out.append((a, b, True))
                pos = b
            if pos < spec.own_end:
                out.append((pos, spec.own_end, False))
            return out
        for k, vid in enumerate(voice_ids):
            a = assoc_by_voice.get(vid)
            st = slots_status[k]
            if st == "none" and reasons.get(k, "") == "insufficient_audio":
                continue                                           # silent slot: no claim, no segment
            if st == "none":                                       # audible but cannot be enrolled: say why, abstain
                why = ("enrollment_blocked: insufficient_eligible_audio" if reasons.get(k) == "insufficient_eligible_audio"
                       else f"identity_not_established: {reasons.get(k, '')} (too little audio to enroll a new speaker)")
                seg("unresolved", track_id=None, association_status=a.status if a else "unmatched",
                    visibility=a.visibility if a else "unknown", stem_availability="unavailable",
                    refusal_reason=why, identity_confidence="none", slot=k)
                continue
            if slot_src[k] is None:                                # ambiguous identity: abstain, never blend
                seg("unresolved", track_id=None, association_status="ambiguous", visibility=a.visibility if a else "unknown",
                    stem_availability="unavailable", refusal_reason=f"identity_ambiguous: {reasons.get(k, '')}".strip(": "),
                    identity_confidence="ambiguous", slot=k)
                continue
            conf = "new" if st == "new" else "resolved"
            if a is not None and a.status == "matched" and a.track_id:
                for p0, p1, on in pieces(a.track_id):
                    seg(slot_src[k], p0, p1, track_id=a.track_id if on else None,
                        association_status="matched" if on else "unmatched", visibility=a.visibility,
                        stem_availability="ok", refusal_reason=None, identity_confidence=conf, slot=k)
            else:
                seg(slot_src[k], track_id=a.track_id if a else None, association_status=a.status if a else "unmatched",
                    visibility=a.visibility if a else "unknown", stem_availability="ok",
                    refusal_reason=None, identity_confidence=conf, slot=k)
        for tid, (stream_id, ext) in res.selection.items():
            vis_cov = res.info.get("tracks", {}).get(tid, {}).get("face_coverage", 0.0)
            visible = "visible" if vis_cov > 0 else "unknown"
            if ext.status in ("ok", "degraded") and f"tse:{tid}" in streams:
                for p0, p1, on in pieces(tid):
                    if on:
                        seg(f"tse:{tid}", p0, p1, track_id=tid, association_status="matched", visibility=visible,
                            stem_availability=ext.status, refusal_reason=ext.reason if ext.status == "degraded" else None)
                    else:
                        seg(f"tse:{tid}", p0, p1, track_id=tid, association_status="unmatched", visibility=visible,
                            stem_availability="unavailable", refusal_reason="target inactive in this part of the chunk")
            else:
                amb = ext.reason.startswith("ambiguous")
                seg(f"tse:{tid}", track_id=tid, association_status="ambiguous" if amb else "unmatched",
                    visibility=visible, stem_availability="unavailable" if ext.status == "unavailable" else ext.status,
                    refusal_reason=ext.reason or "no stem")
        r0, r1 = stitch_mod.final_region(spec, self.n16, self.f)
        out_state = state
        out_state.emitted_samples = r1
        out_state.pending = {"from_chunk": spec.index, "sources": sorted(tails)}
        out_state.write(d / "state_out.json")
        meta.update({
            "chunk": spec.to_dict(), "slots": {str(k): {"source": slot_src[k], "status": slots_status[k],
                                                          "reason": reasons.get(k, "")} for k in slots_status},
            "segments": segs, "boundary_flags": flags, "enrolled_s": enrolled,
            "kinds": {**{s: {"kind": "speech", "role": "default"} for s in resolved_win},
                      **{f"tse:{t}": {"kind": "tse", "role": "diagnostic"} for t in res.tse}},
            "info": {k: v for k, v in res.info.items() if k in ("n_suspected_talkers", "activity_source", "tse_similarity_db")},
            "final_region": [r0, r1], "new_ids": new_ids,
        })
        (d / "meta.json").write_text(json.dumps(meta, sort_keys=True, default=_json_default), encoding="utf-8")

    # ---- driver: two passes so each pass loads its own models once (the GPU lease is exclusive)
    def run_visual(self) -> list[tuple[str, Path]]:
        """Pass A: persistent visual tracks for every chunk (SAM 3 + S3FD live here)."""
        out: list[tuple[str, Path]] = []
        vstate: dict = self.visual_empty_state()
        for spec in self.specs:
            vkey, ventry = self._visual_stage(spec, vstate)
            vstate = json.loads((ventry / "state_out.json").read_text(encoding="utf-8"))
            out.append((vkey, ventry))
        return out

    def run_speech(self, visual: list[tuple[str, Path]]) -> SpeechRunResult:
        """Pass B: TalkNet / MossFormer2 / AV-TSE / identity / stitching for every chunk."""
        records: list[ChunkRecord] = []
        state = PipelineState()
        prev_entry: Path | None = None
        prev_key: str | None = None
        for spec, (vkey, ventry) in zip(self.specs, visual):
            skey, sentry = self._speech_stage(spec, vkey, ventry, state, prev_entry, prev_key)
            state = PipelineState.read(sentry / "state_out.json")
            records.append(ChunkRecord(spec, vkey, skey, ventry, sentry))
            prev_entry, prev_key = sentry, skey
        return SpeechRunResult(records, state, self.counters, self.n16)

    def run(self) -> SpeechRunResult:
        return self.run_speech(self.run_visual())

    def visual_empty_state(self) -> dict:
        f = getattr(self.visual, "empty_state", None)
        return f() if callable(f) else {}


def _mask_to_dict(m):
    return None if m is None else m.to_dict()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)
