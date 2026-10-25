"""Orchestration for `python -m av_scene run`: passes, per-pass model providers, memory sampling, bundle export.

Passes (each loads only its own models, under the GPU lease in production):
  visual (SAM 3 + S3FD) -> speech (TalkNet, MossFormer2, AV-TSE, speaker embedder) -> [diarization, optional]
  -> events (SAM 3 nouns, SAM Audio, judge/CLAP) -> residual -> export.
Every pass is chunk/window cached, so a rerun after interruption restores finished work without models.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager

from replay.chunked import ChunkedSpeechRunner, Counters, SpeechRunResult

from .cache import StageCache
from .config import AvConfig
from .contracts import SPEECH_SR, SpeechModels, VisualFrontEnd
from .pipeline import ResidualStage
from .schemas import SpeakerTrack, StageRecord


@dataclass
class Providers:
    """Per-pass model providers: each returns a context manager that loads that pass's models, yields them, and
    releases them (production: ModelLifecycleManager under the GPU lease; tests: ``nullcontext(fake)``)."""

    visual: Callable[[], ContextManager[tuple[VisualFrontEnd, dict]]]       # -> (front end, model revisions)
    speech: Callable[[], ContextManager[SpeechModels]]
    events: Callable[[], ContextManager[tuple]] | None = None               # -> (EventAdapters, classifier, revisions)
    diarization: Callable[[], object | None] | None = None                  # -> pyannote pipeline or None


class MemorySampler:
    """Whole-process peak RSS (this process + children) and CUDA allocation, sampled off-thread."""

    def __init__(self, interval_s: float = 0.25):
        self.interval, self.series, self.peak_rss_mib, self.peak_gpu_mib = interval_s, [], 0.0, None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t0 = 0.0

    def _rss(self) -> float | None:
        try:
            import psutil
            p = psutil.Process()
            return (p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))) / 2**20
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def sample(self) -> None:
        rss = self._rss()
        if rss is not None:
            self.peak_rss_mib = max(self.peak_rss_mib, rss)
            self.series.append((round(time.perf_counter() - self._t0, 2), round(rss, 1)))
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_gpu_mib = max(self.peak_gpu_mib or 0.0, torch.cuda.max_memory_allocated() / 2**20)
        except Exception:
            pass

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.sample()
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(2)
        self.sample()


@dataclass
class PipelineResult:
    manifest: object
    out_dir: Path
    counters: Counters
    report: dict
    speech: SpeechRunResult


def identity_activity(metas: list[dict]) -> dict[str, list[tuple[int, int]]]:
    from separate.diarization import normalize
    acts: dict[str, list] = {}
    for m in metas:
        for s in m["segments"]:
            if s["source_id"].startswith("src_") and s["stem_availability"] in ("ok", "degraded"):
                acts.setdefault(s["source_id"], []).append((s["start_us"], s["end_us"]))
    return {k: normalize(v) for k, v in acts.items()}


def run_pipeline(source, providers: Providers, cfg: AvConfig, out_dir: str | Path, *, media_id: str, run_id: str,
                 cache_dir: str | Path | None = None, fault=None, evidence_tier: str = "replayed",
                 evidence_note: str = "", code_revision: str | None = None, hardware: dict | None = None,
                 identity_factory=None, yaw_provider=None) -> PipelineResult:
    from . import event_stage, export
    from separate.diarization import compute_anchors

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = StageCache(cache_dir or out / "_cache")
    counters = Counters()
    runner = ChunkedSpeechRunner(source, None, None, cache, cfg, out / "_work", media_id=media_id, counters=counters,
                                 fault=fault, identity_factory=identity_factory, yaw_provider=yaw_provider)
    stages: list[StageRecord] = []
    revisions: dict = {}

    def timed(name: str, fn):
        t0 = time.perf_counter()
        r = fn()
        stages.append(StageRecord(name=name, key=media_id[:16], seconds=round(time.perf_counter() - t0, 2)))
        return r

    with MemorySampler() as mem:
        with providers.visual() as (vis, vrev):
            runner.set_visual(vis, vrev)
            revisions.update({f"visual:{k}": v for k, v in vrev.items()})
            vrecs = timed("visual", runner.run_visual)
        with providers.speech() as models:
            runner.set_models(models)
            revisions.update(models.revisions)
            speech = timed("speech", lambda: runner.run_speech(vrecs))
        metas = [json.loads((rec.speech_dir / "meta.json").read_text(encoding="utf-8")) for rec in speech.records]

        # ---- anchors from identity activity; diarization is OPTIONAL evidence with an explicit status
        activity = identity_activity(metas)
        diar = {"status": "skipped", "reason": "switches.diarization=off"}
        if cfg.switches.diarization != "off":
            from separate.diarization import compute_anchors_with_diarization

            from . import diar_windows
            av = diar_windows.availability()
            pipe = providers.diarization() if (providers.diarization and av["status"] == "ok") else None
            res = timed("diarization", lambda: diar_windows.run_windows(
                lambda a, b: source.read_audio(runner._master(a), runner._master(b), SPEECH_SR).mean(axis=0),
                speech.n_samples, activity, pipeline=pipe, mode=cfg.switches.diarization,
                not_attempted_reason=f"package={av['package']} cache={av['cache']} gated_access={av['gated_access']}"))
            diar = res.to_dict() if hasattr(res, "to_dict") else {"status": getattr(res, "status", "failed")}
            anchors = compute_anchors_with_diarization(activity, (), cfg.anchors, res)
        else:
            anchors = compute_anchors(activity, (), cfg.anchors) if activity else {}
        speakers = [SpeakerTrack(speaker_id=k, turns_us=[tuple(i) for i in activity.get(k, [])], anchors=a.to_dict())
                    for k, a in sorted(anchors.items())]

        events = None
        if cfg.switches.events and providers.events is not None:
            with providers.events() as (adapters, classifier, erev):
                revisions.update({f"events:{k}": v for k, v in erev.items()})
                stage = event_stage.EventStage(runner, cfg, adapters, classifier=classifier, revisions=erev)
                events = timed("events", lambda: stage.run(speech))
        residual = []
        if cfg.switches.residual:
            residual = timed("residual", lambda: ResidualStage(runner, cfg).run(speech, events.refs if events else []))
        report = {
            "resume": {"computed": sorted({f"{s}:{i}" for s, i in counters.chunks_computed}),
                       "restored": len(counters.chunks_restored), "model_calls": dict(counters.calls)},
            "max_frames_resident": counters.max_frames_resident,
        }
        manifest = timed("export", lambda: export.export_bundle(
            out, runner=runner, speech=speech, events=events, residual=residual, cfg=cfg, run_id=run_id,
            config_hash=cfg.hash(), evidence_tier=evidence_tier, evidence_note=evidence_note, stages=stages,
            model_revisions=revisions, report=report, diarization=diar, code_revision=code_revision,
            hardware=hardware, speakers=speakers))
    report["memory"] = {"peak_rss_mib": round(mem.peak_rss_mib, 1), "peak_gpu_mib": mem.peak_gpu_mib,
                        "rss_series": mem.series[-2000:]}
    (out / "run_report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return PipelineResult(manifest, out, counters, report, speech)
