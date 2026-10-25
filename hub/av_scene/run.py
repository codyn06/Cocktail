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
    # Static revisions (and the cheap empty visual state) let cache keys be computed WITHOUT loading any model.
    # When given, a pass opens its provider lazily, only if a stage actually has to compute; a fully warm cache
    # therefore loads no checkpoint at all. When None, the pass opens its provider eagerly (legacy behaviour).
    visual_revisions: dict | None = None
    speech_revisions: dict | None = None
    event_revisions: dict | None = None
    visual_empty_state: Callable[[], dict] | None = None
    # Event DETECTION (AST) is cheap and needs none of the heavy event models. When given, it is built lazily on its own
    # (outside the GPU lease), so SAM 3 / SAM Audio / judge load only if an event window actually has to compute.
    event_classifier: Callable[[], Callable] | None = None


class LazyPass:
    """Enter a provider's context on first use, exit it exactly once. ``opened`` counts how often models loaded."""

    def __init__(self, provider):
        self._provider, self._cm, self._obj, self.opened = provider, None, None, 0

    def get(self):
        if self._cm is None:
            self._cm = self._provider()
            self._obj = self._cm.__enter__()
            self.opened += 1
        return self._obj

    def close(self, *exc) -> None:
        cm, self._cm, self._obj = self._cm, None, None
        if cm is not None:
            cm.__exit__(*(exc or (None, None, None)))


class _LazyVisual:
    def __init__(self, lp: LazyPass, empty_state):
        self._lp, self._empty = lp, empty_state

    def empty_state(self) -> dict:
        return self._empty() if self._empty else self._lp.get()[0].empty_state()

    def step(self, frames, pts_us, state):
        return self._lp.get()[0].step(frames, pts_us, state)


class _LazyCall:
    def __init__(self, lp: LazyPass, getter, what: str):
        self._lp, self._get, self._what = lp, getter, what

    def __call__(self, *a, **kw):
        fn = self._get(self._lp.get())
        if fn is None:
            raise RuntimeError(f"{self._what} is not provided by this run's model set")
        return fn(*a, **kw)


class _LazyAttr:
    """Proxy for an object with named methods (TalkNet scorer, AV-TSE extractor)."""

    def __init__(self, lp: LazyPass, getter):
        self._lp, self._get = lp, getter

    def __getattr__(self, name):
        return lambda *a, **kw: getattr(self._get(self._lp.get()), name)(*a, **kw)


def _lazy_speech(lp: LazyPass, revisions: dict, use_tse: bool) -> SpeechModels:
    return SpeechModels(
        separator=_LazyCall(lp, lambda m: m.separator, "separator"),
        scorer=_LazyAttr(lp, lambda m: m.scorer),
        tse=_LazyAttr(lp, lambda m: m.tse) if use_tse else None,
        embed=_LazyCall(lp, lambda m: m.embed, "speaker embedder"), revisions=revisions)


class _LazyClassifier:
    def __init__(self, factory):
        self._f, self._c = factory, None

    def __call__(self, windows):
        if self._c is None:
            self._c = self._f()
        return self._c(windows)


def _lazy_events(lp: LazyPass, classifier_factory=None):
    from .event_scope import EventAdapters
    ad = EventAdapters(
        sam_audio=_LazyCall(lp, lambda o: o[0].sam_audio, "sam_audio adapter"),
        judge=_LazyCall(lp, lambda o: o[0].judge, "judge adapter"),
        clap=_LazyCall(lp, lambda o: o[0].clap, "clap adapter"),
        tracker=_LazyCall(lp, lambda o: o[0].tracker, "noun tracker"))
    return ad, (_LazyClassifier(classifier_factory) if classifier_factory else _LazyCall(lp, lambda o: o[1], "event classifier"))


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


def identity_activity(metas: list[dict], audio_start_us: int = 0) -> dict[str, list[tuple[int, int]]]:
    """Per-source activity in CLIP time (us from audio sample 0). Timeline segments carry common-clock times, so
    ``audio_start_us`` is removed here; diarization windows are also clip-time and the two must agree."""
    from separate.diarization import normalize
    acts: dict[str, list] = {}
    for m in metas:
        for s in m["segments"]:
            if s["source_id"].startswith("src_") and s["stem_availability"] in ("ok", "degraded"):
                acts.setdefault(s["source_id"], []).append((s["start_us"] - audio_start_us,
                                                            s["end_us"] - audio_start_us))
    return {k: normalize(v) for k, v in acts.items()}


def run_pipeline(source, providers: Providers, cfg: AvConfig, out_dir: str | Path, *, media_id: str, run_id: str,
                 cache_dir: str | Path | None = None, fault=None, evidence_tier: str = "replayed",
                 evidence_note: str = "", code_revision: str | None = None, hardware: dict | None = None,
                 identity_factory=None, yaw_provider=None, progress=None) -> PipelineResult:
    from . import event_stage, export
    from separate.diarization import compute_anchors

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = StageCache(cache_dir or out / "_cache")
    counters = Counters()
    runner = ChunkedSpeechRunner(source, None, None, cache, cfg, out / "_work", media_id=media_id, counters=counters,
                                 fault=fault, identity_factory=identity_factory, yaw_provider=yaw_provider,
                                 progress=progress)
    stages: list[StageRecord] = []
    revisions: dict = {}

    def timed(name: str, fn):
        t0 = time.perf_counter()
        r = fn()
        stages.append(StageRecord(name=name, key=media_id[:16], seconds=round(time.perf_counter() - t0, 2)))
        return r

    opened: dict[str, int] = {}
    with MemorySampler() as mem:
        vp = LazyPass(providers.visual)
        try:
            if providers.visual_revisions is None:                    # eager (legacy): models load up front
                vis, vrev = vp.get()
            else:
                vis, vrev = _LazyVisual(vp, providers.visual_empty_state), providers.visual_revisions
            runner.set_visual(vis, vrev)
            revisions.update({f"visual:{k}": v for k, v in vrev.items()})
            vrecs = timed("visual", runner.run_visual)
            runner.visual = None
            del vis
        finally:
            opened["visual"] = vp.opened
            vp.close()
        sp = LazyPass(providers.speech)
        try:
            if providers.speech_revisions is None:
                models = sp.get()
            else:
                models = _lazy_speech(sp, providers.speech_revisions, cfg.switches.use_tse)
            runner.set_models(models)
            revisions.update(models.revisions)
            speech = timed("speech", lambda: runner.run_speech(vrecs))
            runner.models = None
            del models
        finally:
            opened["speech"] = sp.opened
            sp.close()
        metas = [json.loads((rec.speech_dir / "meta.json").read_text(encoding="utf-8")) for rec in speech.records]

        # ---- anchors from identity activity; diarization is OPTIONAL evidence with an explicit status
        activity = identity_activity(metas, source.av_map.audio_start_us)
        diar = {"status": "skipped", "reason": "switches.diarization=off"}
        if cfg.switches.diarization != "off":
            from separate.diarization import compute_anchors_with_diarization

            from . import diar_windows
            av = diar_windows.availability()
            pipe, load_error = None, None
            if providers.diarization and av["status"] == "ok":
                try:
                    pipe = providers.diarization()
                except Exception as e:      # access/terms/offline at LOAD time is a runtime failure, not a skip
                    load_error = f"{type(e).__name__}: {str(e)[:160]}"
            if load_error:
                diar = {"status": "failed", "reason": f"pipeline load failed: {load_error}"}
                anchors = compute_anchors(activity, (), cfg.anchors) if activity else {}
            else:
                res = timed("diarization", lambda: diar_windows.run_windows(
                    lambda a, b: source.read_audio(runner._master(a), runner._master(b), SPEECH_SR).mean(axis=0),
                    speech.n_samples, activity, pipeline=pipe, mode=cfg.switches.diarization,
                    not_attempted_reason=f"package={av['package']} cache={av['cache']} "
                                         f"gated_access={av['gated_access']}"))
                diar = res.to_dict() if hasattr(res, "to_dict") else {"status": getattr(res, "status", "failed")}
                anchors = compute_anchors_with_diarization(activity, (), cfg.anchors, res)
        else:
            anchors = compute_anchors(activity, (), cfg.anchors) if activity else {}
        speakers = [SpeakerTrack(speaker_id=k, turns_us=[tuple(i) for i in activity.get(k, [])], anchors=a.to_dict())
                    for k, a in sorted(anchors.items())]

        events = None
        if cfg.switches.events and providers.events is not None:
            ep = LazyPass(providers.events)
            try:
                if providers.event_revisions is None:
                    adapters, classifier, erev = ep.get()
                else:
                    (adapters, classifier), erev = _lazy_events(ep, providers.event_classifier), providers.event_revisions
                revisions.update({f"events:{k}": v for k, v in erev.items()})
                stage = event_stage.EventStage(runner, cfg, adapters, classifier=classifier, revisions=erev)
                events = timed("events", lambda: stage.run(speech))
                del stage, adapters, classifier
            finally:
                opened["events"] = ep.opened
                ep.close()
        residual = []
        if cfg.switches.residual:
            residual = timed("residual", lambda: ResidualStage(runner, cfg).run(speech, events.refs if events else []))
        report = {
            "resume": {"computed": sorted({f"{s}:{i}" for s, i in counters.chunks_computed}),
                       "restored": len(counters.chunks_restored), "model_calls": dict(counters.calls)},
            "max_frames_resident": counters.max_frames_resident,
            "model_passes_opened": dict(opened),            # 0 for a pass means no checkpoint was loaded at all
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
