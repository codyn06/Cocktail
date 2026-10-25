"""Production model providers for `python -m av_scene run` (loads real checkpoints; GPU).

UNTESTED unless a run report says otherwise: this module is exercised by the opt-in model runs, never by the CPU
test suite (which uses fakes). Each pass loads only its own models under ``ModelLifecycleManager`` (one GPU lease
at a time, released and measured on exit), so peak VRAM is the largest pass, not the sum:

  visual   SAM 3 image model + S3FD                       (persistent person tracks, head boxes)
  speech   MossFormer2_SS, TalkNet, AV-MossFormer2 TSE, WavLM speaker-verification embedder
  events   SAM 3 (noun tracking) + SAM Audio small-tv core + AST classifier (+ judge/CLAP ranker on CPU)

Model revisions recorded in cache keys are package versions and repo ids; checkpoint file hashes are not
computed (large files), which is a known limitation recorded in the run report.
"""

from __future__ import annotations

import contextlib
import importlib.metadata as md
from pathlib import Path

import numpy as np

from .config import AvConfig
from .contracts import SpeechModels
from .event_scope import EventAdapters
from .run import Providers
from .runtime import ModelLifecycleManager


def _ver(*dists: str) -> str:
    out = []
    for d in dists:
        try:
            out.append(f"{d}=={md.version(d)}")
        except md.PackageNotFoundError:
            out.append(f"{d}==?")
    return ",".join(out)


def real_providers(cfg: AvConfig, *, device: str = "cuda", run_tag: str = "run", reports: list | None = None,
                   vram_fraction: float | None = None) -> Providers:
    reports = reports if reports is not None else []

    @contextlib.contextmanager
    def visual():
        from vision.chunk_tracker import ChunkTracker
        from vision.face_detect import FaceDetector, face_boxes_by_track
        from vision.sam3_tracker import load_sam3_image

        def load():
            return load_sam3_image("person", device=device), FaceDetector(device)

        with ModelLifecycleManager("visual_sam3_s3fd", load, device=device, reports=reports,
                                   vram_fraction=vram_fraction) as (sam3, det):
            front = ChunkTracker(sam3, lambda frames, masks: face_boxes_by_track(det, frames, masks),
                                 concept="person", run_tag=run_tag)
            yield front, {"sam3": _ver("sam3"), "s3fd": "face_detect.FaceDetector", "concept": "person"}
            del front, sam3, det          # nothing may still reference the models when the lease is released

    @contextlib.contextmanager
    def speech():
        from separate.models import VoiceSeparator
        from separate.speaker_embed import load_wavlm_sv
        from vision.talknet import TalkNetScorer

        def load():
            tse = None
            if cfg.switches.use_tse:
                from separate.av_tse import AVTargetSpeakerExtractor
                tse = AVTargetSpeakerExtractor(device=device)
            return VoiceSeparator(device), TalkNetScorer(device=device), tse, load_wavlm_sv(device)

        with ModelLifecycleManager("speech_mossformer2_talknet_tse", load, device=device, reports=reports,
                                   vram_fraction=vram_fraction) as (sep, scorer, tse, embed):
            yield SpeechModels(sep, scorer, tse, embed, {
                "mossformer2": "MossFormer2_SS_16K", "talknet": "talknet", "av_tse": "AV_MossFormer2_TSE" if tse else "off",
                "wavlm_sv": "microsoft/wavlm-base-plus-sv", "clearvoice": _ver("clearvoice"), "pkgs": _ver("torch")})
            del sep, scorer, tse, embed

    @contextlib.contextmanager
    def events():
        from replay import sam_audio as sa
        from replay.judge import JudgeRanker
        from vision.sam3_tracker import MaskIouTracker, dedupe, load_sam3_image

        from .events import ast_classifier

        def load():
            return load_sam3_image("person", device=device), sa.load_sam_audio(device=device), ast_classifier(device)

        with ModelLifecycleManager("events_sam3_samaudio_ast", load, device=device, reports=reports,
                                   vram_fraction=vram_fraction) as (sam3, bundle, clf):
            judge = JudgeRanker(backend="judge", device="cpu")
            clap = JudgeRanker(backend="clap", device="cpu")

            def sam_audio(audio48, prompt, seeds):
                cs = sa.extract_candidates(bundle, audio=np.asarray(audio48, np.float32), sample_rate=48000,
                                           frames=None, masks=None, text=prompt, anchors="full_clip",
                                           n_candidates=len(seeds), seeds=list(seeds))
                return cs.targets

            def tracker(frames, concept):
                sam3.concept = concept
                mt, out = MaskIouTracker(), {}
                for i in range(len(frames)):
                    for local, m, _conf in mt.update(i, dedupe(sam3(frames[i]))):
                        out.setdefault(str(local), [None] * len(frames))[i] = m
                return out

            ad = EventAdapters(sam_audio=sam_audio, judge=lambda c, p, sr, mix=None: judge.rank(c, p, sr, mix),
                               clap=lambda c, p, sr, mix=None: clap.rank(c, p, sr, mix), tracker=tracker)
            yield ad, clf, {"sam_audio": bundle.revision if hasattr(bundle, "revision") else "small-tv",
                            "sam3": _ver("sam3"), "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
                            "judge": "facebook/sam-audio-judge", "clap": "laion_clap"}
            del ad, judge, clap, sam_audio, tracker, sam3, bundle, clf

    def diar():
        from separate.pyannote_diar import load_pipeline
        return load_pipeline()          # raises on gated/offline/bad token: run_pipeline records that as `failed`

    return Providers(visual=visual, speech=speech, events=events, diarization=diar)
