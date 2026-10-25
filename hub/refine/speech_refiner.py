"""Phase 5 speech refiner: does the voice the user selected belong to a visible face?

The hub's live separator is NOT re-run: its slot audio is handed to ``run_speech_branch`` as the "separated
voices" (voice_0 = chosen, voice_1 = the other voice slot). TalkNet + AV-TSE then decide whether a face's
lip-conditioned extraction matches voice_0. Evidence only; nothing here changes routing. Face tracks come from
S3FD alone (``faces.FaceBoxTracker``, new and unproven on real footage). Importing this module needs no torch.
"""

from __future__ import annotations

import time

import numpy as np

from .contracts import RefineConfig, RefineJob, RefineOutcome, RefineState
from .faces import detect_tracks

VOICE_SLOTS = ("s1", "s2")


def _unavail(reason, ms, ev=None):
    return RefineOutcome(RefineState.unavailable, reason, compute_ms=ms, evidence=ev or {})


def decode_frames(frames) -> np.ndarray:
    """JPEG bytes -> uint8 RGB [N,H,W,3], resized to the first frame's size."""
    import cv2
    out, size = [], None
    for _, jpg in frames:
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("undecodable JPEG frame")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if size is None:
            size = img.shape[:2]
        elif img.shape[:2] != size:
            img = cv2.resize(img, (size[1], size[0]))
        out.append(img)
    return np.stack(out)


class SpeechRefiner:
    def __init__(self, detector, scorer, tse, config: RefineConfig | None = None):
        self.detector, self.scorer, self.tse = detector, scorer, tse
        self.config = config or RefineConfig()

    def __call__(self, job: RefineJob) -> RefineOutcome:
        cfg, ms = self.config, {}
        ev: dict = {"n_frames": len(job.frames), "window_s": cfg.window_s,
                    "min_video_fps": cfg.min_video_fps, "min_face_coverage": cfg.min_face_coverage}
        if len(job.frames) < 2:
            return _unavail("insufficient video (no frames)", ms, ev)
        span = (job.frames[-1][0] - job.frames[0][0]) / 1e6
        fps = (len(job.frames) - 1) / span if span > 0 else 0.0
        ev["fps"] = round(fps, 2)
        if fps < cfg.min_video_fps:
            return _unavail(f"insufficient video ({fps:.1f} fps)", ms, ev)

        chosen = [s for s in job.slots if s in VOICE_SLOTS]
        if not job.slots or len(chosen) != len(job.slots) or any(s not in job.slot_audio for s in chosen):
            return _unavail("selected source is not a voice slot", ms, ev)
        n = min(len(job.slot_audio[s]) for s in chosen)
        v0 = np.sum([np.asarray(job.slot_audio[s][:n], np.float32) for s in chosen], axis=0, dtype=np.float32)
        others = [s for s in VOICE_SLOTS if s not in chosen and s in job.slot_audio]
        v1 = (np.asarray(job.slot_audio[others[0]][:n], np.float32).copy() if others
              else np.zeros(n, np.float32))
        voices = np.stack([v0, v1])
        audio = np.asarray(job.mix_16k[:n], np.float32).copy()

        t = time.perf_counter()
        rgb = decode_frames(job.frames)
        ms["decode"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        tracks = detect_tracks(self.detector, rgb)
        ms["faces"] = (time.perf_counter() - t) * 1e3
        cov = {k: float(np.mean([b is not None for b in v])) for k, v in tracks.items()}
        best = max(cov.values()) if cov else 0.0
        ev.update(n_tracks=len(tracks), face_coverage=round(best, 3))
        if not tracks or best < cfg.min_face_coverage:
            return RefineOutcome(RefineState.refused, "no visible face for this voice", compute_ms=ms, evidence=ev)

        from av_scene.speech import run_speech_branch
        t = time.perf_counter()
        res = run_speech_branch(audio, rgb, fps, tracks, separator=lambda _x: voices,
                                scorer=self.scorer, tse=self.tse,
                                activity_gate=False, offscreen_upgrade=False, peel=False)
        ms["speech"] = (time.perf_counter() - t) * 1e3

        assoc = next((a for a in res.associations if a.source_id == "voice_0"), None)
        if assoc is not None and assoc.status == "matched" and assoc.track_id:
            comp = {c.name: c.value for c in assoc.components}
            score = comp.get("tse_stream_si_sdr_db", assoc.score)
            conf = float(np.clip((score or 0.0) / 20.0, 0.1, 1.0))
            return RefineOutcome(RefineState.accepted, f"voice matches visible face {assoc.track_id}",
                                 track_id=assoc.track_id, score=score, margin_db=assoc.runner_up_margin,
                                 assoc_conf=conf, refined_audio=res.streams.get(f"tse:{assoc.track_id}"),
                                 compute_ms=ms, evidence=ev)
        sel = [e.reason for _, e in res.selection.values()]
        if sel and all(r == "target inactive" for r in sel):
            reason = "no face is speaking while this voice is active"
        else:
            note = next((c.note for c in assoc.components if c.note), "") if assoc else ""
            reason = next((r for r in sel if r and r != "target inactive"), "") or note \
                or (assoc.status if assoc else "voice not analysed")
            reason = f"voice not matched to a face: {reason}"
        ev["association"] = assoc.status if assoc else None
        return RefineOutcome(RefineState.refused, reason, compute_ms=ms, evidence=ev)


def load_speech_refiner(device: str = "cuda", config: RefineConfig | None = None) -> SpeechRefiner:
    import av_scene  # noqa: F401  (registers FFmpeg DLLs before torch-side imports)
    from separate.av_tse import AVTargetSpeakerExtractor
    from vision.face_detect import FaceDetector
    from vision.talknet import TalkNetScorer
    return SpeechRefiner(FaceDetector(device), TalkNetScorer(device=device),
                         AVTargetSpeakerExtractor(device=device), config)
