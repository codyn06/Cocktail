"""Event-scoped non-speech pass: streaming AST -> bounded event windows -> per-window SAM 3 / SAM Audio / judge.

Each event window is its own cache entry (``event_window``). Memory is bounded by the window, never the clip:
  * AST runs over lazily read blocks (`iter_detect_events`), the event list is a small JSON entry;
  * a window is at most ``memory.event_max_window_s`` long; a sustained event is split into contiguous windows
    whose entries chain (``state_in`` = previous window's key) so an interruption inside a sustained event
    reruns only the unfinished windows;
  * frames are subsampled so a window never decodes more than ``memory.max_decoded_frames``;
  * candidates are generated and ranked in batches by `run_event_window`; only the chosen stem is kept.

Attribution evidence that is not available stays explicit: no calibrated array -> no spatial evidence; no
event-free span -> no silent control. Neither can produce "matched" (see `nonspeech_attr.attribute_nonspeech`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from replay.chunked import ChunkedSpeechRunner, SpeechRunResult, read_arrays, resample_to
from .cache import code_fingerprint
from .config import AvConfig
from . import event_scope as es, events as ev
from .contracts import SPEECH_SR, sample_to_us
from .pipeline import EventRef
from .schemas import AcousticEvent

SAM_RATE = 48000


def pick_control_span(windows: list, length_s: float, duration_s: float, guard_s: float = 0.5):
    """First span of ``length_s`` that overlaps no detected event window (padded by ``guard_s``); None when the
    clip has no such span, in which case the control stays MISSING (never assumed)."""
    t = 0.0
    for lo, hi in sorted((float(w[0]), float(w[1])) for w in windows):
        if lo - guard_s - t >= length_s:
            return (t, t + length_s)
        t = max(t, hi + guard_s)
    return (t, t + length_s) if duration_s - t >= length_s else None


def assemble_range(records, sid: str, a16: int, b16: int, margin: int = 1600) -> np.ndarray:
    """Final (stitched) audio of one speech source over [a16, b16) at 16 kHz, zeros where it has no segment.
    Reads at most the chunk entries overlapping the span."""
    out = np.zeros(b16 - a16, np.float32)
    for rec in records:
        r0, r1 = rec.spec.own_start - margin, rec.spec.own_end       # finals may start up to half a crossfade early
        if r1 <= a16 or r0 >= b16:
            continue
        got = read_arrays(rec.speech_dir, "final", "final.json").get(sid)
        if got is None:
            continue
        s, arr = got
        lo, hi = max(s, a16), min(s + len(arr), b16)
        if hi > lo:
            out[lo - a16:hi - a16] = arr[lo - s:hi - s]
    return out


@dataclass
class EventRunResult:
    refs: list[EventRef]
    windows: list
    detect_key: str
    timeline: list[dict]                    # TimelineSegment dicts for all event windows (sample_rate = 16 kHz)


class EventStage:
    def __init__(self, runner: ChunkedSpeechRunner, cfg: AvConfig, adapters: es.EventAdapters, *, classifier=None,
                 revisions: dict | None = None):
        self.r, self.cfg, self.ad = runner, cfg, adapters
        self.classifier = classifier
        self.revisions = revisions or {}
        m = cfg.memory
        self.ecfg = es.EventConfig(max_candidates=m.max_candidates, candidate_batch=m.candidate_batch,
                                   max_window_s=m.event_max_window_s)
        self.ecfg.validate()
        self.fp = code_fingerprint(es, ev, __import__(__name__, fromlist=["x"]))
        self.duration_s = runner.src.n_audio_samples / runner.src.audio_rate_hz

    # ---- detection (one small cache entry)
    def detect(self) -> tuple[str, list[AcousticEvent]]:
        r = self.r
        key = r.cache.key("event_detect", inputs={"media": r.media_id}, config={"top_k": 3, "window_s": 1.0, "hop_s": 0.5},
                          upstream={"ingest": r._ingest_key}, fingerprint=self.fp, models=self.revisions)

        def read_block(a_s: float, b_s: float) -> np.ndarray:
            m0 = int(round(a_s * r.src.audio_rate_hz))
            m1 = min(int(round(b_s * r.src.audio_rate_hz)), r.src.n_audio_samples)
            return r.src.read_audio(m0, m1, SPEECH_SR).mean(axis=0).astype(np.float32)

        def produce(d: Path, meta: dict) -> None:
            r.fault("event_detect:mid", 0)
            evs = list(ev.iter_detect_events(read_block, self.duration_s, self.classifier))
            (d / "events.json").write_text(json.dumps([e.model_dump(mode="json") for e in evs]), encoding="utf-8")
            r.counters.chunks_computed.append(("event_detect", 0))

        entry, computed = r.cache.get_or_compute("event_detect", key, produce, required=("events.json",))
        if not computed:
            r.counters.chunks_restored.append(("event_detect", 0))
        return key, [AcousticEvent.model_validate(e) for e in json.loads((entry / "events.json").read_text())]

    # ---- windows
    def _frames(self, lo_s: float, hi_s: float):
        r = self.r
        t0 = r.src.av_map.audio_start_us + int(round(lo_s * 1e6))
        t1 = r.src.av_map.audio_start_us + int(round(hi_s * 1e6))
        from .contracts import video_index_range
        i0, i1 = video_index_range(r._pts, t0, t1)
        cap = self.cfg.memory.max_decoded_frames
        stride = max(1, -(-(i1 - i0) // cap))                       # ceil: never decode more than the cap
        idx = list(range(i0, i1, stride))
        frames = [r.src.read_frames(i, i + 1)[0] for i in idx] if stride > 1 else (
            list(r.src.read_frames(i0, i1)) if i1 > i0 else [])
        r.counters.max_frames_resident = max(r.counters.max_frames_resident, len(frames))
        return (np.stack(frames) if frames else np.zeros((0, 2, 2, 3), np.uint8)), [r._pts[i] for i in idx]

    def run(self, speech: SpeechRunResult) -> EventRunResult:
        r, cfg = self.r, self.cfg
        dkey, events = self.detect()
        nonspeech = ev.non_speech_events(events)
        spans = [(e.start_us / 1e6, e.end_us / 1e6) for e in nonspeech]
        windows = es.bounded_event_windows(spans, self.duration_s, self.ecfg.pad_s, self.ecfg.merge_gap_s,
                                           self.ecfg.max_window_s)
        refs: list[EventRef] = []
        timeline: list[dict] = []
        prev_key: str | None = None
        master = r.src.audio_rate_hz
        for i, w in enumerate(windows):
            lo, hi = float(w[0]), float(w[1])
            inside = [e for e in nonspeech if e.end_us / 1e6 > lo and e.start_us / 1e6 < hi]
            best = max(inside, key=lambda e: e.activity_score) if inside else None
            prompt = (best.audio_prompt if best and best.audio_prompt else "sound")
            nouns = list(best.visual_noun_candidates) if best else []
            a16, b16 = int(round(lo * SPEECH_SR)), int(round(hi * SPEECH_SR))
            over = sorted(rec.speech_key for rec in speech.records if rec.spec.own_end > a16 and rec.spec.own_start < b16)
            sid = f"event_{i}"
            key = r.cache.key(
                "event_window", inputs={"media": r.media_id, "window": [round(lo, 6), round(hi, 6)]},
                config={"event": asdict(self.ecfg), "prompt": prompt, "nouns": nouns, "mem": cfg.memory.max_decoded_frames},
                upstream={"event_detect": dkey, **{f"speech:{k}": k for k in over}}, fingerprint=self.fp,
                models=self.revisions, seeds=[self.ecfg.seed_base + j for j in range(self.ecfg.max_candidates)],
                state_in=prev_key if getattr(w, "continues_prev", False) else None)

            def produce(d: Path, meta: dict, w=w, lo=lo, hi=hi, a16=a16, b16=b16, prompt=prompt, nouns=nouns, sid=sid) -> None:
                r.fault("event_window:mid", i)
                m0, m1 = int(round(lo * master)), min(int(round(hi * master)), r.src.n_audio_samples)
                mix = r.src.read_audio(m0, m1, SAM_RATE).mean(axis=0).astype(np.float32)
                others = []
                margin = r.f // 2 + 1
                near = [rec for rec in speech.records if rec.spec.own_end + margin > a16 and rec.spec.own_start - margin < b16]
                for sp in sorted({s for rec in near for s in read_arrays(rec.speech_dir, "final", "final.json")
                                  if s.startswith("src_")}):
                    x = assemble_range(near, sp, a16, b16, margin)
                    if np.any(x):
                        others.append(resample_to(x[None], SPEECH_SR, SAM_RATE)[0][: len(mix)])
                frames, pts = self._frames(lo, hi)
                ctrl = pick_control_span(windows, hi - lo, self.duration_s)
                cf, cp = (None, None)
                if ctrl is not None:
                    cf, cp = self._frames(*ctrl)
                res = es.run_event_window(w, mix_48k=mix, prompt=prompt, nouns=nouns, frames=frames, frame_pts_us=pts,
                                          adapters=self.ad, config=self.ecfg, master_rate_hz=master,
                                          other_sources=others, spatial=None, tdoa_expected=None,
                                          control_frames=cf, control_pts_us=cp,
                                          window_start_us=r.src.av_map.audio_start_us + int(round(lo * 1e6)))
                if res.stem is not None:
                    np.save(d / "stem.npy", res.stem.astype(np.float32))
                att = res.attribution
                meta.update({
                    "source_id": sid, "window": [lo, hi], "start_master": int(res.window_start_sample),
                    "end_master": int(res.window_end_sample), "has_stem": res.stem is not None,
                    "stem_rate_hz": res.stem_rate_hz, "chosen_seed": res.chosen_seed, "seeds": res.seeds,
                    "ranker_used": res.ranker_used, "gate": res.gate, "prompt": prompt, "nouns": nouns,
                    "continues_prev": res.continues_prev, "continues_next": res.continues_next,
                    "candidates": [asdict(c) for c in res.candidates],
                    "extraction": res.extraction.model_dump(mode="json"),
                    "attribution": None if att is None else {
                        "status": att.status, "visibility": att.visibility, "track_id": att.track_id,
                        "score": att.score, "reason": att.reason, "missing_evidence": att.missing_evidence},
                    "control_available": cf is not None, "spatial_available": False,
                    "frame_stride_cap": cfg.memory.max_decoded_frames,
                })
                (d / "event.json").write_text(json.dumps(meta, sort_keys=True, default=_jd), encoding="utf-8")
                r.counters.chunks_computed.append(("event_window", i))

            entry, computed = r.cache.get_or_compute("event_window", key, produce, required=("event.json",))
            if not computed:
                r.counters.chunks_restored.append(("event_window", i))
            r.fault("event_window:after", i)
            m = json.loads((entry / "event.json").read_text(encoding="utf-8"))
            refs.append(EventRef(sid, m["start_master"], m["end_master"], key, entry, m["stem_rate_hz"], m["has_stem"]))
            timeline.append(self._segment(m, a16, b16))
            prev_key = key
        return EventRunResult(refs, windows, dkey, timeline)

    def _segment(self, m: dict, a16: int, b16: int) -> dict:
        from .schemas import TimelineSegment
        att, ext = m["attribution"], m["extraction"]
        status = att["status"] if att else "unmatched"
        reason = None
        if ext["status"] in ("unavailable", "degraded", "not_attempted"):
            reason = ext["reason"] or m["gate"]
        elif att and att["status"] != "matched":
            reason = att["reason"]
        au = self.r.src.av_map.audio_start_us
        return TimelineSegment(
            source_id=m["source_id"], start_sample=a16, end_sample=b16, start_us=au + sample_to_us(a16, SPEECH_SR),
            end_us=au + sample_to_us(b16, SPEECH_SR), track_id=att["track_id"] if att else None,
            association_status=status, visibility=att["visibility"] if att else "unknown",
            stem_availability=ext["status"], refusal_reason=reason).model_dump(mode="json")


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)
