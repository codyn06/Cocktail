"""Stream a finished run into a long-clip inspection bundle without assembling any full-run array.

Inputs are the on-disk cache entries of the speech, event and residual stages. Audio goes out one stitched
segment at a time, frames/masks one chunk's owned frames at a time, and the timeline is built from the small
per-chunk segment JSON (metadata only, so it may be sorted in memory). Peak memory is one chunk.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from render.long_bundle import LongBundleWriter
from replay.chunked import ChunkedSpeechRunner, SpeechRunResult, read_arrays
from vision.mask_store import MaskRecord

from .config import AvConfig
from .contracts import SPEECH_SR, sample_to_us
from .schemas import (Association, MediaIndex, RunManifest, SceneSource, StageRecord, TimelineSegment,
                      VideoStreamInfo)

SAM_RATE = 48000


def _merge_spans(spans: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in sorted(spans, key=lambda s: s["start_s"]):
        if out and s["start_s"] <= out[-1]["end_s"] + 1e-9:
            out[-1]["end_s"] = max(out[-1]["end_s"], s["end_s"])
            out[-1]["peak_db_rel_mixture"] = max(out[-1]["peak_db_rel_mixture"], s["peak_db_rel_mixture"])
        else:
            out.append(dict(s))
    return out


def classify_sources(metas: list[dict]) -> dict[str, dict]:
    """source id -> {kind, role, label}. A speech source whose spans are mostly `outside_view` (and never matched
    to a visible track) is listed under outside-view voices; `tse:<track>` is always a diagnostic stem."""
    vis: dict[str, Counter] = {}
    matched: Counter = Counter()
    kinds: dict[str, dict] = {}
    for m in metas:
        kinds.update(m.get("kinds", {}))
        for s in m["segments"]:
            vis.setdefault(s["source_id"], Counter())[s["visibility"]] += 1
            if s["association_status"] == "matched":
                matched[s["source_id"]] += 1
    out = {}
    for sid, k in kinds.items():
        kind, role = k["kind"], k["role"]
        if kind == "speech" and not matched[sid] and vis.get(sid) and vis[sid]["outside_view"] * 2 > sum(vis[sid].values()):
            kind = "outside_view"
        out[sid] = {"kind": kind, "role": role, "label": "outside-view voice" if kind == "outside_view" else ""}
    return out


def export_bundle(out_dir: str | Path, *, runner: ChunkedSpeechRunner, speech: SpeechRunResult, events,
                  residual: list, cfg: AvConfig, run_id: str, config_hash: str, evidence_tier: str,
                  evidence_note: str, stages: list[StageRecord], model_revisions: dict, report: dict,
                  diarization: dict, code_revision: str | None = None, hardware: dict | None = None,
                  speakers=()) -> RunManifest:
    out = Path(out_dir)
    src = runner.src
    n16 = speech.n_samples
    master = src.audio_rate_hz
    pts = runner._pts
    metas = [json.loads((rec.speech_dir / "meta.json").read_text(encoding="utf-8")) for rec in speech.records]

    # ---- manifest shell (media index needs the frame size: read one frame)
    w = h = None
    if pts:
        f0 = src.read_frames(0, 1)
        h, w = int(f0.shape[1]), int(f0.shape[2])
    media = MediaIndex(run_id=run_id, input_hashes={"media": runner.media_id},
                       video=[VideoStreamInfo(stream_id="video0", width=w, height=h, pts_us=[int(p) for p in pts])]
                       if w else [])
    manifest = RunManifest(
        run_id=run_id, evidence_tier=evidence_tier, evidence_note=evidence_note, config_hash=config_hash,
        code_revision=code_revision, model_revisions=model_revisions, input_hashes={"media": runner.media_id},
        stages=stages, hardware=hardware or {}, media=media if w else None, capabilities={})
    from .schemas import LongRunInfo
    sw = cfg.switches
    manifest.long_run = LongRunInfo(
        chunking={"window_s": cfg.chunking.window_s, "overlap_s": cfg.chunking.overlap_s,
                  "crossfade_ms": cfg.chunking.crossfade_ms, "n_chunks": len(speech.records), "n_samples": n16,
                  "sample_rate": SPEECH_SR, "master_rate_hz": master,
                  "av_map": {"audio_start_us": src.av_map.audio_start_us, "video_transform": src.av_map.video_transform}},
        switches={k: getattr(sw, k) for k in sw.__dataclass_fields__},
        memory_caps=cfg.memory.__dict__ | {"budgets": cfg.budgets.__dict__}, diarization=diarization,
        events_index=None, resume=report.get("resume", {}))

    wr = LongBundleWriter(out, manifest)
    kinds = classify_sources(metas)
    for sid, k in sorted(kinds.items()):
        wr.register_source(sid, k["kind"], k["role"], k["label"])
    wr.register_source("unresolved", "speech", "diagnostic", "identity ambiguous: abstained, no stem")

    # ---- speech audio, one stitched segment at a time
    for rec in speech.records:
        for sid, (start, arr) in sorted(read_arrays(rec.speech_dir, "final", "final.json").items()):
            k = kinds[sid]
            wr.add_audio_segment(sid, k["kind"], k["role"], SPEECH_SR, SPEECH_SR / 2.0, start, arr)
    for sid in kinds:
        wr.set_source_length(sid, n16)

    # ---- mixture + residual proxy at the master rate (owned regions tile the clip)
    wr.register_source("mixture", "mixture", "mixture", "original mixture")
    wr.register_source("residual", "residual_proxy", "default", "unexplained residual (projection proxy)")
    res_by_spec = {spec.index: entry for spec, entry in residual}
    for rec in speech.records:
        o0, o1 = runner._master(rec.spec.own_start), runner._master(rec.spec.own_end)
        mix = src.read_audio(o0, o1, master).mean(axis=0).astype(np.float32)
        wr.add_audio_segment("mixture", "mixture", "mixture", master, None, o0, mix)
        e = res_by_spec.get(rec.spec.index)
        if e is not None:
            wr.add_audio_segment("residual", "residual_proxy", "default", master, None, o0, np.load(e / "proxy.npy"))
    wr.set_source_length("mixture", runner._master(n16))
    wr.set_source_length("residual", runner._master(n16))

    # ---- events: sparse stems, zero-filled by omission
    segs = [s for m in metas for s in m["segments"]]
    if events is not None:
        for ref in events.refs:
            wr.register_source(ref.source_id, "event", "default", "")
            if ref.has_stem:
                meta = json.loads((ref.entry / "event.json").read_text(encoding="utf-8"))
                lo = meta["window"][0]
                wr.add_audio_segment(ref.source_id, "event", "default", SAM_RATE, None, int(round(lo * SAM_RATE)),
                                     np.load(ref.entry / "stem.npy"))
        segs += events.timeline
        manifest.long_run.events_index = "timeline/index.json"

    # ---- residual unexplained activity as timeline spans (merged across chunk edges)
    spans = []
    for _, e in residual:
        spans += json.loads((e / "residual.json").read_text(encoding="utf-8")).get("unexplained_spans", [])
    au = src.av_map.audio_start_us
    for s in _merge_spans(spans):
        a, b = int(round(s["start_s"] * SPEECH_SR)), int(round(s["end_s"] * SPEECH_SR))
        segs.append(TimelineSegment(
            source_id="residual", start_sample=a, end_sample=b, start_us=au + sample_to_us(a, SPEECH_SR),
            end_us=au + sample_to_us(b, SPEECH_SR), association_status="unmatched", visibility="unknown",
            stem_availability="ok").model_dump(mode="json"))
    for s in sorted(segs, key=lambda s: (s["start_us"], s["source_id"])):
        wr.add_timeline_segment(TimelineSegment.model_validate(s))

    # ---- frames and masks, one chunk's OWNED frames at a time (tiles the video; no frame twice)
    if pts:
        from .contracts import video_index_range
        for rec in speech.records:
            t0 = -(1 << 62) if rec.spec.is_first else au + sample_to_us(rec.spec.own_start, SPEECH_SR)
            t1 = (1 << 62) if rec.spec.is_last else au + sample_to_us(rec.spec.own_end, SPEECH_SR)
            i0, i1 = video_index_range(pts, t0, t1)
            if i1 > i0:
                wr.add_frames(src.read_frames(i0, i1), pts[i0:i1])
            masks = json.loads((rec.visual_dir / "masks.json").read_text(encoding="utf-8"))
            for tid, recs in sorted(masks.items()):
                keep = [MaskRecord.from_dict(r) for r in recs if r is not None and t0 <= r["pts_us"] < t1]
                if keep:
                    wr.add_masks(tid, keep)

    # ---- manifest summaries so `python -m av_scene inspect` works on long bundles
    by_src: dict[str, list[dict]] = {}
    for s in segs:
        by_src.setdefault(s["source_id"], []).append(s)
    for sid, ss in sorted(by_src.items()):
        if sid in ("residual",):
            continue
        st = Counter(s["association_status"] for s in ss).most_common(1)[0][0]
        vi = Counter(s["visibility"] for s in ss).most_common(1)[0][0]
        tr = Counter(s["track_id"] for s in ss if s["track_id"]).most_common(1)
        manifest.sources.append(SceneSource(
            source_id=sid, kind="non_speech" if sid.startswith("event_") else "speech", label=kinds.get(sid, {}).get("label", ""),
            lifecycle="confirmed", visibility=vi, current_association_id=f"assoc_{sid}"))
        manifest.associations.append(Association(
            association_id=f"assoc_{sid}", source_id=sid, track_id=tr[0][0] if tr else None,
            valid_start_us=min(s["start_us"] for s in ss), valid_end_us=max(s["end_us"] for s in ss),
            status=st, visibility=vi))
    manifest.speakers = list(speakers)
    wr.finalize(sample_rate=SPEECH_SR, n_samples=n16)
    (out / "run_report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return manifest
