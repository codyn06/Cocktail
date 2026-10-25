"""Phase 2: first real model slice on a short recorded clip (one speaker, one silent bystander).

    uv run python -m av_scene.phase2 --video clip.mp4 --run-id demo [--speaker-xy 0.3,0.5]

decode -> MOCK discovery (one hardcoded "person speaking" event) -> SAM 3 *image* model per frame
+ mask-IoU tracking ("person") -> MOCK association (a person picks the speaker's masklet) ->
SAM Audio small-tv with that masklet as the visual prompt -> Phase 1 envelope gating, residual
accounting, overlays and bundle. Each model runs alone under `ModelLifecycleManager`: loaded,
used, unloaded, GPU cache emptied, lease released, before the next one loads.

Evidence label: `replayed` (recorded clip); model execution is hardware-tested only on the laptop
named in the manifest. Discovery and association are HARDCODED, not learned.

Two-pass use: without --speaker-xy / --speaker-track the run stops after tracking, writes
`_work/tracks_preview.png` and the track list, and exits 2; re-run with --reuse-tracks and a
selector to skip the (slow) SAM 3 pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from importlib import metadata as md
from pathlib import Path

import numpy as np
import torch

from ingest.media_io import locate_ffmpeg, read_wav, write_png
from render.envelope import compute_envelope
from render.video import OverlayItem, overlay_frame
from render.viewer import BundleInput, build_bundle
from replay import residual as rs
from replay import sam_audio as sa
from vision import sam3_tracker as tr
from vision.mask_store import MaskRecord, read_track_jsonl

from . import preflight
from .cache import hash_config, hash_file
from .config import load_config
from .phase1 import _git_rev, frame_sample_times
from .runtime import StageReport, run_stage
from .schemas import (AcousticEvent, Association, ComponentScore, LabelScore, MaskletTrack, RemainderResult,
                      RunManifest, SceneSource, SpeakerTrack, StageRecord, StemResult)

SPEAKER_ID = "spk0"


# --------------------------------------------------------------------------- decode
@dataclass
class Clip:
    frames: torch.Tensor          # (T, 3, H, W) uint8
    pts_us: list[int]             # relative to the first kept frame
    audio: np.ndarray             # mono float32, windowed to exactly the frames' span
    audio_rate: int
    fps: float
    source: str

    @property
    def duration_us(self) -> int:
        dt = int(np.median(np.diff(self.pts_us))) if len(self.pts_us) > 1 else 40_000
        return self.pts_us[-1] + dt


def decode_clip(path: str | Path, *, max_seconds: float = 10.0, max_side: int = 640, stride: int = 1) -> Clip:
    """Decode with TorchCodec; the audio is cut to the same time window as the kept frames
    (SAM Audio maps video frames onto audio positions uniformly, so the spans must agree)."""
    preflight.configure_ffmpeg_dlls()
    from torchcodec.decoders import AudioDecoder, VideoDecoder

    vd = VideoDecoder(str(path), dimension_order="NCHW")
    batch = vd.get_frames_in_range(0, len(vd))
    pts = batch.pts_seconds.numpy().astype(np.float64)
    keep = np.where(pts - pts[0] < max_seconds)[0][::max(1, stride)]
    if len(keep) < 2:
        raise ValueError("clip has fewer than 2 usable frames")
    frames, pts = batch.data[keep], pts[keep]
    h, w = frames.shape[-2:]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        frames = torch.nn.functional.interpolate(frames.float(), size=(round(h * s / 2) * 2, round(w * s / 2) * 2),
                                                 mode="area").round().clamp(0, 255).to(torch.uint8)
    try:
        samples = AudioDecoder(str(path)).get_all_samples()
    except Exception as e:
        raise ValueError(f"{path} has no decodable audio track: {e}") from e
    rate = int(samples.sample_rate)
    wav = samples.data.float().mean(0).numpy()
    t0 = pts[0]
    dt = float(np.median(np.diff(pts)))
    a0, a1 = int(round(t0 * rate)), int(round((pts[-1] + dt) * rate))
    return Clip(frames=frames, pts_us=[int(round((p - t0) * 1e6)) for p in pts], audio=wav[a0:a1].astype(np.float32),
                audio_rate=rate, fps=1.0 / dt, source=str(path))


# --------------------------------------------------------------------------- association (mock)
def select_speaker_track(records: dict[str, list[MaskRecord]], *, xy: tuple[float, float] | None = None,
                         track_id: str | None = None) -> str:
    """HARDCODED association: the user names the speaker, by track id or by a normalised (x, y)
    point that lies inside the speaker's mask for most of the clip."""
    if track_id:
        if track_id not in records:
            raise SystemExit(f"unknown track {track_id!r}; have {sorted(records)}")
        return track_id
    if xy is None:
        raise ValueError("need xy or track_id")
    hits = {}
    for tid, recs in records.items():
        n = 0
        for r in recs:
            if r.rle is not None:
                m = r.rle.decode()
                n += bool(m[min(int(xy[1] * m.shape[0]), m.shape[0] - 1), min(int(xy[0] * m.shape[1]), m.shape[1] - 1)])
        hits[tid] = n
    best = max(hits, key=hits.get)
    if hits[best] == 0:
        raise SystemExit(f"no track contains point {xy}; hits per track: {hits}")
    return best


def _label(i: int, tid: str, speaker: str | None) -> str:
    return f"P{i + 1}" + (" SPEAKER" if tid == speaker else "")


def write_preview(frame_rgb: np.ndarray, tracks: list[MaskletTrack], records, frame_idx: int, path: Path) -> None:
    items = []
    for i, t in enumerate(tracks):
        rec = next((r for r in records[t.track_id] if r.frame_idx == frame_idx), None) or records[t.track_id][0]
        items.append(OverlayItem(t.track_id, f"P{i + 1}", "possible", rec.rle.decode() if rec.rle else None))
    write_png(path, overlay_frame(frame_rgb, items, None))


# --------------------------------------------------------------------------- helpers
def _rms_dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))), 1e-9)))


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]


def _pkg(name: str) -> str:
    try:
        v = md.version(name)
        raw = md.distribution(name).read_text("direct_url.json")
        sha = json.loads(raw)["vcs_info"]["commit_id"][:8] if raw and "vcs_info" in raw else None
        return v + (f"@{sha}" if sha else "")
    except md.PackageNotFoundError:
        return "missing"


def _tracks_from_disk(work: Path) -> tuple[list[MaskletTrack], dict[str, list[MaskRecord]]]:
    meta = json.loads((work / "tracks.json").read_text(encoding="utf-8"))
    tracks = [MaskletTrack.model_validate(t) for t in meta["tracks"]]
    return tracks, {t.track_id: read_track_jsonl(work / t.mask_file) for t in tracks}


# --------------------------------------------------------------------------- pipeline
def run(video: str | Path, out_dir: str | Path, *, run_id: str = "phase2", speaker_xy=None, speaker_track=None,
        reuse_tracks: bool = False, concept: str = "person", text: str = sa.DEFAULT_TEXT, max_seconds: float = 10.0,
        max_side: int = 640, stride: int = 1, device: str = "cuda", probe_bystanders: int = 2,
        ffmpeg_path: str | None = None, sam3_threshold: float = 0.25) -> RunManifest | None:
    t_run = time.perf_counter()
    out, work = Path(out_dir), Path(out_dir) / "_work"
    work.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    stages: list[StageRecord] = []
    reports: list[StageReport] = []
    chain = {"key": hash_file(Path(video))[:16]}

    def stage(name: str, t0: float, detail: str = "") -> None:
        chain["key"] = hash_config({"stage": name, "upstream": chain["key"], "cfg": cfg.hash()})[:16]
        stages.append(StageRecord(name=name, key=chain["key"], seconds=round(time.perf_counter() - t0, 3), detail=detail))

    # ---- decode
    t0 = time.perf_counter()
    clip = decode_clip(video, max_seconds=max_seconds, max_side=max_side, stride=stride)
    T, _, H, W = clip.frames.shape
    pts = clip.pts_us
    dur_us, rate = clip.duration_us, sa.TARGET_RATE
    mix = sa.resample_audio(clip.audio, clip.audio_rate, rate)[0].numpy().astype(np.float32)
    stage("decode", t0, f"{T} frames {W}x{H} at ~{clip.fps:.1f} fps (stride {stride}), "
                        f"{len(clip.audio) / clip.audio_rate:.2f} s audio {clip.audio_rate}->{rate} Hz")

    # ---- mock discovery
    t0 = time.perf_counter()
    event = AcousticEvent(event_id="ev_mock_0", labels=[LabelScore(label=text, score=1.0)], start_us=0, end_us=dur_us,
                          activity_score=1.0, audio_prompt=text, visual_noun_candidates=[concept],
                          provenance={"label": "MOCK_hardcoded_discovery", "note": "no AST / diarization in Phase 2"})
    speaker = SpeakerTrack(speaker_id=SPEAKER_ID, turns_us=[(0, dur_us)],
                           count_evidence={"method": "MOCK_hardcoded", "speakers": 1},
                           anchors={"mode": "full_clip", "anchors": [["+", 0.0, dur_us / 1e6]]})
    stage("discovery(mock)", t0, "one hardcoded event covering the whole clip")

    # ---- vision: SAM 3 image model per frame + mask-IoU tracking
    if reuse_tracks and (work / "tracks.json").is_file():
        tracks, records = _tracks_from_disk(work)
        stages.append(StageRecord(name="vision(reused)", key=chain["key"], status="skipped",
                                  detail=f"reused {len(tracks)} track(s) from _work/tracks.json"))
    else:
        t0 = time.perf_counter()
        rgb = clip.frames.permute(0, 2, 3, 1).contiguous().numpy()        # (T, H, W, 3), views not copies per frame
        res = run_stage(
            "sam3_image", lambda: tr.load_sam3_image(concept, device=device, confidence_threshold=sam3_threshold),
            lambda mgr, seg: _timed(mgr, "inference", lambda: tr.track_concept_images(
                seg, (rgb[i] for i in range(T)), concept, out_dir=work, frame_pts_us=pts, run_tag=run_id,
                min_track_frames=max(2, T // 10))),
            reports=reports, device=device, workload={"frames": T, "size": f"{W}x{H}", "concept": concept})
        tracks, records = res.tracks, res.records
        (work / "tracks.json").write_text(json.dumps({"tracks": [t.model_dump() for t in tracks]}), encoding="utf-8")
        stage("vision(sam3_image)", t0, f"{len(tracks)} track(s) from {res.frames_seen} frames; model unloaded")
    if not tracks:
        raise SystemExit(f"SAM 3 found no {concept!r} tracks; lower --sam3-threshold or check the clip")

    # ---- mock association
    t0 = time.perf_counter()
    mid = T // 2
    write_preview(clip.frames[mid].permute(1, 2, 0).numpy(), tracks, records, mid, work / "tracks_preview.png")
    if speaker_xy is None and not speaker_track:
        print("tracks found (pick the SPEAKER, the rest are bystanders):")
        for i, t in enumerate(tracks):
            area = np.mean([r.rle.area for r in records[t.track_id] if r.rle]) / (H * W)
            print(f"  P{i + 1}  {t.track_id}  frames={t.n_frames}  mean_area={area:.1%}  conf={t.mean_confidence:.2f}")
        print(f"preview: {work / 'tracks_preview.png'}\nre-run with --reuse-tracks and --speaker-xy X,Y (0..1) "
              "or --speaker-track ID")
        return None
    sid = select_speaker_track(records, xy=speaker_xy, track_id=speaker_track)
    sp_track = next(t for t in tracks if t.track_id == sid)
    bystanders = [t for t in tracks if t.track_id != sid]
    stage("association(mock)", t0, f"HARDCODED: {SPEAKER_ID} -> {sid}; {len(bystanders)} bystander track(s) unassigned")

    # ---- separation: SAM Audio small-tv with the speaker masklet as visual prompt
    t0 = time.perf_counter()
    by_frame = {r.frame_idx: r.rle for r in records[sid]}
    masks = [by_frame.get(i) for i in range(T)]
    diag: dict[str, dict] = {}

    def separate(mgr, bundle):
        with mgr.phase("inference"):
            st, prompts = sa.extract_stem(bundle, audio=clip.audio, sample_rate=clip.audio_rate, frames=clip.frames,
                                          masks=masks, source_id=SPEAKER_ID, out_dir=work, text=text)
            diag["speaker"] = {"result": st, "prompts": prompts}
            # measure visual-prompt selectivity: the same text with a silent bystander's masklet as the prompt
            for t in sorted(bystanders, key=lambda t: -t.n_frames)[:probe_bystanders]:
                bf = {r.frame_idx: r.rle for r in records[t.track_id]}
                bst, _ = sa.extract_stem(bundle, audio=clip.audio, sample_rate=clip.audio_rate, frames=clip.frames,
                                         masks=[bf.get(i) for i in range(T)], source_id="probe_" + t.track_id,
                                         out_dir=work, text=text)
                diag[t.track_id] = {"result": bst}
        mgr.report.output_seconds = round(len(mix) / rate, 3)

    run_stage("sam_audio_small_tv", lambda: sa.load_sam_audio(device=device), separate, reports=reports,
              device=device, workload={"frames": T, "audio_s": round(len(mix) / rate, 2), "text": text,
                                       "bystander_probes": min(probe_bystanders, len(bystanders))})
    stem_res: StemResult = diag["speaker"]["result"]
    stem = _fit(read_wav(work / stem_res.raw_path)[0][:, 0], len(mix))
    stage("separation(sam_audio_small_tv)", t0, f"stem {stem_res.n_samples} samples @ {rate} Hz; model unloaded")

    # ---- render: Phase 1 envelope gating, residual accounting, overlays, bundle
    t0 = time.perf_counter()
    ft = frame_sample_times(pts)
    mask_ok = np.array([i in by_frame and by_frame[i] is not None for i in range(T)])
    env = compute_envelope(stem, rate, ft, cfg.envelope, assoc_valid=np.ones(T, bool), mask_valid=mask_ok)
    states = {t.track_id: ["idle"] * T for t in tracks}
    for i in range(T):
        if env.active[i]:
            states[sid][i] = "speaking"
    activity = {SPEAKER_ID: dict(
        source_id=SPEAKER_ID, kind="speech", label=text, track_id=sid, status="matched", attributed=True,
        segments_us=env.segments_us, active=env.active.astype(int).tolist(), gate_active=env.gate_active.astype(int).tolist(),
        glow=np.round(env.glow, 3).tolist(), level_dbfs_k=np.round(env.level_dbfs_k, 1).tolist(),
        level_dbfs=np.round(env.level_dbfs, 1).tolist(), evidence_tier="replayed")}
    r_wave, _ = rs.raw_waveform_remainder(mix, [stem])
    events_u, proj = rs.find_unexplained_events(mix, [stem], rate, cfg.residual)
    order = {t.track_id: i for i, t in enumerate(tracks)}
    overlays = []
    for i in range(T):
        items = []
        for t in tracks:
            rec = next((r for r in records[t.track_id] if r.frame_idx == i), None)
            items.append(OverlayItem(t.track_id, _label(order[t.track_id], t.track_id, sid), states[t.track_id][i],
                                     rec.rle.decode() if rec and rec.rle else None, float(env.glow[i]) if t.track_id == sid else 1.0))
        overlays.append((items, None))

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    frame_paths = []
    for i in range(T):
        p = frames_dir / f"{i:06d}.png"
        write_png(p, clip.frames[i].permute(1, 2, 0).numpy())
        frame_paths.append(p)

    from ingest.media_io import build_media_index, write_audio_stream
    audio_info = write_audio_stream(out / "media", "mic", mix, rate, layout="mono", rel_to=out)
    index = build_media_index(run_id, video_pts_us=pts, width=W, height=H, audio=[audio_info], nominal_fps=clip.fps,
                              frames_dir="frames", input_hashes={Path(video).name: hash_file(Path(video))},
                              ambient_temperature_c=cfg.acoustics.ambient_temperature_c,
                              temperature_provenance=cfg.acoustics.temperature_provenance, r_min_m=cfg.acoustics.r_min_m)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    revisions = {"sam3": _pkg("sam3"), "sam-audio": _pkg("sam-audio"), "sam_audio_checkpoint": stem_res.model_revision,
                 "torch": _pkg("torch"), "torchcodec": _pkg("torchcodec")}
    manifest = RunManifest(
        run_id=run_id, evidence_tier="replayed",
        evidence_note=(f"Recorded clip {Path(video).name}. SAM 3 (image) and SAM Audio small-tv ran on {gpu}. "
                       "Discovery and association are HARDCODED mocks. No listening review or ground-truth scoring yet."),
        config_hash=cfg.hash(), code_revision=_git_rev(), model_revisions=revisions, input_hashes=index.input_hashes,
        stages=stages, timing={"sign": "positive = late", "reference": "n/a (no ground truth)",
                               "frame_sampling": "display midpoint", "downmix": "mean_over_channels"},
        envelope=cfg.envelope.to_dict(),
        hardware={"device": gpu, "stages": [r.to_dict() for r in reports]},
        media=index, events=[event], speakers=[speaker], tracks=tracks,
        sources=[SceneSource(source_id=SPEAKER_ID, kind="speech", label=text, event_ids=[event.event_id],
                             speaker_id=SPEAKER_ID, lifecycle="confirmed", visibility="visible",
                             association_revision=1, current_association_id=f"assoc_{SPEAKER_ID}")],
        associations=[Association(
            association_id=f"assoc_{SPEAKER_ID}", source_id=SPEAKER_ID, track_id=sid, valid_start_us=0, valid_end_us=dur_us,
            status="matched", visibility="visible", score=None,
            components=[ComponentScore(name="manual_assignment", value=None, note="HARDCODED by the operator; not evidence")])],
        stems=[stem_res.model_copy(update={"linked_remainder_ids": ["rem_projected", "rem_waveform"]})],
        remainders=[
            RemainderResult(remainder_id="rem_waveform", kind="raw_waveform_diagnostic", source_stem_ids=[SPEAKER_ID],
                            sample_rate=rate, n_samples=len(mix), artifact_path="diagnostics/residual_waveform.wav", use="diagnostic"),
            RemainderResult(remainder_id="rem_projected", kind="phase_projected_stft", source_stem_ids=[SPEAKER_ID],
                            sample_rate=rate, n_samples=len(mix), stft=proj.meta, gain_bounds=(0.0, cfg.residual.alpha_max),
                            projection_error_db=proj.projection_error_db, artifact_path="residual.wav", use="playback_proxy",
                            proxy_waveform_provenance="istft(X - sum M_k X); projection proxy, not a separated source"),
            RemainderResult(remainder_id="rem_unexplained", kind="unexplained_power", source_stem_ids=[SPEAKER_ID],
                            sample_rate=rate, n_samples=len(mix), stft={"events": events_u},
                            artifact_path="residuals/unexplained_power.npz", use="discovery")])
    ff = locate_ffmpeg(ffmpeg_path or cfg.ffmpeg or (str(preflight.DEFAULT_FFMPEG_BIN / "ffmpeg.exe")
                                                     if (preflight.DEFAULT_FFMPEG_BIN / "ffmpeg.exe").is_file() else None))
    manifest.capabilities = {"ffmpeg": ff.to_dict(), "learned_models": ["sam3_image", "sam_audio_small_tv"],
                             "discovery": "MOCK", "association": "MOCK", "gpu": gpu}
    build_bundle(BundleInput(
        out_dir=out, manifest=manifest, mixture=mix, sample_rate=rate, stems={SPEAKER_ID: stem},
        residual_proxy=proj.proxy_waveform.astype(np.float32), residual_waveform=r_wave.astype(np.float32),
        projection=proj, residual_cfg=cfg.residual, frame_paths=frame_paths, pts_us=pts, tracks=records,
        activity=activity, track_states=states, overlays=overlays, ffmpeg=ff, write_annotated=True))
    stage("render", t0, f"{int(env.active.sum())}/{T} frames speaking; {len(events_u)} unexplained event(s)")

    # ---- metrics (bystander behaviour is measured, not assumed)
    probes = {}
    for t in bystanders[:probe_bystanders]:
        if t.track_id in diag:
            b = _fit(read_wav(work / diag[t.track_id]["result"].raw_path)[0][:, 0], len(mix))
            probes[t.track_id] = {"stem_rms_dbfs": round(_rms_dbfs(b), 1),
                                  "vs_speaker_stem_db": round(_rms_dbfs(b) - _rms_dbfs(stem), 1),
                                  "corr_with_speaker_stem": round(float(np.corrcoef(b, stem)[0, 1]), 3)}
    metrics = {
        "evidence_tier": "replayed", "clip": Path(video).name, "frames": T, "duration_s": round(dur_us / 1e6, 2),
        "speaker": {"track": sid, "active_frames": int(env.active.sum()), "gate_frames": int(env.gate_active.sum()),
                    "stem_rms_dbfs": round(_rms_dbfs(stem), 1), "mixture_rms_dbfs": round(_rms_dbfs(mix), 1),
                    "stem_minus_mixture_db": round(_rms_dbfs(stem) - _rms_dbfs(mix), 1), "prompts": diag["speaker"]["prompts"]},
        "bystanders": {"tracks": [t.track_id for t in bystanders], "highlighted_frames": 0,
                       "visual_prompt_probe": probes,
                       "probe_note": "same text prompt with the bystander's masklet as visual prompt; a silent bystander "
                                     "should give a stem far below the speaker's (vs_speaker_stem_db strongly negative)"},
        "stages": [r.to_dict() for r in reports],
        "peak_vram_mib": {r.name: {"allocated": r.peak_allocated_mib, "reserved": r.peak_reserved_mib,
                                   "after_release": r.allocated_after_release_mib} for r in reports},
        "total_s": round(time.perf_counter() - t_run, 2),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    manifest.stages.append(StageRecord(name="total", key=hash_config({"stage": "total", "upstream": chain["key"]})[:16],
                                       seconds=metrics["total_s"]))
    (out / "manifest.json").write_text(manifest.to_json(indent=1), encoding="utf-8")
    shutil.rmtree(frames_dir, ignore_errors=True)
    return manifest


def _timed(mgr, name, fn):
    with mgr.phase(name):
        return fn()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m av_scene.phase2", description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--run-id", default="phase2")
    ap.add_argument("--out", default=None, help="bundle dir (default sessions/<run-id>/av)")
    ap.add_argument("--speaker-xy", default=None, help="normalised x,y (0..1) inside the speaker's mask")
    ap.add_argument("--speaker-track", default=None)
    ap.add_argument("--reuse-tracks", action="store_true")
    ap.add_argument("--concept", default="person")
    ap.add_argument("--text", default=sa.DEFAULT_TEXT)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sam3-threshold", type=float, default=0.25)
    ap.add_argument("--probe-bystanders", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ffmpeg", default=None)
    a = ap.parse_args(argv)
    xy = tuple(float(v) for v in a.speaker_xy.split(",")) if a.speaker_xy else None
    out = Path(a.out) if a.out else Path("sessions") / a.run_id / "av"
    m = run(a.video, out, run_id=a.run_id, speaker_xy=xy, speaker_track=a.speaker_track, reuse_tracks=a.reuse_tracks,
            concept=a.concept, text=a.text, max_seconds=a.max_seconds, max_side=a.max_side, stride=a.stride,
            device=a.device, probe_bystanders=a.probe_bystanders, ffmpeg_path=a.ffmpeg, sam3_threshold=a.sam3_threshold)
    if m is None:
        return 2
    print(f"bundle: {out.resolve()}\nserve:  uv run python -m av_scene serve {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
