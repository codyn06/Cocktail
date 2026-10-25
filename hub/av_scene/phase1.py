"""Phase 1 deterministic pipeline over an on-disk synthetic fixture (evidence tier: mocked/synthetic).

fixture dir -> media index -> ORACLE Wiener-mask stems (derived from the truth stems, not a model)
-> K-weighted envelopes and gates -> speaker anchors -> spatial + onset-synchrony + semantic evidence
-> gated assignment with unmatched options -> residual accounting -> overlays -> session bundle.

Every learned-perception step is replaced by a deterministic stand-in, and every output is
labelled `mocked/synthetic`. Each step is implemented by a separately tested module; this file
only orchestrates them. Source labels come from the fixture truth (`fixture_truth_label`).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from ingest.media_io import (MediaTimeline, build_media_index, locate_ffmpeg, read_wav, write_audio_stream)
from locate import av_spatial as sp
from render.envelope import compute_envelope
from render.video import OverlayItem
from render.viewer import BundleInput, build_bundle
from replay import residual as rs
from separate.diarization import compute_anchors
from vision.mask_store import read_track_jsonl
from who.matcher import (Evidence, FaceObservation, LearningEligibility, assign, causal_lag_score, combine,
                         lip_asd_available)

from .cache import hash_config, hash_file
from .config import AvConfig, load_array_calibration, load_config
from .schemas import (AcousticEvent, Association, ComponentScore, EligibilityInterval, LabelScore, MaskletTrack,
                      RemainderResult, RunManifest, SceneSource, SpatialEvidence, SpeakerTrack, StageRecord,
                      StemResult)

PAIR_WINDOW_US = 300_000          # visual onset must lie within +-300 ms of an audio onset to be paired
NOUNS = {"speech": ["person"], "chirp": ["cat"], "knock": ["door", "wall"]}


def _git_rev() -> str | None:
    try:
        cwd = Path(__file__).parent
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, cwd=cwd).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                               timeout=10, cwd=cwd).stdout.strip()
        return (rev + ("+dirty" if dirty else "")) or None
    except Exception:
        return None


def oracle_stems(mix: np.ndarray, truth: dict[str, np.ndarray], cfg: rs.ResidualConfig):
    """Wiener masks M_k = |S_k|^2 / (sum_j |S_j|^2 + |N|^2) from truth magnitudes applied to the mixture.

    This is an ORACLE stand-in for a separator: it leaks where sources overlap in time-frequency and
    leaves the noise floor in the remainder, which makes leakage/remainder metrics non-trivial."""
    X = rs.stft(mix, cfg)
    S = {k: rs.stft(v, cfg) for k, v in truth.items()}
    total = sum(np.abs(s) ** 2 for s in S.values())
    noise = np.abs(rs.stft(mix - sum(truth.values()), cfg)) ** 2
    den = total + noise + 1e-18
    return {k: rs.istft(X * (np.abs(s) ** 2 / den), cfg, len(mix)).astype(np.float32) for k, s in S.items()}


def frame_sample_times(pts: list[int]) -> np.ndarray:
    """Display-midpoint of each frame: a frame is 'true' for [pts, next_pts); sampling there makes
    the quantization error of an onset at most half the local interval (q)."""
    p = np.asarray(pts, np.int64)
    nxt = np.concatenate([p[1:], [p[-1] + (p[-1] - p[-2])]])
    return p + (nxt - p) // 2


def visual_onsets(values, pts, thr: float = 0.5):
    """Mouth/body motion rising through `thr`, estimated midway between the two bracketing frames."""
    out = []
    for i in range(1, len(values)):
        if values[i] > thr >= values[i - 1] and (i + 1 >= len(values) or values[i + 1] > thr):
            out.append((int((pts[i - 1] + pts[i]) // 2), (pts[i] - pts[i - 1]) / 2000.0))
    return out


def run(fixture_dir: str | Path, out_dir: str | Path, cfg: AvConfig | None = None, ffmpeg_path: str | None = None,
        run_id: str = "phase1_fixture", write_annotated: bool = True, remove_stem: str | None = None) -> RunManifest:
    t_run = time.perf_counter()
    fx, out = Path(fixture_dir), Path(out_dir)
    cfg = cfg or load_config()
    cal = load_array_calibration()
    truth = json.loads((fx / "fixture.json").read_text())
    stages: list[StageRecord] = []
    inputs = {str(p.relative_to(fx).as_posix()): hash_file(p) for p in sorted(fx.rglob("*"))
              if p.is_file() and p.suffix in (".wav", ".json", ".jsonl")}
    chain = {"key": ""}

    def stage(name: str, t0: float, detail: str = "") -> None:
        # content-addressed and chained: a changed input, config or upstream stage changes every later key
        chain["key"] = hash_config({"stage": name, "inputs": inputs, "config": cfg.hash(), "upstream": chain["key"]})[:16]
        stages.append(StageRecord(name=name, key=chain["key"], seconds=round(time.perf_counter() - t0, 3), detail=detail))

    # ---- ingest
    t0 = time.perf_counter()
    master, rate = read_wav(fx / "master.wav")
    pts = [int(p) for p in truth["frame_pts_us"]]
    H, W = truth["image"]["h"], truth["image"]["w"]
    out.mkdir(parents=True, exist_ok=True)
    audio = write_audio_stream(out / "media", "mic", master, rate, layout="calibrated_array",
                               channel_order=cal.channel_order, rel_to=out)
    index = build_media_index(run_id, video_pts_us=pts, width=W, height=H, audio=[audio], nominal_fps=25.0,
                              frames_dir="frames",
                              input_hashes=inputs,
                              calibration_id=cal.calibration_id, ambient_temperature_c=cfg.acoustics.ambient_temperature_c,
                              temperature_provenance=cfg.acoustics.temperature_provenance, r_min_m=cfg.acoustics.r_min_m)
    tl = MediaTimeline(index)
    ft_mid = frame_sample_times(pts)
    mix = master.astype(np.float64).mean(axis=1)                       # declared mono downmix: mean over channels
    stage("ingest", t0, f"{len(pts)} frames, {len(index.gaps)} gap(s), master {rate} Hz x {master.shape[1]} ch")

    # ---- stems (oracle), envelopes, events
    t0 = time.perf_counter()
    src_truth = {s["id"]: s for s in truth["sources"]}
    truth_stems = {sid: read_wav(fx / s["stem_file"])[0][:, 0].astype(np.float64) for sid, s in src_truth.items()}
    used = {k: v for k, v in truth_stems.items() if k != remove_stem}
    stems = oracle_stems(mix, used, cfg.residual)
    envs, events, sources = {}, [], []
    for sid, st in stems.items():
        e = compute_envelope(st, rate, ft_mid, cfg.envelope)
        envs[sid] = e
        s = src_truth[sid]
        noun_key = "speech" if s["kind"] == "speech" else ("chirp" if "chirp" in s["label"] else "knock")
        evs = []
        for k, (a, b) in enumerate(e.segments_us):
            ev = AcousticEvent(event_id=f"ev_{sid}_{k}", labels=[LabelScore(label=s["label"], score=1.0)],
                               start_us=a, end_us=b, activity_score=1.0, audio_prompt=s["label"].lower(),
                               visual_noun_candidates=NOUNS[noun_key],
                               provenance={"label": "fixture_truth_label", "timing": "k_weighted_envelope_gate"})
            events.append(ev)
            evs.append(ev.event_id)
        sources.append(SceneSource(source_id=sid, kind=s["kind"], label=s["label"], event_ids=evs,
                                   speaker_id=sid if s["kind"] == "speech" else None, lifecycle="confirmed"))
    stage("separate+envelope", t0, "oracle Wiener-mask stems from truth magnitudes (not a model)")

    # ---- speaker anchors
    speech = [s for s in truth["sources"] if s["kind"] == "speech"]
    acts = {s["id"]: envs[s["id"]].segments_us for s in speech if s["id"] != remove_stem}
    noise = [iv for s in truth["sources"] if s["kind"] != "speech" and s["id"] in envs for iv in envs[s["id"]].segments_us]
    anchors = compute_anchors(acts, noise, cfg.anchors)
    speakers = [SpeakerTrack(speaker_id=k, turns_us=[tuple(iv) for iv in v], count_evidence={"method": "fixture_inventory"},
                             anchors=anchors[k].to_dict()) for k, v in acts.items()]

    # ---- evidence
    t0 = time.perf_counter()
    c = sp.c_sound(cfg.acoustics.ambient_temperature_c)
    pos = np.asarray(cal.mic_positions_m)
    d_pair = float(np.linalg.norm(pos[1] - pos[0]))
    nfft, hop = cfg.spatial.nfft, cfg.spatial.hop
    X0, X1 = sp.stft_frames(master[:, 0].astype(np.float64), nfft, hop), sp.stft_frames(master[:, 1].astype(np.float64), nfft, hop)
    ids = list(stems)
    Ws = dict(zip(ids, sp.wdo_weights([sp.stft_frames(stems[i].astype(np.float64), nfft, hop) for i in ids],
                                      cfg.spatial.wdo_min_weight)))
    t_frames = (np.arange(X0.shape[0]) * hop + nfft / 2) / rate
    par = sp.parallax_sigma(cal.t_cam_to_array_m, cfg.acoustics.r_min_m)   # widens the delay likelihood below
    tracks_truth = {t["id"]: t for t in truth["tracks"]}
    motion = json.loads((fx / "visual_motion.json").read_text())["motion"]
    faces = {}
    yaw_limit = cfg.off_axis.yaw_limit_deg
    for tid, t in tracks_truth.items():
        obs = [FaceObservation(a, b, y) for a, b, y in t["face_yaw"]]
        obs += [FaceObservation(a, b, 0.0, occluded=True) for a, b in t["occlusions_us"]]
        faces[tid] = obs

    def face_ok(tid: str, t_us: float) -> bool:
        if tracks_truth[tid]["concept"] != "person":
            return True                                              # human lip-ASD limits do not apply to the cat
        hit = [o for o in faces[tid] if o.start_us <= t_us < o.end_us]
        return bool(hit) and all(lip_asd_available(o, yaw_limit) for o in hit)

    vis_on = {tid: visual_onsets(motion[tid], pts) for tid in tracks_truth}
    sem = {("speech", "person"): 0.9, ("speech", "cat"): 0.1, ("chirp", "cat"): 0.9, ("chirp", "person"): 0.1}
    scores, spatial_info, lag_info = {}, {}, {}
    for sid in ids:
        s = src_truth[sid]
        e = envs[sid]
        kind = "speech" if s["kind"] == "speech" else ("chirp" if "chirp" in s["label"] else "knock")
        sel = np.zeros(len(t_frames), bool)
        for a, b in e.segments_us:
            sel |= (t_frames >= a / 1e6) & (t_frames < b / 1e6)
        st = stems[sid].astype(np.float64)
        tw, ow, onsets = sp.onset_flux_weights(st, rate, window_ms=cfg.spatial.onset_window_ms,
                                               sustain_weight=cfg.spatial.onset_sustain_weight)
        fc, bw = sp.spectral_centroid_bandwidth(st[np.abs(st) > 1e-6] if np.any(np.abs(st) > 1e-6) else st, rate)
        f_alias = sp.pair_alias_freq(d_pair, cfg.acoustics.ambient_temperature_c)
        alias_flag = bool(bw < cfg.spatial.narrowband_ratio * fc and fc > f_alias)
        cc = None
        if sel.any():
            w = Ws[sid][sel] * np.interp(t_frames[sel], tw, ow)[:, None]
            lags, cc = sp.weighted_gcc_phat(X0[sel], X1[sel], rate, nfft, 0.0005, cfg.spatial.interp, w)
        spatial_info[sid] = dict(peaks=[p[0] for p in sp.retained_peaks(lags, cc, 0.5, d_pair / c)][:6] if cc is not None else [],
                                 flag=alias_flag, bw=bw, fc=fc,
                                 onset_w=float(np.mean(np.interp([a / 1e6 for a, _ in e.segments_us], tw, ow))) if e.segments_us else None)
        for tid, t in tracks_truth.items():
            ev = []
            # onset synchrony against the track's measured visual onsets (unavailable where lips are not visible)
            lags_ms, paired, avail = [], [], 0
            for (a, _b) in e.segments_us:
                if not face_ok(tid, a):
                    continue
                avail += 1
                near = [(abs(a - tv), tv, q) for tv, q in vis_on[tid] if abs(a - tv) <= PAIR_WINDOW_US]
                if near:
                    _, tv, q = min(near)
                    lag = (a - tv) / 1000.0
                    lags_ms.append(lag)
                    paired.append(causal_lag_score(lag, cfg.matcher, q))
                else:
                    paired.append(0.0)                                 # video present, no motion: a real negative
            lag_info[(sid, tid)] = lags_ms
            ev.append(Evidence("onset_synchrony", float(np.mean(paired)) if avail else None, min(1.0, avail / 2.0), 1.0,
                               f"{avail} onset(s) with visible lips; paired lags {np.round(lags_ms, 1).tolist()} ms"
                               if avail else "lips not visible (off-axis/occluded): ASD unavailable"))
            # 3D spatial likelihood: camera ray -> predicted pair delay vs weighted post-PHAT GCC
            if cc is not None and par.confident:
                cam = cal.camera
                ray = sp.ray_from_pixel(*t["center_px"], cam.fx, cam.fy, cam.cx, cam.cy, np.asarray(cal.r_cam_to_array))
                tau = float(sp.predict_pair_tau(cal, ray, 0, 1, cfg.acoustics.ambient_temperature_c))
                # delay uncertainty: base resolution, calibrated channel noise and the camera-to-array parallax floor
                sig_tau = float(np.sqrt(35e-6 ** 2 + cal.channel_noise_s ** 2 + (d_pair / c * par.sigma_rad) ** 2))
                val = sp.ray_delay_score(lags, cc, tau, sig_tau)
                ev.append(Evidence("spatial_delay", val, 0.3 if alias_flag else 1.0, 0.8,
                                   (f"narrowband {fc:.0f} Hz > pair alias cutoff {f_alias:.0f} Hz: multimodal peaks kept"
                                    if alias_flag else "wideband post-PHAT WDO-weighted GCC")))
            else:
                ev.append(Evidence("spatial_delay", None, 0.0, 0.8, "no active frames or parallax withheld"))
            ev.append(Evidence("semantic_compat", sem.get((kind, t["concept"]), 0.3), 0.5, 0.4,
                               "label noun vs track concept (mocked prior)"))
            scores[(sid, tid)] = combine(ev, cfg.matcher)
    stage("evidence", t0, "onset synchrony + 3D pair-delay + semantic prior")

    # ---- association
    t_assoc = time.perf_counter()
    decisions = assign(ids, list(tracks_truth), scores, cfg.matcher)
    assocs, status_of, track_of = [], {}, {}
    speech_slots = {s["id"]: i for i, s in enumerate(speech)}
    for sid, dcs in decisions.items():
        s = src_truth[sid]
        info = spatial_info[sid]
        lags = lag_info.get((sid, dcs.track_id), []) if dcs.track_id else []
        status_of[sid], track_of[sid] = dcs.status, dcs.track_id
        el_records = []
        if dcs.track_id and sid in speech_slots:
            e2 = LearningEligibility(cfg.off_axis.unknown_pose, yaw_limit)
            for o in faces[dcs.track_id]:
                e2.observe(speech_slots[sid], o)
            el_records = [EligibilityInterval(**r) for r in e2.to_records()]
        a0 = min((a for a, _ in envs[sid].segments_us), default=0)
        b0 = max((b for _, b in envs[sid].segments_us), default=0)
        comps = scores[(sid, dcs.track_id)].components if dcs.track_id else (
            scores[(sid, max(dcs.candidates, key=lambda k: dcs.candidates[k]["score"] or -1))].components
            if any(v["score"] is not None for v in dcs.candidates.values()) else [])
        assocs.append(Association(
            association_id=f"assoc_{sid}", source_id=sid, track_id=dcs.track_id, valid_start_us=a0, valid_end_us=b0,
            status=dcs.status, visibility="visible" if dcs.track_id else "unknown", score=dcs.score,
            components=[ComponentScore(name=x.name, value=x.value, quality=x.quality, weight=x.weight, note=x.note)
                        for x in comps],
            runner_up_margin=dcs.margin, evidence_dependencies=["ev_" + sid + "_*", "mask_tracks"],
            spatial=SpatialEvidence(peaks_s=info["peaks"], onset_flux_weight=info["onset_w"],
                                    aliasing_multimodal_flag=info["flag"], aliasing_pairs=["0-1"] if info["flag"] else [],
                                    parallax_sigma_rad=par.sigma_rad, bandwidth_hz=info["bw"]),
            causal_lag_ms=float(np.median(lags)) if lags else None, learning_eligibility=el_records))
        for ss in sources:
            if ss.source_id == sid:
                ss.visibility = "visible" if dcs.track_id else "unknown"
                ss.association_revision, ss.current_association_id = 1, f"assoc_{sid}"

    # ---- masks, gated activity and per-track states
    tracks_masks = {tid: read_track_jsonl(fx / "masks_truth" / f"{tid}.jsonl") for tid in tracks_truth}
    n = len(pts)
    activity, track_states = {}, {tid: ["idle"] * n for tid in tracks_truth}
    caption = [None] * n
    for sid in ids:
        tid = track_of[sid]
        attributed = status_of[sid] in ("matched", "ambiguous") and tid is not None
        mask_ok = np.array([r.rle is not None and r.visibility == "visible" for r in tracks_masks[tid]]) if attributed \
            else np.zeros(n, bool)
        env = compute_envelope(stems[sid], rate, ft_mid, cfg.envelope, assoc_valid=np.full(n, attributed), mask_valid=mask_ok)
        envs[sid] = env
        kind = src_truth[sid]["kind"]
        for i in range(n):
            if env.active[i]:
                track_states[tid][i] = ("possible" if status_of[sid] == "ambiguous" else
                                        ("speaking" if kind == "speech" else "sounding"))
            elif env.gate_active[i] and not attributed:
                caption[i] = "UNATTRIBUTED SOUND: " + src_truth[sid]["label"].upper()
        activity[sid] = dict(source_id=sid, kind=kind, label=src_truth[sid]["label"], track_id=tid, status=status_of[sid],
                             attributed=attributed, segments_us=env.segments_us, active=env.active.astype(int).tolist(),
                             gate_active=env.gate_active.astype(int).tolist(), glow=np.round(env.glow, 3).tolist(),
                             level_dbfs_k=np.round(env.level_dbfs_k, 1).tolist(),
                             level_dbfs=np.round(env.level_dbfs, 1).tolist(), evidence_tier="mocked/synthetic")
    stage("association+gating", t_assoc, "gated assignment with unmatched dummies")

    # ---- residual accounting
    t0 = time.perf_counter()
    est = [stems[i].astype(np.float64) for i in ids]
    r_wave, r_meta = rs.raw_waveform_remainder(mix, est)
    events_u, proj = rs.find_unexplained_events(mix, est, rate, cfg.residual)
    events_ps, _ = rs.find_unexplained_events(mix, est, rate, cfg.residual, basis="power_subtraction")
    stage("residual", t0, f"{len(events_u)} unexplained event(s); projection error {proj.projection_error_db:.1f} dB")

    # ---- overlays + bundle
    overlays = []
    for i in range(n):
        items = []
        for tid, t in tracks_truth.items():
            rec = tracks_masks[tid][i]
            items.append(OverlayItem(tid, tid.upper(), track_states[tid][i], rec.rle.decode() if rec.rle else None,
                                     max([activity[s]["glow"][i] for s in ids if track_of[s] == tid] or [1.0])))
        overlays.append((items, caption[i]))
    manifest = RunManifest(
        run_id=run_id, config_hash=cfg.hash(), code_revision=_git_rev(), model_revisions={"separator": "oracle-wdo-wiener-v1"},
        input_hashes=index.input_hashes, stages=stages,
        timing={"sign": "positive = late", "reference": "t_visual_truth", "tof_range_ms": [cfg.matcher.tof_min_ms, cfg.matcher.tof_max_ms],
                "clock_uncertainty_ms": cfg.matcher.clock_uncertainty_ms,
                "quantization": "q = half of the local bracketing frame interval (1000/(2*FPS) at constant rate)",
                "gates": cfg.timing.__dict__, "temperature_c": cfg.acoustics.ambient_temperature_c,
                "temperature_provenance": cfg.acoustics.temperature_provenance, "frame_sampling": "display midpoint",
                "downmix": "mean_over_channels"},
        envelope=cfg.envelope.to_dict(), hardware={"device": "cpu", "note": "no GPU or model weights used"},
        media=index, calibration=cal, events=events, speakers=speakers,
        tracks=[MaskletTrack(track_id=t, concept=tracks_truth[t]["concept"], mask_file=f"masks/{t}.jsonl", image_height=H,
                             image_width=W, n_frames=n, first_pts_us=pts[0], last_pts_us=pts[-1],
                             visibility_summary={v: sum(r.visibility == v for r in tracks_masks[t])
                                                 for v in {r.visibility for r in tracks_masks[t]}}) for t in tracks_truth],
        sources=sources, associations=assocs,
        stems=[StemResult(source_id=i, model_revision="oracle-wdo-wiener-v1", sample_rate=rate, n_samples=len(mix),
                          raw_path=f"stems/source_{i}.wav", linked_remainder_ids=["rem_projected", "rem_waveform"])
               for i in ids],
        remainders=[
            RemainderResult(remainder_id="rem_waveform", kind="raw_waveform_diagnostic", source_stem_ids=ids, sample_rate=rate,
                            n_samples=len(mix), artifact_path="diagnostics/residual_waveform.wav", use="diagnostic", stft={}),
            RemainderResult(remainder_id="rem_projected", kind="phase_projected_stft", source_stem_ids=ids, sample_rate=rate,
                            n_samples=len(mix), stft=proj.meta, gain_bounds=(0.0, cfg.residual.alpha_max),
                            projection_error_db=proj.projection_error_db, artifact_path="residual.wav", use="playback_proxy",
                            proxy_waveform_provenance="istft(X - sum M_k X); projection proxy, not a separated source"),
            RemainderResult(remainder_id="rem_unexplained", kind="unexplained_power", source_stem_ids=ids, sample_rate=rate,
                            n_samples=len(mix), stft={"n_fft": cfg.residual.n_fft, "hop": cfg.residual.hop,
                                                      "events": events_u, "events_power_subtraction": events_ps},
                            artifact_path="residuals/unexplained_power.npz", use="discovery")])
    ff = locate_ffmpeg(ffmpeg_path or cfg.ffmpeg or None)
    manifest.capabilities = {"ffmpeg": ff.to_dict(), "learned_models": "none", "gpu": "unused"}
    frame_paths = [fx / "frames" / f"{i:06d}.png" for i in range(n)]
    build_bundle(BundleInput(out_dir=out, manifest=manifest, mixture=mix.astype(np.float32), sample_rate=rate,
                             stems=stems, residual_proxy=proj.proxy_waveform.astype(np.float32),
                             residual_waveform=r_wave.astype(np.float32), projection=proj, residual_cfg=cfg.residual,
                             frame_paths=frame_paths, pts_us=pts, tracks=tracks_masks, activity=activity,
                             track_states=track_states, overlays=overlays, ffmpeg=ff, write_annotated=write_annotated))
    manifest.stages.append(StageRecord(name="total", key=hash_config({"stage": "total", "upstream": chain["key"]})[:16], seconds=round(time.perf_counter() - t_run, 3)))
    (out / "manifest.json").write_text(manifest.to_json(indent=1), encoding="utf-8")
    return manifest
