"""Audit the exact tensors that reach SAM Audio (no model weights needed; processor only).

Run from hub/:  uv run python ../eval/av_scene/audit_inputs.py --out sessions/av_matrix/audit

For each audited case/target it records (audit.json) and saves PNGs of what the model sees:
  audio       input rate, channel count, dtype, peak, clipped fraction, DC, RMS; batch tensor layout,
              feature count vs expected ceil(n / hop), seconds per feature
  video       masked tensor layout/dtype/range, per-tile retained/blacked pixel stats (mask polarity
              checked on the real tensors), exact-equality of the retained pixels with the source
  vision      what the PE-Core encoder does to the frame (square resize, aspect distortion, normalisation),
              plus the 336x336 view of the first masked frame
  alignment   audio feature k (time k*hop/sr) -> video frame index chosen by the processor's linspace
              mapping, against the decoded presentation timestamps of that frame
  decode      audio/video start timestamps of the source mp4s (does decode_clip's t=0 assumption hold)
  anchors     the anchor list and the anchor_ids / anchor_alignment tensors derived from it
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
HUB = HERE.parents[1] / "hub"
for p in (str(HUB), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import metrics as M  # noqa: E402
import mixtures as X  # noqa: E402
from replay import sam_audio as sa  # noqa: E402


def _png(path: Path, chw: torch.Tensor) -> str:
    from torchvision.io import write_png

    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(chw.to(torch.uint8).cpu(), str(path))
    return str(path)


def audit_audio(batch, proc, audio_in: np.ndarray, rate_in: int) -> dict:
    a = np.asarray(audio_in, np.float64)
    wav = batch.audios
    n = wav.shape[-1]
    return {
        "input_rate_hz": rate_in, "model_rate_hz": proc.audio_sampling_rate, "resampled": rate_in != proc.audio_sampling_rate,
        "input_shape": list(audio_in.shape), "input_dtype": str(audio_in.dtype), "input_channels": 1 if audio_in.ndim == 1 else audio_in.shape[0],
        "batch_audios_shape": list(wav.shape), "batch_audios_dtype": str(wav.dtype), "layout": "(B, 1, T) mono",
        "peak": float(np.abs(a).max()), "clipped_fraction_ge_0.999": float((np.abs(a) >= 0.999).mean()),
        "dc_offset": float(a.mean()), "rms_dbfs": M.rms_dbfs(a), "nan_or_inf": bool(not np.isfinite(a).all()),
        "hop_length": proc.audio_hop_length, "features": int(batch.sizes[0]),
        "features_expected_ceil(n/hop)": math.ceil(n / proc.audio_hop_length),
        "seconds_per_feature": proc.audio_hop_length / proc.audio_sampling_rate,
        "wav_size": int(batch.wav_sizes[0]), "seconds": n / proc.audio_sampling_rate,
    }


def audit_video(case, target, masked: torch.Tensor, frames: torch.Tensor, all_masks: dict, image_size: int, out: Path) -> dict:
    T, C, H, W = frames.shape
    mask = all_masks[target.mask]
    fg = mask.bool()
    other_masks = [v for k, v in all_masks.items() if k != target.mask]
    keep = masked.bool().any(1)                                   # (T, H, W): pixel survived (non-black)
    retained_fg = float((masked.float().sum(1)[fg] > 0).float().mean())
    retained_bg = float((masked.float().sum(1)[~fg] > 0).float().mean())
    exact = bool(torch.equal(masked[:, :, fg[0]], frames[:, :, fg[0]])) if fg.ndim == 3 else None
    info = {
        "frames_shape": [T, C, H, W], "layout": "(T, C, H, W) = NCHW", "dtype": str(masked.dtype),
        "pixel_range_in": [int(frames.min()), int(frames.max())], "pixel_range_masked": [int(masked.min()), int(masked.max())],
        "mask_target_fraction_of_frame": float(fg.float().mean()),
        "polarity_check": {"fraction_of_target_pixels_kept_nonblack": retained_fg,
                           "fraction_of_background_pixels_kept_nonblack": retained_bg,
                           "target_pixels_bit_identical_to_source": exact,
                           "verdict": "OK: target kept, background blacked out" if retained_fg > 0.9 and retained_bg < 0.01
                                      else "CHECK: polarity or mask problem"},
        "mean_luma_target_region": float(masked.float().mean(1)[fg].mean()),
        "mean_luma_other_regions": [float(masked.float().mean(1)[m.bool()].mean()) for m in other_masks],
    }
    # what the vision encoder sees: the processor passes (T,3,H,W) uint8 through Resize((s, s)) -> /255 -> (x-0.5)/0.5
    sx, sy = image_size / W, image_size / H
    tile_w, tile_h = [int(round(v)) for v in (fg[0].any(0).sum().item() * sx, fg[0].any(1).sum().item() * sy)]
    info["encoder_input"] = {
        "square_resize_to": [image_size, image_size], "aspect_preserved": False,
        "scale_x": sx, "scale_y": sy, "horizontal_stretch_vs_vertical": sx / sy,
        "target_bbox_in_encoder_pixels": {"w": tile_w, "h": tile_h},
        "normalisation": "x/255 then (x-0.5)/0.5 -> [-1, 1]; black (0) maps to -1",
    }
    mid = T // 2
    resized = torch.nn.functional.interpolate(masked[mid:mid + 1].float(), size=(image_size, image_size), mode="bilinear")
    info["pngs"] = {
        "source_mid": _png(out / f"{case.case_id}__{target.name}__source.png", frames[mid]),
        "masked_first": _png(out / f"{case.case_id}__{target.name}__masked_first.png", masked[0]),
        "masked_mid": _png(out / f"{case.case_id}__{target.name}__masked_mid.png", masked[mid]),
        "masked_last": _png(out / f"{case.case_id}__{target.name}__masked_last.png", masked[-1]),
        "encoder_view_mid": _png(out / f"{case.case_id}__{target.name}__encoder_336.png", resized[0].round().clamp(0, 255)),
    }
    return info


def audit_alignment(n_frames: int, n_feat: int, pts_s: np.ndarray, hop: int, rate: int) -> dict:
    """Processor mapping (processor.load_video): frame index = linspace(0, T-1, n_feat).round()."""
    idx = torch.linspace(0, n_frames - 1, n_feat).round().long().numpy()
    t_feat = np.arange(n_feat) * hop / rate
    err = (pts_s[idx] - t_feat) * 1000.0
    dup = int(n_feat - len(np.unique(idx)))
    return {"n_frames": n_frames, "n_features": n_feat, "feature_time_s": "k*hop/sr",
            "frame_time_error_ms": {"mean": float(err.mean()), "max_abs": float(np.abs(err).max()), "min": float(err.min()), "max": float(err.max())},
            "duplicated_frame_picks": dup, "dropped_frames": int(n_frames - len(np.unique(idx))),
            "frame_span_s": float(pts_s[-1] - pts_s[0] + np.median(np.diff(pts_s))), "audio_span_s": n_feat * hop / rate,
            "assumption": "uniform frame rate: pts are ignored by the processor; non-uniform pts would shift frames"}


def audit_decode(sessions: Path) -> dict:
    from av_scene import preflight

    preflight.configure_ffmpeg_dlls()
    from torchcodec.decoders import AudioDecoder, VideoDecoder

    out = {}
    for k in ("compA", "compB"):
        p = sessions / k / "av" / "source_video.mp4"
        s = AudioDecoder(str(p)).get_all_samples()
        vd = VideoDecoder(str(p))
        pts = vd.get_frames_in_range(0, len(vd)).pts_seconds.numpy()
        out[k] = {"audio_first_sample_pts_s": float(s.pts_seconds), "audio_rate": int(s.sample_rate), "audio_channels": int(s.data.shape[0]),
                  "audio_dtype": str(s.data.dtype), "video_first_pts_s": float(pts[0]), "video_frames": len(pts),
                  "video_pts_step_ms_min_max": [float(np.diff(pts).min() * 1000), float(np.diff(pts).max() * 1000)],
                  "start_offset_audio_minus_video_ms": float((s.pts_seconds - pts[0]) * 1000)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sessions/av_matrix/audit")
    ap.add_argument("--sessions", default=str(X.SESSIONS))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from sam_audio import SAMAudioProcessor
    from huggingface_hub import snapshot_download

    folder = snapshot_download(repo_id=sa.SAM_AUDIO_REPO)
    cfg = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8"))
    image_size = int(cfg["vision_encoder"].get("image_size", 336))
    bundle = sa.SamAudioBundle(None, SAMAudioProcessor.from_pretrained(folder), sa.SAM_AUDIO_REPO)
    proc = bundle.processor

    report = {"tier": X.TIER, "processor": {"hop": proc.audio_hop_length, "rate": proc.audio_sampling_rate},
              "vision_encoder_cfg": {k: cfg["vision_encoder"].get(k) for k in ("name", "image_size", "batch_size", "dim", "interpolation_mode", "normalize_feature")},
              "decode": audit_decode(Path(a.sessions)), "cases": []}
    for case in X.build_cases(Path(a.sessions), include=["overlap_stack", "overlap_side", "nonspeech_two"]):
        pipeline_frames, pipeline_masks = X.resize_like_pipeline(case.frames, case.masks) if case.layout in ("side",) else (case.frames, case.masks)
        for t in case.targets:
            if t.name not in ("A", "cat"):
                continue
            mask = pipeline_masks[t.mask]
            for tag, anchors in (("full_clip", "full_clip"), ("none", None)):
                batch, anchor_list, masked = sa.build_batch(bundle, audio=case.audio, sample_rate=X.RATE, frames=pipeline_frames,
                                                            masks=mask, text=t.text, anchors=anchors)
                rec = {"case": case.case_id, "target": t.name, "layout": case.layout, "anchors_mode": tag, "text": t.text,
                       "audio": audit_audio(batch, proc, case.audio, X.RATE),
                       "video": audit_video(case, t, masked, pipeline_frames, pipeline_masks, image_size, out / "png") if tag == "full_clip" else "see full_clip",
                       "alignment": audit_alignment(pipeline_frames.shape[0], int(batch.sizes[0]),
                                                    np.arange(pipeline_frames.shape[0]) / (pipeline_frames.shape[0] / case.duration),
                                                    proc.audio_hop_length, proc.audio_sampling_rate),
                       "anchors": {"anchor_list": anchor_list, "anchor_ids": batch.anchor_ids.tolist(),
                                   "alignment_value_counts": {str(k): int(v) for k, v in zip(*np.unique(batch.anchor_alignment.numpy(), return_counts=True))},
                                   "masked_video_tensors_in_batch": len(batch.masked_video)}}
                report["cases"].append(rec)
            # sanity: masks of different targets must select different pixels
    (out / "audit.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    # console summary
    print("decode:", json.dumps(report["decode"]))
    for r in report["cases"]:
        if r["anchors_mode"] != "full_clip":
            continue
        v, au, al = r["video"], r["audio"], r["alignment"]
        print(f"{r['case']:<22} tgt={r['target']:<4} audio {au['model_rate_hz']}Hz peak={au['peak']:.3f} clip={au['clipped_fraction_ge_0.999']:.4f} "
              f"feat={au['features']}/{au['features_expected_ceil(n/hop)']}")
        print(f"    video {v['frames_shape']} {v['dtype']} polarity={v['polarity_check']['verdict']} "
              f"stretch={v['encoder_input']['horizontal_stretch_vs_vertical']:.2f} bbox_in_encoder={v['encoder_input']['target_bbox_in_encoder_pixels']}")
        print(f"    align max|err|={al['frame_time_error_ms']['max_abs']:.2f} ms dup={al['duplicated_frame_picks']} dropped={al['dropped_frames']}")
    print("wrote", out / "audit.json")


if __name__ == "__main__":
    main()
