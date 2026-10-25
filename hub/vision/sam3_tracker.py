"""SAM 3 video-predictor adapter: text concept -> `MaskletTrack`s with RLE masks.

Uses the stateful API (`build_sam3_video_predictor`; `handle_request` start/add_prompt/close;
`handle_stream_request` propagate). Memory rule: every propagated frame is converted to RLE
immediately and the session is closed in `finally`, before control returns to the orchestrator.

SAFETY: ARCHITECTURE.md D9 records that the SAM 3.1 video tracker crashed this Windows laptop
twice (bugcheck 0x00020001, ~15.5 s/frame). `load_sam3_video_predictor` therefore refuses to
build on win32 unless COCKTAIL_ALLOW_SAM3_VIDEO=1 is set. The tracker logic itself is exercised
by unit tests against a fake predictor; it has NOT been run on hardware.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from av_scene.schemas import MaskletTrack
from vision.mask_store import FrameTransform, MaskRecord, RleMask, encode_mask, write_track_jsonl

BACKEND = "sam3_video"


@dataclass
class TrackingResult:
    session_id: str
    concept: str
    tracks: list[MaskletTrack] = field(default_factory=list)
    records: dict[str, list[MaskRecord]] = field(default_factory=dict)  # track_id -> per-frame records
    frames_seen: int = 0


def load_sam3_video_predictor(**kwargs):
    """Build the predictor (loads weights; expect `facebook/sam3` access). Windows needs an opt-in."""
    if sys.platform == "win32" and os.environ.get("COCKTAIL_ALLOW_SAM3_VIDEO") != "1":
        raise RuntimeError("SAM 3 video predictor crashed this Windows laptop twice (ARCHITECTURE.md D9). "
                           "Set COCKTAIL_ALLOW_SAM3_VIDEO=1 to build it anyway.")
    from sam3.model_builder import build_sam3_video_predictor

    kwargs.setdefault("gpus_to_use", [0])
    return build_sam3_video_predictor(**kwargs)


def shutdown_predictor(predictor) -> None:
    """Lifecycle cleanup hook: terminate workers if the predictor exposes a shutdown."""
    for name in ("shutdown", "close"):
        fn = getattr(predictor, name, None)
        if callable(fn):
            fn()
            return


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _file_stem(track_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", track_id)  # ':' is not a legal Windows filename character


def track_concept(predictor, video_path: str | Path, concept: str = "person", *, out_dir: str | Path,
                  frame_pts_us: Sequence[int] | None = None, fps: float | None = None,
                  prompt_frame: int = 0, min_confidence: float = 0.0) -> TrackingResult:
    """Prompt `concept` on `prompt_frame`, propagate, and export each masklet to RLE.

    Object ids are namespaced `sam3_video:<session8>:<obj>`. Frames are matched to timestamps
    via `frame_pts_us` (preferred) or `fps`. Masks are stored at the predictor's output size with an
    identity transform; the caller attaches a different `FrameTransform` if it resized.
    """
    out_dir = Path(out_dir)
    pts_of = (lambda i: int(frame_pts_us[i])) if frame_pts_us is not None else (
        (lambda i: int(round(i * 1e6 / fps))) if fps else (lambda i: i))
    session_id = predictor.handle_request({"type": "start_session", "resource_path": str(video_path)})["session_id"]
    result = TrackingResult(session_id=session_id, concept=concept)
    shape: dict[str, tuple[int, int]] = {}
    try:
        predictor.handle_request({"type": "add_prompt", "session_id": session_id,
                                  "frame_index": prompt_frame, "text": concept})
        for item in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id,
                                                     "propagation_direction": "both"}):
            result.frames_seen += 1
            _export_frame(result, session_id, item["frame_index"], item["outputs"], pts_of, shape, min_confidence)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})  # free VRAM first

    for track_id, recs in result.records.items():
        recs.sort(key=lambda r: r.frame_idx)
        rel = f"masks/{_file_stem(track_id)}.jsonl"
        write_track_jsonl(out_dir / rel, recs)
        h, w = shape[track_id]
        vis: dict[str, int] = {}
        for r in recs:
            vis[r.visibility] = vis.get(r.visibility, 0) + 1
        result.tracks.append(MaskletTrack(
            track_id=track_id, concept=concept, mask_file=rel, image_height=h, image_width=w, n_frames=len(recs),
            first_pts_us=recs[0].pts_us, last_pts_us=recs[-1].pts_us,
            mean_confidence=float(np.mean([r.confidence for r in recs])), visibility_summary=vis))
    return result


def _export_frame(result: TrackingResult, session_id: str, frame_idx: int, outputs: dict, pts_of, shape: dict,
                  min_confidence: float) -> None:
    obj_ids = _np(outputs.get("out_obj_ids", []))
    if obj_ids.size == 0:
        return
    masks = _np(outputs["out_binary_masks"]).astype(bool)
    probs = _np(outputs["out_probs"]) if outputs.get("out_probs") is not None else np.ones(len(obj_ids))
    for i, obj in enumerate(obj_ids):
        conf = float(probs[i])
        if conf < min_confidence:
            continue
        m = masks[i]
        if m.ndim == 3:
            m = m[0]
        track_id = f"{BACKEND}:{session_id[:8]}:{int(obj)}"
        shape[track_id] = m.shape
        rle: RleMask = encode_mask(m, FrameTransform.identity(*m.shape))
        result.records.setdefault(track_id, []).append(
            MaskRecord(pts_us=pts_of(frame_idx), frame_idx=int(frame_idx), rle=rle, confidence=conf,
                       visibility="visible" if m.any() else "unknown", provenance="native"))
