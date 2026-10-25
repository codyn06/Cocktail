"""SAM 3 image-model adapter with mask-IoU tracking: text concept -> `MaskletTrack`s with RLE masks.

Backend `sam3_image`: the SAM 3 *image* model runs on every frame and a small tracker stitches
the per-frame detections into tracklets by mask IoU. This is the laptop baseline of spec 4.1.
The SAM 3 video predictor is deliberately not used: ARCHITECTURE.md D9 records that the video
tracker crashed this Windows machine twice (bugcheck 0x00020001, ~15.5 s/frame).

Memory rule: tensors never outlive a frame. Each detection is converted to a numpy mask and then
to RLE straight away; the GPU model itself is released by `ModelLifecycleManager`.
D9: the model runs with full-precision weights under bf16 autocast, as upstream does (converting
the weights to bf16 breaks it). "person" can score low on a single prompt (0.27 vs 0.96 for
"child" in D9), so the default confidence threshold is lowered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from av_scene.schemas import MaskletTrack
from vision.mask_store import FrameTransform, MaskRecord, encode_mask, write_track_jsonl

BACKEND = "sam3_image"
Detection = tuple[np.ndarray, float]          # (H, W) bool mask, confidence
Segmenter = Callable[[np.ndarray], list[Detection]]


@dataclass
class TrackingResult:
    run_tag: str
    concept: str
    tracks: list[MaskletTrack] = field(default_factory=list)
    records: dict[str, list[MaskRecord]] = field(default_factory=dict)   # track_id -> per-frame records
    frames_seen: int = 0


# --------------------------------------------------------------------------- model
class Sam3Image:
    """SAM 3 image model + processor behind a `(rgb) -> detections` call. `.to` lets the lifecycle
    manager move the model off the GPU."""

    def __init__(self, model, processor, concept: str):
        self.model, self.processor, self.concept = model, processor, concept

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, rgb: np.ndarray) -> list[Detection]:
        from PIL import Image

        cuda = next(self.model.parameters()).is_cuda
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            state = self.processor.set_image(Image.fromarray(rgb))
            state = self.processor.set_text_prompt(prompt=self.concept, state=state)
        masks = state["masks"].detach().cpu().numpy()
        scores = state["scores"].detach().float().cpu().numpy()
        del state                                                    # free this frame's GPU tensors now
        if masks.size == 0:
            return []
        masks = masks.reshape(-1, *masks.shape[-2:]).astype(bool)
        return [(m, float(s)) for m, s in zip(masks, scores.reshape(-1))]


def load_sam3_image(concept: str = "person", *, device: str = "cuda", confidence_threshold: float = 0.25) -> Sam3Image:
    """Build the image model (downloads `facebook/sam3` on first use: gated, explicit action)."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(device=device)
    return Sam3Image(model, Sam3Processor(model, device=device, confidence_threshold=confidence_threshold), concept)


# --------------------------------------------------------------------------- tracking
def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def dedupe(dets: list[Detection], thresh: float = 0.8) -> list[Detection]:
    """Drop near-duplicate masks of the same object within one frame (keep the higher score)."""
    kept: list[Detection] = []
    for m, s in sorted(dets, key=lambda d: -d[1]):
        if m.any() and all(mask_iou(m, k) < thresh for k, _ in kept):
            kept.append((m, s))
    return kept


class MaskIouTracker:
    """Greedy-optimal frame-to-frame association by mask IoU. A track survives `max_gap` missed
    frames (occlusion / dropped detection) and is matched against its last seen mask."""

    def __init__(self, iou_thresh: float = 0.3, max_gap: int = 5):
        self.iou_thresh, self.max_gap = iou_thresh, max_gap
        self.last: dict[int, tuple[int, np.ndarray]] = {}    # local id -> (last frame, last mask)
        self._next = 0

    def update(self, frame_idx: int, dets: list[Detection]) -> list[tuple[int, np.ndarray, float]]:
        live = [t for t, (f, _) in self.last.items() if frame_idx - f <= self.max_gap + 1]
        assigned: dict[int, int] = {}
        if live and dets:
            iou = np.array([[mask_iou(dets[d][0], self.last[t][1]) for t in live] for d in range(len(dets))])
            rows, cols = linear_sum_assignment(-iou)
            assigned = {int(r): live[c] for r, c in zip(rows, cols) if iou[r, c] >= self.iou_thresh}
        out = []
        for d, (m, s) in enumerate(dets):
            tid = assigned.get(d)
            if tid is None:
                tid, self._next = self._next, self._next + 1
            self.last[tid] = (frame_idx, m)
            out.append((tid, m, s))
        return out


def _file_stem(track_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", track_id)


def track_concept_images(segment: Segmenter, frames: Iterable[np.ndarray], concept: str = "person", *,
                         out_dir: str | Path, frame_pts_us: Sequence[int], run_tag: str = "run",
                         iou_thresh: float = 0.3, max_gap: int = 5, min_confidence: float = 0.0,
                         min_track_frames: int = 1) -> TrackingResult:
    """Segment every frame, stitch detections into tracklets by mask IoU, export RLE `MaskletTrack`s.

    `frames` yields (H, W, 3) uint8 RGB in the same order as `frame_pts_us`. Masks are stored at
    frame size with an identity transform (the caller attaches another `FrameTransform` if it resized).
    Track ids are namespaced `sam3_image.<run_tag>.<n>` (no ':' so they are valid Windows file names).
    """
    out_dir = Path(out_dir)
    tracker = MaskIouTracker(iou_thresh, max_gap)
    result = TrackingResult(run_tag=run_tag, concept=concept)
    shapes: dict[str, tuple[int, int]] = {}
    for i, rgb in enumerate(frames):
        dets = dedupe([d for d in segment(rgb) if d[1] >= min_confidence])
        for local, m, conf in tracker.update(i, dets):
            track_id = f"{BACKEND}.{run_tag}.{local}"
            shapes[track_id] = m.shape
            result.records.setdefault(track_id, []).append(MaskRecord(
                pts_us=int(frame_pts_us[i]), frame_idx=i, rle=encode_mask(m, FrameTransform.identity(*m.shape)),
                confidence=conf, visibility="visible", provenance="native"))
        result.frames_seen += 1

    for track_id in [t for t, r in result.records.items() if len(r) < min_track_frames]:
        del result.records[track_id]                                  # flicker, not a person
    for track_id, recs in result.records.items():
        rel = f"masks/{_file_stem(track_id)}.jsonl"
        write_track_jsonl(out_dir / rel, recs)
        h, w = shapes[track_id]
        result.tracks.append(MaskletTrack(
            track_id=track_id, concept=concept, mask_file=rel, image_height=h, image_width=w, n_frames=len(recs),
            first_pts_us=recs[0].pts_us, last_pts_us=recs[-1].pts_us,
            mean_confidence=float(np.mean([r.confidence for r in recs])), visibility_summary={"visible": len(recs)}))
    return result
