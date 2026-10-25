"""Incremental visual front-end: persistent SAM 3 person tracks across overlapping decode windows.

`ChunkTracker.step(frames, pts_us, state)` implements `av_scene.contracts.VisualFrontEnd`.
Track ids are `sam3_image:<run_tag>:<n>` and stay stable across windows through an explicit,
JSON-safe `state` (no hidden mutable tracker). Frames are identified by pts: a frame whose pts is
already in the state's ring is served from the cache and NOT re-segmented; only new pts run the
segmenter. Tensors/ndarrays never outlive a frame (masks are RLE-encoded immediately; the only
decoded masks are the new frames' of the current window, handed to `face_boxes_fn`).

Identity caveat: ids are VISUAL tracklet ids from mask-IoU stitching. A person who leaves for more
than `max_gap` frames (counted in processed frames) and returns gets a NEW id. This is not visual
re-identification; speaker identity is resolved elsewhere (voice/face-embedding stage).

State schema (all JSON types):
  {"v": 1, "next": int, "n_frames": int,
   "tracklets": [{"n": int, "frame": int, "rle": RleMask.to_dict()}],   # sorted by n
   "ring": [{"pts": int, "tracks": {track_id: {"rle": dict, "conf": float, "box": [x0,y0,x1,y1]|None}}}]}
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from av_scene.contracts import VisualChunk
from vision.mask_store import FrameTransform, MaskRecord, RleMask, encode_mask
from vision.sam3_tracker import BACKEND, MaskIouTracker, Segmenter, dedupe

FaceBoxesFn = Callable[[np.ndarray, dict], dict]


def empty_state() -> dict:
    return {"v": 1, "next": 0, "n_frames": 0, "tracklets": [], "ring": []}


class ChunkTracker:
    def __init__(self, segment: Segmenter, face_boxes_fn: FaceBoxesFn, *, concept: str = "person",
                 run_tag: str, iou_thresh: float = 0.3, max_gap: int = 5, overlap_cache_s: float = 2.5,
                 min_confidence: float = 0.0):
        self.segment, self.face_boxes_fn = segment, face_boxes_fn
        self.concept, self.run_tag = concept, run_tag
        self.iou_thresh, self.max_gap = iou_thresh, max_gap
        self.overlap_cache_us = int(overlap_cache_s * 1_000_000)
        self.min_confidence = min_confidence

    def track_id(self, n: int) -> str:
        return f"{BACKEND}:{self.run_tag}:{n}"

    def step(self, frames: np.ndarray, pts_us: Sequence[int], state: dict) -> VisualChunk:
        if len(frames) != len(pts_us):
            raise ValueError("frames and pts_us length mismatch")
        state = state or empty_state()
        pts = [int(p) for p in pts_us]
        ring = {int(r["pts"]): r["tracks"] for r in state["ring"]}

        tracker = MaskIouTracker(self.iou_thresh, self.max_gap)
        tracker.last = {int(t["n"]): (int(t["frame"]), RleMask.from_dict(t["rle"]).decode())
                        for t in state["tracklets"]}
        tracker._next = int(state["next"])
        counter = int(state["n_frames"])

        per_frame: list[dict] = []          # per window frame: {tid: {"rle","conf","box"}}
        new_idx: list[int] = []
        new_masks: list[dict] = []          # decoded masks of new frames only, for face_boxes_fn
        for i, p in enumerate(pts):
            if p in ring:
                per_frame.append({t: dict(v) for t, v in ring[p].items()})
                continue
            dets = dedupe([d for d in self.segment(frames[i]) if d[1] >= self.min_confidence])
            tracks, masks = {}, {}
            for local, m, conf in tracker.update(counter, dets):
                tid = self.track_id(local)
                tracks[tid] = {"rle": encode_mask(m, FrameTransform.identity(*m.shape)).to_dict(),
                               "conf": float(conf), "box": None}
                masks[tid] = m
            counter += 1
            per_frame.append(tracks)
            new_idx.append(i)
            new_masks.append(masks)

        if new_idx:
            ids = sorted({t for ms in new_masks for t in ms})
            if ids:
                by_track = {t: [ms.get(t) for ms in new_masks] for t in ids}
                boxes = self.face_boxes_fn(frames[new_idx], by_track)
                for j, i in enumerate(new_idx):
                    for t in ids:
                        if t in per_frame[i]:
                            b = boxes.get(t, [None] * len(new_idx))[j]
                            per_frame[i][t]["box"] = None if b is None else [float(v) for v in b]
        del new_masks

        all_ids = sorted({t for pf in per_frame for t in pf})
        boxes_out: dict[str, list] = {t: [] for t in all_ids}
        masks_out: dict[str, list] = {t: [] for t in all_ids}
        for i, pf in enumerate(per_frame):
            for t in all_ids:
                e = pf.get(t)
                if e is None:
                    boxes_out[t].append(None)
                    masks_out[t].append(None)
                else:
                    b = e["box"]
                    boxes_out[t].append(tuple(b) if b is not None else None)
                    masks_out[t].append(MaskRecord(pts_us=pts[i], frame_idx=i, rle=RleMask.from_dict(e["rle"]),
                                                   confidence=e["conf"], visibility="visible", provenance="native"))

        # next state: ring = frames within overlap_cache of the newest pts (window frames + old ring)
        merged = dict(ring)
        for p, pf in zip(pts, per_frame):
            merged[p] = pf
        newest = max(merged) if merged else 0
        keep = sorted(p for p in merged if p >= newest - self.overlap_cache_us)
        tracklets = [{"n": n, "frame": f, "rle": encode_mask(m, FrameTransform.identity(*m.shape)).to_dict()}
                     for n, (f, m) in sorted(tracker.last.items()) if counter - 1 - f <= self.max_gap]
        new_state = {"v": 1, "next": tracker._next, "n_frames": counter, "tracklets": tracklets,
                     "ring": [{"pts": p, "tracks": merged[p]} for p in keep]}
        return VisualChunk(boxes_by_track=boxes_out, masks_by_track=masks_out, state=new_state)
