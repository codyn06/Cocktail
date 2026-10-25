"""S3FD-only face tracks for the Phase 5 refiner (no SAM 3 masks available).

NEW in Phase 5 and UNPROVEN on real footage: greedy IoU association of per-frame S3FD detections.
Output matches ``run_speech_branch``'s ``boxes_by_track``: ``{"f0": [box | None, ...], ...}`` with one entry per
frame, box = (x0, y0, x1, y1) floats, ``None`` where the face was not detected on that frame (a track is carried
through short gaps for matching only; a guessed box is never emitted).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


class FaceBoxTracker:
    def __init__(self, iou_thresh: float = 0.3, max_gap: int = 3, max_tracks: int = 3, min_frames: int = 3):
        self.iou_thresh, self.max_gap, self.max_tracks, self.min_frames = iou_thresh, max_gap, max_tracks, min_frames
        self._tracks: list[dict] = []
        self._n = 0

    def update(self, dets) -> None:
        """Feed one frame of detections ([k,5]: x0,y0,x1,y1,score)."""
        i = self._n
        self._n += 1
        d = np.asarray(dets, dtype=np.float64).reshape(-1, 5) if dets is not None and len(dets) else np.zeros((0, 5))
        boxes = [tuple(float(v) for v in r[:4]) for r in d]
        cands = sorted(((_iou(t["last"], b), ti, bi) for ti, t in enumerate(self._tracks)
                        if i - t["last_i"] <= self.max_gap + 1 for bi, b in enumerate(boxes)), reverse=True)
        used_t, used_b = set(), set()
        for v, ti, bi in cands:
            if v < self.iou_thresh:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            t = self._tracks[ti]
            t["boxes"][i] = boxes[bi]
            t["last"], t["last_i"] = boxes[bi], i
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                self._tracks.append({"boxes": {i: b}, "last": b, "last_i": i})

    def result(self) -> dict[str, list]:
        keep = [t for t in self._tracks if len(t["boxes"]) >= self.min_frames]

        def area(t):
            return float(np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in t["boxes"].values()]))
        keep.sort(key=lambda t: (-len(t["boxes"]), -area(t)))
        return {f"f{k}": [t["boxes"].get(i) for i in range(self._n)] for k, t in enumerate(keep[: self.max_tracks])}


def detect_tracks(detector, frames: Sequence, **kw) -> dict[str, list]:
    """frames: uint8 RGB [N,H,W,3] (or a sequence of them); detector.detect(rgb) -> [k,5]."""
    tr = FaceBoxTracker(**kw)
    for f in frames:
        tr.update(detector.detect(f))
    return tr.result()
