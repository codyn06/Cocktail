"""Internal run-length codec and per-track storage for segmentation masks.

Conventions (recorded in every encoded mask so a reader never has to guess):
  * `order = "row-major"`: pixels are scanned row by row, left to right, top to bottom.
  * `counts` are alternating run lengths that always START with a background run (possibly 0).
    Even indices are background, odd indices are foreground. `foreground = True` records that
    the "foreground" pixels are the ones belonging to the object (the SAM Audio helper expects
    the opposite polarity; convert at that boundary, never here).
  * `bbox_norm = (x0, y0, x1, y1)` is normalised to the ORIGINAL frame, via `transform`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCAN_ORDER = "row-major"


@dataclass(frozen=True)
class FrameTransform:
    """Affine map from mask pixel coordinates to original-frame pixel coordinates.

    `matrix = (a, b, tx, c, d, ty)`: x_o = a*x + b*y + tx, y_o = c*x + d*y + ty.
    Covers resize, crop, letterbox, mirror and quarter-turn rotation."""

    orig_height: int
    orig_width: int
    matrix: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @classmethod
    def identity(cls, height: int, width: int) -> FrameTransform:
        return cls(height, width)

    @classmethod
    def resize(cls, orig_height: int, orig_width: int, mask_height: int, mask_width: int) -> FrameTransform:
        return cls(orig_height, orig_width, (orig_width / mask_width, 0.0, 0.0, 0.0, orig_height / mask_height, 0.0))

    @classmethod
    def crop_resize(cls, orig_height: int, orig_width: int, crop_x: float, crop_y: float,
                    crop_w: float, crop_h: float, mask_height: int, mask_width: int) -> FrameTransform:
        return cls(orig_height, orig_width,
                   (crop_w / mask_width, 0.0, crop_x, 0.0, crop_h / mask_height, crop_y))

    def mirrored(self, mask_width: int) -> FrameTransform:
        """Transform for a mask that was horizontally flipped before encoding."""
        a, b, tx, c, d, ty = self.matrix
        return FrameTransform(self.orig_height, self.orig_width,
                              (-a, b, tx + a * mask_width, -c, d, ty + c * mask_width))

    def apply(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, np.float64).reshape(-1, 2)
        a, b, tx, c, d, ty = self.matrix
        return np.stack([a * p[:, 0] + b * p[:, 1] + tx, c * p[:, 0] + d * p[:, 1] + ty], axis=1)

    def to_dict(self) -> dict:
        return {"orig_height": self.orig_height, "orig_width": self.orig_width, "matrix": list(self.matrix)}

    @classmethod
    def from_dict(cls, d: dict) -> FrameTransform:
        return cls(int(d["orig_height"]), int(d["orig_width"]), tuple(float(v) for v in d["matrix"]))


@dataclass(frozen=True)
class RleMask:
    height: int
    width: int
    counts: tuple[int, ...]
    transform: FrameTransform
    bbox_norm: tuple[float, float, float, float] | None
    order: str = SCAN_ORDER
    foreground: bool = True

    @property
    def area(self) -> int:
        return int(sum(self.counts[1::2]))

    def decode(self) -> np.ndarray:
        return decode_mask(self)

    def to_dict(self) -> dict:
        return {"h": self.height, "w": self.width, "order": self.order, "foreground": self.foreground,
                "counts": list(self.counts), "bbox_norm": list(self.bbox_norm) if self.bbox_norm else None,
                "transform": self.transform.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> RleMask:
        bb = d.get("bbox_norm")
        return cls(int(d["h"]), int(d["w"]), tuple(int(c) for c in d["counts"]),
                   FrameTransform.from_dict(d["transform"]), tuple(bb) if bb else None,
                   d.get("order", SCAN_ORDER), bool(d.get("foreground", True)))


def encode_mask(mask: np.ndarray, transform: FrameTransform | None = None) -> RleMask:
    """Encode a boolean (H, W) mask; True pixels are the object."""
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("mask must be 2-D (H, W)")
    m = m.astype(bool)
    h, w = m.shape
    tf = transform or FrameTransform.identity(h, w)
    flat = m.reshape(-1)  # C order == row-major
    if flat.size == 0:
        counts: list[int] = [0]
    else:
        change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        edges = np.concatenate([[0], change, [flat.size]])
        runs = [int(r) for r in np.diff(edges)]
        counts = [0] + runs if flat[0] else runs
    return RleMask(h, w, tuple(counts), tf, bbox_normalized(m, tf))


def decode_mask(rle: RleMask) -> np.ndarray:
    if rle.order != SCAN_ORDER:
        raise ValueError(f"unsupported scan order {rle.order!r}")
    total = rle.height * rle.width
    if sum(rle.counts) != total:
        raise ValueError(f"RLE covers {sum(rle.counts)} pixels, expected {total}")
    flat = np.zeros(total, bool)
    pos = 0
    for i, n in enumerate(rle.counts):
        if i % 2 == 1:
            flat[pos:pos + n] = True
        pos += n
    out = flat.reshape(rle.height, rle.width)
    return out if rle.foreground else ~out


def bbox_normalized(mask: np.ndarray, transform: FrameTransform) -> tuple[float, float, float, float] | None:
    """Bounding box of the True pixels' outer edges, mapped to the original frame and normalised."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    corners = transform.apply(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float))
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    W, H = transform.orig_width, transform.orig_height
    return (float(np.clip(lo[0] / W, 0, 1)), float(np.clip(lo[1] / H, 0, 1)),
            float(np.clip(hi[0] / W, 0, 1)), float(np.clip(hi[1] / H, 0, 1)))


@dataclass
class MaskRecord:
    """One masklet entry: the mask belongs to the frame with this presentation timestamp."""

    pts_us: int
    frame_idx: int
    rle: RleMask | None            # None while the object has no mask (occluded / not yet seen)
    confidence: float = 1.0
    visibility: str = "visible"    # visible | occluded | outside_view | unknown
    provenance: str = "native"     # native | propagated
    age_frames: int = 0            # frames since the last native mask
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"pts_us": self.pts_us, "frame_idx": self.frame_idx,
                "rle": self.rle.to_dict() if self.rle else None, "confidence": self.confidence,
                "visibility": self.visibility, "provenance": self.provenance,
                "age_frames": self.age_frames, **({"extra": self.extra} if self.extra else {})}

    @classmethod
    def from_dict(cls, d: dict) -> MaskRecord:
        return cls(int(d["pts_us"]), int(d["frame_idx"]), RleMask.from_dict(d["rle"]) if d.get("rle") else None,
                   float(d.get("confidence", 1.0)), d.get("visibility", "visible"),
                   d.get("provenance", "native"), int(d.get("age_frames", 0)), d.get("extra", {}))


def write_track_jsonl(path: str | Path, records: list[MaskRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r.to_dict(), separators=(",", ":")) + "\n" for r in records), encoding="utf-8")


def read_track_jsonl(path: str | Path) -> list[MaskRecord]:
    return [MaskRecord.from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
