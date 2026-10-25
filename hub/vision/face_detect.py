"""S3FD face detection, assigned to SAM 3 person tracks.

TalkNet and AV-TSE need a face crop. A SAM 3 "person" mask is a body; cropping its top is wrong for
anything but upright head-and-shoulders shots. This runs S3FD (the detector TalkNet's and
ClearVoice's demos use) per frame and gives each track the face whose centre lies inside its mask.
A frame with no such face is None (no evidence), never a guessed box.

Weights: hub/checkpoints/talknet/sfd_face.pth (from the TalkNet-ASD demo, MIT), overridable with
COCKTAIL_S3FD_WEIGHTS. The TalkNet package's s3fd/__init__.py downloads weights at import time, so
only its nets.py and box_utils.py are loaded here, by file path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .talknet import DEFAULT_SRC

_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "checkpoints" / "talknet" / "sfd_face.pth"
_MEAN = np.array([123.0, 117.0, 104.0], np.float32)       # RGB order of the demo's BGR mean, net sees RGB


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class FaceDetector:
    def __init__(self, device: str = "cuda", weights: str | None = None, src_dir: str | None = None):
        import torch

        d = Path(src_dir or os.environ.get("COCKTAIL_TALKNET_SRC", DEFAULT_SRC)) / "model" / "faceDetector" / "s3fd"
        _load_module("s3fd_box_utils", d / "box_utils.py")
        src = (d / "nets.py").read_text(encoding="utf-8").replace("from .box_utils", "from s3fd_box_utils")
        nets = type(sys)("s3fd_nets")
        exec(compile(src, str(d / "nets.py"), "exec"), nets.__dict__)
        self.device = device
        self.net = nets.S3FDNet(device=device).to(device).eval()
        w = weights or os.environ.get("COCKTAIL_S3FD_WEIGHTS") or str(_DEFAULT_WEIGHTS)
        self.net.load_state_dict(torch.load(w, map_location=device))

    def detect(self, rgb: np.ndarray, conf_th: float = 0.8, scale: float = 0.5) -> np.ndarray:
        """rgb uint8 [H,W,3] -> [k,5] boxes (x0,y0,x1,y1,score) in pixels."""
        import cv2
        import torch

        h, w = rgb.shape[:2]
        img = cv2.resize(rgb, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x = torch.from_numpy((img - _MEAN).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            det = self.net(x).data
        boxes = []
        sc = torch.tensor([w, h, w, h], dtype=det.dtype)
        for i in range(det.size(1)):
            j = 0
            while j < det.size(2) and det[0, i, j, 0] > conf_th:
                pt = (det[0, i, j, 1:].cpu() * sc).numpy()
                boxes.append((*pt, float(det[0, i, j, 0])))
                j += 1
        if not boxes:
            return np.empty((0, 5))
        b = np.array(boxes)
        return b[_nms(b, 0.1)]


def _nms(b: np.ndarray, thr: float) -> np.ndarray:
    """Greedy IoU NMS (upstream nms_ uses the removed np.int alias)."""
    order, keep = np.argsort(-b[:, 4]), []
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx0, yy0 = np.maximum(b[i, 0], b[order[1:], 0]), np.maximum(b[i, 1], b[order[1:], 1])
        xx1, yy1 = np.minimum(b[i, 2], b[order[1:], 2]), np.minimum(b[i, 3], b[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter) <= thr]
    return np.array(keep, dtype=int)


def assign_faces(faces: np.ndarray, masks: dict[str, np.ndarray | None]) -> dict[str, tuple | None]:
    """Give each track the highest-scoring face whose centre is inside its mask; a face goes to one track."""
    out: dict[str, tuple | None] = {t: None for t in masks}
    used: set[int] = set()
    for k in np.argsort(-faces[:, 4]) if len(faces) else []:
        cx, cy = (faces[k, 0] + faces[k, 2]) / 2, (faces[k, 1] + faces[k, 3]) / 2
        for t, m in masks.items():
            if out[t] is not None or m is None or k in used:
                continue
            yi, xi = int(np.clip(cy, 0, m.shape[0] - 1)), int(np.clip(cx, 0, m.shape[1] - 1))
            if m[yi, xi]:
                out[t] = tuple(float(v) for v in faces[k, :4])
                used.add(k)
    return out


def face_boxes_by_track(detector: FaceDetector, frames: np.ndarray,
                        masks_by_track: dict[str, Sequence[np.ndarray | None]]) -> dict[str, list]:
    """frames uint8 [N,H,W,3]; masks_by_track: per-frame bool mask or None -> per-frame face box or None."""
    res: dict[str, list] = {t: [] for t in masks_by_track}
    for i in range(len(frames)):
        a = assign_faces(detector.detect(frames[i]), {t: ms[i] for t, ms in masks_by_track.items()})
        for t in res:
            res[t].append(a[t])
    return res
