"""TalkNet active-speaker scoring adapter plus the shared face-crop / 25 fps helpers.

In-memory only: no ffmpeg, no files, no ClearVoice file pipeline. Geometry follows TalkNet's
`crop_video` (bs = half box size, cs = 0.40, constant pad 110), then gray, resize 224, centre
crop 112x112. TalkNet normalises (x/255 - 0.4161)/0.1688 *inside* its visual frontend, so this
module feeds raw 0-255 gray crops. AV-MossFormer2 does NOT normalise internally; its adapter
applies the normalisation itself.

Paths are configurable by argument or env var (COCKTAIL_TALKNET_SRC, COCKTAIL_TALKNET_WEIGHTS).
TalkNet's own `talkNet.__init__` hard-codes `.cuda()` and torch.load without map_location, so
this builds `talkNetModel` + `lossAV` directly and loads the state dict with map_location.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Sequence

import numpy as np

CROP_SCALE = 0.40
VIDEO_FPS = 25.0
PAD_VALUE = 110
DEFAULT_SRC = r"C:\Users\nguye\repos\Cocktail\hub\checkpoints\talknet\src"
DEFAULT_WEIGHTS = r"C:\Users\nguye\repos\Cocktail\hub\checkpoints\talknet\pretrain_TalkSet.model"
DURATIONS = (1, 2, 3, 4, 5, 6)


def resample_indices(n_src: int, src_fps: float, dst_fps: float = VIDEO_FPS) -> np.ndarray:
    """Nearest-timestamp source index for each output frame at `dst_fps` (same duration)."""
    if n_src <= 0:
        return np.zeros(0, dtype=np.int64)
    n_dst = max(1, int(round(n_src / src_fps * dst_fps)))
    t = np.arange(n_dst) / dst_fps
    return np.clip(np.rint(t * src_fps), 0, n_src - 1).astype(np.int64)


def hold_boxes(boxes: Sequence, n: int | None = None) -> tuple[np.ndarray, float]:
    """Fill None boxes by holding the last box (leading gaps take the first real box).

    Returns ([N,4] float array, fraction of frames that were filled).
    Raises ValueError when no frame has a box.
    """
    n = len(boxes) if n is None else n
    real = [i for i, b in enumerate(boxes[:n]) if b is not None]
    if not real:
        raise ValueError("no face box in any frame")
    out = np.zeros((n, 4), dtype=np.float64)
    last = np.asarray(boxes[real[0]], dtype=np.float64)
    filled = 0
    for i in range(n):
        b = boxes[i] if i < len(boxes) else None
        if b is None:
            filled += 1
        else:
            last = np.asarray(b, dtype=np.float64)
        out[i] = last
    return out, filled / n


def crop_face(frame_rgb: np.ndarray, box: Sequence[float], out_size: int = 224,
              crop_scale: float = CROP_SCALE) -> np.ndarray:
    """TalkNet/ClearVoice crop of one RGB uint8 frame -> uint8 gray [112,112]."""
    import cv2

    x0, y0, x1, y1 = (float(v) for v in box)
    bs = max(max(x1 - x0, y1 - y0) / 2.0, 1.0)
    cs = crop_scale
    bsi = int(bs * (1 + 2 * cs))
    padded = np.pad(frame_rgb, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=PAD_VALUE)
    my = (y0 + y1) / 2.0 + bsi
    mx = (x0 + x1) / 2.0 + bsi
    face = padded[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (out_size, out_size))
    h = out_size // 2
    return gray[h - 56:h + 56, h - 56:h + 56]


def build_face_stack(frames: np.ndarray, face_boxes: Sequence, fps: float) -> tuple[np.ndarray, dict]:
    """Resample to 25 fps, hold boxes, crop. Returns (uint8 [Tv,112,112], info dict)."""
    idx = resample_indices(len(frames), fps)
    boxes25 = [face_boxes[i] if i < len(face_boxes) else None for i in idx]
    boxes, filled = hold_boxes(boxes25)
    stack = np.stack([crop_face(frames[i], b) for i, b in zip(idx, boxes)])
    return stack, {"box_filled_fraction": filled, "box_coverage": 1.0 - filled,
                   "n_video_frames": len(stack)}


def smooth5(scores: np.ndarray) -> np.ndarray:
    """Demo smoothing: mean over score[max(i-2,0):min(i+3,len-1)] (faithful to demoTalkNet)."""
    n = len(scores)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = scores[max(i - 2, 0):min(i + 3, n - 1)]
        out[i] = float(np.mean(seg)) if len(seg) else float(scores[i])
    return out


def load_talknet(src_dir: str | None = None, weights: str | None = None, device: str | None = None):
    """Build TalkNet (model + lossAV) on `device`, load weights. Returns (model, lossAV, device)."""
    import torch

    src = str(src_dir or os.environ.get("COCKTAIL_TALKNET_SRC", DEFAULT_SRC))
    wts = str(weights or os.environ.get("COCKTAIL_TALKNET_WEIGHTS", DEFAULT_WEIGHTS))
    if src not in sys.path:
        sys.path.insert(0, src)
    from loss import lossAV  # type: ignore  # TalkNet src
    from model.talkNetModel import talkNetModel  # type: ignore

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, loss = talkNetModel().to(dev), lossAV().to(dev)
    state = torch.load(wts, map_location=dev)
    own = {**{"model." + k: v for k, v in model.state_dict().items()},
           **{"lossAV." + k: v for k, v in loss.state_dict().items()}}
    loaded = 0
    for name, param in state.items():
        n = name if name in own else name.replace("module.", "")
        if n in own and own[n].shape == param.shape:
            own[n].copy_(param)
            loaded += 1
    if loaded == 0:
        raise RuntimeError(f"no TalkNet parameters matched in {wts}")
    model.eval()
    loss.eval()
    return model, loss, dev


class TalkNetScorer:
    def __init__(self, src_dir: str | None = None, weights: str | None = None,
                 device: str | None = None):
        self.model, self.loss, self.device = load_talknet(src_dir, weights, device)

    def score_track(self, audio_16k: np.ndarray, face_boxes: Sequence, frames: np.ndarray,
                    fps: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """Per-25fps-frame speaking scores. Returns (scores [Tv], timestamps_s [Tv], info).

        `audio_16k`: float mono in [-1, 1] (scaled to int16 range for MFCC, as the demo's
        wavfile.read does). Scores are raw lossAV logits for class 1 (what the demo averages);
        > 0 means speaking.
        """
        import torch
        from python_speech_features import mfcc

        audio = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        feat = mfcc(audio * 32768.0, 16000, numcep=13, winlen=0.025, winstep=0.010)
        stack, info = build_face_stack(frames, face_boxes, fps)
        length = min((feat.shape[0] - feat.shape[0] % 4) / 100, stack.shape[0] / VIDEO_FPS)
        feat = feat[:int(round(length * 100))]
        stack = stack[:int(round(length * 25))]
        runs = []
        with torch.no_grad():
            for d in DURATIONS:
                sc: list[float] = []
                for i in range(int(math.ceil(length / d))):
                    a = torch.as_tensor(feat[i * d * 100:(i + 1) * d * 100], dtype=torch.float32)
                    v = torch.as_tensor(stack[i * d * 25:(i + 1) * d * 25], dtype=torch.float32)
                    if len(a) == 0 or len(v) == 0:
                        continue
                    a, v = a.unsqueeze(0).to(self.device), v.unsqueeze(0).to(self.device)
                    m = self.model
                    ea, ev = m.forward_audio_frontend(a), m.forward_visual_frontend(v)
                    ea, ev = m.forward_cross_attention(ea, ev)
                    out = m.forward_audio_visual_backend(ea, ev)
                    sc.extend(self.loss.forward(out, labels=None).tolist())
                runs.append(sc)
        n = min(len(r) for r in runs)
        scores = smooth5(np.mean(np.array([r[:n] for r in runs]), axis=0))
        info.update(n_scores=int(n), length_s=float(length))
        return scores, np.arange(n) / VIDEO_FPS, info
