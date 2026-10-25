"""AV-MossFormer2 target-speaker extraction (16 kHz) as an in-memory adapter.

Loads the ClearVoice network directly (not `ClearVoice.process_tse`, whose file pipeline shells
out to rm/ffmpeg and fails on Windows) and calls `decode_one_audio_AV_MossFormer2_TSE_16K`.
CUDA only: clearvoice av_mossformer2.py hard-codes `.cuda()` in its overlap-add.

Checkpoint path is configurable via the `checkpoint` argument or COCKTAIL_AV_TSE_CKPT.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from vision.talknet import VIDEO_FPS, build_face_stack

DEFAULT_CKPT = (r"C:\Users\nguye\repos\Cocktail\hub\checkpoints\AV_MossFormer2_TSE_16K"
                r"\last_best_checkpoint.pt")
SAMPLE_RATE = 16000
VIS_MEAN, VIS_STD = 0.4161, 0.1688


def normalise_visual(stack_u8: np.ndarray) -> np.ndarray:
    """uint8 gray [Tv,112,112] -> float32 [1,Tv,112,112] normalised as ClearVoice does."""
    v = stack_u8.astype(np.float32) / 255.0
    return ((v - VIS_MEAN) / VIS_STD)[None].astype(np.float32)


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) for k, v in d.items()}) if isinstance(d, dict) else d


def load_args(device: str = "cuda"):
    import clearvoice
    import yaml

    cfg = Path(clearvoice.__file__).parent / "config" / "inference" / "AV_MossFormer2_TSE_16K.yaml"
    args = _ns(yaml.safe_load(cfg.read_text(encoding="utf-8")))
    args.device = device
    return args


def load_model(checkpoint: str | None = None, device: str = "cuda"):
    """Build av_mossformer2 and load weights (same key-matching as clearvoice `_load_model`)."""
    import torch
    from clearvoice.models.av_mossformer2_tse.av_mossformer2 import AV_MossFormer2_TSE_16K

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("AV-MossFormer2 TSE needs CUDA (clearvoice hard-codes .cuda())")
    ckpt_path = checkpoint or os.environ.get("COCKTAIL_AV_TSE_CKPT", DEFAULT_CKPT)
    args = load_args(device)
    model = AV_MossFormer2_TSE_16K(args).model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pre = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = model.state_dict()
    matched = 0
    for k in state:
        for cand in (k, k.replace("module.", ""), "module." + k):
            if cand in pre and pre[cand].shape == state[k].shape:
                state[k] = pre[cand]
                matched += 1
                break
    if matched < 0.9 * len(state):
        raise RuntimeError(f"only {matched}/{len(state)} AV-TSE tensors matched in {ckpt_path}")
    model.load_state_dict(state)
    return model.to(device).eval(), args


class AVTargetSpeakerExtractor:
    def __init__(self, checkpoint: str | None = None, device: str = "cuda"):
        self.model, self.args = load_model(checkpoint, device)

    def extract_target(self, audio_16k: np.ndarray, face_boxes: Sequence, frames: np.ndarray,
                       fps: float) -> tuple[np.ndarray, dict]:
        """Extract the speaker whose face track is given.

        audio_16k: mono float in [-1, 1]; frames: uint8 RGB [N,H,W,3] at `fps`;
        face_boxes: per-frame (x0,y0,x1,y1) or None. Returns (1-D float32 [T] at 16 kHz,
        info with box_coverage / box_filled_fraction).
        """
        import torch
        from clearvoice.utils.decode import decode_one_audio_AV_MossFormer2_TSE_16K

        audio = np.asarray(audio_16k, dtype=np.float32).reshape(1, -1).copy()
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio /= peak  # peak-normalise, as ClearVoice does (output is not rescaled)
        stack, info = build_face_stack(frames, face_boxes, fps)
        visual = normalise_visual(stack)
        need = int(np.ceil(audio.shape[1] / SAMPLE_RATE * VIDEO_FPS))
        if visual.shape[1] < need:  # pad with edge, as ClearVoice evaluate_network does
            visual = np.pad(visual, ((0, 0), (0, need - visual.shape[1]), (0, 0), (0, 0)), mode="edge")
        visual = visual[:, :need]
        with torch.no_grad():
            out = decode_one_audio_AV_MossFormer2_TSE_16K(self.model, (audio, visual), self.args)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        info.update(peak_in=peak, n_samples=int(out.shape[0]))
        return out, info
