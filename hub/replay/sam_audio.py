"""SAM Audio `small-tv` adapter: masked-video + text + full-clip anchors -> one source stem.

MASK POLARITY (verified by hub/tests/av_scene/test_mask_polarity.py against the pinned
sam-audio commit bb4c6999): upstream `Processor.mask_videos` computes `video * mask.eq(0)`.
It KEEPS pixels where the mask is 0 and ZEROES pixels where the mask is 1/True, i.e. it expects
a *removal* mask, the opposite of the foreground=True convention used by Cocktail's RLE masks.
Handing it a foreground mask blacks out the speaker and keeps the bystanders and background.
`apply_visual_prompt_mask` therefore measures the helper's polarity on a tiny known frame
(cached per processor class) and inverts the mask when the helper retains zeros, so the
visual prompt always keeps the target. Never call `processor.mask_videos` directly.

Time alignment: the processor maps tensor video onto audio feature positions uniformly
(`linspace(0, T-1, n_features)`), so `frames` must cover exactly the audio window passed in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

from av_scene.schemas import StemResult
from vision.mask_store import RleMask, decode_mask

SAM_AUDIO_REPO = "facebook/sam-audio-small-tv"
TARGET_RATE = 48_000
DEFAULT_TEXT = "person speaking"  # concise lowercase noun/verb phrase (spec 4.2)

Polarity = Literal["retains_zero", "retains_one"]
_POLARITY_CACHE: dict[type, Polarity] = {}


# --------------------------------------------------------------------------- polarity
def probe_mask_polarity(processor) -> Polarity:
    """Run `processor.mask_videos` on a white frame with a centred True block and report which
    side survives: 'retains_zero' means True pixels were blacked out (the upstream behaviour)."""
    video = torch.ones(2, 3, 8, 8)
    mask = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    mask[:, :, 2:6, 2:6] = True
    out = processor.mask_videos([video], [mask])[0]
    centre, edge = float(out[0, 0, 4, 4]), float(out[0, 0, 0, 0])
    if centre == 1.0 and edge == 0.0:
        return "retains_one"
    if centre == 0.0 and edge == 1.0:
        return "retains_zero"
    raise RuntimeError(f"unrecognised mask_videos behaviour (centre={centre}, edge={edge})")


def mask_polarity(processor) -> Polarity:
    cls = type(processor)
    if cls not in _POLARITY_CACHE:
        _POLARITY_CACHE[cls] = probe_mask_polarity(processor)
    return _POLARITY_CACHE[cls]


def _default_processor():
    from sam_audio import SAMAudioProcessor

    # mask_videos does not use the hop length / rate; no checkpoint access is needed.
    return SAMAudioProcessor(audio_hop_length=1920, audio_sampling_rate=TARGET_RATE)


def apply_visual_prompt_mask(video: torch.Tensor, mask: torch.Tensor, processor=None) -> torch.Tensor:
    """Return `video` with everything outside the foreground `mask` blacked out.

    video: (T, 3, H, W); mask: (T, 1, H, W) or (T, H, W), True = target (foreground).
    The foreground is retained regardless of the upstream helper's polarity.
    """
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError(f"video must be (T, 3, H, W), got {tuple(video.shape)}")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be (T, 1, H, W) or (T, H, W), got {tuple(mask.shape)}")
    if mask.shape[0] != video.shape[0] or mask.shape[2:] != video.shape[2:]:
        raise ValueError(f"mask {tuple(mask.shape)} does not match video {tuple(video.shape)}")
    fg = mask.to(torch.bool)
    proc = processor if processor is not None else _default_processor()
    helper_mask = ~fg if mask_polarity(proc) == "retains_zero" else fg
    return proc.mask_videos([video], [helper_mask])[0]


# --------------------------------------------------------------------------- inputs
def resample_audio(audio, sample_rate: int, target_rate: int = TARGET_RATE) -> torch.Tensor:
    """Mono float32 (1, T) at `target_rate`. Accepts (T,), (C, T) arrays/tensors."""
    wav = torch.as_tensor(np.asarray(audio) if not torch.is_tensor(audio) else audio, dtype=torch.float32)
    if wav.ndim == 1:
        wav = wav[None]
    if wav.ndim != 2:
        raise ValueError(f"audio must be (T,) or (C, T), got {tuple(wav.shape)}")
    wav = wav.mean(0, keepdim=True)
    if sample_rate != target_rate:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sample_rate, target_rate)
    return wav


def masks_to_tensor(masks: Sequence[RleMask | None], height: int, width: int) -> torch.Tensor:
    """Decode per-frame RLE into (T, 1, H, W) bool. A frame with no mask is all background."""
    out = np.zeros((len(masks), 1, height, width), bool)
    for i, rle in enumerate(masks):
        if rle is None:
            continue
        if (rle.height, rle.width) != (height, width):
            raise ValueError(f"frame {i}: mask is {rle.height}x{rle.width}, frames are {height}x{width}")
        out[i, 0] = decode_mask(rle)
    return torch.from_numpy(out)


def _sanitize_config(cfg: dict) -> dict:
    """Drop the optional parts Phase 2 must not load: the PE-A-Frame span predictor (Phase 3
    discovery model, instantiated by SAMAudio.__init__ whenever configured) and the rankers
    (unused with one candidate). The checkpoint holds no weights for them."""
    return {**cfg, "span_predictor": None, "visual_ranker": None, "text_ranker": None}


@dataclass
class SamAudioBundle:
    model: object
    processor: object
    revision: str

    def to(self, device):  # lets ModelLifecycleManager move it off the GPU
        self.model.to(device)
        return self


VISION_BATCH = 8


def cap_vision_batch(model, batch_size: int = VISION_BATCH) -> int:
    """Chunk the visual encoder. Upstream's default is 300 frames per forward pass, and the processor
    feeds one frame per 40 ms of audio (an 8 s clip = 200 frames through PE-Core-L14-336 at once). That
    needs far more than a 12 GiB card and was the prime suspect for a hypervisor bugcheck (0x20001) on
    the dev laptop; it also raised a CUDA OOM once capped. Frames are encoded independently, so
    chunking changes memory only, not the features."""
    model.vision_encoder.batch_size = int(batch_size)
    return model.vision_encoder.batch_size


def load_sam_audio(repo: str = SAM_AUDIO_REPO, device: str = "cuda", vision_batch: int = VISION_BATCH) -> SamAudioBundle:
    """Load model + processor. Downloads the checkpoint on first use (explicit action)."""
    from huggingface_hub import snapshot_download
    from sam_audio import SAMAudio, SAMAudioProcessor
    from sam_audio.model.config import SAMAudioConfig

    folder = Path(snapshot_download(repo_id=repo))
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = SAMAudio(SAMAudioConfig(**_sanitize_config(cfg)))
    model.load_state_dict(torch.load(folder / "checkpoint.pt", weights_only=True, map_location="cpu"))
    cap_vision_batch(model, vision_batch)
    model = model.eval().to(device)
    return SamAudioBundle(model, SAMAudioProcessor.from_pretrained(str(folder)), f"{repo}@{folder.name}")


# --------------------------------------------------------------------------- inference
def _first_waveform(x) -> np.ndarray:
    """SeparationResult.target is a list of per-item tensors despite its annotation; accept both."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    arr = x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, np.float32)
    return arr.reshape(-1, arr.shape[-1]).mean(0).astype(np.float32)


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def extract_stem(bundle: SamAudioBundle, *, audio, sample_rate: int, frames: torch.Tensor,
                 masks: Sequence[RleMask | None], source_id: str, out_dir: str | Path,
                 text: str = DEFAULT_TEXT, anchors: list[tuple[str, float, float]] | Literal["full_clip"] | None = "full_clip",
                 seed: int | None = 0) -> tuple[StemResult, dict]:
    """Separate one source. Returns the StemResult and a log of the prompts that reached the model.

    frames: (T, 3, H, W) uint8 covering exactly the audio window; masks: one RleMask|None per frame.
    """
    from ingest.media_io import write_wav

    proc, model = bundle.processor, bundle.model
    rate = proc.audio_sampling_rate
    wav = resample_audio(audio, sample_rate, rate)
    duration = wav.shape[-1] / rate
    if frames.ndim != 4 or len(masks) != frames.shape[0]:
        raise ValueError("need exactly one mask entry (or None) per frame")
    mask_t = masks_to_tensor(masks, frames.shape[2], frames.shape[3])
    masked = apply_visual_prompt_mask(frames, mask_t, proc)

    anchor_list = [("+", 0.0, duration)] if anchors == "full_clip" else anchors
    batch = proc(descriptions=[text], audios=[wav], masked_videos=[masked],
                 anchors=[anchor_list] if anchor_list else None)
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)  # separate() draws its ODE start noise from the global RNG: unseeded runs differ
    with torch.inference_mode():
        result = model.separate(batch.to(device), reranking_candidates=1, predict_spans=False)
    stem = _first_waveform(result.target)

    errors = [] if np.isfinite(stem).all() else ["non-finite samples in output"]
    out_dir = Path(out_dir)
    rel = f"stems/source_{source_id}.wav"
    write_wav(out_dir / rel, np.nan_to_num(stem), rate)
    prompts = {"text": text, "anchors": anchor_list, "visual": True, "mask_polarity": mask_polarity(proc),
               "frames": int(frames.shape[0]), "audio_seconds": round(duration, 3), "seed": seed}
    res = StemResult(source_id=source_id, model_revision=bundle.revision, prompt_hash=_hash([text]),
                     anchor_hash=_hash(anchor_list), sample_rate=rate, n_samples=int(stem.shape[-1]),
                     raw_path=rel, errors=errors)
    return res, prompts
