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


# --------------------------------------------------------------------------- diagnostics (additive)
@dataclass
class CandidateSet:
    """All candidates of one SAM Audio run. Waveforms are mono float32 at `rate`."""
    targets: list[np.ndarray]
    residuals: list[np.ndarray]
    seeds: list[int]
    rate: int
    prompts: dict
    spans: list | None            # spans predicted by the span predictor (anchors=None, predict_spans=True)
    timings: dict


def load_sam_audio_diagnostic(repo: str = SAM_AUDIO_REPO, device: str = "cuda", vision_batch: int = VISION_BATCH,
                              span_predictor: bool = True, span_predictor_device: str = "cpu") -> SamAudioBundle:
    """Like `load_sam_audio`, but optionally KEEPS the PE-A-Frame span predictor (needed for
    `anchors=None, predict_spans=True`). Rankers stay off (their weights are not in the checkpoint;
    score candidates externally). Diagnostics only: Phase 2 uses `load_sam_audio`.

    The predictor's checkpoint is 6.1 GB (fp32). Measured on the 12 GiB RTX 5070 Ti Laptop it takes
    GPU memory to 11.15 GiB allocated together with SAM Audio small-tv, which leaves no room for
    activations, so it stays on `span_predictor_device` (CPU by default) and only the span prediction
    runs there."""
    from huggingface_hub import snapshot_download
    from sam_audio import SAMAudio, SAMAudioProcessor
    from sam_audio.model.config import SAMAudioConfig

    folder = Path(snapshot_download(repo_id=repo))
    raw = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    cfg = _sanitize_config(raw)
    if span_predictor:
        cfg["span_predictor"] = raw["span_predictor"]
    model = SAMAudio(SAMAudioConfig(**cfg))
    model.load_state_dict(torch.load(folder / "checkpoint.pt", weights_only=True, map_location="cpu"))
    cap_vision_batch(model, vision_batch)
    sp = model._modules.pop("span_predictor", None)          # keep it off the GPU
    model = model.eval().to(device)
    if sp is not None:
        model.span_predictor = sp.eval().to(span_predictor_device)
    return SamAudioBundle(model, SAMAudioProcessor.from_pretrained(str(folder)), f"{repo}@{folder.name}")


def _predict_spans(model, batch, fwd):
    """`SAMAudio.predict_spans` with the span predictor allowed to live on another device."""
    sp = model.span_predictor
    pdev = next(sp.parameters()).device
    inp = model.span_predictor_transform(text=batch.descriptions).to(pdev)
    out = sp(input_features=fwd["audio_features"][:, :, :128].to(pdev), padding_mask=fwd["audio_pad_mask"].to(pdev),
             return_spans=True, **inp)
    batch.process_anchors([[["+"] + a for a in spans] for spans in out.spans])
    return batch


def _mask_tensor(masks, height: int, width: int) -> torch.Tensor:
    """RleMask|None per frame, or a bool tensor/array (T, H, W) / (T, 1, H, W) -> (T, 1, H, W) bool."""
    if torch.is_tensor(masks) or isinstance(masks, np.ndarray):
        t = torch.as_tensor(masks).to(torch.bool)
        return t.unsqueeze(1) if t.ndim == 3 else t
    return masks_to_tensor(masks, height, width)


def build_batch(bundle: SamAudioBundle, *, audio, sample_rate: int, frames: torch.Tensor | None, masks,
                text: str, anchors):
    """The processor call used by `extract_stem` / `extract_candidates`.
    anchors: None | "full_clip" | list[(sign, start_s, end_s)]. Returns (batch, anchor_list, masked_video)."""
    proc = bundle.processor
    wav = resample_audio(audio, sample_rate, proc.audio_sampling_rate)
    duration = wav.shape[-1] / proc.audio_sampling_rate
    masked = None
    if frames is not None:
        if frames.ndim != 4 or len(masks) != frames.shape[0]:
            raise ValueError("need exactly one mask entry (or None) per frame")
        masked = apply_visual_prompt_mask(frames, _mask_tensor(masks, frames.shape[2], frames.shape[3]), proc)
    anchor_list = [("+", 0.0, duration)] if isinstance(anchors, str) and anchors == "full_clip" else anchors
    batch = proc(descriptions=[text], audios=[wav], masked_videos=None if masked is None else [masked],
                 anchors=[anchor_list] if anchor_list else None)
    return batch, anchor_list, masked


def extract_candidates(bundle: SamAudioBundle, *, audio, sample_rate: int, frames: torch.Tensor | None, masks,
                       text: str = DEFAULT_TEXT, anchors="full_clip", predict_spans: bool = False,
                       n_candidates: int = 8, seed: int = 0, seeds: Sequence[int] | None = None,
                       ode_opt: dict | None = None, free_cache: bool = True) -> CandidateSet:
    """Return ALL `n_candidates` (<= 8) candidate waveforms of one prompt, not just one.

    Upstream `separate(reranking_candidates=N)` keeps only the ranker's pick (candidate 0 when no
    ranker is configured, which is the case here), so this re-runs its sampling loop from the same
    encoder inputs. Memory is kept bounded as in Phase 2: vision encoder capped at VISION_BATCH frames
    (`cap_vision_batch`), the encoders run ONCE, then each candidate is an independent B=1 ODE solve +
    decode, run sequentially with the CUDA cache freed in between.

    Noise for candidate k comes from a private generator seeded `seeds[k]` (default seed + k) on the
    model's device, so two runs with the same clip length and `seeds` use IDENTICAL start noise: that is
    the matched-noise pairing for prompt / mask / anchor comparisons. Candidate k equals
    `model.separate(batch, noise=<that noise>, reranking_candidates=1)`.

    frames=None runs without the visual prompt (zero video features). anchors=None with
    predict_spans=True needs a bundle from `load_sam_audio_diagnostic(span_predictor=True)`.
    """
    import time

    from torchdiffeq import odeint
    from sam_audio.model.model import DFLT_ODE_OPT

    if not 1 <= n_candidates <= 8:
        raise ValueError("n_candidates must be in 1..8")
    seed_list = list(seeds) if seeds is not None else [seed + k for k in range(n_candidates)]
    if len(seed_list) != n_candidates:
        raise ValueError("need one seed per candidate")
    t0 = time.perf_counter()
    model, proc = bundle.model, bundle.processor
    device = next(model.parameters()).device
    batch, anchor_list, _ = build_batch(bundle, audio=audio, sample_rate=sample_rate, frames=frames, masks=masks,
                                        text=text, anchors=anchors)
    batch = batch.to(device)
    spans = None
    with torch.inference_mode():
        fwd = model._get_forward_args(batch, candidates=1)
        if predict_spans and anchors is None:
            if not hasattr(model, "span_predictor"):
                raise RuntimeError("predict_spans=True needs a model loaded with the span predictor")
            batch = _predict_spans(model, batch, fwd)
            spans = [list(a) for a in batch.anchors[0]]
            fwd.update({"anchor_ids": batch.anchor_ids, "anchor_alignment": batch.anchor_alignment})
        t_enc = time.perf_counter() - t0
        feats = fwd["audio_features"]
        _, T, C2 = feats.shape
        C = C2 // 2
        sizes = model.audio_codec.feature_idx_to_wav_idx(batch.sizes)
        opts = ode_opt or DFLT_ODE_OPT
        targets, residuals, per_cand = [], [], []
        for s in seed_list:
            tc = time.perf_counter()
            gen = torch.Generator(device=device).manual_seed(int(s))
            noise = torch.randn(feats.shape, generator=gen, device=device, dtype=feats.dtype)

            def vector_field(t, noisy, _f=fwd):
                return model.forward(noisy_audio=noisy, time=t.expand(noisy.size(0)), **_f)

            states = odeint(vector_field, noise, torch.tensor([0.0, 1.0], device=device), **opts)
            gen_f = states[-1].transpose(1, 2)
            wavs = model.audio_codec.decode(gen_f.reshape(2, C, T)).view(1, 2, -1)
            n = int(sizes[0])
            targets.append(wavs[0, 0, :n].float().cpu().numpy().astype(np.float32))
            residuals.append(wavs[0, 1, :n].float().cpu().numpy().astype(np.float32))
            del states, gen_f, wavs, noise
            if free_cache and device.type == "cuda":
                torch.cuda.empty_cache()
            per_cand.append(round(time.perf_counter() - tc, 3))
    prompts = {"text": text, "anchors": anchor_list, "predict_spans": bool(predict_spans), "visual": frames is not None,
               "mask_polarity": mask_polarity(proc),
               "audio_seconds": round(batch.audios.shape[-1] / proc.audio_sampling_rate, 3),
               "frames": None if frames is None else int(frames.shape[0]), "seeds": seed_list}
    return CandidateSet(targets, residuals, seed_list, proc.audio_sampling_rate, prompts, spans,
                        {"encode_s": round(t_enc, 3), "per_candidate_s": per_cand,
                         "total_s": round(time.perf_counter() - t0, 3)})
