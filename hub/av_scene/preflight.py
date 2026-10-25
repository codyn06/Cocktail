"""Phase 2 capability report: CUDA, TorchCodec decode, FFmpeg shared DLLs, HF gated access.

`python -m av_scene preflight --av [--media clip.mp4]`. Each capability is reported on its
own with status ok | fail | skipped; nothing is downloaded and no token is ever printed
(only whether one exists). Exit code is 1 when a required capability fails.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import os
import sys
import tempfile
from pathlib import Path

GATED_REPOS = ("facebook/sam3", "facebook/sam-audio-small-tv")
PACKAGES = ("torch", "torchvision", "torchcodec", "sam3", "sam-audio", "triton-windows", "transformers")
HUB = Path(__file__).resolve().parents[1]
DEFAULT_FFMPEG_BIN = HUB / ".local" / "ffmpeg" / "bin"

_dll_dirs: list = []


def configure_ffmpeg_dlls() -> dict:
    """Make FFmpeg 'full-shared' DLLs visible to torchcodec (Windows). Call before first decode.

    Uses COCKTAIL_FFMPEG_BIN, else hub/.local/ffmpeg/bin if present. Static ffmpeg.exe
    builds do not satisfy torchcodec: it needs avcodec/avformat/avutil .dll files.
    """
    raw = os.environ.get("COCKTAIL_FFMPEG_BIN")
    bin_dir = Path(raw) if raw else (DEFAULT_FFMPEG_BIN if DEFAULT_FFMPEG_BIN.is_dir() else None)
    info = {"env": "COCKTAIL_FFMPEG_BIN" if raw else None, "dir": str(bin_dir) if bin_dir else None}
    if bin_dir is None:
        info["status"] = "not configured"
        return info
    if not bin_dir.is_dir():
        info["status"] = "directory missing"
        return info
    dlls = sorted(p.name for p in bin_dir.glob("avcodec*.dll")) + sorted(p.name for p in bin_dir.glob("avformat*.dll"))
    info["dlls"] = dlls
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        _dll_dirs.append(os.add_dll_directory(str(bin_dir)))  # kept alive for the process
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    info["status"] = "registered" if dlls else "registered but no avcodec/avformat DLLs found"
    return info


def _git_sha(dist: str) -> str | None:
    try:
        raw = md.distribution(dist).read_text("direct_url.json")
        return json.loads(raw)["vcs_info"]["commit_id"] if raw else None
    except Exception:
        return None


def check_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = {"version": md.version(name), "git": _git_sha(name)}
        except md.PackageNotFoundError:
            out[name] = None
    return {"status": "ok" if out["sam3"] and out["sam-audio"] and out["torchcodec"] else "fail", "packages": out}


def check_cuda(min_free_gib: float = 8.0) -> dict:
    try:
        import torch
    except Exception as e:
        return {"status": "fail", "error": f"torch import failed: {type(e).__name__}"}
    if not torch.cuda.is_available():
        return {"status": "fail", "error": "torch.cuda.is_available() is False", "torch": torch.__version__}
    free, total = torch.cuda.mem_get_info()
    p = torch.cuda.get_device_properties(0)
    gib = 1024 ** 3
    out = {"status": "ok", "torch": torch.__version__, "cuda": torch.version.cuda, "device": p.name,
           "capability": f"{p.major}.{p.minor}", "total_gib": round(total / gib, 2), "free_gib": round(free / gib, 2)}
    if free / gib < min_free_gib:
        out.update(status="fail", error=f"only {free / gib:.1f} GiB free, need ~{min_free_gib} (another job on the GPU?)")
    return out


def _synth_clip(path: Path) -> bool:
    """Tiny 1 s clip with AAC audio via PyAV, only if PyAV is importable."""
    try:
        import av
        import numpy as np
    except Exception:
        return False
    c = av.open(str(path), "w")
    v = c.add_stream("libx264", rate=25)
    v.width, v.height, v.pix_fmt = 128, 96, "yuv420p"
    a = c.add_stream("aac", rate=48000)
    a.layout = "mono"
    for i in range(25):
        for p in v.encode(av.VideoFrame.from_ndarray(np.full((96, 128, 3), i * 8, np.uint8), format="rgb24")):
            c.mux(p)
    for p in v.encode():
        c.mux(p)
    t = (np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 0.3).astype(np.float32)
    for s in range(0, 48000, 1024):
        fr = av.AudioFrame.from_ndarray(t[None, s:s + 1024], format="fltp", layout="mono")
        fr.sample_rate, fr.pts = 48000, s
        for p in a.encode(fr):
            c.mux(p)
    for p in a.encode():
        c.mux(p)
    c.close()
    return True


def check_decode(media: str | None = None) -> dict:
    ffmpeg = configure_ffmpeg_dlls()
    tmp = None
    if media is None:
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # Windows: decoder may still hold the file
        media = str(Path(tmp.name) / "preflight.mp4")
        if not _synth_clip(Path(media)):
            return {"status": "skipped", "ffmpeg_dlls": ffmpeg, "note": "no --media given and PyAV unavailable to synthesize one"}
    out: dict = {"ffmpeg_dlls": ffmpeg, "media": "synthetic" if tmp else str(media)}
    try:
        from torchcodec.decoders import AudioDecoder, VideoDecoder

        vd = VideoDecoder(media)
        frame = vd[0]
        out["video"] = {"frames": vd.metadata.num_frames, "fps": vd.metadata.average_fps, "frame_shape": list(frame.shape)}
        ad = AudioDecoder(media)
        samples = ad.get_all_samples()
        out["audio"] = {"sample_rate": samples.sample_rate, "channels": int(samples.data.shape[0]), "samples": int(samples.data.shape[1])}
        out["status"] = "ok"
    except Exception as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else ""
        out.update(status="fail", error=f"{type(e).__name__}: {first[:200]}",
                   hint="torchcodec needs FFmpeg 6/7 *shared* DLLs (avcodec/avformat/avutil). Download an "
                        "ffmpeg 'full-shared' build and set COCKTAIL_FFMPEG_BIN to its bin/ directory "
                        "(or unpack it to hub/.local/ffmpeg).")
    finally:
        if tmp:
            tmp.cleanup()
    return out


def check_hf_access(repos=GATED_REPOS) -> dict:
    """Gated-repo access via existing credentials only. Never prints or returns the token."""
    try:
        from huggingface_hub import HfApi, get_token
    except Exception as e:
        return {"status": "fail", "error": f"huggingface_hub unavailable: {type(e).__name__}"}
    has_token = bool(get_token())
    out: dict = {"token_present": has_token, "repos": {}}
    if not has_token:
        out.update(status="fail", error="no Hugging Face credentials found",
                   hint="run `hf auth login` (or set HF_TOKEN), then request access on each model page: "
                        + ", ".join(f"https://huggingface.co/{r}" for r in repos))
        return out
    api = HfApi()
    for repo in repos:
        try:
            api.auth_check(repo_id=repo, repo_type="model")
            out["repos"][repo] = "ok"
        except Exception as e:
            kind = type(e).__name__
            out["repos"][repo] = ("access not granted (gated): request it on the model page" if kind == "GatedRepoError"
                                  else f"{kind}: not found, or no access with this token" if kind == "RepositoryNotFoundError"
                                  else f"{kind}")
    out["status"] = "ok" if all(v == "ok" for v in out["repos"].values()) else "fail"
    return out


REQUIRED = ("versions", "cuda", "decode", "hf_access")


def check_optional(online: bool = False) -> dict:
    """Optional Phase 4 capabilities. None of these is REQUIRED: a missing one is `not_attempted` (an intentional
    skip), distinct from a probe that crashed (`fail`). No credentials are printed, nothing is downloaded, and
    there is no network call unless ``online`` is set (gated access is then probed metadata-only)."""
    out: dict = {}

    def probe(name, fn):
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {"status": "fail", "error": f"{type(e).__name__}: {str(e)[:200]}"}

    from . import diar_windows
    probe("diarization", lambda: diar_windows.availability(online=online))

    def wavlm():
        from separate.speaker_embed import wavlm_sv_status
        st = wavlm_sv_status()
        return {**st, "status": "ok" if st.get("ready") else "not_attempted"}
    probe("wavlm_sv", wavlm)

    def judge():
        from replay.judge import availability
        a = availability()
        return {**a, "status": "ok" if any(v.get("ok") for v in a.values() if isinstance(v, dict)) else "not_attempted"}
    probe("judge_clap", judge)
    return out


def run_av_checks(media: str | None = None, *, checks: dict | None = None) -> dict:
    """`checks` lets tests inject fakes: name -> zero-arg callable."""
    table = checks or {
        "versions": check_versions, "cuda": check_cuda,
        "decode": lambda: check_decode(media), "hf_access": check_hf_access,
    }
    report = {}
    for name in REQUIRED:
        try:
            report[name] = table[name]()
        except Exception as e:  # a crashing probe is a failed capability, not a crashed report
            report[name] = {"status": "fail", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    report["ok"] = all(report[n]["status"] == "ok" for n in REQUIRED)
    if checks is None:                                   # injected fakes stay hermetic
        report["optional"] = check_optional()
    return report
