"""pyannote/speaker-diarization-community-1 adapter (import-safe without pyannote).

Pyannote and its gated weights are loaded lazily by `load_pipeline`. Everything else is pure and
feeds `separate.diarization.compute_anchors`.

Overlapped speech: turns of different speakers may overlap in time. That is intentional and is
kept as-is; `compute_anchors` turns the overlap into competing (negative) spans for each speaker
and excludes it from each speaker's solo (positive) spans.
"""

from __future__ import annotations

from separate.diarization import Interval, normalize

MODEL_ID = "pyannote/speaker-diarization-community-1"
Turn = tuple[str, float, float]  # (speaker, start_s, end_s)


def load_pipeline(token: str | None = None):
    """Lazily import pyannote and load the pipeline. Raises RuntimeError with guidance on failure."""
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Install it in a project-local environment and note "
            f"that {MODEL_ID} is gated: accept its terms on Hugging Face and provide a read token "
            "(argument or HF_TOKEN). Neither is done automatically.") from exc
    try:
        pipe = Pipeline.from_pretrained(MODEL_ID, token=token) if token else Pipeline.from_pretrained(MODEL_ID)
    except Exception as exc:  # gated / offline / bad token
        raise RuntimeError(f"Could not load {MODEL_ID}: accept the model terms on Hugging Face and "
                           f"supply a valid token ({exc})") from exc
    if pipe is None:
        raise RuntimeError(f"{MODEL_ID} returned no pipeline: terms not accepted or token invalid")
    return pipe


def diarize(audio, sr: int, pipeline) -> list[Turn]:
    """Run `pipeline` on mono float audio (1-D array) and return [(speaker, start_s, end_s)].

    Turns of different speakers can overlap (overlapped speech). Sorted by start time.
    """
    import numpy as np
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=0 if wav.shape[0] <= 8 else 1)
    try:
        import torch  # type: ignore
        data = {"waveform": torch.from_numpy(wav)[None, :], "sample_rate": int(sr)}
    except ImportError:
        data = {"waveform": wav[None, :], "sample_rate": int(sr)}
    out = pipeline(data)
    ann = getattr(out, "speaker_diarization", out)  # community-1 wraps; 3.x returns Annotation
    return sorted(((str(spk), float(seg.start), float(seg.end))
                   for seg, _, spk in ann.itertracks(yield_label=True)), key=lambda t: (t[1], t[2], t[0]))


def turns_to_timeline(turns) -> dict[str, list[Interval]]:
    """Group turns by speaker into normalized microsecond intervals (same speaker merged)."""
    raw: dict[str, list[Interval]] = {}
    for spk, s, e in turns:
        raw.setdefault(str(spk), []).append((int(round(s * 1e6)), int(round(e * 1e6))))
    return {k: normalize(v) for k, v in raw.items()}


def turns_to_anchor_inputs(turns, noise=()) -> tuple[dict[str, list[Interval]], list[Interval]]:
    """Return `(activities, noise)` exactly as `compute_anchors(activities, noise, cfg)` expects.

    `compute_anchors` derives solo '+' = A_i minus other speakers/noise and competing '-' =
    other speakers minus A_i, so overlap needs no special handling here. Min-duration filtering is
    done by compute_anchors via AnchorConfig; this function does not pre-filter.
    """
    return turns_to_timeline(turns), normalize(noise)


class FakePipeline:
    """Test double: returns canned turns; mimics Annotation.itertracks."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = 0

    def __call__(self, data):
        self.calls += 1
        turns = self._turns

        class _Seg:
            def __init__(self, s, e):
                self.start, self.end = s, e

        class _Ann:
            def itertracks(self, yield_label=False):
                for i, (spk, s, e) in enumerate(turns):
                    yield _Seg(s, e), i, spk

        return _Ann()
