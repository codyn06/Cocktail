"""Phase 4 shared contracts for chunked long-clip processing. Pure Python/numpy; no models.

Time
  * Canonical audio time is an integer SAMPLE INDEX at the stream's own rate (the original multichannel
    master keeps its capture rate). Microseconds are derived: ``us = round(sample * 1e6 / rate)`` (half up,
    integer arithmetic), and ``sample = round(us * rate / 1e6)``. 16 kHz does not divide 1e6 evenly
    (1 sample = 62.5 us), so conversions are NOT a bijection; never round-trip a boundary through
    microseconds. Chunk boundaries are defined in samples at SPEECH_SR only.
  * Every interval is half-open: ``[start, end)``.
  * Video frames keep their own pts (integer us on the common clock, ``StreamTransform`` applied);
    a chunk selects frames with ``start_us <= pts < end_us`` (`video_index_range`). Nothing derives a
    timestamp from ``frame_idx / fps``.
  * The mapping used between video timestamps and audio samples is recorded in ``AvMap`` and written to the
    manifest's long_run section.

Chunks
  * A chunk has an INFERENCE window ``[win_start, win_end)`` and an OWNED region ``[own_start, own_end)``.
    Owned regions tile ``[0, n_samples)`` exactly (no gap, no duplicate), including a final partial region.
    Consecutive windows overlap by ``overlap_s``; the boundary between chunks k and k+1 is the MIDPOINT of
    their overlap, so every owned sample sits at least ``overlap_s / 2`` from any window edge that is not a
    clip edge. Activity and enrollment durations are counted on OWNED samples only (never on the overlap).

Identity layers (never interchangeable)
  * slot        chunk-local separator output index (``SlotRef``); meaningless across chunks (permutation).
  * source_id   persistent ACOUSTIC identity, ``src_<n>``, assigned by the identity tracker.
  * track_id    VISUAL track identity from the person tracker ("sam3_image:...").
  Associations link a source_id to a track_id per owned span and may be absent, ambiguous or refused.

Status (kept as four independent fields, see schemas.TimelineSegment)
  association_status, visibility, stem_availability, refusal_reason.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SPEECH_SR = 16_000

# Sample rate each adapter REQUIRES. The multichannel master is never replaced by these views.
ADAPTER_RATES_HZ: dict[str, int] = {
    "s3fd": 0,                  # video only
    "talknet": 16_000,
    "mossformer2_ss": 16_000,
    "av_mossformer2_tse": 16_000,
    "ast": 16_000,
    "wavlm_sv": 16_000,
    "pyannote": 16_000,
    "sam_audio": 48_000,
    "clap_judge": 48_000,
}


# ----------------------------------------------------------------------------------------- time
def sample_to_us(sample: int, rate_hz: int) -> int:
    return (int(sample) * 1_000_000 + rate_hz // 2) // rate_hz


def us_to_sample(us: int, rate_hz: int) -> int:
    return (int(us) * rate_hz + 500_000) // 1_000_000


def video_index_range(pts_us: list[int], start_us: int, end_us: int) -> tuple[int, int]:
    """Frame index range ``[i0, i1)`` with ``start_us <= pts < end_us`` (pts sorted ascending)."""
    return bisect.bisect_left(pts_us, start_us), bisect.bisect_left(pts_us, end_us)


@dataclass(frozen=True)
class AvMap:
    """Recorded audio<->video mapping on the common clock (us). ``audio_start_us`` is the common-clock time of
    audio sample 0; video pts are already on the common clock (their StreamTransform applied at ingest)."""

    audio_rate_hz: int
    audio_start_us: int = 0
    video_transform: str = "identity"         # StreamTransform.method that produced the video pts
    clock_offset_confidence: float = 1.0

    def sample_to_common_us(self, sample: int) -> int:
        return self.audio_start_us + sample_to_us(sample, self.audio_rate_hz)

    def common_us_to_sample(self, us: int) -> int:
        return us_to_sample(us - self.audio_start_us, self.audio_rate_hz)


# --------------------------------------------------------------------------------------- chunks
@dataclass(frozen=True)
class ChunkSpec:
    index: int
    win_start: int
    win_end: int
    own_start: int
    own_end: int
    is_first: bool
    is_last: bool

    @property
    def win_len(self) -> int:
        return self.win_end - self.win_start

    @property
    def own_len(self) -> int:
        return self.own_end - self.own_start

    def to_dict(self) -> dict:
        return asdict(self)


def plan_chunks(n_samples: int, window_s: float = 4.0, overlap_s: float = 2.0, sr: int = SPEECH_SR) -> list[ChunkSpec]:
    """Window starts every ``window - overlap``; the last window may be partial. Owned regions tile [0, n)."""
    if n_samples <= 0:
        return []
    w, ov = int(round(window_s * sr)), int(round(overlap_s * sr))
    if w <= 0 or not 0 <= ov < w:
        raise ValueError("need window_s > 0 and 0 <= overlap_s < window_s")
    hop = w - ov
    starts = [0]
    while starts[-1] + w < n_samples:
        starts.append(starts[-1] + hop)
    # boundary between k and k+1 = midpoint of the overlap [starts[k+1], starts[k] + w)
    bounds = [starts[k + 1] + ov // 2 for k in range(len(starts) - 1)]
    specs = []
    for k, s in enumerate(starts):
        own_start = 0 if k == 0 else bounds[k - 1]
        own_end = n_samples if k == len(starts) - 1 else bounds[k]
        specs.append(ChunkSpec(k, s, min(s + w, n_samples), own_start, own_end, k == 0, k == len(starts) - 1))
    check_coverage(specs, n_samples)
    return specs


def crossfade_check(f_samples: int, window_s: float, overlap_s: float, sr: int = SPEECH_SR) -> None:
    """The crossfade (centred on a boundary at the overlap midpoint) must fit in the overlap and in a hop."""
    if f_samples > int(round(overlap_s * sr)) or f_samples > int(round((window_s - overlap_s) * sr)):
        raise ValueError(f"crossfade of {f_samples} samples does not fit overlap={overlap_s}s / hop={window_s - overlap_s}s")


def check_coverage(specs: list[ChunkSpec], n_samples: int) -> None:
    """Raise unless owned regions tile [0, n_samples) with no gap and no duplicate, inside their windows."""
    pos = 0
    for c in specs:
        if c.own_start != pos:
            raise AssertionError(f"chunk {c.index}: owned region starts at {c.own_start}, expected {pos}")
        if c.own_end <= c.own_start:
            raise AssertionError(f"chunk {c.index}: empty owned region")
        if not (c.win_start <= c.own_start and c.own_end <= c.win_end):
            raise AssertionError(f"chunk {c.index}: owned region outside its window")
        pos = c.own_end
    if pos != n_samples:
        raise AssertionError(f"owned regions end at {pos}, clip has {n_samples} samples")


# ------------------------------------------------------------------------------------- identity
@dataclass(frozen=True)
class SlotRef:
    chunk_index: int
    slot: int          # chunk-local separator output; NEVER compare across chunks without an identity mapping


def source_id_for(n: int) -> str:
    return f"src_{n}"


# ----------------------------------------------------------------------------------------- state
@dataclass
class PipelineState:
    """Everything carried from chunk k to chunk k+1. JSON-serialisable; its canonical hash chains the cache
    keys (``state_in`` of chunk k+1 = ``state_hash`` of chunk k's output state). Component dicts are opaque
    to this module: ``identity`` (IdentityStore.to_dict), ``visual`` (tracker state), ``pending`` (per-source
    crossfade tails by file reference, never inline arrays), ``events`` (event-boundary state)."""

    next_chunk: int = 0
    emitted_samples: int = 0                 # final, nonduplicated samples already written (= next own_start)
    identity: dict = field(default_factory=dict)
    visual: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    enrolled_s: dict = field(default_factory=dict)     # source_id -> nonduplicated owned audio seconds enrolled

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineState":
        return cls(**d)

    def hash(self) -> str:
        from .cache import hash_config
        return hash_config(self.to_dict())[:32]

    def write(self, path: str | Path) -> None:
        p = Path(path)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")
        tmp.replace(p)

    @classmethod
    def read(cls, path: str | Path) -> "PipelineState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------------------- budgets
@dataclass
class ChunkConfig:
    window_s: float = 4.0
    overlap_s: float = 2.0
    crossfade_ms: float = 50.0               # centred on the owned boundary, after permutation is resolved

    def validate(self) -> None:
        if not 0 <= self.crossfade_ms / 1000.0 <= self.overlap_s:
            raise ValueError("crossfade must fit inside the overlap")


@dataclass
class MemoryCaps:
    """Hard caps on things that could otherwise grow with clip duration. Enforced by the pipeline (asserted in
    tests with counters), not merely documented."""

    max_decoded_frames: int = 150            # frames resident at once (4 s window @ 25 fps = 100, + margin)
    max_queued_chunks: int = 1               # decoded chunks waiting for inference (prefetch depth)
    max_candidates: int = 4                  # SAM Audio candidates per event window
    candidate_batch: int = 2                 # candidates resident per judge/rank batch
    max_active_identities: int = 8           # identity centroids kept in RAM; the rest persist on disk
    max_resident_models: int = 2             # models loaded at once (ModelLifecycleManager)
    event_max_window_s: float = 12.0         # a sustained event is split into windows this long


@dataclass
class MemoryBudgets:
    """Numeric acceptance budgets, fixed BEFORE any model run (decision recorded in .agents/decisions.md).
    Whole-process figures; browser memory is measured separately."""

    rss_peak_mib: float = 6144.0             # whole-process peak RSS, any workload length
    gpu_peak_mib: float = 9216.0             # peak CUDA memory (12 GiB card, 11.9 usable, headroom for the OS)
    rss_warmup_chunks: int = 5               # growth is judged after this many chunks
    rss_slope_mib_per_audio_min: float = 2.0     # fitted slope after warm-up
    rss_peak_growth_mib_30_to_300s: float = 256.0    # peak(300 s) - peak(30 s)
    browser_heap_mib: float = 700.0          # JS heap at 300 s
    browser_heap_growth_mib_30_to_300s: float = 100.0


@dataclass
class Switches:
    """Every behaviour switch, recorded in every run's manifest (`long_run.switches`)."""

    use_tse: bool = True                     # AV-TSE per active track
    activity_gate: bool = False              # Phase 3 slice 1
    offscreen_upgrade: bool = False          # Phase 3 slice 2
    peel: bool = False                       # Phase 3 slice 3b
    identity: bool = True                    # Phase 4 persistent identities across chunks
    events: bool = True                      # event-scoped non-speech
    diarization: str = "off"                 # "off" | "auto" | "on"
    residual: bool = True


def fingerprint_chunk(spec: ChunkSpec) -> str:
    return hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------- injection protocols
# The chunked pipeline (replay/chunked.py) talks to the world only through these, so CPU/mocked gates and the
# real model run share one code path.
from typing import Protocol, Sequence  # noqa: E402

import numpy as np  # noqa: E402

Box = tuple  # (x1, y1, x2, y2) in decoded-frame pixels


class ChunkSource(Protocol):
    """Bounded access to a recording. Implementations must NOT decode the whole clip and slice it."""

    audio_rate_hz: int                  # capture rate of the multichannel master
    n_audio_samples: int                # at audio_rate_hz
    av_map: AvMap

    def video_pts_us(self) -> list[int]: ...                          # common-clock pts, ascending
    def read_audio(self, start_sample: int, end_sample: int, rate_hz: int) -> np.ndarray:
        """[channels, n] float32 for the half-open master-sample span, resampled to ``rate_hz``
        FROM THE MASTER (never from another view)."""
    def read_frames(self, i0: int, i1: int) -> np.ndarray:
        """uint8 [i1-i0, H, W, 3]; at most MemoryCaps.max_decoded_frames frames per call."""


@dataclass
class VisualChunk:
    boxes_by_track: dict[str, list]            # track_id -> per-frame head box or None, len == n frames in the window
    masks_by_track: dict[str, list]            # track_id -> per-frame vision.mask_store.MaskRecord | None
    state: dict                                # JSON-safe tracker state carried to the next chunk


class VisualFrontEnd(Protocol):
    """Persistent visual track IDs across chunks. ``state`` in, ``VisualChunk.state`` out (never a hidden
    mutable tracker): that state is part of PipelineState and of the cache key chain."""

    def step(self, frames: np.ndarray, pts_us: Sequence[int], state: dict) -> VisualChunk: ...


@dataclass
class SpeechModels:
    """Existing-adapter-shaped callables (see av_scene.speech.run_speech_branch)."""

    separator: object                  # (audio_16k[T]) -> ndarray [S, T]
    scorer: object | None              # TalkNetScorer-like: score_track(audio16k, boxes, frames, fps)
    tse: object | None                 # AVTargetSpeakerExtractor-like: extract_target(audio16k, boxes, frames, fps)
    embed: object                      # (audio_16k[T]) -> embedding vector (speaker embedding, e.g. WavLM-SV)
    revisions: dict = field(default_factory=dict)   # model/checkpoint revision strings (part of cache keys)
