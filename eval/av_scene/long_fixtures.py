"""Deterministic synthetic long-clip scenes plus FAKE models, so the whole Phase 4 chunked pipeline runs on CPU.

EVIDENCE TIER: mocked/synthetic. Nothing here is learned perception; the fake models are oracles with configurable
imperfections. They prove plumbing (chunking, identity resolution, refusal, memory caps, resume), not model quality.

Audio is synthesised PROCEDURALLY per requested span: every output sample is a pure function of its absolute index
at the target rate (integer-exact phase arithmetic), so `read_audio(0, N)[:, a:b] == read_audio(a, b)` holds and a
300 s scene is never allocated whole. 16 kHz reads evaluate the band-limited signal (everything is below 8 kHz)
at 16 kHz sample times, which is the ideal-resampling equivalent of decimating the 48 kHz master.

TIME CODE (why the fakes need no chunk context)
  The protocols hand the models only `audio_16k` (+ boxes/frames): no chunk index, no window start. Call counters
  would break on resume (cached chunks skip calls) and with prefetch. So the mixture carries a deterministic
  time code: three quiet tones (4.5 to 7.9 kHz, 0.01 amplitude each, above every voice harmonic) whose
  frequencies encode the 40 ms block index in base 45. `decode_window_start(audio16k)` reads it back from the
  first 640 samples. REQUIREMENTS on the pipeline: windows start on multiples of 640 samples (default 4 s / 2 s
  chunks do) and models receive the raw window audio (no loudness normalisation that changes tone ratios, no
  crossfade, no lowpass below 8 kHz). A misaligned or filtered window raises `TimecodeError`; it never guesses.
  `FakeSeparator.override_window_start` is an escape hatch if the pipeline cannot satisfy this.

SCENARIOS (named, deterministic; ground truth in `scene.ground_truth`; sizes: 160x90 frames, 25 fps nominal)
  pause_return           60 s. A speaks [1,7) and [33,41): a 26 s (13-chunk) silence between, then the SAME voice must
                         get the SAME source_id. B speaks [8,20) and [44,56). Both visible throughout. One dropped
                         video frame every 211 (variable-rate pts).
  crossing               24 s. A walks left to right, B right to left (y offset keeps boxes apart); they cross at 12 s.
                         A speaks [1,10) and [17,23), B [8,16): 2 s of cross-talk. Track ids stay with the person.
  reentry                40 s. A visible [0,12) and [24,40), absent in between. A speaks [2,10), [14,19) (off-screen:
                         must be refused/outside_view, not given to B), [26,36). B visible always, speaks [20.5,23).
                         The tracker issues a NEW track id when A returns (`reentry_new_track=True`); both ids map to
                         person pA.
  silent_overlap         30 s. A speaks [1,16), B speaks [12,28) (4 s cross-talk). Silent person C is visible the whole
                         clip and her box partially overlaps A's box. C must never be highlighted.
  slot_swap_every_chunk  30 s. A [0,8) [16,24), B [6,18) [22,30), both visible. The separator's slot order flips on
                         every chunk (`recommended_separator = {"perm": "alternate"}`), so identity must resolve it.
  final_partial          11.3 s (not a multiple of the 2 s hop): last window is 3.3 s. A [1,5), B [5.5,11.2).
  silent_bystander       24 s. A speaks [1,12) on screen; bystander B is visible the whole clip and never speaks; an
                         off-screen speaker C (no person) speaks [14,20) and must be refused.
  matrix(duration_s)     any length. Same two visible speakers alternating in 12 s cycles (A [0.5,6.5), B [6,11.5) +
                         k*12 s) plus one silent bystander: identical resolution and source count at 30/120/300 s.
  Spec strings (`build_scene`, `run --mock-scene`): "name", "name:duration_s" (matrix only), "name:duration_s:seed".
"""

from __future__ import annotations

import bisect
import sys
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)) if str(Path(__file__).resolve().parent) not in sys.path else None
import long_metrics as LM  # noqa: E402  (sibling module; torch-free)

_C = LM.load_hub_module("contracts")
AvMap, SpeechModels, VisualChunk = _C.AvMap, _C.SpeechModels, _C.VisualChunk

SR16 = 16_000
MASTER_SR = 48_000
MASTER_CHANNELS = 2
FPS = 25
BLOCK16 = 640                     # 40 ms: time-code block, = one video frame at 25 fps
TC_BASE, TC_F0_HZ, TC_STEP_HZ = 45, 4500.0, 25.0
TIMECODE_AMP = 0.01
RAMP16 = 80                       # 5 ms gate fade, in 16 kHz samples
MAX_FRAMES_PER_READ = _C.MemoryCaps().max_decoded_frames
FRAME_H, FRAME_W = 90, 160
BOX_W, BOX_H = 18, 22

Interval = tuple[int, int]


class TimecodeError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------- voices
@dataclass(frozen=True)
class Voice:
    """Harmonic voice: f0 and its harmonics (to 3.8 kHz) shaped by Gaussian formant bumps, syllable-rate AM."""

    name: str
    f0_hz: float
    formants: tuple = ((500.0, 120.0, 1.0), (1500.0, 200.0, 0.7), (2500.0, 250.0, 0.4))   # (centre, sigma, gain)
    syllable_hz: float = 3.7
    syllable_phase: float = 0.0
    rms: float = 0.1
    seed: int = 0


VOICE_BANK = {
    "A": Voice("A", 110.0, ((500.0, 120.0, 1.0), (1500.0, 200.0, 0.7), (2500.0, 250.0, 0.4)), 3.7, 0.0, seed=1),
    "B": Voice("B", 165.0, ((700.0, 130.0, 1.0), (1200.0, 180.0, 0.8), (2900.0, 250.0, 0.5)), 4.3, 1.1, seed=2),
    "C": Voice("C", 210.0, ((350.0, 90.0, 1.0), (2100.0, 220.0, 0.9), (3100.0, 200.0, 0.5)), 3.1, 2.3, seed=3),
    "D": Voice("D", 130.0, ((850.0, 140.0, 1.0), (1700.0, 160.0, 0.6), (2600.0, 220.0, 0.6)), 4.9, 0.7, seed=4),
    "E": Voice("E", 250.0, ((300.0, 80.0, 1.0), (900.0, 150.0, 0.5), (2700.0, 230.0, 0.7)), 2.7, 1.9, seed=5),
}


def twin_of(v: Voice, name: str) -> Voice:
    """Same f0 and formants (so `fake_embed` is near-identical: it is phase-blind), different phases and syllable
    timing. Use it to force identity ambiguity."""
    return Voice(name, v.f0_hz, v.formants, v.syllable_hz * 1.07, v.syllable_phase + 0.9, v.rms, v.seed + 100)


@lru_cache(maxsize=64)
def _harmonics(v: Voice):
    f0 = int(round(v.f0_hz * 10))                                   # deci-Hz: integer phase arithmetic
    h = np.arange(1, int(3800 * 10 // f0) + 1, dtype=np.int64)
    f = h * f0 / 10.0
    a = 0.02 + sum(g * np.exp(-0.5 * ((f - c) / s) ** 2) for c, s, g in v.formants)
    a = a * v.rms / np.sqrt(0.5 * np.sum(a ** 2) * 0.40375)         # 0.40375 = E[env^2] of 0.55 + 0.45 sin
    ph = np.random.default_rng([v.seed, zlib.crc32(v.name.encode())]).random(len(h))
    return f0, h * f0, a, ph


def voice_signal(v: Voice, start: int, end: int, rate: int) -> np.ndarray:
    """Ungated voice at samples [start, end) of `rate`; a pure function of the absolute sample index."""
    n = np.arange(start, end, dtype=np.int64)
    _, f_dhz, a, ph = _harmonics(v)
    period = rate * 10
    x = np.zeros(len(n), np.float64)
    for fd, amp, p in zip(f_dhz, a, ph):
        x += amp * np.sin(2 * np.pi * (((fd * n) % period) / period + p))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * (v.syllable_hz * n / rate + v.syllable_phase))
    return x * env


def gate_signal(intervals16: Sequence[Interval], start: int, end: int, rate: int) -> np.ndarray:
    k = rate // SR16
    n = np.arange(start, end, dtype=np.int64)
    g = np.zeros(len(n), np.float64)
    ramp = RAMP16 * k
    for a, b in intervals16:
        a, b = a * k, b * k
        if b <= start or a >= end:
            continue
        m = np.clip((n - a) / ramp, 0.0, 1.0) * np.clip((b - n) / ramp, 0.0, 1.0)
        g = np.maximum(g, m)
    return g


def timecode_signal(start: int, end: int, rate: int, amp: float = TIMECODE_AMP) -> np.ndarray:
    blk = rate // FPS
    n = np.arange(start, end, dtype=np.int64)
    b, local = n // blk, (n % blk).astype(np.float64)
    x = np.zeros(len(n), np.float64)
    for k in range(3):
        d = (b // TC_BASE ** k) % TC_BASE
        f = TC_F0_HZ + (TC_BASE * k + d) * TC_STEP_HZ
        x += np.sin(2 * np.pi * f * local / rate)
    return amp * x


def decode_window_start(audio16k: np.ndarray) -> int:
    """Window start sample (16 kHz) from the time code in the first 640 samples. Raises TimecodeError."""
    x = np.asarray(audio16k, np.float64)
    x = x.mean(axis=0) if x.ndim == 2 else x
    x = np.pad(x[:BLOCK16], (0, max(0, BLOCK16 - len(x))))
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    first = int(TC_F0_HZ / TC_STEP_HZ)                                # bin of the lowest code tone (25 Hz bins)
    digits = []
    for k in range(3):
        band = mag[first + TC_BASE * k: first + TC_BASE * (k + 1)]
        o = np.sort(band)[::-1]
        if o[0] <= 0 or o[0] < 3.0 * o[1] or o[0] < 3.0 * np.median(band):
            raise TimecodeError("no clean time code in the first 640 samples: window start is not a multiple of 640 "
                                "samples, or the audio was filtered/normalised/mixed before reaching the model")
        m = first + TC_BASE * k + int(np.argmax(band))
        # a tone that starts at phase 0 on a block boundary is purely imaginary in its bin; a window that starts
        # inside a block rotates it, which catches small misalignments the amplitude test cannot
        if abs(spec[m].real) > 0.25 * mag[m]:
            raise TimecodeError("time code phase is off: the window does not start on a 640-sample boundary")
        digits.append(m - first - TC_BASE * k)
    return (digits[0] + TC_BASE * digits[1] + TC_BASE ** 2 * digits[2]) * BLOCK16


# ----------------------------------------------------------------------------------------- scene definition
@dataclass(frozen=True)
class Person:
    """A visible human. `speaker_id=None` is a silent bystander. `visible` and path are in seconds / fractions:
    head centre moves linearly from path[0] at t=0 to path[1] at t=duration, as fractions of (width, height)."""

    person_id: str
    speaker_id: str | None
    visible: tuple
    color: tuple
    path: tuple = ((0.5, 0.5), (0.5, 0.5))
    track_id: str | None = None


def _to_samples(iv: Sequence[tuple[float, float]], n16: int) -> list[Interval]:
    return LM.merge((max(0, int(round(a * SR16))), min(n16, int(round(b * SR16)))) for a, b in iv)


def _contains(iv: Sequence[Interval], x: int) -> bool:
    i = bisect.bisect_right([a for a, _ in iv], x) - 1
    return i >= 0 and iv[i][0] <= x < iv[i][1]


@dataclass
class GroundTruth:
    """All intervals are half-open and in 16 kHz samples."""

    n_samples: int
    rate: int
    speaker_active: dict[str, list[Interval]]
    person_visible: dict[str, list[Interval]]
    person_speaker: dict[str, str | None]
    track_visible: dict[str, list[Interval]]
    track_person: dict[str, str]
    speaker_person: dict[str, str | None]
    onscreen: dict[str, list[Interval]]                  # speaker active AND own person visible
    offscreen_speakers: list[str]                        # speakers with no person at all
    offscreen_active: dict[str, list[Interval]]          # speaker active while its person is absent / nonexistent
    should_refuse: list[Interval]                        # union of offscreen_active and scene.extra_should_refuse
    silent_visible_by_track: dict[str, list[Interval]]   # visible and not speaking, per visual track
    silent_bystander: dict[str, bool]                    # person never speaks

    def metric_inputs(self) -> dict:
        """Keyword-ready inputs for long_metrics functions."""
        return {"speaker_active": self.speaker_active, "onscreen": self.onscreen, "track_person": self.track_person,
                "person_speaker": self.person_speaker, "should_refuse": self.should_refuse,
                "silent_visible": self.silent_visible_by_track}


class SyntheticScene:
    """`SyntheticScene(duration_s, speakers, schedule, people=(), ...)`.

    speakers: {speaker_id: Voice}; schedule: {speaker_id: [(start_s, end_s), ...]} half-open seconds (rounded to
    16 kHz samples); people: visible humans (a speaker without a person is off-screen)."""

    def __init__(self, duration_s: float, speakers: Mapping[str, Voice], schedule: Mapping[str, Sequence[tuple[float, float]]],
                 people: Sequence[Person] = (), *, name: str = "scene", seed: int = 0, fps: float = 25.0,
                 frame_size: tuple[int, int] = (FRAME_H, FRAME_W), reentry_new_track: bool = False,
                 pts_jitter_us: int = 2000, drop_frame_every: int = 0,
                 extra_should_refuse: Sequence[tuple[float, float]] = (), meta: dict | None = None):
        self.name, self.seed, self.fps = name, seed, fps
        self.n16 = int(round(duration_s * SR16))
        self.duration_s = self.n16 / SR16
        self.speakers = dict(speakers)
        self.active = {s: _to_samples(schedule.get(s, ()), self.n16) for s in self.speakers}
        self.people = list(people)
        self.visible = {p.person_id: _to_samples(p.visible, self.n16) for p in self.people}
        self.reentry_new_track = reentry_new_track
        self.frame_h, self.frame_w = frame_size
        self.pts_jitter_us, self.drop_frame_every = pts_jitter_us, drop_frame_every
        self.extra_should_refuse = _to_samples(extra_should_refuse, self.n16)
        self.meta = dict(meta or {})
        self._pts: list[int] | None = None
        for p in self.people:
            if p.speaker_id is not None and p.speaker_id not in self.speakers:
                raise ValueError(f"person {p.person_id}: unknown speaker {p.speaker_id}")
        if len({p.color for p in self.people}) != len(self.people):
            raise ValueError("person colours must be distinct (the fakes identify people by pixel colour)")

    # -- sizes
    @property
    def n48(self) -> int:
        return self.n16 * (MASTER_SR // SR16)

    # -- audio
    def speaker_signal(self, speaker_id: str, start: int, end: int, rate: int) -> np.ndarray:
        k = rate // SR16
        ivs = [(a, b) for a, b in self.active[speaker_id] if b * k > start and a * k < end]
        if not ivs:
            return np.zeros(end - start, np.float64)
        return voice_signal(self.speakers[speaker_id], start, end, rate) * gate_signal(ivs, start, end, rate)

    def mixture(self, start: int, end: int, rate: int) -> np.ndarray:
        """Mono mixture at `rate` over [start, end) of that rate: all speakers + time code."""
        x = timecode_signal(start, end, rate)
        for s in self.speakers:
            x += self.speaker_signal(s, start, end, rate)
        return x

    def active_in(self, start16: int, end16: int) -> list[str]:
        """Speakers (scene order) with any active sample in [start16, end16)."""
        return [s for s in self.speakers if LM.intersect(self.active[s], [(start16, end16)])]

    # -- people, tracks, video
    def person(self, person_id: str) -> Person:
        return next(p for p in self.people if p.person_id == person_id)

    def person_speaks_at(self, person_id: str, sample16: int) -> bool:
        sp = self.person(person_id).speaker_id
        return sp is not None and _contains(self.active[sp], sample16)

    def track_id_at(self, p: Person, sample16: int) -> str | None:
        vis = self.visible[p.person_id]
        if not _contains(vis, sample16):
            return None
        base = p.track_id or f"trk_{p.person_id}"
        if not self.reentry_new_track:
            return base
        inc = bisect.bisect_right([a for a, _ in vis], sample16) - 1
        return base if inc == 0 else f"{base}.r{inc}"

    def box_at(self, p: Person, sample16: int) -> tuple | None:
        if not _contains(self.visible[p.person_id], sample16):
            return None
        f = sample16 / max(1, self.n16)
        (x0, y0), (x1, y1) = p.path
        cx, cy = int(round((x0 + (x1 - x0) * f) * self.frame_w)), int(round((y0 + (y1 - y0) * f) * self.frame_h))
        return (cx - BOX_W // 2, cy - BOX_H // 2, cx + BOX_W // 2, cy + BOX_H // 2)

    def video_pts_us(self) -> list[int]:
        """Variable-rate common-clock pts: nominal fps + deterministic 0..jitter lag + optional dropped frames."""
        if self._pts is None:
            n, out = int(self.duration_s * self.fps), []
            for i in range(n):
                if self.drop_frame_every and i % self.drop_frame_every == self.drop_frame_every - 1:
                    continue
                jit = ((i * 2654435761 + self.seed * 97) % (self.pts_jitter_us + 1)) if self.pts_jitter_us else 0
                out.append(int(round(i * 1e6 / self.fps)) + jit)
            self._pts = out
        return self._pts

    @staticmethod
    def pts_to_sample16(pts_us: int) -> int:
        return (int(pts_us) * SR16) // 1_000_000

    def detections(self, pts_us: int) -> list[tuple[Person, str, tuple]]:
        s = self.pts_to_sample16(pts_us)
        out = []
        for p in self.people:
            t = self.track_id_at(p, s)
            if t is not None:
                out.append((p, t, self.box_at(p, s)))
        return out

    def render_frame(self, pts_us: int) -> np.ndarray:
        img = np.empty((self.frame_h, self.frame_w, 3), np.uint8)
        img[:] = 40
        img[:, :, 2] += (np.arange(self.frame_w) % 7).astype(np.uint8)      # faint texture, never a person colour
        for p, _, (x1, y1, x2, y2) in self.detections(pts_us):
            img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = p.color
        return img

    def person_at_pixel(self, frame: np.ndarray, box) -> str | None:
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        px = frame[min(max(cy, 0), frame.shape[0] - 1), min(max(cx, 0), frame.shape[1] - 1)].astype(int)
        best = min(self.people, key=lambda p: int(np.abs(px - np.asarray(p.color)).sum()), default=None)
        return best.person_id if best is not None and np.abs(px - np.asarray(best.color)).sum() <= 30 else None

    # -- objects
    def source(self, **kw) -> "SyntheticSource":
        return SyntheticSource(self, **kw)

    @property
    def ground_truth(self) -> GroundTruth:
        n = self.n16
        p_sp = {p.person_id: p.speaker_id for p in self.people}
        sp_p = {s: next((p.person_id for p in self.people if p.speaker_id == s), None) for s in self.speakers}
        onscreen = {s: LM.intersect(self.active[s], self.visible[sp_p[s]]) if sp_p[s] else [] for s in self.speakers}
        offs = {s: LM.subtract(self.active[s], self.visible[sp_p[s]] if sp_p[s] else []) for s in self.speakers}
        track_vis: dict[str, list[Interval]] = {}
        track_person: dict[str, str] = {}
        silent: dict[str, list[Interval]] = {}
        for p in self.people:
            vis = self.visible[p.person_id]
            talk = self.active[p.speaker_id] if p.speaker_id else []
            base = p.track_id or f"trk_{p.person_id}"
            for k, iv in enumerate(vis):
                t = base if (not self.reentry_new_track or k == 0) else f"{base}.r{k}"
                if not self.reentry_new_track:
                    track_vis.setdefault(t, []).append(iv)
                else:
                    track_vis[t] = [iv]
                track_person[t] = p.person_id
        for t, iv in track_vis.items():
            p = self.person(track_person[t])
            silent[t] = LM.subtract(iv, self.active[p.speaker_id] if p.speaker_id else [])
        return GroundTruth(
            n_samples=n, rate=SR16, speaker_active={s: list(v) for s, v in self.active.items()},
            person_visible={k: list(v) for k, v in self.visible.items()}, person_speaker=p_sp,
            track_visible=track_vis, track_person=track_person, speaker_person=sp_p, onscreen=onscreen,
            offscreen_speakers=[s for s in self.speakers if sp_p[s] is None], offscreen_active=offs,
            should_refuse=LM.merge([iv for v in offs.values() for iv in v] + list(self.extra_should_refuse)),
            silent_visible_by_track=silent,
            silent_bystander={p.person_id: not (p.speaker_id and self.active[p.speaker_id]) for p in self.people})


class SyntheticSource:
    """`contracts.ChunkSource` over a scene: 48 kHz x 2 channel master (channel 1 = 0.9 x channel 0), bounded reads.

    Counters in `.stats` let tests assert the memory discipline: `max_audio_samples_per_read` (master samples),
    `max_frames_per_read` (raises above MemoryCaps.max_decoded_frames), `reads`, `frames_read`."""

    audio_rate_hz = MASTER_SR

    def __init__(self, scene: SyntheticScene, max_frames: int = MAX_FRAMES_PER_READ):
        self.scene, self.max_frames = scene, max_frames
        self.n_audio_samples = scene.n48
        self.av_map = AvMap(audio_rate_hz=MASTER_SR, audio_start_us=0, video_transform="identity")
        self.stats = {"audio_reads": 0, "frame_reads": 0, "max_audio_samples_per_read": 0, "max_frames_per_read": 0,
                      "frames_read": 0}

    def video_pts_us(self) -> list[int]:
        return list(self.scene.video_pts_us())

    def read_audio(self, start_sample: int, end_sample: int, rate_hz: int) -> np.ndarray:
        if MASTER_SR % rate_hz:
            raise ValueError(f"cannot resample the 48 kHz master to {rate_hz} Hz exactly (need an integer divisor)")
        if not 0 <= start_sample <= end_sample <= self.n_audio_samples:
            raise ValueError(f"read [{start_sample}, {end_sample}) outside the {self.n_audio_samples}-sample master")
        d = MASTER_SR // rate_hz
        x = self.scene.mixture(start_sample // d, end_sample // d, rate_hz)
        self.stats["audio_reads"] += 1
        self.stats["max_audio_samples_per_read"] = max(self.stats["max_audio_samples_per_read"], end_sample - start_sample)
        return np.stack([x, 0.9 * x]).astype(np.float32)

    def read_frames(self, i0: int, i1: int) -> np.ndarray:
        n = i1 - i0
        if n > self.max_frames:
            raise ValueError(f"read_frames({i0}, {i1}): {n} frames exceed the {self.max_frames}-frame cap")
        pts = self.scene.video_pts_us()
        frames = np.stack([self.scene.render_frame(pts[i]) for i in range(i0, i1)]) if n > 0 else \
            np.zeros((0, self.scene.frame_h, self.scene.frame_w, 3), np.uint8)
        self.stats["frame_reads"] += 1
        self.stats["frames_read"] += n
        self.stats["max_frames_per_read"] = max(self.stats["max_frames_per_read"], n)
        return frames


# ----------------------------------------------------------------------------------------------- fake models
class _SceneModel:
    override_window_start: int | None = None

    def __init__(self, scene: SyntheticScene):
        self.scene = scene

    def locate(self, audio16k: np.ndarray) -> int:
        return decode_window_start(audio16k) if self.override_window_start is None else int(self.override_window_start)


def _perm_for(spec, chunk: int, n: int) -> list[int]:
    if callable(spec):
        p = list(spec(chunk, n))
    elif isinstance(spec, (list, tuple)):
        p = list(spec[chunk % len(spec)])
    elif spec == "fixed":
        p = list(range(n))
    elif spec == "alternate":
        p = [(j + chunk) % n for j in range(n)] if n > 2 else (list(range(n)) if chunk % 2 == 0 else list(range(n))[::-1])
    elif spec == "random":
        p = list(np.random.default_rng([12345, chunk]).permutation(n))
    else:
        raise ValueError(f"unknown permutation schedule {spec!r}")
    if sorted(p) != list(range(n)):
        raise ValueError(f"not a permutation of {n} slots: {p}")
    return [int(i) for i in p]


class FakeSeparator(_SceneModel):
    """`separator(audio_16k[T]) -> ndarray [n_slots, T]`.

    Oracle stems of the speakers active in the window, placed in slots then PERMUTED per chunk (chunk = window
    start // hop_samples) so slot index is meaningless across chunks, exactly like a real separator. Speaker i
    (scene order, among those active in the window) lands in base slot i % n_slots; with more active speakers than
    slots, voices SHARE a slot (a genuine mixture: the degraded/capacity case). Every slot gets small deterministic
    noise. perm: "alternate" (reverse each odd chunk; cyclic shift for >2 slots), "fixed", "random", a sequence of
    permutations cycled by chunk, or callable(chunk, n_slots) -> permutation. `calls` records each invocation."""

    def __init__(self, scene: SyntheticScene, n_slots: int = 2, perm="alternate", hop_samples: int = 32_000,
                 noise_db: float = -45.0, seed: int = 0):
        super().__init__(scene)
        self.n_slots, self.perm, self.hop, self.noise_db, self.seed = n_slots, perm, hop_samples, noise_db, seed
        self.calls: list[dict] = []

    def plan(self, start: int, length: int) -> dict:
        chunk = start // self.hop
        act = self.scene.active_in(start, start + length)
        base = {s: i % self.n_slots for i, s in enumerate(act)}
        perm = _perm_for(self.perm, chunk, self.n_slots)
        return {"window_start": start, "chunk": chunk, "perm": perm, "active": act,
                "slot_of": {s: perm.index(b) for s, b in base.items()}, "_base": base}

    def __call__(self, audio16k: np.ndarray) -> np.ndarray:
        audio16k = np.asarray(audio16k)
        start, T = self.locate(audio16k), len(audio16k)
        plan = self.plan(start, T)
        stems = np.zeros((self.n_slots, T), np.float64)
        for s, b in plan["_base"].items():
            stems[b] += self.scene.speaker_signal(s, start, start + T, SR16)
        out = np.stack([stems[plan["perm"][j]] for j in range(self.n_slots)])
        amp = 10 ** (self.noise_db / 20)
        for j in range(self.n_slots):
            out[j] += amp * np.random.default_rng([self.seed, start, j]).standard_normal(T)
        self.calls.append({k: v for k, v in plan.items() if k != "_base"} | {"n_samples": T})
        return out.astype(np.float32)


class FakeScorer(_SceneModel):
    """TalkNet-like `score_track(audio16k, boxes, frames, fps) -> (raw_logits, None, info)`.

    Logits are +/-`logit` from ground-truth speech of the person whose colour is inside the box, on a 25 fps grid
    anchored to the WINDOW START (length int(T/16000*25), as run_speech_branch expects). `misses` forces -logit and
    `false_alarms` +logit for person_id -> [(start_s, end_s)]."""

    def __init__(self, scene: SyntheticScene, misses: Mapping[str, Sequence[tuple[float, float]]] | None = None,
                 false_alarms: Mapping[str, Sequence[tuple[float, float]]] | None = None, logit: float = 3.0):
        super().__init__(scene)
        self.logit = logit
        self.misses = {k: _to_samples(v, scene.n16) for k, v in (misses or {}).items()}
        self.false_alarms = {k: _to_samples(v, scene.n16) for k, v in (false_alarms or {}).items()}
        self.n_calls = 0

    def score_track(self, audio16k, boxes, frames, fps):
        self.n_calls += 1
        start, n25 = self.locate(np.asarray(audio16k)), int(len(audio16k) / SR16 * FPS)
        raw = np.full(n25, -self.logit, np.float32)
        for j in range(n25):
            k = min(int(np.rint(j / FPS * fps)), len(frames) - 1, len(boxes) - 1)
            if k < 0 or boxes[k] is None:
                continue
            pid = self.scene.person_at_pixel(frames[k], boxes[k])
            if pid is None:
                continue
            t = start + j * BLOCK16 + BLOCK16 // 2
            on = self.scene.person_speaks_at(pid, t)
            if _contains(self.misses.get(pid, []), t):
                on = False
            if _contains(self.false_alarms.get(pid, []), t):
                on = True
            raw[j] = self.logit if on else -self.logit
        return raw, None, {"fake": 1.0}


class FakeTSE(_SceneModel):
    """AV-TSE-like `extract_target(audio16k, boxes, frames, fps) -> (wave, info)`: the ground-truth stem of the
    boxed person's speaker for this window (silence for a bystander), plus small noise; optional `leak_db` mixes the
    other active speakers in at that level (an imperfect extractor)."""

    def __init__(self, scene: SyntheticScene, noise_db: float = -45.0, leak_db: float | None = None, seed: int = 0):
        super().__init__(scene)
        self.noise_db, self.leak_db, self.seed = noise_db, leak_db, seed
        self.calls: list[dict] = []

    def extract_target(self, audio16k, boxes, frames, fps):
        start, T = self.locate(np.asarray(audio16k)), len(audio16k)
        pid = next((self.scene.person_at_pixel(frames[k], b) for k, b in enumerate(boxes) if b is not None), None)
        sp = self.scene.person(pid).speaker_id if pid else None
        x = self.scene.speaker_signal(sp, start, start + T, SR16) if sp else np.zeros(T)
        if self.leak_db is not None:
            for o in self.scene.active_in(start, start + T):
                if o != sp:
                    x = x + 10 ** (self.leak_db / 20) * self.scene.speaker_signal(o, start, start + T, SR16)
        x = x + 10 ** (self.noise_db / 20) * np.random.default_rng([self.seed, start, 99]).standard_normal(T)
        self.calls.append({"window_start": start, "person": pid, "speaker": sp})
        return x.astype(np.float32), {"fake": 1.0}


EMBED_BANDS = 64


def fake_embed(audio16k: np.ndarray) -> np.ndarray:
    """Deterministic spectral signature: L2-normalised, mean-centred log band energy over 0 to 4 kHz in 64 bands
    (level-invariant; blind to phase, so `twin_of` voices embed near-identically). Zeros for near-silent input."""
    x = np.asarray(audio16k, np.float64)
    x = x.mean(axis=0) if x.ndim == 2 else x
    if len(x) < 256:
        return np.zeros(EMBED_BANDS, np.float32)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR16)
    edges = np.searchsorted(freqs, np.linspace(0.0, 4000.0, EMBED_BANDS + 1))
    e = np.add.reduceat(X, edges[:-1])[:EMBED_BANDS] / np.maximum(np.diff(edges), 1)
    v = np.log(e + 1e-9)
    v -= v.mean()
    nrm = np.linalg.norm(v)
    return (v / nrm if nrm > 1e-9 and e.sum() > 1e-6 else np.zeros(EMBED_BANDS)).astype(np.float32)


class FakeVisualFrontEnd:
    """`contracts.VisualFrontEnd` with oracle detections: persistent track ids from the scene (a fresh id on
    re-entry when the scene says so), per-frame head boxes, `masks_by_track` all None (no segmentation here).
    `dropouts`: person_id -> [(start_s, end_s)] where the detector misses (box None). State is JSON-safe and only
    informational (last_pts_us, frame counts), so the output is a pure function of (frames, pts)."""

    def __init__(self, scene: SyntheticScene, dropouts: Mapping[str, Sequence[tuple[float, float]]] | None = None):
        self.scene = scene
        self.dropouts = {k: _to_samples(v, scene.n16) for k, v in (dropouts or {}).items()}
        self.calls = 0

    def step(self, frames: np.ndarray, pts_us: Sequence[int], state: dict):
        self.calls += 1
        n = len(pts_us)
        if len(frames) != n:
            raise ValueError(f"{len(frames)} frames for {n} pts")
        boxes: dict[str, list] = {}
        for i, pts in enumerate(pts_us):
            s = self.scene.pts_to_sample16(pts)
            for p, tid, box in self.scene.detections(pts):
                lst = boxes.setdefault(tid, [None] * n)
                if not _contains(self.dropouts.get(p.person_id, []), s):
                    lst[i] = tuple(int(v) for v in box)
        tracks = {k: dict(v) for k, v in state.get("tracks", {}).items()}
        for tid, lst in boxes.items():
            seen = [i for i, b in enumerate(lst) if b is not None]
            if seen:
                t = tracks.setdefault(tid, {"first_pts_us": int(pts_us[seen[0]]), "frames": 0})
                t["last_pts_us"], t["frames"] = int(pts_us[seen[-1]]), t["frames"] + len(seen)
        new = {"tracks": tracks, "n_steps": int(state.get("n_steps", 0)) + 1}
        return VisualChunk(boxes_by_track=boxes, masks_by_track={t: [None] * n for t in boxes}, state=new)


def make_speech_models(scene: SyntheticScene, *, n_slots: int = 2, perm="alternate", misses=None, false_alarms=None,
                       use_scorer: bool = True, use_tse: bool = True, hop_samples: int = 32_000,
                       leak_db: float | None = None) -> "SpeechModels":
    return SpeechModels(separator=FakeSeparator(scene, n_slots, perm, hop_samples),
                        scorer=FakeScorer(scene, misses, false_alarms) if use_scorer else None,
                        tse=FakeTSE(scene, leak_db=leak_db) if use_tse else None, embed=fake_embed,
                        revisions={"fake_models": "long_fixtures-v1"})


def oracle_segments(scene: SyntheticScene) -> list:
    """The perfect prediction in `long_metrics.Seg` form (16 kHz samples): one persistent source per speaker
    (`src_<index>`), matched to the right visual track while its person is visible, refused as outside_view otherwise."""
    gt, out = scene.ground_truth, []
    for i, (spk, act) in enumerate(scene.active.items()):
        pid = gt.speaker_person[spk]
        pieces = []
        if pid is not None:
            for t, tv in gt.track_visible.items():
                if gt.track_person[t] == pid:
                    pieces += [(a, b, t) for a, b in LM.intersect(act, tv)]
        covered = [(a, b) for a, b, _ in pieces]
        for a, b, t in pieces:
            out.append(LM.Seg(f"src_{i}", a, b, t, "matched", "visible", "ok"))
        for a, b in LM.subtract(act, covered):
            out.append(LM.Seg(f"src_{i}", a, b, None, "unmatched", "outside_view", "ok", "speaker not visible"))
    return sorted(out, key=lambda s: (s.start, s.source_id))


# ------------------------------------------------------------------------------------------------ scenarios
def _faces(n: int = 4) -> list[tuple]:
    return [(220, 40, 40), (40, 200, 60), (50, 90, 230), (230, 210, 40), (210, 50, 200)][:n]


def pause_return(seed: int = 0) -> SyntheticScene:
    r, g = _faces(2)
    return SyntheticScene(
        60.0, {"A": VOICE_BANK["A"], "B": VOICE_BANK["B"]}, {"A": [(1, 7), (33, 41)], "B": [(8, 20), (44, 56)]},
        [Person("pA", "A", ((0, 60),), r, ((0.3, 0.4), (0.3, 0.4))), Person("pB", "B", ((0, 60),), g, ((0.7, 0.6), (0.7, 0.6)))],
        name="pause_return", seed=seed, drop_frame_every=211)


def crossing(seed: int = 0) -> SyntheticScene:
    r, g = _faces(2)
    return SyntheticScene(
        24.0, {"A": VOICE_BANK["A"], "B": VOICE_BANK["B"]}, {"A": [(1, 10), (17, 23)], "B": [(8, 16)]},
        [Person("pA", "A", ((0, 24),), r, ((0.2, 0.35), (0.8, 0.35))), Person("pB", "B", ((0, 24),), g, ((0.8, 0.65), (0.2, 0.65)))],
        name="crossing", seed=seed)


def reentry(seed: int = 0, new_track: bool = True) -> SyntheticScene:
    r, g = _faces(2)
    return SyntheticScene(
        40.0, {"A": VOICE_BANK["A"], "B": VOICE_BANK["B"]}, {"A": [(2, 10), (14, 19), (26, 36)], "B": [(20.5, 23)]},
        [Person("pA", "A", ((0, 12), (24, 40)), r, ((0.3, 0.5), (0.3, 0.5))),
         Person("pB", "B", ((0, 40),), g, ((0.7, 0.5), (0.7, 0.5)))],
        name="reentry", seed=seed, reentry_new_track=new_track)


def silent_overlap(seed: int = 0) -> SyntheticScene:
    r, g, b = _faces(3)
    return SyntheticScene(
        30.0, {"A": VOICE_BANK["A"], "B": VOICE_BANK["B"]}, {"A": [(1, 16)], "B": [(12, 28)]},
        [Person("pA", "A", ((0, 30),), r, ((0.40, 0.5), (0.40, 0.5))),
         Person("pC", None, ((0, 30),), b, ((0.4625, 0.5), (0.4625, 0.5))),      # centres 10 px apart: boxes overlap
         Person("pB", "B", ((0, 30),), g, ((0.8, 0.5), (0.8, 0.5)))],
        name="silent_overlap", seed=seed)


def slot_swap_every_chunk(seed: int = 0) -> SyntheticScene:
    r, g = _faces(2)
    return SyntheticScene(
        30.0, {"A": VOICE_BANK["A"], "B": VOICE_BANK["B"]}, {"A": [(0, 8), (16, 24)], "B": [(6, 18), (22, 30)]},
        [Person("pA", "A", ((0, 30),), r, ((0.3, 0.5), (0.3, 0.5))), Person("pB", "B", ((0, 30),), g, ((0.7, 0.5), (0.7, 0.5)))],
        name="slot_swap_every_chunk", seed=seed, meta={"recommended_separator": {"perm": "alternate"}})


def final_partial(seed: int = 0) -> SyntheticScene:
    r, g = _faces(2)
    return SyntheticScene(
        11.3, {"A": VOICE_BANK["A"], "B": VOICE_BANK["B"]}, {"A": [(1, 5)], "B": [(5.5, 11.2)]},
        [Person("pA", "A", ((0, 11.3),), r, ((0.3, 0.5), (0.3, 0.5))), Person("pB", "B", ((0, 11.3),), g, ((0.7, 0.5), (0.7, 0.5)))],
        name="final_partial", seed=seed)


def silent_bystander(seed: int = 0) -> SyntheticScene:
    r, g = _faces(2)
    return SyntheticScene(
        24.0, {"A": VOICE_BANK["A"], "C": VOICE_BANK["C"]}, {"A": [(1, 12)], "C": [(14, 20)]},
        [Person("pA", "A", ((0, 24),), r, ((0.3, 0.5), (0.3, 0.5))), Person("pB", None, ((0, 24),), g, ((0.7, 0.5), (0.7, 0.5)))],
        name="silent_bystander", seed=seed)


def matrix(duration_s: float = 60.0, seed: int = 0) -> SyntheticScene:
    r, g, b = _faces(3)
    a_iv, b_iv, k = [], [], 0
    while k * 12.0 < duration_s:
        a_iv.append((12.0 * k + 0.5, 12.0 * k + 6.5))
        b_iv.append((12.0 * k + 6.0, 12.0 * k + 11.5))
        k += 1
    return SyntheticScene(
        duration_s, {"A": VOICE_BANK["A"], "B": VOICE_BANK["B"]}, {"A": a_iv, "B": b_iv},
        [Person("pA", "A", ((0, duration_s),), r, ((0.25, 0.4), (0.25, 0.4))),
         Person("pB", "B", ((0, duration_s),), g, ((0.75, 0.4), (0.75, 0.4))),
         Person("pS", None, ((0, duration_s),), b, ((0.5, 0.75), (0.5, 0.75)))],
        name="matrix", seed=seed)


SCENARIOS: dict[str, Callable[..., SyntheticScene]] = {f.__name__: f for f in (
    pause_return, crossing, reentry, silent_overlap, slot_swap_every_chunk, final_partial, silent_bystander, matrix)}


def build_scene(spec: str, duration_s: float | None = None, seed: int | None = None) -> SyntheticScene:
    """"name", "name:duration_s" (matrix only) or "name:duration_s:seed"; arguments override the spec."""
    parts = spec.split(":")
    name = parts[0]
    if name not in SCENARIOS:
        raise ValueError(f"unknown scene {name!r}; known: {', '.join(SCENARIOS)}")
    dur = duration_s if duration_s is not None else (float(parts[1]) if len(parts) > 1 and parts[1] else None)
    sd = seed if seed is not None else (int(parts[2]) if len(parts) > 2 else 0)
    if name == "matrix":
        return matrix(dur if dur is not None else 60.0, sd)
    if dur is not None:
        raise ValueError(f"scene {name!r} has a fixed duration; only 'matrix' takes one")
    return SCENARIOS[name](seed=sd)
