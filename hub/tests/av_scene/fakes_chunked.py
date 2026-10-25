"""CPU fakes for the chunked pipeline (mocked/synthetic evidence only): two voices + one non-speech chirp."""
import hashlib

import numpy as np

from av_scene.contracts import SpeechModels, VisualChunk
from replay.chunked import ArraySource

SR16, MR = 16000, 48000
DUR = 14.5
HARM = {0: (220, 440, 660), 1: (340, 680, 1020)}
ACTIVE = {0: [(0.0, 5.0), (9.0, 14.5)], 1: [(3.0, 8.0), (11.0, 14.5)]}
CHIRP = (6.0, 7.0)           # non-speech burst at 3 kHz, while speaker 1 is also talking
CHIRP_HZ = 3000.0


def _gate(t, v):
    g = np.zeros_like(t)
    for a, b in ACTIVE[v]:
        g[(t >= a) & (t < b)] = 1.0
    return g


def make_audio(a, b, chirp=True):
    t = np.arange(a, b) / MR
    x = sum(_gate(t, v) * sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate(HARM[v])) for v in (0, 1))
    x = 0.2 * x
    if chirp:
        x = x + 0.2 * ((t >= CHIRP[0]) & (t < CHIRP[1])) * np.sin(2 * np.pi * CHIRP_HZ * t)
    return x.astype(np.float32)[None].repeat(2, 0)


def _band(audio, freqs, sr, width=12):
    X = np.fft.rfft(audio)
    fr = np.fft.rfftfreq(len(audio), 1 / sr)
    m = np.zeros_like(fr, bool)
    for f in freqs:
        m |= np.abs(fr - f) < width
    return np.fft.irfft(X * m, len(audio)).astype(np.float32)


def voice_stem(audio, v, sr=SR16):
    return _band(audio, HARM[v], sr)


class FakeSep:
    def __call__(self, x):
        s = [voice_stem(x, 0), voice_stem(x, 1)]
        if hashlib.md5(x.tobytes()).digest()[0] % 2:            # content-dependent slot permutation
            s = s[::-1]
        return np.stack(s)


class FakeScorer:
    def score_track(self, audio, boxes, frames, fps):
        v = 0 if next(b for b in boxes if b is not None)[0] < 30 else 1
        st = voice_stem(audio, v)
        n25 = int(len(audio) / SR16 * 25)
        e = np.sqrt((st[: n25 * 640].reshape(n25, 640) ** 2).mean(1))
        return np.where(e > 0.01, 3.0, -3.0), None, {}


class FakeTSE:
    def extract_target(self, audio, boxes, frames, fps):
        v = 0 if next(b for b in boxes if b is not None)[0] < 30 else 1
        return voice_stem(audio, v), {}


def fake_embed(x):
    X = np.abs(np.fft.rfft(np.asarray(x, np.float64)))
    fr = np.fft.rfftfreq(len(x), 1 / SR16)
    edges = np.linspace(100, 1500, 29)
    v = np.log(np.array([X[(fr >= a) & (fr < b)].sum() for a, b in zip(edges[:-1], edges[1:])]) + 1e-9)
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


class FakeVisual:
    def empty_state(self):
        return {"frames": 0}

    def step(self, frames, pts_us, state):
        n = len(frames)
        return VisualChunk({"p0": [(10.0, 10.0, 30.0, 30.0)] * n, "p1": [(50.0, 10.0, 70.0, 30.0)] * n},
                           {"p0": [None] * n, "p1": [None] * n}, {"frames": state.get("frames", 0) + n})


def make_source(chirp=True):
    n_master = int(DUR * MR)
    pts = [int(i * 40_000) for i in range(int(DUR * 25))]
    return ArraySource(lambda a, b: make_audio(a, b, chirp), MR, pts,
                       lambda i0, i1: np.zeros((i1 - i0, 8, 8, 3), np.uint8), n_audio_samples=n_master)


def speech_models(rev="r1"):
    return SpeechModels(FakeSep(), FakeScorer(), FakeTSE(), fake_embed, {"mf2": rev})


# ---- event fakes -------------------------------------------------------------------------------
def fake_classifier(windows):
    out = []
    for w in windows:
        x = np.asarray(w, np.float32)
        chirp_e = float(np.abs(_band(x, [CHIRP_HZ], SR16, 40)).mean())
        voice_e = float(np.abs(x - _band(x, [CHIRP_HZ], SR16, 40)).mean())
        if chirp_e > 0.02:
            out.append([("Cat", 0.9)])
        elif voice_e > 0.02:
            out.append([("Speech", 0.9)])
        else:
            out.append([])
    return out


class EventFakes:
    """sam_audio(audio_48k, prompt, seeds), judge(...), tracker(...) with call counters."""

    def __init__(self):
        self.sam_calls = []
        self.judge_calls = 0

    def sam_audio(self, audio, prompt, seeds):
        self.sam_calls.append(list(seeds))
        return [_band(np.asarray(audio, np.float32), [CHIRP_HZ], MR, 40) *
                (1.0 if i == 0 else 0.5) for i, _ in enumerate(seeds)]

    def judge(self, candidates, prompt, sample_rate, mix=None):
        self.judge_calls += 1
        return [float(np.sqrt(np.mean(np.square(c)))) for c in candidates]

    def tracker(self, frames, concept):
        return {}
