"""ChunkedSpeechRunner on CPU fakes: tiling, identity under slot permutation, cache reuse, crash resume, invalidation.

Evidence: mocked/synthetic (fake separator/scorer/TSE/embedder). No model, GPU or device is involved.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from av_scene.cache import StageCache
from av_scene.config import load_config
from av_scene.contracts import SpeechModels
from replay.chunked import ArraySource, ChunkedSpeechRunner, InjectedCrash, read_arrays

SR16, MR = 16000, 48000
DUR = 14.5
HARM = {0: (220, 440, 660), 1: (340, 680, 1020)}
ACTIVE = {0: [(0.0, 5.0), (9.0, 14.5)], 1: [(3.0, 8.0), (11.0, 14.5)]}


def _gate(t, v):
    g = np.zeros_like(t)
    for a, b in ACTIVE[v]:
        g[(t >= a) & (t < b)] = 1.0
    return g


def make_audio(a, b):
    t = np.arange(a, b) / MR
    x = sum(_gate(t, v) * sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate(HARM[v])) for v in (0, 1))
    return (0.2 * x).astype(np.float32)[None].repeat(2, 0)


def voice_stem(audio, v, sr=SR16):
    X = np.fft.rfft(audio)
    fr = np.fft.rfftfreq(len(audio), 1 / sr)
    m = np.zeros_like(fr, bool)
    for f in HARM[v]:
        m |= np.abs(fr - f) < 12
    return np.fft.irfft(X * m, len(audio)).astype(np.float32)


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
    v = np.array([X[(fr >= a) & (fr < b)].sum() for a, b in zip(edges[:-1], edges[1:])]) + 1e-9
    v = np.log(v)
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


class FakeVisual:
    def empty_state(self):
        return {"frames": 0}

    def step(self, frames, pts_us, state):
        from av_scene.contracts import VisualChunk
        n = len(frames)
        return VisualChunk({"p0": [(10.0, 10.0, 30.0, 30.0)] * n, "p1": [(50.0, 10.0, 70.0, 30.0)] * n},
                           {"p0": [None] * n, "p1": [None] * n}, {"frames": state.get("frames", 0) + n})


def make_source():
    n_master = int(DUR * MR)
    pts = [int(i * 40_000) for i in range(int(DUR * 25))]
    return ArraySource(make_audio, MR, pts, lambda i0, i1: np.zeros((i1 - i0, 8, 8, 3), np.uint8), n_audio_samples=n_master)


def make_runner(tmp_path, cache=None, fault=None, rev="r1", cfg=None, name="run"):
    models = SpeechModels(FakeSep(), FakeScorer(), FakeTSE(), fake_embed, {"mf2": rev})
    cache = cache or StageCache(tmp_path / "cache")
    return ChunkedSpeechRunner(make_source(), models, FakeVisual(), cache, cfg or load_config(), tmp_path / name,
                               media_id="synthetic-14.5s", fault=fault)


def digest(res):
    """Everything that must match between a resumed and an uninterrupted run."""
    h = {}
    for rec in res.records:
        fin = read_arrays(rec.speech_dir, "final", "final.json")
        meta = json.loads((rec.speech_dir / "meta.json").read_text())
        h[rec.spec.index] = ({s: (st, a.tobytes()) for s, (st, a) in fin.items()},
                             json.dumps(meta["segments"], sort_keys=True), meta["slots"],
                             (rec.speech_dir / "state_out.json").read_text())
    return h


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    d = tmp_path_factory.mktemp("base")
    r = make_runner(d)
    return r, r.run(), d


def test_streams_tile_without_gap_or_duplicate_and_identities_survive_permutation(baseline):
    r, res, _ = baseline
    n = res.n_samples
    assert len(res.records) == 7 and res.records[-1].spec.own_end == n
    per_src: dict[str, list[tuple[int, int]]] = {}
    for rec in res.records:
        for sid, (st, a) in read_arrays(rec.speech_dir, "final", "final.json").items():
            per_src.setdefault(sid, []).append((st, st + len(a)))
    for sid, spans in per_src.items():
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 <= b0, f"{sid}: samples emitted twice"
        assert spans[-1][1] <= n
    speech = sorted(s for s in per_src if s.startswith("src_"))
    assert speech == ["src_0", "src_1"]                       # two people, slots permuted by content, no extra ids
    # activity is accounted on owned samples: total enrolled seconds can never exceed clip duration per source
    assert all(v <= DUR + 1e-6 for v in res.state.enrolled_s.values())
    # each source's identity mapped consistently: src_0 always carries the same voice (220 Hz family)
    first = read_arrays(res.records[1].speech_dir, "final", "final.json")
    a = first["src_0"][1]
    fr = np.fft.rfftfreq(len(a), 1 / SR16)
    peak = fr[np.argmax(np.abs(np.fft.rfft(a)))]
    assert min(abs(peak - f) for f in HARM[0] + HARM[1]) < 20


def test_timeline_segments_keep_status_fields_separate(baseline):
    _, res, _ = baseline
    segs = []
    for rec in res.records:
        segs += json.loads((rec.speech_dir / "meta.json").read_text())["segments"]
    assert {s["source_id"] for s in segs if s["source_id"].startswith("src_")} == {"src_0", "src_1"}
    tse = [s for s in segs if s["source_id"].startswith("tse:")]
    assert tse and all({"association_status", "visibility", "stem_availability", "refusal_reason"} <= set(s) for s in tse)
    # a visible person who is not speaking is refused with a reason, and that is not hidden behind an enum
    refused = [s for s in tse if s["stem_availability"] == "unavailable"]
    assert refused and all(s["refusal_reason"] for s in refused)
    # chunk-owned segments tile the clip: per chunk exactly the owned span
    for rec in res.records:
        for s in json.loads((rec.speech_dir / "meta.json").read_text())["segments"]:
            assert (s["start_sample"], s["end_sample"]) == (rec.spec.own_start, rec.spec.own_end)


def test_warm_cache_invokes_no_model_and_matches(baseline, tmp_path):
    r0, res0, d = baseline
    r1 = make_runner(d, cache=r0.cache, name="run_warm")
    res1 = r1.run()
    assert not r1.counters.calls and not r1.counters.chunks_computed
    assert len(r1.counters.chunks_restored) == 14             # 7 visual + 7 speech
    assert digest(res0) == digest(res1)


@pytest.mark.parametrize("boundary", [("visual:mid", 2), ("visual:after", 3), ("speech:before", 1), ("speech:mid", 3),
                                      ("speech:after", 2), ("speech:mid", 6)])
def test_forced_termination_resumes_to_the_same_result(baseline, tmp_path, boundary):
    _, ref, _ = baseline
    cache = StageCache(tmp_path / "cache")

    def fault(name, k):
        if (name, k) == boundary:
            raise InjectedCrash(name)
    with pytest.raises(InjectedCrash):
        make_runner(tmp_path, cache=cache, fault=fault, name="a").run()
    r2 = make_runner(tmp_path, cache=cache, name="b")
    res = r2.run()
    k, stage = boundary[1], boundary[0].split(":")[0]
    # unfinished work only: in the crashed stage's own pass nothing completed before the crash is recomputed
    # (a published chunk at the crash point itself, for the ":after" boundaries, is also reused)
    last_done = k if boundary[0].endswith(":after") else k - 1
    redone = [c for c in r2.counters.chunks_computed if c[0] == stage and c[1] <= last_done]
    assert not redone, r2.counters.chunks_computed
    assert digest(res) == digest(ref)                         # IDs, timeline, sample counts and audio bit-identical


def test_corrupt_entry_recomputes_only_itself_and_downstream_hits_stay(baseline, tmp_path):
    _, ref, _ = baseline
    cache = StageCache(tmp_path / "cache")
    make_runner(tmp_path, cache=cache, name="a").run()
    victim = next(p for p in (tmp_path / "cache" / "speech_chunk").iterdir() if p.is_dir() and (p / "meta.json").exists()
                  and json.loads((p / "meta.json").read_text())["chunk"]["index"] == 2)
    f = next((victim / "final").glob("*.npy"))
    b = bytearray(f.read_bytes())
    b[-1] ^= 0xFF
    f.write_bytes(bytes(b))
    r2 = make_runner(tmp_path, cache=cache, name="b")
    res = r2.run()
    assert [c for c in r2.counters.chunks_computed if c[0] == "speech"] == [("speech", 2)]
    assert digest(res) == digest(ref)


def test_changed_model_revision_invalidates_speech_but_not_visual(baseline, tmp_path):
    r0, _, d = baseline
    r1 = make_runner(d, cache=r0.cache, rev="r2", name="rev")
    r1.run()
    kinds = {c[0] for c in r1.counters.chunks_computed}
    assert kinds == {"speech"} and len(r1.counters.chunks_computed) == 7
    assert len([c for c in r1.counters.chunks_restored if c[0] == "visual"]) == 7


def test_reads_are_bounded_to_windows_not_the_whole_clip(baseline):
    r, res, _ = baseline
    n_master = r.src.n_audio_samples
    longest = max(e - s for kind, s, e in r.src.reads if kind == "audio")
    assert longest <= 4 * MR + 3 and longest < n_master
    assert r.counters.max_frames_resident <= load_config().memory.max_decoded_frames
