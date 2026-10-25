"""Event-scoped non-speech: windows, bounded candidates, fallback, gate, evidence. All models are fakes (mocked)."""
import numpy as np

from av_scene.contracts import MemoryCaps
from av_scene.event_scope import (EventAdapters, EventConfig, align_evidence, bounded_event_windows,
                                  mask_area_motion, run_event_window)

SR = 48_000
rng = np.random.default_rng(1)


# ------------------------------------------------------------------ windows
def _assert_tiles(wins, lo, hi, cap):
    assert wins[0].lo == lo and wins[-1].hi == hi
    for a, b in zip(wins, wins[1:]):
        assert a.hi == b.lo and a.continues_next and b.continues_prev      # no gap, no duplicate
    assert all(w.hi - w.lo <= cap + 1e-9 for w in wins)


def test_windows_pad_merge_like_padded_event_windows():
    w = bounded_event_windows([(0.1, 0.3), (0.6, 0.8), (5.0, 5.9)], 6.0, 0.25, 0.2, 12.0)
    assert w == [(0.0, 1.05), (4.75, 6.0)]
    assert not any(x.continues_prev or x.continues_next for x in w)


def test_continuous_60s_event_is_bounded_and_tiles():
    wins = bounded_event_windows([(1.0, 61.0)], 70.0, 0.25, 0.2, 12.0)
    assert len(wins) == 6                                                   # ceil(60.5 / 12)
    _assert_tiles(wins, 0.75, 61.25, 12.0)
    assert not wins[0].continues_prev and not wins[-1].continues_next


def test_windows_separate_spans_do_not_continue():
    wins = bounded_event_windows([(0, 30), (40, 41)], 50.0, 0.0, 0.0, 12.0)
    assert (wins[-1].lo, wins[-1].hi) == (40.0, 41.0) and not wins[-1].continues_prev
    _assert_tiles(wins[:-1], 0.0, 30.0, 12.0)


def test_windows_accept_acoustic_events_and_default_cap():
    class E:
        start_us, end_us = 0, 100_000_000
    wins = bounded_event_windows([E()], 100.0)
    assert max(w.hi - w.lo for w in wins) <= MemoryCaps.event_max_window_s


# ------------------------------------------------------------------ fakes
class Fakes:
    def __init__(self, score_fn=None, judge_raises=False):
        self.sam_calls, self.judge_calls, self.clap_calls = [], [], []
        self.score_fn = score_fn or (lambda i: float(i))
        self.judge_raises = judge_raises
        self.gate = None

    def sam(self, audio, prompt, seeds):
        self.sam_calls.append(list(seeds))
        return [0.2 * np.sin(np.linspace(0, 400, len(audio))).astype(np.float32) * (1 + 0.01 * (s % 10))
                for s in seeds]

    def judge(self, cands, prompt, rate, mix=None):
        self.judge_calls.append(len(cands))
        if self.judge_raises:
            raise RuntimeError("no judge weights")
        k = len(self.judge_calls)
        return [self.score_fn(k * 10 + i) for i in range(len(cands))]

    def clap(self, cands, prompt, rate, mix=None):
        self.clap_calls.append(len(cands))
        return [0.1 * i for i in range(len(cands))]

    def tracker(self, frames, concept):
        f = len(frames)
        if frames.max() > 0:                                 # silent control window: the object sits still
            return {"t0": [np.ones((8, 8), bool) for _ in range(f)]}
        ms = []
        for i in range(f):
            m = np.zeros((8, 8), bool)
            on = (i // 5) % 2 == 0
            m[: 2 + 3 * (on and i % 2 == 1)] = True            # flickers only while the 5-frame pulse is on
            ms.append(m)
        return {"t0": ms}


def _inputs(secs=4.0, fps=25):
    n = int(secs * SR)
    f = int(secs * fps)
    return dict(mix_48k=(0.2 * rng.standard_normal(n)).astype(np.float32), prompt="dog barking", nouns=["dog"],
                frames=np.zeros((f, 4, 4, 3), np.uint8), frame_pts_us=[int(i * 1e6 / fps) for i in range(f)])


def _adapters(fk, **over):
    d = dict(sam_audio=fk.sam, judge=fk.judge, clap=fk.clap, tracker=fk.tracker)
    d.update(over)
    return EventAdapters(**d)


# ------------------------------------------------------------------ candidates
def test_candidate_cap_batch_and_recorded_seeds():
    fk = Fakes()
    r = run_event_window((10.0, 14.0), config=EventConfig(max_candidates=4, candidate_batch=2, seed_base=1000),
                         adapters=_adapters(fk), **_inputs())
    assert fk.sam_calls == [[1000, 1001], [1002, 1003]]                    # <=4 total, batches of 2
    assert fk.judge_calls == [2, 2]
    assert [c.seed for c in r.candidates] == [1000, 1001, 1002, 1003] and r.seeds == [1000, 1001, 1002, 1003]
    assert r.ranker_used == "judge" and all(c.ranker == "judge" for c in r.candidates)
    assert r.chosen_seed == 1003                                           # highest judge score (later batch)
    assert r.status in ("ok", "degraded") and r.stem is not None and len(r.stem) == 4 * SR
    assert r.window_start_sample == 10 * SR and r.window_end_sample == 14 * SR and r.stem_start_sample == 10 * SR
    assert set(vars(r.candidates[0])) == {"seed", "rank_score", "ranker", "rms_db", "gate"}


def test_default_caps_follow_memory_caps():
    c = EventConfig()
    assert (c.max_candidates, c.candidate_batch) == (MemoryCaps.max_candidates, MemoryCaps.candidate_batch)


def test_only_best_waveform_retained():
    fk = Fakes()
    r = run_event_window((0, 4), config=EventConfig(), adapters=_adapters(fk), **_inputs())
    assert len([v for v in vars(r).values() if isinstance(v, np.ndarray)]) == 1     # the chosen stem only
    assert not any(isinstance(v, np.ndarray) for c in r.candidates for v in vars(c).values())


def test_judge_missing_falls_back_to_clap_degraded():
    fk = Fakes()
    r = run_event_window((0, 4), config=EventConfig(), adapters=_adapters(fk, judge=None), **_inputs())
    assert r.ranker_used == "clap" and r.status == "degraded" and "judge_missing" in r.extraction.reason


def test_judge_raises_then_clap_missing_marks_unranked_degraded():
    fk = Fakes(judge_raises=True)
    r = run_event_window((0, 4), config=EventConfig(), adapters=_adapters(fk, clap=None), **_inputs())
    assert r.ranker_used == "none" and r.status == "degraded" and r.extraction.reason.startswith("unranked")
    assert "judge_failed" in r.extraction.reason and "clap_missing" in r.extraction.reason
    assert r.chosen_seed == 1000                                           # seed order
    assert all(c.rank_score is None and c.ranker == "none" for c in r.candidates)
    assert fk.judge_calls == [2]                                           # dead after the first failure


def test_ranker_none_is_unranked_even_with_adapters():
    fk = Fakes()
    r = run_event_window((0, 4), config=EventConfig(ranker="none"), adapters=_adapters(fk), **_inputs())
    assert r.status == "degraded" and r.ranker_used == "none" and fk.judge_calls == [] and fk.clap_calls == []


def test_missing_sam_audio_is_unavailable():
    r = run_event_window((0, 4), config=EventConfig(), adapters=EventAdapters(), **_inputs())
    assert r.status == "unavailable" and r.stem is None
    assert r.extraction.reason == "sam_audio_adapter_missing"


# ------------------------------------------------------------------ gate
def test_absent_target_with_loud_leaked_speech_rejected():
    speech = (0.3 * rng.standard_normal(4 * SR)).astype(np.float32)
    inp = _inputs()
    inp["mix_48k"] = speech.copy()
    fk = Fakes()
    ad = _adapters(fk, sam_audio=lambda audio, prompt, seeds: [0.8 * speech for _ in seeds])  # "dog" -> speech
    r = run_event_window((0, 4), config=EventConfig(), adapters=ad, other_sources=[speech], **inp)
    assert r.gate.startswith("leaked_from_other_source(") and r.stem is None
    assert r.status == "unavailable" and "absent_target" in r.extraction.reason
    assert r.attribution.status == "unmatched" and r.attribution.reason == "target_absent"
    r2 = run_event_window((0, 4), config=EventConfig(), adapters=ad, **inp)   # no other_sources known
    assert r2.gate is None


def test_judge_floor_rejects():
    fk = Fakes(score_fn=lambda i: -5.0)
    r = run_event_window((0, 4), config=EventConfig(min_judge_score=0.0), adapters=_adapters(fk), **_inputs())
    assert r.gate.startswith("judge_below_floor") and r.stem is None


# ------------------------------------------------------------------ evidence
def test_align_evidence_missing_stays_missing():
    pts = [0, 40_000, 80_000, 120_000]
    ev = align_evidence(pts, envelope=(np.array([0, 100_000]), np.array([0.0, 1.0])),
                        track_masks={"a": [None, np.ones((2, 2), bool), np.ones((2, 2), bool), None]})
    assert ev.pts_us == pts and ev.spatial is None and ev.spatial_valid is None       # missing series: None
    assert list(ev.envelope_valid) == [True, True, True, False]                        # beyond source span
    assert np.isnan(ev.envelope[3])
    assert list(ev.track_valid["a"]) == [False, True, True, False]
    assert np.isnan(ev.track_motion["a"][1]) and ev.track_motion["a"][2] == 0.0         # frame 1: no previous mask
    assert np.isnan(ev.track_area["a"][0])
    ev2 = align_evidence(pts, spatial=(np.array([0, 120_000]), np.array([1e-4, 3e-4])))
    assert ev2.envelope is None and ev2.spatial_valid.all() and abs(ev2.spatial[2] - (1e-4 + 2e-4 * 80 / 120)) < 1e-12


def test_mask_area_motion_no_zero_fill():
    a, m, v = mask_area_motion([None, None])
    assert np.isnan(a).all() and np.isnan(m).all() and not v.any()


def _pulsed(fk):
    """Mix and every SAM candidate share a 5-frame on/off pulse so the envelope tracks the mask motion."""
    inp = _inputs()
    n = len(inp["mix_48k"])
    g = (((np.arange(n) / SR * 25 / 5).astype(int) % 2) == 0).astype(np.float32)
    inp["mix_48k"] = (0.2 * rng.standard_normal(n) * (0.2 + g)).astype(np.float32)
    ad = _adapters(fk, sam_audio=lambda audio, prompt, seeds: [
        (0.2 * rng.standard_normal(len(audio)) * (0.2 + g)).astype(np.float32) for _ in seeds])
    return inp, ad


SPATIAL = dict(tdoa_expected={"dog:t0": 0.0}, spatial=([0, 4_000_000], [0.0, 0.0]))
CONTROL = np.full((100, 4, 4, 3), 255, np.uint8)


def test_missing_control_or_spatial_never_matched_in_run():
    fk = Fakes()
    inp, ad = _pulsed(fk)
    r = run_event_window((0, 4), adapters=ad, **inp, **SPATIAL)                     # no control
    assert r.attribution.status != "matched" and r.attribution.reason == "control_missing"
    inp, ad = _pulsed(fk)
    r = run_event_window((0, 4), adapters=ad, control_frames=CONTROL, tdoa_expected={"dog:t0": 0.0}, **inp)
    assert r.attribution.status != "matched" and r.attribution.reason == "missing_spatial_evidence"


def test_full_evidence_matches_and_rank_score_is_irrelevant():
    out = []
    for fn in (lambda i: float(i), lambda i: 7.0 + 1000.0 * (i % 3)):               # wildly different judge scales
        fk = Fakes(score_fn=fn)
        inp, ad = _pulsed(fk)
        out.append(run_event_window((0, 4), adapters=ad, control_frames=CONTROL, **inp, **SPATIAL))
    for r in out:
        assert r.attribution.status == "matched" and r.attribution.track_id == "dog:t0", r.attribution
    assert out[0].attribution.scores.keys() == out[1].attribution.scores.keys()


# ------------------------------------------------------------------ review fixes
def test_gated_top_ranked_candidate_does_not_hide_a_passing_one():
    speech = (0.3 * rng.standard_normal(4 * SR)).astype(np.float32)
    inp = _inputs()
    inp["mix_48k"] = speech.copy()
    clean = (0.1 * rng.standard_normal(4 * SR)).astype(np.float32)
    fk = Fakes(score_fn=lambda i: 5.0 if i % 10 == 0 else 1.0)
    ad = _adapters(fk, sam_audio=lambda audio, prompt, seeds: [0.8 * speech, clean][: len(seeds)])
    r = run_event_window((0, 4), config=EventConfig(max_candidates=2, candidate_batch=2, seed_base=500),
                         adapters=ad, other_sources=[speech], **inp)
    assert r.candidates[0].gate and r.candidates[0].gate.startswith("leaked_from_other_source")   # top-ranked, leaked
    assert r.candidates[1].gate is None
    assert r.gate is None and r.stem is not None and r.chosen_seed == 501    # the passing candidate wins


def test_control_from_a_different_location_is_not_borrowed_by_a_colliding_track_id():
    fk = Fakes()
    inp, ad = _pulsed(fk)

    def tracker(frames, concept):
        if frames.max() > 0:                                       # control window: SAME id "t0", other object elsewhere
            return {"t0": [np.concatenate([np.zeros((6, 8), bool), np.ones((2, 8), bool)]) for _ in range(len(frames))]}
        return fk.tracker(frames, concept)
    ad = EventAdapters(sam_audio=ad.sam_audio, judge=ad.judge, clap=ad.clap, tracker=tracker)
    r = run_event_window((0, 4), adapters=ad, control_frames=CONTROL, **inp, **SPATIAL)
    assert r.attribution.status != "matched" and r.attribution.reason == "control_missing"
