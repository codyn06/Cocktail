import numpy as np

from av_scene.nonspeech_attr import (Candidate, absent_target_gate, attribute_nonspeech,
                                     mask_motion_series, padded_event_windows, rank_candidates)

N = 60
rng = np.random.default_rng(0)


def env():
    e = np.zeros(N)
    e[10:20] = 1.0
    e[35:45] = 0.6
    return e


def moving(e):
    return e + 0.05 * rng.standard_normal(N)


def quiet(*tracks):
    """Silent-window control: the same tracks barely move and do not follow the envelope."""
    return {t: 0.02 * np.abs(rng.standard_normal(N)) for t in tracks}


def test_windows_pad_merge_clamp():
    w = padded_event_windows([(0.1, 0.3), (0.6, 0.8), (5.0, 5.9)], 0.25, 6.0, 0.2)
    assert w == [(0.0, 1.05), (4.75, 6.0)]


def test_rank_caches_judge():
    calls = []

    def judge(a, p):
        calls.append(1)
        return float(a.mean())
    cs = [Candidate("a", np.ones(4), "cat"), Candidate("b", np.full(4, 2.0), "cat")]
    cache = {}
    r = rank_candidates(cs, judge, cache)
    r2 = rank_candidates(cs, judge, cache)
    assert [c.key for c, _ in r] == ["b", "a"] and r2 == r and len(calls) == 2


def test_absent_target_rejected():
    mix = 0.1 * rng.standard_normal(16000)
    assert absent_target_gate(np.zeros(16000), mix, -50).startswith("silent")
    assert absent_target_gate(1e-4 * rng.standard_normal(16000), mix, -90).startswith("tiny")
    assert absent_target_gate(0.05 * rng.standard_normal(16000), mix, -50) is None


def test_mask_motion_series():
    a = np.zeros((8, 8), bool)
    b = a.copy()
    b[:4] = True
    m = mask_motion_series([a, a, b])
    assert m[1] == 0 and m[2] > 0


def test_two_cats_one_chirp_ambiguous():
    e = env()
    r = attribute_nonspeech(e, {"c1": moving(e), "c2": moving(e)}, 0.0,
                            {"c1": 0.0, "c2": 1e-4}, 1e-4, control_motion=quiet("c1", "c2"))
    assert r.status == "ambiguous" and r.track_id is None


def test_one_moving_cat_matched():
    e = env()
    r = attribute_nonspeech(e, {"c1": moving(e), "c2": np.zeros(N)}, 0.0,
                            {"c1": 0.0, "c2": 0.0}, 1e-4, control_motion=quiet("c1", "c2"))
    assert r.status == "matched" and r.track_id == "c1"


def test_tdoa_disambiguates_two_movers():
    e = env()
    r = attribute_nonspeech(e, {"c1": moving(e), "c2": moving(e)}, 0.0,
                            {"c1": 0.0, "c2": 5e-4}, 1e-4, control_motion=quiet("c1", "c2"))
    assert r.status == "matched" and r.track_id == "c1"


def test_offscreen_knock_not_attributed():
    e = env()
    r = attribute_nonspeech(e, {"cat": np.zeros(N)}, 4e-4, {"cat": 0.0}, 1e-4)
    assert r.status == "unmatched" and r.visibility == "outside_view" and r.track_id is None


def test_silent_control_blocks_spurious_match():
    e = env()
    m = moving(e)
    r = attribute_nonspeech(e, {"c1": m}, 0.0, {"c1": 0.0}, 1e-4, control_motion={"c1": m})
    assert r.status == "unmatched" and r.reason == "silent_control_explains"


def test_wrong_direction_rejects():
    e = env()
    r = attribute_nonspeech(e, {"c1": moving(e)}, 6e-4, {"c1": 0.0}, 1e-4, control_motion=quiet("c1"))
    assert r.status == "unmatched" and r.reason == "tdoa_inconsistent"


# ---- Phase 4 fixes: missing evidence is missing, not a pass ------------------------------------------
def test_missing_control_never_matches():
    e = env()
    r = attribute_nonspeech(e, {"c1": moving(e), "c2": np.zeros(N)}, 0.0, {"c1": 0.0, "c2": 0.0}, 1e-4)
    assert r.status in ("ambiguous", "unmatched") and r.track_id is None and r.reason == "control_missing"
    assert r.control_scores["c1"] is None and r.missing_evidence["control"] == ["c1"]


def test_missing_control_keeps_competitor_in_play():
    e = env()
    tr = {"c1": moving(e), "c2": moving(e)}
    r = attribute_nonspeech(e, tr, 0.0, {"c1": 0.0, "c2": 5e-4}, 1e-4, control_motion=quiet("c1"))
    # c2 is spatially excluded here, but with c2 control-missing and still correlated it must not be dropped
    # silently: c1 has full evidence and c2 is spatially inconsistent only if its delay is known
    assert r.missing_evidence["control"] == ["c2"]
    assert r.status == "ambiguous"


def test_missing_spatial_never_matches():
    e = env()
    r = attribute_nonspeech(e, {"c1": moving(e)}, None, {"c1": 0.0}, 1e-4, control_motion=quiet("c1"))
    assert r.status == "unmatched" and r.reason == "missing_spatial_evidence" and r.tdoa_ok["c1"] is None
    r2 = attribute_nonspeech(e, {"c1": moving(e)}, 0.0, {}, 1e-4, control_motion=quiet("c1"))
    assert r2.status == "unmatched" and r2.reason == "missing_spatial_evidence"


def test_two_cats_both_move_one_sounds_spatial_cannot_tell_apart():
    e = env()
    cats = {"cat_a": moving(e), "cat_b": moving(e)}          # both move with the sound; only one actually sounds
    r = attribute_nonspeech(e, cats, 1e-4, {"cat_a": 0.0, "cat_b": 2e-4}, 2e-4,
                            control_motion=quiet("cat_a", "cat_b"))
    assert r.tdoa_ok == {"cat_a": True, "cat_b": True}
    assert r.status == "ambiguous" and r.track_id is None


def test_nan_frames_are_missing_not_zero():
    e = env()
    m = moving(e)
    m[:57] = np.nan                                           # only 3 observed frames
    r = attribute_nonspeech(e, {"c1": m}, 0.0, {"c1": 0.0}, 1e-4, control_motion=quiet("c1"))
    assert r.status == "unmatched" and r.reason == "no_motion_correlation"


def test_attribution_has_no_rank_or_quality_inputs():
    import inspect
    params = set(inspect.signature(attribute_nonspeech).parameters)
    assert not params & {"rank_score", "judge_score", "score", "quality", "text_similarity", "si_sdr"}


def test_loud_leak_from_speech_stem_rejected_for_absent_noun():
    speech = 0.1 * rng.standard_normal(16000)
    mix = speech + 0.001 * rng.standard_normal(16000)
    leaked = 0.8 * speech                                   # loud, not tiny, not silent
    assert absent_target_gate(leaked, mix) is None          # old behaviour: passes
    r = absent_target_gate(leaked, mix, other_sources=[speech])
    assert r is not None and r.startswith("leaked_from_other_source(")
    real = 0.05 * rng.standard_normal(16000)                # independent sound passes
    assert absent_target_gate(real, mix + real, other_sources=[speech]) is None


def test_gate_judge_floor():
    mix = 0.1 * rng.standard_normal(16000)
    c = 0.05 * rng.standard_normal(16000)
    assert absent_target_gate(c, mix, judge_score=-1.0, judge_floor=0.0) == "judge_below_floor(-1<0)"
    assert absent_target_gate(c, mix, judge_score=1.0, judge_floor=0.0) is None
    assert absent_target_gate(c, mix, judge_score=None, judge_floor=0.0) is None
