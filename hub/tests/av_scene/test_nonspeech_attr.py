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
                            {"c1": 0.0, "c2": 1e-4}, 1e-4)
    assert r.status == "ambiguous" and r.track_id is None


def test_one_moving_cat_matched():
    e = env()
    r = attribute_nonspeech(e, {"c1": moving(e), "c2": np.zeros(N)}, 0.0,
                            {"c1": 0.0, "c2": 0.0}, 1e-4)
    assert r.status == "matched" and r.track_id == "c1"


def test_tdoa_disambiguates_two_movers():
    e = env()
    r = attribute_nonspeech(e, {"c1": moving(e), "c2": moving(e)}, 0.0,
                            {"c1": 0.0, "c2": 5e-4}, 1e-4)
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
    r = attribute_nonspeech(e, {"c1": moving(e)}, 6e-4, {"c1": 0.0}, 1e-4)
    assert r.status == "unmatched" and r.reason == "tdoa_inconsistent"
