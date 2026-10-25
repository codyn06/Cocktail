import numpy as np

from av_scene.activity_gate import combine_activity

SR, FPS, N = 16000, 25, 100


def _noise(seed, db):
    x = np.random.default_rng(seed).standard_normal(N * SR // FPS)
    return x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)


def test_talknet_miss_rescued_by_strong_tse():
    a, b = _noise(1, -12), _noise(2, -4)
    sc = {"A": np.ones(N), "B": np.full(N, 0.1)}
    act, d = combine_activity(sc, {"A": a, "B": b}, a + b, FPS)
    assert act["A"].all() and act["B"].mean() > 0.95
    assert d["B"]["tse_rescued_frames"] > 90


def test_equal_strength_tracks_not_rescued_without_lead():
    a, b = _noise(1, -6), _noise(2, -6)
    sc = {"A": np.ones(N), "B": np.full(N, 0.1)}
    act, _ = combine_activity(sc, {"A": a, "B": b}, a + b, FPS)
    assert not act["B"].any()


def test_silent_bystander_bleed_not_active():
    a = _noise(3, -6)
    bleed = a * 10 ** (-18.7 / 20)  # the other talker's bleed, -18.7 dB re. mix
    sc = {"A": np.ones(N), "B": np.zeros(N)}
    act, _ = combine_activity(sc, {"A": a, "B": bleed}, a, FPS)
    assert not act["B"].any() and act["A"].all()


def test_both_talknet_active_unchanged():
    a, b = _noise(5, -6), _noise(6, -30)
    sc = {"A": np.ones(N), "B": np.ones(N)}
    act, _ = combine_activity(sc, {"A": a, "B": b}, a + b, FPS)
    assert act["A"].all() and act["B"].all()


def test_strongly_negative_sync_blocks_tse():
    a, b = _noise(7, -30), _noise(8, -6)
    sc = {"A": np.zeros(N), "B": np.full(N, -3.0)}
    act, _ = combine_activity(sc, {"A": a, "B": b}, a + b, FPS)
    assert not act["B"].any()


def test_bleed_stays_gated_near_and_above_the_energy_threshold():
    a = _noise(3, -6)
    sc = {"A": np.ones(N), "B": np.zeros(N)}
    for bleed_db in (-12.0, -15.0, -18.7):
        act, _ = combine_activity(sc, {"A": a, "B": a * 10 ** (bleed_db / 20)}, a, FPS)
        assert not act["B"].any(), bleed_db
    # above the -11 dB threshold the 6 dB lead rule still blocks it: bleed cannot exceed its source track
    act, _ = combine_activity(sc, {"A": a, "B": a * 10 ** (-9.0 / 20)}, a, FPS)
    assert not act["B"].any()


def test_lone_silent_bystander_track_with_bleed_is_not_active():
    a = _noise(3, -6)                                      # only B is a tracked face; its TSE is A's bleed
    for score in (np.zeros(N), np.full(N, np.nan)):        # silent face, or no face evidence (NaN counts as 0)
        act, _ = combine_activity({"B": score}, {"B": a * 10 ** (-15.0 / 20)}, a, FPS)
        assert not act["B"].any()
