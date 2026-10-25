import numpy as np

from av_scene.associate import (associate_streams, continuity, select_for_track,
                                track_activity)

N = 250


def blocks(*spans, n=N):
    x = np.zeros(n)
    for a, b in spans:
        x[a:b] = 1.0
    return x


def env_of(x, seed=0):
    r = np.random.default_rng(seed)
    return x * (0.8 + 0.2 * r.random(len(x))) + 1e-4 * r.random(len(x))


def two_speakers():
    a_on = blocks((20, 100), (160, 200))
    b_on = blocks((60, 140), (200, 240))
    env = {"s1": env_of(a_on, 1), "s2": env_of(b_on, 2)}
    trk = {"A": 0.05 + 0.9 * a_on, "B": 0.05 + 0.9 * b_on}
    return env, trk


def test_overlap_select_A_then_B_differ():
    env, trk = two_speakers()
    assocs = associate_streams(env, trk)
    a, ea = select_for_track("A", assocs, env)
    b, eb = select_for_track("B", assocs, env)
    assert (a, b) == ("s1", "s2") and ea.status == "ok" and eb.status == "ok"


def test_target_silent_other_talks_unavailable():
    env = {"s1": env_of(blocks(), 1), "s2": env_of(blocks((20, 200)), 2)}
    trk = {"A": np.full(N, 0.05), "B": 0.05 + 0.9 * blocks((20, 200))}
    assocs = associate_streams(env, trk)
    sid, ex = select_for_track("A", assocs, env)
    assert sid is None and ex.status == "unavailable"
    sid, ex = select_for_track("A", assocs, env, target_active=False)
    assert sid is None and ex.reason == "target inactive"


def test_all_silent_nothing_matched():
    env = {"s1": np.zeros(N), "s2": np.zeros(N)}
    trk = {"A": np.full(N, 0.02), "B": np.full(N, 0.02)}
    assocs = associate_streams(env, trk)
    assert all(a.status == "unmatched" and a.track_id is None for a in assocs)
    assert all(a.visibility == "unknown" for a in assocs)


def test_offscreen_talker_outside_view():
    env = {"s1": env_of(blocks((20, 200)), 3)}
    trk = {"A": np.full(N, 0.05), "B": np.full(N, 0.08)}
    (a,) = associate_streams(env, trk)
    assert a.status == "unmatched" and a.track_id is None
    assert a.visibility == "outside_view"
    assert select_for_track("A", [a], env)[0] is None


def test_offscreen_not_claimed_when_tracks_unobservable():
    env = {"s1": env_of(blocks((20, 200)), 3)}
    (a,) = associate_streams(env, {"A": np.full(N, np.nan)})
    assert a.status == "unmatched" and a.visibility == "unknown"
    assert all(c.value is None for c in a.components if c.name == "best_track_corr")


def test_near_equal_correlations_ambiguous():
    on = blocks((20, 100), (160, 200))
    env = {"s1": env_of(on, 1)}
    trk = {"A": 0.05 + 0.9 * on, "B": 0.05 + 0.9 * on + 0.001 * np.arange(N) / N}
    (a,) = associate_streams(env, trk)
    assert a.status == "ambiguous"
    assert a.runner_up_margin is not None and a.runner_up_margin < 0.15
    sid, ex = select_for_track(a.track_id, [a], env)
    assert sid is None and ex.reason == "ambiguous"


def test_identity_swap_weak_evidence_flagged():
    env, trk = two_speakers()
    prev = associate_streams(env, trk)
    swapped = [a.model_copy(update={"track_id": "B" if a.track_id == "A" else "A",
                                    "runner_up_margin": 0.2}) for a in prev]
    assert all(a.status == "ambiguous" for a in continuity(prev, swapped))
    strong = [a.model_copy(update={"runner_up_margin": 0.6}) for a in swapped]
    assert all(a.status == "matched" for a in continuity(prev, strong))
    sim = {("s1", "s1"): 0.9, ("s2", "s2"): 0.9}
    assert all(a.status == "matched" for a in continuity(prev, swapped, sim))
    assert all(a.status == "matched" for a in continuity(prev, prev))


def test_track_activity_and_degraded():
    sc = np.full(30, np.nan)
    sc[5:15] = 0.9
    sc[20:22] = 0.9
    assert track_activity({"A": sc}, 0.5, 5) == {"A": [(200000, 600000)]}
    env, trk = two_speakers()
    sid, ex = select_for_track("A", associate_streams(env, trk), env,
                               n_suspected_talkers=3)
    assert sid == "s1" and ex.status == "degraded"
