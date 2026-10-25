"""Speech branch wiring with fake separator / TalkNet (mocked; no weights)."""

import numpy as np

from av_scene.speech import SR16, frame_envelope, run_speech_branch

DUR = 8
N25 = DUR * 25


def _bursts(on: list[tuple[float, float]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(DUR * SR16, np.float32)
    for a, b in on:
        x[int(a * SR16):int(b * SR16)] = rng.standard_normal(int((b - a) * SR16)) * 0.1
    return x


def _speaking(on, n=N25):
    s = np.full(n, -4.0)
    for a, b in on:
        s[int(a * 25):int(b * 25)] = 4.0
    return s


class FakeSep:
    def __init__(self, a, b):
        self.out = np.stack([b, a])            # permuted on purpose

    def __call__(self, x):
        return self.out


class FakeScorer:
    def __init__(self, by_first_box):
        self.by = by_first_box

    def score_track(self, audio, boxes, frames, fps):
        return self.by[boxes[0]], None, {}


A_ON = [(0, 1.5), (3, 4.5), (6, 7)]
B_ON = [(1.5, 3), (4.5, 6)]
BOX_A, BOX_B = (0, 0, 10, 10), (20, 0, 30, 10)
FRAMES = np.zeros((N25, 32, 32, 3), np.uint8)


def _run(a_on, b_on, speak_a, speak_b, offscreen=None):
    a, b = _bursts(a_on, 1), _bursts(b_on, 2)
    if offscreen is not None:
        b = offscreen
    scorer = FakeScorer({BOX_A: _speaking(speak_a), BOX_B: _speaking(speak_b)})
    return run_speech_branch(a + b, FRAMES, 25.0, {"tA": [BOX_A] * N25, "tB": [BOX_B] * N25},
                             separator=FakeSep(a, b), scorer=scorer)


def test_selecting_each_person_returns_different_streams():
    r = _run(A_ON, B_ON, A_ON, B_ON)
    sa, ea = r.selection["tA"]
    sb, eb = r.selection["tB"]
    assert ea.status == "ok" and eb.status == "ok"
    assert sa != sb
    np.testing.assert_array_equal(r.streams[sa][:SR16], _bursts(A_ON, 1)[:SR16])


def test_silent_target_gets_no_stream():
    r = _run([], B_ON, [], B_ON)
    sa, ea = r.selection["tA"]
    assert sa is None and ea.status == "unavailable"
    assert r.selection["tB"][1].status == "ok"


def test_offscreen_talker_not_attributed_to_visible_people():
    off = _bursts([(0, 8)], 3)
    r = _run([], [], [], [], offscreen=off)
    assert all(s is None for s, _ in r.selection.values())
    voice = [x for x in r.associations if x.status != "matched" and x.visibility == "outside_view"]
    assert voice, [(x.source_id, x.status, x.visibility) for x in r.associations]


def test_missing_face_is_no_evidence_not_silence():
    a = _bursts(A_ON, 1)
    r = run_speech_branch(a, FRAMES, 25.0, {"tA": [None] * N25}, separator=FakeSep(a, np.zeros_like(a)),
                          scorer=FakeScorer({}))
    assert np.isnan(r.track_scores["tA"]).all()
    assert r.selection["tA"][0] is None


def test_envelope():
    env = frame_envelope(np.ones(SR16, np.float32), 25)
    assert env.shape == (25,) and np.allclose(env, 1.0)


class FakeTSE:
    """Returns the clean voice of the person whose box it is given (or a leak of another voice)."""

    def __init__(self, by_box):
        self.by = by_box

    def extract_target(self, audio, boxes, frames, fps):
        return self.by[boxes[0]], {}


def test_tse_path_selects_each_person_and_gates_silent_and_conflicts():
    a, b = _bursts(A_ON, 1), _bursts(B_ON, 2)
    boxes = {"tA": [BOX_A] * N25, "tB": [BOX_B] * N25}
    r = run_speech_branch(a + b, FRAMES, 25.0, boxes, separator=FakeSep(a, b),
                          scorer=FakeScorer({BOX_A: _speaking(A_ON), BOX_B: _speaking(B_ON)}),
                          tse=FakeTSE({BOX_A: a, BOX_B: b}))
    (sa, ea), (sb, eb) = r.selection["tA"], r.selection["tB"]
    assert (sa, sb) == ("tse:tA", "tse:tB") and ea.status == eb.status == "ok"
    assert {x.track_id for x in r.associations if x.status == "matched"} == {"tA", "tB"}

    def leak(a_on):
        # silent A is (falsely) "active" and its TSE output is B's voice
        return run_speech_branch(b, FRAMES, 25.0, boxes, separator=FakeSep(np.zeros_like(b), b),
                                 scorer=FakeScorer({BOX_A: _speaking(a_on), BOX_B: _speaking(B_ON)}),
                                 tse=FakeTSE({BOX_A: b, BOX_B: b}))

    # false activity while B's voice is silent: the voice doesn't agree with A's face
    r = leak([(0, 1.2)])
    assert r.selection["tA"][0] is None and "inactive while face speaks" in r.selection["tA"][1].reason
    assert r.selection["tB"][0] == "tse:tB"
    # false activity overlapping B's speech, B clearly more active (3.0 s vs 1.3 s): B keeps it, A refused
    r = leak([(1.5, 2.8)])
    assert r.selection["tA"][0] is None and "same voice as tB" in r.selection["tA"][1].reason
    assert r.selection["tB"][0] == "tse:tB"
    # comparable activity (2.5 s vs 3.0 s): nobody gets the voice
    r = leak([(1.5, 3.0), (4.5, 5.5)])
    assert r.selection["tA"][0] is None and r.selection["tB"][0] is None
    assert any(x.status == "ambiguous" for x in r.associations)


def test_tse_path_offscreen_voice_is_outside_view():
    off = _bursts([(0, 8)], 3)
    boxes = {"tA": [BOX_A] * N25, "tB": [BOX_B] * N25}
    r = run_speech_branch(off, FRAMES, 25.0, boxes, separator=FakeSep(np.zeros_like(off), off),
                          scorer=FakeScorer({BOX_A: _speaking([]), BOX_B: _speaking([])}),
                          tse=FakeTSE({BOX_A: off, BOX_B: off}))
    assert all(s is None for s, _ in r.selection.values())
    assert any(x.visibility == "outside_view" for x in r.associations)
    assert not any(x.status == "matched" for x in r.associations)


def test_tse_path_three_talkers_is_degraded():
    """Three visible talkers into a 2-voice separator: owners keep a stream but it is degraded."""
    box_c = (40, 0, 50, 10)
    a, b, c = _bursts(A_ON, 1), _bursts(B_ON, 2), _bursts([(0, 8)], 3)
    boxes = {"tA": [BOX_A] * N25, "tB": [BOX_B] * N25, "tC": [box_c] * N25}
    r = run_speech_branch(a + b + c, FRAMES, 25.0, boxes, separator=FakeSep(a, b),
                          scorer=FakeScorer({BOX_A: _speaking(A_ON), BOX_B: _speaking(B_ON),
                                             box_c: _speaking([(0, 8)])}),
                          tse=FakeTSE({BOX_A: a, BOX_B: b, box_c: c}))
    assert r.info["n_suspected_talkers"] == 3
    assert r.selection["tA"][1].status == r.selection["tB"][1].status == "degraded"
    assert r.selection["tC"][0] is None
    # explicit count still wins
    r2 = run_speech_branch(a + b, FRAMES, 25.0, {k: boxes[k] for k in ("tA", "tB")}, separator=FakeSep(a, b),
                           scorer=FakeScorer({BOX_A: _speaking(A_ON), BOX_B: _speaking(B_ON)}),
                           tse=FakeTSE({BOX_A: a, BOX_B: b}), n_suspected_talkers=2)
    assert r2.selection["tA"][1].status == "ok" and r2.info["n_suspected_talkers"] == 2


def _tse_run(speak_a, speak_b, **kw):
    a, b = _bursts(A_ON, 1), _bursts(B_ON, 2)
    return run_speech_branch(a + b, FRAMES, 25.0, {"tA": [BOX_A] * N25, "tB": [BOX_B] * N25},
                             separator=FakeSep(a, b),
                             scorer=FakeScorer({BOX_A: _speaking(speak_a), BOX_B: _speaking(speak_b)}),
                             tse=FakeTSE({BOX_A: a, BOX_B: b}), **kw)


def test_activity_gate_rescues_talknet_miss_only_when_enabled():
    off = _tse_run(A_ON, [])                                  # TalkNet misses B entirely
    assert off.selection["tB"][0] is None
    on = _tse_run(A_ON, [], activity_gate=True)
    assert on.selection["tB"][1].status == "ok" and on.selection["tA"][1].status == "ok"
    assert "activity_gate" in on.info


def test_activity_gate_keeps_silent_bystander_gated():
    a, b = _bursts(A_ON, 1), _bursts([], 2)
    leak = 0.12 * a                                            # B's TSE is only A bleeding through (-18.7 dB)
    r = run_speech_branch(a, FRAMES, 25.0, {"tA": [BOX_A] * N25, "tB": [BOX_B] * N25}, separator=FakeSep(a, b),
                          scorer=FakeScorer({BOX_A: _speaking(A_ON), BOX_B: _speaking([])}),
                          tse=FakeTSE({BOX_A: a, BOX_B: leak}), activity_gate=True)
    assert r.selection["tB"][0] is None and r.selection["tA"][0] is not None


def test_peel_adds_third_voice_only_when_enabled():
    a, b, c = _bursts(A_ON, 1), _bursts(B_ON, 2), _bursts([(0, 8)], 3)
    kw = dict(separator=FakeSep(a, b), scorer=None)
    base = run_speech_branch(a + b + c, FRAMES, 25.0, {"tA": [BOX_A] * N25}, **kw)
    peeled = run_speech_branch(a + b + c, FRAMES, 25.0, {"tA": [BOX_A] * N25}, peel=True, **kw)
    assert len(base.streams) == 2
    assert len(peeled.streams) >= 2 and "peeling" in peeled.info


def test_offscreen_upgrade_marks_unexplained_voice_outside_view_with_reason():
    a, b = _bursts(A_ON, 1), _bursts([(0.5, 2), (3.5, 5)], 2)   # overlaps A: old rule says 'unknown'
    r0 = run_speech_branch(a + b, FRAMES, 25.0, {"tA": [BOX_A] * N25}, separator=FakeSep(a, b),
                           scorer=FakeScorer({BOX_A: _speaking(A_ON)}), tse=FakeTSE({BOX_A: a}))
    assert not [x for x in r0.associations if x.visibility == "outside_view"]
    r = run_speech_branch(a + b, FRAMES, 25.0, {"tA": [BOX_A] * N25}, separator=FakeSep(a, b),
                          scorer=FakeScorer({BOX_A: _speaking(A_ON)}), tse=FakeTSE({BOX_A: a}),
                          offscreen_upgrade=True)
    ov = [x for x in r.associations if x.visibility == "outside_view"]
    assert ov and "no visible face's TSE output resembles it" in ov[0].components[0].note


def test_activity_gate_updates_reported_activity():
    off = _tse_run(A_ON, [])
    on = _tse_run(A_ON, [], activity_gate=True)
    assert not off.activity_us["tB"] and on.activity_us["tB"]


def test_peel_empty_audio_falls_back_to_plain_separator():
    z = np.zeros(DUR * SR16, np.float32)
    r = run_speech_branch(z, FRAMES, 25.0, {"tA": [BOX_A] * N25}, separator=FakeSep(z, z), scorer=None, peel=True)
    assert len(r.streams) == 2


def test_offscreen_upgrade_spares_a_visible_face_talknet_missed():
    """stkM-like: B speaks on screen but TalkNet scores B silent; B's voice must not be called off-screen."""
    a, b = _bursts(A_ON, 1), _bursts([(0.5, 2), (3.5, 5)], 2)   # B overlaps A, so the old rule says 'unknown'
    r = run_speech_branch(a + b, FRAMES, 25.0, {"tA": [BOX_A] * N25, "tB": [BOX_B] * N25},
                          separator=FakeSep(a, b),
                          scorer=FakeScorer({BOX_A: _speaking(A_ON), BOX_B: _speaking([])}),
                          tse=FakeTSE({BOX_A: a, BOX_B: b}), offscreen_upgrade=True)
    assert not [x for x in r.associations if x.visibility == "outside_view"]
