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
