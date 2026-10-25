"""Speech branch wiring with fake separator / TalkNet (mocked; no weights)."""

import numpy as np

from av_scene.speech import SR16, frame_envelope, head_box, run_speech_branch

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


def test_head_box_and_envelope():
    m = np.zeros((100, 60), bool)
    m[10:90, 20:40] = True
    x0, y0, x1, y1 = head_box(m)
    assert (x0, y0, x1) == (20, 10, 40) and y1 == 10 + 1.1 * 20
    assert head_box(np.zeros((5, 5), bool)) is None
    env = frame_envelope(np.ones(SR16, np.float32), 25)
    assert env.shape == (25,) and np.allclose(env, 1.0)
