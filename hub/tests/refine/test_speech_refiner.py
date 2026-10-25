import cv2
import numpy as np

from refine.contracts import RefineJob, RefineKey, RefineState
from refine.speech_refiner import SpeechRefiner

SR, DUR, FPS = 16000, 8, 25
N = DUR * FPS
BOX_A, BOX_B = (0, 0, 10, 10), (20, 0, 30, 10)


def bursts(on, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(DUR * SR, np.float32)
    for a, b in on:
        x[int(a * SR):int(b * SR)] = rng.standard_normal(int((b - a) * SR)) * 0.1
    return x


def speaking(on):
    s = np.full(N, -4.0)
    for a, b in on:
        s[int(a * 25):int(b * 25)] = 4.0
    return s


A_ON, B_ON = [(0, 1.5), (3, 4.5), (6, 7)], [(1.5, 3), (4.5, 6)]


class Det:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, rgb):
        return np.array([[*b, 0.9] for b in self.boxes], float).reshape(-1, 5)


class Scorer:
    def __init__(self, by):
        self.by = by

    def score_track(self, audio, boxes, frames, fps):
        return self.by[boxes[0]], None, {}


class Tse:
    def __init__(self, by):
        self.by = by

    def extract_target(self, audio, boxes, frames, fps):
        return self.by[boxes[0]], None


def jpegs(n=N, fps=FPS, size=(32, 32)):
    ok, buf = cv2.imencode(".jpg", np.zeros((*size, 3), np.uint8))
    return [(int(i * 1e6 / fps), buf.tobytes()) for i in range(n)]


def job(a, b, slots=("s1",), frames=None):
    audio = {"s1": a.copy(), "s2": b.copy(), "s3": np.zeros_like(a)}
    return RefineJob(1, RefineKey("t1", 0, 0, 0), slots, 0, DUR * 10**6, a + b, audio,
                     jpegs() if frames is None else frames)


def refiner(boxes, scores, tse_by):
    return SpeechRefiner(Det(boxes), Scorer(scores), Tse(tse_by))


def test_match_accepted():
    a, b = bursts(A_ON, 1), bursts(B_ON, 2)
    j = job(a, b)
    s1_before = j.slot_audio["s1"].copy()
    r = refiner([BOX_A, BOX_B], {BOX_A: speaking(A_ON), BOX_B: speaking(B_ON)}, {BOX_A: a, BOX_B: b})(j)
    assert r.state == RefineState.accepted and r.track_id is not None
    assert r.score > 3 and 0 < r.assoc_conf <= 1 and r.refined_audio is not None
    assert set(r.compute_ms) == {"decode", "faces", "speech"}
    np.testing.assert_array_equal(j.slot_audio["s1"], s1_before)


def test_stkm_like_unmatched():
    a, b = bursts(A_ON, 1), bursts(B_ON, 2)
    junk = bursts([(0, 8)], 9)
    r = refiner([BOX_A], {BOX_A: speaking(A_ON)}, {BOX_A: junk})(job(a, b))
    assert r.state == RefineState.refused and r.track_id is None and r.assoc_conf == 0


def test_no_face_refused():
    a, b = bursts(A_ON, 1), bursts(B_ON, 2)
    r = refiner([], {}, {})(job(a, b))
    assert r.state == RefineState.refused and "no visible face" in r.reason and r.track_id is None


def test_low_fps_unavailable():
    a, b = bursts(A_ON, 1), bursts(B_ON, 2)
    r = refiner([BOX_A], {}, {})(job(a, b, frames=jpegs(10, fps=2)))
    assert r.state == RefineState.unavailable and "insufficient video" in r.reason


def test_s3_unavailable():
    a, b = bursts(A_ON, 1), bursts(B_ON, 2)
    r = refiner([BOX_A], {}, {})(job(a, b, slots=("s3",)))
    assert r.state == RefineState.unavailable and "not a voice slot" in r.reason


def test_two_faces_same_voice_ambiguous():
    a, b = bursts(A_ON, 1), bursts(B_ON, 2)
    both = speaking(A_ON)
    r = refiner([BOX_A, BOX_B], {BOX_A: both, BOX_B: both}, {BOX_A: a, BOX_B: a})(job(a, b))
    assert r.state == RefineState.refused and r.track_id is None and r.assoc_conf == 0


def test_silent_faces_refused():
    a, b = bursts(A_ON, 1), bursts(B_ON, 2)
    r = refiner([BOX_A], {BOX_A: speaking([])}, {BOX_A: a})(job(a, b))
    assert r.state == RefineState.refused and "no face is speaking" in r.reason
