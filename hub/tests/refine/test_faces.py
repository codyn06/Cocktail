import numpy as np

from refine.faces import FaceBoxTracker, detect_tracks


def D(*boxes):
    return np.array([[*b, 0.9] for b in boxes], float).reshape(-1, 5)


A, B = (0, 0, 10, 10), (50, 0, 60, 10)


def test_two_faces_keep_ids():
    t = FaceBoxTracker()
    for i in range(6):
        t.update(D((i, 0, 10 + i, 10), (50, 0, 60, 10)))
    r = t.result()
    assert set(r) == {"f0", "f1"} and all(b is not None for v in r.values() for b in v)
    firsts = {k: (v[0][0], v[5][0]) for k, v in r.items()}
    assert sorted(firsts.values()) == [(0.0, 5.0), (50.0, 50.0)]


def test_gap_gives_none_not_guess_and_reconnects():
    t = FaceBoxTracker()
    for i in range(7):
        t.update(D(A) if i not in (3, 4) else D())
    r = t.result()
    assert list(r) == ["f0"] and r["f0"][3] is None and r["f0"][4] is None and r["f0"][5] == A


def test_short_tracks_dropped_and_cap_three():
    t = FaceBoxTracker()
    for i in range(6):
        t.update(D(A) if i != 2 else D(A, B))      # B seen once
    assert list(t.result()) == ["f0"]
    t = FaceBoxTracker()
    boxes = [(100 * k, 0, 100 * k + 10, 10) for k in range(5)]
    for _ in range(5):
        t.update(D(*boxes))
    assert len(t.result()) == 3


def test_detect_tracks():
    class Det:
        def detect(self, f):
            return D(A)
    assert len(detect_tracks(Det(), [0] * 4)["f0"]) == 4
