"""Face-to-track assignment and NMS (pure numpy; detector not loaded)."""

import numpy as np

from vision.face_detect import _nms, assign_faces, face_boxes_by_track


def _mask(h, w, x0, x1):
    m = np.zeros((h, w), bool)
    m[:, x0:x1] = True
    return m


def test_assign_face_to_track_containing_its_centre():
    faces = np.array([[60, 10, 80, 30, 0.95], [10, 10, 30, 30, 0.9]])
    a = assign_faces(faces, {"L": _mask(50, 100, 0, 50), "R": _mask(50, 100, 50, 100)})
    assert a["L"] == (10, 10, 30, 30) and a["R"] == (60, 10, 80, 30)


def test_no_face_in_mask_is_none_and_face_used_once():
    faces = np.array([[10, 10, 30, 30, 0.9]])
    both = _mask(50, 100, 0, 100)
    a = assign_faces(faces, {"P": both, "Q": both, "R": None})
    assert sum(v is not None for v in a.values()) == 1 and a["R"] is None
    assert assign_faces(np.empty((0, 5)), {"P": both}) == {"P": None}


def test_nms_drops_overlaps():
    b = np.array([[0, 0, 10, 10, 0.9], [1, 1, 10, 10, 0.8], [50, 50, 60, 60, 0.7]])
    assert sorted(_nms(b, 0.1).tolist()) == [0, 2]


def test_boxes_by_track_per_frame():
    class Det:
        def detect(self, rgb):
            return np.array([[10, 10, 30, 30, 0.9]]) if rgb[0, 0, 0] else np.empty((0, 5))

    frames = np.zeros((3, 50, 100, 3), np.uint8)
    frames[1, 0, 0, 0] = 1
    m = _mask(50, 100, 0, 50)
    out = face_boxes_by_track(Det(), frames, {"L": [m, m, None]})
    assert out["L"] == [None, (10.0, 10.0, 30.0, 30.0), None]


def test_overlapping_masks_give_face_to_mask_covering_it():
    near = np.zeros((50, 100), bool)
    near[5:35, 5:35] = True                               # small, covers the face box
    far = np.ones((50, 100), bool)                        # large background person also contains the centre
    faces = np.array([[10, 10, 30, 30, 0.9]])
    for order in ({"far": far, "near": near}, {"near": near, "far": far}):
        a = assign_faces(faces, order)
        assert a["near"] == (10, 10, 30, 30) and a["far"] is None
