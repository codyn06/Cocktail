"""PersonBox: wearer tagging, fresh-single-face gating, bbox/bearing math (scripted fake detector)."""

import math
import threading
from types import SimpleNamespace

import numpy as np

import protocol as p
from events.selection import Selection
from vision.person_box import PersonBox

W, H = 640, 360


def src(sid, level):
    return p.Source(source_id=sid, label="speech", label_conf=0.9, level_db=level, last_observed_us=1)


class Fake:
    def __init__(self, dets):
        self.dets = dets
        self.calls = 0

    def detect(self, rgb):
        self.calls += 1
        return np.array(self.dets, float).reshape(-1, 5)


def yaw_quat(deg):
    a = math.radians(deg) / 2
    return (0.0, math.sin(a), 0.0, math.cos(a))


def make(dets, chosen="t2", ts=1_000_000, yaw=0.0, hfov=90.0, flipped=False):
    header = SimpleNamespace(device_ts_us=ts, hfov_deg=hfov, cam_rot_xyzw=yaw_quat(yaw))
    ingest = SimpleNamespace(pose_epoch=0, video=SimpleNamespace(
        lock=threading.Lock(), header=header, jpeg=b"x"))
    sel = Selection()
    sel.chosen = chosen
    pb = PersonBox(ingest, sel, decode=lambda j: np.zeros((H, W, 3), np.uint8), camera_flipped=flipped)
    pb.detector = Fake(dets)
    return pb, ingest, sel


def feed(pb, n=6, wearer=-40.0, other=-60.0, now=1_100_000):
    out = None
    for _ in range(n):
        out = pb.overlay([src("t1", wearer), src("t2", other)], now)
    return out


FACE = (400, 100, 440, 160, 0.95)  # cx=420/640 -> right of centre


def test_wearer_tag_margin_logic():
    pb, *_ = make([])
    for _ in range(6):
        pb.overlay([src("t1", -40)], 1)
    assert pb.wearer() is None  # single track
    pb, *_ = make([])
    feed(pb, wearer=-45, other=-50)
    assert pb.wearer() is None  # margin too small
    pb, *_ = make([])
    feed(pb)
    assert pb.wearer() == "t1"
    pb, *_ = make([])
    feed(pb, n=3)
    assert pb.wearer() is None  # < 5 hops


def test_chosen_wearer_gets_no_box():
    pb, *_ = make([FACE], chosen="t1")
    feed(pb)
    assert pb.step() == "idle"
    out = feed(pb)
    assert all(s.bbox is None for s in out)
    assert pb.status()["reason_if_not"] == "wearer"


def test_unknown_wearer_reason():
    pb, *_ = make([FACE])
    pb.overlay([src("t2", -60)], 1_100_000)
    assert pb.status()["reason_if_not"] == "unknown_wearer"


def test_one_fresh_face_boxes_with_bearing_sign_and_copies():
    pb, ingest, sel = make([FACE], yaw=30.0, hfov=90.0)
    feed(pb)
    assert pb.step() == "ok"
    originals = [src("t1", -40), src("t2", -60)]
    out = pb.overlay(originals, 1_100_000)
    t2 = next(s for s in out if s.source_id == "t2")
    assert t2.in_view and t2.bearing_source == p.BearingSource.camera and t2.bearing_conf == 0.6
    cx = 420 / 640
    fw, fh = 40 / 640, 60 / 360
    exp = 30.0 + math.degrees(math.atan((2 * cx - 1) * math.tan(math.radians(45))))
    assert abs(t2.bearing_deg - exp) < 1e-3 and t2.bearing_deg > 30.0  # right of centre -> more right
    x0, y0, x1, y1 = t2.bbox
    assert abs(x0 - (cx - 1.5 * fw)) < 1e-6 and abs(x1 - (cx + 1.5 * fw)) < 1e-6
    assert abs(y0 - (100 / 360 - 0.4 * fh)) < 1e-6 and abs(y1 - (160 / 360 + 1.8 * fh)) < 1e-6
    assert originals[1].bbox is None and originals[1].bearing_deg is None  # copies, input untouched
    assert next(s for s in out if s.source_id == "t1").bbox is None
    assert t2.last_observed_us == 1
    assert pb.status()["published"] is True


def test_flipped_mirrors_y():
    pb, *_ = make([FACE], flipped=True)
    feed(pb)
    pb.step()
    t2 = next(s for s in pb.overlay([src("t1", -40), src("t2", -60)], 1_100_000) if s.source_id == "t2")
    assert abs(t2.bbox[1] - (1 - min(1.0, 160 / 360 + 1.8 * 60 / 360))) < 1e-6


def test_two_faces_none():
    pb, *_ = make([FACE, (50, 100, 90, 160, 0.9)])
    feed(pb)
    assert pb.step() == "multiple_faces"
    out = feed(pb)
    assert all(s.bbox is None for s in out)
    assert pb.status()["reason_if_not"] == "multiple_faces"
    assert pb.status()["last_face_count"] == 2


def test_low_conf_or_tiny_face_ignored():
    pb, *_ = make([(400, 100, 440, 160, 0.5), (10, 10, 20, 20, 0.99)])
    feed(pb)
    assert pb.step() == "no_face"


def test_stale_frame_none_and_expiry():
    pb, *_ = make([FACE], ts=100_000)
    feed(pb, now=1_100_000)
    assert pb.step() == "stale_frame"
    pb, *_ = make([FACE], ts=1_000_000)
    feed(pb)
    pb.step()
    out = pb.overlay([src("t1", -40), src("t2", -60)], 1_000_000 + 600_000)
    assert all(s.bbox is None for s in out)
    assert pb.status()["reason_if_not"] == "stale_frame"


def test_no_pose():
    pb, ingest, _ = make([FACE])
    ingest.video.header.hfov_deg = None
    feed(pb)
    assert pb.step() == "no_pose"


def test_selection_change_clears():
    pb, _, sel = make([FACE])
    feed(pb)
    pb.step()
    sel.chosen = "t1"
    out = feed(pb, n=1)
    assert all(s.bbox is None for s in out)
    sel.chosen = "t2"
    assert pb._box is None


def test_epoch_reset_clears():
    pb, ingest, _ = make([FACE])
    feed(pb)
    pb.step()
    ingest.pose_epoch = 1
    out = pb.overlay([src("t1", -40), src("t2", -60)], 1_100_000)
    assert all(s.bbox is None for s in out)


def test_same_frame_not_reprocessed_and_busy_skipped():
    pb, *_ = make([FACE])
    feed(pb)
    assert pb.step() == "ok"
    assert pb.step() == "same_frame"
    assert pb.detector.calls == 1
    pb._last_frame_us = -1
    pb._busy.acquire()
    assert pb.step() == "busy"
    assert pb.detector.calls == 1


def test_detector_loading_reason():
    pb, *_ = make([FACE])
    pb.detector = None
    feed(pb)
    assert pb.status()["reason_if_not"] == "detector_loading"
