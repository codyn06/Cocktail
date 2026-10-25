"""Camera-derived rectangle for the selected NON-wearer speaker (hub-side only; no wire change).

Policy ("tag the wearer, box only others"): the wearer is never in the camera. The wearer is the
track with the clearly loudest running-median level (>= 8 dB above the next track, >= 5 hops); with
one track or a smaller margin the wearer is UNKNOWN and nothing is boxed. A box is published only
when the chosen track is not the wearer and exactly ONE confident face (>= 0.8, height >= 5% of the
frame) is in a FRESH frame (<= 0.5 s old at tick time). Otherwise the original Source is passed
through (the direction marker is the honest fallback). Overlays return model_copy copies and never
touch the Tracker, selection, or last_observed_us.

Orientation assumption: the camera JPEG is upright (row 0 = top). The wire carries no vertical-flip
flag, so ``camera_flipped`` (CLI --camera-flipped) mirrors bbox y if the real frame is upside-down.
Verify on the headset: box the face and check it sits on the person, not mirrored vertically.
Bearing = camera yaw at capture + atan((2*cx-1)*tan(hfov/2)); + = right.
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
from collections import deque

import protocol as p
from ingest.buffers import yaw_pitch

log = logging.getLogger("person_box")

MAX_AGE_US = 500_000
MIN_PERIOD_S = 0.5
MARGIN_DB = 8.0
MIN_HOPS = 5
HIST = 40
MIN_FACE_H = 0.05
MIN_CONF = 0.8
MAX_W = 640


def _default_decode(jpeg: bytes):
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > MAX_W:
        img = cv2.resize(img, (MAX_W, max(1, round(h * MAX_W / w))))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _default_detector():
    from vision.face_detect import FaceDetector

    return FaceDetector()


class PersonBox:
    def __init__(self, ingest, selection, detector_factory=_default_detector, decode=_default_decode,
                 camera_flipped: bool = False, enabled: bool = True):
        self.ingest, self.selection = ingest, selection
        self.detector_factory, self.decode = detector_factory, decode
        self.camera_flipped, self.enabled = camera_flipped, enabled
        self.detector = None
        self.hist: dict[str, deque] = {}
        self.now_us = 0
        self._epoch = None
        self._box: dict | None = None      # {sid, bbox, bearing, frame_us}
        self._last_reason = "no_face"
        self._last_faces = 0
        self._last_frame_us = -1
        self._busy = threading.Lock()
        self._published = False
        self._reason = "no_selection"
        self._age_ms: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- wearer tagging
    def wearer(self) -> str | None:
        med = {k: statistics.median(v) for k, v in self.hist.items() if len(v) >= 1}
        if len(med) < 2:
            return None
        ranked = sorted(med, key=med.get, reverse=True)
        top, nxt = ranked[0], ranked[1]
        if len(self.hist[top]) >= MIN_HOPS and med[top] - med[nxt] >= MARGIN_DB:
            return top
        return None

    def _observe(self, sources) -> None:
        live = {s.source_id for s in sources}
        for k in [k for k in self.hist if k not in live]:
            del self.hist[k]
        for s in sources:
            if s.state == p.TrackState.tracking and s.level_db > -90:
                self.hist.setdefault(s.source_id, deque(maxlen=HIST)).append(s.level_db)

    # ---- overlay (called from EventsState.tick, on the loop)
    def overlay(self, sources: list, now_us: int) -> list:
        self.now_us = now_us
        try:
            return self._overlay(sources, now_us)
        except Exception:
            log.exception("person box overlay failed")
            return sources

    def _overlay(self, sources, now_us):
        self._published, self._age_ms = False, None
        if not self.enabled:
            self._reason = "no_selection"
            return sources
        epoch = getattr(self.ingest, "pose_epoch", None)
        if epoch != self._epoch:
            self._epoch, self._box = epoch, None
        self._observe(sources)
        sid = self.selection.chosen
        src = next((s for s in sources if s.source_id == sid), None) if sid else None
        if src is None or src.state == p.TrackState.lost:
            self._box, self._reason = None, "no_selection"
            return sources
        if self._box and self._box["sid"] != sid:
            self._box = None
        wearer = self.wearer()
        if wearer == sid:
            self._box, self._reason = None, "wearer"
            return sources
        if wearer is None:
            self._box, self._reason = None, "unknown_wearer"
            return sources
        if self.detector is None:
            self._reason = "detector_loading"
            return sources
        b = self._box
        if b is None:
            self._reason = self._last_reason
            return sources
        age = now_us - b["frame_us"]
        self._age_ms = age / 1000
        if not 0 <= age <= MAX_AGE_US:
            self._reason = "stale_frame"
            return sources
        self._published, self._reason = True, ""
        return [s.model_copy(update=dict(
            bbox=b["bbox"], bearing_deg=b["bearing"], bearing_conf=0.6,
            bearing_source=p.BearingSource.camera, in_view=True)) if s.source_id == sid else s
            for s in sources]

    # ---- detection pass (one synchronous attempt; the thread calls it at <= 2 Hz)
    def step(self) -> str:
        if self.detector is None:
            return "detector_loading"
        sid = self.selection.chosen
        wearer = self.wearer()
        if not sid or wearer is None or sid == wearer:
            return "idle"
        if not self._busy.acquire(blocking=False):
            return "busy"
        try:
            with self.ingest.video.lock:
                h, jpeg = self.ingest.video.header, self.ingest.video.jpeg
            if h is None or jpeg is None:
                return self._fail("stale_frame")
            if h.device_ts_us == self._last_frame_us:
                return "same_frame"
            self._last_frame_us = h.device_ts_us
            if self.now_us - h.device_ts_us > MAX_AGE_US:
                return self._fail("stale_frame")
            if h.hfov_deg is None or h.cam_rot_xyzw is None:
                return self._fail("no_pose")
            img = self.decode(jpeg)
            if img is None:
                return self._fail("no_face")
            ih, iw = img.shape[:2]
            dets = self.detector.detect(img)
            faces = [d for d in dets if d[4] >= MIN_CONF and (d[3] - d[1]) / ih >= MIN_FACE_H]
            self._last_faces = len(faces)
            if not faces:
                return self._fail("no_face")
            if len(faces) > 1:
                return self._fail("multiple_faces")
            x0, y0, x1, y1 = (float(v) for v in faces[0][:4])
            fx0, fx1, fy0, fy1 = x0 / iw, x1 / iw, y0 / ih, y1 / ih
            cx, fw, fh = (fx0 + fx1) / 2, fx1 - fx0, fy1 - fy0

            def c(v):
                return min(1.0, max(0.0, v))

            bx0, bx1 = c(cx - 1.5 * fw), c(cx + 1.5 * fw)
            by0, by1 = c(fy0 - 0.4 * fh), c(fy1 + 1.8 * fh)
            if self.camera_flipped:
                by0, by1 = 1 - by1, 1 - by0
            yaw = yaw_pitch(h.cam_rot_xyzw)[0]
            bearing = yaw + math.degrees(math.atan((2 * cx - 1) * math.tan(math.radians(h.hfov_deg) / 2)))
            self._box = dict(sid=sid, bbox=(bx0, by0, bx1, by1), bearing=bearing, frame_us=h.device_ts_us)
            self._last_reason = ""
            return "ok"
        except Exception:
            log.exception("person box detection failed")
            return self._fail("no_face")
        finally:
            self._busy.release()

    def _fail(self, reason: str) -> str:
        self._last_reason = reason
        self._box = None
        return reason

    # ---- thread
    def start(self) -> None:
        if self._thread is None and self.enabled:
            self._thread = threading.Thread(target=self._run, name="person-box", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            self.detector = self.detector_factory()
        except Exception:
            log.exception("person box: face detector unavailable; disabled")
            self.enabled = False
            return
        while not self._stop.wait(MIN_PERIOD_S):
            self.step()

    def status(self) -> dict:
        return dict(enabled=self.enabled, detector_ready=self.detector is not None,
                    last_face_count=self._last_faces, last_age_ms=self._age_ms,
                    published=self._published, reason_if_not=None if self._published else self._reason)
