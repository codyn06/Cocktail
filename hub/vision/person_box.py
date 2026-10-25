"""Camera box for the selected speaker, placed only when the voice is associated with a face.

Pipeline (hub-side only; no wire change; one background thread):
  * every new camera frame (~8-10 fps, newest-only buffer): cheap grayscale decode (<= 320 px wide) and
    a per-face mouth-motion sample (vision.av_match.mouth_motion) on tracker-carried boxes;
  * <= 2 Hz: S3FD face detection on the GPU -> vision.av_match.FaceTracker (ids f1.., <= 4 faces);
  * vision.av_match.AVMatch learns which speaker id (hub/speakers recorder, same device clock) drives
    which face by contrast of mouth motion in "speaker alone" time bins; overlap adds no evidence.
The overlay puts bbox/bearing on the chosen Source ONLY if its speaker has an association, its face track
was detected recently, and the newest frame is fresh. Otherwise the Source passes through unchanged
(honest fallback: no box). Overlays return model_copy copies; Tracker/selection/route are never touched.

When it runs (CPU/GPU budget): while something is selected, plus a 10 s "pre-learn" window starting when
the recorder first reports >= 2 speakers (so association can form before the operator selects). Otherwise
the thread idles (no decode, no detection).

Freshness: the newest processed frame must be <= 0.5 s old at tick time; the face's last detection must be
<= 0.75 s older than that frame (detections come at ~2 Hz, so a strict 0.5 s would flicker); the box is the
tracker-carried box at the newest frame with that frame's camera yaw.

Orientation: camera JPEG assumed upright. ``camera_flipped`` (--camera-flipped) mirrors bbox y.
Bearing = camera yaw at capture + atan((2*cx-1)*tan(hfov/2)); + = right.
"""

from __future__ import annotations

import logging
import math
import threading
import time

import protocol as p
from ingest.buffers import yaw_pitch
from vision.av_match import AVMatch, FaceTracker, mouth_motion, predict_box, HISTORY_US

log = logging.getLogger("person_box")

MAX_AGE_US = 500_000
DET_FRESH_US = 750_000
DET_PERIOD_S = 0.5
POLL_S = 0.04
ASSOC_PERIOD_S = 0.25
PRELEARN_S = 10.0
MIN_FACE_H = 0.05
MIN_CONF = 0.5   # S3FD scores a sideways face only ~0.8 (real headset frame); the mouth-motion match guards false faces
MAX_W = 640
GRAY_W = 320


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


def _default_gray(jpeg: bytes, width: int | None = None):
    import cv2
    import numpy as np

    flag = cv2.IMREAD_GRAYSCALE
    if width and width >= 1280:
        flag = cv2.IMREAD_REDUCED_GRAYSCALE_4
    elif width and width >= 640:
        flag = cv2.IMREAD_REDUCED_GRAYSCALE_2
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), flag)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > GRAY_W:
        img = cv2.resize(img, (GRAY_W, max(1, round(h * GRAY_W / w))), interpolation=cv2.INTER_AREA)
    return img


def _default_detector():
    from vision.face_detect import FaceDetector

    return FaceDetector()


class PersonBox:
    def __init__(self, ingest, selection, detector_factory=_default_detector, decode=_default_decode,
                 camera_flipped: bool = False, enabled: bool = True, recorder_fn=None,
                 decode_gray=_default_gray):
        self.ingest, self.selection = ingest, selection
        self.detector_factory, self.decode, self.decode_gray = detector_factory, decode, decode_gray
        self.camera_flipped, self.enabled = camera_flipped, enabled
        self.recorder_fn = recorder_fn or (lambda: None)
        self.detector = None
        self.tracker, self.av = FaceTracker(), AVMatch()
        self._lock = threading.RLock()
        self._epoch = None
        self._prev_gray = None
        self._prev_ts = -1
        self._last_frame_us = -1
        self._frame_ts = None            # newest processed frame (device us)
        self._pose = None                # (yaw_deg, hfov_deg) of that frame
        self._last_det_t = 0.0
        self._last_assoc_t = 0.0
        self._learn_until = 0.0
        self._learn_started = False
        self._last_faces = 0
        self._published = False
        self._reason = "no_selection"
        self._age_ms: float | None = None
        self._sids: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- overlay (called from EventsState.tick, on the loop)
    def overlay(self, sources: list, now_us: int) -> list:
        try:
            return self._overlay(sources, now_us)
        except Exception:
            log.exception("person box overlay failed")
            self._published = False
            return sources

    def _reset_state(self):
        self.tracker.reset()
        self.av.reset()
        self._prev_gray, self._prev_ts, self._frame_ts = None, -1, None

    def _overlay(self, sources, now_us):
        self._published, self._age_ms = False, None
        if not self.enabled:
            self._reason = "no_selection"
            return sources
        epoch = getattr(self.ingest, "pose_epoch", None)
        with self._lock:
            if epoch != self._epoch:
                self._epoch = epoch
                self._reset_state()
        sid = self.selection.chosen
        src = next((s for s in sources if s.source_id == sid), None) if sid else None
        if src is None or src.state == p.TrackState.lost:
            self._reason = "no_selection"
            return sources
        if self.detector is None:
            self._reason = "detector_loading"
            return sources
        with self._lock:
            a = self.av.assoc.get(sid)
            if a is None:
                self._reason = self.av.reasons.get(sid, "no_association")
                return sources
            face = self.tracker.faces.get(a.face)
            if face is None:
                self._reason = "face_unseen"
                return sources
            fts, pose = self._frame_ts, self._pose
            if fts is None or pose is None:
                self._reason = "stale_frame"
                return sources
            age = now_us - fts
            self._age_ms = age / 1000
            if not 0 <= age <= MAX_AGE_US:
                self._reason = "stale_frame"
                return sources
            if fts - face.det_ts > DET_FRESH_US:
                self._reason = "face_unseen"
                return sources
            box = predict_box(face, fts)
            conf = 0.5 + 0.4 * min(1.0, a.score / 0.4)
        x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in box)
        fw, fh = x1 - x0, y1 - y0
        cx = (x0 + x1) / 2
        bx0, bx1 = min(1.0, max(0.0, cx - 1.5 * fw)), min(1.0, max(0.0, cx + 1.5 * fw))
        by0, by1 = min(1.0, max(0.0, y0 - 0.4 * fh)), min(1.0, max(0.0, y1 + 1.8 * fh))
        if self.camera_flipped:
            by0, by1 = 1 - by1, 1 - by0
        bearing = pose[0] + math.degrees(math.atan((2 * cx - 1) * math.tan(math.radians(pose[1]) / 2)))
        self._published, self._reason = True, ""
        return [s.model_copy(update=dict(
            bbox=(bx0, by0, bx1, by1), bearing_deg=bearing, bearing_conf=conf,
            bearing_source=p.BearingSource.camera, in_view=True)) if s.source_id == sid else s
            for s in sources]

    # ---- worker cycle
    def _active(self) -> bool:
        rec = self.recorder_fn()
        now = time.monotonic()
        if rec is not None and not self._learn_started:
            try:
                if len(rec.ids()) >= 2:
                    self._learn_started, self._learn_until = True, now + PRELEARN_S
            except Exception:
                pass
        return bool(self.selection.chosen) or now < self._learn_until

    def _speaker_state(self):
        rec = self.recorder_fn()
        if rec is None:
            return [], {}
        sids, ivs = rec.ids(), {}
        for s in sids:
            lst = rec.intervals(s) or []
            ivs[s] = [(d["start_us"], d["end_us"]) for d in lst]
        return sids, ivs

    def step(self) -> str:
        """One cycle: motion sample on a new frame, detection when due, evidence + association."""
        if not self.enabled:
            return "disabled"
        if not self._active():
            return "idle"
        epoch = getattr(self.ingest, "pose_epoch", None)
        with self._lock:
            if epoch != self._epoch:
                self._epoch = epoch
                self._reset_state()
        with self.ingest.video.lock:
            h, jpeg = self.ingest.video.header, self.ingest.video.jpeg
        if h is None or jpeg is None or h.device_ts_us == self._last_frame_us:
            return "same_frame"
        if h.hfov_deg is None or h.cam_rot_xyzw is None:
            self._last_reason = "no_pose"
            return "no_pose"
        self._last_frame_us = ts = h.device_ts_us
        gray = self.decode_gray(jpeg, getattr(h, "width", None))
        if gray is None:
            return "bad_frame"
        yaw = yaw_pitch(h.cam_rot_xyzw)[0]
        with self._lock:
            self._frame_ts, self._pose = ts, (yaw, h.hfov_deg)
            if self._prev_gray is not None and 0 < ts - self._prev_ts < 500_000:
                for f in self.tracker.faces.values():
                    m = mouth_motion(self._prev_gray, gray, predict_box(f, ts))
                    if m is not None:
                        f.series.append((ts, m))
                        self.av.add_motion(f.id, ts, m)
                    while f.series and ts - f.series[0][0] > HISTORY_US:
                        f.series.popleft()
            self._prev_gray, self._prev_ts = gray, ts
        out = "motion"
        now = time.monotonic()
        if self.detector is not None and now - self._last_det_t >= DET_PERIOD_S:
            self._last_det_t = now
            out = self._detect(jpeg, ts)
        if now - self._last_assoc_t >= ASSOC_PERIOD_S:
            self._last_assoc_t = now
            try:
                sids, ivs = self._speaker_state()
                with self._lock:
                    self._sids = sids
                    self.av.finalize(ivs)
                    self.tracker.prune(ts)
                    active = [s for s in sids if ivs.get(s)]
                    self.av.update(active, list(self.tracker.faces), ts)
            except Exception:
                log.exception("person box association failed")
        return out

    def _detect(self, jpeg, ts) -> str:
        try:
            img = self.decode(jpeg)
            if img is None:
                return "bad_frame"
            ih, iw = img.shape[:2]
            dets = self.detector.detect(img, conf_th=MIN_CONF)
            faces = [d for d in dets if d[4] >= MIN_CONF and (d[3] - d[1]) / ih >= MIN_FACE_H]
            boxes = [(float(d[0]) / iw, float(d[1]) / ih, float(d[2]) / iw, float(d[3]) / ih) for d in faces]
            self._last_faces = len(boxes)
            with self._lock:
                self.tracker.update(boxes, ts)
            return "ok"
        except Exception:
            log.exception("person box detection failed")
            return "detect_error"

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
        while not self._stop.wait(POLL_S):
            try:
                self.step()
            except Exception:
                log.exception("person box step failed")

    # ---- status / debug
    def status(self) -> dict:
        with self._lock:
            fts = self._frame_ts
            faces = [dict(id=f.id, box=[round(v, 3) for v in f.box],
                          last_seen_ms=None if fts is None else round((fts - f.det_ts) / 1000))
                     for f in self.tracker.faces.values()]
            assoc = {s: dict(face=a.face, score=round(a.score, 3), margin=round(a.margin, 2),
                             evidence_s=round(a.evidence_s, 1)) for s, a in self.av.assoc.items()}
            reasons = dict(self.av.reasons)
        return dict(enabled=self.enabled, detector_ready=self.detector is not None,
                    active=self._active() if self.enabled else False,
                    last_face_count=self._last_faces, last_age_ms=self._age_ms, published=self._published,
                    reason_if_not=None if self._published else self._reason,
                    faces=faces, associations=assoc, speaker_reasons=reasons, speakers=list(self._sids))

    def debug(self) -> dict:
        sids, ivs = self._speaker_state() if self.recorder_fn() else ([], {})
        with self._lock:
            fts = self._frame_ts
            series = {}
            for f in self.tracker.faces.values():
                pts = list(f.series)
                step = max(1, len(pts) // 60)
                series[f.id] = [[round((t - fts) / 1e6, 2), round(m, 2)] for t, m in pts[::step]] if fts else []
            tail = self.av.bins_tail(60)
        act = {}
        if fts is not None:
            for s, lst in ivs.items():
                act[s] = [[round((a - fts) / 1e6, 2), round((b - fts) / 1e6, 2)] for a, b in lst
                          if b > fts - HISTORY_US]
        return dict(frame_ts_us=fts, motion_s_rel=series, speaker_intervals_s_rel=act, bins=tail,
                    latest_candidates={s: (None if c is None else dict(face=c.face, score=round(c.score, 3)))
                                       for s, c in self.av.latest.items()})
