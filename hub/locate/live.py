"""Live direction finding (task C4): feeds hub ingest poses and separation levels into the
head-movement BearingEstimator (A8) and publishes one protocol Source per separated stream.

Use as a SeparationWorker listener:  worker.listeners.append(LiveLocator(state).on_chunks)
Only calls the A8 estimator's public API; it does not change hub/locate's other files.

- source_id is the separator's stream slot (s1, s2, ...), not a person. Slots can change
  occupant (one-talker merges, a new talker after a pause), so the tracker (D1) must decide
  identity from bearing and voice, not from source_id.
- bearing_deg is the estimator's world yaw: 0 = headset tracking-space forward, + = right. It
  equals "initial forward" only if the tracking origin faced forward; confirm on the headset
  (recentring shifts it).
- state: tracking while the stream is heard, last_seen after LAST_SEEN_S of silence, lost after
  LOST_S. last_observed_us is when the stream was last heard, not the latest hop.
- method: "level" (default) uses the head-shadow level estimator (A8). "tdoa" uses the
  inter-mic delay of each stream's two mic images plus head yaw (pair.py), calibrated per device;
  experimental, for closely spaced mics such as the Quest 3S, where "level" fails.
"""

from __future__ import annotations

import threading
import warnings

import numpy as np

import protocol as p

from .estimator import BearingEstimate, BearingEstimator, LocateConfig
from .pair import PairCalibration, PairConfig, PairLocator, frame_delays

# A8 estimator states -> how much to trust the bearing in the published Source.
_SHOW = {"locked", "tracking", "ambiguous"}
AMBIGUOUS_CONF_MAX = 0.5   # mirror/front-back ambiguity: show the bearing, but not confidently
ACTIVE_MIN = 0.2           # level-frame activity above this counts as "the stream is heard"
LAST_SEEN_S, LOST_S = 2.0, 10.0
IDENTITY_MIN_CONF = 0.6    # a chunk's acoustic id counts as known at or above this confidence
TDOA_WIN, TDOA_HOP = 1024, 800  # 64 ms frames every 50 ms at 16 kHz


class LiveLocator:
    def __init__(self, ingest, cfg: LocateConfig | None = None, method: str = "level",
                 calib: PairCalibration | None = None, pair_cfg: PairConfig | None = None):
        assert method in ("level", "tdoa")
        self.ingest, self.cfg, self.method, self.pair_cfg = ingest, cfg or LocateConfig(), method, pair_cfg
        self.calib = (calib or PairCalibration.default()) if method == "tdoa" else None
        if self.calib is not None and self.calib.delay_method != "gcc":
            raise ValueError("live tdoa measures GCC-PHAT delays; this calibration was fitted on "
                             f"{self.calib.delay_method!r} delays")
        if self.calib is not None and (self.calib.rate_hz, self.calib.win) != (16000, TDOA_WIN):
            warnings.warn(f"tdoa calibration was fitted at {self.calib.rate_hz} Hz, {self.calib.win}-sample "
                          f"frames; live measures at 16000 Hz, {TDOA_WIN}: its channel offset may not hold "
                          "(refit with `real_eval calibrate --rate 16000`)", stacklevel=2)
        self.lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.est = PairLocator(self.calib, self.pair_cfg) if self.method == "tdoa" else BearingEstimator(self.cfg)
        self._last_pose_us = None
        self._pose_epoch = self.ingest.pose_epoch
        self.latest: dict[str, tuple] = {}  # stream_id -> (StreamChunk, BearingEstimate)
        # The estimators have no per-stream reset, so a stream whose confident acoustic id changes
        # gets a fresh estimator key (stream_id, then stream_id#1, ...): its direction state starts
        # empty instead of carrying the previous talker's bearing. Published ids stay the stream ids.
        self._gen: dict[str, int] = {}
        self._conf_aid: dict[str, str] = {}
        self.identity: dict[str, tuple[str | None, float]] = {}
        self.last_active_us: dict[str, int] = {}
        self.now_us = 0

    def _feed_poses(self):
        if self.ingest.pose_epoch != self._pose_epoch:  # reconnect: device clock may restart
            self._reset()
        with self.ingest.poses.lock:
            poses = list(self.ingest.poses.poses)
        for q in poses:
            if self._last_pose_us is None or q.device_ts_us > self._last_pose_us:
                self.est.add_head_pose(q.device_ts_us / 1e6, q.yaw_deg, q.pitch_deg)
                self._last_pose_us = q.device_ts_us

    def on_chunks(self, chunks) -> None:
        """SeparationWorker listener: one finished hop for every stream."""
        with self.lock:
            self._feed_poses()
            if self._last_pose_us is None:
                return  # no head pose yet: nothing to locate against
            for c in chunks:
                aid, iconf = getattr(c, "acoustic_id", None), float(getattr(c, "identity_conf", 0.0) or 0.0)
                self.identity[c.stream_id] = (aid, iconf)
                if aid is not None and iconf >= IDENTITY_MIN_CONF:
                    if self._conf_aid.get(c.stream_id, aid) != aid:
                        self._gen[c.stream_id] = self._gen.get(c.stream_id, 0) + 1  # new voice: re-key
                    self._conf_aid[c.stream_id] = aid
                key = self._key(c.stream_id)
                if self.method == "tdoa":
                    self._add_delays(c, key)
                for t_us, lv, w in zip(c.level_ts_us, c.levels_db, c.activity):
                    if self.method == "level":
                        self.est.add_levels(key, t_us / 1e6, lv, float(w))
                    if w > ACTIVE_MIN:
                        self.last_active_us[c.stream_id] = int(t_us)
                self.now_us = max(self.now_us, int(c.level_ts_us[-1]))
                self.latest[c.stream_id] = (c, self.est.estimate(key, c.level_ts_us[-1] / 1e6))

    def _key(self, stream_id: str) -> str:
        g = self._gen.get(stream_id, 0)
        return stream_id if g == 0 else f"{stream_id}#{g}"

    def identities(self) -> dict[str, tuple[str | None, float]]:
        """stream_id -> (acoustic id, confidence) from the latest chunks; internal, never published."""
        with self.lock:
            return {sid: self.identity.get(sid, (None, 0.0)) for sid in self.latest}

    def _add_delays(self, c, key: str | None = None) -> None:
        """Inter-mic delay of this stream's two mic images, frame by frame, weighted by activity
        and kept only for the sharper half of the GCC peaks."""
        if c.mic_audio.shape[0] < 2 or not np.any(c.mic_audio):
            return
        t, tau, sharp = frame_delays(c.mic_audio[:2].T, 16000, TDOA_WIN, TDOA_HOP, self.calib.max_tau_s)
        if len(t) == 0:
            return
        t_us = c.start_ts_us + t * 1e6
        w = np.interp(t_us, c.level_ts_us, c.activity)
        on = w > ACTIVE_MIN
        if not on.any():
            return
        cut = np.median(sharp[on])
        for ti, di, wi, si in zip(t_us, tau, w, sharp):
            if wi > ACTIVE_MIN and si >= cut:
                self.est.add_delay(key or c.stream_id, ti / 1e6, float(di), float(wi))

    def estimates(self) -> dict[str, BearingEstimate]:
        with self.lock:
            return {sid: e for sid, (_, e) in self.latest.items()}

    def sources(self) -> list[p.Source]:
        """Protocol Sources: bearing_deg is world yaw (0 = initial forward, + = right)."""
        out = []
        with self.lock:
            for sid, (c, e) in self.latest.items():
                if sid not in self.last_active_us:
                    continue  # never heard: an empty slot is not a source
                silent_s = (self.now_us - self.last_active_us[sid]) / 1e6
                state = (p.TrackState.tracking if silent_s < LAST_SEEN_S else
                         p.TrackState.last_seen if silent_s < LOST_S else p.TrackState.lost)
                shown = e.state in _SHOW
                conf = min(max(e.confidence, 0.0), 1.0)
                if e.state == "ambiguous":
                    conf = min(conf, AMBIGUOUS_CONF_MAX)
                out.append(p.Source(
                    source_id=sid, label=c.label, label_conf=min(max(c.label_conf, 0.0), 1.0),
                    bearing_deg=e.world_deg if shown else None,
                    bearing_conf=conf if shown else 0.0,
                    bearing_source=p.BearingSource.head_motion if shown else None,
                    state=state,
                    level_db=float(10 * np.log10(np.mean(c.audio ** 2) + 1e-10)),
                    last_observed_us=self.last_active_us[sid],
                ))
        return out
