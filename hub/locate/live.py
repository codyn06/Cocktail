"""Live direction finding (task C4): feeds hub ingest poses and separation levels into the
head-movement BearingEstimator (A8) and publishes one protocol Source per separated stream.

Use as a SeparationWorker listener:  worker.listeners.append(LiveLocator(state).on_chunks)
Only calls the A8 estimator's public API; it does not change hub/locate's other files.
"""

from __future__ import annotations

import threading

import protocol as p

from .estimator import BearingEstimate, BearingEstimator, LocateConfig

# A8 estimator states -> how much to trust the bearing in the published Source.
_SHOW = {"locked", "tracking", "ambiguous"}


class LiveLocator:
    def __init__(self, ingest, cfg: LocateConfig | None = None):
        self.ingest, self.cfg = ingest, cfg or LocateConfig()
        self.lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.est = BearingEstimator(self.cfg)
        self._last_pose_us = None
        self._pose_epoch = self.ingest._pose_epoch
        self.latest: dict[str, tuple] = {}  # stream_id -> (StreamChunk, BearingEstimate)

    def _feed_poses(self):
        if self.ingest._pose_epoch != self._pose_epoch:  # reconnect: device clock may restart
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
                for t_us, lv, w in zip(c.level_ts_us, c.levels_db, c.activity):
                    self.est.add_levels(c.stream_id, t_us / 1e6, lv, float(w))
                self.latest[c.stream_id] = (c, self.est.estimate(c.stream_id, c.level_ts_us[-1] / 1e6))

    def estimates(self) -> dict[str, BearingEstimate]:
        with self.lock:
            return {sid: e for sid, (_, e) in self.latest.items()}

    def sources(self) -> list[p.Source]:
        """Protocol Sources: bearing_deg is world yaw (0 = initial forward, + = right)."""
        import numpy as np

        out = []
        with self.lock:
            for sid, (c, e) in self.latest.items():
                shown = e.state in _SHOW
                out.append(p.Source(
                    source_id=sid, label=c.label, label_conf=min(max(c.label_conf, 0.0), 1.0),
                    bearing_deg=e.world_deg if shown else None,
                    bearing_conf=min(max(e.confidence, 0.0), 1.0) if shown else 0.0,
                    bearing_source=p.BearingSource.head_motion if shown else None,
                    level_db=float(10 * np.log10(np.mean(c.audio ** 2) + 1e-10)),
                    last_observed_us=int(c.level_ts_us[-1]),
                ))
        return out
