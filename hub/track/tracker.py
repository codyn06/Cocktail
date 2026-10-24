"""D1: give every separated sound one stable identity, direction and state.

The separator hands out stream slots (s1, s2, ...) that can swap or go quiet; the locator
gives each slot a world bearing. The tracker turns that into stable `t1`, `t2`, ... sources:

    tracker = Tracker()
    sources = tracker.update(now_us, raw_sources, head_yaw_deg)   # raw = protocol Sources

Identity comes from WORLD bearing (it does not move when the head turns) plus label, so a talker
who pauses and comes back in a different stream slot keeps the same id. A source that goes quiet
becomes last_seen, then lost, and is kept for reacquisition before it is forgotten.
Pure Python, no models: it runs the same on simulated and real inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

import protocol as p


def wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


@dataclass
class TrackConfig:
    active_floor_db: float = -55.0   # raw level_db below this = the stream is silent
    min_bearing_conf: float = 0.25   # bearings less certain than this are not used to match
    gate_deg: float = 30.0           # max bearing difference to count as the same source
    label_penalty_deg: float = 25.0  # added to the distance when labels differ
    stream_bonus_deg: float = 15.0   # subtracted when the same stream slot matched this track before
    merge_deg: float = 12.0          # a same-label extra stream this close to a track is a duplicate
    confirm_hits: int = 3            # updates needed before a new source is announced
    hold_s: float = 1.5              # unheard this long: tracking -> last_seen
    lost_s: float = 8.0              # unheard this long: last_seen -> lost
    forget_s: float = 60.0           # unheard this long: dropped (the id is never reused)
    bearing_alpha: float = 0.3       # bearing smoothing (scaled up by confidence)
    label_decay: float = 0.9
    fov_deg: float = 90.0            # horizontal field of view used for in_view
    silent_db: float = -100.0        # level_db reported while a source is not being heard


@dataclass
class _Track:
    tid: str
    label_scores: dict[str, float]
    bearing: float | None
    bearing_conf: float
    bearing_source: p.BearingSource | None
    stream: str
    first_us: int
    last_obs_us: int
    level_db: float
    hits: int = 1
    confirmed: bool = False
    state: p.TrackState = p.TrackState.tracking

    @property
    def label(self) -> str:
        return max(self.label_scores.items(), key=lambda kv: kv[1])[0]

    @property
    def label_conf(self) -> float:
        tot = sum(self.label_scores.values())
        return min(1.0, self.label_scores[self.label] / tot) if tot > 0 else 0.0


class Tracker:
    def __init__(self, cfg: TrackConfig | None = None):
        self.cfg = cfg or TrackConfig()
        self.reset()

    def reset(self) -> None:
        self.tracks: dict[str, _Track] = {}
        self._next = 1
        self._last_now: int | None = None
        self._dup_of: dict[str, str] = {}  # stream slot -> track it duplicates

    # ------------------------------------------------------------------ matching
    def _cost(self, t: _Track, o: p.Source, has_b: bool) -> float | None:
        """Lower is better; None = not allowed to match."""
        c = self.cfg
        same_stream = t.stream == o.source_id
        if t.bearing is None or not has_b:
            return 0.0 if same_stream else None  # no direction to compare: only a slot's own track
        d = abs(wrap180(t.bearing - o.bearing_deg))
        if d > c.gate_deg:
            return None
        cost = d + (0.0 if o.label == t.label else c.label_penalty_deg)
        return cost - (c.stream_bonus_deg if same_stream else 0.0)

    def update(self, now_us: int, raw: list[p.Source], head_yaw_deg: float | None = None) -> list[p.Source]:
        c = self.cfg
        if self._last_now is not None and now_us < self._last_now - 2_000_000:
            self.reset()  # device clock restarted (headset app relaunched)
        self._last_now = now_us

        heard = [o for o in raw if o.level_db >= c.active_floor_db]
        has_b = {o.source_id: o.bearing_deg is not None and o.bearing_conf >= c.min_bearing_conf for o in heard}
        live = list(self.tracks.values())

        # optimal one-to-one matching of heard streams to existing tracks
        matched: dict[str, _Track] = {}
        if live and heard:
            big = 1e6
            M = np.full((len(heard), len(live)), big)
            for i, o in enumerate(heard):
                for j, t in enumerate(live):
                    cost = self._cost(t, o, has_b[o.source_id])
                    if cost is not None:
                        M[i, j] = cost
            for i, j in zip(*linear_sum_assignment(M)):
                if M[i, j] < big:
                    matched[heard[i].source_id] = live[j]

        fresh: list[_Track] = []  # tracks created in this update
        for o in heard:
            t = matched.get(o.source_id)
            if t is not None:
                self._dup_of.pop(o.source_id, None)
                self._observe(t, o, now_us, has_b[o.source_id])
                continue
            if has_b[o.source_id]:  # a second stream carrying the same source as a track just heard?
                prev = self.tracks.get(self._dup_of.get(o.source_id, ""))
                cands = [m for m in [*matched.values(), *fresh] if m.bearing is not None and m.label == o.label
                         and abs(wrap180(m.bearing - o.bearing_deg)) <= c.merge_deg]
                if not cands and prev is not None and prev.bearing is not None and prev.label == o.label                         and abs(wrap180(prev.bearing - o.bearing_deg)) <= c.gate_deg:
                    cands = [prev]  # same slot was a duplicate of this track before: stay a duplicate
                if cands:
                    self._dup_of[o.source_id] = cands[0].tid
                    continue
            fresh.append(self._spawn(o, now_us, has_b[o.source_id]))

        self._merge_converged()
        for tid, t in list(self.tracks.items()):
            if t.last_obs_us == now_us:
                continue
            age = (now_us - t.last_obs_us) / 1e6
            if not t.confirmed:
                if age > c.hold_s:
                    del self.tracks[tid]  # a blip that never became a source
                continue
            if age > c.forget_s:
                del self.tracks[tid]
            elif age > c.lost_s:
                t.state = p.TrackState.lost
            elif age > c.hold_s:
                t.state = p.TrackState.last_seen
        return self.sources(now_us, head_yaw_deg)

    def _merge_converged(self) -> None:
        """Two same-label tracks that settle on the same bearing were one source split over two
        stream slots: fold the younger into the older (its slot becomes a sticky duplicate)."""
        c = self.cfg
        tr = sorted(self.tracks.values(), key=lambda t: t.first_us)
        for i, a in enumerate(tr):
            for b in tr[i + 1:]:
                if (a.tid in self.tracks and b.tid in self.tracks and a.bearing is not None and b.bearing is not None
                        and a.label == b.label and a.state == b.state == p.TrackState.tracking
                        and abs(wrap180(a.bearing - b.bearing)) <= c.merge_deg / 2):
                    self._dup_of[b.stream] = a.tid
                    a.hits += b.hits
                    a.confirmed = a.confirmed or a.hits >= c.confirm_hits
                    del self.tracks[b.tid]

    def _observe(self, t: _Track, o: p.Source, now_us: int, has_b: bool) -> None:
        c = self.cfg
        t.stream, t.last_obs_us, t.level_db = o.source_id, now_us, o.level_db
        t.hits += 1
        t.confirmed = t.confirmed or t.hits >= c.confirm_hits
        t.state = p.TrackState.tracking
        for k in t.label_scores:
            t.label_scores[k] *= c.label_decay
        t.label_scores[o.label] = t.label_scores.get(o.label, 0.0) + max(o.label_conf, 0.05)
        if has_b:
            if t.bearing is None:
                t.bearing, t.bearing_conf = o.bearing_deg, o.bearing_conf
            else:
                a = min(0.6, c.bearing_alpha * (0.5 + o.bearing_conf))
                t.bearing = wrap180(t.bearing + a * wrap180(o.bearing_deg - t.bearing))
                t.bearing_conf += 0.3 * (o.bearing_conf - t.bearing_conf)
            t.bearing_source = o.bearing_source or p.BearingSource.head_motion

    def _spawn(self, o: p.Source, now_us: int, has_b: bool) -> _Track:
        tid = f"t{self._next}"
        self._next += 1
        self.tracks[tid] = _Track(
            tid=tid, label_scores={o.label: max(o.label_conf, 0.05)},
            bearing=o.bearing_deg if has_b else None, bearing_conf=o.bearing_conf if has_b else 0.0,
            bearing_source=(o.bearing_source or p.BearingSource.head_motion) if has_b else None,
            stream=o.source_id, first_us=now_us, last_obs_us=now_us, level_db=o.level_db,
            confirmed=self.cfg.confirm_hits <= 1)
        return self.tracks[tid]

    # ------------------------------------------------------------------ output
    def sources(self, now_us: int, head_yaw_deg: float | None = None) -> list[p.Source]:
        c = self.cfg
        out = []
        for t in self.tracks.values():
            if not t.confirmed:
                continue
            in_view = (t.bearing is not None and head_yaw_deg is not None
                       and t.state != p.TrackState.lost
                       and abs(wrap180(t.bearing - head_yaw_deg)) <= c.fov_deg / 2)
            out.append(p.Source(
                source_id=t.tid, label=t.label, label_conf=t.label_conf,
                bearing_deg=t.bearing, bearing_conf=min(max(t.bearing_conf, 0.0), 1.0),
                bearing_source=t.bearing_source, state=t.state, in_view=bool(in_view),
                level_db=t.level_db if t.state == p.TrackState.tracking else c.silent_db,
                last_observed_us=t.last_obs_us))
        return out

    def event(self, now_us: int, head_yaw_deg: float | None = None) -> p.SourcesUpdated:
        return p.SourcesUpdated(device_ts_us=now_us, sources=self.sources(now_us, head_yaw_deg))
