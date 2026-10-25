"""D1: give every separated sound one stable identity, direction and state.

The separator hands out stream slots (s1, s2, ...) that can swap or go quiet; the locator
gives each slot a world bearing. The tracker turns that into stable `t1`, `t2`, ... sources:

    tracker = Tracker()
    sources = tracker.update(now_us, raw_sources, head_yaw_deg)   # raw = protocol Sources

Identity comes from WORLD bearing (it does not move when the head turns) plus label, so a talker
who pauses and comes back in a different stream slot keeps the same id. A source that goes quiet
becomes last_seen, then lost, and is kept for reacquisition before it is forgotten.
Call update() on every tick (about 10 Hz), even when nothing is heard: the last_seen and lost
transitions only happen inside update(). Pure Python, no models: it runs the same on simulated
and real inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

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
    bearingless_hits: int = 15       # a source with no direction yet needs this many (and no same-label one active)
    # Speech with no direction (the live Quest withholds unfit bearings) can confirm sooner: None = off, else the
    # hop count for a speech-labelled direction-less source. Identity then rests on slot continuity + voice memory.
    bearingless_speech_hits: int | None = None
    # With bearingless_speech_hits on: a speech source on ANOTHER separator slot is a different voice (the separator
    # emits one voice per slot), so another tracking speech source no longer blocks its announcement.
    bearingless_speech_per_slot: bool = False
    hold_s: float = 1.5              # unheard this long: tracking -> last_seen
    lost_s: float = 8.0              # unheard this long: last_seen -> lost
    forget_s: float = 60.0           # unheard this long: dropped (the id is never reused)
    # Bearing smoothing is a time constant, so it does not depend on how often update() is called.
    # A reading close to the tracked bearing (within follow_from_deg) is noise: it moves the arrow
    # with time constant smooth_tau_s. One far from it (beyond follow_to_deg) is a real move: it is
    # followed with follow_tau_s. In between the two blend (geometric). A confident reading
    # (bearing_conf 1) acts as if the time constant were 1/1.5 as long, an unsure one (0) twice as long.
    smooth_tau_s: float = 0.6
    follow_tau_s: float = 0.3
    follow_from_deg: float = 10.0
    follow_to_deg: float = 30.0
    label_decay: float = 0.9
    fov_deg: float = 90.0            # horizontal field of view used for in_view
    silent_db: float = -100.0        # level_db reported while a source is not being heard
    identity_min_conf: float = 0.6   # acoustic-id confidence needed before an id is trusted (opt-in)
    # A bearing older than this (time since the last usable bearing observation) is not published:
    # bearing_deg None, bearing_conf 0, in_view False. The track keeps it internally to re-find the
    # talker. 8 s = lost_s and the estimator's own memory (tau_s); a 5 s pause is a normal conversational
    # gap (scene slot_reused_by_other_sound expects the direction to survive it). A walking talker
    # (1.4 m/s at 2 m = ~45 deg in 5 s) is the reason not to go longer. None = never expire (old behaviour).
    bearing_max_age_s: float | None = 8.0


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
    b_hits: int = 0  # observations that carried a usable bearing
    confirmed: bool = False
    state: p.TrackState = p.TrackState.tracking
    aid_run: tuple[str, int] | None = None  # (confident acoustic id, consecutive observations)
    bearing_us: int = 0  # when the bearing was last refreshed by a usable observation

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
        self._next = getattr(self, "_next", 1)  # ids are never reused, even after a clock restart
        self._last_now: int | None = None
        self._dup_of: dict[str, str] = {}  # stream slot -> track it duplicates
        self._apart: set[frozenset] = set()  # track pairs that were clearly separate sources at some point
        # Phase 5 identity (opt-in via update(identity=...)): acoustic id <-> tracker id. Cleared with
        # the tracker (epoch/clock restart) and with each track; tracker ids are never reused.
        self.id_map: dict[str, str] = {}
        self._tid_aid: dict[str, str] = {}

    # ------------------------------------------------------------------ identity
    def _aid_of(self, t: _Track) -> str | None:
        """The acoustic id a track stands for: its mapping, else the one it is currently collecting."""
        return self._tid_aid.get(t.tid) or (t.aid_run[0] if t.aid_run else None)

    def _differ(self, t: _Track, oid: str | None) -> bool:
        a = self._aid_of(t)
        return oid is not None and a is not None and a != oid

    def _same_voice_ruled_out(self, a: _Track, b: _Track) -> bool:
        x, y = self._aid_of(a), self._aid_of(b)
        return x is not None and y is not None and x != y

    def _drop(self, tid: str) -> None:
        self.tracks.pop(tid, None)
        aid = self._tid_aid.pop(tid, None)
        if aid is not None and self.id_map.get(aid) == tid:
            del self.id_map[aid]

    def _note_identity(self, t: _Track, oid: str | None) -> None:
        if oid is None:
            return
        t.aid_run = (oid, t.aid_run[1] + 1) if t.aid_run and t.aid_run[0] == oid else (oid, 1)
        if t.aid_run[1] >= self.cfg.confirm_hits and t.tid not in self._tid_aid and oid not in self.id_map:
            self.id_map[oid] = t.tid
            self._tid_aid[t.tid] = oid

    # ------------------------------------------------------------------ matching
    def _cost(self, t: _Track, o: p.Source, has_b: bool, oid: str | None = None) -> float | None:
        """Lower is better; None = not allowed to match."""
        c = self.cfg
        if oid is not None:
            mapped = self._tid_aid.get(t.tid)
            if mapped is not None and mapped != oid:
                return None  # a confidently different voice never updates this track
            if mapped is None and self.id_map.get(oid, t.tid) != t.tid:
                return None  # this voice already has its own track
            if mapped == oid:  # the voice came back: its own track, wherever it went meanwhile
                near_b = abs(wrap180(t.bearing - o.bearing_deg)) * 0.01 if t.bearing is not None and has_b else 0.0
                return -100.0 + near_b
        same_stream = t.stream == o.source_id
        if t.state != p.TrackState.tracking and o.label != t.label:
            return None  # a paused source only comes back as the same kind of sound
        if t.bearing is None or not has_b:
            return 5.0 if same_stream else None  # no direction to compare: only a slot's own track
        d = abs(wrap180(t.bearing - o.bearing_deg))
        if d > c.gate_deg:
            return None
        cost = d + (0.0 if o.label == t.label else c.label_penalty_deg)
        return cost - (c.stream_bonus_deg if same_stream else 0.0)

    def update(self, now_us: int, raw: list[p.Source], head_yaw_deg: float | None = None,
               identity: dict[str, tuple[str | None, float]] | None = None) -> list[p.Source]:
        """identity: stream id -> (acoustic id, confidence), optional. Without it (or without a
        confident entry) the world-bearing + label + slot rules apply unchanged."""
        c = self.cfg
        if self._last_now is not None and now_us < self._last_now - 2_000_000:
            self.reset()  # device clock restarted (headset app relaunched)
        self._last_now = now_us

        heard = [o for o in raw if math.isfinite(o.level_db) and o.level_db >= c.active_floor_db]
        has_b = {o.source_id: o.bearing_deg is not None and math.isfinite(o.bearing_deg)
                 and o.bearing_conf >= c.min_bearing_conf for o in heard}
        oids: dict[str, str] = {}
        for sid, v in (identity or {}).items():
            try:
                aid, conf = v
            except (TypeError, ValueError):
                continue
            if aid is not None and conf is not None and math.isfinite(conf) and conf >= c.identity_min_conf:
                oids[sid] = aid
        live = list(self.tracks.values())

        # optimal one-to-one matching of heard streams to existing tracks
        matched: dict[str, _Track] = {}
        if live and heard:
            big = 1e6
            M = np.full((len(heard), len(live)), big)
            for i, o in enumerate(heard):
                for j, t in enumerate(live):
                    cost = self._cost(t, o, has_b[o.source_id], oids.get(o.source_id))
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
                self._note_identity(t, oids.get(o.source_id))
                continue
            oid = oids.get(o.source_id)
            if has_b[o.source_id]:  # a second stream carrying the same source as a track just heard?
                prev = self.tracks.get(self._dup_of.get(o.source_id, ""))
                cands = [m for m in [*matched.values(), *fresh] if m.bearing is not None and m.label == o.label
                         and abs(wrap180(m.bearing - o.bearing_deg)) <= c.merge_deg and not self._differ(m, oid)]
                if not cands and prev is not None and prev.bearing is not None and prev.label == o.label                         and not self._differ(prev, oid) and prev.state == p.TrackState.tracking                         and abs(wrap180(prev.bearing - o.bearing_deg)) <= 2 * c.merge_deg:
                    cands = [prev]  # same slot was a duplicate of this track before: stay a duplicate
                if cands:
                    self._dup_of[o.source_id] = cands[0].tid
                    continue
            if oid is not None:
                self._dup_of.pop(o.source_id, None)
            nt = self._spawn(o, now_us, has_b[o.source_id])
            self._note_identity(nt, oid)
            fresh.append(nt)

        self._merge_converged()
        self._confirm()
        for tid, t in list(self.tracks.items()):
            if t.last_obs_us == now_us:
                continue
            age = (now_us - t.last_obs_us) / 1e6
            if not t.confirmed:
                if age > c.hold_s:
                    self._drop(tid)  # a blip that never became a source
                continue
            if age > c.forget_s:
                self._drop(tid)
            elif age > c.lost_s:
                t.state = p.TrackState.lost
            elif age > c.hold_s:
                t.state = p.TrackState.last_seen
        return self.sources(now_us, head_yaw_deg)

    def _merge_converged(self) -> None:
        """Two same-label tracks that settle on the same bearing were one source split over two
        stream slots: fold the younger into the older (its slot becomes a sticky duplicate).
        Never merges a pair that was ever clearly apart, so two talkers who cross paths keep
        their own ids."""
        c = self.cfg
        self._apart = {pr for pr in self._apart if all(i in self.tracks for i in pr)}
        tr = sorted(self.tracks.values(), key=lambda t: t.first_us)
        for i, a in enumerate(tr):
            for b in tr[i + 1:]:
                if a.tid not in self.tracks or b.tid not in self.tracks or a.bearing is None or b.bearing is None:
                    continue
                if a.state != p.TrackState.tracking or b.state != p.TrackState.tracking:
                    continue
                d = abs(wrap180(a.bearing - b.bearing))
                pair = frozenset((a.tid, b.tid))
                if self._same_voice_ruled_out(a, b):
                    self._apart.add(pair)  # two different voices are never one source
                    continue
                if d > 2 * c.merge_deg:
                    self._apart.add(pair)
                elif d <= c.merge_deg / 2 and a.label == b.label and pair not in self._apart:
                    self._dup_of[b.stream] = a.tid
                    a.hits += b.hits
                    a.confirmed = a.confirmed or b.confirmed
                    mapped_b = self._tid_aid.get(b.tid)
                    self._drop(b.tid)
                    if mapped_b is not None and a.tid not in self._tid_aid:
                        self.id_map[mapped_b] = a.tid
                        self._tid_aid[a.tid] = mapped_b

    def _confirm(self) -> None:
        """Announce a new source once it has been heard a few times. One with no direction yet is
        held back longer, and while another same-label source is active (it may be a duplicate)."""
        c = self.cfg
        for t in list(self.tracks.values()):
            if t.confirmed or t.hits < c.confirm_hits:
                continue
            if t.bearing is not None and t.b_hits < c.confirm_hits:
                continue  # wait until the bearing has settled
            if t.bearing is not None:
                near = [o for o in self.tracks.values()
                        if o is not t and o.confirmed and o.label == t.label and o.state == p.TrackState.tracking
                        and o.bearing is not None and abs(wrap180(o.bearing - t.bearing)) <= c.merge_deg
                        and frozenset((o.tid, t.tid)) not in self._apart and not self._same_voice_ruled_out(o, t)]
                if near:  # same sound as an announced source, seen through another stream slot
                    self._dup_of[t.stream] = near[0].tid
                    self._drop(t.tid)
                else:
                    t.confirmed = True
            elif t.tid in self._tid_aid:
                t.confirmed = True  # a distinct voice, direction unsure: announce it without a bearing
            elif t.hits >= (c.bearingless_speech_hits if c.bearingless_speech_hits is not None and t.label == "speech"
                            else c.bearingless_hits) and not any(
                    o is not t and o.label == t.label and o.state == p.TrackState.tracking
                    and not (c.bearingless_speech_per_slot and t.label == "speech" and o.stream != t.stream)
                    for o in self.tracks.values()):
                t.confirmed = True

    def _observe(self, t: _Track, o: p.Source, now_us: int, has_b: bool) -> None:
        c = self.cfg
        dt = min(max((now_us - t.last_obs_us) / 1e6, 0.0), 1.0)  # silent gaps count at most 1 s
        t.stream, t.last_obs_us, t.level_db = o.source_id, now_us, o.level_db
        t.hits += 1
        t.state = p.TrackState.tracking
        for k in t.label_scores:
            t.label_scores[k] *= c.label_decay
        t.label_scores[o.label] = t.label_scores.get(o.label, 0.0) + max(o.label_conf, 0.05)
        if has_b:
            t.b_hits += 1
            t.bearing_us = now_us
            if t.bearing is None:
                t.bearing, t.bearing_conf = o.bearing_deg, o.bearing_conf
            else:
                innov = wrap180(o.bearing_deg - t.bearing)
                span = max(c.follow_to_deg - c.follow_from_deg, 1e-6)
                f = min(max((abs(innov) - c.follow_from_deg) / span, 0.0), 1.0)
                tau = c.smooth_tau_s ** (1 - f) * c.follow_tau_s ** f
                tau /= min(max(0.5 + o.bearing_conf, 0.5), 1.5)
                t.bearing = wrap180(t.bearing + (1 - math.exp(-dt / tau)) * innov)
                t.bearing_conf += 0.3 * (o.bearing_conf - t.bearing_conf)
            t.bearing_source = o.bearing_source or p.BearingSource.head_motion

    def _spawn(self, o: p.Source, now_us: int, has_b: bool) -> _Track:
        tid = f"t{self._next}"
        self._next += 1
        self.tracks[tid] = _Track(
            tid=tid, label_scores={o.label: max(o.label_conf, 0.05)},
            bearing=o.bearing_deg if has_b else None, bearing_conf=o.bearing_conf if has_b else 0.0,
            bearing_source=(o.bearing_source or p.BearingSource.head_motion) if has_b else None,
            stream=o.source_id, first_us=now_us, last_obs_us=now_us, level_db=o.level_db, b_hits=int(has_b),
            bearing_us=now_us)
        return self.tracks[tid]

    # ------------------------------------------------------------------ output
    def sources(self, now_us: int, head_yaw_deg: float | None = None) -> list[p.Source]:
        c = self.cfg
        out = []
        for t in self.tracks.values():
            if not t.confirmed:
                continue
            fresh = (t.bearing is not None and (c.bearing_max_age_s is None
                                                or (now_us - t.bearing_us) / 1e6 <= c.bearing_max_age_s))
            in_view = (fresh and head_yaw_deg is not None
                       and t.state != p.TrackState.lost
                       and abs(wrap180(t.bearing - head_yaw_deg)) <= c.fov_deg / 2)
            out.append(p.Source(
                source_id=t.tid, label=t.label, label_conf=t.label_conf,
                bearing_deg=t.bearing if fresh else None,
                bearing_conf=min(max(t.bearing_conf, 0.0), 1.0) if fresh else 0.0,
                bearing_source=t.bearing_source if fresh else None, state=t.state, in_view=bool(in_view),
                level_db=t.level_db if t.state == p.TrackState.tracking else c.silent_db,
                last_observed_us=t.last_obs_us))
        return out

    def event(self, now_us: int, head_yaw_deg: float | None = None) -> p.SourcesUpdated:
        return p.SourcesUpdated(device_ts_us=now_us, sources=self.sources(now_us, head_yaw_deg))
