"""Which tracked source the user chose (M2), kept by the Tracker's stable id.

The headset sends InputEvent select / stop / next / prev on /ingest/pose. A select carries the
public `source_id` shown in `sources.updated` (a tracker id like `t3`, never a separator slot),
or a ray in tracking space, or nothing (then the head's forward direction is the ray).

Continuity: the choice is stored as a Tracker id. The Tracker already keeps one id per sound
through separator slot swaps and pauses (it matches by world bearing + label), so "t3" keeps
meaning the same talker when the separator calls them s1 one second and s2 the next. `route()`
asks the Tracker which slot(s) currently carry the chosen id; that is the only place the slot
is looked up, and it is looked up again for every audio frame.

Order for next / prev: by world bearing ascending, left to right (-180 < bearing <= 180);
sources without a bearing come last, by numeric id. Lost sources are skipped (the chosen one
stays in the cycle so next/prev step away from it). Wraps around. With nothing chosen, next
picks the first and prev the last.

Pure Python: no models, no I/O.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import protocol as p

log = logging.getLogger("events.selection")


def wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def _id_num(source_id: str) -> int:
    m = re.search(r"(\d+)$", source_id)
    return int(m.group(1)) if m else 1 << 30


def cycle_order(sources: list[p.Source]) -> list[p.Source]:
    """Deterministic order used by next / prev: bearing ascending, bearingless last."""
    def key(s: p.Source):
        b = s.bearing_deg
        has = b is not None and math.isfinite(b)
        return (0 if has else 1, wrap180(b) if has else 0.0, _id_num(s.source_id), s.source_id)
    return sorted(sources, key=key)


@dataclass(frozen=True)
class Route:
    """Where the chosen source's audio is right now."""

    target_id: str
    slots: tuple[str, ...]          # separator slots carrying it (empty = not audible now)
    bearing_deg: float | None
    state: p.TrackState


@dataclass
class Selection:
    chosen: str | None = None
    version: int = 0                # bumps on every change of the choice
    max_ray_error_deg: float = 30.0  # a ray further than this from every source selects nothing
    _slot: str | None = None        # slot the chosen track was on when chosen / at the last reconcile()

    def clear(self) -> None:
        self._set(None)

    def _set(self, source_id: str | None, tracker=None) -> None:
        """Change the choice. `tracker` (when given) records the chosen track's slot right away, so a
        merge that happens before the next reconcile() can still be followed."""
        if source_id != self.chosen:
            self.chosen = source_id
            self.version += 1
            log.info("selection -> %s (v%d)", source_id, self.version)
        t = tracker.tracks.get(source_id) if tracker is not None and source_id is not None else None
        self._slot = t.stream if t is not None else None   # never keep the previous choice's slot

    def handle(self, ev: p.InputEvent, sources: list[p.Source], head_yaw_deg: float | None,
               tracker=None) -> None:
        a = ev.action
        if a == p.Action.stop:
            self._set(None)
        elif a in (p.Action.next, p.Action.prev):
            self.step(sources, +1 if a == p.Action.next else -1, tracker)
        elif a == p.Action.select:
            self._select(ev, sources, head_yaw_deg, tracker)
        else:
            log.info("input action %s not handled by selection", a)

    def _select(self, ev: p.InputEvent, sources: list[p.Source], head_yaw_deg: float | None,
                tracker=None) -> None:
        if ev.source_id is not None:
            if any(s.source_id == ev.source_id for s in sources):
                self._set(ev.source_id, tracker)
            else:
                log.info("select of unknown source ignored")
            return
        if ev.ray_dir is not None:
            x, _, z = ev.ray_dir
            if not (math.isfinite(x) and math.isfinite(z)) or (x == 0 and z == 0):
                return
            yaw = math.degrees(math.atan2(x, z))      # tracking space: 0 = +z, + = right
        elif head_yaw_deg is not None:
            yaw = head_yaw_deg                         # no ray: whatever the head faces
        else:
            return
        best, best_err = None, self.max_ray_error_deg
        for s in sources:
            if s.state == p.TrackState.lost or s.bearing_deg is None or not math.isfinite(s.bearing_deg):
                continue
            err = abs(wrap180(s.bearing_deg - yaw))
            if err <= best_err:
                best, best_err = s, err
        if best is not None:
            self._set(best.source_id, tracker)

    def step(self, sources: list[p.Source], direction: int, tracker=None) -> None:
        pool = [s for s in sources if s.state != p.TrackState.lost or s.source_id == self.chosen]
        ordered = cycle_order(pool)
        if not ordered:
            return
        ids = [s.source_id for s in ordered]
        if self.chosen in ids:
            i = (ids.index(self.chosen) + direction) % len(ids)
        else:
            i = 0 if direction > 0 else len(ids) - 1
        self._set(ids[i], tracker)

    def reconcile(self, tracker) -> None:
        """Call after every Tracker.update. If the chosen track disappeared because the Tracker
        folded it into another (one sound seen through two slots), follow the merge; if it was
        forgotten, clear the choice (its id is never reused)."""
        if self.chosen is None:
            return
        t = tracker.tracks.get(self.chosen)
        if t is not None:
            self._slot = t.stream
            return
        merged = getattr(tracker, "_dup_of", {}).get(self._slot or "")
        if merged in tracker.tracks:
            log.info("chosen source %s merged into %s", self.chosen, merged)
            self._set(merged, tracker)
        else:
            log.info("chosen source %s forgotten", self.chosen)
            self._set(None)

    def route(self, tracker) -> Route | None:
        """Slots that currently carry the chosen source, or None when nothing is chosen.

        Reads the Tracker's `tracks[id].stream` (its current slot) and `_dup_of` (extra slots
        folded into the same source). A slot only counts while the track is `tracking` and no other
        track has been heard in it more recently."""
        if self.chosen is None:
            return None
        t = tracker.tracks.get(self.chosen)
        if t is None:
            return Route(self.chosen, (), None, p.TrackState.lost)
        slots: list[str] = []
        if t.state == p.TrackState.tracking:
            # A paused track stays `tracking` for hold_s with a stale slot. If another track has been
            # heard in that slot since, the slot now carries that other sound: do not route it.
            def claimed(slot: str) -> bool:
                return any(o is not t and o.stream == slot and o.last_obs_us > t.last_obs_us
                           for o in tracker.tracks.values())
            dup = getattr(tracker, "_dup_of", {})
            for slot in [t.stream] + sorted(s for s, tid in dup.items() if tid == self.chosen and s != t.stream):
                if not claimed(slot):
                    slots.append(slot)
        return Route(self.chosen, tuple(slots), t.bearing, t.state)
