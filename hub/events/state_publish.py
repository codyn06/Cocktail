"""Publish the hub's authoritative selection state on /events (existing protocol messages only).

Runs as an `EventsState.tick_hooks` entry. Each tick it derives the state from `events.selection`
(chosen id, version, `route(tracker)`) and publishes ONLY WHEN IT CHANGES, via `publish_aux`:

* `state.changed` under aux key ``hub.state`` (always present, so a newly connected client gets it).
* `target.updated` under aux key ``hub.target`` while something is chosen; the key is dropped when
  nothing is chosen. Refinement publishes under ``t_<source_id>``; that is a different key, so the
  two never overwrite each other, and this module skips its own target.updated when refinement
  already holds one for the same source and selection version (refinement owns person claims).

State mapping (nothing chosen -> listening):

  chosen, route has >= 1 slot, track tracking -> playing
  chosen, track last_seen                     -> selected
  chosen, track lost / forgotten / no slots   -> target_lost

target_id is ``t_<source_id>`` (same as refine); the headset resolves it back to the tracker id.
assoc_conf is always 0.0 (no person claim), kind is the track label (default "speech"). Bearing is
informational and NOT part of the change signature (it moves every tick); it is sent when the rest changes.
`mode` changes republish state.changed. Epoch / clock resets clear the selection, so the next tick
publishes listening.
"""

from __future__ import annotations

import protocol as p

STATE_KEY = "hub.state"
TARGET_KEY = "hub.target"


def compute(events) -> tuple[p.StateChanged, p.TargetUpdated | None]:
    sel = events.selection
    chosen = sel.chosen
    if chosen is None:
        return p.StateChanged(state=p.HubState.listening, target_id=None,
                              selection_version=sel.version, mode=events.mode), None
    route = sel.route(events.tracker)
    tstate = route.state if route is not None else p.TrackState.lost
    if route is not None and route.slots and tstate == p.TrackState.tracking:
        hub_state = p.HubState.playing
    elif tstate == p.TrackState.last_seen:
        hub_state = p.HubState.selected
    else:
        hub_state = p.HubState.target_lost
    tid = f"t_{chosen}"
    track = events.tracker.tracks.get(chosen)
    kind = getattr(track, "label", None) or "speech"
    bearing = route.bearing_deg if route is not None else None
    target = p.TargetUpdated(target_id=tid, selection_version=sel.version, source_id=chosen, kind=kind,
                             assoc_conf=0.0, state=tstate, bearing_deg=bearing)
    return (p.StateChanged(state=hub_state, target_id=tid, selection_version=sel.version, mode=events.mode),
            target)


def install_state_publisher(events) -> None:
    last: dict = {"state": None, "target": None}

    def hook() -> None:
        state, target = compute(events)
        sig = (state.state, state.target_id, state.selection_version, state.mode)
        if sig != last["state"]:
            last["state"] = sig
            events.publish_aux(state.model_dump_json(), key=STATE_KEY)
        if target is None:
            if last["target"] is not None:
                last["target"] = None
                events.drop_aux(TARGET_KEY)
            return
        tsig = (target.target_id, target.selection_version, target.kind, target.state)
        if tsig == last["target"]:
            return
        last["target"] = tsig
        refine = events.aux_snapshot.get(target.target_id)
        if refine is not None:
            try:
                held = p.TargetUpdated.model_validate_json(refine)
            except ValueError:
                held = None
            if held is not None and held.selection_version == target.selection_version:
                return  # refinement already describes this selection
        events.publish_aux(target.model_dump_json(), key=TARGET_KEY)

    events.tick_hooks.append(hook)
