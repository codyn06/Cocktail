"""Simulated scenes for the tracker, each with PASS/FAIL checks you can read.

    cd hub
    uv run python -m track            # all scenes
    uv run python -m track pause      # scenes whose name contains "pause"

Inputs are scripted separator+locator outputs (protocol Sources, stream slots s1, s2, ...), 10 per
second, with noisy bearings. This tests the tracking logic only, not real audio.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

import protocol as p

from .tracker import Tracker, wrap180

DT_US = 100_000  # 10 Hz, like sources.updated


def raw(stream: str, label: str, bearing: float | None, conf: float = 0.8, level: float = -30.0, us: int = 0) -> p.Source:
    return p.Source(source_id=stream, label=label, label_conf=0.8, bearing_deg=bearing,
                    bearing_conf=conf if bearing is not None else 0.0,
                    bearing_source=p.BearingSource.head_motion if bearing is not None else None,
                    level_db=level, last_observed_us=us)


@dataclass
class Run:
    t: list[float] = field(default_factory=list)
    out: list[list[p.Source]] = field(default_factory=list)

    def at(self, sec: float) -> list[p.Source]:
        return self.out[int(round(sec * 1e6 / DT_US))]

    def ids_ever(self) -> set[str]:
        return {s.source_id for o in self.out for s in o}


def simulate(frame, seconds: float, head_yaw=lambda t: 0.0, tracker: Tracker | None = None) -> Run:
    """frame(t, rng) -> list of raw Sources heard at second t."""
    tr, rng, run = tracker or Tracker(), np.random.default_rng(0), Run()
    for k in range(int(seconds * 1e6 / DT_US) + 1):
        t = k * DT_US / 1e6
        run.t.append(t)
        run.out.append(tr.update(k * DT_US, frame(t, rng), head_yaw(t)))
    return run


def jit(rng, b: float, sd: float = 5.0) -> float:
    return float(wrap180(b + rng.normal(0, sd)))


Check = tuple[str, bool, str]


def near(s: p.Source, b: float, tol: float = 15.0) -> bool:
    return s.bearing_deg is not None and abs(wrap180(s.bearing_deg - b)) <= tol


def by_bearing(srcs: list[p.Source], b: float, tol: float = 15.0) -> p.Source | None:
    c = [s for s in srcs if near(s, b, tol)]
    return c[0] if c else None


# ----------------------------------------------------------------------------- scenes
def two_talkers_swap_slots() -> list[Check]:
    def frame(t, rng):
        a, b = (("s1", "s2") if t < 5 else ("s2", "s1"))  # the separator swaps its output slots at 5 s
        return [raw(a, "speech", jit(rng, -40)), raw(b, "speech", jit(rng, 60))]
    r = simulate(frame, 10)
    before, after = r.at(4.5), r.at(9.5)
    ida0, ida1 = getattr(by_bearing(before, -40), "source_id", None), getattr(by_bearing(after, -40), "source_id", None)
    idb0, idb1 = getattr(by_bearing(before, 60), "source_id", None), getattr(by_bearing(after, 60), "source_id", None)
    return [("exactly 2 sources at the end", len(after) == 2, f"{len(after)}"),
            ("talker at -40 keeps the same id across the slot swap", ida0 is not None and ida0 == ida1, f"{ida0} -> {ida1}"),
            ("talker at +60 keeps the same id across the slot swap", idb0 is not None and idb0 == idb1, f"{idb0} -> {idb1}"),
            ("no extra ids were ever created", len(r.ids_ever()) == 2, f"{sorted(r.ids_ever())}")]


def pause_and_return() -> list[Check]:
    def frame(t, rng):
        if t < 5:
            return [raw("s1", "speech", jit(rng, 30))]
        if 15 <= t < 20:
            return [raw("s3", "speech", jit(rng, 30))]  # comes back in a different slot
        return []
    r = simulate(frame, 20)
    st = lambda sec: [s.state.value for s in r.at(sec)]
    return [("tracking while talking (t=4)", st(4) == ["tracking"], f"{st(4)}"),
            ("last_seen after a short pause (t=7)", st(7) == ["last_seen"], f"{st(7)}"),
            ("lost after a long pause (t=14)", st(14) == ["lost"], f"{st(14)}"),
            ("tracking again on return (t=19)", st(19) == ["tracking"], f"{st(19)}"),
            ("same id the whole time (one id ever)", len(r.ids_ever()) == 1, f"{sorted(r.ids_ever())}")]


def head_turn_in_view() -> list[Check]:
    def frame(t, rng):
        return [raw("s1", "speech", jit(rng, 70, 3))]  # fixed in the room at +70 deg
    r = simulate(frame, 12, head_yaw=lambda t: min(90.0, 90.0 * t / 10))  # head turns 0 -> 90 deg
    iv = lambda sec: r.at(sec)[0].in_view
    return [("out of view while facing straight ahead (t=1, yaw 9, source at +70)", not iv(1), f"in_view={iv(1)}"),
            ("in view once the head is turned toward it (t=6, yaw 54)", iv(6), f"in_view={iv(6)}"),
            ("world bearing stays near +70 as the head turns", all(near(o[0], 70, 10) for o in r.out[20:]), ""),
            ("one id the whole time", len(r.ids_ever()) == 1, f"{sorted(r.ids_ever())}")]


def blip_ignored() -> list[Check]:
    def frame(t, rng):
        out = [raw("s1", "speech", jit(rng, -20))]
        if 3.0 <= t < 3.2:
            out.append(raw("s2", "other", 150.0))  # 0.2 s click from somewhere else
        return out
    r = simulate(frame, 6)
    return [("a 0.2 s blip never shows up as a source", len(r.ids_ever()) == 1, f"{sorted(r.ids_ever())}")]


def same_direction_two_labels() -> list[Check]:
    def frame(t, rng):
        return [raw("s1", "speech", jit(rng, 20)), raw("s2", "music", jit(rng, 22))]
    r = simulate(frame, 8)
    labs = sorted(s.label for s in r.at(7.5))
    return [("speech and music from the same direction stay two sources", labs == ["music", "speech"], f"{labs}")]


def moving_source() -> list[Check]:
    def frame(t, rng):
        return [raw("s1", "speech", jit(rng, 10 * t, 3))]  # walks 10 deg per second
    r = simulate(frame, 9)
    last = r.at(9)[0]
    return [("one id while it moves", len(r.ids_ever()) == 1, f"{sorted(r.ids_ever())}"),
            ("bearing follows it (within 20 deg at t=9)", near(last, 90, 20), f"{last.bearing_deg:.0f} deg")]


def bearing_arrives_late() -> list[Check]:
    def frame(t, rng):
        return [raw("s1", "speech", None if t < 3 else jit(rng, 30))]
    r = simulate(frame, 8)
    early, late = r.at(2.5), r.at(7.5)
    return [("announced before its direction is known (bearing None)", len(early) == 1 and early[0].bearing_deg is None, f"{[s.bearing_deg for s in early]}"),
            ("gets its direction later, same single source", len(late) == 1 and near(late[0], 30), f"{[(s.source_id, s.bearing_deg) for s in late]}"),
            ("one id ever", len(r.ids_ever()) == 1, f"{sorted(r.ids_ever())}")]


def one_source_in_two_streams() -> list[Check]:
    def frame(t, rng):
        return [raw("s1", "speech", jit(rng, 30)), raw("s2", "speech", jit(rng, 32), level=-38)]
    r = simulate(frame, 8)
    return [("the same source heard in two streams counts once", len(r.at(7.5)) == 1, f"{len(r.at(7.5))} sources")]


def talkers_crossing() -> list[Check]:
    def frame(t, rng):  # A stands at 0 deg; B walks past A from -60 to +60 deg over 12 s
        return [raw("s1", "speech", jit(rng, 0, 3)), raw("s2", "speech", jit(rng, -60 + 10 * t, 3))]
    r = simulate(frame, 12)
    end = r.at(12)
    a, b = by_bearing(end, 0, 15), by_bearing(end, 60, 15)
    a0 = by_bearing(r.at(2), 0, 15)
    return [("two sources the whole time, no id appears or disappears", len(r.ids_ever()) == 2 and all(len(o) == 2 for o in r.out[10:]), f"{sorted(r.ids_ever())}"),
            ("the one who stood still keeps its id through the crossing", a0 is not None and a is not None and a.source_id == a0.source_id, f"{getattr(a0, 'source_id', None)} -> {getattr(a, 'source_id', None)}"),
            ("the walker ends near +60 as a different id", b is not None and a is not None and b.source_id != a.source_id, f"{getattr(b, 'source_id', None)}")]


def duplicate_stream_without_direction() -> list[Check]:
    def frame(t, rng):  # s2 carries the same voice but the locator has no direction for it for 4 s
        return [raw("s1", "speech", jit(rng, 30)), raw("s2", "speech", None if t < 4 else jit(rng, 31), level=-38)]
    r = simulate(frame, 9)
    return [("the direction-less copy is never announced then removed (one id ever)", len(r.ids_ever()) == 1, f"{sorted(r.ids_ever())}"),
            ("one source at the end", len(r.at(9)) == 1, f"{len(r.at(9))}")]


def slot_reused_by_other_sound() -> list[Check]:
    def frame(t, rng):
        if t < 3:
            return [raw("s1", "speech", jit(rng, 0))]
        if 6 <= t < 9:
            return [raw("s1", "other", None)]  # an unrelated sound lands in the same slot, no direction
        return []
    r = simulate(frame, 10)
    t1 = [s for s in r.at(8) if s.source_id == "t1"]
    return [("the paused talker keeps its own label and direction", len(t1) == 1 and t1[0].label == "speech" and near(t1[0], 0, 10), f"{[(s.label, s.bearing_deg) for s in t1]}"),
            ("the paused talker is not 'tracking' while it is silent", len(t1) == 1 and t1[0].state != p.TrackState.tracking, f"{[s.state.value for s in t1]}")]


def wraps_around_behind() -> list[Check]:
    def frame(t, rng):  # directly behind: readings land on both sides of +-180
        return [raw("s1", "speech", jit(rng, 180, 6))]
    r = simulate(frame, 8)
    last = r.at(8)
    return [("one id while readings jump between +170 and -170", len(r.ids_ever()) == 1, f"{sorted(r.ids_ever())}"),
            ("bearing is behind (within 20 deg of 180)", len(last) == 1 and abs(wrap180(last[0].bearing_deg - 180)) < 20, f"{[s.bearing_deg for s in last]}")]


SCENES = [two_talkers_swap_slots, pause_and_return, head_turn_in_view, blip_ignored,
          same_direction_two_labels, moving_source, bearing_arrives_late, one_source_in_two_streams,
          talkers_crossing, duplicate_stream_without_direction, slot_reused_by_other_sound, wraps_around_behind]


# ----------------------------------------------------------------------------- probes
# Edge cases for workstream B (offline identity checks). Each states what SHOULD happen for the
# true sources; tests/track/test_track.py marks the ones the tracker is known to fail (xfail,
# strict) with the reason, so a fix flips them.
def near_talkers(gap: float):
    def scene() -> list[Check]:
        def frame(t, rng):  # two different talkers, same label, gap deg apart, both talking throughout
            return [raw("s1", "speech", jit(rng, 30, 3)), raw("s2", "speech", jit(rng, 30 + gap, 3))]
        r = simulate(frame, 10)
        return [(f"two talkers {gap:.0f} deg apart are two sources at the end", len(r.at(9.5)) == 2, f"{len(r.at(9.5))}"),
                ("never more than two ids", len(r.ids_ever()) <= 2, f"{sorted(r.ids_ever())}")]
    scene.__name__ = f"near_talkers_{gap:.0f}deg"
    return scene


def pause_beyond_forget() -> list[Check]:
    def frame(t, rng):
        if t < 3:
            return [raw("s1", "speech", jit(rng, 0))]
        if 70 <= t < 75:
            return [raw("s2", "speech", jit(rng, 0))]
        return []
    r = simulate(frame, 75)
    ids_end = [s.source_id for s in r.at(74.5)]
    return [("gone after forget_s of silence (t=65)", r.at(65) == [], f"{[s.source_id for s in r.at(65)]}"),
            ("back as one source (by design a NEW id: forget_s passed)", len(ids_end) == 1 and ids_end != ["t1"], f"{ids_end}"),
            ("ids not reused", len(r.ids_ever()) == 2, f"{sorted(r.ids_ever())}")]


def two_bearingless_births() -> list[Check]:
    def frame(t, rng):  # two talkers start together; the locator has no direction for either for 6 s
        if t < 6:
            return [raw("s1", "speech", None), raw("s2", "speech", None)]
        return [raw("s1", "speech", jit(rng, -40)), raw("s2", "speech", jit(rng, 50))]
    r = simulate(frame, 12)
    return [("both announced before directions exist (t=5)", len(r.at(5)) == 2, f"{len(r.at(5))}"),
            ("two sources with directions at the end", len(r.at(11.5)) == 2
             and by_bearing(r.at(11.5), -40) is not None and by_bearing(r.at(11.5), 50) is not None,
             f"{[(s.source_id, s.bearing_deg) for s in r.at(11.5)]}"),
            ("never more than two ids", len(r.ids_ever()) <= 2, f"{sorted(r.ids_ever())}")]


def pause_while_crossing() -> list[Check]:
    def frame(t, rng):  # A stands at 0; B walks -60 -> +60 over 12 s, silent 4-8 s while passing A,
        out = [raw("s1", "speech", jit(rng, 0, 3))]                     # back in another slot
        if not 4 <= t < 8:
            out.append(raw("s2" if t < 4 else "s3", "speech", jit(rng, -60 + 10 * t, 3)))
        return out
    r = simulate(frame, 12)
    b0, b1 = by_bearing(r.at(3.5), -25), by_bearing(r.at(11.5), 55)
    a0, a1 = by_bearing(r.at(3.5), 0), by_bearing(r.at(11.5), 0)
    return [("the one standing still keeps its id", a0 is not None and a1 is not None and a0.source_id == a1.source_id,
             f"{getattr(a0, 'source_id', None)} -> {getattr(a1, 'source_id', None)}"),
            ("the walker keeps its id across a pause it walked 40 deg during", b0 is not None and b1 is not None
             and b0.source_id == b1.source_id, f"{getattr(b0, 'source_id', None)} -> {getattr(b1, 'source_id', None)}")]


def return_near_other() -> list[Check]:
    def frame(t, rng):  # A talks throughout at 0 in s1; B talks at -24 until 3 s, comes back at +10 in s3
        out = [raw("s1", "speech", jit(rng, 0, 2))]
        if t < 3:
            out.append(raw("s2", "speech", jit(rng, -24, 2)))
        elif t >= 6:
            out.append(raw("s3", "speech", jit(rng, 10, 2)))
        return out
    r = simulate(frame, 10)
    a0, a1 = by_bearing(r.at(2.5), 0, 6), by_bearing(r.at(9.5), 0, 6)
    return [("the talker who never stopped keeps its id when another comes back nearby",
             a0 is not None and a1 is not None and a0.source_id == a1.source_id,
             f"{getattr(a0, 'source_id', None)} -> {getattr(a1, 'source_id', None)}")]


def low_conf_second_talker() -> list[Check]:
    def frame(t, rng):  # the locator is sure of A but calls B "ambiguous" (conf 0.15) the whole time
        return [raw("s1", "speech", jit(rng, -30), conf=0.8), raw("s2", "speech", jit(rng, 40), conf=0.15)]
    r = simulate(frame, 20)
    return [("a second talker whose direction stays unsure is still announced", len(r.at(19.5)) == 2,
             f"{[(s.source_id, s.bearing_deg) for s in r.at(19.5)]}")]


def unreliable_bearings() -> list[Check]:
    def frame(t, rng):  # every other reading is junk from anywhere, flagged with low confidence
        if int(t * 10) % 2:
            return [raw("s1", "speech", float(rng.uniform(-180, 180)), conf=0.15)]
        return [raw("s1", "speech", jit(rng, 30), conf=0.8)]
    r = simulate(frame, 10)
    last = r.at(9.5)
    return [("one id", len(r.ids_ever()) == 1, f"{sorted(r.ids_ever())}"),
            ("low-confidence junk does not move the bearing (within 10 deg of 30)", len(last) == 1 and near(last[0], 30, 10),
             f"{[s.bearing_deg for s in last]}")]


def alternating_near_talkers() -> list[Check]:
    def frame(t, rng):  # A at 20 and B at 30 take turns in the same slot (one separated voice at a time)
        b = 30 if 5 <= t < 10 else 20
        return [raw("s1", "speech", jit(rng, b, 3))]
    r = simulate(frame, 15)
    return [("two different talkers get two ids (needs voice identity: same slot, 10 deg apart)",
             len(r.ids_ever()) == 2, f"{sorted(r.ids_ever())}")]


PROBES = [near_talkers(8), near_talkers(15), near_talkers(20), pause_beyond_forget, two_bearingless_births,
          pause_while_crossing, return_near_other, low_conf_second_talker, unreliable_bearings,
          alternating_near_talkers]


def main() -> int:
    want = sys.argv[1:]
    bad = 0
    for sc in SCENES + PROBES:
        if want and not any(w in sc.__name__ for w in want):
            continue
        print(f"\n{sc.__name__}")
        for name, ok, val in sc():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{val}]" if val else ""))
            bad += not ok
    print("\nRESULT:", "ALL PASS" if bad == 0 else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
