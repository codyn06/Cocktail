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


def by_bearing(srcs: list[p.Source], b: float) -> p.Source | None:
    c = [s for s in srcs if near(s, b)]
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


SCENES = [two_talkers_swap_slots, pause_and_return, head_turn_in_view, blip_ignored,
          same_direction_two_labels, moving_source, bearing_arrives_late, one_source_in_two_streams]


def main() -> int:
    want = sys.argv[1:]
    bad = 0
    for sc in SCENES:
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
