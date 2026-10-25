"""Print ONE markdown results table from the scoreboard and the delay test.

    cd hub && uv run python -m locate.bench ../hub/recordings --json ../bench.json
    uv run --project hub python eval/delay_test.py --json delay.json        # with the headset
    uv run --project hub python eval/results_table.py --bench bench.json --delay delay.json
    uv run --project hub python eval/results_table.py --bench bench.json --method tdoa-loo --out table.md

Inputs are the JSON files written by `locate.bench --json` ({"rows", "summaries"}) and by
eval/delay_test.py --json. Either may be left out: its rows say "no data". The table is for the
README once real recordings exist; the example files in eval/examples/ are hand-made numbers
(they carry "example": true and the table is then headed EXAMPLE).

Targets (docs/STATUS.md and hub/locate/bench.py TARGETS): median error <= 15 deg in front and
<= 25 deg behind, front/back right >= 95%, locked within 3 s, two speakers within 25 deg
(not measured by locate.bench: it scores one source per take). Status is judged on the median
error per region and on the median seconds to lock (a take that never locks counts as infinite).
Confidence honesty has no target; it reports the expected calibration error (0 = honest) and how
often the locator is right when it says it is 90% sure. The delay budget is the one in the delay
JSON (an assumption, see eval/delay_test.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

METHOD_PREFERENCE = ("tdoa-loo", "tdoa", "level")
FRONT_TARGET_DEG, BEHIND_TARGET_DEG = 15.0, 25.0
FB_TARGET, LOCK_TARGET_S = 0.95, 3.0
MET, NOT_MET, NO_DATA, NO_TARGET = "met", "NOT MET", "no data", "no target"


def _err(xs: list[float]) -> str:
    return f"{np.median(xs):.1f} / {np.percentile(xs, 90):.1f} deg (n={len(xs)})"


def pick_method(summaries: dict, wanted: str | None) -> str | None:
    if wanted:
        if wanted not in summaries:
            raise SystemExit(f"method {wanted!r} not in the bench JSON (has {sorted(summaries)})")
        return wanted
    return next((m for m in METHOD_PREFERENCE if m in summaries and summaries[m].get("n")), next(iter(summaries), None))


def rows_from_bench(s: dict | None, method: str | None) -> list[tuple[str, str, str, str]]:
    names = ["Error in front, median / p90", "Error behind, median / p90", "Error at the side, median / p90 (info)",
             "Front/back right", "Seconds to lock", "Confidence honesty", "Two speakers within 25 deg"]
    targets = [f"median <= {FRONT_TARGET_DEG:g} deg", f"median <= {BEHIND_TARGET_DEG:g} deg", "none",
               f">= {100 * FB_TARGET:g}%", f"<= {LOCK_TARGET_S:g} s (median)", "none (ECE: 0 = honest)",
               "within 25 deg"]
    if not s or not s.get("n"):
        return [(n, t, "-", NO_DATA) for n, t in zip(names, targets)]
    out = []
    reg = s.get("by_region", {})
    for name, target, key, limit in ((names[0], targets[0], "front", FRONT_TARGET_DEG),
                                     (names[1], targets[1], "behind", BEHIND_TARGET_DEG),
                                     (names[2], targets[2], "side", None)):
        e = reg.get(key) or []
        if not e:
            out.append((name, target, "-", NO_DATA if limit else NO_TARGET))
        else:
            out.append((name, target, _err(e), NO_TARGET if limit is None else (MET if np.median(e) <= limit else NOT_MET)))
    fb = s.get("fb_right_rate")
    out.append((names[3], targets[3], "-" if fb is None else f"{100 * fb:.0f}% (n={s.get('n_fb', 0)})",
                NO_DATA if fb is None else (MET if fb >= FB_TARGET else NOT_MET)))
    locks = s.get("lock_s") or []
    if locks:
        vals = [math.inf if x is None else x for x in locks]
        med = float(np.median(vals))
        within = sum(v <= LOCK_TARGET_S for v in vals)
        never = sum(math.isinf(v) for v in vals)
        out.append((names[4], targets[4], f"median {'never' if math.isinf(med) else f'{med:.1f} s'}; "
                    f"within {LOCK_TARGET_S:g} s: {within}/{len(vals)}; never: {never}",
                    MET if med <= LOCK_TARGET_S else NOT_MET))
    else:
        out.append((names[4], targets[4], "-", NO_DATA))
    if s.get("ece") is None:
        out.append((names[5], targets[5], "-", NO_DATA))
    else:
        sure = (f"; right {100 * s['hit_rate_when_90']:.0f}% of {s['n_90']} when >= 90% sure" if s.get("n_90")
                else "; never >= 90% sure")
        out.append((names[5], targets[5], f"ECE {s['ece']:.2f}{sure}", NO_TARGET))
    out.append((names[6], targets[6], "not measured by locate.bench (one source per take)", NO_DATA))
    return out


def rows_from_delay(d: dict | None) -> list[tuple[str, str, str, str]]:
    names = ("Round trip median / p90", "Round trip worst case")
    if not d or "median_ms" not in d:
        return [(n, "-", "-", NO_DATA) for n in names]
    n_valid, need = d.get("n_valid", 0), d.get("min_trials", 30)
    # Each row is judged on its own number; a run with too few heard trials is judged on neither.
    if n_valid < need:
        med_ok = worst_ok = f"too few trials ({n_valid} of {need})"
    else:
        med_ok = MET if d["median_ms"] <= d["budget_median_ms"] else NOT_MET
        worst_ok = MET if d["worst_ms"] <= d["budget_worst_ms"] else NOT_MET
    # A simulated or example run must not read as a result: say so in the status itself.
    flags = [x for x, on in (("mock", d.get("mode") == "mock"), ("example", d.get("example"))) if on]
    tag = f" ({', '.join(flags)}, not a result)" if flags else ""
    return [
        (names[0], f"<= {d['budget_median_ms']:g} ms (assumed)",
         f"{d['median_ms']:.0f} / {d['p90_ms']:.0f} ms (n={n_valid}/{d['n_trials']})", med_ok + tag),
        (names[1], f"<= {d['budget_worst_ms']:g} ms (assumed)", f"{d['worst_ms']:.0f} ms", worst_ok + tag),
    ]


def build_table(bench: dict | None, delay: dict | None, method: str | None = None) -> str:
    summaries = (bench or {}).get("summaries", {})
    m = pick_method(summaries, method) if summaries else None
    s = summaries.get(m) if m else None
    lines = []
    example = bool((bench or {}).get("example") or (delay or {}).get("example"))
    if example:
        lines.append("**EXAMPLE: hand-made numbers to show the format, not measurements.**\n")
    head = f"Direction method: `{m}`" if m else "Direction method: none (no bench data)"
    if s:
        head += f", {s['n']} valid take(s)" + (f", {s['n_invalid']} invalid and not scored" if s.get("n_invalid") else "")
    lines.append(head + "\n")
    lines += ["| Metric | Target | Result | Status |", "|---|---|---|---|"]
    for name, target, result, status in rows_from_bench(s, m) + rows_from_delay(delay):
        lines.append(f"| {name} | {target} | {result} | {status} |")
    if delay and "evidence" in delay:
        lines.append(f"\nRound-trip evidence: {delay['evidence']}.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", type=Path, help="JSON from `python -m locate.bench --json`")
    ap.add_argument("--delay", type=Path, help="JSON from eval/delay_test.py --json")
    ap.add_argument("--method", help="direction method to report (default tdoa-loo, else tdoa, else level)")
    ap.add_argument("--out", type=Path, help="also write the table here")
    a = ap.parse_args(argv)
    if not (a.bench or a.delay):
        ap.error("give --bench and/or --delay")
    try:
        bench = json.loads(a.bench.read_text()) if a.bench else None
        delay = json.loads(a.delay.read_text()) if a.delay else None
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return 2
    table = build_table(bench, delay, a.method)
    sys.stdout.write(table)
    if a.out:
        a.out.write_text(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
