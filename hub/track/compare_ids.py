"""Compare tracker ids (t1, t2, ...) with Phase 4 persistent source ids (src_1, ...) on one recording.

    cd hub
    uv run python -m track.compare_ids <phase4 run dir> <tracker log.jsonl> --audio-start-us 1000000

Inputs:
* Phase 4 long-run output (docs/AV_SOUND_DISCOVERY_SPEC.md 12, hub/av_scene/schemas.py on the
  Phase 4 branch): timeline/index.json (sample_rate, pages) and its pages of TimelineSegment rows
  (source_id, half-open [start_sample, end_sample)). Only acoustic ids ("src_<n>") are compared.
* The tracker's output as protocol SourcesUpdated messages, one JSON per line (what the hub sends
  as sources.updated; `track.offline --log-dir` writes the same for simulated cases).
* --audio-start-us: device time of Phase 4 sample 0 (a mockclient session: its start_ts_us).

Phase 4 ids carry no bearing, so ids are paired by WHEN they are active, on a 100 ms grid (a
tracker id is active while its state is "tracking"; a src stays active --latency-s after each
span, because the tracker hears a 2 s window and holds a source for 1.5 s after it stops).
"Clean" time, when exactly one src and one tracker id are active, decides the pairing when a src
has --min-clean-s of it (score = share of the src's clean time). Otherwise the score is the lift
P(id active | src active) - P(id active | src silent), so an id that is on the whole time (e.g.
the first talker announced while the second is not yet) does not collect every src. Two talkers
who always overlap have neither: they stay unpaired. Reported:

* overlap table: seconds each tracker id and src were active together (clean seconds in brackets);
* pairing: for every src, the best tracker id, its score, and the margin over the runner-up
  (ambiguous below --margin);
* splits: a src spread over several tracker ids (identity switches the tracker made, or Phase 4
  merged two talkers); merges: one tracker id paired with several srcs;
* timeline of disagreements: grid spans where the src is active alone but its paired id is not.

No real Phase 4 output exists yet (identity task T1 of PR #58 pending), so this runs on
fixtures and simulated logs only so far.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DT_US = 100_000


def load_phase4(run_dir: Path) -> tuple[int, dict[str, list[tuple[int, int]]]]:
    """(sample_rate, src -> sorted [start_sample, end_sample) spans) from a Phase 4 run folder."""
    idx = json.loads((run_dir / "timeline" / "index.json").read_text())
    spans: dict[str, list[tuple[int, int]]] = {}
    for page in idx.get("pages", []):
        doc = json.loads((run_dir / page["path"]).read_text())
        for seg in doc.get("segments", []):
            if seg["source_id"].startswith("src_"):
                spans.setdefault(seg["source_id"], []).append((int(seg["start_sample"]), int(seg["end_sample"])))
    return int(idx["sample_rate"]), {k: sorted(v) for k, v in spans.items()}


def load_tracker_log(path: Path) -> list[tuple[int, set[str]]]:
    """[(device_ts_us, tracker ids in state tracking)] from SourcesUpdated JSON lines."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            m = json.loads(line)
            out.append((int(m["device_ts_us"]), {s["source_id"] for s in m["sources"] if s.get("state") == "tracking"}))
    return sorted(out, key=lambda r: r[0])


def grid(rate: int, spans: dict, log: list, audio_start_us: int, latency_s: float = 0.0):
    """Grid times and, per grid point, the set of active srcs and active tracker ids. A src counts
    as active for latency_s after each span (the tracker hears a window of the past and holds a
    source as tracking for a while after it stops)."""
    t0, t1 = log[0][0], log[-1][0]
    times = np.arange(t0, t1 + 1, DT_US, dtype=np.int64)  # int32 by default on Windows
    lag = latency_s * rate
    act_src = []
    for t in times:
        s = (t - audio_start_us) * rate / 1e6
        act_src.append({k for k, v in spans.items() if any(a <= s < b + lag for a, b in v)})
    li, act_trk = 0, []
    for t in times:
        while li + 1 < len(log) and log[li + 1][0] <= t:
            li += 1
        act_trk.append(log[li][1])
    return times, act_src, act_trk


def lift(on_t: np.ndarray, on_s: np.ndarray) -> float | None:
    """P(tracker id active | src active) - P(tracker id active | src silent); None if the src is
    never or always active (no contrast). An id that is always on scores ~0, not 1."""
    if on_s.all() or not on_s.any():
        return None
    return float(on_t[on_s].mean() - on_t[~on_s].mean())


def compare(rate: int, spans: dict, log: list, audio_start_us: int, min_clean_s: float = 2.0,
            margin: float = 0.2, latency_s: float = 2.0, min_lift: float = 0.3) -> dict:
    times, act_src, act_trk = grid(rate, spans, log, audio_start_us, latency_s)
    dt = DT_US / 1e6
    srcs = sorted(spans)
    tids = sorted({t for _, ids in log for t in ids}, key=lambda s: (len(s), s))
    both = {(t, s): 0.0 for t in tids for s in srcs}
    clean = dict(both)
    for a_s, a_t in zip(act_src, act_trk):
        for s in a_s:
            for t in a_t:
                both[(t, s)] += dt
                if len(a_s) == 1 and len(a_t) == 1:
                    clean[(t, s)] += dt
    on_s = {s: np.array([s in a for a in act_src]) for s in srcs}
    on_t = {t: np.array([t in a for a in act_trk]) for t in tids}
    pairing = {}
    for s in srcs:
        if sum(clean[(t, s)] for t in tids) >= min_clean_s:
            ev, kind = {t: clean[(t, s)] for t in tids}, "clean"
            tot = sum(ev.values())
            ev = {t: v / tot for t, v in ev.items()}  # share of the clean evidence
            floor = 0.0
        else:
            ev = {t: lift(on_t[t], on_s[s]) for t in tids}
            ev, kind, floor = {t: v for t, v in ev.items() if v is not None}, "lift", min_lift
        ranked = sorted(ev.items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] <= floor:
            pairing[s] = {"tid": None, "score": None, "margin": None, "evidence": kind if ranked else "none",
                          "status": "unpaired"}
            continue
        top = ranked[0][1]
        second = max(ranked[1][1], 0.0) if len(ranked) > 1 else 0.0
        pairing[s] = {"tid": ranked[0][0], "score": round(top, 3), "margin": round(top - second, 3),
                      "evidence": kind, "status": "paired" if top - second >= margin else "ambiguous",
                      "other_tids": [t for t, v in ranked[1:] if v >= max(0.1 * top, floor)]}
    by_tid: dict[str, list[str]] = {}
    for s, pr in pairing.items():
        if pr["tid"] is not None and pr["status"] == "paired":
            by_tid.setdefault(pr["tid"], []).append(s)
    disagree = []
    for s in srcs:
        tid = pairing[s]["tid"]
        if tid is None:
            continue
        run = None
        for t, a_s, a_t in zip(times, act_src, act_trk):
            bad = a_s == {s} and tid not in a_t
            if bad and run is None:
                run = [int(t), int(t)]
            elif bad:
                run[1] = int(t)
            elif run is not None:
                disagree.append({"src": s, "tid": tid, "from_us": run[0], "to_us": run[1] + DT_US,
                                 "tracker_active": sorted(act_trk[int((run[0] - times[0]) // DT_US)])})
                run = None
        if run is not None:
            disagree.append({"src": s, "tid": tid, "from_us": run[0], "to_us": run[1] + DT_US,
                             "tracker_active": sorted(act_trk[int((run[0] - times[0]) // DT_US)])})
    return {"overlap_s": {f"{t}|{s}": round(v, 1) for (t, s), v in both.items() if v > 0},
            "clean_s": {f"{t}|{s}": round(v, 1) for (t, s), v in clean.items() if v > 0},
            "pairing": pairing,
            "splits": {s: pr["other_tids"] for s, pr in pairing.items() if pr.get("other_tids")},
            "merges": {t: ss for t, ss in by_tid.items() if len(ss) > 1},
            "disagreements": [d for d in disagree if d["to_us"] - d["from_us"] >= 5 * DT_US]}


def print_report(r: dict) -> None:
    keys = sorted({k.split("|")[1] for k in r["overlap_s"]})
    tids = sorted({k.split("|")[0] for k in r["overlap_s"]}, key=lambda s: (len(s), s))
    print("overlap seconds (clean in brackets):")
    print("      " + "".join(f"{s:>16s}" for s in keys))
    for t in tids:
        cells = [f"{r['overlap_s'].get(f'{t}|{s}', 0):7.1f} [{r['clean_s'].get(f'{t}|{s}', 0):5.1f}] " for s in keys]
        print(f"{t:6s}" + "".join(cells))
    print("\npairing:")
    for s, pr in r["pairing"].items():
        print(f"  {s:8s} -> {pr['tid']}  {pr['status']}  score {pr['score']}  margin {pr['margin']}  "
              f"({pr['evidence']})" + (f"  also {pr['other_tids']}" if pr.get("other_tids") else ""))
    if r["merges"]:
        print(f"merges (one tracker id, several srcs): {r['merges']}")
    print(f"disagreements >= 0.5 s: {len(r['disagreements'])}")
    for d in r["disagreements"][:20]:
        print(f"  {d['src']} alone {d['from_us'] / 1e6:.1f}-{d['to_us'] / 1e6:.1f} s, tracker had {d['tracker_active']} not {d['tid']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m track.compare_ids", description=__doc__.split("\n")[0])
    ap.add_argument("phase4_run", type=Path)
    ap.add_argument("tracker_log", type=Path)
    ap.add_argument("--audio-start-us", type=int, required=True, help="device time of Phase 4 sample 0")
    ap.add_argument("--min-clean-s", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--latency-s", type=float, default=2.0, help="src activity counts this long after it stops")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    rate, spans = load_phase4(a.phase4_run)
    r = compare(rate, spans, load_tracker_log(a.tracker_log), a.audio_start_us, a.min_clean_s, a.margin, a.latency_s)
    print_report(r)
    if a.json:
        a.json.write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
