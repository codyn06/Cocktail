"""Acceptance metrics for Phase 4 long clips. Pure numpy/stdlib; no torch, no model imports.

Everything takes PLAIN DATA so it scores mocked output now and real-model output later:

* predictions are lists of `Seg` (or dicts / `schemas.TimelineSegment`-like objects, see `as_seg`), in samples;
* ground truth is dict[str, list[(start, end)]] of half-open intervals in the SAME sample units
  (16 kHz speech samples by default; `load_timeline(...).segments_at(16000)` converts a bundle);
* `None` in a returned number means "not applicable" (an empty denominator), never "zero".

Vocabulary
  speaker   ground-truth voice (acoustic identity)         person  ground-truth visible human
  source    predicted persistent acoustic id (`src_n`)      track   predicted visual track id
A segment is *attributed* when association_status == "matched", it names a track and it carries no refusal_reason.
Everything else (ambiguous / unmatched / rejected / uncovered time) counts as "no attribution claimed".

Gate numbers (see `GATES`): only the 0.02 silent-bystander ceiling comes from the Phase 4 spec. The 0.80 floors
(active-speaker recall, refusal precision, refusal recall) and the 0.05 wrong-person ceiling are PLACEHOLDERS chosen
by the eval worker before any model run; they await orchestrator/team confirmation and the result always carries
`provisional: true`.
"""

from __future__ import annotations

import bisect
import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

HUB = Path(__file__).resolve().parents[2] / "hub"
SR16 = 16_000

GATES = {
    "active_speaker_recall_floor": 0.80,  # placeholder, unconfirmed
    "refusal_precision_floor": 0.80,      # placeholder, unconfirmed
    "refusal_recall_floor": 0.80,         # placeholder, unconfirmed
    "silent_bystander_ceiling": 0.02,     # the spec gate: highlighted silent person-time < 2 %  (SPEC)
    "wrong_person_rate_ceiling": 0.05,    # placeholder, unconfirmed
}

Interval = tuple[int, int]


# ------------------------------------------------------------------------------------ hub module loading
def load_hub_module(name: str):
    """`av_scene.<name>`, so class identity matches `replay/chunked.py` and the pipeline (frozen-dataclass equality and
    isinstance would fail against a duplicate copy). Falls back to loading the file standalone when the package
    cannot be imported; `av_scene/__init__` itself imports only stdlib (it configures FFmpeg DLL paths)."""
    full = f"av_scene.{name}"
    if full in sys.modules:
        return sys.modules[full]
    if str(HUB) not in sys.path:
        sys.path.insert(0, str(HUB))
    try:
        return importlib.import_module(full)
    except ImportError:
        pass
    alias = f"cocktail_hub_standalone_{name}"
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, HUB / "av_scene" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------- intervals
def merge(iv: Iterable[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for a, b in sorted((int(a), int(b)) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def intersect(x: Iterable[Interval], y: Iterable[Interval]) -> list[Interval]:
    xs, ys, out, i, j = merge(x), merge(y), [], 0, 0
    while i < len(xs) and j < len(ys):
        a, b = max(xs[i][0], ys[j][0]), min(xs[i][1], ys[j][1])
        if b > a:
            out.append((a, b))
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(x: Iterable[Interval], y: Iterable[Interval]) -> list[Interval]:
    ys, out = merge(y), []
    for a, b in merge(x):
        cur = a
        for c, d in ys:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
        if cur < b:
            out.append((cur, b))
    return out


def total(iv: Iterable[Interval]) -> int:
    return sum(b - a for a, b in merge(iv))


def erode(iv: Iterable[Interval], guard: int) -> list[Interval]:
    """Shrink every interval by `guard` samples on each side (boundary tolerance for silent-time scoring)."""
    return [(a + guard, b - guard) for a, b in merge(iv) if b - a > 2 * guard]


def union_of(m: Mapping[str, Iterable[Interval]] | Iterable[Interval]) -> list[Interval]:
    if isinstance(m, Mapping):
        return merge(iv for v in m.values() for iv in v)
    return merge(m)


# ------------------------------------------------------------------------------------------------ segments
@dataclass(frozen=True)
class Seg:
    source_id: str
    start: int
    end: int
    track_id: str | None = None
    status: str = "unmatched"            # association_status
    visibility: str = "unknown"
    stem: str = "not_attempted"          # stem_availability
    refusal: str | None = None
    kind: str = "speech"                 # TimelineSource.kind, when known
    chunk_index: int | None = None
    slot: int | None = None


_ALIASES = {"start": ("start", "start_sample"), "end": ("end", "end_sample"), "status": ("status", "association_status"),
            "stem": ("stem", "stem_availability"), "refusal": ("refusal", "refusal_reason")}


def as_seg(x, kind: str | None = None) -> Seg:
    """Accept a Seg, a dict, or any object with TimelineSegment-like attributes."""
    if isinstance(x, Seg):
        return x
    get = (lambda k, d=None: x.get(k, d)) if isinstance(x, Mapping) else (lambda k, d=None: getattr(x, k, d))

    def pick(key, default=None):
        for k in _ALIASES.get(key, (key,)):
            v = get(k, None)
            if v is not None:
                return v
        return default

    return Seg(source_id=str(get("source_id")), start=int(pick("start")), end=int(pick("end")), track_id=get("track_id"),
               status=pick("status", "unmatched"), visibility=get("visibility", "unknown") or "unknown",
               stem=pick("stem", "not_attempted"), refusal=pick("refusal"),
               kind=kind or get("kind", "speech") or "speech", chunk_index=get("chunk_index"), slot=get("slot"))


def as_segs(xs: Iterable) -> list[Seg]:
    return sorted((as_seg(x) for x in xs), key=lambda s: (s.start, s.end, s.source_id))


def is_attributed(s: Seg) -> bool:
    return s.status == "matched" and s.track_id is not None and s.refusal is None


def is_refused(s: Seg) -> bool:
    """Explicit refusal: a reason was given or the association was left ambiguous/rejected."""
    return s.refusal is not None or s.status in ("ambiguous", "rejected")


def attributed_intervals(segs: Sequence[Seg], pred: Callable[[Seg], bool] = is_attributed) -> list[Interval]:
    return merge((s.start, s.end) for s in segs if pred(s))


def scale_segs(segs: Iterable[Seg], from_rate: int, to_rate: int = SR16) -> list[Seg]:
    if from_rate == to_rate:
        return list(segs)
    f = lambda v: (int(v) * to_rate + from_rate // 2) // from_rate            # half-up, integer arithmetic
    return [Seg(**{**s.__dict__, "start": f(s.start), "end": f(s.end)}) for s in segs]


# ----------------------------------------------------------------------------------- identity continuity
def _overlap_matrix(segs: Sequence[Seg], gt: Mapping[str, Sequence[Interval]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for g, iv in gt.items():
        row: dict[str, int] = {}
        for s in segs:
            o = total(intersect([(s.start, s.end)], iv))
            if o > 0:
                row[s.source_id] = row.get(s.source_id, 0) + o
        out[g] = row
    return out


def _id_totals(segs: Sequence[Seg]) -> dict[str, int]:
    by: dict[str, list[Interval]] = {}
    for s in segs:
        by.setdefault(s.source_id, []).append((s.start, s.end))
    return {k: total(v) for k, v in by.items()}


def _jaccard(segs: Sequence[Seg], gt: Mapping[str, Sequence[Interval]]) -> dict[str, dict[str, float]]:
    """speaker -> source id -> |id & speaker| / |id | speaker|. Raw overlap is ambiguous under cross-talk (a long
    id overlaps every speaker); Jaccard picks the speaker an id actually follows."""
    ov, tot = _overlap_matrix(segs, gt), _id_totals(segs)
    return {g: {i: o / (tot[i] + total(gt[g]) - o) for i, o in row.items()} for g, row in ov.items()}


def _dominant_sequence(segs: Sequence[Seg], active: Sequence[Interval], rank: Mapping[str, int]) -> list[tuple[int, int, str]]:
    """Predicted id over the speaker's active time, cell by cell; where several ids overlap (cross-talk) the id
    with the most overlap with this speaker overall wins. Time with no predicted segment is skipped (a gap is
    not a switch)."""
    active = merge(active)
    pts = sorted({p for s in segs for p in (s.start, s.end)} | {p for a, b in active for p in (a, b)})
    starts = [a for a, _ in active]
    seq: list[tuple[int, int, str]] = []
    for lo, hi in zip(pts, pts[1:]):
        k = bisect.bisect_right(starts, lo) - 1
        if k < 0 or active[k][1] < hi:
            continue
        ids = {s.source_id for s in segs if s.start <= lo and hi <= s.end}
        if not ids:
            continue
        sid = min(ids, key=lambda i: (rank.get(i, 1 << 60), i))
        if seq and seq[-1][2] == sid and seq[-1][1] == lo:
            seq[-1] = (seq[-1][0], hi, sid)
        else:
            seq.append((lo, hi, sid))
    return seq


ABSTAIN_IDS = frozenset({"unresolved"})


def _identified(pred: Iterable) -> list:
    """Segments that carry an acoustic identity. An ABSTENTION ("unresolved": identity ambiguous, blocked, or too
    little audio to enroll) is not an identity, so abstaining and later resolving is not a speaker swap; abstained time
    still counts against `source_coverage` / `active_speaker_recall`, which is what keeps abstaining from being free."""
    return [s for s in as_segs(pred) if s.source_id not in ABSTAIN_IDS]


def identity_switches(pred: Iterable, gt: Mapping[str, Sequence[Interval]], *, min_run_samples: int = 0) -> dict:
    """Times the predicted source id covering a ground-truth speaker changes while that speaker is active.
    Runs shorter than `min_run_samples` are ignored (boundary slivers). Gaps without a segment are skipped."""
    segs = _identified(pred)
    jac = _jaccard(segs, gt)
    per, runs = {}, {}
    for g, iv in gt.items():
        rank = {sid: r for r, (sid, _) in enumerate(sorted(jac[g].items(), key=lambda kv: (-kv[1], kv[0])))}
        seq = [r for r in _dominant_sequence(segs, iv, rank) if r[1] - r[0] >= min_run_samples]
        ids = [r[2] for r in seq]
        collapsed = [i for k, i in enumerate(ids) if k == 0 or i != ids[k - 1]]
        per[g] = max(0, len(collapsed) - 1)
        runs[g] = collapsed
    return {"total": sum(per.values()), "per_speaker": per, "id_sequence": runs}


def switches_per_minute(switches: int | Mapping, duration_samples: int, rate: int = SR16) -> float:
    n = switches["total"] if isinstance(switches, Mapping) else int(switches)
    minutes = duration_samples / rate / 60.0
    return n / minutes if minutes > 0 else float("nan")


def fragmentation(pred: Iterable, gt: Mapping[str, Sequence[Interval]], *, min_overlap_samples: int = 8000) -> dict:
    """Distinct predicted ids per ground-truth speaker beyond the first. An id counts for the one speaker it follows
    most (Jaccard), so cross-talk does not charge both speakers; ids with < `min_overlap_samples` of overlap are
    boundary slivers and are not counted."""
    segs = _identified(pred)
    ov, jac = _overlap_matrix(segs, gt), _jaccard(segs, gt)
    home: dict[str, str] = {}                                          # each id belongs to the speaker it follows most
    for g, row in jac.items():
        for i, v in row.items():
            if i not in home or v > jac[home[i]][i]:
                home[i] = g
    per = {g: max(0, sum(1 for i, o in ov[g].items() if home.get(i) == g and o >= min_overlap_samples) - 1) for g in gt}
    return {"total": sum(per.values()), "per_speaker": per}


# ------------------------------------------------------------------------------------------- attribution
def source_coverage(pred: Iterable, gt: Mapping[str, Sequence[Interval]] | Sequence[Interval]) -> float | None:
    """Share of ground-truth speaking time that has ANY predicted speech segment (acoustic coverage)."""
    d = union_of(gt)
    return None if not d else total(intersect(d, [(s.start, s.end) for s in as_segs(pred)])) / total(d)


def attribution_coverage(pred: Iterable, onscreen: Mapping[str, Sequence[Interval]] | Sequence[Interval]) -> float | None:
    """Share of on-screen speaking time (speaker active AND own person visible) that is attributed to some track,
    right or wrong. Compare with `active_speaker_recall` (right only) and `wrong_person_assignments`."""
    d = union_of(onscreen)
    return None if not d else total(intersect(d, attributed_intervals(as_segs(pred)))) / total(d)


def speaker_tracks(track_person: Mapping[str, str], person_speaker: Mapping[str, str | None]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for t, p in track_person.items():
        g = person_speaker.get(p)
        if g is not None:
            out.setdefault(g, set()).add(t)
    return out


def active_speaker_recall(pred: Iterable, onscreen: Mapping[str, Sequence[Interval]],
                          track_person: Mapping[str, str], person_speaker: Mapping[str, str | None]) -> float | None:
    """Share of on-screen speaking time attributed to the CORRECT person's track."""
    segs = as_segs(pred)
    tracks = speaker_tracks(track_person, person_speaker)
    den = sum(total(iv) for iv in onscreen.values())
    if den == 0:
        return None
    hit = 0
    for g, iv in onscreen.items():
        ok = attributed_intervals(segs, lambda s, ts=tracks.get(g, set()): is_attributed(s) and s.track_id in ts)
        hit += total(intersect(iv, ok))
    return hit / den


def wrong_person_assignments(pred: Iterable, speaker_active: Mapping[str, Sequence[Interval]],
                             track_person: Mapping[str, str], person_speaker: Mapping[str, str | None]) -> dict:
    """Attributed time that lands on a person who is NOT speaking while someone else (or an off-screen speaker)
    is. Attributed time over silence is not counted here; it is the silent-bystander metric's job."""
    any_speech = union_of(speaker_active)
    wrong = right = bad_segs = n = 0
    for s in as_segs(pred):
        if not is_attributed(s):
            continue
        n += 1
        g = person_speaker.get(track_person.get(s.track_id, s.track_id))
        mine = [(s.start, s.end)]
        r = total(intersect(mine, speaker_active.get(g, []))) if g is not None else 0
        w = total(intersect(mine, subtract(any_speech, speaker_active.get(g, []) if g is not None else [])))
        right += r
        wrong += w
        bad_segs += int(w > r)
    assigned = right + wrong
    return {"wrong_samples": wrong, "correct_samples": right, "wrong_segments": bad_segs, "attributed_segments": n,
            "wrong_rate": (wrong / assigned) if assigned else None}


def refusal_precision_recall(pred: Iterable, speaker_active: Mapping[str, Sequence[Interval]] | Sequence[Interval],
                             should_refuse: Sequence[Interval]) -> dict:
    """Positive class = "should refuse" over ground-truth speaking time D. Predicted-refused = D minus attributed
    time (explicit refusals AND silence both count as "no claim"). Report BOTH numbers: refusing everything has
    recall 1 but precision = base rate, and costs `active_speaker_recall`; see `gate_pass`."""
    d = union_of(speaker_active)
    should = intersect(should_refuse, d)
    refused = subtract(d, attributed_intervals(as_segs(pred)))
    tp = total(intersect(should, refused))
    return {"precision": (tp / total(refused)) if total(refused) else None,
            "recall": (tp / total(should)) if total(should) else None,
            "true_positive_samples": tp, "predicted_refused_samples": total(refused),
            "should_refuse_samples": total(should), "speaking_samples": total(d)}


def silent_bystander_highlight_rate(pred: Iterable, silent_visible: Mapping[str, Sequence[Interval]], *,
                                    highlight: Callable[[Seg, str], bool] | None = None,
                                    groups: Mapping[str, set[str]] | None = None,
                                    guard_samples: int = 0) -> float | None:
    """Highlighted silent person-time / annotated visible silent person-time (gate: < 0.02).

    `silent_visible`: key -> intervals where that person is visible and not speaking (key = track id, or a person id
    with `groups` mapping it to all of that person's track ids). `highlight(seg, key)` defaults to "attributed to
    that key's track(s)". Person-time sums over keys (two silent people highlighted for 1 s each = 2 person-seconds)."""
    segs = as_segs(pred)
    hl = highlight or (lambda s, t: is_attributed(s) and s.track_id in (groups or {}).get(t, {t}))
    den = num = 0
    for t, iv in silent_visible.items():
        iv = erode(iv, guard_samples) if guard_samples else merge(iv)
        den += total(iv)
        num += total(intersect(iv, merge((s.start, s.end) for s in segs if hl(s, t))))
    return None if den == 0 else num / den


def gate_pass(m: Mapping[str, float | None], **overrides) -> dict:
    """Combine the headline numbers. Keys of `m`: active_speaker_recall, refusal_precision, refusal_recall,
    silent_bystander_rate, optional wrong_person_rate. A key that is absent FAILS (not measured); a value of None
    means not applicable (empty denominator) and passes, flagged. Refusing everything fails twice: active-speaker
    recall collapses and refusal precision falls to the base rate."""
    g = {**GATES, **overrides}
    spec = [("active_speaker_recall", ">=", g["active_speaker_recall_floor"], True),
            ("refusal_precision", ">=", g["refusal_precision_floor"], True),
            ("refusal_recall", ">=", g["refusal_recall_floor"], True),
            ("silent_bystander_rate", "<", g["silent_bystander_ceiling"], True),
            ("wrong_person_rate", "<=", g["wrong_person_rate_ceiling"], False)]
    checks, ok = {}, True
    for key, op, lim, required in spec:
        if key not in m:
            checks[key] = {"value": None, "limit": lim, "op": op, "status": "fail" if required else "skipped",
                           "note": "not measured"}
        elif m[key] is None:
            checks[key] = {"value": None, "limit": lim, "op": op, "status": "n/a", "note": "empty denominator"}
        else:
            v = float(m[key])
            passed = v >= lim if op == ">=" else (v < lim if op == "<" else v <= lim)
            checks[key] = {"value": v, "limit": lim, "op": op, "status": "pass" if passed else "fail"}
        ok &= checks[key]["status"] != "fail"
    return {"passed": ok, "checks": checks, "provisional": True}


def score_all(pred: Iterable, *, speaker_active, onscreen, track_person, person_speaker, should_refuse,
              silent_visible, duration_samples: int, rate: int = SR16, groups=None, guard_samples: int = 0,
              **gate_overrides) -> dict:
    """Every metric above on one prediction, plus the gate. Keyword inputs match `GroundTruth.metric_inputs()` of the
    synthetic scenes and `long_annotations.Annotations.ground_truth()` of real clips."""
    segs = as_segs(pred)
    sw = identity_switches(segs, speaker_active)
    ref = refusal_precision_recall(segs, speaker_active, should_refuse)
    wrong = wrong_person_assignments(segs, speaker_active, track_person, person_speaker)
    m = {"identity_switches": sw["total"], "switches_per_minute": switches_per_minute(sw, duration_samples, rate),
         "fragmentation": fragmentation(segs, speaker_active)["total"],
         "source_coverage": source_coverage(segs, speaker_active),
         "attribution_coverage": attribution_coverage(segs, onscreen),
         "active_speaker_recall": active_speaker_recall(segs, onscreen, track_person, person_speaker),
         "wrong_person_rate": wrong["wrong_rate"], "refusal_precision": ref["precision"], "refusal_recall": ref["recall"],
         "silent_bystander_rate": silent_bystander_highlight_rate(segs, silent_visible, groups=groups,
                                                                  guard_samples=guard_samples)}
    return {"metrics": m, "gate": gate_pass(m, **gate_overrides),
            "detail": {"identity_switches": sw, "fragmentation": fragmentation(segs, speaker_active),
                       "wrong_person": wrong, "refusal": ref}}


# ---------------------------------------------------------------------------------------- memory growth
def memory_growth(workloads: Sequence[Mapping], budgets=None, warmup_chunks: int | None = None, *,
                  window_s: float = 4.0, hop_s: float = 2.0) -> dict:
    """Judge whole-process memory against `contracts.MemoryBudgets`.

    workloads: [{"duration_s": 300, "series": [(audio_s, rss_mib), ...], "rss_peak_mib": opt (default max series),
                 "gpu_peak_mib": number | None | "unavailable"}, ...]   (>= 1; peak growth needs >= 2)
    Warm-up ends after `warmup_chunks` chunks: window_s + (warmup_chunks - 1) * hop_s audio seconds (12 s for
    5 chunks of 4 s / 2 s hop). The slope (MiB per audio minute, least squares) is judged on the LONGEST workload
    (a 30 s workload leaves ~18 s post-warm-up, too short for a slope); the others are reported. Peak growth is
    peak(longest) - peak(shortest). GPU readings are device-wide and may be absent: `unavailable` never passes
    silently, it is listed and `complete` is False."""
    if budgets is None:
        budgets = load_hub_module("contracts").MemoryBudgets()
    wc = budgets.rss_warmup_chunks if warmup_chunks is None else warmup_chunks
    t0 = window_s + (wc - 1) * hop_s
    per, ws = {}, sorted(workloads, key=lambda w: w["duration_s"])
    for w in ws:
        pts = np.asarray([(float(a), float(r)) for a, r in w["series"]], dtype=np.float64).reshape(-1, 2)
        peak = float(w["rss_peak_mib"]) if w.get("rss_peak_mib") is not None else (float(pts[:, 1].max()) if len(pts) else None)
        post = pts[pts[:, 0] >= t0]
        slope = None
        if len(post) >= 3 and post[-1, 0] - post[0, 0] >= 1.0:
            slope = float(np.polyfit(post[:, 0] / 60.0, post[:, 1], 1)[0])
        gpu = w.get("gpu_peak_mib")
        per[str(w["duration_s"])] = {"duration_s": w["duration_s"], "n_samples": len(pts), "n_post_warmup": len(post),
                                     "rss_peak_mib": peak, "slope_mib_per_audio_min": slope,
                                     "gpu_peak_mib": gpu if isinstance(gpu, (int, float)) else None}
    checks: dict[str, dict] = {}

    def put(name, value, limit, status=None):
        if status is None:
            status = "unavailable" if value is None else ("pass" if value <= limit else "fail")
        checks[name] = {"value": value, "limit": limit, "status": status}

    peaks = [p["rss_peak_mib"] for p in per.values() if p["rss_peak_mib"] is not None]
    put("rss_peak_mib", max(peaks) if peaks else None, budgets.rss_peak_mib)
    longest = per[str(ws[-1]["duration_s"])] if ws else {}
    put("rss_slope_mib_per_audio_min", longest.get("slope_mib_per_audio_min"), budgets.rss_slope_mib_per_audio_min)
    if len(ws) >= 2 and None not in (per[str(ws[0]["duration_s"])]["rss_peak_mib"], longest.get("rss_peak_mib")):
        put("rss_peak_growth_mib", longest["rss_peak_mib"] - per[str(ws[0]["duration_s"])]["rss_peak_mib"],
            budgets.rss_peak_growth_mib_30_to_300s)
    else:
        put("rss_peak_growth_mib", None, budgets.rss_peak_growth_mib_30_to_300s)
    gp = [p["gpu_peak_mib"] for p in per.values() if p["gpu_peak_mib"] is not None]
    put("gpu_peak_mib", max(gp) if gp else None, budgets.gpu_peak_mib)
    return {"warmup_end_audio_s": t0, "per_workload": per, "checks": checks,
            "passed": not any(c["status"] == "fail" for c in checks.values()),
            "complete": all(c["status"] != "unavailable" for c in checks.values()),
            "unavailable": [k for k, c in checks.items() if c["status"] == "unavailable"],
            "note": "RSS = whole process tree; GPU = device-wide memory.used (other processes included)."}


# ---------------------------------------------------------------------------------------------- bundles
@dataclass
class LoadedTimeline:
    sample_rate: int
    n_samples: int
    sources: dict[str, dict]                         # source_id -> TimelineSource dict (kind, role, ...)
    segments: dict[str, list[Seg]] = field(default_factory=dict)

    def all_segments(self) -> list[Seg]:
        return sorted((s for v in self.segments.values() for s in v), key=lambda s: (s.start, s.end, s.source_id))

    def speech_segments(self) -> list[Seg]:
        """Speech voices: kind "speech" plus "outside_view" (an off-screen voice is still a separated speech source,
        listed apart only for the viewer). "tse:<track>" stems, events and the residual are excluded: they would
        double-count the same voice or are not speech."""
        return [s for s in self.all_segments() if s.kind in ("speech", "outside_view")]

    def segments_at(self, rate: int = SR16, *, speech_only: bool = True) -> list[Seg]:
        segs = self.speech_segments() if speech_only else self.all_segments()
        return scale_segs(segs, self.sample_rate, rate)


def load_timeline(bundle_dir: str | Path) -> LoadedTimeline:
    """Read `timeline/index.json` + its pages (`schemas.TimelineIndex` / `TimelinePageDoc`)."""
    sch = load_hub_module("schemas")
    root = Path(bundle_dir)
    index = sch.TimelineIndex.model_validate_json((root / "timeline" / "index.json").read_text(encoding="utf-8"))
    kinds = {s.source_id: s for s in index.sources}
    segs: dict[str, list[Seg]] = {s.source_id: [] for s in index.sources}
    for page in index.pages:
        doc = sch.TimelinePageDoc.model_validate_json((root / page.path).read_text(encoding="utf-8"))
        for t in doc.segments:
            kind = kinds[t.source_id].kind if t.source_id in kinds else None
            segs.setdefault(t.source_id, []).append(as_seg(t, kind=kind or "unknown"))
    for v in segs.values():
        v.sort(key=lambda s: (s.start, s.end))
    return LoadedTimeline(index.sample_rate, index.n_samples, {s.source_id: s.model_dump() for s in index.sources}, segs)


def _read_audio_file(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        x = np.load(path)
    elif ext == ".wav":
        from scipy.io import wavfile
        _, x = wavfile.read(path)
    else:
        raise ValueError(f"no reader for {path.name}: add one for {ext!r} (audio format is owned by the bundle writer)")
    if x.dtype.kind in "iu":
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max + 1)
    x = np.asarray(x, np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def snapshot_bundle(bundle_dir: str | Path, *, with_audio: bool = True) -> dict:
    """Plain-data run snapshot for `compare_runs`: per source the logical length, timeline segments and the
    zero-filled full-length audio (spans without a segment are silent by schema contract)."""
    sch = load_hub_module("schemas")
    root = Path(bundle_dir)
    tl = load_timeline(root)
    snap = {"timeline_samples": tl.n_samples, "sources": {}}
    for sid, segs in tl.segments.items():
        snap["sources"][sid] = {"segments": [s.__dict__ for s in segs], "n_samples": None, "audio": None}
    ai = root / "audio" / "index.json"
    if ai.is_file():
        idx = sch.AudioIndex.model_validate_json(ai.read_text(encoding="utf-8"))
        for sid, e in idx.sources.items():
            ent = snap["sources"].setdefault(sid, {"segments": [], "n_samples": None, "audio": None})
            ent["n_samples"], ent["rate_hz"] = e.n_samples, e.rate_hz
            if with_audio:
                buf = np.zeros(e.n_samples, np.float32)
                for ref in e.segments:
                    x = _read_audio_file(root / ref.path)[: ref.n_samples]
                    buf[ref.start_sample: ref.start_sample + len(x)] = x
                ent["audio"] = buf
    return snap


_CMP_FIELDS = ("source_id", "start", "end", "track_id", "status", "visibility", "stem", "refusal", "chunk_index", "slot")


def compare_runs(a: Mapping, b: Mapping, tol: float = 1e-4, *, ignore_fields: Sequence[str] = ()) -> dict:
    """Resume check on two plain snapshots {"sources": {sid: {"n_samples", "segments", "audio"}}}.
    Identical = same source ids, same timeline segments, same logical sample counts and array lengths, and audio
    within `tol` (max absolute difference over the shared span; reported either way)."""
    sa, sb = a["sources"], b["sources"]
    out = {"tol": tol, "missing_in_b": sorted(set(sa) - set(sb)), "missing_in_a": sorted(set(sb) - set(sa)),
           "timeline_diffs": [], "sample_count_diffs": [], "audio_length_diffs": [], "audio_max_abs_diff": 0.0,
           "audio_over_tol": []}
    keep = [f for f in _CMP_FIELDS if f not in ignore_fields]
    key = lambda s: tuple(getattr(as_seg(s), f) for f in keep)
    for sid in sorted(set(sa) & set(sb)):
        x, y = sa[sid], sb[sid]
        ka, kb = sorted(map(key, x.get("segments", []))), sorted(map(key, y.get("segments", [])))
        if ka != kb:
            out["timeline_diffs"].append({"source_id": sid, "n_a": len(ka), "n_b": len(kb),
                                          "first_diff": next((p for p in zip(ka, kb) if p[0] != p[1]), None)})
        if x.get("n_samples") != y.get("n_samples"):
            out["sample_count_diffs"].append({"source_id": sid, "a": x.get("n_samples"), "b": y.get("n_samples")})
        xa, ya = x.get("audio"), y.get("audio")
        if xa is None and ya is None:
            continue
        if xa is None or ya is None:
            out["audio_length_diffs"].append({"source_id": sid, "a": None if xa is None else len(xa),
                                              "b": None if ya is None else len(ya)})
            continue
        if len(xa) != len(ya):
            out["audio_length_diffs"].append({"source_id": sid, "a": len(xa), "b": len(ya)})
        n = min(len(xa), len(ya))
        d = float(np.max(np.abs(np.asarray(xa[:n], np.float64) - np.asarray(ya[:n], np.float64)))) if n else 0.0
        out["audio_max_abs_diff"] = max(out["audio_max_abs_diff"], d)
        if d > tol:
            out["audio_over_tol"].append({"source_id": sid, "max_abs_diff": d})
    if a.get("timeline_samples") != b.get("timeline_samples"):
        out["sample_count_diffs"].append({"source_id": "<timeline>", "a": a.get("timeline_samples"),
                                          "b": b.get("timeline_samples")})
    out["identical"] = not (out["missing_in_a"] or out["missing_in_b"] or out["timeline_diffs"]
                            or out["sample_count_diffs"] or out["audio_length_diffs"] or out["audio_over_tol"])
    return out


def to_json(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
