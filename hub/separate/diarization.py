"""Interval and anchor math for speaker activity (no models).

All intervals are half-open integer microsecond pairs `(start_us, end_us)` meaning
`start_us <= t < end_us`. The helpers are pure functions so anchors can be tested without
running a diarizer. A diarizer (Phase 3) only has to supply `activities`.

    positive_candidates(i) = A_i minus (union of other A_j and N)
    negative_candidates(i) = union of other A_j minus A_i

Fully overlapped speech legitimately yields NO clean positive anchor; that is returned as an
empty set with a reason, never padded to satisfy an expectation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Interval = tuple[int, int]


def normalize(intervals) -> list[Interval]:
    """Sort, drop empty/inverted intervals, and merge overlapping or touching ones."""
    out: list[Interval] = []
    for s, e in sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s)):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def union(*sets) -> list[Interval]:
    return normalize(iv for s in sets for iv in s)


def subtract(a, b) -> list[Interval]:
    """Parts of `a` not covered by `b`."""
    a, b = normalize(a), normalize(b)
    out: list[Interval] = []
    j = 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                out.append((cur, b[k][0]))
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


def intersect(a, b) -> list[Interval]:
    a, b = normalize(a), normalize(b)
    out: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def merge_gaps(intervals, max_gap_us: int) -> list[Interval]:
    """Join neighbours separated by at most `max_gap_us`."""
    out: list[Interval] = []
    for s, e in normalize(intervals):
        if out and s - out[-1][1] <= max_gap_us:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def trim(intervals, trim_us: int) -> list[Interval]:
    """Shrink every interval by `trim_us` on both sides (uncertain boundaries); drop what vanishes."""
    return [(s + trim_us, e - trim_us) for s, e in normalize(intervals) if e - s > 2 * trim_us]


def filter_min_duration(intervals, min_us: int) -> list[Interval]:
    return [(s, e) for s, e in normalize(intervals) if e - s >= min_us]


def total_us(intervals) -> int:
    return sum(e - s for s, e in normalize(intervals))


@dataclass
class AnchorConfig:
    merge_gap_us: int = 150_000      # bridge short pauses inside one activity
    trim_us: int = 50_000            # distrust boundaries (diarizer onsets/offsets are soft)
    min_duration_us: int = 500_000   # shortest usable anchor
    negative_min_duration_us: int = 200_000


@dataclass
class AnchorSet:
    speaker_id: str
    positive: list[Interval] = field(default_factory=list)
    negative: list[Interval] = field(default_factory=list)
    empty_reason: str | None = None  # set exactly when `positive` is empty
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"speaker_id": self.speaker_id, "positive": [list(i) for i in self.positive],
                "negative": [list(i) for i in self.negative], "empty_reason": self.empty_reason,
                "provenance": self.provenance}


def compute_anchors(activities: dict[str, list], noise=(), cfg: AnchorConfig | None = None) -> dict[str, AnchorSet]:
    """Positive/negative anchor intervals per speaker with provenance and valid empty outcomes."""
    cfg = cfg or AnchorConfig()
    noise = normalize(noise)
    acts = {k: normalize(v) for k, v in activities.items()}
    result: dict[str, AnchorSet] = {}
    for sid, own in acts.items():
        others = union(*(v for k, v in acts.items() if k != sid))
        raw_pos = subtract(subtract(own, others), noise)
        merged = merge_gaps(raw_pos, cfg.merge_gap_us)
        # Merging bridged a gap: re-subtract so the bridge never covers another talker or noise.
        merged = subtract(subtract(merged, others), noise)
        positive = filter_min_duration(trim(merged, cfg.trim_us), cfg.min_duration_us)

        raw_neg = subtract(others, own)
        negative = filter_min_duration(trim(merge_gaps(raw_neg, cfg.merge_gap_us), cfg.trim_us),
                                       cfg.negative_min_duration_us)
        # A negative anchor says "target absent"; it must never cover trusted positive or own speech.
        negative = subtract(negative, union(own, positive))

        reason = None
        if not positive:
            if not own:
                reason = "no_activity"
            elif not subtract(own, union(others, noise)):
                reason = "total_overlap"
            else:
                reason = "too_short"
        result[sid] = AnchorSet(sid, positive, negative, reason, {
            "own_us": total_us(own), "overlapped_by_other_speakers_us": total_us(intersect(own, others)),
            "overlapped_by_noise_us": total_us(intersect(own, noise)),
            "raw_positive_us": total_us(raw_pos), "kept_positive_us": total_us(positive),
            "raw_negative_us": total_us(raw_neg), "kept_negative_us": total_us(negative),
            "config": {"merge_gap_us": cfg.merge_gap_us, "trim_us": cfg.trim_us,
                       "min_duration_us": cfg.min_duration_us},
        })
    return result
