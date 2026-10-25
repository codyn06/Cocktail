"""Ground truth for "who is talking in this frame" (M4: box on the real talker in >= 80% of frames).

Label a take by hand (watch the frames, or the recording while listening) in OUT/talkers.csv:

    start_s,end_s,talker_id,bearing_deg,note
    0.0,4.5,alice,-30,left of me
    4.5,9.0,bob,40,
    7.0,9.0,alice,-30,overlap: both talk

- start_s / end_s: seconds since the first audio sample of the take (Session.start_ts_us), the same
  timeline as mockclient.replay. A frame at device_ts_us is at (device_ts_us - start_ts_us) / 1e6.
- talker_id: any stable name; one row per talking span, rows may overlap (several talkers at once).
- bearing_deg: optional, where the talker is in degrees from where the head faced at the start of the
  take (+ right, 180 behind), the same convention as record_session.py --truth. Leave empty if unknown.
- Silence is simply no row. Lines starting with # and blank lines are ignored.

`frame_truth` gives, per recorded frame, the talkers who are talking in it; a perception result
(a box, its talker id or bearing) is then scored against that list.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .session import Session

FILENAME = "talkers.csv"


@dataclass(frozen=True)
class TalkerSpan:
    start_s: float
    end_s: float
    talker_id: str
    bearing_deg: float | None = None
    note: str = ""


def load_talkers(take: str | Path) -> list[TalkerSpan]:
    """Read OUT/talkers.csv (or the CSV file itself); raises ValueError naming the bad line."""
    path = Path(take)
    if path.is_dir():
        path = path / FILENAME
    lines = [l for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip() and not l.lstrip().startswith("#")]
    spans = []
    for row in csv.DictReader(lines):
        try:
            span = TalkerSpan(float(row["start_s"]), float(row["end_s"]), row["talker_id"].strip(),
                              float(row["bearing_deg"]) if (row.get("bearing_deg") or "").strip() else None,
                              (row.get("note") or "").strip())
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}: bad row {row}: {e!r}") from e
        if not (math.isfinite(span.start_s) and math.isfinite(span.end_s)) or span.end_s <= span.start_s                 or not span.talker_id:
            raise ValueError(f"{path}: bad row {row}: need end_s > start_s and a talker_id")
        spans.append(span)
    return sorted(spans, key=lambda s: (s.start_s, s.end_s))


def active_at(spans: list[TalkerSpan], t_s: float) -> list[TalkerSpan]:
    """Spans covering t_s (start inclusive, end exclusive)."""
    return [s for s in spans if s.start_s <= t_s < s.end_s]


def frame_truth(session: Session, spans: list[TalkerSpan]) -> list[tuple[int, list[TalkerSpan]]]:
    """(device_ts_us, talkers talking at that frame) for every recorded frame, in time order."""
    return [(ts, active_at(spans, (ts - session.start_ts_us) / 1e6)) for ts, _ in session.frames]
