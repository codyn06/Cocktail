"""Human annotations for real long clips: JSON format, validator, and scoring against a run's timeline.

See LONG_ANNOTATIONS.md for how to annotate. Times are INTEGER MICROSECONDS on the clip's common clock,
half-open [start, end). They are converted to 16 kHz sample intervals once, here, at the metrics boundary
(`us_to_sample`, half-up integer arithmetic, same formula as av_scene.contracts); nothing is converted back.

    {
      "version": 1,
      "clip_id": "cafe_02",
      "duration_us": 120000000,
      "people": [
        {"person_id": "p1", "speaker_id": "s1", "track_ids": ["sam3_image:0"],
         "visible_us": [[0, 60000000]], "speaking_us": [[2000000, 9000000]],
         "silent_visible_us": [[9000000, 60000000]]},
        {"person_id": "p2", "speaker_id": null, "visible_us": [[0, 120000000]]}
      ],
      "off_screen_speakers": [{"speaker_id": "o1", "speaking_us": [[70000000, 76000000]]}],
      "refuse_us": []
    }

`silent_visible_us` is optional (default: visible minus speaking). `speaker_id: null` (or omitted with no
speaking_us) marks a silent bystander. `track_ids` are the pipeline's visual track ids, filled in after looking at a
run (or passed to `score_annotations(track_map=...)`); a person may have several tracks (re-entry).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent)) if str(Path(__file__).resolve().parent) not in sys.path else None
import long_metrics as LM  # noqa: E402

VERSION = 1
SR16 = 16_000


class AnnotationError(ValueError):
    pass


def us_to_sample(us: int, rate_hz: int = SR16) -> int:
    return (int(us) * rate_hz + 500_000) // 1_000_000


@dataclass
class Person:
    person_id: str
    speaker_id: str | None
    track_ids: list[str]
    visible_us: list[tuple[int, int]]
    speaking_us: list[tuple[int, int]]
    silent_visible_us: list[tuple[int, int]]


@dataclass
class Annotations:
    clip_id: str
    duration_us: int
    people: list[Person]
    off_screen: dict[str, list[tuple[int, int]]] = field(default_factory=dict)      # speaker_id -> speaking_us
    refuse_us: list[tuple[int, int]] = field(default_factory=list)
    version: int = VERSION

    def ground_truth(self, rate: int = SR16) -> dict:
        """Keyword inputs for `long_metrics.score_all` (intervals in samples at `rate`). Silent-visible time is
        keyed by PERSON, with `groups` mapping a person to all of their track ids."""
        c = lambda iv: LM.merge((us_to_sample(a, rate), us_to_sample(b, rate)) for a, b in iv)
        active, onscreen, p_spk, t_p, silent, groups, refuse = {}, {}, {}, {}, {}, {}, []
        for p in self.people:
            p_spk[p.person_id] = p.speaker_id
            if p.speaker_id is not None:
                active[p.speaker_id] = c(p.speaking_us)
                onscreen[p.speaker_id] = c(p.speaking_us)
            for t in p.track_ids:
                t_p[t] = p.person_id
            silent[p.person_id] = c(p.silent_visible_us)
            groups[p.person_id] = set(p.track_ids)
        for spk, iv in self.off_screen.items():
            active[spk] = c(iv)
            refuse += c(iv)
        refuse += c(self.refuse_us)
        return {"speaker_active": active, "onscreen": onscreen, "track_person": t_p, "person_speaker": p_spk,
                "should_refuse": LM.merge(refuse), "silent_visible": silent, "groups": groups,
                "duration_samples": us_to_sample(self.duration_us, rate), "rate": rate}


def _intervals(raw, where: str, duration: int | None) -> list[tuple[int, int]]:
    if not isinstance(raw, list):
        raise AnnotationError(f"{where}: expected a list of [start_us, end_us]")
    out, prev = [], None
    for k, pair in enumerate(raw):
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in pair)):
            raise AnnotationError(f"{where}[{k}]: need two integers (microseconds), got {pair!r}")
        a, b = pair
        if a < 0 or b <= a:
            raise AnnotationError(f"{where}[{k}]: need 0 <= start < end, got {pair!r}")
        if duration is not None and b > duration:
            raise AnnotationError(f"{where}[{k}]: end {b} beyond duration_us {duration}")
        if prev is not None and a < prev:
            raise AnnotationError(f"{where}[{k}]: intervals must be sorted and non-overlapping (start {a} < previous end {prev})")
        prev = b
        out.append((a, b))
    return out


def parse_annotations(doc: Mapping) -> Annotations:
    """Validate and normalise an annotation document (raises AnnotationError with the offending path)."""
    if doc.get("version") != VERSION:
        raise AnnotationError(f"version must be {VERSION}, got {doc.get('version')!r}")
    dur = doc.get("duration_us")
    if not isinstance(dur, int) or dur <= 0:
        raise AnnotationError("duration_us must be a positive integer")
    people, seen_p, seen_spk, seen_trk = [], set(), set(), {}
    for k, p in enumerate(doc.get("people", [])):
        w = f"people[{k}]"
        pid = p.get("person_id")
        if not pid or pid in seen_p:
            raise AnnotationError(f"{w}: person_id missing or duplicated")
        seen_p.add(pid)
        vis = _intervals(p.get("visible_us", []), f"{w}.visible_us", dur)
        talk = _intervals(p.get("speaking_us", []), f"{w}.speaking_us", dur)
        spk = p.get("speaker_id", pid if talk else None)
        if spk is None and talk:
            raise AnnotationError(f"{w}: speaking_us given but speaker_id is null")
        if spk is not None:
            if spk in seen_spk:
                raise AnnotationError(f"{w}: speaker_id {spk!r} used twice")
            seen_spk.add(spk)
        out = LM.subtract(talk, vis)
        if out:
            raise AnnotationError(f"{w}: speaking outside visible_us at {out[0]}; an unseen voice is an off_screen_speaker")
        if "silent_visible_us" in p:
            sil = _intervals(p["silent_visible_us"], f"{w}.silent_visible_us", dur)
            if LM.subtract(sil, vis):
                raise AnnotationError(f"{w}: silent_visible_us must lie inside visible_us")
            if LM.intersect(sil, talk):
                raise AnnotationError(f"{w}: silent_visible_us overlaps speaking_us")
        else:
            sil = LM.subtract(vis, talk)
        tracks = list(p.get("track_ids", []))
        for t in tracks:
            if t in seen_trk:
                raise AnnotationError(f"{w}: track {t!r} already belongs to {seen_trk[t]!r}")
            seen_trk[t] = pid
        people.append(Person(pid, spk, tracks, vis, talk, sil))
    off = {}
    for k, o in enumerate(doc.get("off_screen_speakers", [])):
        spk = o.get("speaker_id")
        if not spk or spk in seen_spk or spk in off:
            raise AnnotationError(f"off_screen_speakers[{k}]: speaker_id missing or already used")
        off[spk] = _intervals(o.get("speaking_us", []), f"off_screen_speakers[{k}].speaking_us", dur)
    refuse = _intervals(doc.get("refuse_us", []), "refuse_us", dur)
    return Annotations(str(doc.get("clip_id", "")), dur, people, off, refuse)


def load_annotations(path: str | Path) -> Annotations:
    return parse_annotations(json.loads(Path(path).read_text(encoding="utf-8")))


def score_annotations(timeline, annotations: Annotations | Mapping, *, rate: int = SR16,
                      track_map: Mapping[str, str] | None = None, guard_samples: int = 0, **gate_overrides) -> dict:
    """Score a run against human annotations.

    timeline: `long_metrics.LoadedTimeline` (speech sources at its own sample rate) or a list of Seg/dict/segment
    objects already in `rate` samples. track_map: pipeline track id -> annotated person_id, for tracks not yet listed
    in the annotation's `track_ids` (merged in; conflicts raise)."""
    ann = annotations if isinstance(annotations, Annotations) else parse_annotations(annotations)
    gt = ann.ground_truth(rate)
    for t, pid in (track_map or {}).items():
        if gt["track_person"].get(t, pid) != pid:
            raise AnnotationError(f"track {t!r} mapped to {pid!r} but annotated as {gt['track_person'][t]!r}")
        if pid not in gt["groups"]:
            raise AnnotationError(f"track_map names unknown person {pid!r}")
        gt["track_person"][t] = pid
        gt["groups"][pid].add(t)
    segs = timeline.segments_at(rate) if isinstance(timeline, LM.LoadedTimeline) else LM.as_segs(timeline)
    out = LM.score_all(segs, guard_samples=guard_samples, **gt, **gate_overrides)
    out["clip_id"], out["duration_s"] = ann.clip_id, ann.duration_us / 1e6
    out["unmapped_tracks"] = sorted({s.track_id for s in segs if s.track_id and s.track_id not in gt["track_person"]})
    return out
