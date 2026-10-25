# Annotating a long clip for the Phase 4 gates

The silent-bystander gate (highlighted silent person-time under 2 %) and the active-speaker / refusal gates need a
human-labelled clip. Synthetic scenes (`long_fixtures.py`) prove the plumbing; only this annotation can say anything
about real footage. Evidence label for results scored this way: `replayed` (recorded data) until the whole chain runs
on the device.

## What you produce

One JSON file per clip, validated by `long_annotations.parse_annotations`:

```json
{
  "version": 1,
  "clip_id": "cafe_02",
  "duration_us": 120000000,
  "people": [
    {"person_id": "p1", "speaker_id": "s1", "track_ids": ["sam3_image:0"],
     "visible_us": [[0, 60000000], [75000000, 120000000]],
     "speaking_us": [[2000000, 9000000], [80000000, 90000000]],
     "silent_visible_us": [[9000000, 60000000], [90000000, 120000000]]},
    {"person_id": "p2", "speaker_id": null, "visible_us": [[0, 120000000]]}
  ],
  "off_screen_speakers": [{"speaker_id": "o1", "speaking_us": [[62000000, 70000000]]}],
  "refuse_us": [[100000000, 104000000]]
}
```

| Field | Meaning |
|---|---|
| `duration_us` | clip length in integer microseconds on the common clock (video pts start at 0 unless the manifest says otherwise) |
| `person_id` | your label for a human in the clip; unique |
| `speaker_id` | that human's VOICE label; `null` for someone who never speaks (a silent bystander). Defaults to `person_id` when `speaking_us` is non-empty |
| `track_ids` | visual track ids the pipeline gave this person (`inspect` a run, or pass `track_map` to the scorer). A person re-entering may have several |
| `visible_us` | when the person's face/head is on screen and unoccluded enough to be identified by eye |
| `speaking_us` | when this person is audibly talking, lips moving. Must lie inside `visible_us` |
| `silent_visible_us` | when the person is visible and clearly NOT talking. Optional; default = `visible_us` minus `speaking_us` |
| `off_screen_speakers` | voices you can hear that belong to nobody on screen (or to a person who is out of view), with their speaking spans |
| `refuse_us` | extra spans where the right answer is "do not attribute" even though a person is visible, e.g. two voices you cannot tell apart, or heavy cross-talk |

All times are integers in microseconds, half-open `[start, end)`, sorted and non-overlapping within a list. Do not
round-trip them through seconds in floating point.

## What counts as silent-visible

Silent-visible is the denominator of the bystander gate, so it must be conservative. Include a span only if a
reasonable listener would say "this person is not producing speech now". Mark it silent-visible when the person is
listening, reading, eating, looking away, or has a closed mouth. Do NOT mark it silent-visible when the person is
laughing, humming, whispering, mouthing words, or speaking so quietly you cannot judge. Leave those spans out of
`silent_visible_us` (they are neither speaking nor silent) by giving `silent_visible_us` explicitly. Brief pauses
inside someone's own utterance (under about 0.3 s) belong to `speaking_us`.

The scorer erodes nothing by default; pass `guard_samples` (for example 3200 for 200 ms) to keep boundary jitter out
of the rate.

## Tools

Any editor that shows the video with a millisecond clock works. Practical options:

* VLC or mpv with the on-screen timestamp, noting spans in a spreadsheet, then convert with
  `int(round(seconds * 1e6))`.
* ELAN or Audacity labels exported as TSV (start, end, label), then a short script to emit the JSON.
* Annotate against the same decoded video the pipeline used (`media_index` pts), not a re-encoded copy, or offsets drift.

Spot-check: speaking and silent-visible spans of one person must not overlap; the validator rejects speaking outside
visibility and unsorted lists.

## Scoring

```python
import long_annotations as A, long_metrics as M
ann = A.load_annotations("cafe_02.json")
tl = M.load_timeline("out/cafe_02_bundle")          # timeline/index.json + pages
report = A.score_annotations(tl, ann, track_map={"sam3_image:3": "p2"}, guard_samples=3200)
report["metrics"], report["gate"]["passed"], report["unmapped_tracks"]
```

`gate.provisional` is always true: only the 0.02 silent-bystander ceiling comes from the Phase 4 spec; the 0.80
floors and the 0.05 wrong-person ceiling in `long_metrics.GATES` are placeholders awaiting the orchestrator's
confirmation. Report both refusal precision and recall; refusing everything cannot pass because active-speaker recall and
refusal precision both collapse.
