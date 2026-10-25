# feat/separate-measure-chosen
Status: ready for review (2026-10-25)
Owner: perception-engineer subagent for the orchestrator   PR: draft, see GitHub

## Ownership
- Measure script + unit tests -> perception-engineer: `hub/separate/measure_chosen.py`, `hub/tests/test_measure_chosen.py`
- Not touched: `hub/out`, `hub/events` (parallel branch feat/hub-audio-out)

## Decisions
- Measure on the separator's streams directly (not the /audio/out route): SIR-style, each 250 ms window of the chosen stream is least-squares projected on the ground-truth stems; margin = target component vs sum of the other stems' components. Sound matching no stem is reported separately (`unexplained_db`), not counted as "other voice".
- Chosen stream = oracle (best SI-SDR vs the target stem after t0), as in score.py. `carried_by` shows when the voice sits in another stream, so a bad margin can be attributed.
- Windows where the target stem is silent are skipped. Times are the end of the window, in signal time from t0; separator pipeline lag (a hop is final 1 s after it ends, plus compute) is printed, not added.
- Sessions without stems are refused.

## Verification
- `cd hub; uv run pytest -q` -> 97 passed incl. 11 new (mocked, synthetic known-ratio signals)
- `python -m separate.measure_chosen sessions/suite/scene01/session --t0 2,4` and `scene07` -> see PR table (replayed, simulated scenes, RTX 5070 Ti Laptop)

## Next steps
- 3-voice scene: 10 dB / 1 s not met; separator has 2 voice slots + residual. Needs a decision on how a third talker is held before this target can pass.
- When the hub routes the chosen stream to /audio/out, add an end-to-end check against that output.
- Delete this file in the PR before merge.
