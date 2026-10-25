# feat/locate-bench
Status: in progress (2026-10-25)
Owner: Sadie / Claude Code   PR: not opened (GitHub CLI not installed on this machine yet)

Goal (Sadie's plan, milestone "location you can trust"): a scoreboard for direction finding
on real Quest 3S takes, then use it to improve the delay method and fusion. Folders:
`hub/locate`, `hub/track`.

## Ownership
- Scoreboard -> orchestrator: `hub/locate/bench.py`, `hub/tests/locate/test_bench.py`,
  small change in `hub/locate/real_eval.py` (`run(..., timeline_s=)`; result gains
  `go`, `stop`, `timeline`; default behaviour unchanged).

## Decisions
- The bench scores through `real_eval.run`, so its validity rules (head motion >= 8 deg std,
  source >= 6 dB over the room) apply; invalid takes are listed, not scored.
- Methods: `level`, `tdoa` (stored calibration; takes inside it are flagged `*in-calib`),
  `tdoa-loo` (calibration refitted without the scored take). Use `tdoa-loo` for baselines.
- Regions: side = within 15 deg of +-90 (not counted for front/back); front/behind otherwise.
- Seconds to lock: from the first estimate in the turn phase (real_eval blanks ~4 s of cue
  audio after the turn cue) until within 25 deg and staying there. Timeline every 0.25 s.
- Honest confidence: confidence = posterior mass within +-20 deg, so hit = error <= 20 deg;
  reliability bins of 0.2 and ECE over timeline points (correlated, so counts are optimistic).
- Not measured yet: "two speakers within 25 deg" (needs two-source takes).

## Verification
- `uv run pytest -q` (hub) -> 79 passed, 1 skipped (mocked: synthetic takes with the exact
  pair-delay model; checks plumbing and metric definitions, not real-world accuracy).
- `python -m locate.bench` on 4 synthetic takes -> tdoa/tdoa-loo 0 deg error, level
  fails as expected (identical channels carry no level cue). Untested on real takes.

## Next steps
- Get the real takes from Cody and Lalitha (the 6 calibration takes named in
  `hub/locate/calib/quest3s.json`) into `hub/recordings/`, run the bench, record the baseline.
- Install `gh`, `gh auth login`, push the branch, open a draft PR.
- reviewer pass before marking ready.
- Then plan step 3 (delay method): per-headset calibration files, improve until it beats baseline.
