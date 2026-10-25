# feat/locate-streams
Status: in progress (2026-10-25). Stacked on feat/locate-delay-ml (PR #47).
Owner: Sadie / Claude Code   PR: not opened

## Ownership
- orchestrator: `hub/locate/pairsim.py` (--scene), `score_live.py` (--oracle, --locate-method, --calib, --adapt),
  `pair.py` (PairConfig.adapt), tests.

## Findings (simulated; oracle separation = true stems, live masks / mic images / locator)
- Calibration rate matters: the channel offset reads ~5 us higher at 16 kHz than at 48 kHz in pairsim, which
  put a talker at 40 deg 49 deg off on the live path. Calibrations now carry rate_hz/win; the live hub warns
  on a calibration not fitted at 16 kHz / 1024 (fix is on feat/locate-delay-ml, commit 38ad4e3).
- 12 two-talker scenes (suite script: scratchpad scene_suite.py, not committed), calibration refit at 16 kHz:
  fixed model: median 10.0 deg, p90 23.7, front/back right 100%, both talkers within 25 deg in 11/12 scenes,
  mean confidence 0.48 vs 0.79 of estimates within 20 deg (underconfident, mostly "ambiguous").
  adapt (per-source swing and offset): median 8.5, p90 25.0, 10/12 scenes, mean conf 0.41.
  Not clearly better, so adapt stays opt-in (PairConfig(adapt=True), --adapt).
- Caveat: sim only; the sim room is not the real headset (see feat-locate-delay-ml.md).

## Verification
- `uv run pytest -q tests/locate` -> 40 passed (simulated).
- Nothing hardware-tested or replayed.

## Next steps
- Honest confidence: ambiguous/underconfident on 2-talker scenes (state "ambiguous" in most).
- Tracker smoothing (hub/track) on these scenes.
- Real takes: diagnose + bench baseline (waiting on Shayaan's recordings folder).
