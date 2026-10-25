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

## Confidence and tracker (branch feat/locate-confidence, 2026-10-25)
- Added `locate.scenes` (live two-talker scoreboard), `track.jitter` (jitter/lag metric), time-constant
  tracker smoothing, `hub/locate/FUSION.md` (design only, nothing built).
- PairLocator: `sys_sd_deg=16` blur before confidence is read (can only lower confidence; bearing still
  from the sharp peak). The simulation-fitted settings (neff_ratio 0.5, ambiguous_ratio 0.6,
  alt_exclude_deg 60) are NOT defaults: `pair.sim_tuned()` / `locate.scenes --sim-tuned`.
  Why: review (read-only) found the defaults were tuned because the SIM is underconfident while the
  real 3S was overconfident (79 deg miss "locked" at 0.80), and the 16 deg blur caps confidence near
  0.8, so it cannot stop a confidently wrong bearing.
- Held-out simulated scenes (12 two-talker, seed 500, 16 kHz calibration), same accuracy either way
  (median 10.5 deg, p90 27.0, front/back right 95.2%, both talkers within 25 deg in 8/12):
  default: ECE 0.437, says 0.43 / right 0.79, committed within 3 s 0%;
  --sim-tuned: ECE 0.160, says 0.69 / right 0.79, committed within 3 s 17%.
- "When it says 90% it is right 90%" cannot be shown yet: confidence never reaches 0.9 here.
- Tracker (simulated): tick-step p95 0.32 -> 0.20 deg per 0.1 s, 6 deg white-noise sd 2.84 -> 2.04,
  step 80% time 0.45 -> 0.85 s, ramp lag 0.31 -> 0.54 s. 3 of 48 step tests never reach 80% (not investigated).
- Open: a goodness-of-fit gate for `locked` and a per-device `sys_sd_deg` need real takes.
