# feat/locate-delay-ml
Status: in progress (2026-10-25). Stacked on feat/locate-bench.
Owner: Sadie / Claude Code   PR: https://github.com/codyn06/Cocktail/pull/47 (draft, base feat/locate-bench / #46)

Plan step 3 (delay method), done without real takes: none are on this machine
(`hub/recordings/` is gitignored; Cody and Lalitha have them).

## Ownership
- orchestrator: `hub/locate/pair.py` (phase-slope delays, weighted calibration),
  `hub/locate/real_eval.py` (phase calibrations, `--delay`, fast `method="frames"`),
  `hub/locate/bench.py` (`phase-loo`), `hub/locate/pairsim.py`, `hub/locate/diagnose.py`, tests.

## Findings (simulated unless marked)
- Real 3S calibration (`calib/quest3s.json`, 6 takes): delay swing D 14.6 us, residual 13.7 us.
- Per-frame delay noise is not the limit: on a known 10 us delay, 4096-sample frames, both
  GCC-PHAT and phase slope are within ~1 us even at 0 dB SNR (independent mic noise).
- pairsim rooms (reflections, coherent reverb/room noise, mic mismatch): reverb shrinks the fitted
  D (15 -> 4-9 us) and twists the axis, but residuals stay ~1.5-2.5 us even in a very
  reverberant room. Real residuals are 6-9x larger, so room acoustics alone don't explain them.
- Candidates, unranked until real takes are diagnosed: (1) timing: the recorder
  (quest/tools/record_session.py) places audio at `first_audio_ts_us + samples / rate` and
  ignores later chunks' device timestamps, so dropped chunks would misalign audio and pose in
  recordings (the live hub anchors every chunk, `hub/ingest/buffers.py` AudioRing, and the app
  stamps each chunk's first sample, MicStreamer.cs, so clock drift alone should be small);
  (2) headset processing on the "UNPROCESSED" mic path blurring sub-sample delays; (3) device
  constants that differ between takes or rooms. A constant lag in one take is mostly absorbed
  into the fitted axis, so diagnose reports both the best-shift scan and shared vs own fit.
- Phase-slope delays (opt-in, `real_eval calibrate --delay phase`; band 200 Hz-8 kHz, debiased
  coherence, lower-sigma half of frames so evidence matches GCC's sharper half): NOT better on sim
  (16 takes, leave-one-out: median 7.0 vs 8.5 deg for GCC, but p90 23 vs 18.5, locked within 3 s
  62% vs 88%, ECE 0.13 vs 0.07). Kept opt-in because their per-frame sigma may help on real takes
  with many poor frames; the bench compares both. Default stays GCC; live hub rejects phase calibs.
- Review (reviewer role, read-only) fixes applied: phase band cap and coherence debias; lock needs a
  committed state (tracking/locked) and agrees with the final estimate; leave-one-out skips INVALID
  takes and keys takes by path; summary flags in-sample `tdoa`; unobserved points out of ECE;
  diagnose skips unlabeled takes; LiveLocator rejects phase calibrations.

## What each method does badly (for targeting the next change)
- level: no usable cue on the 3S (mics ~5 mm apart); 59-180 deg off on real takes while
  "locked" at 90%+ (hardware-tested; ARCHITECTURE.md, "Real 3S results").
- tdoa (GCC): real LOO median 44 deg, 1/6 within 25; behind/weak-motion takes 52-79 deg;
  overconfident (a 79 deg miss "locked" at 0.80). Fixed noise_s and "sharper half" frame selection.
- tdoa near endfire (source along the pair axis, ~-20 or ~160 deg world for this axis):
  the delay barely changes with angle there, so the head has to turn the source out of endfire.
- phase: biased toward 0 delay by coherent zero-delay sound (reverb, room noise).

## Decisions
- Calibration JSON gains `delay_method` (default "gcc") and `model_s` (default null); old files load
  unchanged; live hub behaviour unchanged.
- `real_eval.run(method="frames")` is internal (calibration only).

## Verification
- `uv run pytest -q` (hub) -> 87 passed, 1 skipped (mocked/simulated).
- `python -m locate.diagnose` on pairsim takes with a 0.6 s pose lag -> finds +0.58 / +0.62 s.
- Nothing hardware-tested or replayed on this branch.

## Next steps
- Get the real takes into `hub/recordings/`, then run:
  `uv run python -m locate.diagnose recordings/*` and
  `uv run python -m locate.bench recordings --json bench-baseline.json`; record the baseline here.
- If diagnose shows a timing shift: ask the recorder/app owners to save per-chunk
  `device_ts_us` (quest/tools/record_session.py, outside hub/locate) and check how
  hub ingest timestamps audio for the live path.
- Per-headset calibration needs a headset id in Hello (protocol + Quest app; shared interface,
  coordinate with the integration owner). Today Hello only has `device: "quest3s"`.
