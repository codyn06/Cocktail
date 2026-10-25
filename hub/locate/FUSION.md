# Fusing delay, level and head-motion history (design only, nothing here is implemented)

Status: a proposal for the next locate step. Everything measured below is from SIMULATED Quest 3S
data (`locate.pairsim`, `locate.scenes`) unless it says "real"; no number here is hardware-tested.

## What exists

| Cue | Where | Evidence it works |
|---|---|---|
| Inter-mic delay + head yaw -> bearing posterior on a 1 degree grid | `pair.py` (`PairLocator`) | simulated: median 8-10 deg, front/back right ~100%; real (6 takes, earlier measurement in `pair.py` docstring): 14-23 deg on clean front/side takes, 57-76 deg on behind/weak takes |
| Per-mic level (head shadow) | `estimator.py` (`BearingEstimator`) | FAILS on the real 3S: the mics are ~5 mm apart, so levels barely differ. 59-180 deg off at 90%+ confidence on real takes |
| Head-motion history | implicit in `PairLocator`: every delay frame is paired with the yaw at its time; evidence decays with `tau_s` = 20 s | front/back resolved by the turn in simulation |

## Proposed model: one log-posterior per voice

    log P(theta | everything) = sum over delay frames  w_d * log L_delay(tau_i | theta, yaw_i)
                              + sum over level frames  w_l * log L_level(lev_j | theta, yaw_j)
                              + log P_prior(theta | previous posterior)

* Delay term: as today (robust inlier/outlier mixture, calibrated D, axis, offset, noise).
* Level term: the head-shadow model, but with its **own measured model error** and a weight
  `w_l` in [0, 1] fitted on labeled real takes. If the fitted weight is ~0 the cue is dropped; it
  never gets to set confidence by itself. The hope (untested) is that head shadow breaks the front/back
  mirror of a single mic pair earlier than head motion does; that is the only reason to keep it.
* History term: replace "exponential forgetting" by a forward filter, `P_t(theta) proportional to
  (P_{t-dt} convolved with a random-walk kernel of sd s*sqrt(dt)) x likelihood`. It keeps the
  evidence for a standing talker (s small) and lets a walking talker's posterior move (s ~ 10-20
  deg/s) instead of being dragged by 20 s of old evidence. s could come from the tracker's
  motion estimate for the track.
* Head-motion weighting: a delay frame carries bearing information only through the sinusoid
  `tau = D sin(theta - axis - yaw)`; frames at the same yaw repeat the same constraint. Weight
  frames by new yaw span covered since the last used frame (replaces the flat `neff_ratio`).
* Confidence: unchanged definition (mass within +-20 deg of the peak after the systematic-error
  blur `sys_sd_deg`), so every term above is scored by the same reliability table.
* Per voice: each separated stream keeps its own posterior; the tracker's stable id decides which
  posterior continues after a stream-slot swap (a posterior handoff keyed by track id).

## Why not build it now

1. The level cue is the only new information, and it is known to fail on the real device. Adding
   it unmeasured would add a way to be confidently wrong.
2. The forward filter only matters for moving talkers; no real moving-talker data exists.
3. The simulated sets cannot tell whether the weights transfer to the headset.

## What measurement would justify it

All on real Quest 3S takes (`quest/tools/record_session.py`), scored with `locate.bench`
(single source) and `locate.scenes`-style two-talker takes recorded with a loudspeaker pair:

1. **Level cue worth keeping**: at least 20 labeled takes covering front, side and behind. Fit
   `w_l` leave-one-take-out. Accept only if the fused leave-one-out median error, front/back
   right rate and ECE are all no worse than delay-only, and the gain in held-out log-likelihood is
   positive in a majority of takes. Otherwise delete the level cue from the plan.
2. **Systematic error floor**: refit `sys_sd_deg` (now 16 deg, learned on simulation) and the
   reliability table on the same real takes. The milestone "when it says 90%, it is right about
   90%" needs real hit rates of >= 90% within +-20 deg for estimates that report 0.9; in
   simulation the ceiling is ~0.8 because the median error is ~10 deg.
3. **Moving talker**: at least 5 takes with a person walking at known angles (floor markers and
   a timestamped sync) to fit the random-walk sd and to check lag against the tracker's.
4. **Two talkers**: at least 10 two-speaker takes. Compare per-talker error against
   `locate.scenes` with the same separator masks, so separation error is accounted for.
