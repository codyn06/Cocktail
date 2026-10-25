# Direction tuning: why the live bearing missed the talkers, and what to switch on

Evidence labels: REAL = `take_4ch_voice3` (4 ch, 48 kHz, 40 s, two people talking, NO truth angle, mixture
audio, no separation). SIM = simulation. MOCK = synthetic delays. Reproduce REAL numbers with
`uv run python -m locate.audit <take_dir>` (offline, CPU).

## What the real take shows (REAL)

| Finding | Number |
|---|---|
| Channels carrying sound | ch0, ch1 only; ch2, ch3 exactly zero |
| ch0 broadband rms is mostly sub-100 Hz | rms -33 dBFS, but -47 dBFS after a 100 Hz high-pass (ch1: -53 / -54) |
| ch0 vs ch1 level gap in speech bands | ch0 is 15-32 dB below ch1 (300-600 Hz -15 dB, 0.6-1.2 kHz -22, 1.2-2.4 kHz -27, 2.4-4.8 kHz -32; loud frames only) |
| ch0 is a different kind of channel, not a quieter twin | its response relative to ch1 falls about 7 dB/octave from 300 Hz up; coherence with ch1 only 0.2-0.35 at 0.5-2 kHz, 0.01-0.1 above 4 kHz |
| ch0 near its noise floor in speech bands | across ch1 loudness quartiles ch0 rose 16 dB for 30 dB of ch1; above 2 kHz it does not follow the speaker at all (slope 0.0) |
| Level difference follows head yaw? | No. Correlation of ch0-ch1 per-band difference with yaw is between -0.33 and +0.11; slope about 0.01-0.06 dB per degree, while the difference itself wanders by 5-6 dB std |
| Inter-mic delay offset | +152 us median (IQR 138-164) in 300-3500 Hz GCC-PHAT, versus the calibrated offset of -0.7 us and a physical swing of only +-15 us |
| Delay follows head yaw? | Not on this mixture: sinusoid fit R2 0.01 (two talkers at unknown positions, so this is not proof the cue is dead) |

Hypotheses (not proven): ch0 is not a matched twin of ch1 (different mic or processing path in the
4-channel mode); the calibration (`calib/quest3s.json`) was fitted in the earlier 2-channel mode and
does not describe this mode.

## Why the live estimate missed (answers to Phase 1 a-d)

a. Gain imbalance (SIM, ear-like geometry, measured per-band gap injected, 6 sources, 4 seeds):
   equal-gain default median error 71.5 deg, 17% within 25 deg. With `profile_diff=True`: median 4.0 deg,
   100% within 25 deg. With the known gap removed via `mic_gain_db`: same as the clean baseline (4 deg).
   The equal-gain model reads a constant 20 dB difference as "the source is on one side of the head", so
   the rectangle sits at about head yaw plus or minus 90 deg and follows the head.
b. Channel meaning (REAL): ch0/ch1 are not a left/right pair of the same signal. The level gap is a
   spectral response difference, not a head-shadow ILD, and it does not change with yaw. Front/back or
   left/right cannot be decided from the level cue on this data.
c. Level method (REAL + SIM): with the live defaults the level estimator reports "locked", world bearing
   +83 deg, confidence 1.00 on the real mixture (reduced chi-square 5.1: the model does not fit). That is the
   confident wrong answer the wearer saw. The documented 30-45 deg male-voice bias (ARCHITECTURE D8) is a
   different effect (simulated head-shadow model mismatch) and does not explain a 20 dB channel gap.
   Even with the gap removed, nearly co-located mics (SIM, mics at +-5 deg) give median 42-75 deg error:
   the level cue has almost nothing to measure on a headset whose two mics are centimetres apart.
d. Dynamics (SIM, ear-like geometry, gap + profile_diff): static talker with +-40 deg head turns, `tau_s`
   8 s: median 5 deg, stays within 20 deg after 3 s (median). `tau_s` 3 s is unstable (p90 60 deg).
   Head still: no bearing information (median 72 deg), and the estimator says insufficient_motion.
   A talker walking at 8 deg/s defeats the static-bearing level estimator at every `tau_s` tried (median
   32-73 deg): it models a fixed bearing with forgetting, not a moving one. The delay path now has a
   forward filter for this (below, MOCK only).

## What changed (all opt-in; defaults unchanged)

| Option | Where | Effect |
|---|---|---|
| `LocateConfig.profile_diff=True` | estimator | removes any fixed per-band inter-mic gap online; no calibration needed |
| `LocateConfig.mic_gain_db` | estimator | subtract a known per-mic (M,) or per-mic-per-band (M,B) gain |
| `LocateConfig.max_resid_chi2` | estimator | withhold ("model_mismatch") when the head-shadow model does not fit |
| `LocateConfig.min_fit_gain` | estimator | withhold when the best bearing explains little more than "no direction cue" |
| `PairConfig.drift_dps` | pair | forward filter for a moving talker (use with `tau_s` about 60); not with `adapt` |
| `PairConfig.max_offset_dev_ratio` | pair | withhold ("calibration_mismatch") when the mean delay is far from the calibrated offset |
| `LiveLocator(mic_channels=...)` | live | choose which delivered channels feed the locator |
| `BearingEstimate.resid_chi2`, `.fit_gain` | estimator | the gate inputs, for logging |

Withheld states are not in `live._SHOW`, so the Source goes out with `bearing_deg=None, bearing_conf=0`.

Gate behaviour (SIM, 6 sources x 4 seeds, `max_resid_chi2=3.0`, `min_fit_gain=0.05`, `profile_diff=True`):
ear-like geometry shows 18 of 24 sources, all within 25 deg (median 4 deg); equal-gain without
profiling shows 0 of 24 (ungated median 71.5 deg); co-located mics show 1 of 24 (ungated median 75 deg).
The gate is a screen: in tests one co-located source in 12 still got through at 88 deg error.
REAL mixture: default config "locked" at 83 deg conf 1.0; recommended config "model_mismatch", withheld.

## Recommended live settings

1. Do not rely on the level method on the 3S. Either turn it off, or run it only with
   `LocateConfig(profile_diff=True, max_resid_chi2=3.0, min_fit_gain=0.05)` so it can only speak when it
   fits.
2. Use `--locate-method tdoa` only after the calibration is refit in the same channel mode as the live
   stream (4-channel). Until then set `PairConfig(max_offset_dev_ratio=2.0)` so a shifted offset withholds
   rather than misleads. In a SIM with a +150 us offset, the default tdoa path picks essentially arbitrary
   bearings and `adapt=True` becomes confidently wrong ("locked" 0.75, median error 90 deg).
3. For a talker who may walk: `PairConfig(tau_s=60.0, drift_dps=8.0)` (MOCK: median error 9 deg vs 41 deg
   for the static filter on a 6 deg/s walk; static talker 1 deg).

## Data needed from new labeled takes

See the report that accompanied this document; in short: per take a known truth angle, 1 m, head
turning +-40 deg for 20 s, 4-channel mode, plus a 2-channel-mode take at the same angles to compare the
calibration offset, for speech and pink noise at 0, 90, 180, 270 deg, and one head-still take per type.
