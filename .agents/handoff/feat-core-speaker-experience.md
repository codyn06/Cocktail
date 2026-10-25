# feat/core-speaker-experience
Status: demo build in progress (2026-10-25). Hub side built; NOT verified end to end with two real people.
Owner: Claude Code orchestrator (Sadie's session)   PR: draft (see `gh pr list`)

## Goal
Two people visible and talking: both selectable (Speaker 1 / 2); selecting one draws a box on that person and
plays their isolated audio; switching moves both; stop clears the box. Box only from a voice-to-face match,
never a guess ("unavailable" instead).

## What exists (all on this branch)
- `hub/events/state_publish.py`: hub publishes `state.changed` + `target.updated` on /events (authoritative selection).
- `hub/speakers/`: per-speaker audio + timestamped intervals (`/speakers`, `/speakers/{id}/audio.wav`, `/intervals`);
  survives audio gaps, clears only on a pose-epoch change.
- `hub/vision/person_box.py` + `av_match.py`: S3FD faces (conf >= 0.5), IoU face tracks, CPU mouth-motion per face,
  voice-to-face association by contrast over single-speaker 100 ms bins (Hungarian, margin 1.5x, floor 0.08), sticky;
  overlay (`bbox`, camera bearing, `in_view`) on the chosen Source only while the association and a fresh detection exist.
  `/person-box/status`, `/person-box/debug`. All thresholds are untuned guesses.
- `hub/track`: `bearing_max_age_s` (8 s), opt-in `bearingless_speech_hits` + `bearingless_speech_per_slot` (hub uses 3 + True).
- `hub/separate`: separator mono = first two mic channels only; `--active-floor-db` (default -88; the band level reads ~29 dB
  under the stream rms, so the class default -60 hid all Quest speech); gated level locator by default (`--locate-ungated` to
  undo); `--no-audio-bearing`; `/ingest/levels`.
- Quest: hand dots on, picker box always drawn (empty when nobody), speech speakers stay listed, A-button HUD cycling only in
  DevMode. `quest/tools/run_demo.ps1` starts everything.

## Decisions (move to .agents/decisions.md at merge prep)
- Direction on the Quest 3S is unknown from audio (mics ~cm apart; the ungated level model "locks" on wrong bearings);
  a box position must come from the camera. Audio bearing is still shown by default (user's call), camera overrides it.
- Quest sends 4 channels, 2 are zero: use the first two (`-e micch 2` makes the app send two).
- Launch config of record: `cd hub && HF_HUB_OFFLINE=1 uv run --group gpu python -m separate --voice-memory`; app launched with
  `-e micch 2`; `adb reverse tcp:8765 tcp:8765`; hub_host.txt = 127.0.0.1.

## Verification (labelled)
- hardware-tested (single wearer + one other, hub side only): streams arrive (2 audio ch, pose, video, events); a speaker forms and
  its isolated stream is clean; selection routes the right slot to /audio/out; lag <= 0.02 s, 0 listener errors, ~0-1 underruns.
- mocked/synthetic only: the whole voice-to-face match, the person box overlay, the tracker options, speakers recorder.
- NOT done: two real people overlapping with a correct box on each; listening verdict on isolated audio; box alignment on a real
  frame (frame is upright; box/bearing offset unmeasured); the headset audible loudness. Tests were intentionally skipped at the
  user's request near the end; `hub/tests/vision/test_person_box.py` is STALE (old single-face API) and will fail.

## Known limits
- Second person must be within ~50 cm and speak up (-57..-65 dBFS at 1-2 m is near the noise floor); faces must face the headset.
- The match needs each person to speak ALONE for ~2-3 s first.
- First seconds of a new speaker's audio can be missing; one person can become two ids after a tracker merge.
- The Quest APK in `quest/Builds/` predates the A-button change: rebuild + reinstall (uninstall only if the signature differs).

## Next steps
- Live two-person run; read `/person-box/status` `reason_if_not`; tune thresholds; fix or delete the stale test; reviewer pass;
  then `git rm` this file in the PR before merge.
