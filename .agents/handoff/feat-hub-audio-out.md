# feat/hub-audio-out
Status: in progress, draft PR (2026-10-25)
Owner: Cody / integration-engineer subagent   PR: see draft PR on this branch

## Ownership
- Route chosen voice to /audio/out (M2) -> integration-engineer: hub/out (new), hub/events/selection.py,
  hub/events/server.py, hub/ingest/server.py (input_hooks), hub/separate/__main__.py, hub/pyproject.toml,
  hub/tests/test_audio_out.py

## Decisions
- Select/stop/next/prev come from InputEvent on /ingest/pose, not /events commands: ingest got `input_hooks`.
- Choice is stored as a Tracker id (stable through slot swaps/pauses, and the id the headset sends).
  Voice memory is complementary (slot stability for two voices), not the key.
- next/prev order: world bearing ascending, bearingless last, lost skipped, wraps.
- Quest AudioOutPlayer applies inner/outer gain itself -> `layers` frames are unity gain; hub only
  applies dB for `?layout=stereo` (or `--out-gains-in-layers`).
- Room (outer) = all slots except the chosen one(s). Hop audio waits 0.3 s (settle) so the route is current.
- Uses private Tracker fields `_dup_of` (via getattr) and `tracks[id].stream`; hub/track is not ours.

## Verification
- `cd hub; uv run pytest -q` -> 102 passed, 1 skipped (mocked: scripted tones, no device/GPU)
- Mutation check: disabling hub gains makes 3 volume tests fail (red-green)
- Not tested: real headset playback, real separator output, latency on hardware

## Next steps
- Agree `?layout=` query param with the protocol owner (Lalitha); no protocol change was made.
- Real-hardware run with the Quest (check settle delay, who applies volumes).
- StateChanged / TargetUpdated events are not emitted (the /events fan-out is latest-only SourcesUpdated).
- Delay/gain printout (docs/STATUS.md M2) is partly in /audio/out/status; end-to-end delay not measured.
