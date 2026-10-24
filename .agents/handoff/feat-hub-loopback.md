# feat/hub-loopback
Status: in progress, draft PR (2026-10-24)
Owner: Lalithaaaa / Claude Code   PR: see `gh pr list --head feat/hub-loopback`

## Ownership
- Hub loopback (P0 step 1, laptop side) -> orchestrator: `hub/`, `mock/`, `.gitignore`

## Decisions
- Hub is Python + FastAPI, as in the PRD's Architecture tab.
- Proposed audio wire format v1: 32-byte binary header + PCM (`hub/protocol.py`,
  `hub/README.md`). Not team-approved yet; the integration owner should confirm it
  before the phone app is written against it.
- Output is fixed at int16 stereo 48 kHz, 20 ms frames, as the PRD's `/audio/out` says.
- A new `/ingest/audio` connection replaces the old one (reconnect), and resets the
  pipeline and sequence tracking.
- 100 ms per-listener out queue, drop oldest: delay never builds up.

## Verification
- `.venv/Scripts/python -m pytest hub` -> 18 passed (mocked)
- `uvicorn hub.main:app` + `python -m mock.phone_sim --seconds 10` -> 500 chunks in,
  499 frames out, 0 gaps / drops; round trip on localhost p50 2.4 ms, p95 3.2 ms;
  hub_latency p50 1.3 ms (mocked, localhost, Windows 11)
- 44.1 kHz float32 input -> 149 of ~150 frames, p50 3.1 ms (mocked)
- Loopback WAV: 440 Hz peak, L == R, no step larger than the ideal sine slope, so no
  chunk-boundary clicks (mocked)

## Next steps
- Blocker: the team has an iPhone and no Mac. The phone app (Meta Wearables toolkit)
  needs either an Android phone, or a Mac / cloud Mac + Apple developer signing for iOS.
- Phone app: stream camera-stream audio into `/ingest/audio`, play `/audio/out` to the
  glasses; then measure the real round trip per stage (P0 step 1 acceptance).
- iPhone link has no `adb reverse`: use Wi-Fi (`--host 0.0.0.0`) or USB via iproxy.
- Then video / motion / input ingest and `/events` (P0 step 2).
