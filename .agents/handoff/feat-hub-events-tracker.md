# feat/hub-events-tracker
Status: implemented, independently reviewed; draft PR (2026-10-24)
Owner: Codex orchestrator   PR: https://github.com/codyn06/Cocktail/pull/37

## Ownership
- Events implementation: integration-engineer, hub/events and two-line separate entry wiring; returned to orchestrator.
- Tests, packaging, fixes, documentation and integration: orchestrator.
- Independent review: reviewer, read-only, no remaining blocking findings.

## Decisions
- Frozen protocol, tracker, Quest files and GET /sources unchanged.
- New events package owns lifespan, periodic tracker updates, bounded per-client queues and stored command state.
- Deduplicate locator observations by slot and capture timestamp; tick the tracker through silence.
- Pose-clock monotonic extrapolation advances without input; reset tracks on epoch/large clock rollback without reusing IDs.
- Allow up to 5s observation capture age independently of hold_s; future snapshots cannot poison deduplication.
- Adapter snapshots LiveLocator internals under a bounded lock and evaluates sources off the event loop.
- Tracker last_observed_us remains delivery time; bearingless confirmation requires 15 distinct hops. See hub/events/README.md.
- Existing checkout preserved: worktree based on origin/main 1bf18d5.

## Verification
- Baseline uv run --group gpu pytest: 76 passed, 2 preexisting warnings.
- New events tests initially: 7 passed; generated real-time replay 9.27Hz, 75 valid events per client, stable t1/t2/t3.
- Capture-age defect reproduced before fix; clock-rollback regression failed before fix and passed after.
- Final uv run --group gpu pytest: 84 passed, 2 preexisting on_event deprecation warnings, 42.61s.
- Actual mockclient.replay CLI on local 12s synth_2talk_music: audio600/pose600; 123 valid events, 9.522Hz, 111 nonempty, stable t1/t2/t3. Scripted separation/bearings.
- uv build --wheel --out-dir sessions/build: passed; wheel contains events/server.py and events/__init__.py.
- graphify update .: passed, 2075 nodes. node .agents/scripts/sync.mjs --check: OK (90 files). git diff --check: clean.
- Evidence: mocked processing, simulated audio and replayed clips; nothing hardware-tested. No models downloaded or GPU inference run.

## Next steps
- Human review and merge. Keep PR draft as requested.
- Before merge, transfer durable notes to PR body and remove this handoff.
- Real headset and model-latency validation remains outside this task; no Tracker API mismatch found.
