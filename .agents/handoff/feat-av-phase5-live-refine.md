# feat/av-phase5-live-refine
Status: in progress (2026-10-25)
Owner: Cody / Claude Code orchestrator   PR: draft, base `feat/av-phase4-long-clips` (Phase 4 PR #58 is unmerged)

## Ownership (new package `hub/refine/`; nothing in protocol/ingest/out/separate changes)
- contracts (shared types, admission rule) -> orchestrator: `hub/refine/contracts.py`, `hub/pyproject.toml` (package list)
- T2 bridge (job lifecycle, admission on the event loop, states) -> integration-engineer: `refine/bridge.py`, `tests/refine/test_bridge.py`
- T3 snapshots (frame ring tap, slot ring, job cut by device time) -> device-engineer: `refine/snapshot.py`, `tests/refine/test_snapshot.py`
- T4 speech refiner (S3FD box tracks, TalkNet, AV-TSE, match to hub slot audio) -> perception-engineer: `refine/faces.py`, `refine/speech_refiner.py`, tests
- T6a fixture converter (mp4 -> mockclient Session) -> perception-engineer: `refine/tools/fixture_to_session.py`
- Wave 2: T5 wiring + events aux channel + status route -> integration-engineer; T6b live replay/smoke harness -> perception-engineer.

## Decisions
- Live path = S3FD faces -> TalkNet -> AV-TSE, matched against the HUB slot audio (no second MossFormer2 run, no SAM 3:
  SAM 3 load was the 7.69 GiB transient in Phase 4). SAM 3 and real non-speech attribution stay out of live publication.
- Refinement never changes selection, route or /audio/out. It publishes a verdict (audio_applied is always False): refined
  stems arrive after their audio has played. Latency is measured in the smoke test, not assumed.
- Results are admitted on the event loop at publish time against (source_id, epoch, selection_version, audio_resets),
  slots, source presence and age; otherwise `stale`. Inputs are snapshotted by device time at job creation.
- Publication: `TargetUpdated` on /events (existing message) carries accepted (kind=person, assoc_conf>0) vs not
  (kind=speech, assoc_conf=0); `GET /refinement/status` carries all five states. Proposed (NOT applied, protocol is
  frozen): an optional `refinement_state` field on `TargetUpdated`, because pending/refused/unavailable/stale are not
  distinguishable on the wire today.

## Verification
- (none yet)

## Next steps
- Wave 1 (T2, T3, T4, T6a in parallel worktrees), then wave 2, review, live smoke (ask user: GPU free?).
