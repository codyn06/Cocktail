# test/locate-track-offline
Status: in progress (2026-10-25)
Owner: Sadie / Claude Code   PR: not opened

Workstream B "Offline position and identity checks": run locate + track on recordings and
synthetic cases, measure bearing error and identity switches where truth exists, test pauses,
crossing, nearby sources and missing/unreliable position evidence, compare tracker ids with
Phase 4 persistent source ids (`src_n`, docs/AV_SOUND_DISCOVERY_SPEC.md 12), propose mapping /
expiry / abstention rules for Phase 5, add regression cases, report evidence gaps.

## Ownership
- orchestrator: `hub/track/scenarios.py` (scripted probes), `hub/locate/pairsim.py` (moving /
  pausing talkers, pose gaps), `hub/track/offline.py` (harness), tests in `hub/tests/track`,
  `hub/tests/locate`.
- later, maybe delegated (integration-engineer, own worktree): `hub/track/compare_ids.py`.

## Decisions
- Scope (Sadie, 2026-10-25): tests and measurements; only one-line tracker fixes proven by a
  test. Merge / gate behaviour changes go in a separate PR (the Phase 5 rules depend on them).
- pairsim default path stays byte-identical (cached `sessions/scenes` and the old 12-scene
  baseline must stay reproducible); new features draw no rng on the default path.
- Identity is scored by stream provenance (oracle stems: which true source each stream carries
  per window, including injected slot swaps), not by a bearing gate. Ground-truth "active" comes
  from stem energy per 100 ms tick, not the script (talker() has its own 0.3-1.5 s gaps).
- Slot swaps are injected after LiveSeparator (it re-aligns voice slots by overlap cosine, so a
  swap inside the oracle separator would be undone).

## Evidence available (2026-10-25)
- Simulated only: all `hub/sessions/*` are pairsim. No real takes on this machine (Shayaan's
  recordings folder not shared yet). No Phase 4 outputs (PR #58, identity task T1 pending).

## Verification
- (none yet)

## Next steps
- Scripted probes -> pairsim extension -> offline harness -> suite baseline -> compare_ids ->
  PHASE5_IDS.md -> regression cases + evidence report.
