# test/locate-track-offline
Status: in progress (2026-10-25)
Owner: Sadie / Claude Code   PR: #59 (draft)

Workstream B "Offline position and identity checks": run locate + track on recordings and
synthetic cases, measure bearing error and identity switches where truth exists, test pauses,
crossing, nearby sources and missing/unreliable position evidence, compare tracker ids with
Phase 4 persistent source ids (`src_n`, docs/AV_SOUND_DISCOVERY_SPEC.md 12), propose mapping /
expiry / abstention rules for Phase 5, add regression cases, report evidence gaps.

## Ownership
- orchestrator: `hub/track/scenarios.py` (scripted probes), `hub/locate/pairsim.py` (moving /
  pausing talkers, pose gaps), `hub/track/offline.py` (harness), tests in `hub/tests/track`,
  `hub/tests/locate`.
- orchestrator (done inline, not delegated): `hub/track/compare_ids.py`, `hub/track/PHASE5_IDS.md`.

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

## Findings (simulated; details and tables in `hub/track/PHASE5_IDS.md`)
- 6 confirmed tracker failures, strict xfails in `tests/track/test_track.py` (near same-label
  talkers fold; direction-less births block each other; a returner near an active talker swaps
  ids with it via forced one-to-one assignment; a low-confidence second talker is never
  announced; alternating near talkers merge).
- Suite (16 cases x 2 seeds): 13 of 64 talker runs never announced (low-conf cause above);
  missed slot swaps reach the locator (per-slot posterior) and swap tracker ids; wrong bearings
  carry the same confidence as right ones (0.26-0.37 vs 0.27-0.50).
- Timing-only pairing with Phase 4 ids: never wrong, abstains on 36 of 51 announced runs.

## Verification
- `uv run pytest -q tests/track` -> 32 passed, 6 xfailed (scripted / fixtures).
- `uv run pytest -q tests/locate tests/track` -> 71 passed, 5 xfailed before compare_ids (simulated).
- pairsim default output byte-identical: scene00 regenerated, 5 files same sha256.
- `python -m track.offline sessions/offline --calib sessions/pairsim-calib16.json --seeds 2`
  (simulated, oracle separation). Nothing replayed or hardware-tested.

## Next steps
- Separate fix PR for the forced-assignment swap (needs a "start a new track" cost; changes gate
  and label behaviour) and the low-confidence-announcement rule, after the owner agrees.
- Phase 5 owner (Cody; Lalitha for wire changes) reviews PHASE5_IDS.md.
- When real takes / Phase 4 outputs exist: `locate.bench` on takes, `track.compare_ids` on the
  same recording, replace oracle rows in the evidence table.
