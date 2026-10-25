# feat/hub-events-tracker
Status: in progress (2026-10-24)
Owner: Codex orchestrator   PR: opening draft

## Ownership
- Events implementation: integration-engineer, hub/events and minimal separate entry wiring.
- Tests, packaging and integration: orchestrator after implementation handoff.
- Independent review: reviewer, read-only.

## Decisions
- Frozen protocol and quest files unchanged. New events package owns lifecycle and command state.
- Latest-only per-client delivery isolates slow clients; periodic tracker updates continue without hops.
- Preserve existing checkout changes using isolated worktree from origin/main 1bf18d5.

## Verification
- Initial graph query and graphify update completed; implementation checks pending.
- Evidence will be mocked/simulated/replayed; no hardware available.

## Next steps
- Implement, test pacing/identity/aging/commands/backpressure and replay, run full gpu-group suite, independent review.
