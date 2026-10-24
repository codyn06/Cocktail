# Team decisions

Durable, team-wide decisions only: stack choices, interface contracts, conventions.
Change this file through a PR approved by the team integration owner. Live task
status and device reservations do not belong here (see `AGENTS.md` > Continuity).

## Integration owner
- Unassigned. Agree on one before parallel work on shared interfaces.

## Hardware reservation owners
One owner per shared device grants exclusive use (see `AGENTS.md` > Hackathon coordination).
- Ray-Ban Meta glasses: unassigned
- Phone: unassigned
- Quest headset: unassigned
- GPU machine: unassigned

## Decisions
- No app stack chosen; no package.json; no dependencies.
- Roles have no model names (inherit). Read-only roles may use bounded shell inspection;
  shell restrictions are instruction-enforced, including in Claude and Pi. Direct
  edit tools remain unavailable to those roles in Claude/Pi.
- Generated adapters are copies, not symlinks; the manifest + hash guard prevents
  overwriting hand edits.
- Project fallbacks cover unavailable metavr/TDD tooling without modifying vendored skills.
- Project-local dependency installs follow one policy; global installs and gated
  downloads still need approval.
- Task handoffs are per branch in `.agents/handoff/`; live task status is tracked in
  GitHub draft PRs. Hardware reservations require acknowledgment from the device's
  reservation owner in team chat or a GitHub issue, not a claim in a branch file.

## Open items
- Each teammate: confirm their tool loads the roles/skills (README "pending runtime
  checks" and smoke checks).
- Install Pi / metavr only with user approval if needed beyond the documentation fallback.
- Choose the app stack, then revisit deferred skills in `sources.json`.
