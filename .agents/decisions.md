# Team decisions

Durable, team-wide decisions only: stack choices, interface contracts, conventions.
Change this file through a PR approved by the team integration owner. Live task
status does not belong here (see `AGENTS.md` > Continuity).

## Integration owner
- Unassigned. Agree on one before parallel work on shared interfaces.

## Decisions
- No app stack chosen; no package.json; no dependencies.
- LLM: Gemma, run directly (not through the Gemini API). Use the vendored `gemma-dev` skill (google-gemma/gemma-skills); the
  `gemini-api-dev` skill was removed. Variant and runtime are undecided.
- Roles have no model names (inherit). Read-only roles may use bounded shell inspection;
  shell restrictions are instruction-enforced, including in Claude and Pi. Direct
  edit tools remain unavailable to those roles in Claude/Pi.
- Generated adapters are copies, not symlinks; the manifest + hash guard prevents
  overwriting hand edits.
- Project fallbacks cover unavailable metavr/TDD tooling without modifying vendored skills.
- Project-local dependency installs follow one policy; global installs and gated
  downloads still need approval.
- Task handoffs are per branch in `.agents/handoff/`; live task status is tracked in
  GitHub draft PRs.

- `/audio/out` takes an optional `?layout=layers|stereo` query parameter (default `layers`, the Quest). `layers` frames are unity gain because the Quest applies inner/outer dB itself; the hub applies `SetVolumes` only for `stereo` clients (or with `--out-gains-in-layers`). Approved by Cody (PR #42); no change to `hub/protocol`.

## Open items
- Each teammate: confirm their tool loads the roles/skills (README "pending runtime
  checks" and smoke checks).
- Install Pi / metavr only with user approval if needed beyond the documentation fallback.
- Choose the app stack, then revisit deferred skills in `sources.json`.
- Choose the Gemma variant and runtime (local server, self-hosted cloud, or on-device).
