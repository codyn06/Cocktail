# Handoff

## Status (2026-10-23): workflow fixes verified locally; PR pending GitHub authentication

- Branch: `fix/agent-workflow-blockers`; existing setup was untracked at task start.
- Current fix and handoff owner: this Codex orchestrator. No parallel writers.
- Team integration owner for implementation: unassigned; agree before parallel work.
- Shared device/GPU reservations: none for this configuration task.

## Ownership
- Canonical: `AGENTS.md`, `CLAUDE.md`, `.agents/roles/`, `.agents/skills/cocktail-perception/`,
  `.agents/scripts/sync.mjs`, `.agents/README.md`, this file.
- Do not edit by hand: vendored skills, `.agents/vendor/`.
- Generated (do not hand-edit): see `.agents/generated-manifest.json`.

## Decisions
- No app stack chosen; no package.json; no dependencies.
- Roles have no model names (inherit). Read-only roles may use bounded shell inspection;
  shell restrictions are instruction-enforced, including in Claude and Pi. Direct
  edit tools remain unavailable to those roles in Claude/Pi.
- Copies, not symlinks; manifest + hash guard against overwriting hand edits.
- Project fallbacks cover unavailable metavr/TDD tooling without modifying vendored skills.
- Related fixes reuse the feature PR; dirty/shared checkouts must not be switched underneath work.
- Project-local dependency installs follow one policy; global installs and gated downloads still need approval.

## Verification run
- `node .agents/scripts/sync.mjs` then `--check`: OK, 62 files.
- `node --check .agents/scripts/sync.mjs`: passed.
- `git diff --cached --check`: passed for the staged setup and fixes.
- Independent review: no blocking findings; reviewer successfully inspected files
  with `Get-Content` and `rg`. This used a default agent assigned reviewer duties,
  not automatic custom-role loading.
- Prior setup evidence (not rerun this task): edited generated file -> `--check` and generate both exit 1; user-authored
  file at generated path -> generate refuses; edited role source -> `--check` reports out of date.
- `python -I` tomllib parse of all `.codex/agents/*.toml`: OK.
- Not tested: actual loading in Claude Code, Codex, Pi, Antigravity.

## Next steps
- `gh auth status` reports no authenticated GitHub host. Authenticate before opening
  the required PR; do not merge without human approval.
- Each teammate: confirm their tool loads the roles/skills (see README "pending runtime checks").
- Run the one-time teammate smoke checks in `.agents/README.md` for tools actually used.
- Install Pi / metavr only with user approval if needed beyond the documentation fallback.
- Choose the app stack, then revisit deferred skills in `sources.json`.
