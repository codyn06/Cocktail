# .agents — canonical agent configuration

Edit only the canonical sources here (plus root `AGENTS.md` / `CLAUDE.md`). Everything
else listed under "Generated" is produced by `node .agents/scripts/sync.mjs`
(requires Node >= 18, built-ins only, no symlinks).

## Layout

| Path | What |
|---|---|
| `../AGENTS.md` | shared project instructions (canonical) |
| `../CLAUDE.md` | `@AGENTS.md` + Claude-specific lines (canonical) |
| `roles/*.md` | canonical roles: frontmatter `name`, `description`, `access`, `skills`, then instructions |
| `skills/*/` | skills: vendored (see `sources.json`) plus project-authored `cocktail-perception` |
| `vendor/pi-subagent/` | vendored Pi subagent extension (unchanged upstream) |
| `sources.json` | provenance of vendored components; `projectSkills` lists project-authored skills |
| `scripts/sync.mjs` | generator / checker |
| `generated-manifest.json` | generated path -> sha256 (LF-normalized) of what sync wrote |
| `handoff.md` | current status and next steps |

Generated: `../GEMINI.md`, `../.claude/skills/`, `../.claude/agents/`, `../.agent/skills/`,
`../.codex/agents/`, `../.pi/agents/`, `../.pi/extensions/subagent/`.

## What each tool discovers

| Tool | Instructions | Skills | Roles |
|---|---|---|---|
| Claude Code | `CLAUDE.md` (imports `AGENTS.md`) | `.claude/skills/` | `.claude/agents/*.md` |
| Codex | `AGENTS.md` | `.agents/skills/` (native) | `.codex/agents/*.toml` |
| Pi | `AGENTS.md` | `.agents/skills/` (native) | `.pi/agents/*.md` via subagent extension |
| Antigravity | `GEMINI.md` (reportedly also `AGENTS.md`) | `.agent/skills/` | none native |

## Enforcement

| Tool | Read-only roles (`docs-scout`, `reviewer`) | Notes |
|---|---|---|
| Claude Code | direct edit tools denied; shell read-only behavior is instruction-enforced (runtime-untested) | `Bash` permits inspection; it is not a read-only security boundary |
| Codex | instruction-only | agent-file sandbox/approval settings do not override the parent, so none are set |
| Pi | direct edit tools omitted; shell read-only behavior is instruction-enforced (runtime-untested; Pi not installed) | read roles have `read, grep, find, ls, bash`; edit roles also have `edit, write` |
| Antigravity | none | roles are run as instructions in a dedicated task |

Skill preloading is native only in Claude (`skills:`). Codex/Pi role prompts name the
skill files to read.

Read-only roles may inspect via shell but may not mutate files, execute repository
code, or run artifact-producing checks. Orchestrators supply build/test output.
When external retrieval is unavailable, the scout requests specific official source
content from its orchestrator. Project fallbacks in `AGENTS.md` govern missing metavr
and TDD tooling; vendored skills remain unchanged.

## One-time teammate smoke check

In each tool the team actually uses, invoke the reviewer to read `AGENTS.md` and
report a file/line finding without edits. Then invoke the docs scout to load one
relevant skill and retrieve one official source (or explicitly request the source
from the orchestrator). Record tool/version, result, and limitations in the handoff.
`sync --check` verifies generated content, not runtime discovery or tool access.

## Setup and invocation

- **Claude Code**: nothing to install. Ask e.g. "Use the docs-scout subagent to verify
  the SAM3 video API." Run parallel editing subagents with worktree isolation.
- **Codex**: trust the project so `.codex/agents/` loads (untested here). Ask e.g.
  "Spawn the reviewer agent to review the current diff." No `.codex/config.toml` is created.
- **Pi**: install Pi yourself (not installed here). The extension is copied to
  `.pi/extensions/subagent/` (project path unverified). Project agents load only when
  the subagent tool is called with `agentScope: "project"` or `"both"`, e.g. "Use the
  subagent tool with agentScope project to run perception-engineer on <task>".
  `confirmProjectAgents` defaults to true: keep the trust prompt. Do not change `~/.pi`
  or global config silently.
- **Antigravity**: open the workspace; it reads `GEMINI.md` and `.agent/skills/`. To use a
  role, start a dedicated task and paste/point to `.agents/roles/<role>.md`.

## Sync workflow

1. Edit the canonical source (`AGENTS.md`, `roles/`, `skills/`, `vendor/`).
2. `node .agents/scripts/sync.mjs` to regenerate.
3. `node .agents/scripts/sync.mjs --check` (writes nothing; exit 1 on validation
   errors, missing/out-of-date outputs, or edited generated files).

Safety: sync refuses (exit 1, writes nothing) if a target exists but was not written by
sync, or a generated file was edited since generation. `--force` overrides. Stale
generated files are removed only if unmodified. Hashes ignore CRLF vs LF.

Updating a vendored skill: resolve the new upstream commit SHA, re-copy the folder
unchanged into `skills/<name>/`, update its `sources.json` entry (`commit`, `url`,
`localModifications`), then sync and `--check`.

## Troubleshooting

- "exists but not generated by sync": a hand-written file sits at a generated path. Move
  its content into the canonical source, delete it, re-run sync.
- "edited since generation": someone edited a copy. Port the edit to the source, then
  `sync --force` (or revert the copy).
- Validation errors: skill `name` must equal its folder; role skills must exist in `skills/`.
- Extra unknown files in generated folders are not detected (only manifest paths are tracked).

## Known limitations / pending runtime checks

- Pi is not installed; extension path and agent loading are unverified.
- Antigravity paths come from third-party sources; unverified.
- Codex project trust and custom-agent loading untested.
- metavr CLI/MCP is not installed; use the official-source fallback in `AGENTS.md`
  for documentation. Device-state claims still require device evidence.
- The TDD skill is not installed; use the repeatable failing/passing check fallback
  in `AGENTS.md` instead of installing the framework.
