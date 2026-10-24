@AGENTS.md

## Claude Code specifics

- Skills in `.claude/skills/` and subagents in `.claude/agents/` are generated from
  `.agents/` by `node .agents/scripts/sync.mjs`. Edit the canonical source, then re-run sync.
- Delegate with the Agent tool using the project subagents (`docs-scout`,
  `perception-engineer`, `device-engineer`, `integration-engineer`, `reviewer`).
- Parallel editing subagents must use worktree isolation; otherwise run them one at a time.
