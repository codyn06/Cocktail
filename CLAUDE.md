@AGENTS.md

## Claude Code specifics

- Skills in `.claude/skills/` and subagents in `.claude/agents/` are generated from
  `.agents/` by `node .agents/scripts/sync.mjs`. Edit the canonical source, then re-run sync.
- Delegate with the Agent tool using the project subagents (`docs-scout`,
  `perception-engineer`, `device-engineer`, `integration-engineer`, `reviewer`).
- Models: the main session (orchestrator) runs Sonnet at high effort with Opus as advisor
  (`.claude/settings.json`); generated subagents run Sonnet at low effort. Override per role
  with `model:`/`effort:` in `.agents/roles/<role>.md`, then re-run sync.
- Parallel editing subagents always use `isolation: "worktree"`; spawn independent tasks
  in one message. Long shell commands (>1 min) use `run_in_background` + `Monitor`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- graphify-out/ is gitignored here and rebuilt per clone by `graphify hook install`; it is never committed. (Re-running `graphify claude install` restores upstream's "committed graph.json" line; remove it again.)
