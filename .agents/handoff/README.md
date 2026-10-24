# Per-task handoffs

One file per task branch: `.agents/handoff/<branch-name-with-slashes-as-dashes>.md`
(e.g. branch `feat/gemma-live-audio` -> `feat-gemma-live-audio.md`). Only that
branch's orchestrator edits it, so parallel branches never conflict.

- Create it on the feature branch when a substantive task starts; commit it with the work
  so any tool can resume from the branch.
- Keep it through draft PRs, active development, and review; opening a PR is not a
  reason to delete it.
- Before merging: move durable decisions into `.agents/decisions.md` (or the PR body)
  and delete the file in the same PR. `main` should hold only this README.
- No secrets, credentials, or conversation transcripts.

## Template

```markdown
# <branch>
Status: <in progress | blocked | ready for review> (<date>)
Owner: <person / orchestrating tool>   PR: <link or "not opened">

## Ownership
- <task> -> <role/agent>: <files>

## Decisions
- <decision and reason>

## Verification
- <command> -> <result> (hardware-tested / mocked / replayed / untested)

## Next steps
- <step>
```
