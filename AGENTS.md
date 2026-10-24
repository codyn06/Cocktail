# Cocktail — Shared Agent Instructions

Canonical instructions for every coding agent (Claude Code, Codex, Pi, Antigravity).
`GEMINI.md` and the tool-specific role/skill copies are generated from `.agents/`
by `node .agents/scripts/sync.mjs`. Do not edit generated copies.

## Project context

Cocktail is a 4-person hackathon project: an audio-first assistive experience using
Ray-Ban Meta glasses, a phone, and possibly a Quest headset as a display. Expected
building blocks: the Gemini API, SAM3 (promptable visual segmentation), and possibly
SAM Audio (prompted audio source separation).

Do not assume:
- an app stack, language, framework, or build system (none is chosen yet);
- that Unity, a Meta SDK integration, or any device bridge exists;
- that model checkpoints are downloaded, accessible, or runnable on available hardware;
- that a device capability (camera stream, mic access, display output) is available
  until it is verified against official docs or on the device.

## Workflow (orchestrator)

1. **Inspect** the repo and relevant docs before planning.
2. **Plan**: outcome and acceptance criteria; tasks with dependencies and order; the
   role assigned to each; file ownership per task; shared decisions (interfaces,
   formats, config); how each task will be verified.
3. **Delegate**:
   - At most 2 active delegated tasks at a time.
   - One editing agent per working tree. Parallel writers need separate worktrees;
     otherwise serialize.
   - Every assignment states: goal and context, allowed files, skills to load,
     dependencies, acceptance/verification, and what to return.
   - Out-of-scope needs are reported back to the orchestrator, not acted on.
4. **Integrate and review**: the orchestrator inspects and integrates results itself,
   then gets an independent review proportional to risk (the `reviewer` role for
   non-trivial or shared-interface changes).
5. **Continuity**: keep `.agents/handoff.md` current (status, ownership, decisions,
   verification, next steps) so another agent or teammate can resume.

Trivial tasks: write a brief plan and do them directly.

## Hackathon coordination and fallbacks

- Each feature has one orchestrator. Before parallel work, agree on a team integration
  owner for shared interfaces and `.agents/handoff.md`; only that owner updates the
  shared handoff. Other agents return handoff notes. Record active ownership and
  exclusive headset/GPU reservations there; separate worktrees do not isolate hardware.
- Read-only roles may run bounded inspection commands (`rg`, file reads, `git status`,
  `git diff`, `git show`) and retrieve public official documentation without saving it.
  They must not mutate files, install software, execute repository code, or run tests
  or builds that may write artifacts. The orchestrator runs those checks and supplies
  the output. This evidence handoff satisfies reviewer verification requirements.
- These project fallbacks take precedence over conflicting vendored skill instructions:
  when metavr is unavailable, verify documentation claims using current official Meta
  documentation or upstream source, recording URL, version/date, and uncertainty. Do
  not install tooling just to answer a docs question. Device-state claims still need
  device evidence; mark them unverified if unavailable. If a role has no web access,
  ask the orchestrator to retrieve the specific sources, then continue independent work.
- The missing `superpowers:test-driven-development` skill is not a blocker: reproduce
  the failure with the smallest test or repeatable check, confirm it fails before the
  fix and passes afterward. Use the local `verification-before-completion` skill.
- Editing agents may install task-required project-local dependencies in an isolated
  environment after coordinating dependency-file changes with the orchestrator.
  Global installs, gated checkpoint access/downloads, and sandbox escalation still
  require the existing approvals. Never install into a shared/global environment by default.

## Shared rules

- Coordinate changes to shared interfaces, root config, and dependency files with the
  orchestrator before making them.
- Load specialist skills only when relevant to the task.
- Preserve scope: no unrequested features, scaffolding, or stack choices.
- Report checks honestly: only claim what you actually ran, with results.
- Label evidence: hardware-tested, mocked, replayed (recorded data), or untested.
- Subagents return: findings/changes, checks run, uncertainty, handoff notes.
- The orchestrator inspects and integrates subagent output; it does not rubber-stamp.
- Apart from the branch/PR flow in **Git workflow**, this configuration grants no
  permission to deploy, publish, message people, install
  software globally, download gated checkpoints, or bypass sandbox/approval controls.
  Ask the user first.

## Git workflow

Every independent feature, fix, or config change goes through its own branch and a
pull request. Follow-up fixes and small adjustments for the same feature stay on its
current branch and PR; do not restart the branch workflow for each agent subtask.

- **Never commit directly to `main`.** Start each task from an up-to-date base:
  `git switch main && git pull`, then `git switch -c <type>/<short-kebab-desc>`.
  First inspect status and the current branch. Do not switch branches underneath
  existing work or stash/discard someone else's changes. For concurrent work, fetch
  the target and create a separate worktree/branch from that updated target instead
  of switching the shared checkout. Reuse the assigned branch for an ongoing task.
  Branch off another feature branch only when the work depends on it, and target the
  PR at that branch.
- **Branch types:** `feat/`, `fix/`, `docs/`, `chore/`, `refactor/`, `test/`,
  `perf/` (e.g. `feat/gemini-live-audio`, `fix/sam3-mask-offset`).
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/), e.g.
  `feat(perception): add SAM3 image prompt wrapper`. Make small focused commits, and
  never commit secrets, `.env` files, checkpoints, or large binaries.
- **Scope:** one feature per branch and PR. Unrelated fixes get their own branch.
- **Before opening a PR:** rebase on or merge the latest target branch, run the
  relevant checks (and `node .agents/scripts/sync.mjs --check` if `.agents/` or
  `AGENTS.md` changed), and get a `reviewer` pass for non-trivial changes.
- **Open the PR** with `gh pr create --base <target>`, using a Conventional Commit
  title. The body covers: summary, linked issue (if any), checks actually run with
  results, evidence labels (hardware-tested / mocked / replayed / untested), and
  known limitations.
- **Agents may** create branches, commit, push their feature branch, and open PRs.
- **Agents must not** merge PRs, push to or force-push `main`, force-push a branch
  someone else is working on, or delete remote branches without the user's explicit
  approval. A human approves and merges.
- Parallel agents work in separate worktrees, each on its own branch.

## Skill index (`.agents/skills/`)

| Skill | Load when |
|---|---|
| `cocktail-perception` | SAM3, SAM Audio, audio/visual timing, source association, perception pipelines (project-authored) |
| `gemini-api-dev` | writing or debugging code that calls the Gemini API |
| `hz-quest-verify-first` | any Meta Quest / Horizon OS claim or code (needs metavr CLI/MCP; ask before `npx`) |
| `systematic-debugging` | any bug, test failure, or unexpected behavior, before proposing fixes |
| `verification-before-completion` | before claiming work is complete, fixed, or passing |

Tooling gaps: the referenced TDD skill and metavr CLI/MCP are not installed.
Use the project fallbacks above; do not repeatedly stop to request their installation.

## Role index (`.agents/roles/`)

| Role | Access | Use for |
|---|---|---|
| `docs-scout` | read-only | verify SDK APIs, hardware capabilities, model requirements, official examples |
| `perception-engineer` | edit | bounded audio/CV tasks (SAM3, SAM Audio, timing, source association) |
| `device-engineer` | edit | bounded phone / glasses / Quest integration tasks |
| `integration-engineer` | edit | bounded Gemini integration and communication/interface tasks |
| `reviewer` | read-only | independent review of correctness, concurrency, latency, secrets, regressions, evidence |

See `.agents/README.md` for per-tool setup, invocation, and enforcement limits.
