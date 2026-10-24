# Contributing to Cocktail

Thanks for helping out. This is a short guide; the full rules live in
[`AGENTS.md`](AGENTS.md), and they apply to people and coding agents alike.

## Before you start

1. Finish [setup](README.md#setup).
2. Check what others are working on: `gh pr list`.
3. Read [`.agents/decisions.md`](.agents/decisions.md). Don't pick a stack, framework,
   or model runtime on your own; propose it through a PR.

## Workflow

1. **Branch from an up-to-date `main`.** Never commit to `main` directly.

   ```sh
   git switch main && git pull
   git switch -c <type>/<short-kebab-desc>
   ```

   Types: `feat/`, `fix/`, `docs/`, `chore/`, `refactor/`, `test/`, `perf/`
   (e.g. `feat/gemma-live-audio`, `fix/sam3-mask-offset`).
   Use a separate git worktree if someone else is using the same checkout.

2. **Open a draft PR early** so the team can see your work:

   ```sh
   gh pr create --draft --base main
   ```

3. **Commit in small steps** using [Conventional Commits](https://www.conventionalcommits.org/):

   ```
   feat(perception): add SAM3 image prompt wrapper
   ```

   Never commit secrets, `.env` files, model checkpoints, or large binaries.

4. **Keep a handoff file** for substantive work in `.agents/handoff/` (format in
   [`.agents/handoff/README.md`](.agents/handoff/README.md)) so a teammate or agent can
   pick it up. Delete it in the PR before merge.

5. **Before marking the PR ready for review:**
   - rebase on or merge the latest `main`;
   - run the relevant checks;
   - if you changed `.agents/` or `AGENTS.md`, run `node .agents/scripts/sync.mjs --check`;
   - get a `reviewer` agent pass (or a teammate review) for non-trivial changes.

6. **PR description** should cover: summary, linked issue, checks run with results,
   evidence labels, and known limitations.

7. **A human approves and merges.** Don't force-push `main` or someone else's branch.

## Evidence labels

Say how each claim was verified:

- **hardware-tested**: ran on the real glasses, phone, or headset
- **mocked**: ran against stubs or fake inputs
- **replayed**: ran on recorded data
- **untested**

## Shared devices and machines

Glasses, phones, headsets, and GPU machines are shared. Ask before installing,
uninstalling, rebooting, clearing data, changing settings, or starting a long GPU job.

## Changing agent configuration

Edit only the canonical sources: `AGENTS.md`, `CLAUDE.md`, and files under `.agents/`.
Then regenerate and check the per-tool copies:

```sh
node .agents/scripts/sync.mjs
node .agents/scripts/sync.mjs --check
```

Don't edit generated files (`GEMINI.md`, `.claude/skills/`, `.claude/agents/`,
`.agent/skills/`, `.codex/agents/`, `.pi/`); sync will flag the edits.

## Shared interfaces and decisions

Changes to shared interfaces, root config, or dependency files need agreement from the
team integration owner. Durable team-wide decisions go in `.agents/decisions.md` via PR.
