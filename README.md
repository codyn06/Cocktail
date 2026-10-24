# Cocktail

An audio-first assistive experience built at a hackathon by a team of four. It uses
Ray-Ban Meta glasses, a phone, and possibly a Meta Quest headset as a display.

Planned building blocks:

- **Gemma**: open-weight language models, run directly. Variant and runtime are not chosen yet.
- **SAM3**: promptable visual segmentation.
- **SAM Audio** (possibly): prompted audio source separation.

> **Status:** early setup. No app stack, language, or build system has been chosen yet.
> The repo currently holds shared agent configuration and team conventions. See
> [`.agents/decisions.md`](.agents/decisions.md) for decisions made so far and open items.

## Repository layout

| Path | What |
|---|---|
| `AGENTS.md` | Shared instructions for every coding agent (canonical) |
| `CLAUDE.md` | Claude Code instructions (imports `AGENTS.md`) |
| `GEMINI.md` | Antigravity instructions (generated) |
| `.agents/` | Canonical roles, skills, sync script, decisions, and branch handoffs |
| `.claude/`, `.codex/`, `.pi/`, `.agent/` | Per-tool copies generated from `.agents/` |
| `graphify-out/` | Local code knowledge graph (gitignored, rebuilt per clone) |

## Setup

### Prerequisites

- [Git](https://git-scm.com/) and the [GitHub CLI](https://cli.github.com/) (`gh auth login`)
- [Node.js](https://nodejs.org/) 18 or newer (for the agent config sync script)
- [uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/) (to install graphify)
- At least one coding agent: Claude Code, Codex, Pi, or Antigravity

### 1. Clone

```sh
git clone https://github.com/codyn06/Cocktail.git
cd Cocktail
```

### 2. Install graphify (once per machine)

graphify builds a local knowledge graph of the repo that agents query before
searching code.

```sh
uv tool install graphifyy==0.9.95
# or: pipx install graphifyy==0.9.95
```

### 3. Set up graphify for this clone

```sh
graphify hook install   # git hooks rebuild the graph after commits, checkouts, merges
graphify update .       # build the first graph (local, no model calls)
```

Run `graphify update .` again after pulling. `hook install` also adds a line to
`.gitattributes`; don't commit it.

### 4. Check the agent configuration

```sh
node .agents/scripts/sync.mjs --check
```

This writes nothing and exits non-zero if the generated per-tool copies are out of date.

### 5. Set up your agent tool

- **Claude Code**: nothing extra. Project skills and subagents load from `.claude/`.
- **Codex**: trust the project so `.codex/agents/` loads.
- **Pi**: install Pi yourself, then use the subagent tool with `agentScope: "project"`.
- **Antigravity**: open the workspace; it reads `GEMINI.md` and `.agent/skills/`.

Details and enforcement limits: [`.agents/README.md`](.agents/README.md).

### Optional: Gemma documentation search

The `gemma-dev` skill can use the Google Developer Knowledge MCP server. Add it at
**user scope** with your own API key. Never put the key in repo config. Without it,
the skill falls back to https://gemma.ai/llms.txt.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

See [LICENSE](LICENSE).
