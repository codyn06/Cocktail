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

## Hub (laptop)

The hub in `hub/` is a Python 3.12 server that receives audio, video and head pose from
the headset (or the mock client) and runs all processing. Run every command from `hub/`.

### Requirements

- [uv](https://docs.astral.sh/uv/). It downloads Python 3.12 for the project if you
  don't have it, so your system Python version doesn't matter.
- For GPU work only: an NVIDIA GPU and a driver that supports CUDA 12.8 or newer
  (check with `nvidia-smi`). RTX 50-series (Blackwell) cards need CUDA 12.8+ builds of PyTorch.

### Install and test

```sh
cd hub
uv sync             # creates hub/.venv with the hub dependencies
uv run pytest       # all tests should pass
```

### GPU environment (PR #13)

```sh
uv sync --group gpu   # adds PyTorch + torchaudio built for CUDA 12.8 (~3 GB download)
uv run --group gpu python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This should print `True` and your GPU name. Measured results and the VRAM budget are in
[`docs/STATUS.md`](docs/STATUS.md).

SAM 3 and SAM Audio weights are gated on Hugging Face. Each person who downloads them
needs their own approved account; don't share tokens.

1. Request access on the model pages: `facebook/sam3`, `facebook/sam3.1`,
   `facebook/sam-audio-small`, `facebook/sam-audio-small-tv` (optionally `-base-tv`).
2. Create a read token at huggingface.co → Settings → Access Tokens.
3. Log in: `uvx --from huggingface_hub hf auth login`. The token stays in your user
   cache, never in the repo.

### Run the hub with the mock headset

No headset needed. Terminal 1 starts the hub (port 8765):

```sh
uv run python -m ingest
```

Terminal 2 makes a simulated session and streams it in real time. Each `--stem` is a
WAV file and a direction in degrees (0 = ahead, + = right, 180 = behind):

```sh
uv run python -m mockclient.simulate --out sessions/demo \
    --stem voice1.wav:-50 --stem voice2.wav:40 --stem music.wav:150
uv run python -m mockclient.replay sessions/demo --fake-video-fps 15
```

Then open http://localhost:8765/ingest/status to see what the hub holds (seconds of
audio, poses, video frames received and dropped, missing messages). Sessions are
saved in `hub/sessions/`, which is gitignored. Never commit recordings.

Replayer options: `--speed 0` sends as fast as possible, `--loop` repeats (each pass
counts as a reconnect), and `--hub ws://<ip>:<port>` targets another machine.

### Reaching the hub from the Quest

- **Same Wi-Fi or hotspot**: find the laptop IP with `ipconfig` (Windows), then open
  `http://<laptop-ip>:8765/ingest/status` in the Quest browser. If it doesn't load,
  allow inbound TCP 8765 on private networks in Windows Defender Firewall.
- **USB**: run `adb reverse tcp:8765 tcp:8765`, then use `http://localhost:8765` on the Quest.

The hub listens on all interfaces with no authentication, so only run it on a private
hotspot or USB link.

### Optional: Gemma documentation search

The `gemma-dev` skill can use the Google Developer Knowledge MCP server. Add it at
**user scope** with your own API key. Never put the key in repo config. Without it,
the skill falls back to https://gemma.ai/llms.txt.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

See [LICENSE](LICENSE).
