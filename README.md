# Cocktail

An audio-first assistive experience built at a hackathon by a team of four. A Meta Quest 3S
headset streams its microphones, camera and head pose to a laptop "hub", which splits the
room's sound into separate labeled sources, works out where each one is, and sends the
chosen sound back to the headset in 3D. Plan: [`docs/QUEST_PRD.txt`](docs/QUEST_PRD.txt). Architecture and decisions:
[`ARCHITECTURE.md`](ARCHITECTURE.md).
Live progress: [`docs/STATUS.md`](docs/STATUS.md).

> **Status (2026-10-24):** the hub runs end to end on simulated data: ingest, a mock
> headset, live separation with labels, and direction finding (`GET /sources`). The Quest
> app streams mic audio (2 raw channels, 48 kHz), head pose and camera to the hub
> (hardware-tested) and has a headset UI. Not yet: the tracker (`hub/track`) in the live
> loop, sources pushed to the headset, sound sent back, and a passing real-world test.

## Repository layout

| Path | What |
|---|---|
| `AGENTS.md` | Shared instructions for every coding agent (canonical) |
| `CLAUDE.md` | Claude Code instructions (imports `AGENTS.md`) |
| `GEMINI.md` | Antigravity instructions (generated) |
| `.agents/` | Canonical roles, skills, sync script, decisions, and branch handoffs |
| `.claude/`, `.codex/`, `.pi/`, `.agent/` | Per-tool copies generated from `.agents/` |
| `graphify-out/` | Local code knowledge graph (gitignored, rebuilt per clone) |
| `hub/` | Laptop hub (Python 3.12): `protocol/` wire format, `ingest/` server, `mockclient/` fake headset, `separate/` live separation, `locate/` direction finding, `track/` stable source ids |
| `quest/` | Unity project for the Quest headset ([setup](quest/README.md)); `quest/tools/` has the mock hub, headset validation and the real-world recorder |
| `eval/` | Streaming separation benchmark ([README](eval/README.md)) |
| `ARCHITECTURE.md` | System overview and architectural decisions, with their status and evidence |
| `docs/` | Plan (`QUEST_PRD.txt`) and status board (`STATUS.md`) |

## Requirements at a glance

What you need depends on what you work on. All setup steps are below or linked.

| For | Hardware | Software | Accounts | Disk |
|---|---|---|---|---|
| Everyone (repo, agents) | any | Git, GitHub CLI, Node.js 18+, uv, a coding agent | GitHub | small |
| Hub on recorded/simulated data | any laptop, Windows/macOS/Linux | uv (installs Python 3.12 itself) | none | ~200 MB |
| Live separation and labels | NVIDIA GPU, ~2 GB free VRAM; driver with CUDA 12.8+ (RTX 50-series needs it) | the hub's `gpu` dependency group | none (models are public) | ~5 GB (PyTorch ~3 GB, models ~1 GB) |
| Separation benchmark (`eval/`) | NVIDIA GPU | `gpu` + `bench` dependency groups | none | several GB + LibriSpeech dev-clean (~330 MB) |
| SAM 3 / SAM Audio (A3, C5, D3; experimental) | NVIDIA GPU; 12 GB is tight (see STATUS budget) | separate environment, not in the hub yet | Hugging Face account approved for the `facebook/sam3*` and `facebook/sam-audio-*` repos | ~35 GB in the Hugging Face cache (SAM Audio alone pulls ~28 GB with its companion models) |
| Quest app | Quest 3S (camera, main) or Quest 2 (display only); USB-C data cable | Unity 6000.0.80f1 with Android Build Support, Meta XR packages | Meta developer organization, Developer Mode on the headset | Unity install + project |
| Headset to hub | laptop and headset on the same private network, or USB | Windows Firewall allows TCP 8765, or `adb reverse` | none | none |

Tested so far on Windows 11 with an RTX 5070 Ti Laptop GPU (12 GB). Use Git Bash or
PowerShell. In Git Bash, give Windows Python programs paths like `C:/...` (not `/c/...`).

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
- Port 8765 free on the laptop (the hub's default; change it with `--port`).

### Install and test

```sh
cd hub
uv sync             # creates hub/.venv with the hub dependencies
uv run pytest       # all tests should pass
```

### GPU environment

```sh
uv sync --group gpu   # adds PyTorch + torchaudio built for CUDA 12.8 (~3 GB download)
uv run --group gpu python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This should print `True` and your GPU name. Measured results and the VRAM budget are in
[`docs/STATUS.md`](docs/STATUS.md).

SAM 3 and SAM Audio weights are gated on Hugging Face. Each person who downloads them
needs their own approved account; don't share tokens. They aren't wired into the hub yet;
the measurements are tracked under A3 in STATUS.

1. Request access on the model pages: `facebook/sam3`, `facebook/sam3.1`,
   `facebook/sam-audio-small`, `facebook/sam-audio-small-tv` (optionally `-base-tv`).
2. Create a read token at huggingface.co → Settings → Access Tokens.
3. Log in: `uvx --from huggingface_hub hf auth login`. The token stays in your user
   cache, never in the repo.

### Run the hub with the mock headset

No headset needed. First make test audio and a simulated session (once). `demo_stems`
downloads two openly licensed LibriSpeech readers and a Brahms recording and writes
12 s clips. Any mono WAV files work instead. Each `--stem` is a WAV file and a direction
in degrees (0 = ahead, + = right, 180 = behind):

```sh
uv run --with librosa python -m mockclient.demo_stems sessions/stems
uv run python -m mockclient.simulate --out sessions/demo \
    --stem sessions/stems/voice1.wav:-50 --stem sessions/stems/voice2.wav:40 \
    --stem sessions/stems/music.wav:150
```

Terminal 1 starts the hub (port 8765). Terminal 2 streams the session into it in real time:

```sh
uv run python -m ingest
uv run python -m mockclient.replay sessions/demo --fake-video-fps 15
```

Then open http://localhost:8765/ingest/status to see what the hub holds (seconds of
audio, poses, video frames received and dropped, missing messages). Sessions are
saved in `hub/sessions/`, which is gitignored. Never commit recordings.

Replayer options: `--speed 0` sends as fast as possible, `--loop` repeats (each pass
counts as a reconnect), and `--hub ws://<ip>:<port>` targets another machine.

### Live separation (GPU)

Needs the GPU environment above. The first run downloads the separation and labeling
models (~1 GB) into `hub/checkpoints/` and the Hugging Face cache; both are gitignored.

```sh
uv run --group gpu python -m separate        # hub + live separation (replaces `python -m ingest`)
uv run python -m mockclient.replay sessions/demo
```

http://localhost:8765/separate/status shows each separated stream with its label, level
and compute time per 1 s hop. http://localhost:8765/sources shows the sources with their
direction (world bearing in degrees, + = right), confidence and state (`tracking`,
`last_seen`, `lost`). A source's `source_id` is a separator slot (`s1`, `s2`, ...), not a person.

Score separation on a simulated session (it needs ground truth, so only `mockclient.simulate`
sessions work). It prints the SI-SDR improvement per source, which needs to be at least +8 dB,
and the label for each. On the demo session, expect about +18, +17 and +18 dB, labeled
speech, speech and music:

```sh
uv run --group gpu python -m separate.score sessions/demo
```

Add `--no-labels` to either command to skip the labeling model.

More scoring on simulated sessions:

```sh
uv run --group gpu --with librosa python -m separate.suite   # 11 scenes: 1-3 voices, music, bird, turn-taking
uv run --group gpu python -m locate.score_live sessions/demo  # direction error per source vs truth
```

The suite writes each scene's mix and output streams as WAVs to `hub/sessions/suite/sceneNN/out/`
for listening.

### Real-world direction test (Quest 3S)

Records a take on the headset with a sound source at a known angle and scores the direction
finder on it. Needs the headset set up as in [`quest/README.md`](quest/README.md) and the app
installed. From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File quest/tools/run_locate_take.ps1
```

A voice tells you when to turn your head; it prints PASS/FAIL per take. The pieces can also run
separately: `quest/tools/record_session.py` records (audio + head pose, to `hub/recordings/`,
gitignored), and `uv run python -m locate.real_eval recordings/<take>` (from `hub/`) scores.

### Quest app

Unity, headset and developer-account setup are in [`quest/README.md`](quest/README.md).
The app streams mic audio, head pose and camera to the hub and has a headset UI; in Unity run
**Cocktail > Add Streaming Components** (and **Add UI Components** for the UI) once, then
Build And Run. To check a headset end to end against a mock hub:

```powershell
uv run --project hub python quest/tools/mock_hub.py                 # terminal 1
powershell -ExecutionPolicy Bypass -File quest/tools/validate_quest.ps1   # terminal 2
```

It prints PASS/FAIL for hub reachable, audio and pose rates, camera frame rate, raw mic and
controller input.

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
