# Cocktail

An audio-first assistive experience built at a hackathon by a team of four. A Meta Quest 3S
headset streams its microphones, camera and head pose to a laptop "hub", which splits the
room's sound into separate labeled sources, works out where each one is, and sends the
chosen sound back to the headset in 3D. Plan: [`docs/QUEST_PRD.txt`](docs/QUEST_PRD.txt). Architecture and decisions:
[`ARCHITECTURE.md`](ARCHITECTURE.md).
Live progress: [`docs/STATUS.md`](docs/STATUS.md).
Research and implementation proposal for autonomous audio-visual discovery:
[`docs/AV_SOUND_DISCOVERY_SPEC.md`](docs/AV_SOUND_DISCOVERY_SPEC.md).

> **Status (2026-10-25):** the whole hub loop exists and runs on simulated and replayed data:
> ingest, live separation with labels, direction finding, the tracker (stable ids), `/events`
> pushed to the headset every 100 ms, and the chosen voice plus the room sent back on `/audio/out`.
> The Quest app streams mic audio (2 raw channels, 48 kHz), head pose and camera to the hub
> (hardware-tested) and has a speaker HUD. Tools to record, check and score real takes are in
> place. **Not yet:** a passing real-world direction test (no labeled real takes have been
> scored on the new tools), the chosen-voice audio heard on the headset, camera-based
> outlines (SAM 3), and the full acceptance test. Nothing about accuracy on the real headset
> is measured yet: every direction number in this repo is from simulation unless it says
> "real 3S".

## Repository layout

| Path | What |
|---|---|
| `AGENTS.md` | Shared instructions for every coding agent (canonical) |
| `CLAUDE.md` | Claude Code instructions (imports `AGENTS.md`) |
| `GEMINI.md` | Antigravity instructions (generated) |
| `.agents/` | Canonical roles, skills, sync script, decisions, and branch handoffs |
| `.claude/`, `.codex/`, `.pi/`, `.agent/` | Per-tool copies generated from `.agents/` |
| `graphify-out/` | Local code knowledge graph (gitignored, rebuilt per clone) |
| `hub/` | Laptop hub (Python 3.12), one package per task: `protocol/` wire format, `ingest/` server, `mockclient/` fake headset and recorded-session replay, `separate/` live separation and labels, `locate/` direction finding and its scoreboards, `track/` stable source ids, `events/` the tracker and `/events` feed ([README](hub/events/README.md)), `out/` the chosen voice on `/audio/out` ([README](hub/out/README.md)). Early phases only (no models yet): `vision/` mask store, `who/` audio-visual matcher, `replay/` residual math, `render/` viewer bundle, `av_scene/` autonomous sound discovery |
| `quest/` | Unity project for the Quest headset ([setup](quest/README.md)); `quest/tools/` has the mock hub, headset validation, the take recorder (`record_session.py`) and the real-world take script (`run_locate_take.ps1`) |
| `eval/` | Streaming separation benchmark, plus the real-data tools: `dataset_check.py`, `delay_test.py`, `alerts_check.py`, `results_table.py` ([README](eval/README.md)) |
| `ARCHITECTURE.md` | System overview and architectural decisions, with their status and evidence |
| `docs/` | Plan (`QUEST_PRD.txt`), status board (`STATUS.md`), audio-visual discovery spec (`AV_SOUND_DISCOVERY_SPEC.md`) |

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
uv run pytest       # all tests should pass, except test_events.py::test_simulated_two_voice_music_replay_events,
                    # a real-time timing test that fails about half the time (also on main)
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
Add `--voice-memory` to the live hub, scorer, or scenario suite to opt into WavLM speaker
embeddings. Voice profiles are learned from active speech and consulted when a stream returns
after a pause; low-confidence matches keep the separator's current order. The first use loads
`microsoft/wavlm-base-plus-sv` from Hugging Face. Its model card links a license whose scope for
the checkpoint weights is unclear, so do not redistribute the downloaded weights.

Replay a saved recording (including `sessions/live1`) and write the mix, per-stream WAVs,
embedding similarities, per-hop energy, compute timing, and CUDA memory metrics without
ground-truth stems:

```sh
uv run --group gpu python -m separate.replay sessions/live1 --out sessions/live1/out_voicememory
```

More scoring on simulated sessions:

```sh
uv run --group gpu --with librosa python -m separate.suite   # 11 scenes: 1-3 voices, music, bird, turn-taking
uv run --group gpu python -m locate.score_live sessions/demo  # direction error per source vs truth
```

`separate.suite --sessions-root <folder>` reuses previously generated `sceneNN/session`
recordings when librosa's example clips are unavailable offline. Combine it with
`--voice-memory` to compare the baseline and remembered stream assignments on the same inputs.

The suite writes each scene's mix and output streams as WAVs to `hub/sessions/suite/sceneNN/out/`
for listening. `--voice-memory` adds identity similarity traces and per-hop compute timing to
`results.json`, with CUDA memory in `cuda_memory.json`.

### The live loop: tracked sources, events and sound back

The same `python -m separate` entry point serves the whole loop. Nothing extra to start.

| Endpoint | What |
|---|---|
| `GET /sources` | The locator's raw view: one entry per separator slot (`s1`, `s2`, ...) |
| `GET /tracks` | The tracker's view: stable ids `t1`, `t2`, ... with label, world bearing, confidence and state |
| `WS /events` | The same tracked sources pushed to the headset every 100 ms (also empty updates, and state changes after audio stops). Details: [`hub/events/README.md`](hub/events/README.md) |
| `WS /audio/out` | The chosen voice (inner layer) plus the rest of the room (outer layer), 48 kHz, one frame every 20 ms. `GET /audio/out/status` shows the choice, queue, drops and underruns. Details: [`hub/out/README.md`](hub/out/README.md) |

The headset's select / stop / next / prev arrive as `InputEvent` on `/ingest/pose`. A selection is
a tracker id, so it follows the same talker through separator slot swaps and pauses. Delay: the
hub holds audio about 0.3 s so the tracker has seen the same hop, plus up to 0.2 s of jitter margin
after idle; the separator adds its own 1 to 2 s (see `ARCHITECTURE.md`, D7 and D15).

Headset-free checks: `uv run pytest tests/test_events.py tests/test_audio_out.py` (no GPU needed). The
events test uses a scripted replay with mocked perception, so it checks transport, ids and timing, not
acoustic accuracy. `tests/test_events.py::test_simulated_two_voice_music_replay_events` asserts an
8 to 12 Hz event rate from a real-time replay and is known to fail about half the time (also on `main`).

How much louder is the chosen voice than the rest, and how soon? (simulated sessions with ground
truth; refuses real recordings):

```sh
uv run --group gpu python -m separate.measure_chosen sessions/suite/scene01/session
```

### Real-world direction test (Quest 3S)

Records a take on the headset with a sound source at a known angle and scores the direction
finder on it. Needs the headset set up as in [`quest/README.md`](quest/README.md) and the app
installed. From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File quest/tools/run_locate_take.ps1
```

A voice tells you when to turn your head; it prints PASS/FAIL per take. The pieces can also run
separately: `quest/tools/record_session.py` records (audio + head pose + camera frames, to
`hub/recordings/`, gitignored), and `uv run python -m locate.real_eval recordings/<take>` (from `hub/`)
scores.

Camera frames land in `<take>/video/<device_ts_us>.jpg` with the per-frame header (size, camera pose,
field of view) in `<take>/video/frames.jsonl`, the layout `mockclient.Session.load` and
`python -m mockclient.replay <take>` read, so a recorded take can be replayed into the hub and fed to
SAM 3. Frames take about 0.5 MB/s; use `--video-fps N`, `--max-frames N` or `--no-video` to limit them, and
`--camera-flipped` if the headset's `flipVertical` was on (that flag is not sent on the wire). To score
"box on the real talker", label who talks when in `<take>/talkers.csv`
(`start_s,end_s,talker_id,bearing_deg,note`, seconds since the first audio sample; format and helpers in
`hub/mockclient/talkers.py`).

Label every take so the coverage grid can be checked: `run_locate_take.ps1 -Distance 1 -Motion normal
-Stimulus pink` (`-Motion` small / normal / large, `-Stimulus` pink / speech; speech needs
`-StimulusWav <file.wav>`). The recorder also writes `audio_chunks.csv` (per-chunk device timestamps)
so dropped audio and clock drift can be detected.

### Checking and scoring real takes

All of these take a folder of takes (`hub/recordings/`, gitignored; recordings go in the team's
shared folder, never in git).

```sh
# from the repo root: coverage grid (12 angles x 0.5 / 1 / 2 m) and PASS/FAIL per take
uv run --project hub python eval/dataset_check.py hub/recordings

# from hub/: every direction method on all labeled takes (median / p90 error by region,
# front/back right, seconds to lock, whether the confidence is honest); --json saves it
uv run python -m locate.bench recordings
# why does the delay method miss? (fit residual, audio-vs-pose timing shift, clock span, channel coherence)
uv run python -m locate.diagnose recordings/<take> ...
# fit a delay calibration at the rate the live hub measures at (16 kHz)
uv run python -m locate.real_eval calibrate recordings/<take> ... --rate 16000 --out my-calib.json
```

`locate.bench` reports `tdoa-loo` (calibration refitted without the take being scored); use that, not
plain `tdoa`, for any baseline, because the stored calibration was fitted on a handful of takes. A
calibration records the sample rate it was fitted at; the stored `locate/calib/quest3s.json` is a 48 kHz
one, and the live hub (which measures at 16 kHz) warns about that. The live entry point has no
`--calib` option: it loads `locate/calib/quest3s.json`, so swapping in a 16 kHz fit means replacing that
file, which is a team decision (open a PR).

No headset? `python -m locate.pairsim` writes simulated Quest 3S takes in the recorder's format (and,
with `--scene`, a two-talker session), and `locate.scenes` / `locate.score_live --oracle` score locating
each separated stream with the true stems standing in for the separation model (no GPU). These are
simulations: they test the method and the plumbing, not the real headset.

```sh
cd hub
uv run python -m locate.pairsim sessions/pairsim --angles 0,90,180,-90,45,135,-45,-135 --seeds 2
uv run python -m locate.real_eval calibrate sessions/pairsim/sim-* --rate 16000 --out sessions/pairsim-calib16.json
uv run python -m locate.scenes sessions/scenes --calib sessions/pairsim-calib16.json
```

Round-trip delay (headset to hub and back, 30 or more trials) and the alerts rules:

```sh
uv run --project hub python eval/delay_test.py --mock          # on the laptop, mocked headset
uv run --project hub python eval/alerts_check.py --help       # rules in eval/alerts_rules.json
uv run --project hub python eval/results_table.py --help      # one markdown table against the targets
```

The delay test has only run against the mock; running it on a Quest needs `adb reverse` and the app
(see the top of `eval/delay_test.py`). Targets for locating: 15 degrees or less in front, 25 behind,
front/back right 95% of the time, locked within 3 s, two speakers within 25 degrees, and a confidence
that is right about as often as it says.

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
- **Point the app at the hub**: write the address to `/sdcard/Android/data/com.DefaultCompany.quest/files/hub_host.txt`
  (`127.0.0.1` over USB), then restart the app. The forward is dropped on every reinstall, so run `adb reverse` again.
  Using the headset HUD (picker, pinch, DevMode, the DEV chip, the event log) and fixes for problems seen in testing:
  [`quest/README.md`](quest/README.md#using-the-app-headset-hud).

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
