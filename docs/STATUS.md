# Cocktail status board

Shared progress for everyone on every computer. Plan source of truth: [QUEST_PRD.txt](QUEST_PRD.txt).
**Edit only your own row** (one row per task, so git merges cleanly). Pull before you
start; push right after you claim or finish a task.

Statuses: `todo`, `doing`, `done`, `blocked`. Evidence: hardware-tested, mocked,
replayed, untested.

## How tasks stay flexible
- Owners are defaults, not walls. Anyone with free time can take a `todo` row: put your
  name and branch in the row and set `doing` (pull first, push right away).
- Taking over a `doing` row: ask the current owner in person first, then update the row.
- Stuck for 20 minutes? Set the row to `blocked`, say why in Notes, and tell the team.
- Each task has its own folder and branch, so switching owners never means moving code.
  Only `hub/protocol` stays gated (integration owner approves changes).
- Demo work (pitch, filming, README) starts only after the product works on the headset.

## People and machines
People and computers can swap, except Cody and his computer (the GPU machine).

| Person | Machine / Claude | Role (default) |
|---|---|---|
| Cody (codyn06) | RTX 5070 Ti, 12 GB (Blackwell: PyTorch cu128); Unity | Compute owner: live hub, separation, chosen-voice audio; runs GPU jobs (SAM 3) for everyone. Gets priority on compute tasks |
| Lalitha (Lalithaaaa) | RTX 4060 8 GB laptop, Unity; $20 Claude | Headset experience and integration owner (`quest/`, protocol changes) |
| Sadie | $100 Claude; no Unity or GitHub yet | Location accuracy: the Claude-heavy work |
| Shayaan | weak GPU; no Unity or GitHub yet | Real-world data and testing lead |

Headsets: **Quest 3S** (Cody's, shared by everyone; the only one with camera access) and **Quest 2** (Lalitha's; mic, head pose, UI and audio, but Meta's camera API is Quest 3/3S only).

### First steps for every person's AI
1. Ask the person their name and role, and remember it.
2. GPU check: `nvidia-smi --query-gpu=name,memory.total --format=csv`. If it has 12 GB or more, or is RTX 4070 Ti / 4080 / 4090 / 5070 Ti / 5080 / 5090 class, tell the team: that machine can share Cody's compute tasks.
3. Read `AGENTS.md` and this file; work only on that person's milestone, one PR per subtask, each with "Expected / You check" steps the person can run.

### Setup without Unity (Sadie, Shayaan)
GitHub account, then Cody invites them to `codyn06/Cocktail`; install Git, GitHub CLI (`gh auth login`), `uv`, Claude Code; clone; `cd hub && uv sync && uv run pytest -q` (expected: all pass). Headset work needs only `adb` (an `.apk` built by Lalitha or Cody, installed with `adb install`).

## Milestones (updated Oct 25)
Product target 10 AM Sunday is best-case, not a hard stop; ask the team before cutting corners.
M1 to M4 run in parallel; M5 pulls them together.

| # | Milestone | Owner | Done when (testable) |
|---|---|---|---|
| M1 | **Live loop on the headset** | Lalitha | Two people talk; within 1 s the outline is on the talker in view, and the dot and arrow point to the one out of view; ids stay stable for 60 s |
| M2 | **Hear the chosen voice** | Cody | Select a voice: it is at least 10 dB over the rest within 1 s; switching to the other voice works |
| M3 | **Location you can trust** | Sadie | Real 4-angle test: error 25 deg or less on 3 of 4 takes. Then the targets below |
| M3d | **Real data for M3** | Shayaan | Labeled recordings cover every angle (coverage grid full), 90% or more pass the data check; shared test scene recorded; round-trip delay measured |
| M4 | **Outline on the actual person** (after M2) | Cody | Camera detection puts the outline box on the real talker in 80% or more of frames (Quest 3S only) |
| M5 | **Demo-ready** | everyone | Acceptance test F2 passes; metrics (G1) measured; video recorded |

Subtasks:
- **M1 (Lalitha):** speaker HUD (#38) on the headset with fake data, then with Cody's live hub (#37); tune outline size, view angle, dot and arrow; design pass; "point the controller at the sound and pull the trigger" truth labels for Shayaan's recordings.
- **M2 (Cody):** merge #37 and #39; stream the selected source to `/audio/out` with inner and outer volumes; keep the voice locked through pauses; print gain in dB and delay in ms.
- **M3 (Sadie):** `python -m locate.bench recordings/` scoreboard (median and p90 error, front/back rate, time to lock, confidence calibration); tune the inter-mic delay method (#36) with per-device calibration; locate each separated stream; combine delay, loudness and head-motion history in one posterior with honest confidence; tracker smoothing on real data; camera snap once M4 exists.
- **M3d (Shayaan):** tape angle marks every 30 deg at 0.5, 1 and 2 m; record pink noise and speech at small, normal and large head motion on both headsets with `quest/tools/run_locate_take.ps1`; build `eval/dataset_check.py` (run: `uv run --project hub python eval/dataset_check.py <recordings folder>`) (coverage grid, PASS/FAIL per take); record the shared test scene (C2); round-trip delay test (C1); later run the scoreboard for G1.
- **M4 (Cody):** save camera frames in the recorder; person detection on the hub; boxes into sources; match the box to the speaker by direction (E2).

### Scope: what each person owns
Edit only your own folders. Need a change somewhere else? Ask its owner (or open a PR and tag them).

| Person | Owns (edits freely) | Gets from others | Gives to others | Does NOT touch |
|---|---|---|---|---|
| Cody | `hub/separate`, `hub/events`, `hub/ingest`, `hub/mockclient`, new `hub/out` (audio to the headset), new `hub/vision` | live headset stream; protocol (Lalitha) | separated streams and tracked sources on `/events`; chosen-voice audio on `/audio/out`; GPU runs for others | `quest/`, `hub/locate`, `hub/track` |
| Lalitha | `quest/` (Unity app, UI, input, headset audio), `hub/protocol` (integration owner: approves every protocol change) | `/events` and `/audio/out` from Cody's hub | headset builds (`.apk`) for Shayaan; truth-label tool; UI feedback | hub Python other than the protocol |
| Sadie | `hub/locate`, `hub/track` (direction and tracking math, the `locate.bench` scoreboard) | recordings from Shayaan; separated streams from Cody | better bearings and confidence in every source | `quest/`, `hub/separate`, the protocol (ask Lalitha) |
| Shayaan | `eval/` (data check, delay test, metrics), `quest/tools/record_session.py` and `run_locate_take.ps1` (recording tools), the shared recordings folder | headset build from Lalitha or Cody; new code to score from Sadie and Cody | labeled recordings, the test scene, measured numbers | `hub/locate` math, `quest/` Unity code |

Shared rules: `docs/STATUS.md` (edit only your own rows), dependency files (`hub/pyproject.toml`, `uv.lock`, `quest/Packages`) are changed only after telling the team. **Recordings are not in git** (too big): they go in one shared drive folder (Shayaan sets it up and posts the link); a few short takes may be committed as regression tests.

### Location targets (M3, change them as we learn)
In front: median 15 deg or less. Behind: median 25 deg or less. Front/back correct 95% or more. Lock within 3 s of head movement. Two speakers: both within 25 deg. Confidence honest: when it says 90% sure, it is right about 90% of the time.

### Also to do
Unity scene commit and TMP essentials decision (Lalitha); A1 one-command scripts; F1 modes; G2 echo and delay tuning; G3 glasses mode; A10 network check; alerts A7/D2 and clean replay D3 for whoever frees up first; G4 README, A9 pitch, H1/H2 video and submission at the end.

## Tasks

**Main goal: split every sound into its own source and track what it is and where it
is (marked ★).** Tasks are listed in the order they unlock each other. "Needs" means
the earlier tasks that must be usable first; mocks and recordings let you start sooner.

### Setup and foundations (start now, no dependencies)

| ID | Task | What it means / done when | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|
| A1 | Repo and run scripts | Public repo with MIT license, README skeleton, one-command scripts: run the hub, replay a recorded session, deploy to the headset, run the evaluation. | repo root, `scripts/` | | | todo | |
| A2 | Message format between headset and laptop | The frozen set of messages (audio, video, head pose, button actions, events back) with generated Python and C# code. | `hub/protocol` | | feat/a1-a2-repo-protocol | done | merged in PR #10; mocked (5 tests) |
| A3 | Check all the AI models fit on one GPU | Run separation, speech enhancement, SAM 3 (object outlines), depth and SAM Audio together on the RTX 5070 Ti. Done when we know which ones fit, how fast, and how much memory. | `eval/` | Cody | chore/gpu-env | PR #13 merged: cu128 env, 12 GB budget, C3 candidates benchmarked (hardware-tested). done | SAM 3 image ~4 GB, SAM Audio small-tv ~4.4 GB (core only), live separation ~1.1 GB: fit together in 12 GB. SAM 3.1 tracker crashed Windows (see Open decisions) |
| A4 | Unity and headset ready | Unity with the Meta XR and Meta XR Audio SDKs builds and runs hello-world on the Quest 2 and Quest 3S. Confirm the 3S software version supports the camera API (v74+). | `quest/` | Lalitha | test/quest-3s-all | done (hardware-tested) | Unity 6000.0.80f1 builds and runs on the Quest 3S; PR #18 merged |
| A5 | Headset screen design with fake data | Labels and outlines on each sound in view, edge arrows and clock positions (like "4 o'clock") for sounds out of view, state banners, a small glasses-style view. Runs on the Quest 2 with made-up sources. | `quest/ui` | Lalitha | feat/a5-headset-ui | done (mocked) | PR #28 merged. Fake-data screen compiles and passes its self-test; NOT yet viewed on the headset. Teal "lost" colour fails the 3:1 contrast check |
| A6 | ★ Choose the audio AI models and build the scoring test | Pick the model that splits mixed audio into voices, the sound labeler, the voice-ID model and the speech enhancer. Build a test that mixes known clean recordings and scores the result. | `hub/separate`, `eval/` | Cody | feat/c3-live-separation | done (simulated) | PR #19 merged. MossFormer2 (voices) + residual + AST labels; scoring test `python -m separate.score` |
| A7 | Alerts engine | Rules file that matches sounds (smoke alarm, siren) and the wearer's name, with a name-detection model. Tested with the laptop mic. | `hub/alerts` | | | todo | |
| A8 | ★ Work out where a sound is from head movement | Math that turns "this sound got louder and brighter as the head turned" into a direction, all 360 degrees, in view or not. Tested on simulated audio and head turns. | `hub/locate` | Sadie | feat/a8-locate | done (simulated) | PR #11 merged. Median error ~3 deg at +-30 deg head motion, ambiguous with no motion. Real-world check: PR #29 recorder and scorer, not yet passed (needs head turns and a louder source) |
| A9 | Pitch and demo outline | Outline of the video and slides. Start after the product works. | `docs/` | | | todo | later |
| A10 | Network | Laptop and headset on the same network. University Wi-Fi is good enough for now; a private hotspot is later and only if delay or jitter becomes a problem. | none | | | todo | low priority, later |

### Getting data flowing

| ID | Task | What it means / done when | Needs | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| B1 | Headset sends microphone sound and head pose | Live mic audio and head position stream to the laptop. Report how many mic channels there are and whether the audio is raw. Add a native plugin if Unity's mic is too processed. | A2, A4, A10 | `quest/stream` | Lalitha | feat/b1-quest-streaming | done (hardware-tested) | PR #18 merged. Mic: UNPROCESSED capture gives 2 live raw channels at 48 kHz, nearly co-located, so no usable time-difference direction; direction comes from head movement (A8). Other audio sources give duplicated processed mono |
| B2 | Headset sends camera video | Quest 3S passthrough camera frames compressed on the headset and streamed. Report frame rate, size and delay. | A2, A4, A10 | `quest/stream` | Lalitha | feat/b1-quest-streaming | done (hardware-tested) | PR #18 merged. validate_quest.ps1 requires >= 8 camera frames per second and passed; exact rate and delay not recorded yet |
| B3 | Laptop hub receives everything | Hub accepts audio, video and pose on separate channels so big video never delays audio; lines up their clocks. | A2, A3 | `hub/ingest` | Cody | feat/hub-ingest-mock | done (simulated) | PR #15 merged; last 10 s audio (16 kHz) + pose, newest video frame. Assumes chunk ts = first sample; confirm with B1 |
| B4 | Fake headset for testing | A program that plays a recorded or simulated session into the hub, so nobody needs the headset to test. | A2 | `hub/mockclient` | Cody | feat/hub-ingest-mock | done (simulated) | PR #15 merged; simulator (2 mics, head turns) + real-time replayer. See README "Hub (laptop)" |
| B5 | Choosing a sound | Look at it or point with hand or controller, pinch to confirm. Actions: select, stop, replay, next. | A4, A5 | `quest/input` | Lalitha | feat/b1-quest-streaming | done (hardware-tested) | PR #18 merged. Controllers: right trigger = select; grip or B = stop; A or X = replay; Y or stick right = next; stick left = previous. Presses reaching the hub were hardware-checked (validate_quest.ps1) |
| B6 | Play sound in 3D with two layers | Play the chosen sound at its direction, plus the rest of the room quieter. Separate volumes. | A4 | `quest/audio` | Lalitha | feat/b1-quest-streaming | done (built, not heard) | PR #18 merged. Plays hub audio out with separate inner and outer volumes (applied on the headset). Compiles and runs; playback of real hub audio NOT yet checked by ear |

### Making it work for real

| ID | Task | What it means / done when | Needs | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| C1 | Round-trip delay test | Send mic sound to the hub and straight back, unprocessed. Measure delay and jitter. | B1, B3 | `eval/` | Shayaan | | todo | quick test |
| C2 | Record the test scene | On the 3S: two people talking at once plus music, with natural head movement. Everyone tests on it. | B1, B2 | `mock/` | Shayaan | | todo | needs the headset |
| C3 | ★ Live sound separation and labels | Hub splits the live mix into separate voices and sounds and names each (speech, music, alarm). Tested on the fake headset, then on the recording. | A6, B3 | `hub/separate` | Cody | feat/c3-live-separation | done (simulated) | PR #19 merged. +18.3/+17.1/+17.9 dB on simulated 2 talkers + music, labels correct, ~460 ms per 1 s hop, ~1.1 GB VRAM. Not yet on the 3S recording |
| C4 | ★ Live direction finding | Hub gives every separated sound a direction using the head-movement math. | A8, B3 | `hub/locate` | Sadie | feat/c4-live-locate | doing | PR #21, waits on #11. Simulated errors 40/9/16 deg; estimator overconfident (see #11 comment) |
| C5 | Camera confirmation | SAM 3 outlines objects in the camera view and depth gives distance, to confirm which object makes a sound. | A3, B2 | `hub/vision` | Cody | | todo | |
| C6 | Live headset screen | The designed screen driven by real hub events. | A5, B1, B3 | `quest/ui` | Lalitha | | todo | |

### Tracking, selecting, hearing

| ID | Task | What it means / done when | Needs | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| D1 | ★ Track every sound | Each sound keeps one identity, direction and state (tracking, last seen, lost), in view or out of view. | C3, C4 | `hub/track` | Lalitha | feat/d1-track | doing | PR #30 draft. Simulated scenes only; not wired to the live hub until #21 and #27 merge |
| D2 | Alerts that interrupt playback | A smoke alarm, siren or the wearer's name lowers the playback and speaks with direction. | A7, C3, B6 | `hub/alerts` | | | todo | |
| D3 | Clean replay of the last 5 seconds | SAM Audio re-processes the last 5 s of the chosen sound and plays it back clean. | A3, B3, B6, C3 | `hub/replay` | | | todo | |
| E1 | Choose a sound and hear it | The chosen person or sound is amplified and enhanced over a quieter room. A voice fingerprint keeps hold of them. | D1, B5, B6, A6 | `hub/separate`, `quest/audio` | Cody | | todo | |
| E2 | Who is talking | Direction decides which voice belongs to which person; lip and body motion confirm. | D1, C5 | `hub/who` | Cody | | todo | needs the headset |
| E3 | Show every tracked sound | Labels and arrows for every tracked sound on the headset. | D1, C6 | `quest/ui` | Lalitha | | todo | |

**Checkpoint 1 (target tonight): E1 + E3 working live on the Quest 3S with D1 tracking.**

### Finishing

| ID | Task | What it means / done when | Needs | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| F1 | Three modes | Fast (no lookahead), Balanced (~100 ms lookahead), Accurate (phrase-level, played in pauses). Switchable live. | E1 | `hub/modes` | | | todo | |
| F2 | Full acceptance test | Two people talking plus music: select A, then B, then the music; turn away and keep tracking; alerts fire; replay works. | E1, E2, E3, D2, D3 | all | | | todo | |
| G1 | Measure everything | At least 30 trials each: separation gain, wrong-speaker rate, direction error, delay per mode, switch time, false alerts. | F2, A6 | `eval/` | Shayaan | | todo | |
| G2 | Echo and delay tuning | Make the enhanced voice stand out from the room's real sound; push delay down. | F1, C1 | `hub/modes` | | | todo | |
| G3 | Glasses-like test mode | Lower camera quality and added delay to mimic glasses. | F2, A5 | `quest/ui` | | | todo | |
| G4 | README | Setup, architecture, metrics, limitations. | F2 | `README.md` | | | todo | after product works |
| H1 | Film the demo video | From the headset view, ending with measured numbers. No merges while filming. | F2, G1, A9 | none | | | todo | after product works |
| H2 | Submit video and repo | Before 10 AM. | H1, G4 | none | | | todo | |

If time remains: guidance to a sound, captions, voice commands, "what's that?", Twinly
alert messages, Meta glasses version, search and rescue.

## Open decisions / blockers
- Integration owner: unassigned.
- Unity: repo pins 6000.0.80f1; this machine has 6000.6.7f1.
- A8 needs per-microphone levels for each separated sound. Confirm the Quest mic count
  (task B1) and that separation (C3) can provide per-mic levels.
- GPU: the live-run machine is an RTX 5070 Ti **Laptop** GPU with **12 GB** VRAM
  (11.9 GiB usable, `nvidia-smi` 2026-10-24; not 16 GB). Blackwell (sm_120): install with
  `cd hub && uv sync --group gpu` (torch 2.11.0+cu128, CUDA available, ~14.5 TFLOPS fp16
  matmul). Hub laptop (RTX 4060, 8 GB) is for development. Still to measure:
  do separation, SAM 3, enhancement and SAM Audio fit together on 12 GB (A3)?

  Draft 12 GB budget (docs research 2026-10-24, **not yet measured on device**):

  | Component | Residency | Est. VRAM | Basis |
  |---|---|---|---|
  | OS / compositor / CUDA contexts | always | ~1-1.5 GB | assumption |
  | Live separation (C3) + enhancement | resident | ~1-2 GB | small models |
  | SAM 3 (C5) | resident | ~4-6 GB | extrapolated from 848M params |
  | SAM Audio small(-tv), bf16 (D3) | on demand; unload SAM 3 | ~7.4 GB | third-party, RTX 5090 |
  | Gemma 4 E2B Q4_0 | on demand | 2.9 GB + KV | gemma.ai Table 1 |

  SAM Audio is batch-only (no streaming), so it cannot do C3. Base/large and bf16 Gemma do not fit
  beside other models.

  SAM 3 image model (hardware-tested on the 5070 Ti Laptop, Windows 11, 2026-10-24; separate
  environment, not in the hub yet):

  | | |
  |---|---|
  | VRAM | 3.5 GB weights, ~4.0 GB peak |
  | Load | ~13 s |
  | Per frame | ~320 ms image encode + ~80 ms per text prompt (~2.5 fps with one prompt) |
  | Precision | fp32 weights under bf16 autocast (as upstream). Converting weights to bf16 breaks it |

  - Windows needs the community `triton-windows` package (no official Triton for Windows), plus
    `pycocotools`, `psutil` and `opencv-python-headless` that the image path imports.
  - Prompt wording matters a lot: on a photo of six children, "child" scored 0.96 but "person"
    only 0.27 (below the default 0.5 threshold, so 0 detections). C5 should try several prompts or
    lower the threshold.
  - **SAM 3.1 video tracker: not usable on this laptop.** On Windows it needs `use_fa3=False`, a
    workaround for an upstream bug (`start_session` passes `offload_state_to_cpu`, which the 3.1
    tracker rejects), and allowing non-flash attention in `sam3/model/decoder.py` (it forces the
    flash kernel, which Windows PyTorch lacks). With those, it found people on frame 0, then
    tracked at ~15.5 s per frame, and **Windows crashed both times** (bugcheck 0x00020001,
    hypervisor error, 2026-10-24 14:33 and 15:24). Do not run it on this machine. Use the SAM 3
    image model per frame for C5.
  - SAM Audio small: first load also downloads ~16 GB of companion models (`sam-audio-judge`
    12.3 GB, CLAP 1.8 GB, T5/RoBERTa/BART/BERT ~2.3 GB) on top of the 5.1 GB checkpoint. Needs
    `transformers` 4.x (>=4.54) with `huggingface_hub` < 1.0, `soundfile` for audio I/O
    (`torchaudio.load` needs FFmpeg DLLs on Windows), and the span predictor at the revision
    SAM Audio pins (`facebook/pe-a-frame-large@40187271`, 6.1 GB; the latest revision is a
    different file).

  SAM Audio small-tv (hardware-tested, same machine; text prompts, 1 candidate, no span
  prediction, fp32; 5 s of the synthetic 2 talkers + music mix):

  | | |
  |---|---|
  | VRAM, everything on GPU | 22.3 GB: does not fit (Windows spills into shared system memory) |
  | VRAM, core only on GPU | 2.7 GB weights, **4.4 GB peak**. Codec, T5 text encoder and DiT on GPU; span predictor, judge/CLAP rankers and vision encoder stay on CPU (unused for text prompts with 1 candidate) |
  | Load | ~64 s from disk (keep it loaded if replay is used often) |
  | Time per 5 s clip | ~2.1 s per prompted sound |
  | SI-SDR improvement | voice 1 +1.9 dB, voice 2 +5.4 dB, music +2.5 dB (MossFormer2: +18.1 / +16.8 / +17.8) |

  SI-SDR penalizes generative output (SAM Audio re-synthesizes audio, so small phase and timing
  differences count as error). The low numbers may understate it; judge D3 quality by listening.
  For C3 (live) MossFormer2 stays: SAM Audio is batch-only and ~2 s per sound per clip.
  - All SAM weights together use ~35 GB of disk in the Hugging Face cache.

  C3 candidate benchmark (hardware-tested on the 5070 Ti, 2026-10-24, torch 2.11+cu128, fp32,
  **synthetic input: speed/VRAM only, separation quality not measured yet**):

  | Model | Task | Peak VRAM | Latency (p50) | RTF |
  |---|---|---|---|---|
  | GTCRN (MIT) | speech enhancement, 16 kHz | 30 MiB | ~7 ms per call (64 ms - 2 s windows) | 0.003-0.13 |
  | Waveformer (MIT) | streaming target-sound extraction, 44.1 kHz, 41 classes | 66 MiB | 6.5 ms per 9.4 ms hop | 0.69 |
  | MossFormer2_SS_16K (Apache-2.0) | 2-speaker separation, 16 kHz, non-streaming | 714 MiB | 141 ms / 1 s window, 266 ms / 2 s | ~0.13 |

  Waveformer's classes include instruments but no speech or generic "music" class, so it is a
  weak fit for the 2 talkers + music test. MossFormer2 latency = window length + compute.

  C3 quality on a synthetic mix (replayed: librosa examples libri2 + libri3 + Brahms at -5 dB,
  12 s, 16 kHz mono, no reverb; 2026-10-24). MossFormer2_SS_16K in 2 s windows / 1 s hop,
  speaker order kept via the overlap; music = mix minus gain-fitted voices:

  | Stream | SI-SDR | Improvement over mix |
  |---|---|---|
  | voice 1 | 16.8 dB | +18.0 dB |
  | voice 2 | 15.2 dB | +16.3 dB |
  | music (residual) | 8.9 dB | +17.0 dB |

  Offline 12 s gives the same numbers. Caveat: clean read speech from LibriSpeech, which may overlap
  the model's training data; expect lower on the real 3S recording (reverb, mic noise, crosstalk).

  Multi-source AV separation and attribution (2026-10-25). Evidence: **SYNTHETIC mixtures of recorded
  clips**; models hardware-run on the RTX 5070 Ti Laptop (12 GiB); timing measured while another GPU job
  was running.

  SAM Audio small-tv, `overlap_stack_AB` (8 s clip, A = woman top, B = man bottom, RMS-matched, 8
  candidates, seeds 100-107):
  - **Prompting A:** every non-oracle candidate resembles B. Candidate 0 SI-SDRi on A is about -29 dB;
    best is -12.7 to -14.4 dB; 0/8 candidates reach >= +3 dB.
  - Holds for prompts "person speaking", "speech, single speaker", "speech, one person" and "woman
    speaking", with no anchors / predicted spans or with full-clip anchors. With the visual prompt off it
    is still -28 dB.
  - Oracle spans from clean refs (**DIAGNOSTIC only**) gave one candidate at +6.2 dB, which cannot be
    selected without the reference.
  - **Prompting B:** +7.3 dB for candidate 0, 8/8 candidates.
  - Conclusion: the visual mask does not steer which voice is returned. The bias is signal-dependent: the
    team's earlier report favored the woman, here it favors B.
  - Input audit passed: 48 kHz mono, no clipping, 200 frames <-> 200 features (0 ms mapping error); mask
    polarity correct (target kept 0.9996, background 0).
  - The vision encoder squashes frames to 336x336 without preserving aspect ratio (stretch 1.125 for the
    stack layout, 0.28 for side-by-side). Untested possible contributor.
  - Cost: 8 candidates ~50 s per condition; peak 8.26 GiB; span predictor kept on CPU.
  - The installed sam_audio bb4c6999 includes the issue #98 span fix. Span prediction runs only when
    anchors is None and is driven by the text prompt.
  - BA ordering (B top, A bottom), prompt "person speaking", full-clip anchor: prompting A still gives
    cand0 -28.8 dB, best -14.4 dB; prompting B +7.3 dB. Same as AB, so the bias follows the voice, not the
    screen position. Judge ranker: not run (the full matrix was stopped at 49/94 conditions once the
    selection question was answered; ImageBind visual ranker weights are not cached).
  - Non-speech (Phase 1 fixture, cat chirp + off-screen knock, prompt "cat chirp"): candidates span
    -93 to +10.6 dB SI-SDRi (cand0 -93.5 with the full-clip anchor, +10.1 with predicted spans). Useful
    candidates exist but are seed/anchor-sensitive, so prompted non-speech extraction needs a ranker or a
    listening check before it is trusted. For the silent person in that scene the output was -41.2 dBFS
    against a -39.5 dBFS mixture: SAM Audio returns sound for an absent target.
  - Speech over music, prompted A: cand0 +6.6 dB.

  MossFormer2_SS on the same mixtures: best-matching output stream SI-SDRi +14.5 dB for A and +17.9 dB for
  B, identical across stack/side and AB/BA layouts (same audio). Separation succeeds where SAM Audio
  selection fails.

  Attribution, first run (head-crop heuristic): failed. The eval masks are whole-tile rectangles, so the
  "head box" was the whole tile, TalkNet saw no face, every track scored inactive, and the system refused
  all selections (no wrong voice returned). Fix: S3FD face detection.

  Attribution after the fix (S3FD faces, TalkNet, AV-TSE matched to MossFormer2 voices; commit 1f49aab;
  `eval/av_scene/run_speech.py --tse`, output in `hub/sessions/av_matrix/speech4/`):

  | case | A | B | voices |
  |---|---|---|---|
  | overlap, stack/side x AB/BA (4 layouts) | ok, +16.9 dB, correct | ok, +16.9 dB, correct | both matched |
  | A alone | ok (A's voice) | refused: target inactive | 2nd voice unmatched |
  | B alone | refused: target inactive | ok (B's voice) | 2nd voice unmatched |
  | nobody speaks | refused | refused | none matched |
  | A + off-screen V3 | ok, +17.8 dB, correct | refused | V3 unmatched, `unknown` (overlaps A's speech) |
  | only off-screen V3 | refused | refused | both `outside_view` |
  | A + off-screen music | ok, +11.5 dB, correct | (not scored) | |

  - TalkNet speaking probability with S3FD faces: 0.87 / 0.91 for the talker alone, 0.07-0.19 when silent;
    in overlap 0.52 / 0.64 (it under-detects simultaneous speech, 3.8 s / 5.6 s active of ~7 s).
  - SI-SDRi is not meaningful for solo cases (the mixture equals the reference); those rows are judged by
    selection only. No listening review yet.
  - Wall 2.7-7.9 s per 8 s case, peak 1.6 GiB (MossFormer2 + TalkNet + S3FD + AV-TSE loaded together).
  - Limits: one speaker pair; an off-screen voice that overlaps visible speech stays `unknown`.

  AV-TSE (per face, still with the bad crops):
  - offscreen3, A: +4.7 dB, correct selection (margin 13.5 dB)
  - speech+music, A: +4.5 dB, correct (margin 22.6 dB)
  - silent target: TSE output -16.7 dBFS, i.e. not silent

  Speech-branch peak VRAM 1.52 GiB (MossFormer2 + TalkNet + TSE together). Wall time 6-16 s per 8 s case,
  measured while the SAM Audio run was also on the GPU.

  Limitations: 2-talker separator limit; a single speaker pair; seeds are not generalization; reference B
  is derived (compB - compA); no recorded natural-scene or device test yet; off-screen attribution relies
  on visible faces being confidently silent; latency is not real-time.

  **Phase 3 Row 0 baseline** (2026-10-25, branch `feat/av-phase3-overlap-identity`, commit e842c7c + Step 0 fixes;
  `eval/av_scene/run_speech.py --tse`, output `hub/sessions/av_matrix/row0/`; GPU idle, hardware-run models on
  SYNTHETIC mixtures; no listening review). Ablation rows below compare against this.

  | case | A | B | C | notes |
  |---|---|---|---|---|
  | overlap, 4 layouts | ok +16.9 dB, correct | ok +16.9 dB, correct | | both voices matched; TalkNet p 0.52 / 0.64 (3.7-3.8 s / 5.2-5.6 s active) |
  | A alone / B alone | ok / refused (inactive) | refused / ok | | 2nd voice `unmatched`, `unknown` |
  | nobody speaks | refused | refused | | none matched |
  | A + off-screen V3 | ok +17.8 dB, correct | refused | | V3 `unmatched`, `unknown` |
  | only off-screen V3 | refused | refused | | both voices `outside_view` |
  | A + off-screen music | ok +11.5 dB, correct | | | |
  | **overlap3_side_ABC** (new) | refused: "TSE output matches no separated voice" (TSE itself +15.9 dB) | `degraded` +18.2 dB, correct | refused: inactive (TalkNet p 0.12) | 3 suspected talkers > 2-voice separator |

  - Wall 2.7-8.4 s per case, peak VRAM 1.6 GiB.
  - overlap3 caveat: C's video tile is A's video rolled in time, so C's lips are NOT synced to C's audio
    (TalkNet correctly scores it inactive). It exercises the 3-track path, not lip-sync identity for a third talker.
  - Second speaker pair: no real recording has clean per-speaker references. `pair2_*` (demo voice1/voice2) is
    synthetic and audio-only, so it is not in the video-based Row 0.
  - Not yet run: the real AV clip `stkM` through the Row 0 path (it has no clean references, so it can only be
    scored for attribution/refusal, not SI-SDRi).
  - Phase 3 modules merged but mocked/synthetic only: slice 1 activity gate, slice 2 off-screen rule, slice 3b
    residual peeling, 4a identity windows, 5 pyannote adapter (mocked, no token), 6a/6b non-speech attribution,
    6c Nyquist guard (activity-gated: a tone overlapping active speech is not reported). 4b was already wired.

  **Phase 3 ablation rows vs Row 0** (2026-10-25, same runner and cases, `--tse` plus one switch;
  `hub/sessions/av_matrix/abl_{gate,offscreen,peel,all}/`; GPU hardware-run, SYNTHETIC mixtures; first attempts
  crashed/predated review fixes and were re-run on the fixed code):

  | row | change vs Row 0 | cost (mean wall / case; Row 0 5.9 s) |
  |---|---|---|
  | `--peel` (3b) | overlap3 A: refused -> `ok` +15.9 dB (matched to peeled `voice_2`); B: `degraded` -> `ok`. 2-talker cases identical. Adds an unmatched 4th stream (`voice_3`). | 6.7 s |
  | `--offscreen` (2) | no target status change. Off-screen V3 voice: `unknown` -> `outside_view` (closes the "overlapping off-screen voice stays `unknown`" limitation); both off-screen cases still `outside_view`. A voice is off-screen only if no visible face's TSE output, including faces TalkNet called silent, resembles it. | 7.9 s (extra TSE probes) |
  | `--gate` (1) | no status change; reported activity +0.4 s in the overlap cases (TalkNet misses rescued). Silent-bystander statuses unchanged, but the silent case shows 0.8 s of rescued activity (< the 1.0 s `MIN_ACTIVE_S` line), the off-screen case 0.36 s. | 7.8 s |
  | all three | union of the above | 8.5 s |

  - `overlap3` C stays refused (its tile's lips are not synced to its audio; a fixture artifact).
  - 4b (off-axis yaw limit): already wired through `cfg.off_axis.yaw_limit_deg`; tests only, no ablation row.
  - 4a, 5, 6a/6b, 6c: library modules with mocked/synthetic tests, not attached to a pipeline stage, so no row.

  **Real clip `stkM`** (`hub/sessions/{row0,gate,offscreen,peel,all}_stkM/`; replayed re-encode of
  `stack_man_top.mp4`, 6 s, 2 people stacked; `phase2 --tse` plus switch; no clean references, so this is attribution
  only, not SI-SDRi). Ground truth from the user: both people speak. Track `.0` = bottom, `.1` = top.

  | config | `voice_0` (second voice) | `voice_1` |
  |---|---|---|
  | baseline | `unmatched`, `unknown` | `.0`, matched |
  | `--gate` | **`.1` matched (correct)** | `.0`, matched |
  | `--offscreen` (fixed) | `unmatched`, `unknown` (the first version wrongly said `outside_view`: fixed by probing TalkNet-silent faces) | `.0`, matched |
  | `--peel` | unchanged | `.0`, matched |
  | all three | `.1` matched | `.0`, matched |

  - The gate recovers the second talker that TalkNet misses; this is the first evidence on a real clip.
  - Caveat: one clip, one run each; the silent-bystander side of the gate is only tested on the synthetic silent
    case (0.8 s of rescued activity there, under the 1 s line).

## AV Phase 4: long clips, chunked pipeline, inspection output (branch `feat/av-phase4-long-clips`, PR #58)

`python -m av_scene run <media> --out DIR [--cache DIR]` runs one chunked offline pipeline (visual, speech, optional
diarization, events, residual, export). It is NOT real time and makes no device claim. Details and decisions:
`.agents/decisions.md` ("AV long clips, Phase 4").

Evidence labels: **mocked** = CPU fakes through the real pipeline code; **replayed / hardware-tested** = real models on the
RTX 5070 Ti Laptop; **untested** = not run.

| Gate | Result | Evidence |
|---|---|---|
| Chunk coverage, no missing or duplicated sample (incl. final partial region) | pass, 11 durations | mocked |
| Identity survives pauses, crossings, re-entry, slot permutation every chunk | 0 switches, 0 fragmentation, 0 wrong-person on 7 deterministic scenes (60 s pause_return, crossing, reentry, silent_overlap, slot_swap_every_chunk, final_partial, silent_bystander) | mocked |
| Silent-bystander highlight rate (< 2%) | 0.0 to 0.2% on the synthetic scenes. Whole-chunk status first gave 6 to 20%; segments are now split at the speaking activity | mocked |
| Active-speaker recall and refusal precision/recall reported together | recall >= 0.97; reentry refusal P 0.91 / R 1.0; a refuse-everything prediction fails the gate | mocked |
| Forced termination at 6 boundaries resumes bit-identically; corrupt entry recomputes only itself; model-revision change reruns speech only | pass | mocked |
| Warm cache invokes no model and loads no checkpoint | pass (`model_passes_opened` all 0) | mocked |
| Continuous events bounded (windows <= `event_max_window_s`, candidates <= 4, batch 2, frames <= 150) | pass | mocked |
| Missing control / missing spatial evidence never "matched"; two-cat ambiguity kept; loud leaked audio for an absent target rejected | pass | mocked |
| Memory, 30/120/300 s, pipeline plumbing only | peak RSS 645 / 654 / 659 MiB, growth 13.7 MiB (budget 256), slope -63 MiB/audio-min (budget 2) | mocked (fake models; says nothing about model memory) |
| Viewer in a real browser (Chrome) | listen groups, four separate badges + refusal text, focus kept on toggle, aria-live, diagnostic-vs-default muting, play/pause/seek, windowed fetches (about 4 s back / 10 s ahead), no `_` paths served. One defect found and fixed (sources with no audio were not listed) | synthetic bundle, real browser; audio quality not judged |

### Real-model smoke run (hardware-tested, one short clip)
`stkM` (6 s, 2 people stacked, both speak), defaults, no Phase 3 switches, RTX 5070 Ti Laptop:
- All passes ran on real S3FD, SAM 3, TalkNet, MossFormer2, AV-TSE and WavLM-SV. Events pass loaded no model (no event).
- Peak GPU 3.97 GiB (budget 9.0 GiB: pass). Visual 80 s for 150 frames (SAM 3 per frame), speech 13 s.
- **Peak RSS 7.69 GiB: exceeds the pre-set 6.0 GiB budget (FAIL).** It is a transient at about 9 s (SAM 3 checkpoint load);
  steady state is about 2.4 GiB, and it does not depend on clip length. A first attempt peaked at 14.9 GiB because
  event detection loaded SAM 3 and SAM Audio; fixed (AST detection is now separate, heavy event models load only when a
  window computes).
- Quality is not good enough to claim anything: SAM 3 produced 6 person tracks for 2 people, `src_1` was matched to a
  track in 3 of 6 spans, `src_0` stayed unmatched (as in the Phase 3 baseline; the activity gate was off). No listening review.

### Not done (untested)
- Real-model 30 / 120 / 300 s workloads: the looped-clip run (`--loop-seconds`, duration-matched only, content repeats)
  was started and stopped by the user's time limit before the 30 s case finished. No real long-run memory, identity
  switches/minute or cold / warm / resume timing exist.
- Additional real clip with visible silent people, annotated (silent-bystander gate on real data, `<2%`): no such clip exists
  locally (longest recordings are 6 to 8 s; none has a visibly silent person). Annotation format and scorer are ready
  (`eval/av_scene/LONG_ANNOTATIONS.md`, `long_annotations.py`).
- `stkM` missed-speaker regression through the Phase 4 path with `--activity-gate`; listening review of stitched boundaries
  and accepted / refused stems.
- Non-speech on a real clip (SAM 3 nouns, SAM Audio, judge / CLAP ranking): adapters written, never exercised on a real
  event. The pipeline passes no calibrated array, so non-speech attribution is expected to stay unmatched.
- Diarization on real audio (pyannote not installed; stays OFF). `TorchcodecSource` works on `stkM`; other containers untested.
- Gate floors in `long_metrics` (0.80 recall / refusal, 0.05 wrong-person) are provisional placeholders; only `<2%` is from the spec.

Worktree note: gitignored checkpoints and FFmpeg DLLs live in the main checkout; a worktree needs
`COCKTAIL_FFMPEG_BIN=<main>/hub/.local/ffmpeg/bin` and a directory junction for `hub/checkpoints`.


## AV Phase 5: live selected-voice refinement bridge (branch `feat/av-phase5-live-refine`, PR #62; base `feat/av-phase4-long-clips`)

`python -m refine [--refine]` is the live hub (refinement default OFF) (same composition as `separate/__main__.py`) plus `hub/refine/`: when a source is chosen, a
bounded job (S3FD faces -> TalkNet -> AV-TSE, matched against the HUB slot audio; no SAM 3, no second MossFormer2) decides whether
the chosen voice belongs to a visible face. It publishes a verdict only: it never changes selection, route, `/audio/out` audio
or `sources.updated` (`audio_applied` is always false: measured job latency is 4.7-6.2 s, longer than the playback queue).

States: `pending`, `accepted`, `refused`, `stale`, `unavailable` on `GET /refinement/status`. On `/events` the existing
`TargetUpdated` carries `kind=person, assoc_conf>0` only for `accepted`; the other states are indistinguishable there
(`kind=speech, assoc_conf=0`); stale is never published. Protocol proposal (NOT applied; frozen): an optional
`refinement_state` field on `TargetUpdated`.

Evidence labels: **mocked** = fakes; **replayed** = recorded/synthetic fixtures streamed into a real hub process by the mock
client; **live** = real hub process with real models (RTX 5070 Ti Laptop); no headset was used.

| Check | Result | Evidence |
|---|---|---|
| Stale epoch / selection / audio-reset / route / age / source-gone rejection, selection changed between completion and publish | pass (`tests/refine/test_bridge.py`, `test_install.py`, `test_ids_bridge.py`) | mocked |
| Safe fallback: slow / raising / refusing / never-finishing runner leaves `/audio/out` frames, seq, `sources.updated` and underruns identical | pass (`test_fallback.py`) | mocked |
| Real jobs through the live hub: stale results observed and rejected (`route moved`, `route or selection changed inside the window`) | observed in 4 runs, never published as current | replayed + live models |
| Correct-match: stkM_x3 (18 s, 3x repeat of the 6 s clip), selected voice | `accepted`: matches visible face `f0`, TSE vs slot SI-SDR 17.5 dB, assoc_conf 0.87, latency 5.45 s, `TargetUpdated kind=person` | replayed + live models, looped content |
| stkM unmatched voice stays refused | NOT shown: the select-all run only ever selected one source (stale), so the second voice was not exercised through the live path | untested live |
| Memory (whole hub process tree, live separator + voice memory + refinement models + CUDA) | peak host RSS 3.2-3.7 GiB (budget 6.0: pass); GPU total peak 1.4-2.0 GiB over a 0.5 GiB desktop baseline | live |
| Audio continuity with jobs running | seq breaks 0, ts regressions 0, hops_dropped 0, separator lag <= 0.1 s; BUT output underruns 4-8 with a job running vs 2-3 baseline (longest gap 0.30 s vs 0.14-0.18 s) | live; NOT mitigated. Refinement is therefore opt-in: `python -m refine --refine` (default off). A mitigation attempt (frame subsampling, yielding, pacing) was started and stopped unfinished |

Not done / unverified: real long-clip, silent-bystander, listening review, non-speech and diarization gates (as scoped);
fixtures are single-mic mono duplicated to 2 channels, so bearings are meaningless and sources are announced only through the
identity rules (voice memory on); the Quest 3S real frame rate and the real mic/camera alignment are untested; PR #59's
identity rules are simulated/mocked only (see decisions).
