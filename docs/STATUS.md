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

## Machines
| Machine | Person | Hardware / notes |
|---|---|---|
| Cody's | Cody (codyn06) | RTX 5070 Ti Laptop, 12 GB (Blackwell: use PyTorch cu128). Live-run GPU machine (hub + models) |
| Lalitha's | Lalitha | RTX 4060 8 GB, Unity installed, Claude $100 plan |
| Third | unassigned | |

## Milestones
Target: working product by 10 AM tomorrow (Sunday). This is the best-case target, not a
hard stop: the demo video is separate, and the deadline can be extended as the time
approaches. Don't cut corners or rush to hit it; ask the team first. Each milestone is done when its
"proof" works; move on as soon as it does. Everything after M3 only improves and proves it.

| # | Milestone | Proof it's done | Tasks |
|---|---|---|---|
| M0 | Everyone can build and run | Every machine pulled, models fit on the GPU, headset connects over the private network | A1, A3, A4, A10 |
| M1 | Data flows end to end | Headset audio, video and head pose reach the hub; sound returns to the headset; delay measured; a recorded test scene exists | B1, B2, B3, B4, B6, C1, C2 |
| M2 | Sounds separated and located (main goal) | On the test recording, each sound is its own labeled source with a direction | A6, A8, C3, C4, C5 |
| M3 | **Checkpoint 1: working product** | On the 3S, sounds are tracked live on screen; pick one and hear it alone in 3D | D1, B5, C6, E1, E3 |
| M4 | Complete feature set | Alerts, clean replay, who-is-talking, three modes; acceptance test passes | A7, D2, D3, E2, F1, F2 |
| M5 | Proven and submitted | Metrics (30 trials each), tuning, README, video, submitted before 10 AM | G1-G4, A9, H1, H2 |

## Tasks

**Main goal: split every sound into its own source and track what it is and where it
is (marked ★).** Tasks are listed in the order they unlock each other. "Needs" means
the earlier tasks that must be usable first; mocks and recordings let you start sooner.

### Setup and foundations (start now, no dependencies)

| ID | Task | What it means / done when | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|
| A1 | Repo and run scripts | Public repo with MIT license, README skeleton, one-command scripts: run the hub, replay a recorded session, deploy to the headset, run the evaluation. | repo root, `scripts/` | | | todo | |
| A2 | Message format between headset and laptop | The frozen set of messages (audio, video, head pose, button actions, events back) with generated Python and C# code. | `hub/protocol` | | feat/a1-a2-repo-protocol | done | merged in PR #10; mocked (5 tests) |
| A3 | Check all the AI models fit on one GPU | Run separation, speech enhancement, SAM 3 (object outlines), depth and SAM Audio together on the RTX 5070 Ti. Done when we know which ones fit, how fast, and how much memory. | `eval/` | Cody | chore/gpu-env | doing | PR #13 merged: cu128 env, 12 GB budget, C3 candidates benchmarked (hardware-tested). SAM 3 image measured (see Open decisions). Left: SAM 3.1 tracker, SAM Audio small/small-tv |
| A4 | Unity and headset ready | Unity with the Meta XR and Meta XR Audio SDKs builds and runs hello-world on the Quest 2 and Quest 3S. Confirm the 3S software version supports the camera API (v74+). | `quest/` | | | todo | repo pins Unity 6000.0.80f1 |
| A5 | Headset screen design with fake data | Labels and outlines on each sound in view, edge arrows and clock positions (like "4 o'clock") for sounds out of view, state banners, a small glasses-style view. Runs on the Quest 2 with made-up sources. | `quest/ui` | | | todo | |
| A6 | ★ Choose the audio AI models and build the scoring test | Pick the model that splits mixed audio into voices, the sound labeler, the voice-ID model and the speech enhancer. Build a test that mixes known clean recordings and scores the result. | `hub/separate`, `eval/` | Cody | feat/c3-live-separation | done (simulated) | PR #19 merged. MossFormer2 (voices) + residual + AST labels; scoring test `python -m separate.score` |
| A7 | Alerts engine | Rules file that matches sounds (smoke alarm, siren) and the wearer's name, with a name-detection model. Tested with the laptop mic. | `hub/alerts` | | | todo | |
| A8 | ★ Work out where a sound is from head movement | Math that turns "this sound got louder and brighter as the head turned" into a direction, all 360 degrees, in view or not. Tested on simulated audio and head turns. | `hub/locate` | | feat/a8-locate | done (simulated) | PR #11; needs per-mic levels per source from separation |
| A9 | Pitch and demo outline | Outline of the video and slides. Start after the product works. | `docs/` | | | todo | later |
| A10 | Network | Laptop and headset on the same network. University Wi-Fi is good enough for now; a private hotspot is later and only if delay or jitter becomes a problem. | none | | | todo | low priority, later |

### Getting data flowing

| ID | Task | What it means / done when | Needs | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| B1 | Headset sends microphone sound and head pose | Live mic audio and head position stream to the laptop. Report how many mic channels there are and whether the audio is raw. Add a native plugin if Unity's mic is too processed. | A2, A4, A10 | `quest/stream` | | | todo | |
| B2 | Headset sends camera video | Quest 3S passthrough camera frames compressed on the headset and streamed. Report frame rate, size and delay. | A2, A4, A10 | `quest/stream` | | | todo | |
| B3 | Laptop hub receives everything | Hub accepts audio, video and pose on separate channels so big video never delays audio; lines up their clocks. | A2, A3 | `hub/ingest` | Cody | feat/hub-ingest-mock | done (simulated) | PR #15 merged; last 10 s audio (16 kHz) + pose, newest video frame. Assumes chunk ts = first sample; confirm with B1 |
| B4 | Fake headset for testing | A program that plays a recorded or simulated session into the hub, so nobody needs the headset to test. | A2 | `hub/mockclient` | Cody | feat/hub-ingest-mock | done (simulated) | PR #15 merged; simulator (2 mics, head turns) + real-time replayer. See README "Hub (laptop)" |
| B5 | Choosing a sound | Look at it or point with hand or controller, pinch to confirm. Actions: select, stop, replay, next. | A4, A5 | `quest/input` | | | todo | |
| B6 | Play sound in 3D with two layers | Play the chosen sound at its direction, plus the rest of the room quieter. Separate volumes. | A4 | `quest/audio` | | | todo | |

### Making it work for real

| ID | Task | What it means / done when | Needs | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| C1 | Round-trip delay test | Send mic sound to the hub and straight back, unprocessed. Measure delay and jitter. | B1, B3 | `eval/` | | | todo | quick test |
| C2 | Record the test scene | On the 3S: two people talking at once plus music, with natural head movement. Everyone tests on it. | B1, B2 | `mock/` | | | todo | needs the headset |
| C3 | ★ Live sound separation and labels | Hub splits the live mix into separate voices and sounds and names each (speech, music, alarm). Tested on the fake headset, then on the recording. | A6, B3 | `hub/separate` | Cody | feat/c3-live-separation | done (simulated) | PR #19 merged. +18.3/+17.1/+17.9 dB on simulated 2 talkers + music, labels correct, ~460 ms per 1 s hop, ~1.1 GB VRAM. Not yet on the 3S recording |
| C4 | ★ Live direction finding | Hub gives every separated sound a direction using the head-movement math. | A8, B3 | `hub/locate` | Cody | feat/c4-live-locate | doing | PR #21, waits on #11. Simulated errors 40/9/16 deg; estimator overconfident (see #11 comment) |
| C5 | Camera confirmation | SAM 3 outlines objects in the camera view and depth gives distance, to confirm which object makes a sound. | A3, B2 | `hub/vision` | | | todo | |
| C6 | Live headset screen | The designed screen driven by real hub events. | A5, B1, B3 | `quest/ui` | | | todo | |

### Tracking, selecting, hearing

| ID | Task | What it means / done when | Needs | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| D1 | ★ Track every sound | Each sound keeps one identity, direction and state (tracking, last seen, lost), in view or out of view. | C3, C4 | `hub/track` | | | todo | |
| D2 | Alerts that interrupt playback | A smoke alarm, siren or the wearer's name lowers the playback and speaks with direction. | A7, C3, B6 | `hub/alerts` | | | todo | |
| D3 | Clean replay of the last 5 seconds | SAM Audio re-processes the last 5 s of the chosen sound and plays it back clean. | A3, B3, B6, C3 | `hub/replay` | | | todo | |
| E1 | Choose a sound and hear it | The chosen person or sound is amplified and enhanced over a quieter room. A voice fingerprint keeps hold of them. | D1, B5, B6, A6 | `hub/separate`, `quest/audio` | | | todo | |
| E2 | Who is talking | Direction decides which voice belongs to which person; lip and body motion confirm. | D1, C5 | `hub/who` | | | todo | needs the headset |
| E3 | Show every tracked sound | Labels and arrows for every tracked sound on the headset. | D1, C6 | `quest/ui` | | | todo | |

**Checkpoint 1 (target tonight): E1 + E3 working live on the Quest 3S with D1 tracking.**

### Finishing

| ID | Task | What it means / done when | Needs | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| F1 | Three modes | Fast (no lookahead), Balanced (~100 ms lookahead), Accurate (phrase-level, played in pauses). Switchable live. | E1 | `hub/modes` | | | todo | |
| F2 | Full acceptance test | Two people talking plus music: select A, then B, then the music; turn away and keep tracking; alerts fire; replay works. | E1, E2, E3, D2, D3 | all | | | todo | |
| G1 | Measure everything | At least 30 trials each: separation gain, wrong-speaker rate, direction error, delay per mode, switch time, false alerts. | F2, A6 | `eval/` | | | todo | |
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
  - SAM 3.1 video tracker on Windows: needs `use_fa3=False` (no FlashAttention 3), a workaround
    for an upstream bug (`start_session` passes `offload_state_to_cpu`, which the 3.1 tracker
    rejects), and allowing non-flash attention in `sam3/model/decoder.py` (it forces the flash
    kernel, which Windows PyTorch lacks). Not measured yet.
  - SAM Audio small: first load also downloads ~16 GB of companion models (`sam-audio-judge`
    12.3 GB, CLAP 1.8 GB, T5/RoBERTa/BART/BERT ~2.3 GB) on top of the 5.1 GB checkpoint. Needs
    `transformers` 4.x (>=4.54) with `huggingface_hub` < 1.0. VRAM not measured yet. If the 6.1 GB
    judge stays resident, D3 cannot fit in 12 GB with it.
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
