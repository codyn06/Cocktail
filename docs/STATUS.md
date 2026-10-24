# Cocktail status board

Shared progress for everyone on every computer. **Edit only your own row** (one row per
task, so git merges cleanly). Pull before you start; push after you finish a task.
Plan source of truth: the Quest PRD (sections 14 to 16). Statuses: `todo`, `doing`,
`done`, `blocked`. Evidence: hardware-tested, mocked, replayed, untested.

Folder rule (PRD section 14): one owner per folder.
`hub/protocol` is frozen; change it only with the integration owner's agreement and
re-run `cd hub && uv run python -m protocol.export`.

| Task | What | Owner | Branch | Status | Evidence / notes |
|---|---|---|---|---|---|
| A1 | Repo, run scripts, layout | | feat/a1-a2-repo-protocol | doing | |
| A2 | Protocol + target record schema | | feat/a1-a2-repo-protocol | doing | Python + JSON Schema + generated C# |
| A3 | GPU setup + benchmark all models together | | | todo | |
| A4 | Unity setup, 3S OS version | | | todo | repo pins 6000.0.80f1 |
| A5 | MR UI on Quest 2 with fake data | | | todo | |
| A6 | Audio models + scoring harness | | feat/a6-audio-models | doing | agent running |
| A7 | Alerts engine + name keyword spotting | | | todo | |
| A8 | Locating algorithm (head movement) | | feat/a8-locate | doing | agent running; uses published math |
| A9 | Pitch / demo outline | | | todo | |
| A10 | Private hotspot network | | | todo | |
| B1 | Quest audio + pose streaming | | | todo | report mic channels, raw access |
| B2 | 3S camera streaming | | | todo | |
| B3 | Hub ingest | | | todo | |
| B4 | Mock headset client | | | todo | |
| B5 | Selection input | | | todo | |
| B6 | 3D audio playback | | | todo | |
| C1 | Loopback delay test | | | todo | |
| C2 | Test recording on 3S | | | todo | |
| C3 | Live separation + labels | | | todo | |
| C4 | Live locating | | | todo | |
| C5 | Camera confirmation (SAM 3, depth) | | | todo | |
| C6 | Live UI | | | todo | |
| D1 | Tracking + target record | | | todo | |
| D2 | Alerts live | | | todo | |
| D3 | Clean replay (SAM Audio) | | | todo | |
| E1 | Select and hear | | | todo | |
| E2 | Who is talking | | | todo | |
| E3 | Display tracked sources | | | todo | |
| F1 | Three modes | | | todo | |
| F2 | Acceptance test | | | todo | |
| G1-G4 | Metrics, echo tuning, glasses mode, README | | | todo | |
| H1-H2 | Film and submit before 10 AM | | | todo | |

## Flexible tasks
- Owners are defaults, not walls. Anyone with free time can take a `todo` row: put your
  name and branch in the row and set `doing` (pull first, push right away).
- Taking over a `doing` row: ask the current owner in person first, then update the row.
- Stuck for 20 minutes? Set the row to `blocked`, say why in Notes, and tell the team.
  Someone else may pick it up.
- Each task has its own folder and branch, so switching owners never means moving code.
  Only `hub/protocol` stays gated (integration owner approves changes).

## Demo timing
Demo work (A9 pitch, H1 filming, G4 README) starts only after the product is finished
(Checkpoint 1 working on the 3S). Until then everyone builds the product.

## Machines
| Machine | Person | Hardware / notes |
|---|---|---|
| Cody's | Cody (codyn06) | RTX 5070 Ti: live-run GPU machine (hub + models) |
| Lalitha's | Lalitha | RTX 4060 8 GB, Unity installed, Claude $100 plan: Quest app + dev |
| Third | unassigned | |

## Open decisions / blockers
- Integration owner: unassigned.
- Unity: repo pins 6000.0.80f1; hub laptop has 6000.6.7f1.
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

  C3 candidate benchmark (hardware-tested on the 5070 Ti, 2026-10-24, torch 2.11+cu128, fp32,
  **synthetic input: speed/VRAM only, separation quality not measured yet**):

  | Model | Task | Peak VRAM | Latency (p50) | RTF |
  |---|---|---|---|---|
  | GTCRN (MIT) | speech enhancement, 16 kHz | 30 MiB | ~7 ms per call (64 ms - 2 s windows) | 0.003-0.13 |
  | Waveformer (MIT) | streaming target-sound extraction, 44.1 kHz, 41 classes | 66 MiB | 6.5 ms per 9.4 ms hop | 0.69 |
  | MossFormer2_SS_16K (Apache-2.0) | 2-speaker separation, 16 kHz, non-streaming | 714 MiB | 141 ms / 1 s window, 266 ms / 2 s | ~0.13 |

  Waveformer's classes include instruments but no speech or generic "music" class, so it is a
  weak fit for the 2 talkers + music test. MossFormer2 latency = window length + compute.
