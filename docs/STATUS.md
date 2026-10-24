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
- GPU: the team has an RTX 5070 Ti machine (16 GB per spec; confirm with `nvidia-smi`).
  Make it the live-run machine. It is Blackwell, so install PyTorch built for CUDA 12.8+
  (cu128 wheels). Hub laptop (RTX 4060, 8 GB) is for development. Still to measure:
  do separation, SAM 3, enhancement and SAM Audio fit together on 16 GB (A3)?
