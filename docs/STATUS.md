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
| Cody's | Cody (codyn06) | RTX 5070 Ti, 16 GB (Blackwell: use PyTorch cu128). Live-run GPU machine (hub + models) |
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
| A3 | Check all the AI models fit on one GPU | Run separation, speech enhancement, SAM 3 (object outlines), depth and SAM Audio together on the RTX 5070 Ti. Done when we know which ones fit, how fast, and how much memory. | `eval/` | | | todo | |
| A4 | Unity and headset ready | Unity with the Meta XR and Meta XR Audio SDKs builds and runs hello-world on the Quest 2 and Quest 3S. Confirm the 3S software version supports the camera API (v74+). | `quest/` | | | todo | repo pins Unity 6000.0.80f1 |
| A5 | Headset screen design with fake data | Labels and outlines on each sound in view, edge arrows and clock positions (like "4 o'clock") for sounds out of view, state banners, a small glasses-style view. Runs on the Quest 2 with made-up sources. | `quest/ui` | | | todo | |
| A6 | ★ Choose the audio AI models and build the scoring test | Pick the model that splits mixed audio into voices, the sound labeler, the voice-ID model and the speech enhancer. Build a test that mixes known clean recordings and scores the result. | `hub/separate`, `eval/` | | feat/a6-audio-models | doing | agent running; not merged |
| A7 | Alerts engine | Rules file that matches sounds (smoke alarm, siren) and the wearer's name, with a name-detection model. Tested with the laptop mic. | `hub/alerts` | | | todo | |
| A8 | ★ Work out where a sound is from head movement | Math that turns "this sound got louder and brighter as the head turned" into a direction, all 360 degrees, in view or not. Tested on simulated audio and head turns. | `hub/locate` | | feat/a8-locate | done (simulated) | PR #11; needs per-mic levels per source from separation |
| A9 | Pitch and demo outline | Outline of the video and slides. Start after the product works. | `docs/` | | | todo | later |
| A10 | Private network | Laptop hotspot or phone hotspot that the laptop and headset both join (no venue Wi-Fi). | none | | | todo | |

### Getting data flowing

| ID | Task | What it means / done when | Needs | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| B1 | Headset sends microphone sound and head pose | Live mic audio and head position stream to the laptop. Report how many mic channels there are and whether the audio is raw. Add a native plugin if Unity's mic is too processed. | A2, A4, A10 | `quest/stream` | | | todo | |
| B2 | Headset sends camera video | Quest 3S passthrough camera frames compressed on the headset and streamed. Report frame rate, size and delay. | A2, A4, A10 | `quest/stream` | | | todo | |
| B3 | Laptop hub receives everything | Hub accepts audio, video and pose on separate channels so big video never delays audio; lines up their clocks. | A2, A3 | `hub/ingest` | | | todo | |
| B4 | Fake headset for testing | A program that plays a recorded or simulated session into the hub, so nobody needs the headset to test. | A2 | `hub/ingest`, `mock/` | | | todo | |
| B5 | Choosing a sound | Look at it or point with hand or controller, pinch to confirm. Actions: select, stop, replay, next. | A4, A5 | `quest/input` | | | todo | |
| B6 | Play sound in 3D with two layers | Play the chosen sound at its direction, plus the rest of the room quieter. Separate volumes. | A4 | `quest/audio` | | | todo | |

### Making it work for real

| ID | Task | What it means / done when | Needs | Folder | Owner | Branch | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| C1 | Round-trip delay test | Send mic sound to the hub and straight back, unprocessed. Measure delay and jitter. | B1, B3 | `eval/` | | | todo | quick test |
| C2 | Record the test scene | On the 3S: two people talking at once plus music, with natural head movement. Everyone tests on it. | B1, B2 | `mock/` | | | todo | needs the headset |
| C3 | ★ Live sound separation and labels | Hub splits the live mix into separate voices and sounds and names each (speech, music, alarm). Tested on the fake headset, then on the recording. | A6, B3 | `hub/separate` | | | todo | |
| C4 | ★ Live direction finding | Hub gives every separated sound a direction using the head-movement math. | A8, B3 | `hub/locate` | | | todo | |
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
- GPU: do separation, SAM 3, enhancement and SAM Audio fit together on 16 GB (task A3)?
