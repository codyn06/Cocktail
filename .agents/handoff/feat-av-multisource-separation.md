# feat/av-multisource-separation

**Status:** draft PR. The independent review is running and the final speech eval rerun is pending. The PR targets `feat/av-phase2-models`.

## What exists now
- **Speech branch (`hub/av_scene/speech.py`, `phase2 --speech-backend mossformer2`, which is the CLI default):**
  1. MossFormer2_SS separates the audio into 2 voices.
  2. S3FD face boxes are assigned to SAM 3 person masks.
  3. TalkNet speaking probability gives each person's activity.
  4. AV_MossFormer2_TSE runs for each active person, i.e. at least min(1 s, 25% of the window) of activity.
  5. Each TSE output is matched to the separated voice it resembles, with SI-SDR ≥ 3 dB and a 3 dB margin.
  6. If two people claim the same voice, the more active person keeps it and the other gets `unavailable`.
  7. A person who is not active, or who matches no voice, gets `unavailable`, never another person's voice.
- **Fallback without TSE:** correlation association (`associate.py`). It is measured as weak (see below).
- **Non-speech (`hub/av_scene/events.py`):** AST windowed labels with SAM Audio prompts. Visual nouns are candidates only, never attribution.
- **Schemas:** `SceneSource.label_confidence`, `activity_us`, and `extraction` (an `Extraction` object).
- **Eval:**
  - `eval/av_scene/{mixtures,metrics,audit_inputs,run_matrix,run_speech,summarize}.py`
  - Outputs go to `hub/sessions/av_matrix/` (gitignored).

## Ownership
The orchestrator owns `schemas.py`, `phase2.py`, `speech.py` and the docs. The other modules came from the merged subagent branches.

## Decisions
- The user approved downloads. All of these are project-local and gitignored under `hub/checkpoints/`:
  - **AV_MossFormer2_TSE_16K:** `last_best_checkpoint.pt` (Apache-2.0)
  - **TalkNet-ASD:** source at `6d68214` (MIT) and `pretrain_TalkSet.model` (sha256 `d985ddd0…`)
  - **S3FD:** `sfd_face.pth` (sha256 `d54a87c2…`, 90 MB)
- The user approved more than 2 parallel agents (Sonnet, low effort).
- SAM Audio is not used to select a speaker. Its visual mask did not steer which voice it returned (see `docs/STATUS.md`). It stays for non-speech prompts.
- The person-mask head-crop heuristic was removed. It fed whole tiles to TalkNet.
- Envelope-correlation association was demoted to the no-TSE fallback. It rejected every overlap case and matched an off-screen voice to a silent person (`offscreen_only`).
- Docs number this decision D18, because D11 was already taken.

## Verification
- `uv run pytest tests/av_scene`: 175 passed.
- Speech eval is SYNTHETIC. See STATUS once the run lands in `sessions/av_matrix/speech3`.

## Next
- Fill the TBDs in STATUS from `speech3` and from the narrow SAM run (`run1_sam_narrow.log`).
- Address the reviewer findings.
- Do a listening check.
- Run the speech path on a recorded natural clip.
