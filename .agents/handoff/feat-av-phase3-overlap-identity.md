# feat/av-phase3-overlap-identity

Status: in progress (2026-10-25) - Step 0 merged, Wave 1 + Wave 2 pure modules merged (uncommitted), Row 0 running
Owner: Cody / Claude Code orchestrator   PR: not opened

## Ownership
- Orchestrator: `hub/av_scene/speech.py`, `associate.py` contracts, integration, GPU runs, docs.
- Subagent worktrees were merged into the working tree and removed.

## Decisions
- Frozen in Phase 3: `hub/protocol/messages.py`, `PairCalibration`, `VoiceMemory.reorder`.
- Downloads approved: wavlm, sam-audio-judge, CLAP, LR-ASD (gitignored `hub/checkpoints/`). pyannote: mocked until the user supplies an HF token.
- Row 0 replayed clip: an AV session from `hub/sessions/` with video (`stkM`; `real_uf3` has audio only). `hub/recordings/` is audio-only.
- `read_png` filters were already masked; the fix rejects filter > 4 (and masks filter 2).
- GPU timed runs require the user's go-ahead each time. Row 0 go-ahead given 2026-10-25.
- TSE path: `n_suspected_talkers` defaults to TalkNet-active tracks + unmatched active voices; above 2 the owner's extraction is `degraded`.
- 3-speaker separator: none usable via ClearVoice (scout: only `alibabasglab/mossformer2-wsj0mix-3spk`, 8 kHz, other loader, untested). Keep `MAX_SEPARABLE_TALKERS = 2`; use residual peeling (3b).
- Second speaker pair is SYNTHETIC audio-only (demo voice1/voice2), not a real second pair; no real recording has clean per-speaker refs.
- Slice 4b: yaw limit was already wired (`config.py` OffAxisConfig); tests only, no ablation row.
- Slice 6c: Nyquist guard is activity-gated; a tone overlapping active speech is not reported (limitation).
- Non-speech status vocabulary: real `AssociationStatus` is matched|ambiguous|unmatched|rejected; use unmatched + visibility.
- `.claude/settings.json` model opus->sonnet change is kept on purpose (user said not to revert).

## Verification
- Full suite on Step 0 + Wave 1: 438 passed (hub, COCKTAIL_FFMPEG_BIN set). Wave 2 files: 25 passed. All mocked/synthetic.
- Row 0 GPU run: in progress, output `hub/sessions/av_matrix/row0`.

## Next steps
- Record Row 0 in `docs/STATUS.md`; real AV clip `stkM` still needs a runner path (no refs/masks).
- Wire into `speech.py` behind default-off switches: slice 1 `activity_gate.combine_activity`, slice 2 `associate_streams(active_face_offscreen=True)`, 6c `stem_nyquist_hz`, 3b `peeling`, 4a, 5, 6.
- Ablation row per slice vs Row 0, independent review, then PR.
