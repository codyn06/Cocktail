# feat/av-scene-phase1
Status: in progress (2026-10-25)
Owner: codyn06 / Claude Code orchestrator   PR: not opened

## Ownership
- Spec cleanups -> orchestrator: docs/AV_SOUND_DISCOVERY_SPEC.md
- Phase 1 foundation -> orchestrator (serial, single worktree): hub/av_scene, hub/ingest/media_io.py,
  hub/vision, hub/separate/diarization.py, hub/locate/av_spatial.py, hub/who, hub/replay, hub/render,
  hub/tests/av_scene, eval/av_scene

## Decisions
- Skew gate is symmetric: abs(t_mask_onset - t_audio_onset) <= 20 + q; audio/mask vs visual truth stay on [-15-q, +45+q].
- aliasing_multimodal_flag is per calibrated pair; wdo_min_weight floors WDO weights before post-PHAT weighting.
- Off-axis profile-learning block is a wrapper (who/matcher.py) that restores VoiceMemory learning state for blocked
  slots; separate/voice_memory.py and its signatures are untouched.
- No new dependencies; no ffmpeg on PATH on this machine, so MP4 output is optional and discovered via config/env.
- Zero-width-space scan of the spec found none.

## Verification
- baseline `uv run pytest` in hub/ -> 87 passed (before changes)

## Next steps
- Implement modules per plan, then tests, fixture, bundle, viewer check, reviewer pass.
