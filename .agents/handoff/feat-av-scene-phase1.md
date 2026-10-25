# feat/av-scene-phase1
Status: ready for review pending reviewer pass (2026-10-25)
Owner: codyn06 / Claude Code orchestrator   PR: draft (see `gh pr list`)

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
- No new dependencies; no ffmpeg on PATH on this machine, so MP4 output is optional and discovered via config/env
  (COCKTAIL_FFMPEG / FFMPEG_BINARY / PATH / imageio_ffmpeg). Verified once with the uv-cache imageio ffmpeg 7.1 (libx264).
- Projection residual uses amplitude-ratio masks (mask_exponent=1): mask-derived stems are coherent partitions of the
  mixture, so the power-subtraction form over-reports (2 false events on the fixture). Both bases are reported.
- Frames are sampled at their display midpoint so onset quantization is at most q = half the local interval.
- Fixture stems are ORACLE Wiener masks from truth magnitudes; labels come from fixture truth. Evidence tier mocked/synthetic everywhere.
- Zero-width-space scan of the spec found none.

## Verification
- baseline `uv run pytest` in hub/ -> 87 passed (before changes)
- final `uv run pytest` in hub/ -> 175 passed, 1 skipped (MP4 mux test needs ffmpeg) (mocked/synthetic)
- `COCKTAIL_FFMPEG=<imageio ffmpeg> pytest tests/av_scene -k mp4` -> passed (annotated.mp4 + source_video.mp4)
- `python -m av_scene fixture|inspect|preflight|serve` -> run
- viewer driven in headless Edge through DevTools (Claude-in-Chrome extension was not connected): seek, click-to-select,
  exclusive/additive solo, mute, residual, original-mixture, seek-while-playing, reset; no external requests, no console errors
- `node .agents/scripts/sync.mjs --check` -> OK

## Next steps
- Reviewer pass; fix findings; mark PR ready; at merge prep move decisions to .agents/decisions.md and delete this file.
- Phase 2 (real SAM3 / SAM Audio / preflight / mask polarity probe) is out of scope here.
