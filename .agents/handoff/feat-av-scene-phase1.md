# feat/av-scene-phase1
Status: ready for review (reviewer pass applied) (2026-10-25)
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
- final `uv run pytest` in hub/ -> 180 passed, 1 skipped (MP4 mux test needs ffmpeg) (mocked/synthetic)
- `COCKTAIL_FFMPEG=<imageio ffmpeg> pytest tests/av_scene -k mp4` -> passed (annotated.mp4 + source_video.mp4)
- `python -m av_scene fixture|inspect|preflight|serve` -> run
- viewer driven in headless Edge through DevTools (Claude-in-Chrome extension was not connected): seek, click-to-select,
  exclusive/additive solo, mute, residual, original-mixture, seek-while-playing, reset; no external requests, no console errors
- `node .agents/scripts/sync.mjs --check` -> OK

## Reviewer findings applied
- viewer <video> mode now plays and corrects drift at 45 ms (measured ~20 ms drift in headless Edge with an MP4 bundle)
- off_axis.yaw_limit_deg, calibration camera/rotation/parallax/channel noise now drive the pipeline; dead `elig` removed
- stage keys are content-addressed and chained over all fixture inputs; code_revision marks +dirty
- remove_stem no longer crashes (tested through the pipeline); read_png filter wrap fixed (tested, all 5 filters); concat paths quoted
- envelope segment end back-tracked like the onset; directory listing disabled; alpha_max limitation documented + tested

## Known gaps (deferred, not claimed)
- Off-axis DRR/high-band uncertainty widening (spec 5.5.4) and SpatialEvidence.drr_confidence are not implemented.
- `rejected` is a per-candidate label only; sources are matched/ambiguous/unmatched. Unmatched Decision.score is the best losing candidate.
- EligibilityGatedVoiceMemory is unit-tested but not called by the fixture pipeline (no voice encoder in Phase 1).
- StageCache is unit-tested; the pipeline records chained keys but does not skip stages on a hit yet.
- Viewer ignores FrameTransform / stream transforms (fine for the fixture), AudioContext output latency, reduced motion/glow,
  and re-renders the source list on each action (focus loss). Server has no Host-header check (localhost, synthetic data only).
- Metrics are partly tautological on an oracle fixture (mask onset follows the audio gate); only omitted-source recall is an independent check.
- alpha_max can absorb a quieter missing source overlapping an included stem's bins.

## Next steps
- Reviewer pass; fix findings; mark PR ready; at merge prep move decisions to .agents/decisions.md and delete this file.
- Phase 2 (real SAM3 / SAM Audio / preflight / mask polarity probe) is out of scope here.
