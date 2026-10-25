# Team decisions

Durable, team-wide decisions only: stack choices, interface contracts, conventions.
Change this file through a PR approved by the team integration owner. Live task
status does not belong here (see `AGENTS.md` > Continuity).

## Integration owner
- Unassigned. Agree on one before parallel work on shared interfaces.

## Decisions
- No app stack chosen; no package.json; no dependencies.
- LLM: Gemma, run directly (not through the Gemini API). Use the vendored `gemma-dev` skill (google-gemma/gemma-skills); the
  `gemini-api-dev` skill was removed. Variant and runtime are undecided.
- Roles have no model names (inherit). Read-only roles may use bounded shell inspection;
  shell restrictions are instruction-enforced, including in Claude and Pi. Direct
  edit tools remain unavailable to those roles in Claude/Pi.
- Generated adapters are copies, not symlinks; the manifest + hash guard prevents
  overwriting hand edits.
- Project fallbacks cover unavailable metavr/TDD tooling without modifying vendored skills.
- Project-local dependency installs follow one policy; global installs and gated
  downloads still need approval.
- Task handoffs are per branch in `.agents/handoff/`; live task status is tracked in
  GitHub draft PRs.
- Voice memory (hub `LiveSeparator`, PR #39): optional speaker embeddings accumulated from clean active
  speech keep a separated voice's slot across >1s silence; if confidence/margin is insufficient, the
  prior matching behavior is kept. `LiveSeparator.step` stays synchronous and outputs, overlap-add
  tails and labels stay in stable logical slot order. No protocol or shared-interface change.
- Speaker embeddings (WavLM `microsoft/wavlm-base-plus-sv`, `Wav2Vec2FeatureExtractor` +
  `AutoModelForAudioXVector`) are validated on whole accumulated voices, not short probes; no checkpoint
  is packaged and the weights license scope is unclear. Same-speaker pause-return is covered only by a
  deterministic fake-encoder test, not real speaker labels.
- AV scene Phase 1 (hub `av_scene/`, PR #48, evidence tier mocked/synthetic): model-free CPU foundation; no new
  dependencies and no change to `protocol/messages.py`, `PairCalibration` or `VoiceMemory.reorder`. Sync gates:
  `delta_audio_ms`/`delta_mask_ms` vs `t_visual_truth` on `[-15-q, +45+q]`, `abs(t_mask_onset - t_audio_onset) <= 20 + q`
  (q = half the local frame interval; frames sampled at display midpoint). `aliasing_multimodal_flag` is per calibrated
  pair against `c/(2*d_ij)`; `wdo_min_weight` floors WDO weights before post-PHAT weighting.
- Off-axis (`abs(yaw) > 60 deg` or occluded) profile-learning block is `who.matcher.EligibilityGatedVoiceMemory`, a wrapper
  that restores `VoiceMemory` learning fields for blocked slots (bound to audio interval + logical slot); `voice_memory.py`
  is untouched. Unknown pose follows an explicit policy (`off_axis.unknown_pose`, default allow).
- Residual accounting uses amplitude-ratio masks (`mask_exponent=1`) because mask-derived stems are coherent partitions of the
  mixture; the power-subtraction form over-reports and is reported alongside. The raw waveform remainder is diagnostic only.
  `alpha_max` can absorb a quieter missing source that shares an included stem's bins.
- `assign` statuses are per source (matched/ambiguous/unmatched); `rejected` is a per-candidate label. MP4 output is optional
  and needs an ffmpeg found via config, `COCKTAIL_FFMPEG`/`FFMPEG_BINARY`, PATH or `imageio_ffmpeg`. Deferred to later phases:
  off-axis DRR widening, cache-hit stage skipping, viewer FrameTransform/output-latency handling, real models.

- `/audio/out` takes an optional `?layout=layers|stereo` query parameter (default `layers`, the Quest). `layers` frames are unity gain because the Quest applies inner/outer dB itself; the hub applies `SetVolumes` only for `stereo` clients (or with `--out-gains-in-layers`). Approved by Cody (PR #42); no change to `hub/protocol`.
- Chosen voice routing (hub `events/selection.py`, `out/`, PR #42): select/stop/next/prev arrive as `InputEvent` on `/ingest/pose`
  (ingest got `input_hooks`), not as `/events` commands. The choice is kept as a Tracker id, stable through slot swaps and
  pauses; voice memory is complementary, not the key. next/prev cycle by world bearing, left to right, wrapping, bearingless
  last, lost tracks skipped. A slot is routed only while its track is `tracking` and no other track was heard in it more
  recently; the room (outer) is every other slot. Hop audio waits `--out-settle-s` (0.3 s) so the route is current. The hub
  reads private Tracker state (`tracks[id].stream`, `_dup_of`) until the track owner adds a public accessor; `hub/track` is
  untouched. Hardware-pending: real jitter/underrun behaviour, settle/jitter/queue-cap tuning, end-to-end delay.

## Open items
- Each teammate: confirm their tool loads the roles/skills (README "pending runtime
  checks" and smoke checks).
- Install Pi / metavr only with user approval if needed beyond the documentation fallback.
- Choose the app stack, then revisit deferred skills in `sources.json`.
- Choose the Gemma variant and runtime (local server, self-hosted cloud, or on-device).
