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
- Recorded camera frames (`quest/tools/record_session.py`, `mockclient`, PR #44): layout `video/<device_ts_us>.jpg` plus
  `video/frames.jsonl` (one VideoFrame header per line); `Session.load` reads it into `Session.frame_meta` and also loads recorder
  takes (`meta.json` + `pose.csv`, rotation only so `pos = 0`), and `mockclient.replay` sends the recorded header fields. The
  camera vertical-flip flag is not on the wire (protocol is frozen), so it is the recorder flag `--camera-flipped` stored in
  `meta.json`. Frames are buffered in memory and written at save (a headset clock restart discards the earlier session); limits
  `--no-video`, `--video-fps`, `--max-frames`; h264 frames are skipped. Talker ground truth is `talkers.csv`
  (`start_s,end_s,talker_id,bearing_deg,note`, seconds since the first audio sample) mapped to frames by `mockclient.talkers`.
  Untested on a real headset: frame orientation/colour, real header values and disk use.
- Chosen-voice measurement (`separate.measure_chosen`, PR #43): SIR-style on the separator's streams, not the `/audio/out` route.
  Each 250 ms window of the chosen stream is least-squares projected on the session's ground-truth stems; margin = target vs
  the sum of the other stems, and sound matching no stem is reported as `unexplained_db`. The chosen stream is picked by oracle
  (best SI-SDR vs the target after t0); `carried_by` shows when the voice sits in another stream. Sessions without stems are
  refused. Replayed result: the 10 dB / 1 s target is met on the 2-voice scene and not met on the 3-voice scene (two voice
  outputs plus a residual); a decision on how a third talker is held is needed before that target can pass.

- `/audio/out` takes an optional `?layout=layers|stereo` query parameter (default `layers`, the Quest). `layers` frames are unity gain because the Quest applies inner/outer dB itself; the hub applies `SetVolumes` only for `stereo` clients (or with `--out-gains-in-layers`). Approved by Cody (PR #42); no change to `hub/protocol`.
- Chosen voice routing (hub `events/selection.py`, `out/`, PR #42): select/stop/next/prev arrive as `InputEvent` on `/ingest/pose`
  (ingest got `input_hooks`), not as `/events` commands. The choice is kept as a Tracker id, stable through slot swaps and
  pauses; voice memory is complementary, not the key. next/prev cycle by world bearing, left to right, wrapping, bearingless
  last, lost tracks skipped. A slot is routed only while its track is `tracking` and no other track was heard in it more
  recently; the room (outer) is every other slot. Hop audio waits `--out-settle-s` (0.3 s) so the route is current. The hub
  reads private Tracker state (`tracks[id].stream`, `_dup_of`) until the track owner adds a public accessor; `hub/track` is
  untouched. Hardware-pending: real jitter/underrun behaviour, settle/jitter/queue-cap tuning, end-to-end delay.
- AV speech attribution (PR #53): MossFormer2_SS separates 2 voices; S3FD faces are assigned to SAM 3 person masks;
  TalkNet gives per-person activity; AV_MossFormer2_TSE runs per active person and must match one separated voice
  (SI-SDR >= 3 dB, 3 dB margin) that is active while the face speaks. Contested voices go to one claimant only with a
  2x / 1 s activity lead, else all are refused; an inactive or unmatched person gets `unavailable`, never another voice.
  Envelope correlation (`associate.py`) is the no-TSE fallback only. SAM Audio is not used to select a speaker (its
  visual mask did not steer the output); it stays for non-speech prompts. Checkpoints (AV_MossFormer2_TSE_16K, TalkNet
  `6d68214` + `pretrain_TalkSet.model`, S3FD `sfd_face.pth`) are project-local and gitignored under `hub/checkpoints/`.
- AV overlap and identity, Phase 3 (branch `feat/av-phase3-overlap-identity`): three speech-branch switches, all default
  OFF in `run_speech_branch` (`activity_gate`, `offscreen_upgrade`, `peel`); Row 0 and ablation rows are in
  `docs/STATUS.md`. `peel` (bounded residual speech peeling, `av_scene/peeling.py`) is the 3-talker route because no
  3-speaker checkpoint loads through ClearVoice (only `alibabasglab/mossformer2-wsj0mix-3spk`: 8 kHz, other loader,
  untested), so `MAX_SEPARABLE_TALKERS` stays 2. `activity_gate` rescues TalkNet misses from TSE energy, guarded by an
  absolute floor, a 6 dB lead over other tracks, and a mouth-sync veto (bleed at -18.7 dBFS stays gated; a lone bystander
  relies on the -11 dB threshold alone). `offscreen_upgrade` calls a voice `outside_view` only if no visible face's TSE
  output, including faces TalkNet called silent, resembles it. Library modules without a pipeline stage yet, tested on
  mocks only: identity windows (`continuity_windows.py`), pyannote adapter (`separate/pyannote_diar.py`, mocked, needs an
  HF token and accepted terms), non-speech attribution (`nonspeech_attr.py`), residual Nyquist guard
  (`stem_nyquist_hz`, activity-gated: a tone overlapping active speech is not reported). The viewer exports a
  `frame_transform` (manifest `capabilities` and `frames.json`). The synthetic second pair (demo voice1/voice2) is not a
  real second speaker pair; no recording has clean per-speaker references.

## Open items
- Each teammate: confirm their tool loads the roles/skills (README "pending runtime
  checks" and smoke checks).
- Install Pi / metavr only with user approval if needed beyond the documentation fallback.
- Choose the app stack, then revisit deferred skills in `sources.json`.
- Choose the Gemma variant and runtime (local server, self-hosted cloud, or on-device).
