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

## Open items
- Each teammate: confirm their tool loads the roles/skills (README "pending runtime
  checks" and smoke checks).
- Install Pi / metavr only with user approval if needed beyond the documentation fallback.
- Choose the app stack, then revisit deferred skills in `sources.json`.
- Choose the Gemma variant and runtime (local server, self-hosted cloud, or on-device).
