# docs/av-discovery-spec
Status: ready for review (2026-10-25)
Owner: Codex orchestrator   PR: https://github.com/codyn06/Cocktail/pull/45

## Ownership
- Research and specification -> orchestrator: docs/AV_SOUND_DISCOVERY_SPEC.md, README.md, ARCHITECTURE.md.
- Branch continuity -> orchestrator: this file.
- Two read-only research delegations failed before returning findings because the agent service reported a usage limit. Primary agent performed source research directly.

## Decisions
- Documentation-only expansion of the user-provided specification; no scaffolding, dependencies, model downloads or device work.
- Fit the existing hub Python project and PRD ownership boundaries: ingest, separate, vision, who, locate, replay; propose av_scene and render.
- Keep protocol extensions and runtime changes as proposals; do not modify shared schemas or team decisions.
- Use a separate worktree from fetched origin/main 3b9878ff7f63ac148245bbf085eb17b83843042e; preserve dirty primary checkout.
- Include verified SAM Audio mask polarity/return shape, SAM 3 semantic prompt reset, regular versus exclusive diarization, missing-anchor handling and instance-association refinement.

## Verification
- Public official repositories/model docs and current repository source inspected; citations in specification.
- Read-only GPU/runtime inspection: RTX 5070 Ti Laptop 12227 MiB; existing hub Python 3.12.15, torch 2.11.0+cu128, transformers 5.20.2. FFmpeg/ffprobe not resolved on inspected PATH. No model inference or GPU benchmarks.
- Documentation validation: PASS (15 local reference occurrences, balanced code fences, 13 sections, entry-point links); node .agents/scripts/sync.mjs --check: PASS (90 generated files); git diff --check: PASS. No model/inference tests for this documentation-only change.
- Independent reviewer completed source/document review: no blocking defect. Addressed its channel-domain clarification by specifying a mono reconstruction mixture and a distinct-channel stereo fixture. Initial research delegations had failed due to service limits; reviewer later succeeded.

## Next steps
- Draft PR #45 opened. Final documentation/link/channel-contract checks and generated-agent sync passed after reviewer clarification; staged diff checked before commit.
- Human review and approval remain; PR stays draft as a research proposal.
- Before merge, retain durable rationale in specification/PR and delete this branch handoff.
- Implementation starts at Phase 0 capability probes/Phase 1 deterministic artifacts; proposed CLI and modules do not exist yet.
