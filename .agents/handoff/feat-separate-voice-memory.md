# feat/separate-voice-memory
Status: in progress (2026-10-24)
Owner: Codex orchestrator   PR: not opened

## Ownership
- WavLM API/version research: docs scout (read-only).
- LiveSeparator design, implementation, tests, scoring and integration: orchestrator.
- Independent review: reviewer after implementation.

## Decisions
- Base: origin/main 1bf18d5, isolated branch feat/separate-voice-memory; preserve other dirty checkout and events PR worktree.
- Goal: retain separated voice-slot assignment across >1s silence using optional speaker embeddings accumulated from clean active speech. Keep current matching behavior if confidence/margin is insufficient.
- No protocol, Quest or shared interface changes. `LiveSeparator.step` remains synchronous and model-heavy work only when needed.
- Verify embeddings on whole accumulated voices; the user-supplied prior 2s WavLM probe had overlapping cosine scores and is not sufficient evidence by itself.

## Verification
- Initial graph query on equivalent base tree identified LiveSeparator at hub/separate/live.py:66, `test_merged_voice_stays_in_one_stream_across_a_pause` at hub/tests/test_separate.py:179, and suite scene 6 in hub/separate/suite.py.
- `nvidia-smi`: RTX 5070 Ti Laptop, 12,227 MiB total, 0 MiB in use at 2026-10-24 21:47 local time; approval requested before long shared GPU jobs.
- Implementation checks pending. No hardware available.

## Next steps
- Validate official WavLM usage and installed runtime; implement optional voice memory; add flipped-output-after-pause regression.
- Measure live1 and suite scene 6; full suite and latency/VRAM only after GPU availability authorization.
- Independent review, update graph and handoff; push branch and keep draft PR for review.
