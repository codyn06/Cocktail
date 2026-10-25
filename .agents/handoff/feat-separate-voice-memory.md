# feat/separate-voice-memory
Status: draft PR open (2026-10-24)
Owner: Codex orchestrator   PR: https://github.com/codyn06/Cocktail/pull/39

## Ownership
- WavLM API/version research: docs scout (read-only).
- LiveSeparator design, implementation, tests, scoring and integration: orchestrator.
- Independent review: complete; one learning-block collision bug found and fixed with a regression test. No remaining correctness findings.

## Decisions
- Base: origin/main 1bf18d5, isolated branch feat/separate-voice-memory; preserve other dirty checkout and events PR worktree.
- Goal: retain separated voice-slot assignment across >1s silence using optional speaker embeddings accumulated from clean active speech. Keep current matching behavior if confidence/margin is insufficient.
- No protocol, Quest or shared interface changes. `LiveSeparator.step` remains synchronous; the encoder runs only to seed/update a profile or identify a returning voice. The stable raw-to-logical map is independent of separator overlap ordering, so outputs, overlap-add tails and smoothed labels stay in logical slot order.
- Verify embeddings on whole accumulated voices; the user-supplied prior 2s WavLM probe had overlapping cosine scores and is not sufficient evidence by itself.
- WavLM uses the model card's `Wav2Vec2FeatureExtractor` plus `AutoModelForAudioXVector` API; no checkpoint is packaged. The weights license scope remains unclear.

## Verification
- Initial graph query on the equivalent base tree identified LiveSeparator at hub/separate/live.py:66, `test_merged_voice_stays_in_one_stream_across_a_pause` at hub/tests/test_separate.py:179, and suite scene 6 in hub/separate/suite.py.
- `graphify update .` completed after code changes; `node .agents/scripts/sync.mjs --check`: OK.
- `tests/test_separate.py`: 16 passed. Runnable suite: 75 passed, 2 deselected, with `tests/test_mock_hub.py` (2 tests) ignored. The four omitted tests hang before assertions in this Windows sandbox: asyncio Proactor socketpair setup in the uvicorn and Starlette local-server fixtures. Faulthandler traces identify the environment-level setup stalls.
- Replay evidence (RTX 5070 Ti Laptop, 12 GB; user confirmed use when free; no headset): local `hub/sessions/live1` replayed for 44 s. Mean step 405.15 ms, p95 528.81 ms, max 695.99 ms (1,000 ms hop budget); WavLM mean 8.21 ms, p95 22.74 ms, max 292.68 ms; peak CUDA allocated 669.5 MB. Three output WAVs and `metrics.json` are in ignored `hub/sessions/live1/out_voicememory`.
- Same saved simulated inputs in scenes 1–7 compared with voice memory off/on. Scenes 1–5 and 7 match baseline scores/stream assignments; scene 6 changes from both speakers sharing s1 (+0.0/−0.1 dB, 0 swaps) to man s1 (+26.9 dB) and woman s2 (+27.3 dB), both 0 swaps. Scene 5's existing score remains −15.8 dB with no change. Scenes 8–11 were not regenerated because librosa's example-host DNS/network is unavailable; only saved sessions 1–7 existed locally.
- WavLM inference ran against the cached `microsoft/wavlm-base-plus-sv` checkpoint. Real WavLM different-speaker scene 6 was replayed; same-speaker pause-return is covered by the scripted test using a deterministic fake encoder, not verified with real speaker labels. No headset or physical hardware was tested.

## Next steps
- Update draft PR body with verification and limitations; commit and push implementation while leaving the PR in draft.
- Retain this handoff until the PR is prepared for merge; delete it in the PR before merge.
