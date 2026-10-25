# feat/av-phase4-long-clips
Status: in progress (2026-10-25)
Owner: Cody / Claude Code orchestrator   PR: draft (see `gh pr list`)

## Ownership (disjoint within a wave; each editing worker has its own worktree + branch)
- T0 contracts, cache, config, schemas, CLI, pipeline -> orchestrator:
  `hub/av_scene/{contracts,cache,config,schemas,pipeline,preflight,__main__}.py`, `hub/av_scene/configs/default.toml`,
  `hub/replay/chunked.py`, `hub/tests/av_scene/test_{contracts_chunks,cache_resume,pipeline_*}.py`, `docs/STATUS.md`, this file.
- T1 identity -> integration-engineer (`feat/av-phase4-identity`): `hub/av_scene/continuity_windows.py`,
  `hub/separate/speaker_embed.py`, `hub/tests/av_scene/test_identity_*.py`.
- T2 events -> perception-engineer (`feat/av-phase4-events`): `hub/av_scene/{events,nonspeech_attr,event_scope}.py`,
  `hub/replay/{judge,sam_audio}.py`, `hub/tests/av_scene/test_{event_scope,nonspeech_attr,events_labels}.py`.
- T3 viewer -> integration-engineer (`feat/av-phase4-viewer`): `hub/render/{viewer,long_bundle}.py`,
  `hub/render/assets/*`, `hub/tests/test_viewer_*.py`, `hub/tests/av_scene/test_long_bundle.py`.
- T4 eval + fixtures -> general worker (`feat/av-phase4-eval`): `eval/av_scene/long_*.py`, `hub/tests/eval/test_long_*.py`,
  `hub/tests/av_scene/fixtures_long.py`.
- T5 diarization -> integration-engineer (`feat/av-phase4-diar`): `hub/separate/pyannote_diar.py`,
  `hub/av_scene/diar_windows.py`, `hub/tests/test_pyannote_diar.py`, `hub/tests/av_scene/test_diar_windows.py`.
- T6 visual front-end -> perception-engineer (`feat/av-phase4-visual`): `hub/vision/chunk_tracker.py`,
  `hub/tests/av_scene/test_chunk_tracker.py`.
- Review -> reviewer (read-only) on the combined diff.

## Decisions (promote to `.agents/decisions.md` at merge prep)
- Canonical time = integer sample index at the stream's rate, half-open; us is derived and never round-tripped.
  16 kHz chunking; the multichannel master is preserved; adapters keep their own rates (`ADAPTER_RATES_HZ`).
- Chunking 4 s window / 2 s overlap; boundary = overlap midpoint; owned regions tile the clip exactly. Activity and
  enrollment seconds are counted on owned samples only.
- Identity layers: slot (chunk-local) / source_id `src_n` (acoustic, persistent) / track_id (visual).
- Four independent status fields everywhere (`TimelineSegment`): association_status, visibility, stem_availability,
  refusal_reason.
- Inspection schema extension is versioned separately (`LONG_RUN_SCHEMA_VERSION = 1`) as an optional `RunManifest.long_run`
  plus `timeline/` and `audio/` indexes; Phase 1-3 bundles load unchanged. `SCHEMA_VERSION` (cache key) is NOT bumped.
- Cache entries are checksummed, validated on lookup, quarantined on damage; keys include fingerprint, models, seeds and
  incoming-state hash; one writer per key (O_EXCL lock; dead-pid locks stolen; Windows-safe liveness probe).
- Numeric budgets fixed before any model run (`MemoryBudgets` in `contracts.py`): whole-process RSS peak <= 6144 MiB;
  GPU peak <= 9216 MiB; after 5 warm-up chunks RSS slope <= 2 MiB per audio minute; peak(300 s) - peak(30 s) <= 256 MiB;
  browser JS heap <= 700 MiB at 300 s and <= +100 MiB from 30 s. Memory caps: <=150 decoded frames, 1 queued chunk,
  <=4 candidates, batch 2, <=8 active identities, <=2 resident models, event windows <= 12 s.
- Diarization stays OFF by default; enabled only if bounded, cross-window-consistent behaviour is demonstrated.
- The optional 8 kHz three-speaker checkpoint experiment is permanently skipped.

## Verification
- Baseline before changes: `tests/av_scene` + viewer/pyannote tests -> 3 failures in `test_mixtures.py`
  (torchcodec cannot load FFmpeg DLLs in this shell; environmental, pre-existing).
- T0: `test_contracts_chunks.py`, `test_cache_resume.py`, `test_cache_schemas.py` -> 34 passed (mocked).

## Next steps
- Commit T0, spawn T1-T6 from the T0 commit, write `pipeline.py` / `replay/chunked.py` meanwhile, merge serially,
  CPU gates, review, then (after the user confirms the GPU is free) model runs at 30/120/300 s.
- Needs the user: a real clip with visible silent-person intervals + annotations (silent-bystander gate); gated downloads
  (WavLM-SV, judge, CLAP, pyannote token) only when adapters are ready; GPU free confirmation.
