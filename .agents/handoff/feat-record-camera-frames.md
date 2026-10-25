# feat/record-camera-frames
Status: ready for review (2026-10-25)
Owner: perception-engineer subagent for the Cocktail orchestrator   PR: draft, see GitHub

## Ownership
- Save camera frames in the session recorder -> perception-engineer: quest/tools/record_session.py
- Load/replay recorded frames + headers, talker labels -> perception-engineer: hub/mockclient/{session,replay,talkers,__init__}.py
- Tests -> hub/tests/test_record_session.py; docs -> README.md

## Decisions
- Layout `video/<device_ts_us>.jpg` (what Session.load already parsed) plus `video/frames.jsonl`, one VideoFrame header per line (Session.load did not read a metadata file before; it now does, into `Session.frame_meta`, and `mockclient.replay` uses it instead of hard-coded 1280x960).
- Recorder takes (meta.json, pose.csv) did not load with Session.load (it needed session.json, pose.jsonl). Session.load now falls back to the recorder layout: start_ts_us = first audio ts, device/camera_hfov_deg from the recorded Hello, poses from pose.csv with pos = 0,0,0 (the recorder stores rotation only).
- The vertical-flip flag is not on the wire (VideoFrame/Hello have no field, protocol is frozen), so it is a recorder CLI flag `--camera-flipped` stored in meta.json (`video_camera_flipped`), not per-frame.
- Frames are buffered in memory like audio and written at save (needed because a headset clock restart discards the earlier session). Limits: `--no-video`, `--video-fps`, `--max-frames`.
- Talker ground truth: `talkers.csv` (start_s,end_s,talker_id,bearing_deg,note), seconds since the first audio sample; `mockclient.talkers` loads it and maps frames to active talkers. Bearing convention matches `--truth`.

## Verification
- `uv run --project hub pytest -q hub` -> 92 passed, 1 skipped (main baseline 86 passed, 1 skipped; +6 new) (mocked)
- Round trip: synthetic AudioChunk/VideoFrame through the recorder websockets -> files -> Session.load -> replay into an in-process hub, header fields identical (mocked)
- No headset run, no real JPEG decode, no SAM 3 run (untested)

## Next steps
- Record a real take with the headset (`--camera-flipped` if needed), check frames.jsonl against the real header, orientation of frames, disk size at 15 fps.
- Hand-label talkers.csv for that take; run SAM 3 over `Session.frames` and score box-on-talker with `frame_truth`.
- Drop this file in the PR before merge.
