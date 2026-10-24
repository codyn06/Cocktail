# feat/quest-unity-setup
Status: in progress (2026-10-24)
Owner: codyn06 / Claude Code   PR: not opened

## Ownership
- Unity project scaffold -> orchestrator: `quest/`

## Decisions
- Unity 6000.0.80f1 LTS, OpenXR + Meta XR SDK core/MRUK 207.0.1 (OculusXR plugin is deprecated).
- Quest 3S: full prototype (passthrough camera, depth). Quest 2: hub-event display and audio only
  (Passthrough Camera API is Quest 3/3S, Horizon OS v74+).
- Android build settings come from Meta's Project Setup Tool rather than being hand-set.

## Verification
- Batch-mode `-createProject` and package import on 6000.0.80f1 -> packages resolved, 0 `error CS` (mocked, editor only).
- Android build / headset run -> untested (Android module not installed on this machine).

## Next steps
- Install Android Build Support for 6000.0.80f1, then run the project configuration steps in `quest/README.md`
  and commit the resulting ProjectSettings.
- Build and run on Quest 3S and Quest 2. Then add a hub `/events` WebSocket client.
