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
- `ConfigureBatch` + `BuildBatch` -> 20 Project Setup fixes applied, APK built, manifest targets quest2|quest3s (mocked, no device).
- Quest 3S (build 4069240017700610): APK installed via adb; the runtime logs `PT is: ON numLayers: 1`; the user sees the marker cube (hardware-tested, 2026-10-24). The first build showed the skybox instead of passthrough because it had no OVRPassthroughLayer; that is fixed.
- Quest 2 -> untested.

## Next steps
- Install the APK on Quest 2 and confirm grayscale passthrough.
- Then add a hub `/events` WebSocket client.
