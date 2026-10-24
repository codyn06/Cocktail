# quest/: Quest display prototype (Unity)

The PRD calls this a later layer. It shows Cocktail on display glasses and reads the
same hub events as the phone app (`WS /events`, `WS /audio/out`). It is a thin client:
the laptop hub still does all processing.

## Target devices

| Device | Role | Why |
|---|---|---|
| Quest 3S | Full prototype: glasses-view panel, mixed-reality mode, passthrough camera frames to SAM 3, Depth API | Passthrough Camera API supports only Quest 3 / 3S on Horizon OS v74+ |
| Quest 2 | Hub display only: glasses-view panel and spatial audio driven by hub events | No Passthrough Camera API, and its passthrough is grayscale |

Source: [Passthrough Camera API overview](https://developers.meta.com/horizon/documentation/unity/unity-pca-overview/)
(read 2026-10-24). Camera access needs `horizonos.permission.HEADSET_CAMERA` (or
`android.permission.CAMERA`) with Passthrough enabled. It does not run in the simulator,
so device behaviour is untested until we try it on hardware.

## Pinned stack

- Unity **6000.0.80f1** (Unity 6 LTS)
- `com.meta.xr.sdk.core` 207.0.1 and `com.meta.xr.mrutilitykit` 207.0.1, from Meta's
  scoped registry `https://npm.developer.oculus.com`
- `com.unity.xr.openxr` 1.14.3. Meta says the OculusXR plugin is deprecated, so use OpenXR.

## One-time machine setup

1. In Unity Hub, add these modules to **6000.0.80f1**: **Android Build Support**
   with **OpenJDK** and **Android SDK & NDK Tools**.
2. Install [Meta Quest Developer Hub](https://developers.meta.com/horizon/documentation/unity/ts-odh/)
   (it includes `adb`). You can also use the `adb` that Unity's Android module installs.
3. Turn on developer mode for the headset: join or create a developer organization at
   developers.meta.com, then switch on **Developer Mode** for the headset in the Meta
   Horizon phone app.
4. Connect the headset by USB-C, accept the USB debugging prompt inside the headset,
   then run `adb devices` and check that the headset is listed.
5. Optional: install Meta XR Simulator to test UI without a headset. Passthrough camera
   access will not work there.

## One-time project configuration

These steps need the Android module, so run them in the editor after machine setup.
Commit the `ProjectSettings/` changes they produce.

1. Open `quest/` in Unity Hub with 6000.0.80f1.
2. **File > Build Profiles**: switch to **Android**.
3. **Project Settings > XR Plug-in Management**: on both the Android and Standalone
   tabs, enable **OpenXR** and the **Meta XR** feature group.
4. **Meta > Tools > Project Setup Tool**: apply every fix on the Android tab. It sets
   ARM64, IL2CPP, the graphics API and the minimum API level, so we don't hard-code them.
5. **OVRManager / Quest features**: target **Quest 3S** and **Quest 2**, and set
   Passthrough support to Supported.
6. Build and run once on each headset.

## Status

The project and packages are scaffolded, and a batch-mode import compiles with 0 errors
(mocked: editor only, no device). XR configuration, the Android build and anything on
a headset are untested.
