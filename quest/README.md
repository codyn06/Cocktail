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

## Project configuration

This is already committed. To re-apply it, use **Cocktail > Configure Quest**
([Assets/Editor/QuestProjectSetup.cs](Assets/Editor/QuestProjectSetup.cs)) or run it
in batch mode:

```bash
Unity.exe -batchmode -projectPath quest -executeMethod Cocktail.Editor.QuestProjectSetup.ConfigureBatch
```

It does the following:
- Switches the build target to Android.
- Enables the OpenXR loader and the Meta XR feature group for Android and Standalone.
- Restricts the Meta Quest feature and `OVRProjectConfig` to **Quest 2** and **Quest 3S**,
  and sets Passthrough support to Supported.
- Creates `Assets/Scenes/Main.unity`: an `OVRCameraRig` with passthrough on, plus a light.
- Applies every Meta Project Setup Tool fix for Android: min API 32, target API 34,
  GameActivity, ASTC, the manifest and the controller profile, among others.

Build a smoke-test APK to `quest/Builds/cocktail-quest.apk`:

```bash
Unity.exe -batchmode -projectPath quest -executeMethod Cocktail.Editor.QuestProjectSetup.BuildBatch
```

Install it on a headset connected by USB:

```bash
adb install -r quest/Builds/cocktail-quest.apk
```

## Status

- Configuration and APK build: done in batch mode, and the build succeeded (mocked: no
  device). The generated manifest has `com.oculus.supportedDevices = quest2|quest3s`.
- On a headset: untested. No headset was connected.
