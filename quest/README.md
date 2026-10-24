# quest/: Quest display prototype (Unity)

The PRD calls this a later layer. It shows Cocktail on display glasses and reads the
same hub events as the phone app (`WS /events`, `WS /audio/out`). It is a thin client:
the laptop hub still does all processing.

## Target devices

| Device | Planned role | Why |
|---|---|---|
| Quest 3S | Full prototype: glasses-view panel, mixed-reality mode, passthrough camera frames to SAM 3, Depth API | Passthrough Camera API supports only Quest 3 / 3S on Horizon OS v74+ |
| Quest 2 | Hub display only: glasses-view panel and spatial audio driven by hub events | No Passthrough Camera API, and its passthrough is grayscale |

None of these roles is built yet; today the app is an empty passthrough scene. Camera
access will need `horizonos.permission.HEADSET_CAMERA` (or `android.permission.CAMERA`)
with Passthrough enabled. It doesn't run in the simulator.
Source: [Passthrough Camera API overview](https://developers.meta.com/horizon/documentation/unity/unity-pca-overview/)
(read 2026-10-24).

## Pinned stack

- Unity **6000.0.80f1** (Unity 6 LTS)
- `com.meta.xr.sdk.core` 207.0.1 and `com.meta.xr.mrutilitykit` 207.0.1, from Meta's
  scoped registry `https://npm.developer.oculus.com`
- `com.unity.xr.openxr` 1.14.3. Meta says the OculusXR plugin is deprecated, so use OpenXR.

## 1. Install Unity (once per computer)

1. Install [Unity Hub](https://unity.com/download).
2. In Unity Hub, go to **Installs**, then **Install Editor**. Choose **6000.0.80f1**
   (it's under **Archive** if it isn't listed).
3. Tick the **Android Build Support** module, with **OpenJDK** and
   **Android SDK & NDK Tools** under it. If the editor is already installed, use the
   gear icon next to it and choose **Add modules**.
4. Open the project: in **Projects**, choose **Add > Add project from disk** and pick
   the `quest/` folder. The first open downloads packages and takes a few minutes.

## 2. Set up the headset (once per headset)

Developer account (in your browser):

1. Sign in at [developers.meta.com](https://developers.meta.com) with the Meta account
   your headset uses.
2. [Create an organization](https://developers.meta.com/horizon/manage/organizations/create)
   with any name, and accept the developer agreement. Meta may ask you to verify the
   account with a phone number or payment method.

Developer Mode (in the Meta Horizon app on your phone):

3. Tap **Devices**, select the headset, then tap **Headset settings > Developer Mode**.
4. Switch **Developer Mode** on, then restart the headset.

USB connection:

5. Connect the headset to the PC with a USB-C **data** cable (many cables only charge).
6. Put the headset on. When it asks **Allow USB debugging?**, tick
   **Always allow from this computer** and choose **Allow**.
7. Check the connection. Unity's Android module includes `adb`. In PowerShell:

   ```powershell
   & "C:\Program Files\Unity\Hub\Editor\6000.0.80f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe" devices
   ```

   In Git Bash:

   ```bash
   "/c/Program Files/Unity/Hub/Editor/6000.0.80f1/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb.exe" devices
   ```

   The commands below use `adb` for this path. You can also add `platform-tools` to
   your PATH, or install [Meta Quest Developer Hub](https://developers.meta.com/horizon/documentation/unity/ts-odh/),
   which includes `adb`.

8. Read the output:
   - A serial number followed by `device` means it's connected.
   - `unauthorized` means the prompt wasn't accepted. Put the headset on and accept it.
   - If nothing is listed, try another cable or port, or install the
     [Oculus ADB driver](https://developers.meta.com/horizon/downloads/package/oculus-adb-drivers/).

## 3. Build and run

The project configuration is already committed, so you can build right away. In the
editor, use **File > Build Profiles > Android > Build And Run**. From the command line
(Unity closed), run this from the repo root:

```bash
Unity.exe -batchmode -projectPath quest -executeMethod Cocktail.Editor.QuestProjectSetup.BuildBatch
adb install -r quest/Builds/cocktail-quest.apk
adb shell am start -n com.DefaultCompany.quest/com.unity3d.player.UnityPlayerGameActivity
```

On the headset, the app is in **App Library > Unknown Sources > quest**. It shows your
room through passthrough, with a small gray cube about 1.2 m above eye level (look up).
The rig uses the eye-level tracking origin.

## Re-applying the configuration

If settings drift, run **Cocktail > Configure Quest**
([Assets/Editor/QuestProjectSetup.cs](Assets/Editor/QuestProjectSetup.cs)), or:

```bash
Unity.exe -batchmode -projectPath quest -executeMethod Cocktail.Editor.QuestProjectSetup.ConfigureBatch
```

It exits with code 1 if a step fails. The script does the following:
- Switches the build target to Android.
- Enables the OpenXR loader and the Meta XR feature group for Android and Standalone.
- Restricts the Meta Quest feature and `OVRProjectConfig` to **Quest 2** and **Quest 3S**,
  and sets Passthrough support to Supported.
- Creates or repairs `Assets/Scenes/Main.unity`:
  - an `OVRCameraRig` with an `OVRPassthroughLayer` underlay;
  - eye cameras that clear to transparent black;
  - a marker cube;
  - a light, added only when the scene is first created.
- Keeps any other scenes in the build list, with Main first.
- Applies Meta's Project Setup Tool fixes for Android, including minimum API 32,
  GameActivity, ASTC, the manifest and the controller profile.

Meta's editor tools write per-user files to `Assets/Resources/`: `DevAgentSettings`
(it holds a local access token and LAN IP) and `OVRPlatformToolSettings`. They are
gitignored. Don't commit them.

## Status

- Configuration and APK build: built in batch mode, not run on a device. The manifest
  has `com.oculus.supportedDevices = quest2|quest3s`.
- Quest 3S (hardware-tested, 2026-10-24):
  - the app installs and runs;
  - the runtime reports one passthrough layer on;
  - the user saw the room in color and the marker cube.
- Quest 2: untested.
