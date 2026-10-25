# quest/: Cocktail headset app (Unity)

The Quest app streams mic audio, head pose and camera frames to the laptop hub and shows a speaker HUD in passthrough:
a picker of talking sources, hand pointing, pinch to choose, and marks on the chosen source. It reads the hub's
`WS /events` and plays `WS /audio/out`. It is a thin client: the laptop hub does all processing. Glasses mode is out of
scope for now; the Quest is the product.

## Target devices

| Device | Planned role | Why |
|---|---|---|
| Quest 3S | Full prototype: glasses-view panel, mixed-reality mode, passthrough camera frames to SAM 3, Depth API | Passthrough Camera API supports only Quest 3 / 3S on Horizon OS v74+ |
| Quest 2 | Hub display only: glasses-view panel and spatial audio driven by hub events | No Passthrough Camera API, and its passthrough is grayscale |

Built so far: streaming (mic, pose, camera), the speaker HUD and DevMode (see **Using the app**). Camera
access needs `horizonos.permission.HEADSET_CAMERA` (or `android.permission.CAMERA`)
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
room through passthrough with the HUD (the marker cube from the first prototype was removed).
The rig uses the eye-level tracking origin.

## Using the app (headset HUD)

What the wearer sees (Quest 3S, passthrough):
- **Picker** (lower right): one row per talking source, a coloured block and a name (human, object, unknown), nearest first.
- **Pointing**: the right hand points (head gaze if no hand is tracked); a cursor dot shows where. Pinch to choose a speaker
  or a picker row; pinch the chosen one again to stop. Nothing is chosen automatically.
- **Marks**: a pointed-at speaker gets a tint; the chosen one gets a rectangle with a label tab; out of view, an arrow at the
  edge of the view; a source with no direction, a plain dot. The rectangle sits at the sound's direction with a default size:
  the hub does not send a box around the person yet (`bbox` is empty, C5/M4).
- **DevMode** (off by default, `UiTheme.devMode`): pinch the **DEV** chip (upper left) to toggle it; B + Y together also
  toggles it when controllers are held. It shows the DEMO panel (fake-data fallback, test cubes and their sound, numbers on
  rows, mode, volumes, stop / next / prev), a readout (DATA FAKE/REAL, HUB UP/DOWN, chosen source, hand flags T/P/pinch,
  sources), the last events (`--- LOG`) and the diagnostics text. Fake data only runs in DevMode, so with DevMode off and no
  hub the HUD is empty. Turn `UiTheme.showDevToggle` off to hide the chip for a demo.
- **Event log**: while DevMode is on, events also go to logcat (tag `[Dev]`) and to
  `/sdcard/Android/data/com.DefaultCompany.quest/files/devlog.txt` (`adb pull` it).

Controls with controllers: right trigger = choose, B or right grip = stop, Y or right stick right = next, left stick click
or right stick left = previous, right stick up/down = scroll the picker, A = cycle the older HUD modes (keep speaker-only).

### Connecting to a hub

1. Start the hub on the laptop (`cd hub && uv run --group gpu python -m separate`), or the mock hub
   (`uv run --project hub python quest/tools/mock_hub.py`, no GPU).
2. USB: `adb reverse tcp:8765 tcp:8765` and point the app at `127.0.0.1`:
   `adb shell "echo 127.0.0.1 > /sdcard/Android/data/com.DefaultCompany.quest/files/hub_host.txt"`.
   Wi-Fi/hotspot: put the laptop IP there instead (the default `192.168.137.1` is the Windows mobile hotspot address).
3. Start or restart the app: `adb shell am force-stop com.DefaultCompany.quest` then
   `adb shell monkey -p com.DefaultCompany.quest -c android.intent.category.LAUNCHER 1`.
   The hub's 0 deg is where the headset faced when the app started, so face the scene when you launch it.
4. Check: the DEMO readout says `DATA REAL` and `HUB UP`; the hub's `/ingest/status` shows audio messages rising.

### Things that went wrong in testing (and the fix)

| Symptom | Cause | Fix |
|---|---|---|
| All five sockets say "Unable to connect" after a reinstall | the `adb reverse` forward is dropped on install / reconnect | run `adb reverse tcp:8765 tcp:8765` again |
| `adb devices` says `offline` | the headset's USB debugging link dropped | replug the cable, accept **Allow USB debugging** in the headset |
| Hub audio counter frozen | the app is not in the foreground (home, room setup or boundary screen) | open the app (launch command above) |
| Room layout / boundary screen in the way | the system boundary or space setup UI pauses the app | finish or dismiss it in the headset; see Meta's boundary settings |
| App closes a few seconds after the hub connects (signal 11 in logcat) | fixed: `EnsureInit` recursed when the hub sent an empty list (PR #61 / #66) | update to a build with the fix |
| Hub `separate/status` stuck at `hops_done: 100` | display only: it is the length of a list capped at 100 | check that the streams' `start_ts_us` still advance |

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
  - tracked hands under the hand anchors (hand tracking: controllers and hands, frequency high);
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
- Quest 3S, live loop (hardware-tested, 2026-10-25): the app connects to the real hub over USB, streams 4 mic
  channels at 48 kHz (only ch0 and ch1 carry sound), camera frames and pose; DevMode toggles by pinch; the picker,
  pointing and pinch work with fake data. With real voices the hub now locates sources (PR #63), but the rectangle is
  not yet on the talker: the direction estimate is off (see ARCHITECTURE.md D8 and `hub/locate/TUNING.md`).
- Quest 2: untested.
