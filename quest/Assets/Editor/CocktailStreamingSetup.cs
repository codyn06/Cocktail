// Adds the streaming client to the scene without hand-editing Main.unity.
//   Menu:   Cocktail > Add Streaming Components     (idempotent: safe to re-run, fixes missing pieces)
//           Cocktail > Install Meta XR Audio SDK     (opt-in spatializer package)
//   Batch:  Unity.exe -batchmode -projectPath quest -executeMethod Cocktail.Editor.CocktailStreamingSetup.AddBatch -quit
using System.Linq;
using Cocktail.Audio;
using Cocktail.Stream;
using Cocktail.UserInput;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Cocktail.Editor
{
    public static class CocktailStreamingSetup
    {
        const string ScenePath = "Assets/Scenes/Main.unity";
        const string RootName = "CocktailStreaming";
        const string AudioPackageId = "com.meta.xr.sdk.audio";
        const string AudioPackageVersion = "85.0.0"; // latest on npm.developer.oculus.com as of 2026-10-24 (published 2026-02-10)
        const string SpatializerName = "Meta XR Audio";

        [MenuItem("Cocktail/Add Streaming Components")]
        public static void AddMenu()
        {
            if (!Add()) Debug.LogError("[StreamingSetup] finished with errors");
        }

        public static void AddBatch()
        {
            bool ok = false;
            try { ok = Add(); }
            catch (System.Exception e) { Debug.LogException(e); }
            EditorApplication.Exit(ok ? 0 : 1);
        }

        [MenuItem("Cocktail/Install Meta XR Audio SDK (spatializer)")]
        public static void InstallAudioSdk()
        {
            Debug.Log($"[StreamingSetup] adding {AudioPackageId}@{AudioPackageVersion} from the Meta registry. " +
                      "If the project then fails to compile on your Unity version, remove the package (the player falls back to Unity's built-in 3D panner).");
            Client.Add($"{AudioPackageId}@{AudioPackageVersion}");
        }

        static bool Add()
        {
            var active = EditorSceneManager.GetActiveScene();
            if (string.IsNullOrEmpty(active.path) && AssetDatabase.LoadAssetAtPath<SceneAsset>(ScenePath) != null)
                active = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

            var root = GameObject.Find(RootName);
            if (root == null) root = new GameObject(RootName);

            var rig = Object.FindFirstObjectByType<OVRCameraRig>();
            if (rig == null)
                Debug.LogWarning("[StreamingSetup] no OVRCameraRig in the scene; run Cocktail > Configure Quest first. Components are added anyway.");

            var hub = Ensure<HubClient>(root);
            var mic = Ensure<MicStreamer>(root);
            var pose = Ensure<PoseStreamer>(root);
            var cam = Ensure<CameraStreamer>(root);
            var audioOut = Ensure<AudioOutPlayer>(root);
            var input = Ensure<InputSender>(root);
            var diag = Ensure<DiagnosticsOverlay>(root);

            Wire(hub, ("mic", mic), ("cameraStreamer", cam));
            Wire(mic, ("hub", hub));
            Wire(pose, ("hub", hub), ("rig", rig));
            Wire(cam, ("hub", hub), ("rig", rig));
            Wire(audioOut, ("hub", hub), ("rig", rig));
            Wire(input, ("hub", hub), ("rig", rig));
            Wire(diag, ("hub", hub), ("mic", mic), ("cam", cam), ("pose", pose), ("audioOut", audioOut), ("input", input), ("rig", rig));

            ConfigureProject();
            CheckManifest();

            EditorSceneManager.MarkSceneDirty(active);
            bool saved = !string.IsNullOrEmpty(active.path) ? EditorSceneManager.SaveScene(active) : true;
            Debug.Log($"[StreamingSetup] '{RootName}' ready in scene '{active.path}' (saved={saved}): " +
                      "HubClient, MicStreamer, PoseStreamer, CameraStreamer, AudioOutPlayer, InputSender, DiagnosticsOverlay.");
            return saved;
        }

        static T Ensure<T>(GameObject go) where T : Component
        {
            var c = go.GetComponent<T>();
            return c != null ? c : Undo.AddComponent<T>(go);
        }

        static void Wire(Object target, params (string field, Object value)[] refs)
        {
            var so = new SerializedObject(target);
            foreach (var (field, value) in refs)
            {
                var p = so.FindProperty(field);
                if (p == null) { Debug.LogWarning($"[StreamingSetup] {target.GetType().Name}.{field} not found"); continue; }
                if (value != null && p.objectReferenceValue != value) p.objectReferenceValue = value;
            }
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static void ConfigureProject()
        {
            // 1. Passthrough Camera Access: this flag makes the build add horizonos.permission.HEADSET_CAMERA to the manifest.
            var config = OVRProjectConfig.CachedProjectConfig;
            if (config != null && !config.isPassthroughCameraAccessEnabled)
            {
                config.isPassthroughCameraAccessEnabled = true;
                OVRProjectConfig.CommitProjectConfig(config);
                Debug.Log("[StreamingSetup] OVRProjectConfig.isPassthroughCameraAccessEnabled = true");
            }

            // 2. Spatializer: only if the Meta XR Audio package is installed.
            bool audioInstalled = UnityEditor.PackageManager.PackageInfo.FindForPackageName(AudioPackageId) != null;
            var assets = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/AudioManager.asset");
            if (assets != null && assets.Length > 0)
            {
                var so = new SerializedObject(assets[0]);
                bool changed = false;
                if (audioInstalled)
                {
                    changed |= SetString(so, "m_SpatializerPlugin", SpatializerName);
                    changed |= SetString(so, "m_AmbisonicDecoderPlugin", SpatializerName);
                }
                var dsp = so.FindProperty("m_RequestedDSPBufferSize"); // inspector "DSP Buffer Size"; 256 = Best latency
                if (dsp != null && dsp.intValue != 256) { dsp.intValue = 256; changed = true; }
                if (changed)
                {
                    so.ApplyModifiedProperties();
                    AssetDatabase.SaveAssets();
                    Debug.Log($"[StreamingSetup] AudioManager updated (spatializer={(audioInstalled ? SpatializerName : "unchanged: package not installed")}, DSP buffer = best latency)");
                }
            }
            if (!audioInstalled)
                Debug.Log("[StreamingSetup] Meta XR Audio SDK not installed: inner layer uses Unity's built-in 3D panner. Run Cocktail > Install Meta XR Audio SDK for HRTF spatialization.");
        }

        static bool SetString(SerializedObject so, string name, string value)
        {
            var p = so.FindProperty(name);
            if (p == null || p.stringValue == value) return false;
            p.stringValue = value;
            return true;
        }

        static void CheckManifest()
        {
            const string path = "Assets/Plugins/Android/AndroidManifest.xml";
            if (!System.IO.File.Exists(path)) { Debug.LogWarning("[StreamingSetup] AndroidManifest.xml not found at " + path); return; }
            var text = System.IO.File.ReadAllText(path);
            foreach (var need in new[] { "android.permission.INTERNET", "android.permission.RECORD_AUDIO", "horizonos.permission.HEADSET_CAMERA", "usesCleartextTraffic" })
                if (!text.Contains(need)) Debug.LogWarning($"[StreamingSetup] AndroidManifest.xml is missing '{need}' (cleartext ws:// / mic / camera will fail)");
        }
    }
}
