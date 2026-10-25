// Reproducible Quest project configuration. Run from the menu (Cocktail > Configure Quest)
// or in batch mode:
//   Unity.exe -batchmode -projectPath quest -executeMethod Cocktail.Editor.QuestProjectSetup.ConfigureBatch
// and build a smoke-test APK with -executeMethod Cocktail.Editor.QuestProjectSetup.BuildBatch.
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.SceneManagement;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEditor.XR.OpenXR.Features;
using UnityEngine;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features.MetaQuestSupport;

namespace Cocktail.Editor
{
    public static class QuestProjectSetup
    {
        const string ScenePath = "Assets/Scenes/Main.unity";
        const string CameraRigPath = "Packages/com.meta.xr.sdk.core/Prefabs/OVRCameraRig.prefab";
        const string HandPrefabPath = "Packages/com.meta.xr.sdk.core/Prefabs/OVRHandPrefab.prefab";
        const string MetaFeatureSetId = "com.meta.openxr.featureset.metaxr";
        const string ApkPath = "Builds/cocktail-quest.apk";

        // Quest 2 is a hub-event display; Quest 3S runs the passthrough camera features.
        static readonly OVRProjectConfig.DeviceType[] TargetDevices =
        {
            OVRProjectConfig.DeviceType.Quest2,
            OVRProjectConfig.DeviceType.Quest3S,
        };
        static readonly string[] ManifestDevices = { "quest2", "quest3s" };

        [MenuItem("Cocktail/Configure Quest")]
        public static async void ConfigureMenu()
        {
            try { await Configure(); }
            catch (System.Exception e) { Debug.LogException(e); }
        }

        // async void must catch everything, or a throw leaves batch-mode Unity running forever.
        public static async void ConfigureBatch()
        {
            var ok = false;
            try { ok = await Configure(); }
            catch (System.Exception e) { Debug.LogException(e); }
            EditorApplication.Exit(ok ? 0 : 1);
        }

        public static void BuildBatch()
        {
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = ApkPath,
                target = BuildTarget.Android,
                options = BuildOptions.None,
            });
            Debug.Log($"[QuestSetup] Build result: {report.summary.result}, {report.summary.totalSize} bytes");
            EditorApplication.Exit(report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded ? 0 : 1);
        }

        static async Task<bool> Configure()
        {
            EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android);

            var ok = true;
            foreach (var group in new[] { BuildTargetGroup.Android, BuildTargetGroup.Standalone })
            {
                AssignOpenXR(group);
                ok &= EnableMetaFeatures(group);
            }
            ok &= ConfigureMetaQuestFeature();
            ConfigureOVRProject();
            ok &= CreateScene();

            await OVRProjectSetup.FixAllAsync(BuildTargetGroup.Android);
            AssetDatabase.SaveAssets();
            Debug.Log(ok ? "[QuestSetup] Done." : "[QuestSetup] Finished with errors.");
            return ok;
        }

        static void AssignOpenXR(BuildTargetGroup group)
        {
            if (!EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.k_SettingsKey,
                    out XRGeneralSettingsPerBuildTarget perTarget))
            {
                if (!AssetDatabase.IsValidFolder("Assets/XR")) AssetDatabase.CreateFolder("Assets", "XR");
                perTarget = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
                AssetDatabase.CreateAsset(perTarget, "Assets/XR/XRGeneralSettingsPerBuildTarget.asset");
                EditorBuildSettings.AddConfigObject(XRGeneralSettings.k_SettingsKey, perTarget, true);
            }
            if (!perTarget.HasSettingsForBuildTarget(group))
                perTarget.CreateDefaultSettingsForBuildTarget(group);
            if (!perTarget.HasManagerSettingsForBuildTarget(group))
                perTarget.CreateDefaultManagerSettingsForBuildTarget(group);

            var settings = perTarget.SettingsForBuildTarget(group);
            settings.InitManagerOnStart = true;
            XRPackageMetadataStore.AssignLoader(settings.AssignedSettings,
                typeof(OpenXRLoader).FullName, group);
            EditorUtility.SetDirty(settings);
        }

        static bool EnableMetaFeatures(BuildTargetGroup group)
        {
            FeatureHelpers.RefreshFeatures(group);
            var set = OpenXRFeatureSetManager.GetFeatureSetWithId(group, MetaFeatureSetId);
            if (set == null)
            {
                Debug.LogError($"[QuestSetup] Meta XR feature group not found for {group}");
                return false;
            }
            set.isEnabled = true;
            OpenXRFeatureSetManager.SetFeaturesFromEnabledFeatureSets(group);
            return true;
        }

        static bool ConfigureMetaQuestFeature()
        {
            var feature = OpenXRSettings.GetSettingsForBuildTargetGroup(BuildTargetGroup.Android)
                .GetFeature<MetaQuestFeature>();
            if (feature == null)
            {
                Debug.LogError("[QuestSetup] Meta Quest OpenXR feature not found");
                return false;
            }
            feature.enabled = true;

            // targetDevices is internal; edit it through serialization.
            var so = new SerializedObject(feature);
            var devices = so.FindProperty("targetDevices");
            if (devices == null)
            {
                Debug.LogError("[QuestSetup] MetaQuestFeature.targetDevices not found; the SDK layout changed");
                return false;
            }
            for (var i = 0; i < devices.arraySize; i++)
            {
                var device = devices.GetArrayElementAtIndex(i);
                var name = device.FindPropertyRelative("manifestName");
                var enabled = device.FindPropertyRelative("enabled");
                if (name == null || enabled == null) return false;
                enabled.boolValue = ManifestDevices.Contains(name.stringValue);
            }
            so.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(feature);
            return true;
        }

        static void ConfigureOVRProject()
        {
            var config = OVRProjectConfig.CachedProjectConfig;
            config.targetDeviceTypes = new List<OVRProjectConfig.DeviceType>(TargetDevices);
            config.insightPassthroughSupport = OVRProjectConfig.FeatureSupport.Supported;
            // Hands plus controllers: the app is operated by pinching, and switches to hands when the controllers are put down.
            config.handTrackingSupport = OVRProjectConfig.HandTrackingSupport.ControllersAndHands;
            OVRProjectConfig.CommitProjectConfig(config);
        }

        /// <summary>Tracked hands under the rig's hand anchors, so OVRHand pinch states exist (CocktailUI reads them). Skipped when already present.</summary>
        static void EnsureHands(OVRCameraRig rig)
        {
            if (rig.GetComponentInChildren<OVRHand>(true) != null) return;
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(HandPrefabPath);
            if (prefab == null) { Debug.LogError($"[QuestSetup] Hand prefab missing at {HandPrefabPath}"); return; }
            foreach (var side in new[] { ("LeftHandAnchor", 0), ("RightHandAnchor", 1) })
            {
                var anchor = rig.transform.Find("TrackingSpace/" + side.Item1);
                if (anchor == null) { Debug.LogError($"[QuestSetup] {side.Item1} not found under the rig"); continue; }
                var hand = (GameObject)PrefabUtility.InstantiatePrefab(prefab, anchor);
                hand.name = side.Item2 == 0 ? "LeftHand" : "RightHand";
                // 0 = left, 1 = right on OVRHand.HandType, OVRSkeleton._skeletonType and OVRMesh._meshType
                SetEnum(hand.GetComponent<OVRHand>(), "HandType", side.Item2);
                SetEnum(hand.GetComponent<OVRSkeleton>(), "_skeletonType", side.Item2);
                SetEnum(hand.GetComponent<OVRMesh>(), "_meshType", side.Item2);
            }
        }

        static void SetEnum(Component c, string field, int value)
        {
            if (c == null) return;
            var so = new SerializedObject(c);
            var p = so.FindProperty(field);
            if (p == null) { Debug.LogWarning($"[QuestSetup] {c.GetType().Name}.{field} not found"); return; }
            p.intValue = value;   // the enum's value, not its list index (OVRSkeleton/OVRMesh enums start at None = -1)
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        static bool CreateScene()
        {
            UnityEngine.SceneManagement.Scene scene;
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(ScenePath) != null)
            {
                scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            }
            else
            {
                AssetDatabase.CreateFolder("Assets", "Scenes");
                scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var light = new GameObject("Directional Light").AddComponent<Light>();
                light.type = LightType.Directional;
            }

            // Re-add the rig if it was deleted, so a re-run repairs the scene.
            var rig = Object.FindFirstObjectByType<OVRCameraRig>();
            if (rig == null)
            {
                var rigPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(CameraRigPath);
                if (rigPrefab == null)
                {
                    Debug.LogError($"[QuestSetup] Camera rig prefab missing at {CameraRigPath}");
                    return false;
                }
                rig = ((GameObject)PrefabUtility.InstantiatePrefab(rigPrefab)).GetComponent<OVRCameraRig>();
            }
            var manager = rig.gameObject;

            // Visible marker so an empty passthrough scene is distinguishable from the home view.
            if (GameObject.Find("Marker Cube") == null)
            {
                var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
                cube.name = "Marker Cube";
                cube.transform.position = new Vector3(0f, 1.2f, 1f);
                cube.transform.localScale = Vector3.one * 0.2f;
            }

            // Unity's fake-null breaks ??, so check components explicitly.
            var ovr = manager.GetComponent<OVRManager>();
            if (ovr == null) ovr = manager.AddComponent<OVRManager>();
            ovr.isInsightPassthroughEnabled = true;

            EnsureHands(rig);

            // Passthrough renders as an underlay, so the eye camera must clear to transparent
            // black instead of drawing the skybox over it.
            if (manager.GetComponent<OVRPassthroughLayer>() == null)
                manager.AddComponent<OVRPassthroughLayer>();
            foreach (var cam in manager.GetComponentsInChildren<Camera>(true))
            {
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = new Color(0, 0, 0, 0);
            }

            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene, ScenePath);

            // Keep teammates' scenes; just make sure Main is first and enabled.
            var scenes = EditorBuildSettings.scenes.Where(s => s.path != ScenePath).ToList();
            scenes.Insert(0, new EditorBuildSettingsScene(ScenePath, true));
            EditorBuildSettings.scenes = scenes.ToArray();
            return true;
        }
    }
}
