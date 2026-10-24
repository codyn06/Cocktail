// Reproducible Quest project configuration. Run from the menu (Cocktail > Configure Quest)
// or in batch mode:
//   Unity.exe -batchmode -projectPath quest -executeMethod Cocktail.Editor.QuestProjectSetup.ConfigureBatch
// and build a smoke-test APK with -executeMethod Cocktail.Editor.QuestProjectSetup.BuildBatch.
using System.Collections.Generic;
using System.IO;
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
        public static void ConfigureMenu() => _ = Configure();

        public static async void ConfigureBatch()
        {
            var ok = await Configure();
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

            foreach (var group in new[] { BuildTargetGroup.Android, BuildTargetGroup.Standalone })
            {
                AssignOpenXR(group);
                EnableMetaFeatures(group);
            }
            ConfigureMetaQuestFeature();
            ConfigureOVRProject();
            CreateScene();

            await OVRProjectSetup.FixAllAsync(BuildTargetGroup.Android);
            AssetDatabase.SaveAssets();
            Debug.Log("[QuestSetup] Done.");
            return true;
        }

        static void AssignOpenXR(BuildTargetGroup group)
        {
            if (!EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.k_SettingsKey,
                    out XRGeneralSettingsPerBuildTarget perTarget))
            {
                Directory.CreateDirectory("Assets/XR");
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

        static void EnableMetaFeatures(BuildTargetGroup group)
        {
            FeatureHelpers.RefreshFeatures(group);
            var set = OpenXRFeatureSetManager.GetFeatureSetWithId(group, MetaFeatureSetId);
            if (set == null)
            {
                Debug.LogWarning($"[QuestSetup] Meta XR feature group not found for {group}");
                return;
            }
            set.isEnabled = true;
            OpenXRFeatureSetManager.SetFeaturesFromEnabledFeatureSets(group);
        }

        static void ConfigureMetaQuestFeature()
        {
            var feature = OpenXRSettings.GetSettingsForBuildTargetGroup(BuildTargetGroup.Android)
                .GetFeature<MetaQuestFeature>();
            if (feature == null) return;
            feature.enabled = true;

            // targetDevices is internal; edit it through serialization.
            var so = new SerializedObject(feature);
            var devices = so.FindProperty("targetDevices");
            for (var i = 0; i < devices.arraySize; i++)
            {
                var device = devices.GetArrayElementAtIndex(i);
                var name = device.FindPropertyRelative("manifestName").stringValue;
                device.FindPropertyRelative("enabled").boolValue = ManifestDevices.Contains(name);
            }
            so.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(feature);
        }

        static void ConfigureOVRProject()
        {
            var config = OVRProjectConfig.CachedProjectConfig;
            config.targetDeviceTypes = new List<OVRProjectConfig.DeviceType>(TargetDevices);
            config.insightPassthroughSupport = OVRProjectConfig.FeatureSupport.Supported;
            OVRProjectConfig.CommitProjectConfig(config);
        }

        static void CreateScene()
        {
            if (File.Exists(ScenePath)) return;
            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var rigPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(CameraRigPath);
            var rig = (GameObject)PrefabUtility.InstantiatePrefab(rigPrefab);
            var manager = rig.GetComponent<OVRManager>() ?? rig.AddComponent<OVRManager>();
            manager.isInsightPassthroughEnabled = true;

            var light = new GameObject("Directional Light").AddComponent<Light>();
            light.type = LightType.Directional;

            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        }
    }
}
