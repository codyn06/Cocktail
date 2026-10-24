// Regenerates the Meta AI Agent Bridge access token and syncs it into DevAgentSettings.
// Meta's own Regenerate button lives in the AI Agent Bridge settings; this is a shortcut.
// Both Meta types are internal, so they are reached through reflection.
using System;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace Cocktail.Editor
{
    public static class RotateDevAgentToken
    {
        const BindingFlags Static = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
        const BindingFlags Instance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        [MenuItem("Cocktail/Rotate DevAgent Token")]
        public static void Rotate()
        {
            var remote = FindType("Meta.XR.AI.AgentBridge.RemoteAgentSettings");
            var runtime = FindType("Meta.XR.ImmersiveDebugger.DevAgent.RuntimeSettings");
            if (remote == null || runtime == null)
            {
                Debug.LogError("[RotateToken] Meta AgentBridge/DevAgent types not found; the SDK layout changed.");
                return;
            }

            var token = (string)remote.GetMethod("RegenerateToken", Static).Invoke(null, null);

            var settings = runtime.GetProperty("Instance", Static)?.GetValue(null) as UnityEngine.Object;
            if (settings != null)
            {
                runtime.GetMethod("SetAccessToken", Instance).Invoke(settings, new object[] { token });
                EditorUtility.SetDirty(settings);
                AssetDatabase.SaveAssets();
            }

            // Show only a prefix so the full token never lands in logs.
            Debug.Log($"[RotateToken] New DevAgent token {token.Substring(0, 6)}…; rebuild the APK.");
            EditorUtility.DisplayDialog("DevAgent token rotated",
                $"New token starts with {token.Substring(0, 6)}. Rebuild the APK so it uses the new token.", "OK");
        }

        static Type FindType(string fullName) => AppDomain.CurrentDomain.GetAssemblies()
            .Select(a => a.GetType(fullName, false)).FirstOrDefault(t => t != null);
    }
}
