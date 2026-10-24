using UnityEngine;
using UnityEngine.Android;

namespace Cocktail.Stream
{
    /// <summary>
    /// Runtime permission requests, one dialog at a time (two simultaneous requests drop one on Android).
    /// Poll <see cref="Ensure"/> each frame from a component that needs the permission; it requests at most
    /// once per attempt window and returns true when granted. Off-device (Editor/Windows) it returns true.
    /// </summary>
    public static class PermissionHelper
    {
        public const string RecordAudio = "android.permission.RECORD_AUDIO";
        // Passthrough Camera API permission (Horizon OS v74+). Source: developers.meta.com PCA overview, 2026-10-24.
        public const string HeadsetCamera = "horizonos.permission.HEADSET_CAMERA";

        static string s_pending;
        static float s_pendingSince;
        static readonly System.Collections.Generic.HashSet<string> s_denied = new System.Collections.Generic.HashSet<string>();

        static bool OnAndroid => Application.platform == RuntimePlatform.Android;

        public static bool Has(string permission) =>
            !OnAndroid || Permission.HasUserAuthorizedPermission(permission);

        public static bool Denied(string permission) => s_denied.Contains(permission);

        /// <summary>Returns true if granted. Otherwise requests it (when no other request is on screen) and returns false.</summary>
        public static bool Ensure(string permission)
        {
            if (Has(permission)) { if (s_pending == permission) s_pending = null; return true; }
            if (!OnAndroid) return true;
            // A request that got no answer for 30 s is considered lost.
            if (s_pending != null && Time.realtimeSinceStartup - s_pendingSince > 30f) s_pending = null;
            if (s_pending == null && !s_denied.Contains(permission))
            {
                s_pending = permission;
                s_pendingSince = Time.realtimeSinceStartup;
                var cb = new PermissionCallbacks();
                string p = permission;
                cb.PermissionGranted += _ => { if (s_pending == p) s_pending = null; };
                cb.PermissionDenied += _ => { s_denied.Add(p); if (s_pending == p) s_pending = null; Debug.LogWarning($"[Permission] denied: {p}"); };
                cb.PermissionDeniedAndDontAskAgain += _ => { s_denied.Add(p); if (s_pending == p) s_pending = null; Debug.LogWarning($"[Permission] denied (don't ask again): {p}"); };
                Debug.Log($"[Permission] requesting {permission}");
                Permission.RequestUserPermission(permission, cb);
            }
            return false;
        }
    }
}
