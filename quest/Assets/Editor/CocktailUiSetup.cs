// Adds the headset UI to the scene and provides a headset-free self test.
//   Menu:   Cocktail > Add UI Components        (idempotent: creates/repairs the CocktailUI object, default theme, TMP Essentials)
//           Cocktail > Create Default UI Theme   (only the theme asset)
//   Batch:  Unity.exe -batchmode -nographics -projectPath <copy of quest/> -executeMethod Cocktail.Editor.CocktailUiSetup.AddBatch -quit
//           Unity.exe -batchmode -nographics -projectPath <copy of quest/> -executeMethod Cocktail.Editor.UiSelfTest.Run -logFile ui.log
//   Optional screenshots (needs a GPU, so no -nographics): ... -executeMethod Cocktail.Editor.UiSelfTest.Screenshots  (env COCKTAIL_SHOT_DIR)
// The self test needs TMP Essential Resources (run AddBatch once in the same project first).
using System.Collections.Generic;
using System.IO;
using Cocktail.Protocol;
using Cocktail.Stream;
using Cocktail.Ui;
using Cocktail.UserInput;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Cocktail.Editor
{
    public static class CocktailUiSetup
    {
        const string ScenePath = "Assets/Scenes/Main.unity";
        const string RootName = "CocktailUI";
        public const string ThemeDir = "Assets/Cocktail/UI/Resources";
        public const string ThemePath = ThemeDir + "/CocktailUiTheme.asset";
        const string TmpSettingsPath = "Assets/TextMesh Pro/Resources/TMP Settings.asset";
        const string TmpEssentials = "Packages/com.unity.ugui/Package Resources/TMP Essential Resources.unitypackage";

        [MenuItem("Cocktail/Add UI Components")]
        public static void AddMenu()
        {
            if (!Add()) Debug.LogError("[UiSetup] finished with errors");
        }

        /// <summary>Batch entry. Run WITHOUT -quit: the TMP Essentials import finishes asynchronously, then this exits by itself.</summary>
        public static void AddBatch()
        {
            if (!HasTmpEssentials() && File.Exists(TmpEssentials))
            {
                int frames = 0;
                AssetDatabase.importPackageCompleted += _ => FinishBatch();
                AssetDatabase.importPackageFailed += (_, err) => { Debug.LogError("[UiSetup] TMP import failed: " + err); EditorApplication.Exit(1); };
                EditorApplication.update += () => { if (++frames > 3000) { Debug.LogError("[UiSetup] TMP import timed out"); EditorApplication.Exit(1); } };
                EnsureTmpEssentials();
                return;
            }
            FinishBatch();
        }

        static void FinishBatch()
        {
            bool ok = false;
            try { ok = Add(); }
            catch (System.Exception e) { Debug.LogException(e); }
            EditorApplication.Exit(ok ? 0 : 1);
        }

        [MenuItem("Cocktail/Create Default UI Theme")]
        public static void CreateThemeMenu() { EnsureTheme(); }

        public static bool HasTmpEssentials() => File.Exists(TmpSettingsPath);

        public static void EnsureTmpEssentials()
        {
            if (HasTmpEssentials()) return;
            if (!File.Exists(TmpEssentials)) { Debug.LogWarning("[UiSetup] TMP Essential Resources package not found at " + TmpEssentials + "; use Window > TextMeshPro > Import TMP Essential Resources."); return; }
            Debug.Log("[UiSetup] importing TMP Essential Resources (default font asset and shaders) into Assets/TextMesh Pro");
            AssetDatabase.ImportPackage(TmpEssentials, false);
            AssetDatabase.Refresh();
        }

        public static UiTheme EnsureTheme()
        {
            var theme = AssetDatabase.LoadAssetAtPath<UiTheme>(ThemePath);
            if (theme != null) return theme;
            Directory.CreateDirectory(ThemeDir);
            theme = ScriptableObject.CreateInstance<UiTheme>();
            AssetDatabase.CreateAsset(theme, ThemePath);
            AssetDatabase.SaveAssets();
            Debug.Log("[UiSetup] created default theme " + ThemePath);
            return theme;
        }

        static bool Add()
        {
            var active = EditorSceneManager.GetActiveScene();
            if (string.IsNullOrEmpty(active.path) && AssetDatabase.LoadAssetAtPath<SceneAsset>(ScenePath) != null)
                active = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

            EnsureTmpEssentials();
            var theme = EnsureTheme();

            var root = GameObject.Find(RootName);
            if (root == null) { root = new GameObject(RootName); Undo.RegisterCreatedObjectUndo(root, "Add Cocktail UI"); }

            var ui = Ensure<CocktailUI>(root);
            var fake = Ensure<FakeSourceFeed>(root);
            var hub = Object.FindFirstObjectByType<HubClient>();
            var input = Object.FindFirstObjectByType<InputSender>();
            var rig = Object.FindFirstObjectByType<OVRCameraRig>();
            var cam = Object.FindFirstObjectByType<CameraStreamer>();
            if (hub == null) Debug.LogWarning("[UiSetup] no HubClient in the scene (run Cocktail > Add Streaming Components); the UI will run on the fake feed only.");
            if (rig == null) Debug.LogWarning("[UiSetup] no OVRCameraRig in the scene; run Cocktail > Configure Quest first. The UI falls back to Camera.main.");
            Wire(ui, ("theme", theme), ("hub", hub), ("input", input), ("rig", rig), ("cameraStreamer", cam), ("fake", fake));

            EditorSceneManager.MarkSceneDirty(active);
            bool saved = !string.IsNullOrEmpty(active.path) ? EditorSceneManager.SaveScene(active) : true;
            Debug.Log($"[UiSetup] '{RootName}' ready in scene '{active.path}' (saved={saved}): CocktailUI + FakeSourceFeed wired to " +
                      $"hub={(hub != null)} input={(input != null)} rig={(rig != null)} camera={(cam != null)} theme={ThemePath} tmpEssentials={HasTmpEssentials()}");
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
                if (p == null) { Debug.LogWarning($"[UiSetup] {target.GetType().Name}.{field} not found"); continue; }
                if (value != null && p.objectReferenceValue != value) p.objectReferenceValue = value;
            }
            so.ApplyModifiedPropertiesWithoutUndo();
        }
    }

    /// <summary>
    /// Headset-free check of the UI logic: builds CocktailUI with the fake feed in an empty scene (edit mode, no OVR, no hub), advances simulated
    /// time and asserts markers, arrows, banner changes, alerts, hover/select, HUD toggle and absence of logged errors. Prints
    /// [UISELFTEST] PASS/FAIL lines and exits 0/1. It does NOT test controllers, pinch, passthrough, real hub data or on-device rendering.
    /// </summary>
    public static class UiSelfTest
    {
        public static void Run()
        {
            int code = 1;
            try { code = RunInner() ? 0 : 1; }
            catch (System.Exception e) { Debug.LogException(e); }
            EditorApplication.Exit(code);
        }

        static bool RunInner()
        {
            var fails = new List<string>();
            void Check(bool ok, string what) { Debug.Log($"[UISELFTEST] {(ok ? "PASS" : "FAIL")} {what}"); if (!ok) fails.Add(what); }

            var logged = new List<string>();
            Application.LogCallback cb = (msg, stack, type) =>
            {
                if (type == LogType.Error || type == LogType.Exception || type == LogType.Assert) logged.Add(type + ": " + msg);
            };
            Application.logMessageReceived += cb;
            try
            {
                Check(CocktailUiSetup.HasTmpEssentials(), "TMP Essential Resources present (run Cocktail.Editor.CocktailUiSetup.AddBatch first if not)");
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var go = new GameObject("CocktailUI_SelfTest");
                var feed = go.AddComponent<FakeSourceFeed>(); feed.autoUpdate = false;
                var ui = go.AddComponent<CocktailUI>(); ui.pollInput = false; ui.useFake = true; Cocktail.Dev.DevMode.Set(true);
                void Step(float dt) { feed.Tick(dt); ui.Tick(dt); }
                void Run(float seconds, float dt = 0.05f, System.Action each = null) { for (float t = 0; t < seconds; t += dt) { Step(dt); each?.Invoke(); } }

                ui.EnsureInit();
                var theme = ui.Theme;
                ui.SpeakerOnly = false;   // the checks below are for the full HUD; the speaker-only HUD has its own section
                // ---- t = 0..3.9: listening, 3 markers, arrows for out-of-view sources
                var titles = new HashSet<string>();
                void Note() { if (!string.IsNullOrEmpty(ui.BannerText)) titles.Add(ui.BannerText.Split('|')[0].Trim()); }
                Run(3.9f, 0.05f, Note);
                Check(ui.UsingFake, "fake feed is the data source (no hub)");
                Check(ui.MarkerCount == 3, $"3 source markers exist (got {ui.MarkerCount})");
                Check(ui.ActiveArrowCount >= 2, $"edge arrows for out-of-view sources (got {ui.ActiveArrowCount}; s_2 and s_3 are out of view at head yaw 0)");
                var a2 = ui.GetArrow("s_2");
                Check(a2 != null && a2.DetailText.Contains("O'CLOCK"), $"arrow for s_2 shows a clock position ('{a2?.DetailText}')");
                var a3 = ui.GetArrow("s_3");
                Check(a3 != null && a3.DetailText.Contains("O'CLOCK"), $"arrow for s_3 shows a clock position ('{a3?.DetailText}')");
                Check(ui.RadarPipCount == 3, $"radar ring shows a pip per located source (got {ui.RadarPipCount})");
                Check(ui.RadarOutsideFovCount >= 2, $"radar pips outside the field-of-view cone for out-of-view sources (got {ui.RadarOutsideFovCount})");
                var m1 = ui.GetMarker("s_1"); int m1id = m1.GetInstanceID();
                var m2 = ui.GetMarker("s_2"); int m2id = m2.GetInstanceID();
                Check(m1.TitleText.ToLower().Contains("speech"), $"marker shows the label ('{m1.TitleText}')");
                Check(m1.StatusText.Contains("TRK"), $"tracking code on tracking marker ('{m1.StatusText}')");
                Check(!m1.IsExpanded, "unselected marker shows only the short label (data box collapsed)");
                Check(ui.BannerText.StartsWith("Listening"), $"banner starts as Listening ('{ui.BannerText}')");

                // ---- state banner changes through the demo script
                var m2states = new HashSet<string>();
                bool alertSeen = false, alertGone = false; string alertText = "";
                Run(20f - 3.9f, 0.05f, () =>
                {
                    Note();
                    if (ui.AlertText.Length > 0) { alertSeen = true; alertText = ui.AlertText; }
                });
                Check(titles.Contains("Selected") && titles.Contains("Playing"), $"banner showed Selected and Playing (saw: {string.Join(", ", titles)})");
                Check(alertSeen && alertText.Contains("Smoke"), $"alert banner shown with spoken text ('{alertText}')");
                Run(25f - 20f, 0.05f, () => { Note(); if (alertSeen && ui.AlertText.Length == 0) alertGone = true; });
                Check(alertGone, "alert auto-dismissed after its display time");
                Check(titles.Contains("Replay processing...") || titles.Contains("Replay processing.") || titles.Contains("Replay processing..") , $"banner showed Replay processing (saw: {string.Join(", ", titles)})");
                Check(titles.Contains("Replay playing"), "banner showed Replay playing");

                // ---- s_2: last seen, lost, back; marker identity must stay stable
                Run(41f - 25f, 0.05f, Note);
                m2states.Add(ui.GetMarker("s_2").StatusText);
                Check(ui.GetMarker("s_2").StatusText.Contains("LST"), $"s_2 marker shows last seen code ('{ui.GetMarker("s_2").StatusText}')");
                Run(48f - 41f, 0.05f, Note);
                Check(ui.GetMarker("s_2").StatusText.Contains("LOST"), $"s_2 marker shows lost ('{ui.GetMarker("s_2").StatusText}')");
                Check(titles.Contains("Target lost"), "banner showed Target lost");
                Run(54f - 48f, 0.05f, Note);
                Check(ui.GetMarker("s_2").StatusText.Contains("TRK"), "s_2 tracking again");
                Check(ui.GetMarker("s_1").GetInstanceID() == m1id && ui.GetMarker("s_2").GetInstanceID() == m2id, "markers are stable per source_id (same objects after 54 s, no re-create)");
                Check(ui.MarkerCount == 3, "still exactly 3 markers");

                // ---- gaze hover + select (fake mode)
                feed.ResetDemo();
                Run(1f);
                float b1 = feed.Current.Sources[0].BearingDeg.Value;
                ui.Head.rotation = Quaternion.Euler(0f, b1, 0f);
                Run(1.2f);
                Check(ui.HoveredSourceId == "s_1", $"gaze on s_1 hovers it (hover='{ui.HoveredSourceId}', head yaw {b1:F0})");
                bool sent = ui.TrySelectGazed("head");
                Run(0.3f);
                Check(sent && ui.SelectedSourceId == "s_1", $"select on gazed marker selects s_1 (sent={sent}, selected='{ui.SelectedSourceId}')");
                Check(ui.BannerText.StartsWith("Selected") || ui.BannerText.StartsWith("Playing"), $"banner reflects selection ('{ui.BannerText}')");
                Check(ui.ActiveArrowCount >= 1, "out-of-view arrows follow head yaw (s_2/s_3 still out of view)");
                ui.Head.rotation = Quaternion.identity;

                // ---- speaker-only HUD (the default): outline on speakers in view, arrow at the edge toward speakers out of view; nothing is auto-picked
                ui.SpeakerOnly = true; feed.ResetDemo(); ui.Head.rotation = Quaternion.identity; Run(2f);
                float rel1 = UiMath.NormalizeDeg(feed.Current.Sources[0].BearingDeg.Value - ui.Head.eulerAngles.y);
                float rel2 = UiMath.NormalizeDeg(feed.Current.Sources[1].BearingDeg.Value - ui.Head.eulerAngles.y);
                Check(!ui.WorldRoot.gameObject.activeSelf && ui.MarkerCount >= 0 && !ui.HeadLockedRoot.Find("RadarRing").gameObject.activeSelf, "speaker-only HUD hides the full HUD (markers, radar ring)");
                Check(ui.GetSpeaker("s_1") == null && ui.GetSpeaker("s_2") == null, "nothing is marked in the world until the wearer chooses");
                Check(ui.ListRowCount >= 2, $"the picker lists the talking sources (rows: {ui.ListRowCount})");
                feed.Select("s_1"); Run(1.5f);
                var sp1 = ui.GetSpeaker("s_1"); var sp2 = ui.GetSpeaker("s_2");
                Check(sp1 != null && sp1.OutlineVisible && !sp1.DotVisible, $"the chosen speaker s_1, in view, gets the rectangle (rel {rel1:F0} deg)");
                Check(sp1 != null && sp1.OutlineSize.x > 0.05f && sp1.OutlineSize.y > sp1.OutlineSize.x, $"the rectangle is taller than wide ({(sp1 != null ? sp1.OutlineSize.ToString() : "none")})");
                Check(sp2 == null || (!sp2.OutlineVisible && !sp2.DotVisible), $"the other speaker (s_2 at {rel2:F0} deg) gets no marker");
                Check(ui.GetSpeaker("s_3") == null, "music (not a speaker) gets no speaker indicator");
                Check(ui.SpeakerOutlineCount == 1 && ui.SpeakerDotCount == 0, $"exactly one rectangle and no arrow (got {ui.SpeakerOutlineCount} and {ui.SpeakerDotCount})");
                ui.Head.rotation = Quaternion.Euler(0f, 180f, 0f); Run(1.5f);
                Check(sp1 != null && sp1.DotVisible && !sp1.OutlineVisible, "turned away from the chosen speaker, an arrow at the edge replaces the rectangle");
                feed.Stop(); Run(0.8f);
                ui.SpeakerOnly = false; ui.Head.rotation = Quaternion.identity; Run(0.3f);

                // ---- glasses HUD mode
                ui.HudMode = HudMode.Glasses; Run(0.2f);
                Check(!ui.WorldRoot.gameObject.activeSelf && ui.GlassesText.Contains("S_1") == false && ui.GlassesText.Length > 3, $"glasses mode hides world markers and shows the panel ('{ui.GlassesText}')");
                Check(ui.GlassesText.ToLower().Contains("s_1"), "glasses panel names the selected target");
                ui.HudMode = HudMode.Full; Run(0.2f);
                Check(ui.WorldRoot.gameObject.activeSelf, "full mode restores the markers");

                // ---- contrast (WCAG) for the default theme; report only
                if (theme != null) ContrastReport(theme, Check);

                Object.DestroyImmediate(go);
            }
            finally { Application.logMessageReceived -= cb; }

            Check(logged.Count == 0, logged.Count == 0 ? "no errors/exceptions logged" : "logged errors: " + string.Join(" || ", logged));
            Debug.Log($"[UISELFTEST] {(fails.Count == 0 ? "ALL PASSED" : fails.Count + " FAILED")}");
            return fails.Count == 0;
        }

        static void ContrastReport(UiTheme t, System.Action<bool, string> check)
        {
            var white = Color.white; var black = Color.black;
            void Pair(string name, Color fg, Color bg, float min)
            {
                float cw = UiMath.Contrast(UiMath.Over(fg, UiMath.Over(bg, white)), UiMath.Over(bg, white));
                float cb = UiMath.Contrast(UiMath.Over(fg, UiMath.Over(bg, black)), UiMath.Over(bg, black));
                float worst = Mathf.Min(cw, cb);
                Debug.Log($"[UISELFTEST] CONTRAST {name}: {worst:F2}:1 (over white {cw:F2}, over black {cb:F2}) {(worst >= min ? "ok" : "BELOW " + min)}");
            }
            // text pairs (>= 4.5:1 wanted) over the worst passthrough (white or black behind the 85% plate)
            Pair("text-light on plate", t.textLight, t.plateColor, 4.5f);
            Pair("text-secondary on plate", t.TextSecondary, t.plateColor, 4.5f);
            var solidOrange = t.alertColor; solidOrange.a = 1f;
            Pair("text-dark on alert orange", t.textDark, solidOrange, 4.5f);
            // non-text accents against the plate (>= 3:1 wanted)
            Pair("speech cyan on plate", t.speechColor, t.plateColor, 3f);
            Pair("music hot pink on plate", t.otherSoundColor, t.plateColor, 3f);
            Pair("selected lime on plate", t.selectedColor, t.plateColor, 3f);
            Pair("alert orange on plate", t.alertColor, t.plateColor, 3f);
            Pair("lost muted teal on plate", t.lostColor, t.plateColor, 3f);
            Pair("outline navy on plate", t.outlineColor, t.plateColor, 1.5f);
            // the hard gate: primary text on plate over the worst case background
            float c = Mathf.Min(UiMath.Contrast(t.textLight, UiMath.Over(t.plateColor, white)), UiMath.Contrast(t.textLight, UiMath.Over(t.plateColor, black)));
            check(c >= 4.5f, $"primary text contrast on plate >= 4.5:1 (worst case {c:F2}:1)");
            float c2 = UiMath.Contrast(t.textDark, solidOrange);
            check(c2 >= 4.5f, $"dark text on orange alert >= 4.5:1 ({c2:F2}:1)");
        }

        // ------------------------------------------------------------------ screenshots (design review only)
        public static void Screenshots()
        {
            int code = 1;
            try { RenderShots(); code = 0; }
            catch (System.Exception e) { Debug.LogException(e); }
            EditorApplication.Exit(code);
        }

        static void RenderShots()
        {
            string dir = System.Environment.GetEnvironmentVariable("COCKTAIL_SHOT_DIR");
            if (string.IsNullOrEmpty(dir)) dir = Path.Combine(Application.dataPath, "../Shots");
            Directory.CreateDirectory(dir);
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var go = new GameObject("CocktailUI_Shots");
            var feed = go.AddComponent<FakeSourceFeed>(); feed.autoUpdate = false;
            var ui = go.AddComponent<CocktailUI>(); ui.pollInput = false; ui.useFake = true; Cocktail.Dev.DevMode.Set(true);
            ui.EnsureInit();
            var camGo = new GameObject("ShotCam");
            var cam = camGo.AddComponent<Camera>();
            cam.transform.SetParent(ui.Head, false);
            cam.fieldOfView = 70f; cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = new Color(0.42f, 0.44f, 0.46f);
            cam.nearClipPlane = 0.05f;
            var rt = new RenderTexture(1600, 900, 24);
            cam.targetTexture = rt;

            void Step(float seconds) { for (float t = 0; t < seconds; t += 0.05f) { feed.Tick(0.05f); ui.Tick(0.05f); } }
            void Shot(string name)
            {
                cam.Render();
                RenderTexture.active = rt;
                var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0); tex.Apply();
                File.WriteAllBytes(Path.Combine(dir, name + ".png"), tex.EncodeToPNG());
                RenderTexture.active = null;
                Object.DestroyImmediate(tex);
                Debug.Log("[UISHOT] " + Path.Combine(dir, name + ".png"));
            }
            Step(8f); Shot("1_front_playing");
            ui.Head.rotation = Quaternion.Euler(0, feed.Current.Sources[0].BearingDeg.Value, 0); Step(1.5f); Shot("2_gaze_s1");
            ui.Head.rotation = Quaternion.Euler(0, -60, 0); Step(1.5f); Shot("3_turned_left");
            ui.Head.rotation = Quaternion.identity; Step(6.5f); Shot("4_alert");
            Step(10f); ui.HudMode = HudMode.Glasses; Step(0.3f); Shot("5_glasses");
            ui.HudMode = HudMode.Full; Step(23f); Shot("6_lost");
            Object.DestroyImmediate(camGo); Object.DestroyImmediate(go);
        }
    }
}
