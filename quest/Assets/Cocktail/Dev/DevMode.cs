using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace Cocktail.Dev
{
    /// <summary>
    /// One switch for everything behind the scenes: the DEMO panel, the test cubes and their sound, the fake-data fallback, the
    /// diagnostics text and the event log file. Off in normal use. Starts from <c>UiTheme.devMode</c>; B + Y pressed together on the
    /// controllers toggles it on the headset. Everything dev-only checks <see cref="On"/>, so turning it off hides it all.
    /// </summary>
    public static class DevMode
    {
        static bool _touched;
        public static bool On { get; private set; }
        public static event Action<bool> Changed;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() { On = false; _touched = false; Changed = null; }

        /// <summary>The starting value (from the theme). Ignored once something has called <see cref="Set"/>.</summary>
        public static void Init(bool on) { DevLog.Prepare(); if (!_touched) On = on; }

        public static void Set(bool on)
        {
            _touched = true;
            DevLog.Prepare();
            if (On == on) return;
            On = on;
            DevLog.Add("dev", on ? "DevMode ON" : "DevMode OFF");
            Changed?.Invoke(on);
        }

        public static void Toggle() => Set(!On);
    }

    /// <summary>
    /// Short event log for debugging on the headset: always kept in memory (last lines shown in the DEMO panel); while DevMode is on it is
    /// also written to logcat (tag [Dev]) and to <c>devlog.txt</c> in the app's files folder:
    /// <c>adb pull /sdcard/Android/data/&lt;package&gt;/files/devlog.txt</c>.
    /// </summary>
    public static class DevLog
    {
        const int Keep = 40;
        static readonly string[] Lines = new string[Keep];
        static readonly object Gate = new object();
        static int _next, _count;
        static StreamWriter _file;
        static bool _fileFailed;
        static string _path;

        /// <summary>Call on the main thread once (Application.persistentDataPath is main-thread only); later writes may come from any thread.</summary>
        public static void Prepare() { if (_path == null) _path = Path.Combine(Application.persistentDataPath, "devlog.txt"); }

        public static void Add(string tag, string message)
        {
            string line = DateTime.Now.ToString("HH:mm:ss.f") + " " + tag + " " + message;
            lock (Gate)
            {
                Lines[_next] = line; _next = (_next + 1) % Keep; if (_count < Keep) _count++;
                if (!DevMode.On) return;
                WriteFile(line);
            }
            Debug.Log("[Dev] " + line);
        }

        /// <summary>The last n lines, oldest first, each ending in a newline.</summary>
        public static string Recent(int n)
        {
            lock (Gate)
            {
                n = Mathf.Min(n, _count);
                var sb = new StringBuilder();
                for (int i = n; i >= 1; i--) sb.Append(Lines[(_next - i + Keep) % Keep]).Append('\n');
                return sb.ToString();
            }
        }

        static void WriteFile(string line)
        {
            if (_fileFailed || _path == null) return;
            try
            {
                if (_file == null) _file = new StreamWriter(_path, false) { AutoFlush = true };
                _file.WriteLine(line);
            }
            catch (Exception e) { _fileFailed = true; Debug.LogWarning("[Dev] log file unavailable: " + e.Message); }
        }
    }
}
