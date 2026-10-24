using System.Text;
using Cocktail.Protocol;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Cocktail.Stream
{
    public enum MicBackend
    {
        /// <summary>Native AudioRecord on Android first; fall back to Unity Microphone if it fails or yields no data.</summary>
        Auto,
        /// <summary>Native Android AudioRecord only (UNPROCESSED source, multi-channel attempt).</summary>
        Native,
        /// <summary>Unity Microphone API only (processed mono on Quest).</summary>
        UnityMicrophone,
    }

    /// <summary>
    /// B1: captures the microphone and sends ~20 ms AudioChunk frames to /ingest/audio.
    ///
    /// device_ts_us of an AudioChunk = capture time of the FIRST sample in that chunk, on the shared headset clock
    /// (<see cref="ClockUs"/>). The hub can compute the end time as ts + frames/sample_rate.
    ///
    /// How mic sample time is aligned to the clock:
    ///  - Native path: AudioRecord.getTimestamp(AudioTimestamp.TIMEBASE_MONOTONIC) returns (frame position, CLOCK_MONOTONIC ns)
    ///    pairs, so the first frame of every chunk gets an exact monotonic time; the plugin converts it to microseconds and
    ///    this class adds the measured Java-nanoTime-to-Stopwatch offset (logged at start).
    ///  - Unity Microphone path: no timestamps exist, so sample n gets t0 + n / rate where t0 is the minimum over all polls of
    ///    (poll time - samplesCaptured / rate). Polling only ever makes this estimate too late, so the minimum converges on the
    ///    true start. Hardware input latency (tens of ms) is not included on either path.
    /// </summary>
    public class MicStreamer : MonoBehaviour
    {
        [SerializeField] HubClient hub;
        [SerializeField] MicBackend backend = MicBackend.Auto;
        [Tooltip("Chunk length in ms (20 ms = 960 frames at 48 kHz).")]
        [SerializeField] int chunkMs = 20;
        [SerializeField] int preferredSampleRate = 48000;
        [Tooltip("Native path: try this many channels first, then fewer.")]
        [SerializeField] int wantNativeChannels = 4;
        [SerializeField] bool preferUnprocessed = true;
        [Tooltip("Auto: seconds to wait for the first native chunk before falling back to Unity Microphone.")]
        [SerializeField] float nativeNoDataTimeout = 3f;

        // ---- report (read by HubClient on a network thread: plain volatile fields only) ----
        public volatile bool Active;
        public volatile bool Resolved;
        public volatile int Channels = 1;
        public volatile int SampleRate;
        public volatile bool IsRaw;
        public string BackendInUse = "none";
        public string Report = "mic: not started";
        public string NativeStatusJson = "";
        public string TimestampMode = "";
        public bool Streaming => _started && (_java == null || _gotNativeData);
        public long ChunksSent { get; private set; }
        public float[] LevelDb = new float[0];      // dBFS per channel (smoothed)
        public float InterChannelCorrelation = float.NaN; // ch0/ch1 correlation over ~1 s; ~1.0 = identical (not a real array)

        bool _started;
        bool _permWarned;
        float _startedAt;
        bool _gotNativeData;

        // native
        AndroidJavaObject _java;
        System.IntPtr _pollMethod;
        System.IntPtr _javaObj;
        long _javaToClockOffsetUs;
        readonly jvalue[] _noArgs = new jvalue[0];

        // unity microphone
        string _device;
        AudioClip _clip;
        int _lastPos;
        long _wraps;
        long _samplesTaken;
        double _t0Us = double.MaxValue;
        bool _unityWaitingForData;

        // correlation accumulators
        double _sxy, _sxx, _syy; int _corrN;

        void Awake() { if (hub == null) hub = GetComponent<HubClient>(); if (hub == null) hub = FindFirstObjectByType<HubClient>(); }
        void OnEnable() { Active = true; }
        void OnDisable() { Active = false; StopAll(); }

        void OnApplicationPause(bool paused)
        {
            if (paused) StopAll();
        }

        void Update()
        {
            if (!_started)
            {
                if (!PermissionHelper.Ensure(PermissionHelper.RecordAudio))
                {
                    if (PermissionHelper.Denied(PermissionHelper.RecordAudio) && !_permWarned)
                    {
                        _permWarned = true;
                        Report = "mic: RECORD_AUDIO permission denied";
                        Debug.LogError("[MicStreamer] " + Report);
                        Resolved = true; // let Hello go out (mic_channels=1, raw=false)
                    }
                    else if (!_permWarned) Report = "mic: waiting for RECORD_AUDIO permission";
                    return;
                }
                StartBackend();
                return;
            }

            if (_java != null)
            {
                PollNative();
                if (backend == MicBackend.Auto && !_gotNativeData && Time.realtimeSinceStartup - _startedAt > nativeNoDataTimeout)
                {
                    Debug.LogWarning("[MicStreamer] native capture produced no data; falling back to Unity Microphone");
                    StopAll();
                    StartUnityMic();
                }
            }
            else if (_clip != null) PollUnityMic();
        }

        // ------------------------------------------------------------------ start / stop

        void StartBackend()
        {
            bool onDevice = Application.platform == RuntimePlatform.Android;
            var want = backend;
            string forced = ReadIntentMic();
            if (forced == "unity") want = MicBackend.UnityMicrophone;
            else if (forced == "native") want = MicBackend.Native;

            if (want != MicBackend.UnityMicrophone && onDevice)
            {
                if (StartNative()) return;
                if (want == MicBackend.Native) { Report = "mic: native start failed (see log)"; _started = false; Resolved = true; return; }
            }
            StartUnityMic();
        }

        static string ReadIntentMic()
        {
            if (Application.platform != RuntimePlatform.Android) return null;
            try
            {
                using (var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                using (var activity = player.GetStatic<AndroidJavaObject>("currentActivity"))
                using (var intent = activity.Call<AndroidJavaObject>("getIntent"))
                    return intent.Call<string>("getStringExtra", "mic"); // adb shell am start ... -e mic unity|native
            }
            catch { return null; }
        }

        bool StartNative()
        {
            try
            {
                _java = new AndroidJavaObject("com.cocktail.mic.CocktailMic");
                string json = _java.Call<string>("start", wantNativeChannels, preferredSampleRate, chunkMs, preferUnprocessed);
                NativeStatusJson = json;
                var st = JObject.Parse(json);
                if (!(bool)st["ok"])
                {
                    Debug.LogWarning("[MicStreamer] native AudioRecord failed: " + json);
                    _java.Dispose(); _java = null;
                    return false;
                }
                Channels = (int)st["channels"];
                SampleRate = (int)st["rate"];
                IsRaw = (bool)st["raw"];
                TimestampMode = (string)st["ts_mode"];
                BackendInUse = "native AudioRecord/" + (string)st["source"];
                _javaObj = _java.GetRawObject();
                _pollMethod = AndroidJNIHelper.GetMethodID(_java.GetRawClass(), "poll", "()[B", false);
                CalibrateJavaClock();
                LevelDb = new float[Channels];
                _started = true; _startedAt = Time.realtimeSinceStartup; _gotNativeData = false;
                // Resolved (-> Hello may be sent) only after the first chunk really arrives, so a native failure that
                // falls back to Unity Microphone never leaves a Hello claiming N raw channels.
                Report = $"mic: {BackendInUse} {Channels}ch {SampleRate}Hz raw={IsRaw}";
                Debug.Log($"[MicStreamer] REPORT channels={Channels} rate={SampleRate} raw={IsRaw} source={(string)st["source"]} " +
                          $"timestamps={TimestampMode} javaClockOffsetUs={_javaToClockOffsetUs} status={json}");
                return true;
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("[MicStreamer] native start exception: " + e);
                _java?.Dispose(); _java = null;
                return false;
            }
        }

        void CalibrateJavaClock()
        {
            // Offset such that clockUs = javaNanoTime/1000 + offset. Take the trial with the smallest JNI round trip.
            long bestRtt = long.MaxValue, bestOffset = 0;
            for (int i = 0; i < 8; i++)
            {
                long a = ClockUs.RawMonotonicNowUs;
                long javaUs = _java.Call<long>("nowNs") / 1000;
                long b = ClockUs.RawMonotonicNowUs;
                long rtt = b - a;
                if (rtt < bestRtt) { bestRtt = rtt; bestOffset = (a + b) / 2 - javaUs; }
            }
            // bestOffset maps Java raw us -> raw Stopwatch us; subtract the clock origin to land on ClockUs.Now.
            _javaToClockOffsetUs = ClockUs.FromRawUs(bestOffset) + 0; // FromRawUs(x) = x - origin
            Debug.Log($"[MicStreamer] clock offset Java nanoTime->ClockUs: {bestOffset} us raw (JNI rtt {bestRtt} us); " +
                      "a value near 0 means Java nanoTime and Stopwatch share CLOCK_MONOTONIC");
        }

        void StartUnityMic()
        {
            string[] devices = Microphone.devices;
            Debug.Log("[MicStreamer] Unity Microphone devices: " + string.Join(", ", devices));
            if (devices.Length == 0)
            {
                Report = "mic: no Unity Microphone device";
                Debug.LogWarning("[MicStreamer] " + Report);
                Resolved = true; // Hello with mic_channels=1, raw=false
                return;
            }
            _device = null; // default device
            Microphone.GetDeviceCaps(_device, out int minF, out int maxF);
            int rate = preferredSampleRate;
            if (minF != 0 || maxF != 0) rate = Mathf.Clamp(preferredSampleRate, minF, maxF);
            _clip = Microphone.Start(_device, true, 2, rate);
            if (_clip == null) { Report = "mic: Microphone.Start failed"; Debug.LogError("[MicStreamer] " + Report); Resolved = true; return; }
            _lastPos = 0; _wraps = 0; _samplesTaken = 0; _t0Us = double.MaxValue;
            _unityWaitingForData = true;
            _started = true; _startedAt = Time.realtimeSinceStartup;
            Channels = _clip.channels; SampleRate = _clip.frequency; IsRaw = false;
            BackendInUse = "Unity Microphone";
            TimestampMode = "poll-min-estimate";
            LevelDb = new float[Channels];
            Resolved = true;
            Report = $"mic: Unity Microphone {Channels}ch {SampleRate}Hz raw=False (device caps {minF}-{maxF} Hz)";
            Debug.Log($"[MicStreamer] REPORT channels={Channels} rate={SampleRate} raw=False backend=UnityMicrophone caps={minF}-{maxF}");
        }

        void StopAll()
        {
            if (_java != null)
            {
                try { _java.Call("stop"); } catch { }
                _java.Dispose(); _java = null;
            }
            if (_clip != null)
            {
                Microphone.End(_device);
                _clip = null;
            }
            _started = false;
        }

        // ------------------------------------------------------------------ native polling

        void PollNative()
        {
            // At most 25 chunks per frame (0.5 s) so a stall cannot freeze the main thread.
            for (int i = 0; i < 25; i++)
            {
                System.IntPtr arr = AndroidJNI.CallObjectMethod(_javaObj, _pollMethod, _noArgs);
                if (arr == System.IntPtr.Zero) break;
                byte[] data;
                try { data = AndroidJNI.FromByteArray(arr); }
                finally { AndroidJNI.DeleteLocalRef(arr); }
                if (data == null || data.Length <= 8) continue;
                if (!_gotNativeData) { _gotNativeData = true; Resolved = true; }

                long javaUs = System.BitConverter.ToInt64(data, 0); // little-endian on all Unity platforms
                long tsUs = javaUs + _javaToClockOffsetUs;
                int pcmBytes = data.Length - 8;
                var pcm = new byte[pcmBytes];
                System.Buffer.BlockCopy(data, 8, pcm, 0, pcmBytes);
                Emit(pcm, pcmBytes, tsUs);
            }
        }

        // ------------------------------------------------------------------ Unity Microphone polling

        void PollUnityMic()
        {
            int pos = Microphone.GetPosition(_device);
            if (_unityWaitingForData)
            {
                if (pos <= 0) return;
                _unityWaitingForData = false;
            }
            if (pos < _lastPos) _wraps++;
            _lastPos = pos;
            int clipLen = _clip.samples;
            long captured = _wraps * clipLen + pos;            // frames the mic has delivered so far
            double nowUs = ClockUs.Now;
            double cand = nowUs - captured * 1e6 / SampleRate; // time of frame 0 if polling had zero delay
            if (cand < _t0Us) _t0Us = cand;

            int chunkFrames = SampleRate * chunkMs / 1000;
            int ch = _clip.channels;
            while (captured - _samplesTaken >= chunkFrames)
            {
                int readPos = (int)(_samplesTaken % clipLen);
                var f = new float[chunkFrames * ch];
                int first = Mathf.Min(chunkFrames, clipLen - readPos);
                if (first == chunkFrames)
                {
                    _clip.GetData(f, readPos);
                }
                else
                {
                    var a = new float[first * ch]; _clip.GetData(a, readPos);
                    var b = new float[(chunkFrames - first) * ch]; _clip.GetData(b, 0);
                    System.Array.Copy(a, 0, f, 0, a.Length);
                    System.Array.Copy(b, 0, f, a.Length, b.Length);
                }
                var pcm = new byte[chunkFrames * ch * 2];
                for (int i = 0; i < f.Length; i++)
                {
                    int s = Mathf.Clamp(Mathf.RoundToInt(f[i] * 32767f), -32768, 32767);
                    pcm[2 * i] = (byte)s; pcm[2 * i + 1] = (byte)(s >> 8);
                }
                long tsUs = (long)(_t0Us + _samplesTaken * 1e6 / SampleRate);
                _samplesTaken += chunkFrames;
                Emit(pcm, pcm.Length, tsUs);
            }
        }

        // ------------------------------------------------------------------ send + meters

        void Emit(byte[] pcm, int pcmBytes, long tsUs)
        {
            int ch = Channels;
            int frames = pcmBytes / (2 * ch);
            if (frames < 1) return;
            UpdateMeters(pcm, frames, ch);
            ChunksSent++;
            if (hub == null || hub.Connection == null) return;
            hub.Connection.SendAudio(new AudioChunk
            {
                DeviceTsUs = tsUs,
                SampleRate = SampleRate,
                Channels = ch,
                SampleFormat = "s16le",
                Frames = frames,
            }, pcm, pcmBytes);
        }

        void UpdateMeters(byte[] pcm, int frames, int ch)
        {
            if (LevelDb.Length != ch) LevelDb = new float[ch];
            for (int c = 0; c < ch; c++)
            {
                double sum = 0;
                for (int i = 0; i < frames; i++)
                {
                    int o = 2 * (i * ch + c);
                    short s = (short)(pcm[o] | (pcm[o + 1] << 8));
                    sum += (double)s * s;
                }
                float db = 20f * Mathf.Log10(Mathf.Max(1e-6f, (float)System.Math.Sqrt(sum / frames) / 32768f));
                LevelDb[c] = Mathf.Lerp(LevelDb[c], db, 0.2f);
            }
            if (ch >= 2)
            {
                for (int i = 0; i < frames; i++)
                {
                    int o = 2 * i * ch;
                    double x = (short)(pcm[o] | (pcm[o + 1] << 8));
                    double y = (short)(pcm[o + 2] | (pcm[o + 3] << 8));
                    _sxy += x * y; _sxx += x * x; _syy += y * y;
                }
                _corrN += frames;
                if (_corrN >= SampleRate)
                {
                    InterChannelCorrelation = (float)(_sxy / System.Math.Sqrt(System.Math.Max(1e-9, _sxx * _syy)));
                    Debug.Log($"[MicStreamer] ch0/ch1 correlation over 1 s = {InterChannelCorrelation:F3} " +
                              "(~1.000 and equal levels = duplicated channel, not independent mics)");
                    _sxy = _sxx = _syy = 0; _corrN = 0;
                }
            }
        }

        public string LevelsText()
        {
            if (LevelDb == null || LevelDb.Length == 0) return "-";
            var sb = new StringBuilder();
            for (int i = 0; i < LevelDb.Length; i++) sb.Append(LevelDb[i].ToString("F0")).Append(i + 1 < LevelDb.Length ? "/" : "");
            sb.Append(" dB");
            if (!float.IsNaN(InterChannelCorrelation)) sb.Append("  corr ").Append(InterChannelCorrelation.ToString("F2"));
            return sb.ToString();
        }
    }
}
