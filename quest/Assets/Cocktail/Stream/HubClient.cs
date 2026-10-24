using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Cocktail.Protocol;
using UnityEngine;

namespace Cocktail.Stream
{
    /// <summary>
    /// Scene entry point for the hub link. Owns the HubConnection (5 reconnecting sockets), the hub address
    /// and the main-thread dispatch of hub events. Other components (MicStreamer, PoseStreamer, CameraStreamer,
    /// InputSender, AudioOutPlayer, the UI) talk to the hub through <see cref="Connection"/> or the helpers here.
    ///
    /// Hub address precedence (first that is set wins):
    ///   1. Android intent extra:  adb shell am start -n &lt;pkg&gt;/com.unity3d.player.UnityPlayerGameActivity -e hub_host 192.168.137.1
    ///   2. file &lt;persistentDataPath&gt;/hub_host.txt   (one line)
    ///   3. PlayerPrefs "cocktail.hub_host" (set by SetHost)
    ///   4. the serialized default below (192.168.137.1 = laptop address on a Windows mobile hotspot;
    ///      use 127.0.0.1 with `adb reverse tcp:8765 tcp:8765` for the USB-cable link).
    /// </summary>
    [DefaultExecutionOrder(-100)]
    public class HubClient : MonoBehaviour
    {
        public const string PrefsHostKey = "cocktail.hub_host";
        const string PrefsEpochKey = "cocktail.epoch_base";

        [Tooltip("Default hub host/IP when no override is set (see class comment).")]
        [SerializeField] string defaultHost = "192.168.137.1";
        [SerializeField] int port = ProtocolInfo.DefaultPort;
        [Tooltip("Device name sent in Hello.")]
        [SerializeField] string deviceName = "quest3s";
        [Tooltip("Max seconds the pose socket waits for the mic/camera to report before sending Hello anyway.")]
        [SerializeField] float helloWaitSeconds = 6f;

        [SerializeField] MicStreamer mic;
        [SerializeField] CameraStreamer cameraStreamer;

        public HubConnection Connection { get; private set; }
        public string Host => Connection != null ? Connection.Host : defaultHost;

        public event System.Action<SourcesUpdated> OnSources;
        public event System.Action<StateChanged> OnState;
        public event System.Action<TargetUpdated> OnTarget;
        public event System.Action<AlertFired> OnAlert;
        public event System.Action<ReplayReady> OnReplayReady;
        public event System.Action<Metrics> OnMetrics;
        /// <summary>Latest events, for UI that polls instead of subscribing.</summary>
        public SourcesUpdated LastSources { get; private set; }
        public StateChanged LastState { get; private set; }
        public Metrics LastMetrics { get; private set; }

        public static HubClient Instance { get; private set; }

        void Awake()
        {
            Instance = this;
            if (mic == null) mic = GetComponent<MicStreamer>();
            if (cameraStreamer == null) cameraStreamer = GetComponent<CameraStreamer>();

            long epochBase = PlayerPrefs.GetInt(PrefsEpochKey, 0);
            // Reserve a block of epochs per app launch so an app restart is also seen as a new epoch.
            PlayerPrefs.SetInt(PrefsEpochKey, (int)(epochBase + 1000));
            PlayerPrefs.Save();

            Connection = new HubConnection(ResolveHost(), epochBase, port);
            Connection.HelloProvider = BuildHelloAsync;
        }

        void OnEnable() { Connection.Start(); Debug.Log($"[HubClient] hub={Connection.Host}:{port}"); }
        void OnDisable() { Connection?.Stop(); }

        void Update()
        {
            while (Connection.IncomingEvents.TryDequeue(out var ev))
            {
                switch (ev)
                {
                    case SourcesUpdated e: LastSources = e; OnSources?.Invoke(e); break;
                    case StateChanged e: LastState = e; OnState?.Invoke(e); break;
                    case TargetUpdated e: OnTarget?.Invoke(e); break;
                    case AlertFired e: OnAlert?.Invoke(e); break;
                    case ReplayReady e: OnReplayReady?.Invoke(e); break;
                    case Metrics e: LastMetrics = e; OnMetrics?.Invoke(e); break;
                }
            }
        }

        /// <summary>Change the hub address at runtime (persisted); all sockets reconnect.</summary>
        public void SetHost(string host)
        {
            host = host.Trim();
            if (host.Length == 0) return;
            PlayerPrefs.SetString(PrefsHostKey, host);
            PlayerPrefs.Save();
            Connection.SetHost(host);
            Debug.Log($"[HubClient] hub host set to {host}");
        }

        public void SendCommand(object command) => Connection.SendCommand(command);

        string ResolveHost()
        {
            string h = null;
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                using (var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                using (var activity = player.GetStatic<AndroidJavaObject>("currentActivity"))
                using (var intent = activity.Call<AndroidJavaObject>("getIntent"))
                    h = intent.Call<string>("getStringExtra", "hub_host");
            }
            catch (System.Exception e) { Debug.LogWarning("[HubClient] intent read failed: " + e.Message); }
#endif
            if (string.IsNullOrWhiteSpace(h))
            {
                try
                {
                    var f = Path.Combine(Application.persistentDataPath, "hub_host.txt");
                    if (File.Exists(f)) h = File.ReadAllText(f).Trim();
                }
                catch { }
            }
            if (string.IsNullOrWhiteSpace(h)) h = PlayerPrefs.GetString(PrefsHostKey, "");
            if (string.IsNullOrWhiteSpace(h)) h = defaultHost;
            return h.Trim();
        }

        // Runs on a network thread right after the pose socket connects: wait until the mic (and camera, if used)
        // have reported their real parameters, then build Hello from them.
        async Task<Hello> BuildHelloAsync(CancellationToken ct)
        {
            // No Unity API calls here (not the main thread): only reference checks and volatile fields.
            bool hasMic = !ReferenceEquals(mic, null);
            bool hasCam = !ReferenceEquals(cameraStreamer, null);
            float waited = 0;
            while (waited < helloWaitSeconds && !ct.IsCancellationRequested)
            {
                bool micOk = !hasMic || !mic.Active || mic.Resolved;
                bool camOk = !hasCam || !cameraStreamer.Active || cameraStreamer.Resolved;
                if (micOk && camOk) break;
                await Task.Delay(100, ct).ConfigureAwait(false);
                waited += 0.1f;
            }
            bool micRes = hasMic && mic.Resolved;
            bool camAvail = hasCam && cameraStreamer.CameraAvailable;
            return new Hello
            {
                Device = deviceName,
                MicChannels = micRes ? mic.Channels : 1,
                MicRaw = micRes && mic.IsRaw,
                Camera = camAvail,
                CameraHfovDeg = camAvail && cameraStreamer.HfovDeg > 0 ? cameraStreamer.HfovDeg : (float?)null,
            };
        }
    }
}
