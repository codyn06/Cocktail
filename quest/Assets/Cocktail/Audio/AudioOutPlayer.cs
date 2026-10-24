using System.Threading;
using Cocktail.Protocol;
using Cocktail.Stream;
using UnityEngine;

namespace Cocktail.Audio
{
    /// <summary>Single-producer / single-consumer float ring: network thread writes, audio thread reads.</summary>
    public sealed class FloatRing
    {
        readonly float[] _buf;
        readonly long _mask;
        long _w, _r;

        public FloatRing(int capacityPow2)
        {
            _buf = new float[capacityPow2];
            _mask = capacityPow2 - 1;
        }

        public int Count => (int)(Volatile.Read(ref _w) - Volatile.Read(ref _r));

        public bool Write(float[] src, int n)
        {
            long w = Volatile.Read(ref _w), r = Volatile.Read(ref _r);
            if (_buf.Length - (w - r) < n) return false;
            for (int i = 0; i < n; i++) _buf[(w + i) & _mask] = src[i];
            Volatile.Write(ref _w, w + n);
            return true;
        }

        public int Read(float[] dst, int n)
        {
            long w = Volatile.Read(ref _w), r = Volatile.Read(ref _r);
            int m = (int)System.Math.Min(n, w - r);
            for (int i = 0; i < m; i++) dst[i] = _buf[(r + i) & _mask];
            Volatile.Write(ref _r, r + m);
            return m;
        }

        /// <summary>Consumer side only: discard the oldest data so that at most <paramref name="keep"/> remain.</summary>
        public int TrimTo(int keep)
        {
            long w = Volatile.Read(ref _w), r = Volatile.Read(ref _r);
            long target = w - keep;
            if (target <= r) return 0;
            Volatile.Write(ref _r, target);
            return (int)(target - r);
        }
    }

    /// <summary>
    /// B6: receives /audio/out (48 kHz s16le, 20 ms frames) and plays it in 3D.
    ///  - layout "layers": channel 0 = inner layer (the selected target) -> mono AudioSource placed at
    ///    target_bearing_deg around the head and spatialized (Meta XR Audio if installed and selected as the
    ///    project spatializer, otherwise Unity's built-in 3D panner); channel 1 = outer layer (rest of the room) ->
    ///    non-spatialized ambient source.
    ///  - layout "stereo": already panned stereo, played 2D.
    /// A small jitter buffer (prefill N frames) absorbs network jitter; underruns output silence and re-prime.
    /// Inner and outer layers have separate volumes (dB, applied when the frame is written).
    /// Audio is delivered straight from the network thread to lock-free ring buffers (no main-thread hop).
    /// </summary>
    [DefaultExecutionOrder(50)]
    public class AudioOutPlayer : MonoBehaviour
    {
        const int FrameSamples = 960;

        [SerializeField] HubClient hub;
        [SerializeField] OVRCameraRig rig;
        [Tooltip("Frames (20 ms each) buffered before playback starts / resumes after an underrun.")]
        [SerializeField, Range(1, 10)] int prefillFrames = 3;
        [Tooltip("Frames above which the oldest audio is discarded (bounds latency).")]
        [SerializeField, Range(3, 25)] int maxBufferedFrames = 8;
        [Tooltip("Distance of the inner source from the head, metres.")]
        [SerializeField] float innerDistance = 1.5f;
        [SerializeField] float innerDb = 0f;
        [SerializeField] float outerDb = -12f;
        [Tooltip("Also send SetVolumes to the hub when the volumes change. Leave off if the hub does not apply gains itself; Quest applies them locally.")]
        [SerializeField] bool forwardVolumesToHub = false;
        [Tooltip("Max turn rate of the inner source around the head, deg/s (smooths bearing jumps).")]
        [SerializeField] float maxSlewDegPerSec = 720f;
        [SerializeField] string preferredSpatializer = "Meta XR Audio";

        // ---- stats (read by DiagnosticsOverlay) ----
        public long FramesReceived;
        public long FramesDropped;      // ring overflow or oldest-trim
        public long Underruns;
        public volatile string Layout = "-";
        public float BufferedMs => (_inner.Count) * 1000f / 48000f;
        public float LastFrameAgeMs;    // arrival time minus device_ts_us of the newest input sample (mic -> hub -> headset)
        public float RoundTripMs => LastFrameAgeMs + BufferedMs; // + device output latency not included
        public string SpatializerInfo = "";
        public float InnerDb => innerDb;
        public float OuterDb => outerDb;
        public float CurrentBearingDeg => _bearingNow;
        public bool HasBearing => _hasBearing;

        readonly FloatRing _inner = new FloatRing(1 << 15);   // 32768 samples ~ 680 ms mono
        readonly FloatRing _outer = new FloatRing(1 << 15);
        readonly FloatRing _stereo = new FloatRing(1 << 16);  // interleaved
        volatile bool _innerPrimed, _outerPrimed, _stereoPrimed;
        float _innerGain = 1f, _outerGain = 0.25f;
        volatile bool _hasBearing;
        float _bearingTarget, _bearingNow;
        readonly object _bearingLock = new object();
        readonly float[] _scratchIn = new float[FrameSamples];
        readonly float[] _scratchOut = new float[FrameSamples];
        readonly float[] _scratchSt = new float[FrameSamples * 2];

        AudioSource _innerSrc, _outerSrc, _stereoSrc;
        Transform _head;
        bool _started;

        void Awake()
        {
            if (hub == null) hub = GetComponent<HubClient>();
            if (hub == null) hub = FindFirstObjectByType<HubClient>();
            SetVolumesDb(innerDb, outerDb, false);
        }

        void OnEnable()
        {
            ConfigureSpatializer();
            EnsureSources();
            if (hub != null && hub.Connection != null) hub.Connection.AudioOutReceived += OnFrame;
            _started = true;
        }

        void OnDisable()
        {
            if (hub != null && hub.Connection != null) hub.Connection.AudioOutReceived -= OnFrame;
            _started = false;
        }

        void ConfigureSpatializer()
        {
            try
            {
                // The spatializer plugin is chosen in Project Settings > Audio (set by the
                // Cocktail setup menu). The runtime list/set APIs are not in Unity 6000.0.
                string current = AudioSettings.GetSpatializerPluginName();
                bool available = current == preferredSpatializer;
                SpatializerInfo = $"spatializer='{current}'";
                if (!available)
                    SpatializerInfo += " (Meta XR Audio not selected: using Unity built-in 3D panning)";
                Debug.Log("[AudioOutPlayer] " + SpatializerInfo + $" outputSampleRate={AudioSettings.outputSampleRate}Hz");
            }
            catch (System.Exception e) { SpatializerInfo = "spatializer query failed: " + e.Message; Debug.LogWarning("[AudioOutPlayer] " + SpatializerInfo); }
        }

        void EnsureSources()
        {
            if (_innerSrc != null) return;
            if (FindFirstObjectByType<AudioListener>() == null)
            {
                if (rig == null) rig = FindFirstObjectByType<OVRCameraRig>();
                var host = rig != null ? rig.centerEyeAnchor.gameObject : gameObject;
                host.AddComponent<AudioListener>();
            }
            _innerSrc = MakeSource("CocktailInnerLayer", 1, _inner, () => _innerPrimed, v => _innerPrimed = v, spatial: true);
            _outerSrc = MakeSource("CocktailOuterLayer", 1, _outer, () => _outerPrimed, v => _outerPrimed = v, spatial: false);
            _stereoSrc = MakeSource("CocktailStereoLayer", 2, _stereo, () => _stereoPrimed, v => _stereoPrimed = v, spatial: false);
        }

        AudioSource MakeSource(string name, int channels, FloatRing ring, System.Func<bool> getPrimed, System.Action<bool> setPrimed, bool spatial)
        {
            var go = new GameObject(name);
            go.transform.SetParent(transform, false);
            var src = go.AddComponent<AudioSource>();
            int prefill = prefillFrames * FrameSamples * channels;
            int keep = maxBufferedFrames * FrameSamples * channels;
            var clip = AudioClip.Create(name, 48000, channels, 48000, true, data =>
            {
                // Audio thread: no allocation, no Unity API.
                int n = data.Length;
                if (ring.Count > keep) { Interlocked.Add(ref FramesDropped, ring.TrimTo(prefill) / (FrameSamples * channels)); }
                if (!getPrimed())
                {
                    if (ring.Count >= prefill) setPrimed(true);
                    else { System.Array.Clear(data, 0, n); return; }
                }
                int got = ring.Read(data, n);
                if (got < n)
                {
                    System.Array.Clear(data, got, n - got);
                    setPrimed(false);
                    Interlocked.Increment(ref Underruns);
                }
            });
            src.clip = clip;
            src.loop = true;
            src.playOnAwake = false;
            src.dopplerLevel = 0f;
            src.spatialBlend = spatial ? 1f : 0f;
            src.spatialize = spatial;
            src.spatializePostEffects = false;
            src.rolloffMode = AudioRolloffMode.Linear;
            src.minDistance = 1f;
            src.maxDistance = 100f;
            src.volume = 1f;
            src.Play();
            return src;
        }

        /// <summary>Set inner (target) and outer (room) layer volumes in dB. Gains above 0 dB amplify (applied to samples, clipped at full scale).</summary>
        public void SetVolumesDb(float newInnerDb, float newOuterDb, bool sendToHub = true)
        {
            innerDb = newInnerDb; outerDb = newOuterDb;
            _innerGain = Mathf.Pow(10f, newInnerDb / 20f);
            _outerGain = Mathf.Pow(10f, newOuterDb / 20f);
            if (sendToHub && forwardVolumesToHub && hub != null && hub.Connection != null)
                hub.Connection.SendCommand(new SetVolumes { InnerDb = newInnerDb, OuterDb = newOuterDb });
        }

        // ------------------------------------------------------------ network thread

        void OnFrame(AudioOut h, byte[] buf, int offset, int bytes)
        {
            if (!_started) return;
            Interlocked.Increment(ref FramesReceived);
            Layout = h.Layout;
            LastFrameAgeMs = Mathf.Lerp(LastFrameAgeMs, (ClockUs.Now - h.DeviceTsUs) / 1000f, 0.1f);

            int frames = Mathf.Min(bytes / 4, FrameSamples); // 2 channels x 2 bytes per frame
            if (frames <= 0) return;
            if (h.Layout == "stereo")
            {
                for (int i = 0; i < frames * 2; i++)
                    _scratchSt[i] = Mathf.Clamp((short)(buf[offset + 2 * i] | (buf[offset + 2 * i + 1] << 8)) / 32768f, -1f, 1f);
                if (!_stereo.Write(_scratchSt, frames * 2)) Interlocked.Increment(ref FramesDropped);
                return;
            }

            float gi = _innerGain, go = _outerGain;
            for (int i = 0; i < frames; i++)
            {
                int o = offset + 4 * i;
                short a = (short)(buf[o] | (buf[o + 1] << 8));
                short b = (short)(buf[o + 2] | (buf[o + 3] << 8));
                _scratchIn[i] = Mathf.Clamp(a / 32768f * gi, -1f, 1f);
                _scratchOut[i] = Mathf.Clamp(b / 32768f * go, -1f, 1f);
            }
            if (!_inner.Write(_scratchIn, frames)) Interlocked.Increment(ref FramesDropped);
            if (!_outer.Write(_scratchOut, frames)) Interlocked.Increment(ref FramesDropped);

            if (h.TargetBearingDeg.HasValue)
            {
                lock (_bearingLock) { _bearingTarget = h.TargetBearingDeg.Value; }
                _hasBearing = true;
            }
            else _hasBearing = false;
        }

        // ------------------------------------------------------------ main thread

        void LateUpdate()
        {
            if (_innerSrc == null) return;
            if (rig == null) rig = FindFirstObjectByType<OVRCameraRig>();
            _head = rig != null ? rig.centerEyeAnchor : (Camera.main != null ? Camera.main.transform : null);
            if (_head == null) return;

            float target;
            lock (_bearingLock) { target = _bearingTarget; }
            _bearingNow = Mathf.MoveTowardsAngle(_bearingNow, target, maxSlewDegPerSec * Time.unscaledDeltaTime);

            if (_hasBearing)
            {
                // Bearing is world yaw in tracking space: 0 = tracking +z (initial forward), + = right.
                float rad = _bearingNow * Mathf.Deg2Rad;
                Vector3 dirTracking = new Vector3(Mathf.Sin(rad), 0f, Mathf.Cos(rad));
                Vector3 dirWorld = rig != null && rig.trackingSpace != null ? rig.trackingSpace.TransformDirection(dirTracking) : dirTracking;
                _innerSrc.transform.position = _head.position + dirWorld * innerDistance;
                _innerSrc.spatialBlend = 1f;
            }
            else
            {
                // No bearing: play the inner layer centered in the head (no spatial cue).
                _innerSrc.transform.position = _head.position;
                _innerSrc.spatialBlend = 0f;
            }
        }

        void OnDestroy()
        {
            if (_innerSrc != null) _innerSrc.Stop();
            if (_outerSrc != null) _outerSrc.Stop();
            if (_stereoSrc != null) _stereoSrc.Stop();
        }
    }
}
