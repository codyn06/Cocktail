using System.Reflection;
using Cocktail.Protocol;
using Meta.XR;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Cocktail.Stream
{
    /// <summary>
    /// B2: streams Quest 3/3S passthrough camera frames as JPEG VideoFrame messages to /ingest/video.
    ///
    /// API used (verified against the installed package and Meta docs, see quest/README or the hand-off):
    ///  - Meta.XR.PassthroughCameraAccess (MRUK package com.meta.xr.mrutilitykit 207.0.1, Core/Scripts/PassthroughCameraAccess.cs).
    ///    Needs Horizon OS v74+, Quest 3 / 3S, permission horizonos.permission.HEADSET_CAMERA in the manifest AND granted at runtime,
    ///    Passthrough enabled (OVRManager), and OVRProjectConfig.isPassthroughCameraAccessEnabled (the editor setup script sets it).
    ///  - Frame: GetTexture() (RenderTexture, RGBA8) -> Graphics.Blit to a small RT -> AsyncGPUReadback -> ImageConversion.EncodeNativeArrayToJPG.
    ///    The encode runs on the main thread (measured and logged as encode ms); lower sendResolution/targetFps if it costs too much.
    ///  - Frame time: the component keeps the capture time in the XrTime base (CLOCK_MONOTONIC ns on Android) in a private field,
    ///    read here by reflection and mapped onto <see cref="ClockUs"/>; the result is sanity checked (0..500 ms old) and falls back to
    ///    "now" if implausible. Camera pose (GetCameraPose) is at that same timestamp.
    /// </summary>
    public class CameraStreamer : MonoBehaviour
    {
        [SerializeField] HubClient hub;
        [SerializeField] OVRCameraRig rig;
        [SerializeField] PassthroughCameraAccess.CameraPositionType eye = PassthroughCameraAccess.CameraPositionType.Left;
        [Tooltip("Resolution requested from the camera. 1280x960 is the original supported size (Meta PCA overview).")]
        [SerializeField] Vector2Int sourceResolution = new Vector2Int(1280, 960);
        [Tooltip("Width of the JPEG sent to the hub; height follows the camera aspect ratio.")]
        [SerializeField] int sendWidth = 640;
        [SerializeField, Range(1, 30)] float targetFps = 15f;
        [SerializeField, Range(10, 95)] int jpegQuality = 60;
        [Tooltip("Camera stream max fps requested from the OS (the component only sets it while disabled).")]
        [SerializeField] int cameraMaxFps = 30;
        [Tooltip("Flip vertically before encoding, if the first dumped frame is upside down.")]
        [SerializeField] bool flipVertical = false;
        [Tooltip("Write the first JPEG to persistentDataPath/first_frame.jpg (adb pull to check orientation/colors).")]
        [SerializeField] bool dumpFirstFrame = true;
        [SerializeField] float setupTimeoutSeconds = 12f;

        // ---- report (volatile fields: HubClient reads them on a network thread) ----
        public volatile bool Active;
        public volatile bool Resolved;
        public volatile bool CameraAvailable;
        public volatile float HfovDeg;
        public string Report = "camera: not started";
        public Vector2Int CurrentResolution { get; private set; }
        public Vector2Int SentResolution { get; private set; }

        // ---- live stats ----
        public long FramesSent { get; private set; }
        public float Fps { get; private set; }
        public int LastJpegBytes { get; private set; }
        public float LastEncodeMs { get; private set; }
        public float LastReadbackMs { get; private set; }
        public float LastCaptureAgeMs { get; private set; }   // camera exposure -> our Update
        public float HeadsetPipelineMs { get; private set; }  // exposure -> JPEG queued for send (end-to-end delay on the headset side)
        public bool TimestampFromXrTime { get; private set; }

        PassthroughCameraAccess _pca;
        GameObject _pcaHost;
        RenderTexture _small;
        bool _setupStarted;
        float _setupAt;
        float _nextSendAt;
        int _inFlight;
        long _fpsCountStart; long _fpsCount0;
        FieldInfo _tsField;
        bool _dumped;
        bool _loggedTsInfo;

        void Awake()
        {
            if (hub == null) hub = GetComponent<HubClient>();
            if (hub == null) hub = FindFirstObjectByType<HubClient>();
            _tsField = typeof(PassthroughCameraAccess).GetField("_timestampNsMonotonic", BindingFlags.NonPublic | BindingFlags.Instance);
        }

        void OnEnable() { Active = true; }

        void OnDisable()
        {
            Active = false;
            if (_pcaHost != null) Destroy(_pcaHost);
            _pcaHost = null; _pca = null; _setupStarted = false;
            CameraAvailable = false; Resolved = false;   // set up again on the next enable
            if (_small != null) { _small.Release(); Destroy(_small); _small = null; }
        }

        void Update()
        {
            if (!_setupStarted)
            {
                if (!PermissionHelper.Ensure(PermissionHelper.HeadsetCamera))
                {
                    if (PermissionHelper.Denied(PermissionHelper.HeadsetCamera))
                        Fail("camera: horizonos.permission.HEADSET_CAMERA denied");
                    else Report = "camera: waiting for HEADSET_CAMERA permission";
                    return;
                }
                if (Application.platform == RuntimePlatform.Android && !PassthroughCameraAccess.IsSupported)
                {
                    Fail("camera: Passthrough Camera Access not supported (needs Quest 3/3S and Horizon OS v74+)");
                    return;
                }
                CreatePca();
                return;
            }

            if (_pca == null) return;

            if (!CameraAvailable)
            {
                if (_pca.IsPlaying) OnCameraPlaying();
                else if (Time.realtimeSinceStartup - _setupAt > setupTimeoutSeconds)
                    Fail($"camera: no frame within {setupTimeoutSeconds:F0}s (permission not granted, passthrough disabled, or OS < v74?)");
                return;
            }

            if (!_pca.enabled) { CameraAvailable = false; Fail("camera: PassthroughCameraAccess disabled itself (see log)"); return; }
            if (!_pca.IsUpdatedThisFrame) return;
            if (Time.unscaledTime < _nextSendAt || _inFlight >= 2) return;
            _nextSendAt = Time.unscaledTime + 1f / targetFps;
            Capture();
        }

        void Fail(string why)
        {
            Report = why;
            if (!Resolved) Debug.LogWarning("[CameraStreamer] " + why);
            CameraAvailable = false;
            Resolved = true; // Hello goes out with camera=false
        }

        void CreatePca()
        {
            // Create the component on an inactive GameObject so its fields (and MaxFramerate, which is only settable
            // while disabled) can be configured before OnEnable starts the camera.
            _pcaHost = new GameObject("PassthroughCameraAccess");
            _pcaHost.transform.SetParent(transform, false);
            _pcaHost.SetActive(false);
            _pca = _pcaHost.AddComponent<PassthroughCameraAccess>();
            _pca.CameraPosition = eye;
            _pca.RequestedResolution = sourceResolution;
            _pca.MaxFramerate = cameraMaxFps;
            _pcaHost.SetActive(true);
            _setupStarted = true; _setupAt = Time.realtimeSinceStartup;
            Report = "camera: starting";
        }

        void OnCameraPlaying()
        {
            CurrentResolution = _pca.CurrentResolution;
            var intr = _pca.Intrinsics;
            // Horizontal FOV of the (possibly cropped) image that is delivered, from the pinhole intrinsics.
            var sensor = (Vector2)intr.SensorResolution;
            var cur = (Vector2)CurrentResolution;
            float hfov = 0;
            if (sensor.x > 0 && intr.FocalLength.x > 0 && cur.x > 0)
            {
                Vector2 scale = cur / sensor;
                scale /= Mathf.Max(scale.x, scale.y);
                float cropW = sensor.x * scale.x;
                hfov = 2f * Mathf.Atan(cropW / (2f * intr.FocalLength.x)) * Mathf.Rad2Deg;
            }
            HfovDeg = hfov;
            int sw = Mathf.Max(16, sendWidth);
            int sh = Mathf.Max(16, Mathf.RoundToInt(sw * (float)CurrentResolution.y / Mathf.Max(1, CurrentResolution.x)));
            SentResolution = new Vector2Int(sw, sh);
            _small = new RenderTexture(sw, sh, 0, RenderTextureFormat.ARGB32) { name = "CocktailCamSmall" };
            _small.Create();
            CameraAvailable = true;
            Resolved = true;
            Report = $"camera: {eye} eye {CurrentResolution.x}x{CurrentResolution.y} -> JPEG {sw}x{sh} q{jpegQuality} @{targetFps}fps hfov={hfov:F1}deg";
            Debug.Log($"[CameraStreamer] REPORT {Report} intrinsics: f=({intr.FocalLength.x:F1},{intr.FocalLength.y:F1}) " +
                      $"pp=({intr.PrincipalPoint.x:F1},{intr.PrincipalPoint.y:F1}) sensor={intr.SensorResolution}");
        }

        void Capture()
        {
            Texture src = _pca.GetTexture();
            if (src == null || hub == null || hub.Connection == null) return;

            long nowUs = ClockUs.Now;
            long tsUs = nowUs;
            TimestampFromXrTime = false;
            if (_tsField != null)
            {
                long ns = (long)_tsField.GetValue(_pca);
                long cand = ClockUs.FromMonotonicNs(ns);
                long age = nowUs - cand;
                if (ns > 0 && age >= 0 && age <= 500_000) { tsUs = cand; TimestampFromXrTime = true; }
                if (!_loggedTsInfo)
                {
                    _loggedTsInfo = true;
                    double dtAge = (System.DateTime.UtcNow - _pca.Timestamp).TotalMilliseconds;
                    Debug.Log($"[CameraStreamer] first frame timestamp check: XrTime-mapped age {age / 1000.0:F1} ms (accepted={TimestampFromXrTime}); " +
                              $"DateTime-based age {dtAge:F1} ms. If neither is plausible the frame time falls back to arrival time.");
                }
            }
            LastCaptureAgeMs = (nowUs - tsUs) / 1000f;

            // Camera pose at the frame time, converted from Unity world to the rig's tracking space.
            Pose wp = _pca.GetCameraPose();
            Vector3 pos = wp.position; Quaternion rot = wp.rotation;
            if (rig == null) rig = FindFirstObjectByType<OVRCameraRig>();
            if (rig != null && rig.trackingSpace != null)
            {
                pos = rig.trackingSpace.InverseTransformPoint(wp.position);
                rot = Quaternion.Inverse(rig.trackingSpace.rotation) * wp.rotation;
            }

            if (flipVertical) Graphics.Blit(src, _small, new Vector2(1, -1), new Vector2(0, 1));
            else Graphics.Blit(src, _small);

            _inFlight++;
            float reqTime = Time.realtimeSinceStartup;
            int w = _small.width, h = _small.height;
            float hfov = HfovDeg;
            AsyncGPUReadback.Request(_small, 0, TextureFormat.RGBA32, req =>
            {
                _inFlight--;
                if (req.hasError || hub == null || hub.Connection == null) return;
                LastReadbackMs = (Time.realtimeSinceStartup - reqTime) * 1000f;
                NativeArray<byte> raw = req.GetData<byte>();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                NativeArray<byte> jpg = ImageConversion.EncodeNativeArrayToJPG(raw, GraphicsFormat.R8G8B8A8_UNorm, (uint)w, (uint)h, 0, jpegQuality);
                try
                {
                    LastEncodeMs = (float)sw.Elapsed.TotalMilliseconds;
                    int n = jpg.Length;
                    var bytes = new byte[n];
                    jpg.CopyTo(bytes);
                    LastJpegBytes = n;
                    hub.Connection.SendVideo(new VideoFrame
                    {
                        DeviceTsUs = tsUs,
                        Width = w,
                        Height = h,
                        Codec = "jpeg",
                        CamPos = new[] { pos.x, pos.y, pos.z },
                        CamRotXyzw = new[] { rot.x, rot.y, rot.z, rot.w },
                        HfovDeg = hfov > 0 ? hfov : (float?)null,
                    }, bytes, n);
                    FramesSent++;
                    HeadsetPipelineMs = (ClockUs.Now - tsUs) / 1000f;
                    if (dumpFirstFrame && !_dumped)
                    {
                        _dumped = true;
                        try
                        {
                            var path = System.IO.Path.Combine(Application.persistentDataPath, "first_frame.jpg");
                            System.IO.File.WriteAllBytes(path, bytes);
                            Debug.Log($"[CameraStreamer] wrote {path} ({n} bytes) - adb pull it to verify orientation/colour");
                        }
                        catch (System.Exception e) { Debug.LogWarning("[CameraStreamer] dump failed: " + e.Message); }
                    }
                    UpdateFps();
                }
                finally { jpg.Dispose(); }
            });
        }

        void UpdateFps()
        {
            long t = ClockUs.Now;
            if (_fpsCountStart == 0) { _fpsCountStart = t; _fpsCount0 = FramesSent; return; }
            if (t - _fpsCountStart >= 2_000_000)
            {
                Fps = (FramesSent - _fpsCount0) * 1e6f / (t - _fpsCountStart);
                Debug.Log($"[CameraStreamer] {Fps:F1} fps, {LastJpegBytes / 1024f:F0} KB/frame, encode {LastEncodeMs:F1} ms, " +
                          $"readback {LastReadbackMs:F1} ms, capture age {LastCaptureAgeMs:F0} ms, headset pipeline {HeadsetPipelineMs:F0} ms");
                _fpsCountStart = t; _fpsCount0 = FramesSent;
            }
        }
    }
}
