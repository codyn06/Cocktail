using System.Text;
using Cocktail.Audio;
using Cocktail.Dev;
using Cocktail.UserInput;
using UnityEngine;

namespace Cocktail.Stream
{
    /// <summary>
    /// Small world-space text panel (legacy TextMesh, no extra packages) that follows the head loosely: connection state per
    /// socket, counts, mic channels/raw, camera fps/delay, audio-out buffer and the mic-to-headset round-trip estimate.
    /// Toggle with a click of the right thumbstick. All values are also in logcat (tags [MicStreamer], [CameraStreamer], [Hub:*]).
    /// </summary>
    public class DiagnosticsOverlay : MonoBehaviour
    {
        [SerializeField] HubClient hub;
        [SerializeField] MicStreamer mic;
        [SerializeField] CameraStreamer cam;
        [SerializeField] PoseStreamer pose;
        [SerializeField] AudioOutPlayer audioOut;
        [SerializeField] InputSender input;
        [SerializeField] OVRCameraRig rig;
        [SerializeField] bool visible = true;
        [SerializeField] Vector3 offsetFromHead = new Vector3(-0.30f, 0.05f, 1.2f); // x right, y up, z forward (metres)
        [SerializeField] float characterSize = 0.0045f;
        [SerializeField] int fontSize = 64;

        TextMesh _text;
        float _nextRefresh;
        readonly StringBuilder _sb = new StringBuilder(1024);

        void Awake()
        {
            if (hub == null) hub = GetComponent<HubClient>();
            if (mic == null) mic = GetComponent<MicStreamer>();
            if (cam == null) cam = GetComponent<CameraStreamer>();
            if (pose == null) pose = GetComponent<PoseStreamer>();
            if (audioOut == null) audioOut = GetComponent<AudioOutPlayer>();
            if (input == null) input = GetComponent<InputSender>();

            var go = new GameObject("CocktailDiagnosticsText");
            go.transform.SetParent(transform, false);
            _text = go.AddComponent<TextMesh>();
            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _text.font = font;
            _text.fontSize = fontSize;
            _text.characterSize = characterSize;
            _text.anchor = TextAnchor.UpperLeft;
            _text.alignment = TextAlignment.Left;
            _text.color = new Color(0.85f, 1f, 0.85f, 1f);
            var mr = go.GetComponent<MeshRenderer>();
            mr.sharedMaterial = font.material;
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows = false;
            go.SetActive(false);   // LateUpdate shows it, and only while DevMode is on
        }

        float _nextRigScan;

        void LateUpdate()
        {
            if (rig == null && Time.unscaledTime >= _nextRigScan) { _nextRigScan = Time.unscaledTime + 1f; rig = FindFirstObjectByType<OVRCameraRig>(); }
            if (rig == null) return;
            if (DevMode.On && OVRInput.GetDown(OVRInput.Button.SecondaryThumbstick, OVRInput.Controller.RTouch)) visible = !visible;
            bool show = visible && DevMode.On;
            _text.gameObject.SetActive(show);
            if (!show) return;

            Transform head = rig.centerEyeAnchor;
            Vector3 target = head.position + head.right * offsetFromHead.x + head.up * offsetFromHead.y + head.forward * offsetFromHead.z;
            // Lazy follow so the panel stays readable instead of being glued to the face.
            var t = _text.transform;
            t.position = Vector3.Lerp(t.position, target, 1f - Mathf.Exp(-3f * Time.unscaledDeltaTime));
            Vector3 away = t.position - head.position;
            if (away.sqrMagnitude > 1e-4f) t.rotation = Quaternion.Slerp(t.rotation, Quaternion.LookRotation(away, Vector3.up), 1f - Mathf.Exp(-6f * Time.unscaledDeltaTime));

            if (Time.unscaledTime < _nextRefresh) return;
            _nextRefresh = Time.unscaledTime + 0.25f;
            _text.text = Build();
        }

        string Build()
        {
            _sb.Clear();
            var c = hub != null ? hub.Connection : null;
            _sb.Append("COCKTAIL  hub ").Append(hub != null ? hub.Host : "?").Append('\n');
            if (c != null)
                foreach (var ch in c.All)
                {
                    _sb.Append(ch.Name.PadRight(9)).Append(StateText(ch.State)).Append("  e").Append(ch.Epoch)
                       .Append(" tx ").Append(ch.SentMsgs).Append(" (").Append(Kb(ch.SentBytes)).Append(")")
                       .Append(" rx ").Append(ch.RecvMsgs);
                    if (ch.Queue.Dropped > 0) _sb.Append(" drop ").Append(ch.Queue.Dropped);
                    if (ch.Reconnects > 0) _sb.Append(" rc ").Append(ch.Reconnects);
                    if (ch.State != LinkState.Connected && ch.LastError.Length > 0) _sb.Append("  ").Append(ch.LastError.Length > 40 ? ch.LastError.Substring(0, 40) : ch.LastError);
                    _sb.Append('\n');
                }
            if (mic != null) _sb.Append(mic.Report).Append("  lvl ").Append(mic.LevelsText()).Append("  ts:").Append(mic.TimestampMode).Append('\n');
            if (cam != null)
            {
                _sb.Append(cam.Report).Append('\n');
                if (cam.CameraAvailable)
                    _sb.Append("  cam ").Append(cam.Fps.ToString("F1")).Append(" fps  ").Append((cam.LastJpegBytes / 1024f).ToString("F0")).Append(" KB  enc ")
                       .Append(cam.LastEncodeMs.ToString("F1")).Append(" ms  age ").Append(cam.LastCaptureAgeMs.ToString("F0")).Append(" ms  pipe ")
                       .Append(cam.HeadsetPipelineMs.ToString("F0")).Append(" ms").Append(cam.TimestampFromXrTime ? "" : " (ts=arrival)").Append('\n');
            }
            if (pose != null) _sb.Append("pose ").Append(pose.MeasuredHz.ToString("F0")).Append(" Hz");
            if (input != null) _sb.Append("   input ").Append(input.Sent).Append(" last: ").Append(input.LastSent);
            _sb.Append('\n');
            if (audioOut != null)
                _sb.Append("audio out ").Append(audioOut.Layout).Append("  rx ").Append(audioOut.FramesReceived).Append("  buf ").Append(audioOut.BufferedMs.ToString("F0"))
                   .Append(" ms  under ").Append(audioOut.Underruns).Append("  drop ").Append(audioOut.FramesDropped)
                   .Append("\n  mic->hub->headset age ").Append(audioOut.LastFrameAgeMs.ToString("F0")).Append(" ms  +buf = ").Append(audioOut.RoundTripMs.ToString("F0"))
                   .Append(" ms  vol in/out ").Append(audioOut.InnerDb.ToString("F0")).Append('/').Append(audioOut.OuterDb.ToString("F0")).Append(" dB")
                   .Append(audioOut.HasBearing ? "  brg " + audioOut.CurrentBearingDeg.ToString("F0") : "  brg -").Append('\n');
            return _sb.ToString();
        }

        static string StateText(LinkState s)
        {
            switch (s)
            {
                case LinkState.Connected: return "UP  ";
                case LinkState.Connecting: return "conn";
                case LinkState.Backoff: return "wait";
                default: return "idle";
            }
        }

        static string Kb(long bytes) => bytes < 1024 * 1024 ? (bytes / 1024) + " KB" : (bytes / 1048576f).ToString("F1") + " MB";
    }
}
