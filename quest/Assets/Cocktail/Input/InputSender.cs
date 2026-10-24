using Cocktail.Protocol;
using Cocktail.Stream;
using UnityEngine;
using ProtoAction = Cocktail.Protocol.Action;

namespace Cocktail.UserInput
{
    /// <summary>
    /// B5 (minimal): turns controller buttons and hand pinches into InputEvent messages on /ingest/pose.
    ///
    ///   select  = right index trigger, or index-finger pinch (either tracked hand). Carries a ray in TRACKING SPACE:
    ///             head gaze by default (centre eye position + forward), or the hand/controller pointer if rayMode=Pointer.
    ///   stop    = B (right controller)
    ///   replay  = X (left controller)
    ///   next    = Y (left controller)
    ///   prev    = left thumbstick click
    ///
    /// Other code (UI buttons, source list) can call <see cref="Send"/> directly, e.g. with a source_id.
    /// Pinch needs hand tracking enabled on OVRManager and OVRHand components in the rig (not done by this script).
    /// </summary>
    public class InputSender : MonoBehaviour
    {
        public enum RayMode { HeadGaze, Pointer }

        [SerializeField] HubClient hub;
        [SerializeField] OVRCameraRig rig;
        [SerializeField] RayMode rayMode = RayMode.HeadGaze;
        [SerializeField] bool usePinch = true;

        public long Sent { get; private set; }
        public string LastSent = "-";

        OVRHand[] _hands = new OVRHand[0];
        bool[] _pinchWas = new bool[0];
        float _nextHandScan;

        void Awake()
        {
            if (hub == null) hub = GetComponent<HubClient>();
            if (hub == null) hub = FindFirstObjectByType<HubClient>();
        }

        void Update()
        {
            if (hub == null || hub.Connection == null) return;

            bool select = OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch);
            string modality = "controller";
            Transform pointer = null;
            if (select && rayMode == RayMode.Pointer && rig != null) pointer = rig.rightHandAnchor;

            if (usePinch)
            {
                ScanHands();
                for (int i = 0; i < _hands.Length; i++)
                {
                    var h = _hands[i];
                    bool pinching = h != null && h.IsTracked && h.GetFingerIsPinching(OVRHand.HandFinger.Index);
                    if (pinching && !_pinchWas[i] && !select)
                    {
                        select = true; modality = "hand";
                        if (rayMode == RayMode.Pointer && h.PointerPose != null) pointer = h.PointerPose;
                    }
                    _pinchWas[i] = pinching;
                }
            }

            if (select) SendSelect(pointer, modality);
            // Right controller only (one controller is enough):
            //   front trigger = select, side trigger (grip) or B = stop, A = replay,
            //   joystick flick right/left = next/prev. Left X/Y/stick-click still work if present.
            if (OVRInput.GetDown(OVRInput.Button.PrimaryHandTrigger, OVRInput.Controller.RTouch)
                || OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.RTouch)) Send(ProtoAction.Stop);
            if (OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.RTouch)
                || OVRInput.GetDown(OVRInput.Button.Three, OVRInput.Controller.LTouch)) Send(ProtoAction.Replay);
            if (OVRInput.GetDown(OVRInput.Button.Four, OVRInput.Controller.LTouch)) Send(ProtoAction.Next);
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstick, OVRInput.Controller.LTouch)) Send(ProtoAction.Prev);

            float sx = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.RTouch).x;
            if (_stickLatched) { if (Mathf.Abs(sx) < 0.4f) _stickLatched = false; }
            else if (sx > 0.7f) { _stickLatched = true; Send(ProtoAction.Next); }
            else if (sx < -0.7f) { _stickLatched = true; Send(ProtoAction.Prev); }
        }

        bool _stickLatched;

        void ScanHands()
        {
            if (Time.unscaledTime < _nextHandScan) return;
            _nextHandScan = Time.unscaledTime + 2f;
            var found = FindObjectsByType<OVRHand>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            if (found.Length != _hands.Length)
            {
                _hands = found;
                _pinchWas = new bool[found.Length];
            }
            else _hands = found;
        }

        /// <summary>Send a select with a ray (tracking space). pointer == null uses head gaze.</summary>
        public void SendSelect(Transform pointer, string modality)
        {
            Vector3 origin, dir;
            string mod = modality;
            if (pointer != null)
            {
                ToTracking(pointer.position, pointer.forward, out origin, out dir);
            }
            else
            {
                Transform eye = rig != null ? rig.centerEyeAnchor : (Camera.main != null ? Camera.main.transform : null);
                if (eye == null) { Send(ProtoAction.Select); return; }
                ToTracking(eye.position, eye.forward, out origin, out dir);
                mod = "head"; // the ray is head gaze regardless of which input confirmed it
            }
            Send(ProtoAction.Select, null, mod, origin, dir);
        }

        void ToTracking(Vector3 worldPos, Vector3 worldDir, out Vector3 pos, out Vector3 dir)
        {
            if (rig != null && rig.trackingSpace != null)
            {
                pos = rig.trackingSpace.InverseTransformPoint(worldPos);
                dir = rig.trackingSpace.InverseTransformDirection(worldDir);
            }
            else { pos = worldPos; dir = worldDir; }
        }

        /// <summary>Generic send for UI code. modality must be one of "head", "hand", "controller", "ui".</summary>
        public void Send(ProtoAction action, string sourceId = null, string modality = "ui", Vector3? rayOrigin = null, Vector3? rayDir = null)
        {
            if (hub == null || hub.Connection == null) return;
            hub.Connection.SendInput(new InputEvent
            {
                DeviceTsUs = ClockUs.Now,
                Action = action,
                SourceId = sourceId,
                Modality = modality,
                RayOrigin = rayOrigin.HasValue ? new[] { rayOrigin.Value.x, rayOrigin.Value.y, rayOrigin.Value.z } : null,
                RayDir = rayDir.HasValue ? new[] { rayDir.Value.x, rayDir.Value.y, rayDir.Value.z } : null,
            });
            Sent++;
            LastSent = action + (sourceId != null ? " " + sourceId : "") + " (" + modality + ")";
            Debug.Log("[InputSender] " + LastSent);
        }
    }
}
