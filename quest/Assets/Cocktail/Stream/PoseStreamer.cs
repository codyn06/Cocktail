using Cocktail.Protocol;
using UnityEngine;

namespace Cocktail.Stream
{
    /// <summary>
    /// Sends the head pose (OVRCameraRig center eye) to /ingest/pose at ~50 Hz as PoseSample.
    /// Coordinates are the rig's TRACKING SPACE (not Unity world), +x right, +y up, +z forward, so they stay
    /// consistent if the rig object is ever moved. device_ts_us comes from <see cref="ClockUs"/>, the same clock
    /// as audio chunks and video frames.
    /// </summary>
    public class PoseStreamer : MonoBehaviour
    {
        [SerializeField] HubClient hub;
        [SerializeField] OVRCameraRig rig;
        [SerializeField] float rateHz = 50f;

        public long Sent { get; private set; }
        public float MeasuredHz { get; private set; }

        double _nextUs;
        long _lastCountTime; long _lastCount;

        void Awake()
        {
            if (hub == null) hub = GetComponent<HubClient>();
            if (hub == null) hub = FindFirstObjectByType<HubClient>();
        }

        void Update()
        {
            if (hub == null || hub.Connection == null) return;
            if (rig == null) rig = FindFirstObjectByType<OVRCameraRig>();
            Transform eye = rig != null ? rig.centerEyeAnchor : (Camera.main != null ? Camera.main.transform : null);
            if (eye == null) return;

            long now = ClockUs.Now;
            double interval = 1e6 / rateHz;
            // Send on the frame closest to each 20 ms slot, so a 72/90 Hz frame loop averages 50 Hz instead of 36 Hz.
            if (now < _nextUs - 0.5 * Time.unscaledDeltaTime * 1e6) return;
            _nextUs = (_nextUs == 0 || now - _nextUs > 5 * interval) ? now + interval : _nextUs + interval;

            Vector3 p; Quaternion q;
            if (rig != null && rig.trackingSpace != null)
            {
                p = rig.trackingSpace.InverseTransformPoint(eye.position);
                q = Quaternion.Inverse(rig.trackingSpace.rotation) * eye.rotation;
            }
            else { p = eye.position; q = eye.rotation; }

            hub.Connection.SendPose(new PoseSample
            {
                DeviceTsUs = now,
                Pos = new[] { p.x, p.y, p.z },
                RotXyzw = new[] { q.x, q.y, q.z, q.w },
            });
            Sent++;

            if (now - _lastCountTime > 1_000_000)
            {
                MeasuredHz = (Sent - _lastCount) * 1e6f / Mathf.Max(1, now - _lastCountTime);
                _lastCountTime = now; _lastCount = Sent;
            }
        }
    }
}
