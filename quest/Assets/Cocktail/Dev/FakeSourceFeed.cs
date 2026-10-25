using System.Collections.Generic;
using Cocktail.Protocol;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// Fabricates hub events so the UI works with no hub (editor, headset on a table). Three sources: s_1 speech and s_2 speech
    /// with slowly drifting bearings, s_3 music behind the wearer. s_2 goes tracking -> last_seen -> lost -> tracking again.
    /// A 60 s demo script drives the state banner (listening, selected, playing, replay processing/playing, target lost) and
    /// fires a critical alert; it pauses as soon as the user selects/stops/replays something (user control).
    /// Events are raised from <see cref="Tick"/> at 10 Hz. Update calls Tick in play mode; the headset-free self test calls it directly.
    /// Data is FAKE: nothing here is real audio analysis.
    /// </summary>
    public class FakeSourceFeed : MonoBehaviour
    {
        public event System.Action<SourcesUpdated> OnSources;
        public event System.Action<StateChanged> OnState;
        public event System.Action<TargetUpdated> OnTarget;
        public event System.Action<AlertFired> OnAlert;
        public event System.Action<ReplayReady> OnReplayReady;

        [Tooltip("Run Tick from Update (play mode). The self test turns this off and calls Tick itself.")] public bool autoUpdate = true;
        [Tooltip("Seconds the demo script takes before it repeats.")] public float period = 60f;
        public float hfovDeg = 90f;
        /// <summary>Head yaw used to decide in_view. CocktailUI sets it every frame; 0 if nothing sets it.</summary>
        public float headYawDeg;
        [Tooltip("Half field of view (deg) inside which a fake source counts as in view.")] public float inViewHalfDeg = 40f;

        public float Clock { get; private set; }
        public SourcesUpdated Current { get; private set; }
        public StateChanged CurrentState { get; private set; }
        public bool UserControlled { get; private set; }

        float _acc, _stateClock; bool _started;
        HubState _state = HubState.Listening; string _target; Mode _mode = Mode.Balanced;
        int _selVersion;
        bool _alertFired; int _lastCycle = -1;
        float _pendingTo = -1f; HubState _pendingState;

        void Update() { if (autoUpdate) Tick(Time.unscaledDeltaTime); }

        public void ResetDemo() { Clock = 0f; _acc = 0f; _started = false; UserControlled = false; _state = HubState.Listening; _target = null; _alertFired = false; _lastCycle = -1; }

        public void Tick(float dt)
        {
            Clock += dt; _acc += dt; _stateClock += dt;
            if (!_started) { _started = true; Emit(); EmitState(); }
            if (_pendingTo >= 0f && Clock >= _pendingTo) { _pendingTo = -1f; SetState(_pendingState, _target); }
            if (!UserControlled && _anchors.Count == 0) RunScript();   // with the cubes in the room, nothing is chosen until the wearer chooses
            if (_acc >= 0.1f) { _acc = 0f; Emit(); }
        }

        // ---- user actions (CocktailUI calls these instead of the hub while the fake feed is the data source)
        public void Select(string sourceId)
        {
            UserControlled = true; _selVersion++;
            _target = sourceId;
            OnTarget?.Invoke(new TargetUpdated { TargetId = "t_" + sourceId, SelectionVersion = _selVersion, SourceId = sourceId, Kind = "speech", AssocConf = 0.9f, State = TrackState.Tracking, BearingDeg = BearingFor(sourceId, Clock) });
            SetState(HubState.Selected, sourceId);
            _pendingTo = Clock + 0.8f; _pendingState = HubState.Playing;
        }

        public void ClearAnchors() => _anchors.Clear();
        public void SetMode(Mode m) { _mode = m; EmitState(); }

        public void Stop() { UserControlled = true; _target = null; _pendingTo = -1f; SetState(HubState.Listening, null); }

        public void Replay()
        {
            UserControlled = true;
            SetState(HubState.ReplayProcessing, _target);
            _pendingTo = Clock + 2f; _pendingState = HubState.ReplayPlaying;
            OnReplayReady?.Invoke(new ReplayReady { TargetId = "t_" + _target, SelectionVersion = _selVersion, DurationS = 5f });
        }

        // ---- script
        void RunScript()
        {
            int cycle = (int)(Clock / period);
            if (cycle != _lastCycle) { _lastCycle = cycle; _alertFired = false; }
            float t = Clock % period;
            if (t >= 56f) Goto(HubState.Listening, null, 56f);
            else if (t >= 52f) Goto(HubState.Playing, "s_2", 52f);
            else if (t >= 46f) Goto(HubState.TargetLost, "s_2", 46f);
            else if (t >= 31f) Goto(HubState.Playing, "s_2", 31f);
            else if (t >= 30f) Goto(HubState.Selected, "s_2", 30f);
            else if (t >= 24f) Goto(HubState.ReplayPlaying, "s_1", 24f);
            else if (t >= 22f) Goto(HubState.ReplayProcessing, "s_1", 22f);
            else if (t >= 5f) Goto(HubState.Playing, "s_1", 5f);
            else if (t >= 4f) Goto(HubState.Selected, "s_1", 4f);
            else Goto(HubState.Listening, null, 0f);
        }

        string _gotoKey;
        void Goto(HubState s, string target, float key)
        {
            string k = s + "|" + target + "|" + key + "|" + _lastCycle;
            if (k == _gotoKey) return;
            _gotoKey = k;
            if (target != null && (s == HubState.Selected))
            {
                _selVersion++;
                OnTarget?.Invoke(new TargetUpdated { TargetId = "t_" + target, SelectionVersion = _selVersion, SourceId = target, Kind = "speech", AssocConf = 0.9f, State = TrackState.Tracking, BearingDeg = BearingFor(target, Clock) });
            }
            if (s == HubState.ReplayPlaying)
                OnReplayReady?.Invoke(new ReplayReady { TargetId = "t_" + target, SelectionVersion = _selVersion, DurationS = 5f });
            SetState(s, target);
        }

        void SetState(HubState s, string target)
        {
            _state = s; _target = target; _stateClock = 0f;
            EmitState();
        }

        void EmitState()
        {
            CurrentState = new StateChanged { State = _state, TargetId = _target != null ? "t_" + _target : null, SelectionVersion = _selVersion, Mode = _mode };
            OnState?.Invoke(CurrentState);
        }

        // ---- sources
        // sources placed by the scene (the audio cubes) override the scripted motion
        // x = bearing, y = elevation, z = angular size of the object (degrees, so the box covers it), w = distance in metres
        readonly System.Collections.Generic.Dictionary<string, Vector4> _anchors = new System.Collections.Generic.Dictionary<string, Vector4>();
        public void SetAnchor(string id, float bearingDeg, float elevationDeg, float sizeDeg, float distanceM) => _anchors[id] = new Vector4(bearingDeg, elevationDeg, sizeDeg, distanceM);
        float BearingFor(string id, float t) => _anchors.TryGetValue(id, out var a) ? a.x : BearingOf(id, t);

        static float BearingOf(string id, float t)
        {
            switch (id)
            {
                case "s_1": return 20f + 25f * Mathf.Sin(t * 0.17f);
                case "s_2": return -70f + 15f * Mathf.Sin(t * 0.11f + 1f);
                case "s_3": return UiMath.NormalizeDeg(160f + 12f * Mathf.Sin(t * 0.07f) + t * 0.5f);
                default: return 0f;
            }
        }

        TrackState S2State(float t)
        {
            float p = t % period;
            if (p >= 52f || p < 38f) return TrackState.Tracking;
            return p < 46f ? TrackState.LastSeen : TrackState.Lost;
        }

        void Emit()
        {
            long now = (long)(Clock * 1_000_000.0);
            var list = new List<Source>(3);
            list.Add(Make("s_1", "speech", 0.91f, Clock, TrackState.Tracking, -31f + 7f * Mathf.Sin(Clock * 2.1f), now, now));
            var s2state = S2State(Clock);
            long s2obs = s2state == TrackState.Tracking ? now : (long)((Clock - (Clock % period - 38f)) * 1_000_000.0);
            list.Add(Make("s_2", "speech", 0.84f, Clock, s2state, s2state == TrackState.Tracking ? -38f + 6f * Mathf.Sin(Clock * 1.7f) : -55f, now, s2obs));
            list.Add(Make("s_3", "music", 0.77f, Clock, TrackState.Tracking, -24f + 4f * Mathf.Sin(Clock * 0.9f), now, now));
            // two more talkers, only while the scene has objects standing in for them (the extra cubes)
            if (_anchors.ContainsKey("s_4")) list.Add(Make("s_4", "speech", 0.88f, Clock, TrackState.Tracking, -30f, now, now));
            if (_anchors.ContainsKey("s_5")) list.Add(Make("s_5", "speech", 0.86f, Clock, TrackState.Tracking, -30f, now, now));
            Current = new SourcesUpdated { DeviceTsUs = now, Sources = list };
            OnSources?.Invoke(Current);
        }

        Source Make(string id, string label, float conf, float t, TrackState st, float db, long now, long observed)
        {
            bool anchored = _anchors.TryGetValue(id, out var anchor);
            // a real object in the room does not drop out or go quiet (the scripted source 2 fades to -55 dB for 14 s of every minute)
            if (anchored) { st = TrackState.Tracking; observed = now; db = -30f + 3f * Mathf.Sin(t * 1.7f); }
            float b = BearingFor(id, t);
            float rel = UiMath.NormalizeDeg(b - headYawDeg);
            bool inView = Mathf.Abs(rel) < inViewHalfDeg && st != TrackState.Lost;
            float[] bbox = null;
            if (inView)
            {
                float th = Mathf.Tan(hfovDeg * 0.5f * Mathf.Deg2Rad);
                float cx = 0.5f + Mathf.Tan(rel * Mathf.Deg2Rad) / (2f * th);
                if (anchored)
                {
                    // box sized to the object: normalized half extents for the object's angular size (4:3 camera)
                    float half = Mathf.Tan(anchor.z * 0.5f * Mathf.Deg2Rad);
                    float hw = half / (2f * th), hh = half / (2f * th / (4f / 3f));
                    bbox = new[] { cx - hw, 0.5f - hh, cx + hw, 0.5f + hh };
                }
                else bbox = new[] { cx - 0.05f, 0.38f, cx + 0.05f, 0.64f };
            }
            return new Source
            {
                SourceId = id, Label = label, LabelConf = conf, BearingDeg = b, BearingConf = 0.8f, BearingSource = BearingSource.Fused,
                DistanceM = anchored ? anchor.w : (float?)null,
                ElevationDeg = anchored ? anchor.y : id == "s_3" ? 5f : (float?)null, State = st, LevelDb = db, InView = inView, Bbox = bbox, LastObservedUs = observed,
            };
        }
    }
}
