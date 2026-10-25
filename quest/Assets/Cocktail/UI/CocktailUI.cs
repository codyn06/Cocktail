using System.Collections.Generic;
using System.Text;
using Cocktail.Audio;
using Cocktail.Dev;
using Cocktail.Protocol;
using Cocktail.Stream;
using Cocktail.UserInput;
using UnityEngine;
using ProtoAction = Cocktail.Protocol.Action;

namespace Cocktail.Ui
{
    public enum HudMode { Full, Glasses }

    /// <summary>
    /// A5 / C6 / E3: the mixed-reality screen the wearer sees, driven by hub events (or by <see cref="FakeSourceFeed"/> when there is no hub).
    ///
    /// World content (source markers) lives under the UI world root, which is the OVRCameraRig tracking space, so the protocol bearing
    /// (world yaw in tracking space, 0 = tracking +Z, + = right) maps directly: marker position = head position + (sin b, 0, cos b) * markerDistance.
    /// Head-locked content (state banner, alerts, compass strip, edge arrows, unlocated chips, glasses panel, reticle) lives under the
    /// centre-eye anchor. All looks come from <see cref="UiTheme"/>; this class is logic only.
    ///
    /// Controls (do not clash with InputSender: right trigger/pinch = select, B = stop, X = replay, Y = next, left stick click = prev,
    /// right stick click = diagnostics overlay): A (right controller, Button.One) cycles speaker-only -> full -> glasses HUD (<see cref="ToggleHudMode"/>).
    ///
    /// Selection (full HUD): head-gaze reticle; resting the gaze on a marker highlights it and fills a dwell bar; right trigger or pinch while a
    /// marker is gazed calls <see cref="TrySelectGazed"/>, which sends InputSender.Send(select, source_id, "head", ray). InputSender's own
    /// ray-only select fires on the same press (this duplicates the select unless InputSender is changed).
    /// Selection (speaker-only HUD): point at a speaker or a picker row with the hand ray (or the head when no hand is tracked) and pinch / trigger.
    /// </summary>
    [DefaultExecutionOrder(50)]
    public class CocktailUI : MonoBehaviour
    {
        [Tooltip("Show fabricated sources when the hub sends nothing. Real hub data is preferred as soon as it arrives.")]
        public bool useFake = true;
        [SerializeField] UiTheme theme;
        [SerializeField] HubClient hub;
        [SerializeField] InputSender input;
        [SerializeField] OVRCameraRig rig;
        [SerializeField] CameraStreamer cameraStreamer;
        [SerializeField] FakeSourceFeed fake;
        [Tooltip("Read controller/pinch input (play mode on the headset). The self test turns this off.")] public bool pollInput = true;
        [Tooltip("Send an explicit select(source_id) when the user triggers on a gazed marker.")] public bool sendExplicitSelect = true;
        [SerializeField] HudMode hudMode = HudMode.Full;
        [Tooltip("Seconds without real hub sources before the fake feed takes over (if useFake).")] public float realDataTimeout = 3f;

        sealed class Entry
        {
            public Source src; public float lastUpdate;
            public SourceMarkerView marker, chip; public EdgeArrowView arrow;
            public Vector2 box; public bool hasBox; public float elevFallback;
            public SpeakerView speaker; public float lastSpeaking = -1000f;
        }

        readonly Dictionary<string, Entry> _entries = new Dictionary<string, Entry>();
        readonly Dictionary<string, string> _targetToSource = new Dictionary<string, string>();
        readonly List<Entry> _located = new List<Entry>(), _unlocated = new List<Entry>();
        readonly List<string> _remove = new List<string>();
        readonly List<CompassPip> _pips = new List<CompassPip>();
        static readonly System.Comparison<Entry> ByBearing = (a, b) => (a.src.BearingDeg ?? 0f).CompareTo(b.src.BearingDeg ?? 0f);
        // FindObjectsByType has no guaranteed order: sort so each hand keeps its _pinchWas slot across rescans; names descending puts RightHand before LeftHand, so the right hand is the pointer when both are tracked
        static readonly System.Comparison<OVRHand> ByHandName = (a, b) => { int c = string.CompareOrdinal(b.name, a.name); return c != 0 ? c : a.GetInstanceID().CompareTo(b.GetInstanceID()); };

        Transform _worldRoot, _headLocked, _arrowRoot, _chipRoot, _fakeHead, _speakerWorld;
        bool _speakerOnly;
        SpeakerListView _list; int _listCursor; int _hoverRowNow = -1; bool _scrollCursor;
        DemoPanelView _demo; AudioOutPlayer _audioOut;
        PointerCursorView _cursor; float _tgtDist; bool _tgtFound;
        int _dData, _dCubes, _dNums, _dMode, _dSound;
        float _nextReadout;
        readonly Dictionary<string, int> _speakerNum = new Dictionary<string, int>();
        readonly List<SpeakerRow> _rows = new List<SpeakerRow>();
        StateBannerView _banner; AlertBannerView _alert; GlassesPanelView _glasses; CompassStripView _compass; RadarRingView _radar; ReticleView _reticle;
        bool _init, _subscribed;
        float _time, _realAt = -1000f, _nextRigScan;
        Transform _lastHead, _lastWorldParent;

        // state
        HubState _state = HubState.Listening; string _stateTargetId; Mode _mode = Mode.Balanced; float _replaySeconds;
        float? _alertBearing;
        string _hoverId; float _hoverTime;
        string _pendingId; float _pendingLeft;
        // the real hub does not send state.changed yet, so the wearer's own choice is kept here until it does (then the hub wins)
        string _localChosen; bool _hubStateSeen, _wasFake = true, _devChordWas;
        long _latestTs;
        OVRHand[] _hands = new OVRHand[0]; bool[] _pinchWas = new bool[0]; float _nextHandScan;

        // ------------------------------------------------------------------ public API (also used by the self test)
        public UiTheme Theme => theme;
        public HudMode HudMode { get => hudMode; set => hudMode = value; }
        public bool UsingFake => useFake && DevMode.On && (hub == null || _time - _realAt > realDataTimeout);   // fake data is a developer aid: never shown outside DevMode
        public Transform Head
        {
            get
            {
                if (rig != null && rig.centerEyeAnchor != null) return rig.centerEyeAnchor;
                if (Application.isPlaying && Camera.main != null) return Camera.main.transform;
                return _fakeHead != null ? _fakeHead : transform;
            }
        }
        public string HoveredSourceId => _hoverId;
        public string SelectedSourceId => ResolveSelectedSourceId();
        public HubState State => _state;
        public string BannerText => _banner != null ? _banner.Text : "";
        public string AlertText => _alert != null && _alert.Visible ? _alert.Text : "";
        public string GlassesText => _glasses != null ? _glasses.TitleText + " | " + _glasses.DetailText : "";
        public int MarkerCount { get { int n = 0; foreach (var e in _entries.Values) if (e.marker != null && e.marker.gameObject.activeSelf) n++; return n; } }
        public int ChipCount { get { int n = 0; foreach (var e in _entries.Values) if (e.chip != null && e.chip.gameObject.activeSelf) n++; return n; } }
        public int ActiveArrowCount { get { int n = 0; foreach (var e in _entries.Values) if (e.arrow != null && e.arrow.gameObject.activeSelf) n++; return n; } }
        public int RadarPipCount => _radar != null ? _radar.ActivePips : 0;
        public int RadarOutsideFovCount => _radar != null ? _radar.OutsideFovPips : 0;
        public SourceMarkerView GetMarker(string id) => _entries.TryGetValue(id, out var e) ? e.marker : null;
        public EdgeArrowView GetArrow(string id) => _entries.TryGetValue(id, out var e) ? e.arrow : null;
        public Transform WorldRoot => _worldRoot;
        public Transform HeadLockedRoot => _headLocked;
        public FakeSourceFeed Fake => fake;

        /// <summary>Speaker-only HUD: just the outline on whoever is speaking, a dot at the edge if they are out of view, and the arrow toward the noise. Only applies in HudMode.Full.</summary>
        public bool SpeakerOnly { get => _speakerOnly; set => _speakerOnly = value; }
        public SpeakerView GetSpeaker(string id) => _entries.TryGetValue(id, out var e) ? e.speaker : null;
        public int SpeakerOutlineCount { get { int n = 0; foreach (var e in _entries.Values) if (e.speaker != null && e.speaker.OutlineVisible) n++; return n; } }
        public int SpeakerDotCount { get { int n = 0; foreach (var e in _entries.Values) if (e.speaker != null && e.speaker.DotVisible) n++; return n; } }

        /// <summary>A on the right controller: speaker-only -> full HUD -> glasses -> speaker-only.</summary>
        public void ToggleHudMode()
        {
            if (hudMode == HudMode.Full && _speakerOnly) _speakerOnly = false;
            else if (hudMode == HudMode.Full) hudMode = HudMode.Glasses;
            else { hudMode = HudMode.Full; _speakerOnly = true; }
        }

        // ------------------------------------------------------------------ lifecycle
        void OnEnable() { EnsureInit(); }
        void OnDisable() { Unsubscribe(); }
        void Update() { Tick(Time.unscaledDeltaTime); TickCursor(Time.unscaledDeltaTime); if (pollInput && Application.isPlaying) PollInput(); }

        public void EnsureInit()
        {
            if (!_init)
            {
                _init = true;
                if (theme == null) theme = Resources.Load<UiTheme>("CocktailUiTheme");
                if (theme == null) { theme = ScriptableObject.CreateInstance<UiTheme>(); theme.name = "CocktailUiTheme (runtime default)"; }
                DevMode.Init(theme.devMode);
                if (hub == null) hub = HubClient.Instance != null ? HubClient.Instance : FindFirstObjectByType<HubClient>();
                if (input == null) input = FindFirstObjectByType<InputSender>();
                if (input != null) input.autoSelect = false;   // this class sends every select, by source id; InputSender's own head-ray select never fires
                if (cameraStreamer == null) cameraStreamer = FindFirstObjectByType<CameraStreamer>();
                if (fake == null) fake = GetComponent<FakeSourceFeed>();
                if (fake == null && useFake) fake = gameObject.AddComponent<FakeSourceFeed>();
                if (rig == null) rig = FindFirstObjectByType<OVRCameraRig>();

                _fakeHead = new GameObject("UiFakeHead").transform;
                _fakeHead.SetParent(transform, false);
                _worldRoot = new GameObject("UiWorld").transform;
                _headLocked = new GameObject("UiHeadLocked").transform;
                _arrowRoot = new GameObject("Arrows").transform;
                _chipRoot = new GameObject("Chips").transform;
                _speakerWorld = new GameObject("UiSpeakers").transform;
                _speakerOnly = theme.speakerOnlyHud;
                ReparentRoots();
                _arrowRoot.SetParent(_headLocked, false);
                _chipRoot.SetParent(_headLocked, false); _chipRoot.localPosition = theme.chipRowOffset;

                _banner = Child<StateBannerView>("StateBanner", theme.bannerOffset); _banner.Build(theme);
                _alert = Child<AlertBannerView>("AlertBanner", theme.alertOffset); _alert.Build(theme);
                _glasses = Child<GlassesPanelView>("GlassesPanel", theme.glassesOffset); _glasses.Build(theme);
                _compass = Child<CompassStripView>("CompassStrip", theme.compassOffset); _compass.Build(theme);
                _radar = Child<RadarRingView>("RadarRing", theme.radarOffset); _radar.Build(theme);
                _list = Child<SpeakerListView>("SpeakerList", theme.speakerListOffset); _list.Build(theme);
                _demo = Child<DemoPanelView>("DemoPanel", theme.demoPanelOffset); BuildDemo();
                if (Application.isPlaying) { _cursor = new GameObject("PointerCursor").AddComponent<PointerCursorView>(); _cursor.Build(theme); }
                _reticle = Child<ReticleView>("Reticle", new Vector3(0f, 0f, theme.reticleDistance)); _reticle.Build(theme);
            }
            Subscribe();
            if (hub != null && hub.LastSources != null && _entries.Count == 0) { ApplySources(hub.LastSources, false); }
            if (hub != null && hub.LastState != null) ApplyState(hub.LastState, false);
        }

        T Child<T>(string name, Vector3 localPos) where T : Component
        {
            var go = new GameObject(name);
            go.transform.SetParent(_headLocked, false);
            go.transform.localPosition = localPos;
            return go.AddComponent<T>();
        }

        void ReparentRoots()
        {
            Transform head = Head;
            Transform wp = rig != null && rig.trackingSpace != null ? rig.trackingSpace : transform;
            if (_lastHead != head) { _headLocked.SetParent(head, false); _headLocked.localPosition = Vector3.zero; _headLocked.localRotation = Quaternion.identity; _lastHead = head; }
            if (_lastWorldParent != wp)
            {
                _worldRoot.SetParent(wp, false); _worldRoot.localPosition = Vector3.zero; _worldRoot.localRotation = Quaternion.identity;
                _speakerWorld.SetParent(wp, false); _speakerWorld.localPosition = Vector3.zero; _speakerWorld.localRotation = Quaternion.identity;
                _lastWorldParent = wp;
            }
        }

        void Subscribe()
        {
            if (_subscribed) return;
            _subscribed = true;
            if (hub != null)
            {
                hub.OnSources += OnRealSources; hub.OnState += OnRealState; hub.OnTarget += OnRealTarget; hub.OnAlert += OnRealAlert; hub.OnReplayReady += OnRealReplay;
            }
            if (fake != null)
            {
                fake.OnSources += OnFakeSources; fake.OnState += OnFakeState; fake.OnTarget += OnFakeTarget; fake.OnAlert += OnFakeAlert; fake.OnReplayReady += OnFakeReplay;
            }
            if (input != null) input.OnSent += OnInputSent;
        }

        void Unsubscribe()
        {
            if (!_subscribed) return;
            _subscribed = false;
            if (hub != null)
            {
                hub.OnSources -= OnRealSources; hub.OnState -= OnRealState; hub.OnTarget -= OnRealTarget; hub.OnAlert -= OnRealAlert; hub.OnReplayReady -= OnRealReplay;
            }
            if (fake != null)
            {
                fake.OnSources -= OnFakeSources; fake.OnState -= OnFakeState; fake.OnTarget -= OnFakeTarget; fake.OnAlert -= OnFakeAlert; fake.OnReplayReady -= OnFakeReplay;
            }
            if (input != null) input.OnSent -= OnInputSent;
        }

        void OnDestroy()
        {
            Unsubscribe();
            if (_worldRoot != null) UiObjects.Kill(_worldRoot.gameObject);
            if (_speakerWorld != null) UiObjects.Kill(_speakerWorld.gameObject);
            if (_headLocked != null) UiObjects.Kill(_headLocked.gameObject);
        }

        void OnRealSources(SourcesUpdated e) { _realAt = _time; ApplySources(e, false); }
        void OnRealState(StateChanged e)
        {
            _realAt = _time;
            if (e != null && e.State != HubState.Listening) _hubStateSeen = true;
            if (e != null) DevLog.Add("hub", "state " + e.State + " target " + (e.TargetId ?? "-"));
            ApplyState(e, false);
        }
        void OnRealTarget(TargetUpdated e) { ApplyTarget(e, false); }
        void OnRealAlert(AlertFired e) { ShowAlert(e); }
        void OnRealReplay(ReplayReady e) { _replaySeconds = e.DurationS; }
        void OnFakeSources(SourcesUpdated e) { ApplySources(e, true); }
        void OnFakeState(StateChanged e) { ApplyState(e, true); }
        void OnFakeTarget(TargetUpdated e) { ApplyTarget(e, true); }
        void OnFakeAlert(AlertFired e) { if (UsingFake) ShowAlert(e); }
        void OnFakeReplay(ReplayReady e) { if (UsingFake) _replaySeconds = e.DurationS; }

        // ------------------------------------------------------------------ data in
        public void ApplySources(SourcesUpdated e, bool fromFake)
        {
            EnsureInit();
            if (e == null || e.Sources == null || fromFake != UsingFake) return;
            _latestTs = e.DeviceTsUs;
            string selId = ResolveSelectedSourceId();
            var sorted = new List<Source>(e.Sources);
            sorted.Sort((a, b) => b.LevelDb.CompareTo(a.LevelDb));
            int kept = 0;
            foreach (var s in sorted)
            {
                if (s == null || string.IsNullOrEmpty(s.SourceId)) continue;
                bool isSel = s.SourceId == selId;
                if (kept >= theme.maxSources && !isSel) continue;
                if (!isSel) kept++;
                if (!_entries.TryGetValue(s.SourceId, out var en)) { en = new Entry(); _entries[s.SourceId] = en; DevLog.Add("src", "+" + s.SourceId + " " + s.Label + (s.BearingDeg.HasValue ? " " + s.BearingDeg.Value.ToString("0") + "deg" : " no-dir")); }
                en.src = s; en.lastUpdate = _time;
            }
        }

        public void ApplyState(StateChanged e, bool fromFake)
        {
            EnsureInit();
            if (e == null || fromFake != UsingFake) return;
            _state = e.State; _stateTargetId = e.TargetId; _mode = e.Mode;
            if (_pendingId != null && _state != HubState.Listening) _pendingId = null;
        }

        public void ApplyTarget(TargetUpdated e, bool fromFake)
        {
            if (e == null || fromFake != UsingFake) return;
            if (!string.IsNullOrEmpty(e.TargetId) && !string.IsNullOrEmpty(e.SourceId)) _targetToSource[e.TargetId] = e.SourceId;
        }

        public void ShowAlert(AlertFired a)
        {
            EnsureInit();
            if (a == null) return;
            _alertBearing = a.BearingDeg;
            _alert.Show(a, a.BearingDeg.HasValue ? UiMath.NormalizeDeg(a.BearingDeg.Value - HeadYaw()) : (float?)null);
        }

        float? RelAlert(float headYaw) => _alertBearing.HasValue ? UiMath.NormalizeDeg(_alertBearing.Value - headYaw) : (float?)null;

        string ResolveSelectedSourceId()
        {
            string fromHub = ResolveHubChoice();
            if (fromHub != null || UsingFake || _hubStateSeen) return fromHub;
            return _localChosen != null && _entries.ContainsKey(_localChosen) ? _localChosen : null;
        }

        string ResolveHubChoice()
        {
            if (string.IsNullOrEmpty(_stateTargetId) || _state == HubState.Listening) return null;
            if (_targetToSource.TryGetValue(_stateTargetId, out var sid)) return sid;
            if (_entries.ContainsKey(_stateTargetId)) return _stateTargetId;
            if (_stateTargetId.StartsWith("t_") && _entries.ContainsKey(_stateTargetId.Substring(2))) return _stateTargetId.Substring(2);
            return null;
        }

        // ------------------------------------------------------------------ frame
        float HeadYaw()
        {
            if (_worldRoot == null) return 0f;
            return UiMath.YawOf(_worldRoot.InverseTransformDirection(Head.forward));
        }

        /// <summary>Advance the UI by dt seconds. Update calls this every frame; the headset-free self test calls it directly.</summary>
        public void Tick(float dt)
        {
            EnsureInit();
            _time += dt;
            bool fakeNow = UsingFake;
            if (fakeNow != _wasFake) { _wasFake = fakeNow; ResetForNewSource(); }
            if (_demo != null && _demo.gameObject.activeSelf != DevMode.On) _demo.gameObject.SetActive(DevMode.On);
            if (rig == null && _time >= _nextRigScan) { _nextRigScan = _time + 1f; rig = FindFirstObjectByType<OVRCameraRig>(); }
            ReparentRoots();
            Vector3 headPosL = _worldRoot.InverseTransformPoint(Head.position);
            float headYaw = HeadYaw();
            if (fake != null) fake.headYawDeg = headYaw;
            TickCubes(headPosL, headYaw);
            TickDemo(dt);

            bool full = hudMode == HudMode.Full;
            bool spk = full && _speakerOnly;   // speaker-only HUD: nothing but the speaker outline, dot and arrow
            bool rich = full && !spk;
            _worldRoot.gameObject.SetActive(rich);
            _speakerWorld.gameObject.SetActive(spk);
            _list.gameObject.SetActive(spk);
            _arrowRoot.gameObject.SetActive(rich && theme.showEdgeArrows);
            _chipRoot.gameObject.SetActive(rich);
            _compass.gameObject.SetActive(rich && theme.showCompass);
            _radar.gameObject.SetActive(rich && theme.showRadar);
            _banner.gameObject.SetActive(rich);
            _glasses.gameObject.SetActive(!full);
            _reticle.gameObject.SetActive(rich);

            // expire entries that stopped appearing in updates
            _remove.Clear();
            foreach (var kv in _entries) if (_time - kv.Value.lastUpdate > theme.removeGraceSeconds) _remove.Add(kv.Key);
            foreach (var id in _remove) DestroyEntry(id);
            if (_pendingLeft > 0f) { _pendingLeft -= dt; if (_pendingLeft <= 0f) _pendingId = null; }

            string selId = ResolveSelectedSourceId();
            float hfov = cameraStreamer != null && cameraStreamer.HfovDeg > 1f ? cameraStreamer.HfovDeg : (UsingFake && fake != null ? fake.hfovDeg : theme.fallbackCameraHfovDeg);

            _located.Clear(); _unlocated.Clear();
            foreach (var en in _entries.Values)
            {
                if (en.src.BearingDeg.HasValue) _located.Add(en); else _unlocated.Add(en);
                en.hasBox = false; en.elevFallback = 0f;
                if (en.src.InView && en.src.Bbox != null && en.src.Bbox.Length >= 4)
                {
                    UiMath.BboxToAngles(en.src.Bbox, hfov, theme.cameraAspect, out _, out float pitch, out float wd, out float hd);
                    en.hasBox = true;
                    en.box = new Vector2(UiMath.AngleToSize(wd, theme.markerDistance), UiMath.AngleToSize(hd, theme.markerDistance));
                    en.elevFallback = pitch;
                }
            }
            _located.Sort(ByBearing);

            if (spk)
            {
                TickSpeakers(dt, headPosL, headYaw);
                _alert.Tick(dt, RelAlert(headYaw));   // keep alerts expiring in this mode
                return;
            }

            // stack markers whose bearings are close so their data boxes do not overlap
            float prevB = float.NaN; int stack = 0;
            foreach (var en in _located)
            {
                float b = en.src.BearingDeg.Value;
                if (!float.IsNaN(prevB) && Mathf.Abs(UiMath.NormalizeDeg(b - prevB)) < theme.stackWithinDeg) stack++; else stack = 0;
                prevB = b;
                var marker = EnsureMarker(en, false);
                var mi = MakeInput(en, selId, dt);
                mi.relDeg = UiMath.NormalizeDeg(en.src.BearingDeg.Value - headYaw);
                mi.stackY = stack * theme.stackSpacing; mi.headPosL = headPosL;
                marker.Tick(mi);
                if (en.chip != null) en.chip.gameObject.SetActive(false);
            }
            int chipI = 0;
            foreach (var en in _unlocated)
            {
                if (en.marker != null) en.marker.gameObject.SetActive(false);
                if (chipI >= theme.maxChips) { if (en.chip != null) en.chip.gameObject.SetActive(false); continue; }
                var chip = EnsureMarker(en, true);
                chip.transform.localScale = Vector3.one * theme.chipScale;
                chip.transform.localPosition = new Vector3((chipI - (Mathf.Min(_unlocated.Count, theme.maxChips) - 1) * 0.5f) * theme.chipSpacing * 2f, 0f, 0f);
                chip.transform.localRotation = Quaternion.identity;
                var mi = MakeInput(en, selId, dt); mi.isChip = true; mi.inView = false;
                chip.Tick(mi);
                chipI++;
            }

            // edge arrows for located sources that are out of view; compass pips for all located sources
            _pips.Clear();
            int leftStack = 0, rightStack = 0;
            foreach (var en in _located)
            {
                var s = en.src;
                float rel = UiMath.NormalizeDeg(s.BearingDeg.Value - headYaw);
                bool outOfView = !s.InView;
                if (outOfView && full && theme.showEdgeArrows)
                {
                    if (en.arrow == null)
                    {
                        var go = new GameObject("Arrow_" + s.SourceId); go.transform.SetParent(_arrowRoot, false);
                        en.arrow = go.AddComponent<EdgeArrowView>(); en.arrow.Build(theme);
                    }
                    en.arrow.gameObject.SetActive(true);
                    int si = rel >= 0f ? rightStack++ : leftStack++;
                    float a = s.State == TrackState.Tracking ? theme.alphaTracking : (s.State == TrackState.LastSeen ? theme.alphaLastSeen : theme.alphaLost);
                    en.arrow.Tick(s, rel, s.ElevationDeg ?? 0f, si, a, dt);
                }
                else if (en.arrow != null) en.arrow.gameObject.SetActive(false);
                _pips.Add(new CompassPip
                {
                    id = s.SourceId, bearingDeg = s.BearingDeg.Value, color = s.State == TrackState.Tracking ? theme.LabelColor(s.Label) : theme.lostColor,
                    selected = s.SourceId == selId, lost = s.State == TrackState.Lost, alpha = 1f,
                });
            }
            foreach (var en in _unlocated) if (en.arrow != null) en.arrow.gameObject.SetActive(false);
            if (full && theme.showCompass) _compass.Tick(headYaw, _pips);
            if (full && theme.showRadar) _radar.Tick(headYaw, _pips);

            // gaze hover
            UpdateHover(dt, full);

            // banner / glasses / alert / reticle
            string targetText = TargetText(selId);
            if (full) _banner.Tick(_state, targetText, _mode, _replaySeconds, dt);
            else TickGlasses(selId, targetText, headYaw);
            _alert.Tick(dt, RelAlert(headYaw));
            if (full) _reticle.Tick(_hoverId != null, 0f, dt);
        }

        // ------------------------------------------------------------------ speaker-only HUD
        /// <summary>Speaking = labelled speech, tracking and loud enough.</summary>
        bool IsSpeaking(Source s) => s.Label == "speech" && s.State == TrackState.Tracking && s.LevelDb >= theme.speakingMinDb;

        void TickSpeakers(float dt, Vector3 headPosL, float headYaw)
        {
            // nothing is chosen automatically: every talking speaker is a row in the picker, the wearer chooses one
            string selId = ResolveSelectedSourceId();
            float headPitch = Mathf.Asin(Mathf.Clamp(_worldRoot.InverseTransformDirection(Head.forward).y, -1f, 1f)) * Mathf.Rad2Deg;
            _rows.Clear();
            var ptr = Pointer();
            string best = null; float bestScore = float.MaxValue, bestDist = 1.5f;
            foreach (var en in _located)
            {
                var s = en.src;
                if (IsSpeaking(s)) en.lastSpeaking = _time;
                bool visible = _time - en.lastSpeaking <= theme.speakerHoldSeconds;
                float rel = UiMath.NormalizeDeg(s.BearingDeg.Value - headYaw);
                float relElev = (s.ElevationDeg ?? en.elevFallback) - headPitch;
                bool inView = Mathf.Abs(rel) <= theme.speakerViewHalfDeg && Mathf.Abs(relElev) <= theme.speakerViewHalfVertDeg;
                int num = NumberFor(s.SourceId, visible);
                Color own = num > 0 ? SpeakerColor(num) : default;
                bool chosen = s.SourceId == selId;
                if (visible) AddRow(s, num, own, chosen);
                // pointing at a speaker who is in view: a light colour layer over them (the pointer is the hand ray, or the head if no hand is tracked)
                if (visible && inView)
                {
                    Vector3 wp = _worldRoot.TransformPoint(headPosL + UiMath.DirFromYawElev(s.BearingDeg.Value, s.ElevationDeg ?? en.elevFallback) * theme.markerDistance);
                    Vector2 half = en.hasBox ? en.box * 0.5f
                        : new Vector2(UiMath.AngleToSize(theme.speakerBoxDeg.x, theme.markerDistance), UiMath.AngleToSize(theme.speakerBoxDeg.y, theme.markerDistance)) * 0.5f;
                    if (GazeHitFrom(ptr, wp, half, out float sc) && sc < bestScore) { bestScore = sc; best = s.SourceId; bestDist = Vector3.Dot(wp - ptr.position, ptr.forward); }
                }
                bool hovered = s.SourceId == _hoverId && inView;
                // the chosen speaker is marked: a rectangle if in view, an edge arrow if not. A pointed-at speaker gets only the tint.
                bool show = visible && (chosen || hovered);
                if (show && en.speaker == null) en.speaker = NewSpeaker(s.SourceId);
                if (en.speaker == null) continue;
                en.speaker.Tick(new SpeakerInput
                {
                    color = own,
                    src = s, relDeg = rel, relElevDeg = relElev, inView = inView, headPosL = headPosL,
                    hasBox = en.hasBox, boxSize = en.box, elevFallbackDeg = en.elevFallback, visible = show, dt = dt,
                    selected = chosen, hovered = hovered, number = num, name = NameFor(s),
                });
            }
            // heard but not located: still a row in the picker; if chosen, a plain dot instead of an arrow
            foreach (var en in _unlocated)
            {
                var s = en.src;
                if (IsSpeaking(s)) en.lastSpeaking = _time;
                bool visible = _time - en.lastSpeaking <= theme.speakerHoldSeconds;
                int num = NumberFor(s.SourceId, visible);
                Color own = num > 0 ? SpeakerColor(num) : default;
                bool chosen = s.SourceId == selId;
                if (visible) AddRow(s, num, own, chosen);
                bool show = visible && chosen;
                if (show && en.speaker == null) en.speaker = NewSpeaker(s.SourceId);
                if (en.speaker != null)
                    en.speaker.Tick(new SpeakerInput { src = s, noDirection = true, color = own, visible = show, selected = chosen, dt = dt });
            }
            if (best == _hoverId) _hoverTime += dt;
            else { _hoverId = best; _hoverTime = 0f; }
            if (best != null) NoteTarget(bestDist);
            // nearest first, so a source that is hard to point at can still be picked from the list; no distance = last
            // distances are bucketed to 0.5 m so two sources about as far away keep a steady order (by number) instead of swapping every frame
            _rows.Sort((a, b) => { int c = Bucket(a.dist).CompareTo(Bucket(b.dist)); return c != 0 ? c : a.number.CompareTo(b.number); });
            _listCursor = _rows.Count == 0 ? 0 : Mathf.Clamp(_listCursor, 0, _rows.Count - 1);
            // pointing at a world speaker lights that speaker's row; pointing at a row lights it (and a pinch then chooses it)
            int hoverRow = -1;
            if (_hoverId != null) { hoverRow = _rows.FindIndex(r => r.id == _hoverId); if (hoverRow >= 0) _listCursor = hoverRow; }
            else
            {
                float rowBest = float.MaxValue, rowDist = 1.2f;
                for (int i = 0; i < _list.VisibleRows; i++)
                    if (_list.RowAt(i, out int ri, out Vector3 rc, out Vector2 rh) && GazeHitFrom(ptr, rc, rh, out float rs) && rs < rowBest) { rowBest = rs; _listCursor = ri; hoverRow = ri; rowDist = Vector3.Dot(rc - ptr.position, ptr.forward); }
                if (rowBest < float.MaxValue) NoteTarget(rowDist);
            }
            _hoverRowNow = hoverRow;
            if (hoverRow >= 0) _scrollCursor = false;
            _list.Tick(_rows, hoverRow >= 0 ? hoverRow : (_scrollCursor ? _listCursor : -1), dt);
        }

        void AddRow(Source s, int num, Color own, bool chosen) => _rows.Add(new SpeakerRow { id = s.SourceId, number = num, name = NameFor(s), color = own, selected = chosen, dist = s.DistanceM ?? float.MaxValue });

        SpeakerView NewSpeaker(string id)
        {
            var go = new GameObject("Speaker_" + id); go.transform.SetParent(_speakerWorld, false);
            var sv = go.AddComponent<SpeakerView>(); sv.Build(theme, _headLocked);
            return sv;
        }

        // ------------------------------------------------------------------ demo / test panel
        void BuildDemo()
        {
            _demo.Build(theme);
            _audioOut = FindFirstObjectByType<AudioOutPlayer>();
            _dData = _demo.AddButton("FAKE FALLBACK", () => useFake = !useFake, 0, 1, 0);
            _dCubes = _demo.AddButton("CUBES", () => spawnCubeSources = !spawnCubeSources, 1, 1, 0);
            _dNums = _demo.AddButton("NUMBERS", () => theme.speakerListShowNumbers = !theme.speakerListShowNumbers, 0, 1, 1);
            _dMode = _demo.AddButton("MODE", CycleMode, 1, 1, 1);
            _demo.AddButton("VOICE -", () => VolStep(-3f, 0f), 0, 1, 2);
            _demo.AddButton("VOICE +", () => VolStep(3f, 0f), 1, 1, 2);
            _demo.AddButton("ROOM -", () => VolStep(0f, -3f), 0, 1, 3);
            _demo.AddButton("ROOM +", () => VolStep(0f, 3f), 1, 1, 3);
            _demo.AddButton("REPLAY", () => DemoCmd(ProtoAction.Replay), 0, 1, 4);
            _demo.AddButton("STOP", () => DemoCmd(ProtoAction.Stop), 1, 1, 4);
            _demo.AddButton("PREV", () => DemoCmd(ProtoAction.Prev), 0, 1, 5);
            _demo.AddButton("NEXT", () => DemoCmd(ProtoAction.Next), 1, 1, 5);
            _dSound = _demo.AddButton("CUBE SOUND", () => cubeSound = !cubeSound, 0, 2, 6);
            _demo.SetOpen(false);
        }

        void VolStep(float innerDb, float outerDb)
        {
            if (_audioOut == null) _audioOut = FindFirstObjectByType<AudioOutPlayer>();
            if (_audioOut != null) _audioOut.SetVolumesDb(_audioOut.InnerDb + innerDb, _audioOut.OuterDb + outerDb, false);   // the headset applies the levels itself
        }

        void CycleMode()
        {
            var next = (Mode)(((int)_mode + 1) % 3);
            if (UsingFake) { if (fake != null) fake.SetMode(next); }
            else if (hub != null && hub.Connection != null) hub.Connection.SendCommand(new SetMode { Mode = next });
        }

        void DemoCmd(ProtoAction a)
        {
            if (UsingFake)
            {
                if (fake == null) return;
                if (a == ProtoAction.Replay) fake.Replay();
                else if (a == ProtoAction.Stop) fake.Stop();
                else StepSelection(a == ProtoAction.Next ? 1 : -1);
            }
            else if (input != null) input.Send(a, null, "ui");
        }

        // fake mode has no hub to cycle for us: step through the picker rows
        void StepSelection(int dir)
        {
            if (_rows.Count == 0) return;
            int cur = _rows.FindIndex(r => r.selected);
            int next = cur < 0 ? (dir > 0 ? 0 : _rows.Count - 1) : (cur + dir + _rows.Count) % _rows.Count;
            if (next != cur) TrySelectId(_rows[next].id, "ui");
        }

        // ------------------------------------------------------------------ pointer cursor
        void NoteTarget(float metresAlongRay) { _tgtFound = true; _tgtDist = Mathf.Max(0.3f, metresAlongRay); }

        void TickCursor(float dt)
        {
            if (_cursor == null) return;
            var ptr = Pointer(); bool hand = ptr != Head;
            float pinch = 0f;
            foreach (var h in _hands) if (h != null && h.IsTracked) pinch = Mathf.Max(pinch, h.GetFingerPinchStrength(OVRHand.HandFinger.Index));
            _cursor.Tick(ptr, _tgtFound ? _tgtDist : (hand ? 1.4f : 1.1f), _tgtFound, pinch, hand, dt);
            _tgtFound = false;
        }

        // per hand: T = tracked, P = pointer pose valid, pinch strength of the index finger, * while it counts as a pinch
        string HandDebug()
        {
            if (_hands.Length == 0) return "none found";
            var sb = new StringBuilder();
            foreach (var h in _hands)
            {
                if (h == null) continue;
                sb.Append(h.name.StartsWith("L") ? "L" : "R").Append(h.IsTracked ? "T" : "-").Append(h.IsPointerPoseValid ? "P" : "-")
                  .Append(h.GetFingerPinchStrength(OVRHand.HandFinger.Index).ToString("0.0")).Append(h.GetFingerIsPinching(OVRHand.HandFinger.Index) ? "*" : "").Append(' ');
            }
            return sb.ToString();
        }

        void TickDemo(float dt)
        {
            if (_demo == null || !DevMode.On) return;
            _demo.Tick(Pointer(), dt);
            if (_demo.TryGetHoverWorld(out var demoAt)) NoteTarget(Vector3.Dot(demoAt - Pointer().position, Pointer().forward));
            if (_time < _nextReadout) return;
            _nextReadout = _time + 0.25f;
            _demo.SetLabel(_dData, "FAKE FALLBACK " + (useFake ? "ON" : "OFF"));
            _demo.SetLabel(_dCubes, "CUBES " + (spawnCubeSources ? "ON" : "OFF"));
            _demo.SetLabel(_dSound, "CUBE SOUND " + (cubeSound ? "ON" : "OFF"));
            _demo.SetLabel(_dNums, "NUMBERS " + (theme.speakerListShowNumbers ? "ON" : "OFF"));
            _demo.SetLabel(_dMode, "MODE " + _mode);
            if (!_demo.Open) return;
            var sb = new StringBuilder();
            bool conn = hub != null && hub.Connection != null && hub.Connection.Events.State == LinkState.Connected;
            sb.Append("DATA ").Append(UsingFake ? "FAKE" : "REAL").Append("  HUB ").Append(conn ? "UP " : "DOWN ").Append(hub != null ? hub.Host : "-").Append('\n');
            sb.Append("STATE ").Append(_state).Append("  CHOSEN ").Append(ResolveSelectedSourceId() ?? "-").Append('\n');
            sb.Append("HANDS ").Append(HandDebug()).Append("  POINT ").Append(_demo.HoverIndex >= 0 ? "panel" : _hoverId ?? (_listCursor >= 0 && _rows.Count > 0 ? "row" : "-")).Append('\n');
            if (_audioOut != null) sb.Append("VOICE ").Append(_audioOut.InnerDb.ToString("0")).Append("  ROOM ").Append(_audioOut.OuterDb.ToString("0")).Append(" dB\n");
            int n = 0;
            foreach (var en in _entries.Values)
            {
                if (n++ >= 3) break;
                var s = en.src;
                sb.Append(s.SourceId).Append(' ').Append(s.Label).Append(' ').Append(s.LevelDb.ToString("0")).Append("dB ")
                  .Append(s.BearingDeg.HasValue ? s.BearingDeg.Value.ToString("0") + "deg " : "no-dir ")
                  .Append(s.DistanceM.HasValue ? s.DistanceM.Value.ToString("0.0") + "m " : "")
                  .Append(s.State).Append('\n');
            }
            sb.Append("--- LOG\n").Append(DevLog.Recent(6));
            _demo.SetReadout(sb.ToString());
        }

        // ------------------------------------------------------------------ test audio cubes (fake data only)
        [Tooltip("In fake mode, put two cubes in the room that act as the two speakers: the HUD follows where they are.")]
        [SerializeField] bool spawnCubeSources = true;
        Transform[] _cubes;
        // four cubes at different directions and depths: ids match the fake feed's talkers (s_3 stays the scripted music)
        static readonly string[] CubeIds = { "s_1", "s_2", "s_4", "s_5" };
        static readonly float[] CubeYawOffsetDeg = { -25f, 30f, 5f, -70f };
        static readonly float[] CubeDist = { 2f, 2f, 1.0f, 3.5f };       // metres: two at arm-and-a-bit, one close, one far
        static readonly float[] CubeElev = { 0f, 0f, -8f, 6f };          // degrees: the close one low, the far one a little high
        static readonly float[] CubeF0 = { 180f, 260f, 140f, 330f };
        static readonly float[] CubeRate = { 3.0f, 2.25f, 1.75f, 2.5f };   // multiples of 0.25 so each 4 s clip loops with no click

        [Tooltip("Cubes make sound: the chosen cube plays at the VOICE level, the other at the ROOM level (nothing chosen: both at full).")]
        [SerializeField] bool cubeSound = true;
        AudioSource[] _cubeAudio;
        float[] _cubeGain = new float[4];

        // the product's focus behaviour, played locally for the cubes: chosen speaker = inner level, everyone else = outer level
        void TickCubeSound(float dtForSound)
        {
            if (_cubeAudio == null) return;
            string sel = ResolveSelectedSourceId();
            float inner = _audioOut != null ? _audioOut.InnerDb : 0f, outer = _audioOut != null ? _audioOut.OuterDb : -12f;
            for (int i = 0; i < _cubeAudio.Length; i++)
            {
                if (_cubeAudio[i] == null) continue;
                // an AudioSource stops when its object is deactivated (CUBES off, real data arriving): start it again
                if (_cubeAudio[i].gameObject.activeInHierarchy && !_cubeAudio[i].isPlaying) _cubeAudio[i].Play();
                float db = !cubeSound ? -80f : sel == null ? 0f : (sel == CubeIds[i] ? inner : outer);
                float target = db <= -79f ? 0f : 0.5f * Mathf.Pow(10f, db / 20f);
                _cubeGain[i] = UiObjects.Ease(_cubeGain[i], target, 8f, dtForSound);   // glide, no clicks
                _cubeAudio[i].volume = Mathf.Clamp01(_cubeGain[i]);
            }
        }

        // a looping voice-like sound: harmonics of f0 with a little vibrato, chopped into syllables at the given rate; loops with no click
        static AudioClip VoiceClip(string name, float f0, float syllableHz)
        {
            const int rate = 16000, seconds = 4;
            var data = new float[rate * seconds];
            var rng = new System.Random(f0.GetHashCode());
            for (int n = 0; n < data.Length; n++)
            {
                float t = n / (float)rate;
                float vib = 1f + 0.015f * Mathf.Sin(2f * Mathf.PI * 5f * t);
                float s = 0f;
                for (int h = 1; h <= 8; h++) s += Mathf.Sin(2f * Mathf.PI * f0 * h * vib * t) / h;
                float env = 0.5f * (1f + Mathf.Sin(2f * Mathf.PI * syllableHz * t - 1.2f)); env *= env;
                float breath = ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
                data[n] = (s * 0.35f + breath) * env;
            }
            var clip = AudioClip.Create(name, data.Length, 1, rate, false);
            clip.SetData(data, 0);
            return clip;
        }

        void TickCubes(Vector3 headPosL, float headYaw)
        {
            if (_cubes != null)
            {
                bool on = spawnCubeSources && UsingFake;
                foreach (var c in _cubes) if (c != null && c.gameObject.activeSelf != on) c.gameObject.SetActive(on);
            }
            if (!spawnCubeSources && fake != null) fake.ClearAnchors();
            if (!spawnCubeSources || fake == null || !UsingFake || !Application.isPlaying) return;
            if (_cubes == null)
            {
                if (_time < 1.5f) return;   // let head tracking settle before placing them
                Transform wp = rig != null && rig.trackingSpace != null ? rig.trackingSpace : transform;
                var root = new GameObject("AudioCubes").transform; root.SetParent(wp, false);
                _cubes = new Transform[CubeIds.Length]; _cubeAudio = new AudioSource[CubeIds.Length];
                for (int i = 0; i < CubeIds.Length; i++)
                {
                    var c = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    c.name = "AudioCube" + (i + 1);
                    UiObjects.Kill(c.GetComponent<Collider>());
                    c.transform.SetParent(root, false);
                    c.transform.localScale = Vector3.one * 0.3f;
                    c.transform.localPosition = headPosL + UiMath.DirFromYawElev(headYaw + CubeYawOffsetDeg[i], CubeElev[i]) * CubeDist[i];
                    _cubes[i] = c.transform;
                    // a sound on the cube: a looping voice-like tone (a different pitch and rhythm per cube), 3D so it comes from the cube
                    var src = c.AddComponent<AudioSource>();
                    src.clip = VoiceClip("CubeVoice" + (i + 1), CubeF0[i], CubeRate[i]);
                    src.loop = true; src.spatialBlend = 1f; src.dopplerLevel = 0f;
                    src.rolloffMode = AudioRolloffMode.Logarithmic; src.minDistance = 0.6f; src.maxDistance = 12f;
                    src.volume = 0f; src.playOnAwake = false;
                    _cubeAudio[i] = src;
                    src.Play();
                }
            }
            for (int i = 0; i < _cubes.Length; i++)
            {
                Vector3 d = _cubes[i].localPosition - headPosL;
                float horiz = new Vector2(d.x, d.z).magnitude;
                // the overlay box covers the whole cube (a little larger than its width, from where you stand)
                float sizeDeg = 2f * Mathf.Atan2(_cubes[i].localScale.x * 0.5f * 1.2f, Mathf.Max(0.3f, d.magnitude)) * Mathf.Rad2Deg;
                string id = CubeIds[i];
                fake.SetAnchor(id, UiMath.YawOf(d), Mathf.Atan2(d.y, horiz) * Mathf.Rad2Deg, sizeDeg, d.magnitude);
                // the cube wears the same colour as its row in the picker
                if (_speakerNum.TryGetValue(id, out int cn))
                {
                    var r = _cubes[i].GetComponent<Renderer>();
                    if (r != null) r.material.color = SpeakerColor(cn);
                }
            }
            TickCubeSound(dtForSound: Time.unscaledDeltaTime);
        }

        static float Bucket(float distM) => distM >= float.MaxValue * 0.5f ? float.MaxValue : Mathf.Round(distM / 0.5f);

        // numbers (for the team, not the wearer) go only to sources that actually talk, in the order they first do; music never takes one
        int NumberFor(string id, bool talking)
        {
            if (_speakerNum.TryGetValue(id, out int n)) return n;
            if (!talking) return 0;
            n = _speakerNum.Count + 1; _speakerNum[id] = n;
            return n;
        }

        /// <summary>One colour per speaker number, so a speaker keeps the same colour on its outline, its arrow and its row.</summary>
        Color SpeakerColor(int number)
        {
            // numbers 1 to 5 are the design's colours; later numbers take the midpoint of two neighbours around the loop
            var p = theme.lookSources;
            if (p == null || p.Length == 0) return theme.cyan;
            int i = number - 1;
            if (i < p.Length) return p[i];
            int k = (i - p.Length) % p.Length;
            return Color.Lerp(p[k], p[(k + 1) % p.Length], 0.5f);
        }

        // base names (human, object, unknown; animal once the hub sends one) stand in for the hub's coarse labels; a specific name passes through
        static string NameFor(Source s)
        {
            string l = (s.Label ?? "").Trim().ToLowerInvariant();
            switch (l)
            {
                case "speech": case "human": case "person": case "voice": return "human";
                case "music": case "alarm": case "object": return "object";
                case "other": case "unknown": case "": return "unknown";
                default: return l;
            }
        }

        MarkerInput MakeInput(Entry en, string selId, float dt)
        {
            bool sel = en.src.SourceId == selId;
            bool hov = en.src.SourceId == _hoverId && _hoverTime >= theme.hoverHighlightSeconds;
            return new MarkerInput
            {
                src = en.src, ageSeconds = Mathf.Max(0f, (_latestTs - en.src.LastObservedUs) / 1e6f),
                selected = sel, hovered = hov, pending = en.src.SourceId == _pendingId && !sel,
                dwell01 = en.src.SourceId == _hoverId ? Mathf.Clamp01(_hoverTime / Mathf.Max(0.05f, theme.dwellSeconds)) : 0f,
                inView = en.src.InView, hasBox = en.hasBox, boxSize = en.box, elevFallbackDeg = en.elevFallback, dt = dt,
            };
        }

        SourceMarkerView EnsureMarker(Entry en, bool chip)
        {
            ref SourceMarkerView slot = ref (chip ? ref en.chip : ref en.marker);
            if (slot == null)
            {
                var go = new GameObject((chip ? "Chip_" : "Marker_") + en.src.SourceId);
                go.transform.SetParent(chip ? _chipRoot : _worldRoot, false);
                slot = go.AddComponent<SourceMarkerView>();
                slot.Build(theme, chip);
            }
            if (!slot.gameObject.activeSelf) slot.gameObject.SetActive(true);
            return slot;
        }

        void DestroyEntry(string id)
        {
            if (!_entries.TryGetValue(id, out var en)) return;
            if (en.marker != null) UiObjects.Kill(en.marker.gameObject);
            if (en.chip != null) UiObjects.Kill(en.chip.gameObject);
            if (en.arrow != null) UiObjects.Kill(en.arrow.gameObject);
            if (en.speaker != null) UiObjects.Kill(en.speaker.gameObject);
            _entries.Remove(id);
            DevLog.Add("src", "-" + id);
            if (_hoverId == id) _hoverId = null;
            if (_localChosen == id) _localChosen = null;   // the hub forgets a vanished source too, and never reuses its id
        }

        string TargetText(string selId)
        {
            if (selId != null && _entries.TryGetValue(selId, out var en))
                return theme.DisplayLabel(en.src.Label) + " " + selId;
            if (!string.IsNullOrEmpty(_stateTargetId) && _state != HubState.Listening) return _stateTargetId;
            return theme.noTarget;
        }

        void TickGlasses(string selId, string targetText, float headYaw)
        {
            float? rel = null; Color accent = theme.StateColor(_state);
            if (selId != null && _entries.TryGetValue(selId, out var en) && en.src.BearingDeg.HasValue) rel = UiMath.NormalizeDeg(en.src.BearingDeg.Value - headYaw);
            var st = theme.GetState(_state);
            _glasses.Tick(UiMath.Fmt(theme.glassesTitle, "{target}", targetText, "{state}", st.title), UiMath.Fmt(theme.glassesDetail, "{state}", st.title, "{target}", targetText), rel, accent);
        }

        // ------------------------------------------------------------------ choice kept on the headset (real hub)
        void ResetForNewSource()
        {
            // fake and real sources never mix: start clean (numbers and colours too) when the data source changes
            DevLog.Add("data", UsingFake ? "fake" : "real");
            string keep = _localChosen;
            _remove.Clear(); _remove.AddRange(_entries.Keys);
            foreach (var id in _remove) DestroyEntry(id);
            _speakerNum.Clear(); _targetToSource.Clear();
            _localChosen = keep;   // a short stall must not forget the choice; it only shows once that source is listed again (the hub never reuses an id)
            _pendingId = null; _pendingLeft = 0f; _hoverId = null;
            _state = HubState.Listening; _stateTargetId = null;
        }

        // stop / next / prev can also come from InputSender's buttons: keep the local choice in step with what the hub does
        void OnInputSent(ProtoAction a, string sourceId)
        {
            DevLog.Add("in", a + (sourceId != null ? " " + sourceId : ""));
            if (UsingFake || _hubStateSeen) return;
            if (a == ProtoAction.Stop) _localChosen = null;
            else if (a == ProtoAction.Next || a == ProtoAction.Prev) _localChosen = StepLocal(a == ProtoAction.Next ? 1 : -1);
        }

        // the same order and wrap-around as the hub's next/prev (hub/events/selection.py): left to right by bearing, no bearing last, lost skipped
        string StepLocal(int dir)
        {
            var pool = new List<Source>();
            foreach (var en in _entries.Values) if (en.src.State != TrackState.Lost || en.src.SourceId == _localChosen) pool.Add(en.src);
            if (pool.Count == 0) return _localChosen;
            pool.Sort(HubCycleOrder);
            int i = pool.FindIndex(s => s.SourceId == _localChosen);
            return pool[i >= 0 ? (i + dir + pool.Count) % pool.Count : (dir > 0 ? 0 : pool.Count - 1)].SourceId;
        }

        static int HubCycleOrder(Source a, Source b)
        {
            bool ha = a.BearingDeg.HasValue, hb = b.BearingDeg.HasValue;
            if (ha != hb) return ha ? -1 : 1;
            int c = ha ? UiMath.NormalizeDeg(a.BearingDeg.Value).CompareTo(UiMath.NormalizeDeg(b.BearingDeg.Value)) : 0;
            if (c != 0) return c;
            c = TrailingNumber(a.SourceId).CompareTo(TrailingNumber(b.SourceId));
            return c != 0 ? c : string.CompareOrdinal(a.SourceId, b.SourceId);
        }

        static int TrailingNumber(string id)
        {
            int end = id.Length, start = end;
            while (start > 0 && char.IsDigit(id[start - 1])) start--;
            return start < end && int.TryParse(id.Substring(start), out int n) ? n : int.MaxValue;
        }

        // ------------------------------------------------------------------ gaze selection
        bool GazeHit(Vector3 worldCenter, Vector2 half, out float score) => GazeHitFrom(Head, worldCenter, half, out score);

        /// <summary>The hand's pointing ray when a hand is tracked, otherwise where the head points.</summary>
        Transform Pointer()
        {
            foreach (var h in _hands)
                if (h != null && h.IsTracked && h.IsPointerPoseValid && h.PointerPose != null) return h.PointerPose;
            return Head;
        }

        bool GazeHitFrom(Transform from, Vector3 worldCenter, Vector2 half, out float score)
        {
            float pad = from == Head ? 0.03f : 0.07f;   // a hand ray is harder to hold steady: a more forgiving target
            return DemoPanelView.Hit(from, worldCenter, half, pad, out score);
        }

        void UpdateHover(float dt, bool full)
        {
            string best = null; float bestScore = float.MaxValue;
            if (full)
                foreach (var kv in _entries)
                {
                    TestHover(kv.Key, kv.Value.marker, ref best, ref bestScore);
                    TestHover(kv.Key, kv.Value.chip, ref best, ref bestScore);
                }
            if (best == _hoverId) _hoverTime += dt;
            else { _hoverId = best; _hoverTime = 0f; }
        }

        void TestHover(string id, SourceMarkerView mv, ref string best, ref float bestScore)
        {
            if (mv == null || !mv.gameObject.activeInHierarchy) return;
            if (GazeHit(mv.PlateCenterWorld, mv.PlateHalf, out float sc) && sc < bestScore) { bestScore = sc; best = id; }
            if (mv.BoxVisible && GazeHit(mv.BoxCenterWorld, mv.BoxHalf, out sc) && sc < bestScore) { bestScore = sc; best = id; }
        }

        /// <summary>Select the source under the head gaze, if any. Returns true when a select was issued.</summary>
        public bool TrySelectGazed(string modality = "head")
        {
            if (string.IsNullOrEmpty(_hoverId)) return false;
            return TrySelectId(_hoverId, modality);
        }

        /// <summary>Select the speaker on the picker's cursor row (speaker-only HUD).</summary>
        public bool TrySelectListRow(string modality = "hand")
        {
            // only a row you are pointing at (or scrolled to with the thumbstick) can be chosen: a stray pinch does nothing
            if (!_speakerOnly || _rows.Count == 0 || (_hoverRowNow < 0 && !_scrollCursor)) return false;
            _scrollCursor = false;
            return TrySelectId(_rows[Mathf.Clamp(_listCursor, 0, _rows.Count - 1)].id, modality);
        }

        /// <summary>Move the picker cursor by one row (+1 = down).</summary>
        public void ScrollList(int step) { if (_speakerOnly && _rows.Count > 0) { _scrollCursor = true; _listCursor = Mathf.Clamp(_listCursor + step, 0, _rows.Count - 1); } }
        public int ListCursor => _listCursor;
        public int ListRowCount => _rows.Count;

        bool TrySelectId(string id, string modality)
        {
            DevLog.Add("ui", (id == ResolveSelectedSourceId() ? "stop " : "choose ") + id + " (" + modality + ")");
            // choosing the speaker who is already chosen switches the focus off
            if (id == ResolveSelectedSourceId())
            {
                if (UsingFake) { if (fake != null) fake.Stop(); }
                else if (input != null) input.Send(ProtoAction.Stop, null, modality);
                _localChosen = null; _pendingId = null; _pendingLeft = 0f;
                return true;
            }
            if (UsingFake) { if (fake != null) fake.Select(id); }
            else if (input != null && sendExplicitSelect)
            {
                var head = Head;
                Vector3 o = _worldRoot.InverseTransformPoint(head.position);
                Vector3 d = _worldRoot.InverseTransformDirection(head.forward);
                input.Send(ProtoAction.Select, id, modality, o, d);
                _localChosen = id;
            }
            _pendingId = id; _pendingLeft = theme.pendingSelectSeconds;
            return true;
        }

        // ------------------------------------------------------------------ input (headset only)
        void PollInput()
        {
            bool select = OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch);
            string modality = "controller";
            ScanHands();
            for (int i = 0; i < _hands.Length; i++)
            {
                var h = _hands[i];
                bool pinching = h != null && h.IsTracked && h.GetFingerIsPinching(OVRHand.HandFinger.Index);
                if (pinching && !_pinchWas[i] && !select) { select = true; modality = "hand"; DevLog.Add("pinch", h.name + " at " + (_hoverId ?? (_hoverRowNow >= 0 ? "row " + _hoverRowNow : "nothing"))); }
                _pinchWas[i] = pinching;
            }
            if (select && DevMode.On && _demo != null && _demo.Press()) { }   // a pinch on a demo-panel button
            else if (select && !TrySelectGazed("head")) TrySelectListRow(modality);
            // testing fallback when hand tracking is not working: right thumbstick up/down scrolls the picker
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstickUp, OVRInput.Controller.RTouch)) ScrollList(-1);
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstickDown, OVRInput.Controller.RTouch)) ScrollList(1);
            // B + Y together toggle DevMode (InputSender skips Stop / Next while the other button is held)
            bool chord = OVRInput.Get(OVRInput.Button.Two, OVRInput.Controller.RTouch) && OVRInput.Get(OVRInput.Button.Four, OVRInput.Controller.LTouch);
            if (chord && !_devChordWas) DevMode.Toggle();
            _devChordWas = chord;
            if (OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.RTouch)) ToggleHudMode();
            if (UsingFake && fake != null)
            {
                if (OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.RTouch) && !OVRInput.Get(OVRInput.Button.Four, OVRInput.Controller.LTouch)) fake.Stop();
                if (OVRInput.GetDown(OVRInput.Button.Three, OVRInput.Controller.LTouch)) fake.Replay();
            }
        }

        void ScanHands()
        {
            if (Time.unscaledTime < _nextHandScan) return;
            _nextHandScan = Time.unscaledTime + 2f;
            var found = FindObjectsByType<OVRHand>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            System.Array.Sort(found, ByHandName);
            foreach (var h in found)
            {
                HandGridSkin.Apply(h, theme);
                // the hand's pointer pose is written in tracking-space coordinates, so it must live under the tracking space
                if (rig != null && rig.trackingSpace != null && h.PointerPose.parent != rig.trackingSpace) h.PointerPose.SetParent(rig.trackingSpace, false);
            }
            if (found.Length != _hands.Length) _pinchWas = new bool[found.Length];
            _hands = found;
        }
    }
}
