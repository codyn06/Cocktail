using System.Collections.Generic;
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
    /// right stick click = diagnostics overlay): A (right controller, Button.One) toggles Full / Glasses HUD.
    ///
    /// Selection: head-gaze reticle; resting the gaze on a marker highlights it and fills a dwell bar; right trigger or pinch while a
    /// marker is gazed calls <see cref="TrySelectGazed"/>, which sends InputSender.Send(select, source_id, "head", ray). InputSender's own
    /// ray-only select fires on the same press (see report: this duplicates the select unless InputSender is changed).
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
            public Source src; public long tsUs; public float lastUpdate;
            public SourceMarkerView marker, chip; public EdgeArrowView arrow;
            public Vector2 box; public bool hasBox; public float elevFallback;
            public SpeakerView speaker; public float lastSpeaking = -1000f;
        }

        readonly Dictionary<string, Entry> _entries = new Dictionary<string, Entry>();
        readonly Dictionary<string, string> _targetToSource = new Dictionary<string, string>();
        readonly List<Entry> _tmp = new List<Entry>(), _located = new List<Entry>(), _unlocated = new List<Entry>();
        readonly List<string> _remove = new List<string>();
        readonly List<CompassPip> _pips = new List<CompassPip>();
        static readonly System.Comparison<Entry> ByBearing = (a, b) => (a.src.BearingDeg ?? 0f).CompareTo(b.src.BearingDeg ?? 0f);

        Transform _worldRoot, _headLocked, _arrowRoot, _chipRoot, _fakeHead, _speakerWorld;
        SpeakerArrowView _speakerArrow; bool _speakerOnly;
        StateBannerView _banner; AlertBannerView _alert; GlassesPanelView _glasses; CompassStripView _compass; RadarRingView _radar; ReticleView _reticle;
        bool _init, _subscribed;
        float _time, _realAt = -1000f, _nextRigScan;
        Transform _lastHead, _lastWorldParent;

        // state
        HubState _state = HubState.Listening; string _stateTargetId; Mode _mode = Mode.Balanced; float _replaySeconds;
        float? _alertBearing;
        string _hoverId, _lastHoverId; float _hoverTime;
        string _pendingId; float _pendingLeft;
        long _latestTs;
        OVRHand[] _hands = new OVRHand[0]; bool[] _pinchWas = new bool[0]; float _nextHandScan;

        // ------------------------------------------------------------------ public API (also used by the self test)
        public UiTheme Theme => theme;
        public HudMode HudMode { get => hudMode; set => hudMode = value; }
        public bool UsingFake => useFake && (hub == null || _time - _realAt > realDataTimeout);
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
        public SpeakerArrowView SpeakerArrow => _speakerArrow;
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
        void Update() { Tick(Time.unscaledDeltaTime); if (pollInput && Application.isPlaying) PollInput(); }

        public void EnsureInit()
        {
            if (!_init)
            {
                _init = true;
                if (theme == null) theme = Resources.Load<UiTheme>("CocktailUiTheme");
                if (theme == null) { theme = ScriptableObject.CreateInstance<UiTheme>(); theme.name = "CocktailUiTheme (runtime default)"; }
                if (hub == null) hub = HubClient.Instance != null ? HubClient.Instance : FindFirstObjectByType<HubClient>();
                if (input == null) input = FindFirstObjectByType<InputSender>();
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
                _reticle = Child<ReticleView>("Reticle", new Vector3(0f, 0f, theme.reticleDistance)); _reticle.Build(theme);
                _speakerArrow = Child<SpeakerArrowView>("SpeakerArrow", theme.speakerArrowOffset); _speakerArrow.Build(theme);
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
        }

        void OnDestroy()
        {
            Unsubscribe();
            if (_worldRoot != null) UiObjects.Kill(_worldRoot.gameObject);
            if (_speakerWorld != null) UiObjects.Kill(_speakerWorld.gameObject);
            if (_headLocked != null) UiObjects.Kill(_headLocked.gameObject);
        }

        void OnRealSources(SourcesUpdated e) { _realAt = _time; ApplySources(e, false); }
        void OnRealState(StateChanged e) { _realAt = _time; ApplyState(e, false); }
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
            _tmp.Clear();
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
                if (!_entries.TryGetValue(s.SourceId, out var en)) { en = new Entry(); _entries[s.SourceId] = en; }
                en.src = s; en.tsUs = e.DeviceTsUs; en.lastUpdate = _time;
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

        string ResolveSelectedSourceId()
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
            if (rig == null && _time >= _nextRigScan) { _nextRigScan = _time + 1f; rig = FindFirstObjectByType<OVRCameraRig>(); }
            ReparentRoots();
            Vector3 headPosL = _worldRoot.InverseTransformPoint(Head.position);
            float headYaw = HeadYaw();
            if (fake != null) fake.headYawDeg = headYaw;

            bool full = hudMode == HudMode.Full;
            bool spk = full && _speakerOnly;   // speaker-only HUD: nothing but the speaker outline, dot and arrow
            bool rich = full && !spk;
            _worldRoot.gameObject.SetActive(rich);
            _speakerWorld.gameObject.SetActive(spk);
            _speakerArrow.gameObject.SetActive(spk);
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

            if (spk) { _hoverId = null; _hoverTime = 0f; TickSpeakers(dt, headPosL, headYaw); return; }

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
            float? relAlert = _alertBearing.HasValue ? UiMath.NormalizeDeg(_alertBearing.Value - headYaw) : (float?)null;
            _alert.Tick(dt, relAlert);
            if (full) _reticle.Tick(_hoverId != null, 0f, dt);
        }

        // ------------------------------------------------------------------ speaker-only HUD
        /// <summary>Speaking = labelled speech, tracking and loud enough.</summary>
        bool IsSpeaking(Source s) => s.Label == "speech" && s.State == TrackState.Tracking && s.LevelDb >= theme.speakingMinDb;

        void TickSpeakers(float dt, Vector3 headPosL, float headYaw)
        {
            Entry primary = null, loudest = null; float primaryDb = float.MinValue, loudestDb = float.MinValue;
            foreach (var en in _located)
            {
                var s = en.src;
                if (IsSpeaking(s)) en.lastSpeaking = _time;
                bool visible = _time - en.lastSpeaking <= theme.speakerHoldSeconds;
                float rel = UiMath.NormalizeDeg(s.BearingDeg.Value - headYaw);
                if (visible && en.speaker == null)
                {
                    var go = new GameObject("Speaker_" + s.SourceId); go.transform.SetParent(_speakerWorld, false);
                    en.speaker = go.AddComponent<SpeakerView>(); en.speaker.Build(theme, _headLocked);
                }
                if (en.speaker != null)
                    en.speaker.Tick(new SpeakerInput
                    {
                        src = s, relDeg = rel, inView = Mathf.Abs(rel) <= theme.speakerViewHalfDeg, headPosL = headPosL,
                        hasBox = en.hasBox, boxSize = en.box, elevFallbackDeg = en.elevFallback, visible = visible, dt = dt,
                    });
                if (visible && s.LevelDb > primaryDb) { primary = en; primaryDb = s.LevelDb; }
                if (s.State == TrackState.Tracking && s.LevelDb > loudestDb) { loudest = en; loudestDb = s.LevelDb; }
            }
            foreach (var en in _unlocated)
                if (en.speaker != null)
                    en.speaker.Tick(new SpeakerInput { src = en.src, visible = false, dt = dt });
            // the arrow always points toward the noise: the loudest speaker, else the loudest tracked sound of any kind
            var target = primary ?? loudest;
            _speakerArrow.Tick(target != null ? UiMath.NormalizeDeg(target.src.BearingDeg.Value - headYaw) : (float?)null, dt);
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
            if (_hoverId == id) _hoverId = null;
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

        // ------------------------------------------------------------------ gaze selection
        bool GazeHit(Vector3 worldCenter, Vector2 half, out float score)
        {
            Vector3 local = Head.InverseTransformPoint(worldCenter);
            score = float.MaxValue;
            if (local.z < 0.15f) return false;
            float pad = 0.03f;
            float hax = Mathf.Atan2(half.x + pad, local.z), hay = Mathf.Atan2(half.y + pad, local.z);
            float dx = Mathf.Abs(Mathf.Atan2(local.x, local.z)), dy = Mathf.Abs(Mathf.Atan2(local.y, local.z));
            score = Mathf.Max(dx / hax, dy / hay);
            return score <= 1f;
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
            string id = _hoverId;
            if (UsingFake) { if (fake != null) fake.Select(id); }
            else if (input != null && sendExplicitSelect)
            {
                var head = Head;
                Vector3 o = _worldRoot.InverseTransformPoint(head.position);
                Vector3 d = _worldRoot.InverseTransformDirection(head.forward);
                input.Send(ProtoAction.Select, id, modality, o, d);
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
                if (pinching && !_pinchWas[i] && !select) { select = true; modality = "hand"; }
                _pinchWas[i] = pinching;
            }
            if (select) TrySelectGazed("head");
            if (OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.RTouch)) ToggleHudMode();
            if (UsingFake && fake != null)
            {
                if (OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.RTouch)) fake.Stop();
                if (OVRInput.GetDown(OVRInput.Button.Three, OVRInput.Controller.LTouch)) fake.Replay();
            }
        }

        void ScanHands()
        {
            if (Time.unscaledTime < _nextHandScan) return;
            _nextHandScan = Time.unscaledTime + 2f;
            var found = FindObjectsByType<OVRHand>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            if (found.Length != _hands.Length) { _hands = found; _pinchWas = new bool[found.Length]; } else _hands = found;
        }
    }
}
