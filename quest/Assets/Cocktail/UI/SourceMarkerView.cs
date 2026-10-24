using Cocktail.Protocol;
using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>Per-frame input for one marker (filled by CocktailUI).</summary>
    public struct MarkerInput
    {
        public Source src;
        public float ageSeconds;          // device_ts - last_observed
        public bool selected, hovered, pending;
        public float dwell01;
        public bool inView;
        public bool hasBox; public Vector2 boxSize;   // metres at marker distance (from bbox), if any
        public float elevFallbackDeg;
        public float relDeg;              // bearing relative to the head (+ right), used to put the data box on the inner side
        public Vector3 headPosL;          // head position in the UI world root space
        public float stackY;
        public float dt;
        public bool isChip;               // chip = placed by the parent (head-locked), always shows its data box
    }

    /// <summary>
    /// One source in the world. Unselected: a small dim bracket (in view) or dot (out of view) with a short label plate.
    /// Selected or gazed: the corner brackets snap out to the full source box (lock-on), a thin callout line leads to a
    /// chamfered data box (label + track code, bearing + confidence, level readout + meter). Layout only: every colour, size,
    /// width, shape and text format comes from <see cref="UiTheme"/>.
    /// Lives in the UI world root space (tracking space): the object sits at head + direction(bearing) * markerDistance.
    /// </summary>
    public class SourceMarkerView : MonoBehaviour
    {
        UiTheme t;
        bool _chip;
        Transform _boxRoot, _dataRoot;
        readonly SpriteBox[] _bars = new SpriteBox[8];
        readonly SpriteBox[] _ticks = new SpriteBox[4];
        SpriteBox _dot, _callout, _node;
        SpriteBox _lblPlate, _lblOutline;
        SpriteBox _glow, _plate, _outline, _rule, _barBg, _barFill, _dwell;
        TextMeshPro _lbl, _title, _conf, _brg, _status, _level, _tag;

        public string SourceId { get; private set; }
        public bool BoxVisible { get; private set; }
        public bool IsExpanded { get; private set; }
        public Vector3 PlateCenterWorld { get; private set; }
        public Vector2 PlateHalf { get; private set; }
        public Vector3 BoxCenterWorld => _boxRoot.position;
        public Vector2 BoxHalf { get; private set; }
        public string TitleText => _title != null ? _title.text : "";
        public string LabelText => _lbl != null ? _lbl.text : "";
        public string StatusText => _status != null ? _status.text : "";
        public TrackState State { get; private set; }

        float _yaw, _elev, _stack, _fade, _expandA, _bw, _bh, _dataX, _side = 1f;
        bool _placed, _wasExpanded, _lastBox;
        int _pink;
        TrackState _lastState = TrackState.Tracking;
        int _sigConf = int.MinValue, _sigBrg = int.MinValue, _sigDb = int.MinValue, _sigAge = int.MinValue; TrackState _sigState; bool _sigHasBrg, _sigInit; string _sigLabel;
        Vector2 _box = new Vector2(0.4f, 0.5f);

        public void Build(UiTheme theme, bool chip)
        {
            t = theme; _chip = chip;
            UiLayers.Base = chip ? UiLayers.Chips : UiLayers.World;
            var plateSprite = t.plateSprite != null ? t.plateSprite : UiAssets.Plate;
            var outlineSprite = t.outlineSprite != null ? t.outlineSprite : (t.plateSprite != null ? t.plateSprite : UiAssets.Outline);
            var barSprite = t.barSprite != null ? t.barSprite : UiAssets.White;
            var bracketSprite = t.bracketSprite != null ? t.bracketSprite : UiAssets.White;

            _boxRoot = new GameObject("Box").transform; _boxRoot.SetParent(transform, false);
            _dataRoot = new GameObject("Data").transform; _dataRoot.SetParent(transform, false);

            for (int i = 0; i < _bars.Length; i++) _bars[i] = new SpriteBox(_boxRoot, "Corner" + i, bracketSprite, 4);
            for (int i = 0; i < _ticks.Length; i++) _ticks[i] = new SpriteBox(_boxRoot, "Tick" + i, bracketSprite, 4);
            _dot = new SpriteBox(transform, "Dot", UiAssets.Diamond, 4);
            _callout = new SpriteBox(transform, "Callout", barSprite, 3);
            _node = new SpriteBox(transform, "Node", UiAssets.Diamond, 4);

            _lblPlate = new SpriteBox(transform, "LabelFill", plateSprite, 1);
            _lblOutline = new SpriteBox(transform, "LabelOutline", outlineSprite, 2);
            _lbl = UiText.Make(transform, "Label", t, t.labelEm, TextAlignmentOptions.Center, t.labelPlateSize.x, t.labelEm * 1.4f, 5, true);

            _glow = new SpriteBox(_dataRoot, "Glow", UiAssets.Glow, 0);
            _plate = new SpriteBox(_dataRoot, "Fill", plateSprite, 1);
            _outline = new SpriteBox(_dataRoot, "Outline", outlineSprite, 2);
            _rule = new SpriteBox(_dataRoot, "Rule", barSprite, 3);
            _barBg = new SpriteBox(_dataRoot, "BarBg", barSprite, 3);
            _barFill = new SpriteBox(_dataRoot, "BarFill", barSprite, 4);
            _dwell = new SpriteBox(_dataRoot, "Dwell", barSprite, 4);

            float pw = t.plateSize.x, ph = t.plateSize.y;
            float left = -pw * 0.5f + t.platePadding + t.accentRuleWidth + 0.012f;
            float right = pw * 0.5f - t.platePadding;
            float inner = right - left;
            float rowTitle = ph * 0.5f - t.platePadding - t.titleEm * 0.65f;
            float rowData = rowTitle - t.titleEm * 0.65f - t.dataEm * 0.8f;
            float rowLevel = rowData - t.dataEm * 1.35f;
            _title = Row("Title", t.titleEm, TextAlignmentOptions.Left, left, rowTitle, inner * 0.62f, true);
            _status = Row("Status", t.dataEm, TextAlignmentOptions.Right, right - inner * 0.38f, rowTitle, inner * 0.38f, false);
            _brg = Row("Bearing", t.dataEm, TextAlignmentOptions.Left, left, rowData, inner * 0.6f, false);
            _conf = Row("Conf", t.dataEm, TextAlignmentOptions.Right, right - inner * 0.4f, rowData, inner * 0.4f, false);
            _level = Row("Level", t.dataEm, TextAlignmentOptions.Left, left, rowLevel, inner, false);
            _tag = UiText.Make(transform, "Tag", t, t.dataEm, TextAlignmentOptions.Center, 0.6f, t.dataEm * 1.4f, 5, true);

            _plate.SetSize(pw, ph); _outline.SetSize(pw, ph);
            _rule.SetSize(t.accentRuleWidth, ph - 0.09f); _rule.SetLocal(-pw * 0.5f + t.accentRuleWidth * 0.5f + 0.02f, -0.012f);
            float barY = -ph * 0.5f + t.platePadding + t.levelBarHeight * 0.5f;
            _barBg.SetSize(inner, t.levelBarHeight); _barBg.SetLocal(left + inner * 0.5f, barY);
            _lblPlate.SetSize(t.labelPlateSize.x, t.labelPlateSize.y); _lblOutline.SetSize(t.labelPlateSize.x, t.labelPlateSize.y);
            _bw = _bh = t.smallBoxSize;
            gameObject.name = "Marker";
        }

        TextMeshPro Row(string name, float em, TextAlignmentOptions al, float x, float y, float w, bool title)
        {
            var tmp = UiText.Make(_dataRoot, name, t, em, al, w, em * 1.4f, 5, title);
            tmp.transform.localPosition = new Vector3(x + w * 0.5f, y, 0f);
            return tmp;
        }

        public void Tick(in MarkerInput m)
        {
            var s = m.src;
            SourceId = s.SourceId;
            State = s.State;
            float dt = Mathf.Max(0f, m.dt);
            float pw = t.plateSize.x, ph = t.plateSize.y;

            // ---- signal-lost flicker: the colour flashes to the flicker colour for a couple of frames
            if (_placed && s.State == TrackState.Lost && _lastState != TrackState.Lost) _pink = Mathf.Max(0, t.flickerFrames);
            _lastState = s.State;
            bool pinkNow = _pink > 0; if (_pink > 0) _pink--;

            _fade = t.fadeInSeconds <= 0f ? 1f : Mathf.Min(1f, _fade + dt / t.fadeInSeconds);
            float stateAlpha = s.State == TrackState.Tracking ? t.alphaTracking : (s.State == TrackState.LastSeen ? t.alphaLastSeen : t.alphaLost);
            float alpha = stateAlpha * _fade;

            // ---- placement (world marker)
            float yawT = s.BearingDeg ?? _yaw;
            float elevT = s.ElevationDeg ?? m.elevFallbackDeg;
            if (!_placed) { _yaw = yawT; _elev = elevT; _stack = m.stackY; _placed = true; }
            else
            {
                float k = 1f - Mathf.Exp(-t.positionSmoothing * dt);
                _yaw = Mathf.LerpAngle(_yaw, yawT, k); _elev = Mathf.Lerp(_elev, elevT, k); _stack = Mathf.Lerp(_stack, m.stackY, k);
            }
            if (!m.isChip)
            {
                Vector3 pos = m.headPosL + UiMath.DirFromYawElev(_yaw, _elev) * t.markerDistance + Vector3.up * _stack;
                transform.localPosition = pos;
                Vector3 flat = pos - m.headPosL; flat.y = 0f;
                if (flat.sqrMagnitude > 1e-6f) transform.localRotation = Quaternion.LookRotation(flat.normalized, Vector3.up);
            }

            UpdateTexts(m, s);

            // ---- state of the marker
            bool active = m.selected || m.hovered || m.pending;
            bool expanded = m.isChip || active;
            bool box = m.inView && !m.isChip && s.State != TrackState.Lost;
            IsExpanded = expanded; BoxVisible = box;
            _expandA = UiObjects.Ease(_expandA, expanded ? 1f : 0f, t.easing, dt);
            if (m.relDeg > 6f) _side = -1f; else if (m.relDeg < -6f) _side = 1f;   // data box goes on the inner side (toward the view centre)

            Color baseCol = s.State == TrackState.Tracking ? t.LabelColor(s.Label) : t.lostColor;
            Color col = pinkNow ? t.lostFlickerColor : (active ? t.selectedColor : baseCol);
            float colA = pinkNow ? 1f : alpha;

            // ---- brackets / dot
            _box = m.hasBox ? new Vector2(Mathf.Clamp(m.boxSize.x, t.minBoxSize.x, t.maxBoxSize.x), Mathf.Clamp(m.boxSize.y, t.minBoxSize.y, t.maxBoxSize.y)) : t.defaultBoxSize;
            Vector2 target = expanded ? _box : new Vector2(t.smallBoxSize, t.smallBoxSize);
            if (box && expanded && !_wasExpanded) { _bw = _box.x * t.lockOnScale; _bh = _box.y * t.lockOnScale; }   // snap in from larger
            _wasExpanded = expanded && box;
            _bw = UiObjects.Ease(_bw, target.x, t.easing, dt); _bh = UiObjects.Ease(_bh, target.y, t.easing, dt);
            float bracketA = colA * (active ? 1f : t.dimAlpha);

            _boxRoot.gameObject.SetActive(box);
            float halfX, halfY;
            if (box)
            {
                float w = _bw, h = _bh;
                float th = m.selected ? t.bracketThicknessSelected : t.bracketThickness;
                float arm = Mathf.Min(w, h) * t.bracketArm;
                int i = 0;
                for (int sx = -1; sx <= 1; sx += 2)
                    for (int sy = -1; sy <= 1; sy += 2)
                    {
                        var hb = _bars[i++]; hb.SetSize(arm, th); hb.SetLocal(sx * (w * 0.5f - arm * 0.5f), sy * (h * 0.5f - th * 0.5f)); hb.SetColor(col); hb.SetAlpha(bracketA); hb.SetActive(true);
                        var vb = _bars[i++]; vb.SetSize(th, arm); vb.SetLocal(sx * (w * 0.5f - th * 0.5f), sy * (h * 0.5f - arm * 0.5f)); vb.SetColor(col); vb.SetAlpha(bracketA); vb.SetActive(true);
                    }
                bool ticksOn = expanded;
                float tl = t.tickLength, tw = t.tickThickness;
                SetTick(0, 0f, h * 0.5f - tl * 0.5f, tw, tl, col, bracketA, ticksOn);
                SetTick(1, 0f, -h * 0.5f + tl * 0.5f, tw, tl, col, bracketA, ticksOn);
                SetTick(2, -w * 0.5f + tl * 0.5f, 0f, tl, tw, col, bracketA, ticksOn);
                SetTick(3, w * 0.5f - tl * 0.5f, 0f, tl, tw, col, bracketA, ticksOn);
                halfX = w * 0.5f; halfY = h * 0.5f;
                BoxHalf = new Vector2(Mathf.Max(0.08f, target.x * 0.5f), Mathf.Max(0.08f, target.y * 0.5f));
            }
            else { halfX = halfY = t.dotSize * 0.5f; BoxHalf = Vector2.zero; }
            _dot.SetActive(!box && !m.isChip);
            if (!box && !m.isChip) { _dot.SetSize(t.dotSize, t.dotSize); _dot.SetLocal(0f, 0f); _dot.SetColor(col); _dot.SetAlpha(bracketA); }

            // ---- short label (unselected)
            bool lblOn = !expanded && !m.isChip;
            _lblPlate.SetActive(lblOn); _lblOutline.SetActive(lblOn); _lbl.gameObject.SetActive(lblOn);
            if (lblOn)
            {
                float ly = -(halfY + t.labelGap + t.labelPlateSize.y * 0.5f);
                _lblPlate.SetLocal(0f, ly); _lblOutline.SetLocal(0f, ly); _lbl.transform.localPosition = new Vector3(0f, ly, 0f);
                var pc = t.plateColor; pc.a *= alpha; _lblPlate.SetColor(pc);
                _lblOutline.SetColor(t.outlineColor); _lblOutline.SetAlpha(alpha);
                var lc = t.textLight; lc.a *= alpha * Mathf.Lerp(1f, 0.85f, 0f); _lbl.color = lc;
                PlateCenterWorld = _lblPlate.transform.position; PlateHalf = t.labelPlateSize * 0.5f;
            }

            // ---- data box
            bool dataOn = _expandA > 0.02f;
            _dataRoot.gameObject.SetActive(dataOn);
            float dataXT = m.isChip ? 0f : _side * (halfX + t.calloutLength + pw * 0.5f);
            _dataX = _dataX == 0f ? dataXT : UiObjects.Ease(_dataX, dataXT, t.easing, dt);
            _dataRoot.localPosition = new Vector3(_dataX, 0f, 0f);
            if (dataOn)
            {
                float da = alpha * _expandA;
                Color plateC = t.plateColor; plateC.a *= da; _plate.SetColor(plateC);
                _outline.SetColor(active ? t.selectedColor : t.outlineColor); _outline.SetAlpha(da);
                bool glow = m.selected;
                _glow.SetActive(glow);
                if (glow) { _glow.SetSize(pw + 2f * t.glowSpread, ph + 2f * t.glowSpread); _glow.SetColor(t.selectedColor); _glow.SetAlpha(t.glowAlpha * da); }
                _rule.SetColor(col); _rule.SetAlpha(colA * _expandA);

                float left = -pw * 0.5f + t.platePadding + t.accentRuleWidth + 0.012f;
                float inner = pw * 0.5f - t.platePadding - left;
                float lvl = UiMath.Level01(s.LevelDb, t);
                float barY = -ph * 0.5f + t.platePadding + t.levelBarHeight * 0.5f;
                _barBg.SetColor(t.BarBackground); _barBg.SetAlpha(da);
                _barFill.SetSize(Mathf.Max(0.0005f, inner * lvl), t.levelBarHeight); _barFill.SetLocal(left + inner * lvl * 0.5f, barY);
                _barFill.SetColor(col); _barFill.SetAlpha(da);
                bool dwellOn = m.hovered && m.dwell01 > 0.01f && !m.selected;
                _dwell.SetActive(dwellOn);
                if (dwellOn)
                {
                    float dw = pw * m.dwell01;
                    _dwell.SetSize(dw, t.levelBarHeight * 0.5f); _dwell.SetLocal(-pw * 0.5f + dw * 0.5f, -ph * 0.5f - t.levelBarHeight * 0.5f - 0.006f);
                    _dwell.SetColor(t.selectedColor); _dwell.SetAlpha(da);
                }
                var txt = t.textLight; txt.a *= da; var sec = t.TextSecondary; sec.a *= da;
                _title.color = txt; _status.color = txt; _brg.color = sec; _conf.color = sec; _level.color = sec;
                PlateCenterWorld = _plate.transform.position; PlateHalf = new Vector2(pw, ph) * 0.5f;
            }

            // ---- callout line + node (expanded, with a box or dot)
            bool calloutOn = expanded && !m.isChip && dataOn;
            _callout.SetActive(calloutOn); _node.SetActive(calloutOn);
            if (calloutOn)
            {
                float len = t.calloutLength;
                _callout.SetSize(len, t.calloutWidth); _callout.SetLocal(_side * (halfX + len * 0.5f), 0f); _callout.SetColor(col); _callout.SetAlpha(colA * _expandA);
                _node.SetSize(0.03f, 0.03f); _node.SetLocal(_side * halfX, 0f); _node.SetColor(col); _node.SetAlpha(colA * _expandA);
            }

            // ---- lock tag above the shape
            bool tagOn = m.selected || m.pending;
            _tag.gameObject.SetActive(tagOn);
            if (tagOn)
            {
                UiText.Set(_tag, t.selectedTag);
                _tag.transform.localPosition = new Vector3(0f, halfY + t.dataEm * 0.9f, 0f);
                var tc = t.selectedColor; tc.a *= alpha; _tag.color = tc;
            }
        }

        void SetTick(int i, float x, float y, float w, float h, Color c, float a, bool on)
        {
            var b = _ticks[i]; b.SetActive(on);
            if (!on) return;
            b.SetSize(w, h); b.SetLocal(x, y); b.SetColor(c); b.SetAlpha(a * 0.8f);
        }

        void UpdateTexts(in MarkerInput m, Source s)
        {
            int conf = Mathf.RoundToInt(Mathf.Clamp01(s.LabelConf) * 100f);
            bool hasB = s.BearingDeg.HasValue;
            int brg = hasB ? Mathf.RoundToInt(UiMath.NormalizeDeg(s.BearingDeg.Value)) : 0;
            int db = Mathf.RoundToInt(s.LevelDb);
            int age = Mathf.RoundToInt(m.ageSeconds);
            if (_sigInit && conf == _sigConf && brg == _sigBrg && db == _sigDb && age == _sigAge && s.State == _sigState && hasB == _sigHasBrg && _sigLabel == s.Label) return;
            _sigInit = true; _sigConf = conf; _sigBrg = brg; _sigDb = db; _sigAge = age; _sigState = s.State; _sigHasBrg = hasB; _sigLabel = s.Label;

            string label = UiMath.Fmt(t.markerTitle, "{label}", t.DisplayLabel(s.Label), "{conf}", conf.ToString(), "{id}", s.SourceId);
            UiText.Set(_title, label); UiText.Set(_lbl, label);
            UiText.Set(_conf, UiText.Mono(t, UiMath.Fmt(t.markerConf, "{conf}", conf.ToString())));
            string brgS = hasB ? brg.ToString("+0;-0;0") : "---";
            UiText.Set(_brg, UiText.Mono(t, UiMath.Fmt(t.markerBearing, "{brg}", brgS)));
            UiText.Set(_level, UiText.Mono(t, UiMath.Fmt(t.markerLevel, "{db}", db.ToString())));
            string st;
            switch (s.State)
            {
                case TrackState.LastSeen: st = UiMath.Fmt(t.statusLastSeen, "{age}", age.ToString()); break;
                case TrackState.Lost: st = t.statusLost; break;
                default: st = hasB ? t.statusTracking : t.statusUnlocated; break;
            }
            UiText.Set(_status, UiText.Mono(t, st));
        }
    }
}
