using Cocktail.Protocol;
using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    public struct SpeakerInput
    {
        public Source src;
        /// <summary>bearing relative to the head (+ right), wrapped to (-180,180]</summary>
        public float relDeg;
        public bool inView;
        public Vector3 headPosL;
        public bool hasBox; public Vector2 boxSize; public float elevFallbackDeg;
        /// <summary>false = fade out and hide</summary>
        public bool visible;
        /// <summary>the wearer chose this speaker / is looking at it</summary>
        public bool selected, hovered;
        /// <summary>elevation of the speaker relative to where the head points (+ up), degrees</summary>
        public float relElevDeg;
        /// <summary>this speaker's own colour; alpha 0 = use the theme speaker colour</summary>
        public Color color;
        /// <summary>heard but not located: shown as a plain dot instead of an arrow</summary>
        public bool noDirection;
        /// <summary>the source's number and name, for the label tab</summary>
        public int number; public string name;
        public float dt;
    }

    /// <summary>
    /// The indicator for one source, in the Cocktail look. Pointed at (not chosen): a soft tint of the source colour over it.
    /// Chosen and in view: a thin rectangle in the source colour with an ink hairline and a label tab (number, name) inside the
    /// top-left corner; the tint and the rectangle cross-fade. Out of view: an arrow at the edge toward it. No direction: a plain dot.
    /// Sizes are the design's degree values; layout only.
    /// </summary>
    public class SpeakerView : MonoBehaviour
    {
        UiTheme t;
        Transform _dotRoot;
        readonly SpriteBox[] _dark = new SpriteBox[8], _bar = new SpriteBox[8];
        SpriteBox _dotDark, _dot, _ring, _fill, _plain, _plainDark, _tabBg;
        TextMeshPro _tabNum, _tabName;
        float _yaw, _elev, _w, _h, _fade, _dx, _dy, _ringD, _arrowAng;
        bool _arrowPlaced;
        float _selA, _tintA, _flash;
        bool _wasSelected;
        bool _placed, _sized, _dotPlaced;

        public string SourceId { get; private set; }
        public bool OutlineVisible { get; private set; }
        public bool DotVisible { get; private set; }
        public Vector2 OutlineSize => new Vector2(_w, _h);
        public Vector2 DotLocal => new Vector2(_dx, _dy);
        public float RingDiameter => _ringD;

        public void Build(UiTheme theme, Transform headLockedRoot)
        {
            t = theme;
            UiLayers.Base = UiLayers.World;
            _fill = new SpriteBox(transform, "HoverTint", UiAssets.White, 2);
            for (int i = 0; i < 8; i++) _dark[i] = new SpriteBox(transform, "OutlineDark" + i, UiAssets.White, 3);
            for (int i = 0; i < 8; i++) _bar[i] = new SpriteBox(transform, "Edge" + i, UiAssets.White, 4);
            _tabBg = new SpriteBox(transform, "TabFill", UiAssets.White, 5);
            _tabNum = UiText.Make(transform, "TabNumber", t, UiTheme.Deg(t.lookTabNumDeg, t.markerDistance), TextAlignmentOptions.Center, 0.1f, 0.05f, 7, true);
            _tabName = UiText.Make(transform, "TabName", t, UiTheme.Deg(t.lookTabNameDeg, t.markerDistance), TextAlignmentOptions.Center, 0.2f, 0.05f, 7);
            _tabName.fontStyle &= ~FontStyles.UpperCase;   // the design's names are lowercase

            _dotRoot = new GameObject("SpeakerDot").transform;
            _dotRoot.SetParent(headLockedRoot, false);
            UiLayers.Base = UiLayers.Arrows;
            _ring = new SpriteBox(_dotRoot, "Ring", UiAssets.DashedRing, 1);
            _dotDark = new SpriteBox(_dotRoot, "ArrowDark", UiAssets.Chevron, 2);
            _dot = new SpriteBox(_dotRoot, "Arrow", UiAssets.Chevron, 3);
            _plainDark = new SpriteBox(_dotRoot, "PlainDotDark", UiAssets.Disc, 2);
            _plain = new SpriteBox(_dotRoot, "PlainDot", UiAssets.Disc, 3);
            _plainDark.SetActive(false); _plain.SetActive(false);
            gameObject.name = "Speaker";
            Show(false, false);
        }

        void OnDestroy() { if (_dotRoot != null) UiObjects.Kill(_dotRoot.gameObject); }
        // the edge arrow lives under the head-locked root, so hide it with this object (e.g. when the HUD leaves speaker-only mode)
        void OnDisable() { if (_dotRoot != null) _dotRoot.gameObject.SetActive(false); }

        void Show(bool outline, bool dot)
        {
            OutlineVisible = outline; DotVisible = dot;
            if (!outline)
            {
                foreach (var b in _bar) b.SetActive(false);
                foreach (var b in _dark) b.SetActive(false);
                _fill.SetActive(false); _tabBg.SetActive(false);
                _tabNum.gameObject.SetActive(false); _tabName.gameObject.SetActive(false);
            }
            if (_dotRoot != null && _dotRoot.gameObject.activeSelf != dot) _dotRoot.gameObject.SetActive(dot);
        }

        public void Tick(in SpeakerInput m)
        {
            var s = m.src;
            SourceId = s.SourceId;
            float dt = Mathf.Max(0f, m.dt);
            _fade = UiObjects.Fade(_fade, m.visible ? 1f : 0f, m.visible ? t.tAppear : t.tDisappear, dt);
            if (_fade <= 0f && !m.visible) { _selA = 0f; _tintA = 0f; _wasSelected = false; Show(false, false); return; }
            if (m.noDirection) { Show(false, true); TickPlainDot(m); return; }

            float yawT = s.BearingDeg ?? _yaw, elevT = s.ElevationDeg ?? m.elevFallbackDeg;
            if (!_placed) { _yaw = yawT; _elev = elevT; _placed = true; }
            else
            {
                float k = 1f - Mathf.Exp(-t.positionSmoothing * dt);
                _yaw = Mathf.LerpAngle(_yaw, yawT, k); _elev = Mathf.Lerp(_elev, elevT, k);
            }
            Show(m.inView, !m.inView);
            if (m.inView) TickOutline(m, dt); else TickDot(m, dt);
        }

        void TickOutline(in SpeakerInput m, float dt)
        {
            float D = t.markerDistance;
            Vector3 pos = m.headPosL + UiMath.DirFromYawElev(_yaw, _elev) * D;
            transform.localPosition = pos;
            Vector3 flat = pos - m.headPosL; flat.y = 0f;
            if (flat.sqrMagnitude > 1e-6f) transform.localRotation = Quaternion.LookRotation(flat.normalized, Vector3.up);

            float pad = UiTheme.Deg(t.lookBoxPadDeg, D);
            float tw = (m.hasBox ? m.boxSize.x : UiMath.AngleToSize(t.speakerBoxDeg.x, D)) + 2f * pad;
            float th = (m.hasBox ? m.boxSize.y : UiMath.AngleToSize(t.speakerBoxDeg.y, D)) + 2f * pad;
            if (!_sized) { _w = tw; _h = th; _sized = true; }
            else { _w = UiObjects.Ease(_w, tw, t.positionSmoothing, dt); _h = UiObjects.Ease(_h, th, t.positionSmoothing, dt); }

            Color own = m.color.a > 0f ? m.color : t.lookSources[0];
            own.a = 1f;
            float thick = UiTheme.Deg(t.lookBoxLineDeg, D), e = UiTheme.Deg(t.lookHairlineDeg, D);
            Color ink = t.lookInk;

            // tint (pointed) and the rectangle (chosen) cross-fade
            bool tint = m.hovered && !m.selected;
            _selA = UiObjects.Fade(_selA, m.selected ? 1f : 0f, t.tCross, dt);
            _tintA = UiObjects.Fade(_tintA, tint ? 1f : 0f, tint ? t.tAppear : t.tDisappear, dt);
            if (m.selected && !_wasSelected) _flash = 1f;   // just chosen: the tab flashes
            _wasSelected = m.selected;
            _flash = Mathf.Max(0f, _flash - dt / Mathf.Max(0.05f, t.tFlash));

            // the box: a full thin rectangle, or four corner brackets
            int used;
            if (t.lookFullBox)
            {
                used = 4;
                Place(0, 0f, _h * 0.5f - thick * 0.5f, _w, thick, e, own, ink);
                Place(1, 0f, -_h * 0.5f + thick * 0.5f, _w, thick, e, own, ink);
                Place(2, -_w * 0.5f + thick * 0.5f, 0f, thick, _h - 2f * thick, e, own, ink);
                Place(3, _w * 0.5f - thick * 0.5f, 0f, thick, _h - 2f * thick, e, own, ink);
            }
            else
            {
                used = 8;
                float arm = Mathf.Clamp(Mathf.Min(_w, _h) * t.speakerBracketArm, UiTheme.Deg(0.6f, D), UiTheme.Deg(2.0f, D));
                int i = 0;
                for (int sx = -1; sx <= 1; sx += 2)
                    for (int sy = -1; sy <= 1; sy += 2)
                    {
                        Place(i++, sx * (_w * 0.5f - arm * 0.5f), sy * (_h * 0.5f - thick * 0.5f), arm, thick, e, own, ink);
                        Place(i++, sx * (_w * 0.5f - thick * 0.5f), sy * (_h * 0.5f - arm * 0.5f), thick, arm, e, own, ink);
                    }
            }
            bool boxOn = _selA > 0f;
            for (int k = 0; k < 8; k++)
            {
                bool on = boxOn && k < used;
                _bar[k].SetActive(on); _dark[k].SetActive(on);
                if (on) { _bar[k].SetAlpha(_fade * _selA * t.lookBoxLine); _dark[k].SetAlpha(_fade * _selA * t.lookHairline); }
            }

            // the label tab, inside the top-left corner of the box: number then name
            _tabBg.SetActive(boxOn); _tabNum.gameObject.SetActive(boxOn); _tabName.gameObject.SetActive(boxOn);
            if (boxOn)
            {
                float tabH = UiTheme.Deg(t.lookTabHeightDeg, D), tpad = UiTheme.Deg(t.lookTabPadDeg, D), gap = UiTheme.Deg(t.lookTabGapDeg, D);
                string num = m.number > 0 ? m.number.ToString() : "", nm = m.name ?? "";
                float numSize = UiTheme.Deg(t.lookTabNumDeg, D), nameSize = UiTheme.Deg(t.lookTabNameDeg, D);
                float numW = num.Length == 0 ? 0f : num.Length * numSize * 1.0f;
                float nameW = nm.Length * nameSize * 0.52f;
                float tabW = Mathf.Min(_w - 2f * thick, tpad + numW + (numW > 0f && nameW > 0f ? gap : 0f) + nameW + tpad);
                float left = -_w * 0.5f + thick, top = _h * 0.5f - thick;
                _tabBg.SetSize(tabW, tabH); _tabBg.SetLocal(left + tabW * 0.5f, top - tabH * 0.5f);
                float f2 = _flash * _flash;   // just chosen: white fill with source-colour text, easing back to normal
                _tabBg.SetColor(Color.Lerp(own, t.lookPaper, f2));
                _tabBg.SetAlpha(_fade * _selA * t.lookTab);
                Color txt = Color.Lerp(ink, own, f2); txt.a = _fade * _selA;
                float cy = top - tabH * 0.5f;
                UiText.SetBox(_tabNum, t, Mathf.Max(0.01f, numW), tabH); UiText.Set(_tabNum, num);
                _tabNum.transform.localPosition = new Vector3(left + tpad + numW * 0.5f, cy, 0f); _tabNum.color = txt;
                UiText.SetBox(_tabName, t, Mathf.Max(0.01f, nameW), tabH); UiText.Set(_tabName, nm);
                _tabName.transform.localPosition = new Vector3(left + tpad + numW + (numW > 0f ? gap : 0f) + nameW * 0.5f, cy, 0f); _tabName.color = txt;
            }

            _fill.SetActive(_tintA > 0f);
            if (_tintA > 0f) { _fill.SetSize(_w, _h); _fill.SetLocal(0f, 0f); _fill.SetColor(own); _fill.SetAlpha(_fade * _tintA * t.lookTint); }
        }

        void Place(int i, float x, float y, float w, float h, float e, Color col, Color dark)
        {
            _dark[i].SetSize(w + 2f * e, h + 2f * e); _dark[i].SetLocal(x, y); _dark[i].SetColor(dark);
            _bar[i].SetSize(w, h); _bar[i].SetLocal(x, y); _bar[i].SetColor(col);
        }

        // a sound with no known direction: one plain dot low in the view, no arrow, no ring
        void TickPlainDot(in SpeakerInput m)
        {
            float hy = Mathf.Tan(t.edgeHalfHeightDeg * Mathf.Deg2Rad) * t.edgeDistance;
            _dotRoot.localPosition = new Vector3(0f, -hy * 0.85f, t.edgeDistance);
            float d = t.speakerDotSize * 1.8f, e = t.speakerOutlineExtra;
            Color col = m.color.a > 0f ? m.color : t.lookSources[0];
            _ring.SetActive(false); _dot.SetActive(false); _dotDark.SetActive(false);
            _plainDark.SetActive(true); _plain.SetActive(true);
            _plainDark.SetSize(d + 2f * e, d + 2f * e); _plainDark.SetColor(t.lookInk); _plainDark.SetAlpha(_fade * t.lookHairline);
            _plain.SetSize(d, d); _plain.SetColor(col); _plain.SetAlpha(_fade * t.lookBoxLine);
        }

        void TickDot(in SpeakerInput m, float dt)
        {
            _plain.SetActive(false); _plainDark.SetActive(false); _dot.SetActive(true); _dotDark.SetActive(true);
            float rel = m.relDeg, sign = rel >= 0f ? 1f : -1f;
            float hx = Mathf.Tan(t.edgeHalfWidthDeg * Mathf.Deg2Rad) * t.edgeDistance;
            float hy = Mathf.Tan(t.edgeHalfHeightDeg * Mathf.Deg2Rad) * t.edgeDistance;
            float tx = Mathf.Abs(rel) < 89f ? Mathf.Clamp(Mathf.Tan(rel * Mathf.Deg2Rad) * t.edgeDistance, -hx, hx) : sign * hx;
            // vertical placement follows where the speaker is relative to where you are looking, so looking above them puts the arrow at the bottom edge
            float ty = Mathf.Clamp(m.relElevDeg / 45f, -1f, 1f) * hy;
            if (!_dotPlaced) { _dx = tx; _dy = ty; _dotPlaced = true; }
            else { _dx = UiObjects.Ease(_dx, tx, t.positionSmoothing, dt); _dy = UiObjects.Ease(_dy, ty, t.positionSmoothing, dt); }
            _dotRoot.localPosition = new Vector3(_dx, _dy, t.edgeDistance);

            Color acol = m.color.a > 0f ? m.color : t.lookSources[0];
            _ring.SetActive(false);   // no outline for speakers you cannot see: just the arrow
            // an arrow at the screen edge that points toward the speaker (it is not in view)
            float d = t.speakerDotSize * 1.8f * (m.hovered ? 1.2f : 1f), e = t.speakerOutlineExtra;
            // 0 = up, clockwise positive: toward the speaker's direction in the view, not toward the clamped edge position
            float angT = -Mathf.Atan2(Mathf.Sin(rel * Mathf.Deg2Rad), Mathf.Sin(m.relElevDeg * Mathf.Deg2Rad)) * Mathf.Rad2Deg;
            _arrowAng = _arrowPlaced ? Mathf.LerpAngle(_arrowAng, angT, 1f - Mathf.Exp(-t.positionSmoothing * dt)) : angT;
            _arrowPlaced = true;
            var turn = Quaternion.Euler(0f, 0f, _arrowAng);
            _dotDark.transform.localRotation = turn; _dot.transform.localRotation = turn;
            _dotDark.SetSize(d + 2f * e, d + 2f * e); _dotDark.SetColor(t.lookInk); _dotDark.SetAlpha(_fade * t.lookHairline);
            _dot.SetSize(d, d); _dot.SetColor(acol); _dot.SetAlpha(_fade * t.lookBoxLine);
        }
    }
}
