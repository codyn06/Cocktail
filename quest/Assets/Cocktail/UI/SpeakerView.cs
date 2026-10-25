using Cocktail.Protocol;
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
        public float dt;
    }

    /// <summary>
    /// The speaker indicator for one source. In view: four corner brackets around the speaker (lives in the UI world root, same
    /// placement as <see cref="SourceMarkerView"/>). Out of view: a dot at the screen edge toward them with a dashed ring that grows
    /// with the uncertainty of the bearing (head-locked). Every shape has a thin dark outer line so it reads on a light wall.
    /// Layout only; colours and sizes come from <see cref="UiTheme"/>.
    /// </summary>
    public class SpeakerView : MonoBehaviour
    {
        UiTheme t;
        Transform _dotRoot;
        readonly SpriteBox[] _dark = new SpriteBox[8], _bar = new SpriteBox[8];
        SpriteBox _dotDark, _dot, _ring;
        float _yaw, _elev, _w, _h, _fade, _dx, _dy, _ringD;
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
            for (int i = 0; i < 8; i++) _dark[i] = new SpriteBox(transform, "OutlineDark" + i, UiAssets.White, 3);
            for (int i = 0; i < 8; i++) _bar[i] = new SpriteBox(transform, "Corner" + i, UiAssets.White, 4);

            _dotRoot = new GameObject("SpeakerDot").transform;
            _dotRoot.SetParent(headLockedRoot, false);
            UiLayers.Base = UiLayers.Arrows;
            _ring = new SpriteBox(_dotRoot, "Ring", UiAssets.DashedRing, 1);
            _dotDark = new SpriteBox(_dotRoot, "DotDark", UiAssets.Disc, 2);
            _dot = new SpriteBox(_dotRoot, "Dot", UiAssets.Disc, 3);
            gameObject.name = "Speaker";
            Show(false, false);
        }

        void OnDestroy() { if (_dotRoot != null) UiObjects.Kill(_dotRoot.gameObject); }

        void Show(bool outline, bool dot)
        {
            OutlineVisible = outline; DotVisible = dot;
            foreach (var b in _bar) b.SetActive(outline);
            foreach (var b in _dark) b.SetActive(outline);
            if (_dotRoot != null && _dotRoot.gameObject.activeSelf != dot) _dotRoot.gameObject.SetActive(dot);
        }

        public void Tick(in SpeakerInput m)
        {
            var s = m.src;
            SourceId = s.SourceId;
            float dt = Mathf.Max(0f, m.dt);
            _fade = Mathf.MoveTowards(_fade, m.visible ? 1f : 0f, dt / Mathf.Max(0.01f, t.fadeInSeconds));
            if (_fade <= 0f && !m.visible) { Show(false, false); return; }

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
            Vector3 pos = m.headPosL + UiMath.DirFromYawElev(_yaw, _elev) * t.markerDistance;
            transform.localPosition = pos;
            Vector3 flat = pos - m.headPosL; flat.y = 0f;
            if (flat.sqrMagnitude > 1e-6f) transform.localRotation = Quaternion.LookRotation(flat.normalized, Vector3.up);

            float tw = m.hasBox ? m.boxSize.x : UiMath.AngleToSize(t.speakerBoxDeg.x, t.markerDistance);
            float th = m.hasBox ? m.boxSize.y : UiMath.AngleToSize(t.speakerBoxDeg.y, t.markerDistance);
            if (!_sized) { _w = tw; _h = th; _sized = true; }
            else { _w = UiObjects.Ease(_w, tw, t.positionSmoothing, dt); _h = UiObjects.Ease(_h, th, t.positionSmoothing, dt); }

            float thick = t.speakerBracketThickness, arm = Mathf.Min(_w, _h) * t.speakerBracketArm, e = t.speakerOutlineExtra;
            Color col = t.speakerColor, dark = t.speakerOutlineColor;
            int i = 0;
            for (int sx = -1; sx <= 1; sx += 2)
                for (int sy = -1; sy <= 1; sy += 2)
                {
                    Place(i++, sx * (_w * 0.5f - arm * 0.5f), sy * (_h * 0.5f - thick * 0.5f), arm, thick, e, col, dark);
                    Place(i++, sx * (_w * 0.5f - thick * 0.5f), sy * (_h * 0.5f - arm * 0.5f), thick, arm, e, col, dark);
                }
        }

        void Place(int i, float x, float y, float w, float h, float e, Color col, Color dark)
        {
            _dark[i].SetSize(w + 2f * e, h + 2f * e); _dark[i].SetLocal(x, y); _dark[i].SetColor(dark); _dark[i].SetAlpha(_fade);
            _bar[i].SetSize(w, h); _bar[i].SetLocal(x, y); _bar[i].SetColor(col); _bar[i].SetAlpha(_fade);
        }

        void TickDot(in SpeakerInput m, float dt)
        {
            float rel = m.relDeg, sign = rel >= 0f ? 1f : -1f;
            float hx = Mathf.Tan(t.edgeHalfWidthDeg * Mathf.Deg2Rad) * t.edgeDistance;
            float hy = Mathf.Tan(t.edgeHalfHeightDeg * Mathf.Deg2Rad) * t.edgeDistance;
            float tx = Mathf.Abs(rel) < 89f ? Mathf.Clamp(Mathf.Tan(rel * Mathf.Deg2Rad) * t.edgeDistance, -hx, hx) : sign * hx;
            float ty = Mathf.Clamp(_elev / 45f, -1f, 1f) * hy * 0.6f;
            if (!_dotPlaced) { _dx = tx; _dy = ty; _dotPlaced = true; }
            else { _dx = UiObjects.Ease(_dx, tx, t.positionSmoothing, dt); _dy = UiObjects.Ease(_dy, ty, t.positionSmoothing, dt); }
            _dotRoot.localPosition = new Vector3(_dx, _dy, t.edgeDistance);

            float conf = Mathf.Clamp01(m.src.BearingConf);
            _ringD = UiObjects.Ease(_ringD <= 0f ? t.speakerRingMax : _ringD, Mathf.Lerp(t.speakerRingMax, t.speakerRingMin, conf), t.positionSmoothing, dt);
            _ring.SetSize(_ringD, _ringD); _ring.SetColor(t.speakerColor); _ring.SetAlpha(_fade * 0.9f);
            float d = t.speakerDotSize, e = t.speakerOutlineExtra;
            _dotDark.SetSize(d + 2f * e, d + 2f * e); _dotDark.SetColor(t.speakerOutlineColor); _dotDark.SetAlpha(_fade);
            _dot.SetSize(d, d); _dot.SetColor(t.speakerColor); _dot.SetAlpha(_fade);
        }
    }

    /// <summary>
    /// Head-locked arrow that always points toward the noise: up = straight ahead, right = to the right, down = behind.
    /// </summary>
    public class SpeakerArrowView : MonoBehaviour
    {
        UiTheme t;
        SpriteBox _dark, _arrow;
        float _ang, _fade;
        bool _placed;

        public bool Visible => _arrow != null && _arrow.go.activeSelf;
        /// <summary>current z rotation in degrees (0 = pointing up, -90 = pointing right)</summary>
        public float AngleDeg => _ang;

        public void Build(UiTheme theme)
        {
            t = theme;
            UiLayers.Base = UiLayers.Arrows;
            _dark = new SpriteBox(transform, "ArrowDark", UiAssets.Chevron, 1);
            _arrow = new SpriteBox(transform, "Arrow", UiAssets.Chevron, 2);
            gameObject.name = "SpeakerArrow";
            _dark.SetActive(false); _arrow.SetActive(false);
        }

        /// <param name="relDeg">bearing of the noise relative to the head (+ right); null hides the arrow</param>
        public void Tick(float? relDeg, float dt)
        {
            _fade = Mathf.MoveTowards(_fade, relDeg.HasValue ? 1f : 0f, dt / Mathf.Max(0.01f, t.fadeInSeconds));
            if (_fade <= 0f && !relDeg.HasValue) { _dark.SetActive(false); _arrow.SetActive(false); return; }
            if (relDeg.HasValue)
            {
                float target = -relDeg.Value;   // 0 = up; + bearing (right) turns the arrow clockwise
                if (!_placed) { _ang = target; _placed = true; }
                else _ang = Mathf.LerpAngle(_ang, target, 1f - Mathf.Exp(-t.positionSmoothing * Mathf.Max(0f, dt)));
            }
            transform.localPosition = t.speakerArrowOffset;
            transform.localRotation = Quaternion.Euler(0f, 0f, _ang);
            float e = t.speakerOutlineExtra * 2f, a = t.speakerArrowSize;
            _dark.SetActive(true); _arrow.SetActive(true);
            _dark.SetSize(a + e, a + e); _dark.SetColor(t.speakerOutlineColor); _dark.SetAlpha(_fade);
            _arrow.SetSize(a, a); _arrow.SetColor(t.speakerColor); _arrow.SetAlpha(_fade);
        }
    }
}
