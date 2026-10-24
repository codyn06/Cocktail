using Cocktail.Protocol;
using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// Secondary, small periphery indicator for a source that is not in view: a triangle pointing toward it and a compact data box with
    /// the label, clock position and bearing (12 o'clock = ahead, 3 = right, 6 = behind). Head-locked. Layout only; look from <see cref="UiTheme"/>.
    /// </summary>
    public class EdgeArrowView : MonoBehaviour
    {
        UiTheme t;
        SpriteBox _tri, _plate, _outline;
        TextMeshPro _title, _detail;
        int _sigRel = int.MinValue; string _sigLabel;
        float _x, _y; bool _placed;

        public string SourceId { get; private set; }
        public string TitleText => _title != null ? _title.text : "";
        public string DetailText => _detail != null ? _detail.text : "";

        public void Build(UiTheme theme)
        {
            t = theme;
            UiLayers.Base = UiLayers.Arrows;
            _plate = new SpriteBox(transform, "Fill", t.plateSprite != null ? t.plateSprite : UiAssets.Plate, 1);
            _outline = new SpriteBox(transform, "Outline", t.outlineSprite != null ? t.outlineSprite : (t.plateSprite != null ? t.plateSprite : UiAssets.Outline), 2);
            _tri = new SpriteBox(transform, "Arrow", t.arrowSprite != null ? t.arrowSprite : UiAssets.Triangle, 3);
            _title = UiText.Make(transform, "Title", t, t.arrowTitleEm, TextAlignmentOptions.Right, t.arrowBoxSize.x, t.arrowTitleEm * 1.4f, 5, true);
            _detail = UiText.Make(transform, "Detail", t, t.arrowDetailEm, TextAlignmentOptions.Right, t.arrowBoxSize.x, t.arrowDetailEm * 1.4f, 5);
            gameObject.name = "EdgeArrow";
        }

        /// <param name="relDeg">bearing relative to the head (+ right), wrapped to (-180,180]</param>
        public void Tick(Source s, float relDeg, float elevDeg, int stackIndex, float alpha, float dt)
        {
            SourceId = s.SourceId;
            float sign = relDeg >= 0f ? 1f : -1f;
            float hx = Mathf.Tan(t.edgeHalfWidthDeg * Mathf.Deg2Rad) * t.edgeDistance;
            float hy = Mathf.Tan(t.edgeHalfHeightDeg * Mathf.Deg2Rad) * t.edgeDistance;
            float tx = sign * hx;
            float ty = Mathf.Clamp(elevDeg / 45f, -1f, 1f) * hy * 0.6f - stackIndex * t.arrowStackSpacing;
            if (!_placed) { _x = tx; _y = ty; _placed = true; }
            else { _x = UiObjects.Ease(_x, tx, t.positionSmoothing, dt); _y = UiObjects.Ease(_y, ty, t.positionSmoothing, dt); }
            transform.localPosition = new Vector3(_x, _y, t.edgeDistance);

            // apex points outward (toward the source), tilted by elevation, a bit down when it is behind
            float behind = Mathf.Abs(relDeg) > 120f ? -20f : 0f;
            float ang = (sign > 0 ? 0f : 180f) + sign * (Mathf.Clamp(elevDeg, -45f, 45f) * 0.4f + behind);
            _tri.SetSize(t.arrowSize, t.arrowSize);
            _tri.SetLocal(0f, 0f);
            _tri.transform.localRotation = Quaternion.Euler(0f, 0f, ang - 90f);
            Color accent = s.State == TrackState.Tracking ? t.LabelColor(s.Label) : t.lostColor;
            accent.a *= alpha; _tri.SetColor(accent);

            float bw = t.arrowBoxSize.x, bh = t.arrowBoxSize.y;
            float plateCx = -sign * (t.arrowSize * 0.5f + t.arrowTextGap + bw * 0.5f);
            Color pc = t.plateColor; pc.a *= alpha;
            _plate.SetSize(bw, bh); _plate.SetLocal(plateCx, 0f); _plate.SetColor(pc);
            _outline.SetSize(bw, bh); _outline.SetLocal(plateCx, 0f); _outline.SetColor(t.outlineColor); _outline.SetAlpha(alpha);

            var al = sign > 0 ? TextAlignmentOptions.Right : TextAlignmentOptions.Left;
            float textW = bw - t.platePadding * 1.6f;
            Place(_title, plateCx, bh * 0.2f, textW, al, t.arrowTitleEm);
            Place(_detail, plateCx, -bh * 0.2f, textW, al, t.arrowDetailEm);

            int rel = Mathf.RoundToInt(relDeg);
            if (rel != _sigRel || _sigLabel != s.Label)
            {
                _sigRel = rel; _sigLabel = s.Label;
                UiText.Set(_title, UiMath.Fmt(t.arrowTitle, "{label}", t.DisplayLabel(s.Label), "{id}", s.SourceId));
                UiText.Set(_detail, UiMath.Fmt(t.arrowDetail, "{clock}", UiMath.ClockPosition(relDeg).ToString(), "{bearing}", UiMath.BearingText(relDeg, t)));
            }
            var tc = t.textLight; tc.a *= alpha; _title.color = tc;
            var dc = t.TextSecondary; dc.a *= alpha; _detail.color = dc;
        }

        void Place(TextMeshPro tmp, float cx, float y, float w, TextAlignmentOptions al, float em)
        {
            tmp.alignment = al;
            UiText.SetBox(tmp, t, w, em * 1.4f);
            tmp.transform.localPosition = new Vector3(cx, y, 0f);
        }
    }
}
