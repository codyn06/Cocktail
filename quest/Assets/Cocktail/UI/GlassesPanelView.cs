using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// Glasses-style minimal HUD: one small plate with a direction arrow (compass style: up = ahead, right = to your right)
    /// and the selected target's label plus state. Shown instead of the full mixed-reality UI when the HUD mode is toggled.
    /// </summary>
    public class GlassesPanelView : MonoBehaviour
    {
        UiTheme t;
        SpriteBox _plate, _outline, _rule, _arrow, _dot;
        TextMeshPro _title, _detail;
        public string TitleText => _title != null ? _title.text : "";
        public string DetailText => _detail != null ? _detail.text : "";

        public void Build(UiTheme theme)
        {
            t = theme;
            var sz = t.glassesSize;
            UiLayers.Base = UiLayers.Banner;
            _plate = new SpriteBox(transform, "Fill", t.plateSprite != null ? t.plateSprite : UiAssets.Plate, 1);
            _outline = new SpriteBox(transform, "Outline", t.plateSprite != null ? t.plateSprite : UiAssets.Outline, 2);
            _rule = new SpriteBox(transform, "Rule", t.barSprite != null ? t.barSprite : UiAssets.White, 3);
            _arrow = new SpriteBox(transform, "Arrow", t.arrowSprite != null ? t.arrowSprite : UiAssets.Triangle, 3);
            _dot = new SpriteBox(transform, "Dot", UiAssets.Diamond, 3);
            float ax = -sz.x * 0.5f + t.glassesArrowSize * 0.5f + t.platePadding + 0.02f;
            float tl = ax + t.glassesArrowSize * 0.5f + 0.03f;
            float w = sz.x * 0.5f - t.platePadding - tl;
            _title = UiText.Make(transform, "Title", t, t.glassesTitleEm, TextAlignmentOptions.Left, w, t.glassesTitleEm * 1.4f, 5, true);
            _detail = UiText.Make(transform, "Detail", t, t.glassesDetailEm, TextAlignmentOptions.Left, w, t.glassesDetailEm * 1.4f, 5);
            _title.transform.localPosition = new Vector3(tl + w * 0.5f, 0.03f, 0f);
            _detail.transform.localPosition = new Vector3(tl + w * 0.5f, -0.035f, 0f);
            _plate.SetSize(sz.x, sz.y); _outline.SetSize(sz.x, sz.y);
            _rule.SetSize(t.bannerRuleWidth, sz.y - 0.06f); _rule.SetLocal(-sz.x * 0.5f + 0.02f, 0f);
            _arrow.SetSize(t.glassesArrowSize, t.glassesArrowSize); _arrow.SetLocal(ax + 0.01f, 0f);
            _dot.SetSize(0.03f, 0.03f); _dot.SetLocal(ax + 0.01f, 0f);
            gameObject.name = "GlassesPanel";
        }

        public void Tick(string title, string detail, float? relBearingDeg, Color accent)
        {
            UiText.Set(_title, title);
            UiText.Set(_detail, UiText.Mono(t, detail));
            _plate.SetColor(t.plateColor); _outline.SetColor(t.outlineColor); _rule.SetColor(accent);
            _arrow.SetActive(relBearingDeg.HasValue); _dot.SetActive(!relBearingDeg.HasValue);
            _arrow.SetColor(accent); _dot.SetColor(t.TextSecondary);
            if (relBearingDeg.HasValue) _arrow.transform.localRotation = Quaternion.Euler(0f, 0f, -relBearingDeg.Value);
            _title.color = t.textLight; _detail.color = t.TextSecondary;
        }
    }
}
