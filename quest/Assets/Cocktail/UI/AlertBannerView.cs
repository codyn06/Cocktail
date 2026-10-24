using Cocktail.Protocol;
using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// High-contrast alert plate with a diagonal hazard-stripe accent, the spoken text, a direction arrow (compass style:
    /// up = ahead, right = to your right) and the clock position. Critical alerts are coral with near-black text; others use the
    /// dark plate with a coral rule. Auto-dismisses after theme.alertSecondsCritical / alertSecondsNormal. A soft halo is used
    /// here and on the selected target only.
    /// </summary>
    public class AlertBannerView : MonoBehaviour
    {
        UiTheme t;
        SpriteBox _glow, _plate, _outline, _stripe, _stripeBack, _arrow, _rule;
        TextMeshPro _title, _detail;
        float _remaining, _total, _age;
        int _flicker;
        public string AlertId { get; private set; }
        public bool Visible => _remaining > 0f;
        public string Text => _title != null ? _title.text : "";
        public string DetailText => _detail != null ? _detail.text : "";
        public bool Critical { get; private set; }

        public void Build(UiTheme theme)
        {
            t = theme;
            var sz = t.alertSize;
            UiLayers.Base = UiLayers.Alert;
            _glow = new SpriteBox(transform, "Glow", UiAssets.Glow, 0);
            _plate = new SpriteBox(transform, "Fill", t.plateSprite != null ? t.plateSprite : UiAssets.Plate, 1);
            _outline = new SpriteBox(transform, "Outline", t.plateSprite != null ? t.plateSprite : UiAssets.Outline, 2);
            _stripeBack = new SpriteBox(transform, "HazardBack", t.barSprite != null ? t.barSprite : UiAssets.White, 3);
            _stripe = new SpriteBox(transform, "Hazard", UiAssets.Hazard, 4, true);
            _rule = new SpriteBox(transform, "Rule", t.barSprite != null ? t.barSprite : UiAssets.White, 3);
            _arrow = new SpriteBox(transform, "Arrow", t.arrowSprite != null ? t.arrowSprite : UiAssets.Triangle, 3);
            float textLeft = -sz.x * 0.5f + t.alertStripeWidth + 0.04f;
            float textRight = sz.x * 0.5f - t.alertArrowSize - 0.06f;
            float w = textRight - textLeft;
            _title = UiText.Make(transform, "Text", t, t.alertEm, TextAlignmentOptions.Left, w, t.alertEm * 1.5f, 5, true);
            _detail = UiText.Make(transform, "Detail", t, t.alertDetailEm, TextAlignmentOptions.Left, w, t.alertDetailEm * 1.4f, 5);
            _title.transform.localPosition = new Vector3(textLeft + w * 0.5f, sz.y * 0.5f - 0.035f - t.alertEm * 0.6f, 0f);
            _detail.transform.localPosition = new Vector3(textLeft + w * 0.5f, -sz.y * 0.5f + 0.03f + t.alertDetailEm * 0.7f, 0f);
            _plate.SetSize(sz.x, sz.y); _outline.SetSize(sz.x, sz.y);
            _stripe.SetSize(t.alertStripeWidth, sz.y - 0.06f);
            _stripe.SetLocal(-sz.x * 0.5f + t.alertStripeWidth * 0.5f + 0.03f, 0f);
            _stripeBack.SetSize(t.alertStripeWidth, sz.y - 0.06f);
            _stripeBack.SetLocal(-sz.x * 0.5f + t.alertStripeWidth * 0.5f + 0.03f, 0f);
            _rule.SetSize(0.01f, sz.y - 0.07f);
            _arrow.SetSize(t.alertArrowSize, t.alertArrowSize);
            _arrow.SetLocal(sz.x * 0.5f - t.alertArrowSize * 0.5f - 0.05f, 0f);
            gameObject.name = "AlertBanner";
            gameObject.SetActive(false);
        }

        /// <param name="relBearingDeg">bearing relative to the head or null</param>
        public void Show(AlertFired a, float? relBearingDeg)
        {
            bool same = AlertId == a.AlertId && Visible;
            AlertId = a.AlertId; Critical = a.Critical;
            _total = _remaining = a.Critical ? t.alertSecondsCritical : t.alertSecondsNormal;
            if (!same) { _flicker = Mathf.Max(0, t.flickerFrames); _age = 0f; }
            gameObject.SetActive(true);
            UiText.Set(_title, string.IsNullOrEmpty(a.SpokenText) ? a.Label : a.SpokenText);
            _lastRel = relBearingDeg;
            ApplyColors(1f);
            UpdateDirection(relBearingDeg);
        }

        float? _lastRel;

        /// <summary>Keep the arrow pointing at the source while the head turns.</summary>
        public void Tick(float dt, float? relBearingDeg)
        {
            if (_remaining <= 0f) { if (gameObject.activeSelf) gameObject.SetActive(false); return; }
            _remaining -= dt; _age += dt;
            float a = 1f;
            if (_flicker > 0) { a = t.flickerAlpha; _flicker--; }
            else if (_remaining < 0.4f) a = Mathf.Clamp01(_remaining / 0.4f);
            ApplyColors(a);
            UpdateDirection(relBearingDeg ?? _lastRel);
            if (_remaining <= 0f) gameObject.SetActive(false);
        }

        void UpdateDirection(float? rel)
        {
            _arrow.SetActive(rel.HasValue);
            if (rel.HasValue)
            {
                _arrow.transform.localRotation = Quaternion.Euler(0f, 0f, -rel.Value);
                UiText.Set(_detail, UiText.Mono(t, UiMath.Fmt(t.alertDetail, "{clock}", UiMath.ClockPosition(rel.Value).ToString(), "{bearing}", UiMath.BearingText(rel.Value, t))));
            }
            else UiText.Set(_detail, "");
        }

        void ApplyColors(float a)
        {
            var sz = t.alertSize;
            Color bg = Critical ? t.alertColor : t.plateColor;
            Color fg = Critical ? t.textDark : t.textLight;
            bg.a *= a; fg.a *= a;
            _plate.SetSize(sz.x, sz.y); _plate.SetColor(bg);
            _outline.SetColor(Critical ? t.textDark : t.alertColor); _outline.SetAlpha(a);
            _stripeBack.SetColor(t.textDark); _stripeBack.SetAlpha(a);
            _stripe.SetColor(t.hazardColor); _stripe.SetAlpha(a);
            _rule.SetActive(false);
            _arrow.SetColor(Critical ? t.textDark : t.alertColor); _arrow.SetAlpha(a);
            _title.color = fg;
            var dc = fg; dc.a *= Critical ? 1f : t.secondaryTextAlpha; _detail.color = dc;
            _glow.SetActive(true);
            _glow.SetSize(sz.x + 2f * t.glowSpread, sz.y + 2f * t.glowSpread);
            _glow.SetColor(t.alertColor); _glow.SetAlpha(t.glowAlpha * a);
        }
    }
}
