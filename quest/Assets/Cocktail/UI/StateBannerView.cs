using Cocktail.Protocol;
using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// Head-locked status plate: state headline (listening / selected / playing / target lost / replay processing / replay playing),
    /// the active target and the processing mode. A state change dips the plate for a couple of frames (no motion).
    /// </summary>
    public class StateBannerView : MonoBehaviour
    {
        UiTheme t;
        SpriteBox _plate, _outline, _rule;
        TextMeshPro _title, _detail;
        HubState _state = HubState.Listening; bool _hasState;
        int _flicker; float _clock;
        string _sig;

        public string TitleText => _title != null ? _title.text : "";
        public string DetailText => _detail != null ? _detail.text : "";
        public string Text => TitleText + " | " + DetailText;

        public void Build(UiTheme theme)
        {
            t = theme;
            var sz = t.bannerSize;
            UiLayers.Base = UiLayers.Banner;
            _plate = new SpriteBox(transform, "Fill", t.plateSprite != null ? t.plateSprite : UiAssets.Plate, 1);
            _outline = new SpriteBox(transform, "Outline", t.plateSprite != null ? t.plateSprite : UiAssets.Outline, 2);
            _rule = new SpriteBox(transform, "Rule", t.barSprite != null ? t.barSprite : UiAssets.White, 3);
            float left = -sz.x * 0.5f + t.platePadding + t.bannerRuleWidth + 0.01f;
            float w = sz.x - 2f * t.platePadding - t.bannerRuleWidth;
            _title = UiText.Make(transform, "Title", t, t.bannerTitleEm, TextAlignmentOptions.Left, w, t.bannerTitleEm * 1.4f, 5, true);
            _detail = UiText.Make(transform, "Detail", t, t.bannerDetailEm, TextAlignmentOptions.Left, w, t.bannerDetailEm * 1.4f, 5);
            _title.transform.localPosition = new Vector3(left + w * 0.5f, sz.y * 0.5f - t.platePadding - t.bannerTitleEm * 0.75f, 0f);
            _detail.transform.localPosition = new Vector3(left + w * 0.5f, -sz.y * 0.5f + t.platePadding + t.bannerDetailEm * 0.75f, 0f);
            _plate.SetSize(sz.x, sz.y); _outline.SetSize(sz.x, sz.y);
            _rule.SetSize(t.bannerRuleWidth, sz.y - 0.07f);
            _rule.SetLocal(-sz.x * 0.5f + t.bannerRuleWidth * 0.5f + 0.018f, -0.008f);
            gameObject.name = "StateBanner";
        }

        public void Tick(HubState state, string targetText, Mode mode, float replaySeconds, float dt)
        {
            _clock += dt;
            if (_hasState && state != _state) _flicker = Mathf.Max(0, t.flickerFrames);
            _state = state; _hasState = true;
            var st = t.GetState(state);
            float a = 1f;
            if (_flicker > 0) { a = t.flickerAlpha; _flicker--; }

            string title = UiMath.Fmt(t.bannerTitle, "{state}", st.title);
            if (st.animated)
            {
                int dots = 1 + (int)(_clock / Mathf.Max(0.05f, t.processingDotsPeriod)) % 3;
                title += new string('.', dots);
            }
            string detail = UiMath.Fmt(t.bannerDetail, "{target}", targetText ?? t.noTarget, "{mode}", t.ModeName(mode).ToString());
            if (state == HubState.ReplayPlaying && replaySeconds > 0f)
                detail += "  //  " + UiMath.Fmt(t.replayDetail, "{seconds}", replaySeconds.ToString("0"));
            string sig = title + "\n" + detail;
            if (sig != _sig) { _sig = sig; UiText.Set(_title, title); UiText.Set(_detail, UiText.Mono(t, detail)); }

            Color accent = t.StateColor(state);
            var pc = t.plateColor; pc.a *= a; _plate.SetColor(pc);
            _outline.SetColor(t.outlineColor); _outline.SetAlpha(a);
            _rule.SetColor(accent); _rule.SetAlpha(a);
            var tc = t.textLight; tc.a *= a; _title.color = tc;
            var dc = t.TextSecondary; dc.a *= a; _detail.color = dc;
        }
    }
}
