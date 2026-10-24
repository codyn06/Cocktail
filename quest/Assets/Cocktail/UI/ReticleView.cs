using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>Head-gaze reticle: a thin ring with four tick marks and a centre dot. Grows and takes the highlight colour while a source is under gaze.</summary>
    public class ReticleView : MonoBehaviour
    {
        UiTheme t;
        SpriteBox _ring, _dot;
        SpriteBox[] _ticks = new SpriteBox[4];
        float _d;

        public void Build(UiTheme theme)
        {
            t = theme;
            UiLayers.Base = UiLayers.Reticle;
            _ring = new SpriteBox(transform, "Ring", t.reticleSprite != null ? t.reticleSprite : UiAssets.Ring, 6);
            _dot = new SpriteBox(transform, "Dot", UiAssets.Diamond, 6);
            for (int i = 0; i < 4; i++) _ticks[i] = new SpriteBox(transform, "Tick" + i, t.barSprite != null ? t.barSprite : UiAssets.White, 6);
            _d = t.reticleDiameter;
            gameObject.name = "Reticle";
        }

        public void Tick(bool hovering, float dwell01, float dt)
        {
            _d = UiObjects.Ease(_d, hovering ? t.reticleHoverDiameter : t.reticleDiameter, t.easing, dt);
            Color c = hovering ? t.selectedColor : t.textLight;
            c.a = t.reticleAlpha;
            _ring.SetSize(_d, _d); _ring.SetColor(c);
            _dot.SetSize(t.reticleDotDiameter, t.reticleDotDiameter); _dot.SetColor(c);
            float r = _d * 0.5f, len = t.tickLength * 0.6f, w = t.tickThickness;
            Place(0, 0f, r + len * 0.5f, w, len, c); Place(1, 0f, -r - len * 0.5f, w, len, c);
            Place(2, -r - len * 0.5f, 0f, len, w, c); Place(3, r + len * 0.5f, 0f, len, w, c);
        }

        void Place(int i, float x, float y, float w, float h, Color c)
        {
            _ticks[i].SetSize(w, h); _ticks[i].SetLocal(x, y); _ticks[i].SetColor(c);
        }
    }
}
