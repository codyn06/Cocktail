using System.Collections.Generic;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// Scanner-style radial ring, head-locked around the gaze point: one pip per source at its bearing relative to the head (0 = ahead at the top,
    /// 90 = right, 180 = behind at the bottom), the selected pip bigger and in the selection colour, tick marks every radarTickDeg
    /// and a faint cone showing the field of view inside which a source counts as in view. This is the primary out-of-view
    /// indicator (the edge arrows are the secondary one). Layout only; look comes from <see cref="UiTheme"/>.
    /// </summary>
    public class RadarRingView : MonoBehaviour
    {
        UiTheme t;
        SpriteBox _ring, _fov, _caret;
        readonly List<SpriteBox> _ticks = new List<SpriteBox>();
        readonly List<SpriteBox> _pips = new List<SpriteBox>();
        public int ActivePips { get; private set; }
        public int OutsideFovPips { get; private set; }

        public void Build(UiTheme theme)
        {
            t = theme;
            UiLayers.Base = UiLayers.Radar;
            float d = t.radarRadius * 2f;
            _fov = new SpriteBox(transform, "Fov", UiAssets.Sector(t.radarFovDeg), 0);
            _fov.SetSize(d, d); _fov.SetColor(WithAlpha(t.textLight, t.radarFovAlpha));
            _ring = new SpriteBox(transform, "Ring", t.reticleSprite != null ? t.reticleSprite : UiAssets.ThinRing, 1);
            _ring.SetSize(d, d); _ring.SetColor(WithAlpha(t.textLight, 0.5f * t.radarAlpha));
            _caret = new SpriteBox(transform, "Ahead", t.arrowSprite != null ? t.arrowSprite : UiAssets.Triangle, 3);
            _caret.SetSize(0.03f, 0.03f); _caret.SetLocal(0f, t.radarRadius + 0.03f); _caret.SetColor(WithAlpha(t.textLight, t.radarAlpha));
            int n = Mathf.Max(1, Mathf.RoundToInt(360f / Mathf.Max(5f, t.radarTickDeg)));
            for (int i = 0; i < n; i++)
            {
                float a = i * 360f / n;
                bool major = Mathf.Abs(Mathf.Repeat(a, 90f)) < 0.01f;
                float len = major ? t.radarTickMajor : t.radarTick;
                var tk = new SpriteBox(transform, "Tick" + i, t.barSprite != null ? t.barSprite : UiAssets.White, 2);
                tk.SetSize(t.radarLineWidth * 1.2f, len);
                float r = t.radarRadius - len * 0.5f;
                tk.SetLocal(Mathf.Sin(a * Mathf.Deg2Rad) * r, Mathf.Cos(a * Mathf.Deg2Rad) * r);
                tk.transform.localRotation = Quaternion.Euler(0f, 0f, -a);
                tk.SetColor(WithAlpha(t.textLight, (major ? 0.9f : 0.55f) * t.radarAlpha));
                _ticks.Add(tk);
            }
            gameObject.name = "RadarRing";
        }

        static Color WithAlpha(Color c, float a) { c.a = a; return c; }

        SpriteBox Pip(int i)
        {
            while (_pips.Count <= i)
            {
                UiLayers.Base = UiLayers.Radar;
                _pips.Add(new SpriteBox(transform, "Pip" + _pips.Count, UiAssets.Diamond, 4));
            }
            return _pips[i];
        }

        public void Tick(float headYawDeg, List<CompassPip> pips)
        {
            int n = 0, outside = 0;
            if (pips != null)
                for (int i = 0; i < pips.Count; i++)
                {
                    var p = pips[i];
                    float rel = UiMath.NormalizeDeg(p.bearingDeg - headYawDeg);
                    float r = t.radarRadius;
                    var pip = Pip(n++);
                    pip.SetActive(true);
                    float size = p.selected ? t.radarPipSizeSelected : t.radarPipSize;
                    pip.SetSize(size, size);
                    pip.SetLocal(Mathf.Sin(rel * Mathf.Deg2Rad) * r, Mathf.Cos(rel * Mathf.Deg2Rad) * r);
                    pip.SetColor(p.selected ? t.selectedColor : p.color);
                    pip.SetAlpha(p.alpha * (p.lost ? 0.6f : 1f) * t.radarAlpha);
                    if (Mathf.Abs(rel) > t.radarFovDeg * 0.5f) outside++;
                }
            for (int i = n; i < _pips.Count; i++) _pips[i].SetActive(false);
            ActivePips = n; OutsideFovPips = outside;
        }
    }
}
