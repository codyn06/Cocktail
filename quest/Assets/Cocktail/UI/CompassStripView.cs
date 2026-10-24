using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    public struct CompassPip
    {
        public string id;
        public float bearingDeg;   // world bearing (protocol convention)
        public Color color;
        public bool selected, lost;
        public float alpha;
    }

    /// <summary>
    /// Scanner-style compass strip, head-locked above the view: tick marks every compassTickMinorDeg (longer + numbered every
    /// compassTickMajorDeg) scrolling with the head heading, a fixed caret at the centre (where you look), and one pip per source at its
    /// world bearing. Sources outside the strip span stick to the nearest end as a triangle, so the strip also tells where to
    /// turn for out-of-view sources. Layout only; look comes from <see cref="UiTheme"/>.
    /// </summary>
    public class CompassStripView : MonoBehaviour
    {
        UiTheme t;
        SpriteBox _baseline, _caret;
        readonly List<SpriteBox> _ticks = new List<SpriteBox>();
        readonly List<TextMeshPro> _labels = new List<TextMeshPro>();
        readonly List<SpriteBox> _pips = new List<SpriteBox>();
        readonly List<string> _labelText = new List<string>();
        public int ActivePips { get; private set; }
        public int EdgePips { get; private set; }

        public void Build(UiTheme theme)
        {
            t = theme;
            UiLayers.Base = UiLayers.Radar;
            _baseline = new SpriteBox(transform, "Baseline", t.barSprite != null ? t.barSprite : UiAssets.White, 1);
            _baseline.SetSize(t.compassWidth, t.compassTickWidth); _baseline.SetColor(t.outlineColor);
            _caret = new SpriteBox(transform, "Caret", t.arrowSprite != null ? t.arrowSprite : UiAssets.Triangle, 3);
            _caret.SetSize(0.03f, 0.03f); _caret.SetLocal(0f, t.compassTickMajor + 0.012f); _caret.transform.localRotation = Quaternion.Euler(0, 0, 180f);
            _caret.SetColor(t.textLight);
            int n = Mathf.CeilToInt(t.compassSpanDeg / t.compassTickMinorDeg) + 2;
            for (int i = 0; i < n; i++)
                _ticks.Add(new SpriteBox(transform, "Tick" + i, t.barSprite != null ? t.barSprite : UiAssets.White, 2));
            int nl = Mathf.CeilToInt(t.compassSpanDeg / t.compassTickMajorDeg) + 2;
            for (int i = 0; i < nl; i++)
            {
                var l = UiText.Make(transform, "Num" + i, t, t.compassEm, TextAlignmentOptions.Center, 0.14f, t.compassEm * 1.4f, 5);
                _labels.Add(l); _labelText.Add("");
            }
            gameObject.name = "CompassStrip";
        }

        SpriteBox Pip(int i)
        {
            UiLayers.Base = UiLayers.Radar;
            while (_pips.Count <= i) _pips.Add(new SpriteBox(transform, "Pip" + _pips.Count, UiAssets.Diamond, 4));
            return _pips[i];
        }

        public void Tick(float headYawDeg, List<CompassPip> pips)
        {
            float half = t.compassSpanDeg * 0.5f;
            float halfW = t.compassWidth * 0.5f;
            float step = t.compassTickMinorDeg;
            float first = Mathf.Round(headYawDeg / step) * step;
            int ti = 0, li = 0;
            int steps = Mathf.CeilToInt(half / step);
            for (int i = -steps; i <= steps; i++)
            {
                float b = first + i * step;
                float rel = b - headYawDeg;
                if (Mathf.Abs(rel) > half || ti >= _ticks.Count) continue;
                float rem = Mathf.Repeat(b, t.compassTickMajorDeg);
                bool major = rem < 0.01f || t.compassTickMajorDeg - rem < 0.01f;
                float len = major ? t.compassTickMajor : t.compassTickMinor;
                float x = rel / half * halfW;
                var tk = _ticks[ti++];
                tk.SetActive(true); tk.SetSize(t.compassTickWidth, len); tk.SetLocal(x, len * 0.5f);
                tk.SetColor(t.textLight); tk.SetAlpha(major ? 0.9f : 0.5f * Mathf.Clamp01(2f - 2f * Mathf.Abs(rel) / half + 0.5f));
                if (major && li < _labels.Count)
                {
                    var l = _labels[li];
                    l.gameObject.SetActive(true);
                    l.transform.localPosition = new Vector3(x, t.compassTickMajor + t.compassEm * 0.9f + 0.03f, 0f);
                    int wb = Mathf.RoundToInt(Mathf.Repeat(b, 360f)) % 360;
                    if (_labelText[li] == null || _labelText[li] != wb.ToString())
                    {
                        _labelText[li] = wb.ToString();
                        l.text = UiText.Mono(t, wb.ToString("000"));
                    }
                    var c = t.TextSecondary; l.color = c;
                    li++;
                }
            }
            for (; ti < _ticks.Count; ti++) _ticks[ti].SetActive(false);
            for (; li < _labels.Count; li++) _labels[li].gameObject.SetActive(false);

            int n = 0, edge = 0;
            if (pips != null)
                for (int i = 0; i < pips.Count; i++)
                {
                    var p = pips[i];
                    float rel = UiMath.NormalizeDeg(p.bearingDeg - headYawDeg);
                    bool off = Mathf.Abs(rel) > half;
                    float x = Mathf.Clamp(rel, -half, half) / half * halfW;
                    var pip = Pip(n++);
                    pip.SetActive(true);
                    float size = t.compassPipSize * (p.selected ? 1.4f : 1f);
                    pip.sr.sprite = off ? (t.arrowSprite != null ? t.arrowSprite : UiAssets.Triangle) : UiAssets.Diamond;
                    pip.SetSize(size, size);
                    pip.SetLocal(x, -size * 0.8f);
                    pip.transform.localRotation = off ? Quaternion.Euler(0, 0, rel > 0 ? -90f : 90f) : Quaternion.identity;
                    pip.SetColor(p.selected ? t.selectedColor : p.color);
                    pip.SetAlpha(p.alpha * (p.lost ? 0.6f : 1f));
                    if (off) edge++;
                }
            for (int i = n; i < _pips.Count; i++) _pips[i].SetActive(false);
            ActivePips = n; EdgePips = edge;
        }
    }
}
