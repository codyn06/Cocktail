using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// Demo / test panel, head-locked at the upper left: a small "DEMO" chip that opens a grid of buttons (volumes, mode, replay,
    /// stop, next/prev, data source, ...) and a few lines of live readout. Buttons are pointed at with the hand ray (or the head) and
    /// pressed with the pinch, same as the picker. Plain theme styling; restyled later. Layout only: what the buttons do lives in CocktailUI.
    /// </summary>
    public class DemoPanelView : MonoBehaviour
    {
        const float Bw = 0.105f, Bh = 0.038f, Gap = 0.007f, Pad = 0.012f;

        class Btn
        {
            public string label; public Action click; public SpriteBox bg, line; public TextMeshPro text;
            public bool chip; public int col, span, row;
        }

        UiTheme t;
        readonly List<Btn> _b = new List<Btn>();
        SpriteBox _plate;
        TextMeshPro _readout;
        bool _open;
        int _hover = -1;
        float _flash; int _flashIdx = -1;
        int _rows;

        public bool Open => _open;
        public int HoverIndex => _hover;
        public bool TryGetHoverWorld(out Vector3 p)
        {
            p = default;
            if (_hover < 0 || _hover >= _b.Count) return false;
            p = _b[_hover].bg.transform.position; return true;
        }

        public void Build(UiTheme theme)
        {
            t = theme;
            UiLayers.Base = UiLayers.Banner;
            _plate = new SpriteBox(transform, "Plate", UiAssets.Plate, 0);
            _readout = UiText.Make(transform, "Readout", t, 0.018f, TextAlignmentOptions.TopLeft, 2f * Bw + Gap, 0.2f, 5);
            gameObject.name = "DemoPanel";
            AddButton("DEMO", () => SetOpen(!_open), 0, 2, -1, true);
            Layout();
        }

        public void SetOpen(bool open) { _open = open; Layout(); }

        public int AddButton(string label, Action click, int col, int span, int row, bool chip = false)
        {
            var b = new Btn { label = label, click = click, col = col, span = span, row = row, chip = chip };
            b.bg = new SpriteBox(transform, "Btn" + _b.Count, UiAssets.Plate, 1);
            b.line = new SpriteBox(transform, "BtnLine" + _b.Count, UiAssets.Outline, 2);
            b.text = UiText.Make(transform, "BtnText" + _b.Count, t, 0.02f, TextAlignmentOptions.Center, span * Bw + (span - 1) * Gap, Bh, 5, true);
            UiText.Set(b.text, label);
            _b.Add(b);
            if (row + 1 > _rows) _rows = row + 1;
            return _b.Count - 1;
        }

        public void SetLabel(int idx, string s)
        {
            if (idx < 0 || idx >= _b.Count) return;
            _b[idx].label = s; UiText.Set(_b[idx].text, s);
        }

        public void SetReadout(string s) { if (_open) UiText.Set(_readout, s); }

        float ColX(Btn b) => -Bw - Gap * 0.5f + b.col * (Bw + Gap) + (b.span * Bw + (b.span - 1) * Gap) * 0.5f;
        float RowY(Btn b) => b.chip ? 0f : -(Bh + Gap) * (b.row + 1) - Gap;

        void Layout()
        {
            // the plate runs from just under the chip to below the readout lines
            float top = -Bh * 0.5f - Gap, bottom = RowY(new Btn { row = _rows - 1 }) - Bh * 0.5f - 0.2f;
            _plate.SetActive(_open);
            if (_open)
            {
                _plate.SetSize(2f * Bw + Gap + 2f * Pad, top - bottom + 2f * Pad);
                _plate.SetLocal(0f, (top + bottom) * 0.5f);
                _plate.SetColor(t.plateColor);
            }
            foreach (var b in _b)
            {
                bool on = b.chip || _open;
                b.bg.SetActive(on); b.line.SetActive(on); b.text.gameObject.SetActive(on);
                if (!on) continue;
                float w = b.span * Bw + (b.span - 1) * Gap;
                b.bg.SetSize(w, Bh); b.line.SetSize(w, Bh);
                b.bg.SetLocal(ColX(b), RowY(b)); b.line.SetLocal(ColX(b), RowY(b));
                b.text.transform.localPosition = new Vector3(ColX(b), RowY(b), 0f);
                b.text.color = t.textLight;
            }
            _readout.gameObject.SetActive(_open);
            _readout.transform.localPosition = new Vector3(-Bw - Gap * 0.5f, RowY(new Btn { row = _rows - 1 }) - Bh * 0.5f - 2f * Gap, 0f);
            _readout.color = t.TextSecondary;
        }

        /// <summary>Point at the buttons with this ray (hand pointer or head); highlights the one under it.</summary>
        public void Tick(Transform pointer, float dt)
        {
            _flash = Mathf.Max(0f, _flash - dt);
            int best = -1; float bestScore = float.MaxValue;
            for (int i = 0; i < _b.Count; i++)
            {
                var b = _b[i];
                if (!(b.chip || _open)) continue;
                float w = b.span * Bw + (b.span - 1) * Gap;
                if (Hit(pointer, b.bg.transform.position, new Vector2(w, Bh) * 0.5f, out float sc) && sc < bestScore) { bestScore = sc; best = i; }
            }
            _hover = best;
            for (int i = 0; i < _b.Count; i++)
            {
                var b = _b[i];
                if (!(b.chip || _open)) continue;
                bool hov = i == _hover, fl = i == _flashIdx && _flash > 0f;
                Color c = fl ? t.cyan : hov ? t.navy : t.navyDark;
                c.a = hov || fl ? 0.95f : 0.8f;
                b.bg.SetColor(c);
                b.line.SetColor(hov ? t.offWhite : t.outlineColor);
            }
        }

        /// <summary>Press the button under the pointer. False when nothing is pointed at.</summary>
        public bool Press()
        {
            if (_hover < 0 || _hover >= _b.Count) return false;
            _flashIdx = _hover; _flash = 0.25f;
            _b[_hover].click?.Invoke();
            return true;
        }

        static bool Hit(Transform from, Vector3 center, Vector2 half, out float score)
        {
            Vector3 local = from.InverseTransformPoint(center);
            score = float.MaxValue;
            if (local.z < 0.15f) return false;
            float pad = 0.01f;
            float hax = Mathf.Atan2(half.x + pad, local.z), hay = Mathf.Atan2(half.y + pad, local.z);
            float dx = Mathf.Abs(Mathf.Atan2(local.x, local.z)), dy = Mathf.Abs(Mathf.Atan2(local.y, local.z));
            score = Mathf.Max(dx / hax, dy / hay);
            return score <= 1f;
        }
    }
}
