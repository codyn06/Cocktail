using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    public struct SpeakerRow
    {
        public string id;
        public int number;
        public string name;
        public Color color;
        public bool selected;
        /// <summary>distance to the wearer in metres; float.MaxValue when unknown (sorts last)</summary>
        public float dist;
    }

    /// <summary>
    /// The audio picker, in the Cocktail look: head-locked plate with two corners cut at 45 degrees, one row per source (a square
    /// number block in the source colour and its name to the right). The chosen source's block has a white cut-corner frame; the row
    /// being pointed at gets a steel-teal band. Sizes are the design's degree values at 1.1 m; motion is the design's easing.
    /// </summary>
    public class SpeakerListView : MonoBehaviour
    {
        const int MaxRows = 5;
        const float D = 1.1f;   // metres: where the picker sits, and what the degree sizes are measured at

        UiTheme t;
        SpriteBox _plate, _frame, _chevUp, _chevDown;
        readonly SpriteBox[] _band = new SpriteBox[MaxRows], _bg = new SpriteBox[MaxRows];
        readonly TextMeshPro[] _num = new TextMeshPro[MaxRows], _name = new TextMeshPro[MaxRows];
        readonly float[] _rowA = new float[MaxRows], _bandA = new float[MaxRows], _flash = new float[MaxRows];
        readonly string[] _rowId = new string[MaxRows];
        float _frameY, _frameA, _plateH;
        bool _frameSet;
        int _top;
        float _y0, _rowH, _gap, _pad, _blockW, _nameW, _nameGap;
        string _lastChosen;

        public int VisibleRows { get; private set; }

        public void Build(UiTheme theme)
        {
            t = theme;
            _rowH = UiTheme.Deg(t.lookRowDeg, D); _gap = UiTheme.Deg(t.lookRowGapDeg, D); _pad = UiTheme.Deg(t.lookPadDeg, D);
            _blockW = UiTheme.Deg(t.lookBlockWDeg, D); _nameW = UiTheme.Deg(t.lookNameColDeg, D); _nameGap = UiTheme.Deg(t.lookNameGapDeg, D);
            UiLayers.Base = UiLayers.Banner;
            _plate = new SpriteBox(transform, "Plate", UiAssets.PlateScaled(UiTheme.Deg(t.lookChamferDeg, D)), 1);
            var bandSprite = UiAssets.PlateScaled(UiTheme.Deg(t.lookBandChamferDeg, D));
            for (int i = 0; i < MaxRows; i++)
            {
                _band[i] = new SpriteBox(transform, "Band" + i, bandSprite, 2);
                _bg[i] = new SpriteBox(transform, "Block" + i, UiAssets.White, 3);
                _num[i] = UiText.Make(transform, "Num" + i, t, UiTheme.Deg(t.lookNumDeg, D), TextAlignmentOptions.Center, _blockW, _rowH, 5, true);
                _name[i] = UiText.Make(transform, "Name" + i, t, UiTheme.Deg(t.lookNameDeg, D), TextAlignmentOptions.Left, _nameW, _rowH, 5);
                _name[i].fontStyle &= ~FontStyles.UpperCase;   // the design's names are lowercase
            }
            _frame = new SpriteBox(transform, "ChosenFrame", UiAssets.PlateScaled(UiTheme.Deg(t.lookFrameChamferDeg, D), true), 6);
            _chevUp = new SpriteBox(transform, "ChevUp", UiAssets.Chevron, 5);
            _chevDown = new SpriteBox(transform, "ChevDown", UiAssets.Chevron, 5);
            gameObject.name = "SpeakerList";
            Hide();
        }

        void Hide()
        {
            VisibleRows = 0;
            _plate.SetActive(false); _frame.SetActive(false); _chevUp.SetActive(false); _chevDown.SetActive(false);
            for (int i = 0; i < MaxRows; i++) { _band[i].SetActive(false); _bg[i].SetActive(false); _num[i].gameObject.SetActive(false); _name[i].gameObject.SetActive(false); }
        }

        /// <summary>World centre and half size of the i-th visible row (block + name), and its index in the row list, for pointing at a row.</summary>
        public bool RowAt(int i, out int rowIndex, out Vector3 worldCenter, out Vector2 half)
        {
            rowIndex = _top + i; half = new Vector2(_blockW + _nameGap + _nameW, _rowH) * 0.5f; worldCenter = default;
            if (i < 0 || i >= VisibleRows) return false;
            worldCenter = transform.TransformPoint(0f, _y0 - i * (_rowH + _gap), 0f);
            return true;
        }

        /// <param name="hover">index into rows of the row being pointed at, or -1</param>
        public void Tick(IReadOnlyList<SpeakerRow> rows, int hover, float dt)
        {
            int n = rows.Count;
            if (n == 0) { Hide(); return; }
            int shown = Mathf.Min(n, MaxRows);
            int anchor = hover >= 0 ? hover : 0;
            int top = Mathf.Clamp(anchor - shown / 2, 0, n - shown);
            VisibleRows = shown; _top = top;

            float chevZone = n > shown ? UiTheme.Deg(0.55f, D) : 0f;
            float w = 2f * _pad + _blockW + _nameGap + _nameW;
            float h = 2f * _pad + shown * _rowH + (shown - 1) * _gap + 2f * chevZone;
            _plateH = _plateH <= 0f ? h : UiObjects.Ease(_plateH, h, 1f / Mathf.Max(0.05f, t.tGlide) * 2f, dt);
            Color plate = t.lookPetrol; plate.a = t.lookPlate;
            _plate.SetActive(true); _plate.SetSize(w, _plateH); _plate.SetColor(plate);

            float left = -w * 0.5f + _pad;
            _y0 = h * 0.5f - _pad - chevZone - _rowH * 0.5f;
            string chosen = null; float chosenY = 0f;
            for (int i = 0; i < MaxRows; i++)
            {
                bool on = i < shown;
                var r = on ? rows[top + i] : default;
                float y = _y0 - i * (_rowH + _gap);
                _rowA[i] = UiObjects.Fade(_rowA[i], on ? 1f : 0f, on ? t.tAppear : t.tDisappear, dt);
                bool vis = on || _rowA[i] > 0f;
                _band[i].SetActive(vis); _bg[i].SetActive(vis); _num[i].gameObject.SetActive(vis); _name[i].gameObject.SetActive(vis);
                if (!vis) continue;
                if (!on)   // fading out: same look, falling opacity
                {
                    _bg[i].SetAlpha(_rowA[i]); _band[i].SetActive(false);
                    var fn = _name[i].color; fn.a *= 0.8f; _name[i].color = fn;
                    var fk = _num[i].color; fk.a = _rowA[i]; _num[i].color = fk;
                    continue;
                }

                bool hov = top + i == hover;
                _bandA[i] = UiObjects.Fade(_bandA[i], hov ? 1f : 0f, hov ? t.tGlide : t.tDisappear, dt);
                float bo = UiTheme.Deg(t.lookBandOutsetDeg, D);
                Color bc = t.lookFocus; bc.a = t.lookHoverBand * _bandA[i] * _rowA[i];
                _band[i].SetActive(_bandA[i] > 0.001f);
                _band[i].SetSize(_blockW + _nameGap + _nameW + 2f * bo, _rowH + 2f * bo);
                _band[i].SetLocal(left + (_blockW + _nameGap + _nameW) * 0.5f, y); _band[i].SetColor(bc);

                // the block: source colour at 65%, 90% when chosen; flashes white for a moment after a pinch
                float blockA = r.selected ? t.lookBlockChosen : t.lookBlock;
                Color col = r.color.a > 0f ? r.color : t.lookSources[0];
                if (_flash[i] > 0f) { _flash[i] = Mathf.Max(0f, _flash[i] - dt / Mathf.Max(0.05f, t.tFlash)); col = Color.Lerp(col, t.lookPaper, _flash[i] * _flash[i]); }
                col.a = blockA * _rowA[i];
                _bg[i].SetSize(_blockW, _rowH); _bg[i].SetLocal(left + _blockW * 0.5f, y); _bg[i].SetColor(col);

                _num[i].transform.localPosition = new Vector3(left + _blockW * 0.5f, y, 0f);
                UiText.Set(_num[i], t.speakerListShowNumbers ? r.number.ToString() : "");
                var ink = t.lookInk; ink.a = _rowA[i]; _num[i].color = ink;

                _name[i].transform.localPosition = new Vector3(left + _blockW + _nameGap + _nameW * 0.5f, y, 0f);
                UiText.Set(_name[i], r.name ?? "");
                var nc = t.lookPaper; nc.a = (hov ? 1f : t.lookName) * _rowA[i]; _name[i].color = nc;

                if (r.selected) { chosen = r.id; chosenY = y; if (_lastChosen != r.id) _flash[i] = 1f; }
            }
            _lastChosen = chosen;

            // the chosen frame glides between rows and fades out when nothing is chosen
            _frameA = UiObjects.Fade(_frameA, chosen != null ? 1f : 0f, chosen != null ? t.tAppear : t.tDisappear, dt);
            if (chosen != null) { _frameY = _frameSet ? UiObjects.Ease(_frameY, chosenY, 6.9f / Mathf.Max(0.05f, t.tGlide), dt) : chosenY; _frameSet = true; }
            else _frameSet = false;
            _frame.SetActive(_frameA > 0.001f);
            float fo = UiTheme.Deg(t.lookFrameOutsetDeg, D);
            Color fc = t.lookPaper; fc.a = t.lookFrame * _frameA;
            _frame.SetSize(_blockW + 2f * fo, _rowH + 2f * fo); _frame.SetLocal(left + _blockW * 0.5f, _frameY); _frame.SetColor(fc);

            // chevrons when there are more rows above or below
            float cs = UiTheme.Deg(0.45f, D);
            Color cc = t.lookPaper; cc.a = t.lookChevron;
            bool up = top > 0, down = top + shown < n;
            _chevUp.SetActive(up); _chevDown.SetActive(down);
            if (up) { _chevUp.SetSize(cs, cs); _chevUp.SetLocal(0f, h * 0.5f - _pad - chevZone * 0.5f); _chevUp.SetColor(cc); _chevUp.transform.localRotation = Quaternion.identity; }
            if (down) { _chevDown.SetSize(cs, cs); _chevDown.SetLocal(0f, -h * 0.5f + _pad + chevZone * 0.5f); _chevDown.SetColor(cc); _chevDown.transform.localRotation = Quaternion.Euler(0f, 0f, 180f); }
        }
    }
}
