using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// Built-in flat sprites used when the theme leaves a sprite slot empty. All are generated at runtime from a few
    /// pixels (no asset files, no shaders to strip): chamfered plate + outline + halo, ring, triangle, diamond, hazard stripes.
    /// Chamfered = the top-left and bottom-right corners are cut diagonally (the shape reused by every plate in the UI).
    /// </summary>
    public static class UiAssets
    {
        const int PlateN = 48, PlateBorder = 16, PlateCut = 14;
        const float PlatePpu = 512f;
        static Sprite _white, _plate, _outline, _glow, _ring, _tri, _diamond, _hazard;

        public static Sprite White => _white != null ? _white : (_white = Solid());
        public static Sprite Plate => _plate != null ? _plate : (_plate = MakePlate(false));
        public static Sprite Outline => _outline != null ? _outline : (_outline = MakePlate(true));
        public static Sprite Glow => _glow != null ? _glow : (_glow = MakeGlow());
        public static Sprite ThinRing => _thinRing != null ? _thinRing : (_thinRing = MakeRing(256, 2f));
        static Sprite _thinRing;
        public static Sprite Ring => _ring != null ? _ring : (_ring = MakeRing());
        public static Sprite Triangle => _tri != null ? _tri : (_tri = MakeTriangle());
        public static Sprite Diamond => _diamond != null ? _diamond : (_diamond = MakeDiamond());
        public static Sprite Disc => _disc != null ? _disc : (_disc = MakeDisc());
        public static Sprite DashedRing => _dashed != null ? _dashed : (_dashed = MakeDashedRing());
        /// <summary>Arrowhead pointing up (notched base), the speaker-direction arrow.</summary>
        public static Sprite Chevron => _chevron != null ? _chevron : (_chevron = MakeChevron());
        static Sprite _disc, _dashed, _chevron;
        static readonly System.Collections.Generic.Dictionary<int, Sprite> _sectors = new System.Collections.Generic.Dictionary<int, Sprite>();
        /// <summary>Filled sector pointing up, symmetric, with the given full angle (deg); 1 unit diameter.</summary>
        public static Sprite Sector(float fullAngleDeg)
        {
            int key = Mathf.RoundToInt(fullAngleDeg);
            if (_sectors.TryGetValue(key, out var sp) && sp != null) return sp;
            int n = 128;
            var t = NewTex(n, n);
            var px = new Color32[n * n];
            float c = (n - 1) * 0.5f, half = key * 0.5f;
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    float dx = x - c, dy = y - c;
                    float d = Mathf.Sqrt(dx * dx + dy * dy);
                    float ang = Mathf.Abs(Mathf.Atan2(dx, dy) * Mathf.Rad2Deg);   // 0 = up
                    float a = (d <= 62f && ang <= half) ? 1f : 0f;
                    px[y * n + x] = new Color32(255, 255, 255, (byte)(a * 255f));
                }
            sp = Finish(t, px, n);
            _sectors[key] = sp;
            return sp;
        }

        public static Sprite Hazard => _hazard != null ? _hazard : (_hazard = MakeHazard());

        static Texture2D NewTex(int w, int h, TextureWrapMode wrap = TextureWrapMode.Clamp)
        {
            var t = new Texture2D(w, h, TextureFormat.RGBA32, false)
            {
                hideFlags = HideFlags.HideAndDontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = wrap,
                name = "CocktailUiGenerated",
            };
            return t;
        }

        static Sprite Finish(Texture2D t, Color32[] px, float ppu, Vector4 border = default)
        {
            t.SetPixels32(px);
            t.Apply(false, true);
            var s = Sprite.Create(t, new Rect(0, 0, t.width, t.height), new Vector2(0.5f, 0.5f), ppu, 0, SpriteMeshType.FullRect, border);
            s.hideFlags = HideFlags.HideAndDontSave;
            return s;
        }

        static Sprite Solid()
        {
            var t = NewTex(4, 4);
            var px = new Color32[16];
            for (int i = 0; i < px.Length; i++) px[i] = new Color32(255, 255, 255, 255);
            return Finish(t, px, 4f);
        }

        // inside test of the chamfered shape in a PlateN x PlateN texture (y up)
        static bool InShape(int x, int y, int n, int cut)
        {
            if (x < 0 || y < 0 || x >= n || y >= n) return false;
            if (x + (n - 1 - y) < cut) return false;      // top-left cut
            if ((n - 1 - x) + y < cut) return false;      // bottom-right cut
            return true;
        }

        static Sprite MakePlate(bool outlineOnly)
        {
            int n = PlateN;
            var t = NewTex(n, n);
            var px = new Color32[n * n];
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    bool inside = InShape(x, y, n, PlateCut);
                    byte a = 0;
                    if (inside)
                    {
                        if (!outlineOnly) a = 255;
                        else
                        {
                            bool edge = false;
                            for (int d = 1; d <= 2 && !edge; d++)
                                edge = !InShape(x + d, y, n, PlateCut) || !InShape(x - d, y, n, PlateCut) ||
                                       !InShape(x, y + d, n, PlateCut) || !InShape(x, y - d, n, PlateCut);
                            if (edge) a = 255;
                        }
                    }
                    px[y * n + x] = new Color32(255, 255, 255, a);
                }
            return Finish(t, px, PlatePpu, new Vector4(PlateBorder, PlateBorder, PlateBorder, PlateBorder));
        }

        static Sprite MakeGlow()
        {
            // soft halo: chamfered shape inset in a 64px texture, box-blurred twice. One extra sprite, flat colour * alpha.
            int n = 64, inset = 18, cut = 22, r = 7;
            var a = new float[n * n];
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                    a[y * n + x] = InShape(x - inset, y - inset, n - 2 * inset, cut - 4) ? 1f : 0f;
            for (int pass = 0; pass < 2; pass++)
            {
                var b = new float[n * n];
                for (int y = 0; y < n; y++)
                    for (int x = 0; x < n; x++)
                    {
                        float s = 0; int c = 0;
                        for (int dy = -r; dy <= r; dy += 2)
                            for (int dx = -r; dx <= r; dx += 2)
                            {
                                int xx = Mathf.Clamp(x + dx, 0, n - 1), yy = Mathf.Clamp(y + dy, 0, n - 1);
                                s += a[yy * n + xx]; c++;
                            }
                        b[y * n + x] = s / c;
                    }
                a = b;
            }
            var t = NewTex(n, n);
            var px = new Color32[n * n];
            for (int i = 0; i < px.Length; i++) px[i] = new Color32(255, 255, 255, (byte)(Mathf.Clamp01(a[i]) * 255f));
            return Finish(t, px, PlatePpu, new Vector4(26, 26, 26, 26));
        }

        static Sprite MakeRing() => MakeRing(128, 4f);

        static Sprite MakeRing(int n, float thick)
        {
            var t = NewTex(n, n);
            var px = new Color32[n * n];
            float c = (n - 1) * 0.5f, ro = n * 0.5f - 3f;
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    float d = Mathf.Sqrt((x - c) * (x - c) + (y - c) * (y - c));
                    float a = Mathf.Clamp01(1f - (Mathf.Abs(d - (ro - thick * 0.5f)) - thick * 0.5f));
                    px[y * n + x] = new Color32(255, 255, 255, (byte)(a * 255f));
                }
            return Finish(t, px, n);
        }

        static Sprite MakeTriangle()
        {
            int n = 64;
            var t = NewTex(n, n);
            var px = new Color32[n * n];
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    // apex up; half width grows linearly from 0 at the top to 28 px at the bottom
                    float v = y / (float)(n - 1);                 // 0 bottom .. 1 top
                    float half = 28f * (1f - v) + 0.5f;
                    float a = Mathf.Clamp01(half - Mathf.Abs(x - (n - 1) * 0.5f));
                    if (y < 4 || y > n - 4) a = 0;
                    px[y * n + x] = new Color32(255, 255, 255, (byte)(a * 255f));
                }
            return Finish(t, px, n);
        }

        static Sprite MakeDisc()
        {
            int n = 64;
            var t = NewTex(n, n);
            var px = new Color32[n * n];
            float c = (n - 1) * 0.5f, r = n * 0.5f - 2f;
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    float d = Mathf.Sqrt((x - c) * (x - c) + (y - c) * (y - c));
                    px[y * n + x] = new Color32(255, 255, 255, (byte)(Mathf.Clamp01(r - d + 0.5f) * 255f));
                }
            return Finish(t, px, n);
        }

        static Sprite MakeDashedRing()
        {
            int n = 256, dashes = 24;
            var t = NewTex(n, n);
            var px = new Color32[n * n];
            float c = (n - 1) * 0.5f, ro = n * 0.5f - 3f, thick = 4f;
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    float dx = x - c, dy = y - c;
                    float d = Mathf.Sqrt(dx * dx + dy * dy);
                    float a = Mathf.Clamp01(1f - (Mathf.Abs(d - (ro - thick * 0.5f)) - thick * 0.5f));
                    float ang = (Mathf.Atan2(dy, dx) + Mathf.PI) / (2f * Mathf.PI) * dashes;
                    if (ang - Mathf.Floor(ang) > 0.55f) a = 0f;   // dash, then gap
                    px[y * n + x] = new Color32(255, 255, 255, (byte)(a * 255f));
                }
            return Finish(t, px, n);
        }

        static Sprite MakeChevron()
        {
            int n = 64;
            var t = NewTex(n, n);
            var px = new Color32[n * n];
            // arrowhead, apex up: (0.5,1) -> (1,0) -> (0.5,0.28) -> (0,0), 3x3 supersampled
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    int hit = 0;
                    for (int sy = 0; sy < 3; sy++)
                        for (int sx = 0; sx < 3; sx++)
                        {
                            float u = (x + (sx + 0.5f) / 3f) / n, v = (y + (sy + 0.5f) / 3f) / n;   // v: 0 bottom .. 1 top
                            float half = 0.5f * (1f - v);                                          // outer edge: width shrinks to the apex
                            bool inOuter = Mathf.Abs(u - 0.5f) <= half;
                            float notch = 0.28f * (1f - Mathf.Abs(u - 0.5f) / 0.5f);               // base notch: V cut up to 0.28 at the middle
                            if (inOuter && v >= notch) hit++;
                        }
                    px[y * n + x] = new Color32(255, 255, 255, (byte)(hit / 9f * 255f));
                }
            return Finish(t, px, n);
        }

        static Sprite MakeDiamond()
        {
            int n = 32;
            var t = NewTex(n, n);
            var px = new Color32[n * n];
            float c = (n - 1) * 0.5f;
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    float a = Mathf.Clamp01(14f - (Mathf.Abs(x - c) + Mathf.Abs(y - c)));
                    px[y * n + x] = new Color32(255, 255, 255, (byte)(a * 255f));
                }
            return Finish(t, px, n);
        }

        static Sprite MakeHazard()
        {
            int n = 16;
            var t = NewTex(n, n, TextureWrapMode.Repeat);
            var px = new Color32[n * n];
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                    px[y * n + x] = new Color32(255, 255, 255, (byte)(((x + y) % n) < n / 2 ? 255 : 0));
            return Finish(t, px, 256f);
        }
    }

    /// <summary>A flat-colour rectangle/shape drawn by a SpriteRenderer (works with built-in shapes and theme sprites).</summary>
    public sealed class SpriteBox
    {
        public readonly GameObject go;
        public readonly SpriteRenderer sr;
        readonly bool _sliced, _tiled;
        Vector2 _size = Vector2.one;
        Color _color = Color.white;

        public Transform transform => go.transform;
        public Color Color => _color;

        public SpriteBox(Transform parent, string name, Sprite sprite, int order, bool tiled = false)
        {
            go = new GameObject(name);
            go.transform.SetParent(parent, false);
            sr = go.AddComponent<SpriteRenderer>();
            sr.sprite = sprite;
            sr.sortingOrder = order + UiLayers.Base;
            sr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            sr.receiveShadows = false;
            _tiled = tiled;
            _sliced = !tiled && sprite != null && sprite.border != Vector4.zero;
            if (_sliced) sr.drawMode = SpriteDrawMode.Sliced;
            else if (_tiled) { sr.drawMode = SpriteDrawMode.Tiled; sr.tileMode = SpriteTileMode.Continuous; }
        }

        public void SetSize(float w, float h)
        {
            _size = new Vector2(Mathf.Max(0.0001f, w), Mathf.Max(0.0001f, h));
            if (_sliced || _tiled) { sr.size = _size; go.transform.localScale = Vector3.one; }
            else
            {
                var b = sr.sprite.bounds.size;
                go.transform.localScale = new Vector3(_size.x / b.x, _size.y / b.y, 1f);
            }
        }

        public void SetColor(Color c) { _color = c; sr.color = c; }
        public void SetAlpha(float a) { var c = _color; c.a *= a; sr.color = c; }
        public void SetActive(bool on) { if (go.activeSelf != on) go.SetActive(on); }
        public void SetLocal(float x, float y, float z = 0f) { go.transform.localPosition = new Vector3(x, y, z); }
    }

    /// <summary>TextMeshPro world-text factory driven by the theme.</summary>
    public static class UiText
    {
        public static TextMeshPro Make(Transform parent, string name, UiTheme t, float em, TextAlignmentOptions align,
            float boxW, float boxH, int order, bool title = false)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var tmp = go.AddComponent<TextMeshPro>();
            if (t.font != null) tmp.font = t.font;
            tmp.fontSize = t.FontSizeForEm(em);
            tmp.alignment = align;
            tmp.textWrappingMode = TextWrappingModes.NoWrap;
            tmp.overflowMode = TextOverflowModes.Overflow;
            tmp.characterSpacing = t.characterSpacing;
            tmp.fontStyle = (title ? t.titleStyle : t.bodyStyle) | (t.uppercase ? FontStyles.UpperCase : FontStyles.Normal);
            tmp.richText = true;
            tmp.raycastTarget = false;
            go.transform.localScale = new Vector3(t.fontCondense, 1f, 1f);
            tmp.rectTransform.sizeDelta = new Vector2(boxW / t.fontCondense, boxH);
            var mr = go.GetComponent<MeshRenderer>();
            if (mr != null) { mr.sortingOrder = order + UiLayers.Base; mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off; mr.receiveShadows = false; }
            tmp.text = "";
            return tmp;
        }

        /// <summary>Set the text box size in metres (the horizontal condense of the text object is compensated).</summary>
        public static void SetBox(TMP_Text tmp, UiTheme t, float w, float h) => tmp.rectTransform.sizeDelta = new Vector2(w / t.fontCondense, h);

        /// <summary>Wrap in TMP monospace so digits do not jitter (theme.monospaceEm = 0 turns it off).</summary>
        public static string Mono(UiTheme t, string s) =>
            t.monospaceEm > 0f ? "<mspace=" + t.monospaceEm.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture) + "em>" + s + "</mspace>" : s;

        /// <summary>Set text only when it changed (avoids TMP re-layout and garbage every frame).</summary>
        public static void Set(TMP_Text tmp, string s)
        {
            if (tmp.text != s) tmp.text = s;
        }
    }

    /// <summary>Sorting-order bases so head-locked content always draws over world content. Views set UiLayers.Base before creating sprites/text.</summary>
    public static class UiLayers
    {
        public const int World = 0, Chips = 10, Radar = 20, Arrows = 30, Banner = 40, Alert = 60, Reticle = 80;
        public static int Base;
    }

    public static class UiObjects
    {
        public static void Kill(Object o)
        {
            if (o == null) return;
            if (Application.isPlaying) Object.Destroy(o); else Object.DestroyImmediate(o);
        }

        public static float Ease(float current, float target, float rate, float dt) =>
            Mathf.Lerp(current, target, 1f - Mathf.Exp(-rate * dt));
    }
}
