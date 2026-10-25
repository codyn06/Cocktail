using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// Shows where the pointer is aiming: a faint thin ray from the hand and a dot where it lands. The dot lands at the distance of
    /// whatever is being pointed at (so it sits on the target from the wearer's eyes), grows when it is on a target and shrinks while
    /// pinching. With no tracked hand it falls back to a dot straight ahead of the head. Plain theme styling.
    /// </summary>
    public class PointerCursorView : MonoBehaviour
    {
        UiTheme t;
        LineRenderer _line;
        SpriteRenderer _dot, _dark;
        float _dist = 1.5f, _size;
        bool _placed;

        public void Build(UiTheme theme)
        {
            t = theme;
            gameObject.name = "PointerCursor";
            _dark = NewDot("CursorDark", UiLayers.Reticle);
            _dot = NewDot("Cursor", UiLayers.Reticle + 1);
            var lg = new GameObject("PointerRay"); lg.transform.SetParent(transform, false);
            _line = lg.AddComponent<LineRenderer>();
            _line.useWorldSpace = true; _line.positionCount = 2; _line.numCapVertices = 2;
            _line.sharedMaterial = _dot.sharedMaterial;   // the sprite material, always in the build
            _line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off; _line.receiveShadows = false;
            _line.sortingOrder = UiLayers.Reticle - 1;
        }

        SpriteRenderer NewDot(string name, int order)
        {
            var go = new GameObject(name); go.transform.SetParent(transform, false);
            var sr = go.AddComponent<SpriteRenderer>();
            sr.sprite = UiAssets.Disc; sr.sortingOrder = order;
            return sr;
        }

        /// <param name="dist">metres along the ray to the target (or a default distance when nothing is pointed at)</param>
        public void Tick(Transform pointer, float dist, bool onTarget, float pinch01, bool isHand, float dt)
        {
            if (pointer == null) return;
            _dist = _placed ? UiObjects.Ease(_dist, dist, t.positionSmoothing * 1.5f, dt) : dist;
            Vector3 origin = pointer.position, dir = pointer.forward;
            Vector3 end = origin + dir * _dist;

            float target = t.pointerDotSize * (onTarget ? 1.5f : 1f) * (1f - 0.45f * Mathf.Clamp01(pinch01));
            _size = _placed ? UiObjects.Ease(_size, target, 14f, dt) : target;
            _placed = true;

            var cam = Camera.main != null ? Camera.main.transform : null;
            _dot.transform.position = end; _dark.transform.position = end;
            if (cam != null) { var face = Quaternion.LookRotation(end - cam.position); _dot.transform.rotation = face; _dark.transform.rotation = face; }
            float unit = Mathf.Max(0.0001f, _dot.sprite.bounds.size.x);
            _dot.transform.localScale = Vector3.one * (_size / unit);
            _dark.transform.localScale = Vector3.one * ((_size + 2f * t.speakerOutlineExtra) / unit);
            Color c = onTarget ? t.cyan : t.offWhite; c.a = 0.9f;
            _dot.color = c;
            _dark.color = new Color(t.speakerOutlineColor.r, t.speakerOutlineColor.g, t.speakerOutlineColor.b, 0.8f);

            // the ray: only for a hand, faint at the hand and a little stronger toward the dot; the head fallback is just the dot
            _line.enabled = isHand;
            if (isHand)
            {
                _line.SetPosition(0, origin + dir * 0.06f); _line.SetPosition(1, end);
                _line.startWidth = t.pointerLineWidth; _line.endWidth = t.pointerLineWidth;
                var a = t.offWhite; a.a = 0.08f; var b = t.offWhite; b.a = onTarget ? 0.6f : 0.35f;
                _line.startColor = a; _line.endColor = b;
            }
        }
    }
}
