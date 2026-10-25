using System.Collections.Generic;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// Draws each tracked hand as dots on its joints, down the middle of every finger and across the palm, joined by thin lines
    /// (a see-through constellation instead of a solid hand model). Applied once per hand at runtime; nothing in the scene changes.
    /// </summary>
    public static class HandGridSkin
    {
        static readonly HashSet<int> Done = new HashSet<int>();

        public static void Apply(OVRHand hand, UiTheme t)
        {
            if (hand == null || !Done.Add(hand.GetInstanceID())) return;

            // no solid hand model, no bone-capsule overlay
            var meshRenderer = hand.GetComponent<OVRMeshRenderer>();
            if (meshRenderer != null) meshRenderer.enabled = false;
            var skinned = hand.GetComponent<SkinnedMeshRenderer>();
            if (skinned != null) skinned.enabled = false;
            var skeletonRenderer = hand.GetComponent<OVRSkeletonRenderer>();
            if (skeletonRenderer != null) skeletonRenderer.enabled = false;

            // the dots-and-lines skeleton is off by default (your real hands show through the passthrough); the theme can turn it back on
            if (t.showHandDots)
            {
                var dots = hand.gameObject.AddComponent<HandDots>();
                dots.Init(hand.GetComponent<OVRSkeleton>(), t);
            }
        }
    }

    public class HandDots : MonoBehaviour
    {
        OVRSkeleton _skel;
        UiTheme _t;
        Material _mat;
        Transform _root;
        readonly List<SpriteRenderer> _dots = new List<SpriteRenderer>();
        readonly List<LineRenderer> _lines = new List<LineRenderer>();

        public void Init(OVRSkeleton skeleton, UiTheme t)
        {
            _skel = skeleton; _t = t;
            _root = new GameObject("HandDots").transform;
            _root.SetParent(transform, false);
        }

        void LateUpdate()
        {
            bool ok = _skel != null && _skel.IsInitialized && _skel.IsDataValid && _skel.IsDataHighConfidence && _skel.Bones != null;
            if (!ok) { if (_root.gameObject.activeSelf) _root.gameObject.SetActive(false); return; }
            if (!_root.gameObject.activeSelf) _root.gameObject.SetActive(true);

            var cam = Camera.main != null ? Camera.main.transform : null;
            var bones = _skel.Bones;
            while (_dots.Count < bones.Count) AddJoint();
            Color c = _t.handDotColor;
            for (int i = 0; i < bones.Count; i++)
            {
                var b = bones[i];
                var dot = _dots[i];
                if (b.Transform == null) { dot.enabled = false; _lines[i].enabled = false; continue; }
                Vector3 p = b.Transform.position;
                dot.enabled = true;
                dot.transform.position = p;
                dot.transform.localScale = Vector3.one * (_t.handDotSize / Mathf.Max(0.0001f, dot.sprite.bounds.size.x));   // diameter in metres
                if (cam != null) dot.transform.rotation = Quaternion.LookRotation(p - cam.position);
                dot.color = c;

                var line = _lines[i];
                int pi = b.ParentBoneIndex;
                bool joined = pi >= 0 && pi < bones.Count && bones[pi].Transform != null;
                line.enabled = joined;
                if (joined)
                {
                    line.SetPosition(0, bones[pi].Transform.position); line.SetPosition(1, p);
                    line.startWidth = line.endWidth = _t.handLineWidth;
                    line.startColor = line.endColor = c;
                }
            }
        }

        void AddJoint()
        {
            var go = new GameObject("Dot" + _dots.Count);
            go.transform.SetParent(_root, false);
            var sr = go.AddComponent<SpriteRenderer>();
            sr.sprite = UiAssets.Disc;
            sr.sortingOrder = UiLayers.Reticle;
            if (_mat == null) _mat = sr.sharedMaterial;   // the sprite material, which is always in the build
            _dots.Add(sr);

            var lg = new GameObject("Line" + _lines.Count);
            lg.transform.SetParent(_root, false);
            var lr = lg.AddComponent<LineRenderer>();
            lr.useWorldSpace = true; lr.positionCount = 2; lr.numCapVertices = 2;
            lr.sharedMaterial = _mat;
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off; lr.receiveShadows = false;
            lr.sortingOrder = UiLayers.Reticle - 1;
            _lines.Add(lr);
        }
    }
}
