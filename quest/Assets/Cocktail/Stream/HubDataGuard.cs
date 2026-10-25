using System.Collections.Generic;
using Cocktail.Protocol;

namespace Cocktail.Stream
{
    /// <summary>
    /// Data from the hub is checked once, where it enters the app (HubClient.Update), so nothing past that point has to doubt it:
    /// no null or duplicate sources, no NaN or Infinity in anything a Transform or a smoothing filter will see, and a hard cap on the
    /// list length. A bad value becomes "unknown" (null) or a safe default instead of travelling on. Mutates in place: the objects
    /// belong to the event queue and are used once.
    /// </summary>
    public static class HubDataGuard
    {
        /// <summary>The most sources one update may carry; the tracker never sends more than a handful.</summary>
        public const int MaxSources = 64;
        /// <summary>Level reported for a source whose level is missing or not a number.</summary>
        public const float SilentDb = -100f;

        static bool Ok(float v) => !float.IsNaN(v) && !float.IsInfinity(v);

        public static SourcesUpdated Clean(SourcesUpdated e)
        {
            if (e == null) return null;
            var list = e.Sources ?? new List<Source>();
            var seen = new HashSet<string>();
            int keep = 0;
            for (int i = 0; i < list.Count; i++)
            {
                var s = list[i];
                if (s == null || string.IsNullOrEmpty(s.SourceId) || !seen.Add(s.SourceId)) continue;
                Fix(s);
                list[keep++] = s;
                if (keep >= MaxSources) break;
            }
            list.RemoveRange(keep, list.Count - keep);
            e.Sources = list;
            return e;
        }

        public static TargetUpdated Clean(TargetUpdated e)
        {
            if (e == null) return null;
            if (e.BearingDeg.HasValue && !Ok(e.BearingDeg.Value)) e.BearingDeg = null;
            if (!Ok(e.AssocConf)) e.AssocConf = 0f;
            return e;
        }

        public static AlertFired Clean(AlertFired e)
        {
            if (e == null) return null;
            if (e.BearingDeg.HasValue && !Ok(e.BearingDeg.Value)) e.BearingDeg = null;
            return e;
        }

        static void Fix(Source s)
        {
            if (s.Label == null) s.Label = "";
            if (!Ok(s.LabelConf)) s.LabelConf = 0f;
            if (s.BearingDeg.HasValue && !Ok(s.BearingDeg.Value)) s.BearingDeg = null;
            if (!Ok(s.BearingConf)) s.BearingConf = 0f;
            if (s.ElevationDeg.HasValue && !Ok(s.ElevationDeg.Value)) s.ElevationDeg = null;
            if (s.DistanceM.HasValue && (!Ok(s.DistanceM.Value) || s.DistanceM.Value < 0f)) s.DistanceM = null;
            if (!Ok(s.LevelDb)) s.LevelDb = SilentDb;
            if (s.Bbox != null && !(s.Bbox.Length >= 4 && Ok(s.Bbox[0]) && Ok(s.Bbox[1]) && Ok(s.Bbox[2]) && Ok(s.Bbox[3]))) s.Bbox = null;
        }
    }
}
