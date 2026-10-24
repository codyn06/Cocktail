using System.Text;
using UnityEngine;

namespace Cocktail.Ui
{
    /// <summary>
    /// Pure geometry/text helpers (no scene access, so they are unit-testable).
    ///
    /// Bearing convention (protocol Source.bearing_deg): WORLD YAW in the Quest TRACKING SPACE, 0 = tracking-space +Z
    /// (the initial forward direction when the app started / tracking origin), + = turning right (clockwise seen from above),
    /// i.e. direction = (sin b, 0, cos b) in tracking space. The same convention the hub derives from PoseSample and the one
    /// AudioOutPlayer uses. "Relative" angle = bearing - current head yaw, wrapped to (-180, 180]; 0 = straight ahead.
    /// </summary>
    public static class UiMath
    {
        public static float NormalizeDeg(float a)
        {
            a %= 360f;
            if (a > 180f) a -= 360f;
            else if (a <= -180f) a += 360f;
            return a;
        }

        /// <summary>Yaw (deg, + right) of a direction in tracking space.</summary>
        public static float YawOf(Vector3 dir) => Mathf.Atan2(dir.x, dir.z) * Mathf.Rad2Deg;

        public static Vector3 DirFromYawElev(float yawDeg, float elevDeg)
        {
            float y = yawDeg * Mathf.Deg2Rad, e = elevDeg * Mathf.Deg2Rad;
            float ce = Mathf.Cos(e);
            return new Vector3(Mathf.Sin(y) * ce, Mathf.Sin(e), Mathf.Cos(y) * ce);
        }

        /// <summary>12 = ahead, 3 = right, 6 = behind, 9 = left.</summary>
        public static int ClockPosition(float relDeg)
        {
            float a = ((relDeg % 360f) + 360f) % 360f;
            int h = Mathf.RoundToInt(a / 30f) % 12;
            return h == 0 ? 12 : h;
        }

        /// <summary>Human text for a relative bearing, e.g. "90 deg right", "ahead", "behind".</summary>
        public static string BearingText(float relDeg, UiTheme t)
        {
            float r = NormalizeDeg(relDeg);
            float abs = Mathf.Abs(r);
            if (abs < 5f) return t.aheadWord;
            if (abs > 175f) return t.behindWord;
            var sb = new StringBuilder(16);
            sb.Append(Mathf.RoundToInt(abs)).Append(t.degreeUnit).Append(' ').Append(r > 0 ? t.rightWord : t.leftWord);
            return sb.ToString();
        }

        public static float Level01(float levelDb, UiTheme t) =>
            Mathf.Clamp01(Mathf.InverseLerp(t.levelMinDb, t.levelMaxDb, levelDb));

        /// <summary>
        /// APPROXIMATE camera bbox -> angular box. bbox = normalized (x0,y0,x1,y1) in the latest camera frame, origin top-left.
        /// Pinhole model with horizontal fov hfov and frame aspect (w/h) giving the vertical fov. Assumes the camera axis equals
        /// the head forward direction and ignores head motion since the frame was captured (the protocol sends no camera pose
        /// with SourcesUpdated), so use it only to size the bracket and to estimate elevation, never as a precise position.
        /// Returns angular centre offset (yaw right+, pitch up+) and angular size, all degrees.
        /// </summary>
        public static void BboxToAngles(float[] bbox, float hfovDeg, float aspect,
            out float centerYawDeg, out float centerPitchDeg, out float widthDeg, out float heightDeg)
        {
            float th = Mathf.Tan(0.5f * hfovDeg * Mathf.Deg2Rad);
            float tv = th / Mathf.Max(0.1f, aspect);
            float x0 = bbox[0], y0 = bbox[1], x1 = bbox[2], y1 = bbox[3];
            float AngX(float nx) => Mathf.Atan((nx - 0.5f) * 2f * th) * Mathf.Rad2Deg;
            float AngY(float ny) => Mathf.Atan((0.5f - ny) * 2f * tv) * Mathf.Rad2Deg; // image y grows downward
            float ax0 = AngX(x0), ax1 = AngX(x1), ay0 = AngY(y0), ay1 = AngY(y1);
            centerYawDeg = 0.5f * (ax0 + ax1);
            centerPitchDeg = 0.5f * (ay0 + ay1);
            widthDeg = Mathf.Abs(ax1 - ax0);
            heightDeg = Mathf.Abs(ay1 - ay0);
        }

        /// <summary>Angular extent (deg) to a size in metres at the given distance.</summary>
        public static float AngleToSize(float deg, float distance) => 2f * distance * Mathf.Tan(0.5f * Mathf.Clamp(deg, 0.1f, 170f) * Mathf.Deg2Rad);

        // ------------------------------------------------------------------ contrast (WCAG 2.x)
        static float Lin(float c) => c <= 0.04045f ? c / 12.92f : Mathf.Pow((c + 0.055f) / 1.055f, 2.4f);
        public static float RelativeLuminance(Color c) => 0.2126f * Lin(c.r) + 0.7152f * Lin(c.g) + 0.0722f * Lin(c.b);

        /// <summary>WCAG contrast ratio of two opaque colours (alpha ignored; composite first if needed).</summary>
        public static float Contrast(Color a, Color b)
        {
            float la = RelativeLuminance(a), lb = RelativeLuminance(b);
            if (la < lb) { var t = la; la = lb; lb = t; }
            return (la + 0.05f) / (lb + 0.05f);
        }

        public static Color Over(Color fg, Color bg)
        {
            float a = fg.a;
            return new Color(fg.r * a + bg.r * (1 - a), fg.g * a + bg.g * (1 - a), fg.b * a + bg.b * (1 - a), 1f);
        }

        // ------------------------------------------------------------------ small text helper
        public static string Fmt(string template, string k1, string v1, string k2 = null, string v2 = null, string k3 = null, string v3 = null)
        {
            if (string.IsNullOrEmpty(template)) return "";
            var s = template.Replace(k1, v1);
            if (k2 != null) s = s.Replace(k2, v2);
            if (k3 != null) s = s.Replace(k3, v3);
            return s;
        }
    }
}
