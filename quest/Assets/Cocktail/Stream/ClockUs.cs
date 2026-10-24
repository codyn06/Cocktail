using System.Diagnostics;

namespace Cocktail.Stream
{
    /// <summary>
    /// The one headset clock every device_ts_us comes from: microseconds on a monotonic timer, zero at process
    /// start (Stopwatch = CLOCK_MONOTONIC on Android/IL2CPP, QueryPerformanceCounter in the Windows editor).
    /// Thread-safe. All streamers (audio, video, pose, input) use this so the hub can align the channels.
    ///
    /// Raw CLOCK_MONOTONIC sources can be mapped onto it exactly:
    ///  - Java System.nanoTime() and AudioRecord.getTimestamp(TIMEBASE_MONOTONIC) (native mic plugin; MicStreamer
    ///    measures the Java-to-C# offset at start and logs it),
    ///  - the passthrough camera frame time (MRUK PassthroughCameraAccess keeps it in the OpenXR XrTime base,
    ///    which is CLOCK_MONOTONIC ns on Android) via <see cref="FromMonotonicNs"/>.
    /// Where an assumption can't be proven off-device, the streamers sanity-check the result and log it.
    /// </summary>
    public static class ClockUs
    {
        static readonly double s_usPerTick = 1_000_000.0 / Stopwatch.Frequency;
        static readonly long s_originUs = RawNowUs();

        static long RawNowUs() => (long)(Stopwatch.GetTimestamp() * s_usPerTick);

        /// <summary>Microseconds since process start on the monotonic clock.</summary>
        public static long Now => RawNowUs() - s_originUs;

        /// <summary>Convert a raw CLOCK_MONOTONIC nanosecond value to this clock (Android only meaningful).</summary>
        public static long FromMonotonicNs(long monotonicNs) => monotonicNs / 1000 - s_originUs;

        /// <summary>Convert raw monotonic microseconds (same base as <see cref="RawMonotonicNowUs"/>) to this clock.</summary>
        public static long FromRawUs(long rawUs) => rawUs - s_originUs;

        /// <summary>Raw monotonic microseconds (boot-relative on Android) to compare with Java System.nanoTime()/1000.</summary>
        public static long RawMonotonicNowUs => RawNowUs();
    }
}
