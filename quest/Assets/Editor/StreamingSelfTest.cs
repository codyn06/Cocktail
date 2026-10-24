// Headset-free network test of the streaming client against a running hub (e.g. quest/tools/mock_hub.py).
//   uv run --project hub python quest/tools/mock_hub.py
//   Unity.exe -batchmode -nographics -projectPath quest -executeMethod Cocktail.Editor.StreamingSelfTest.Run -logFile st.log
// Env: COCKTAIL_TEST_HOST (default 127.0.0.1). Exit code 0 = all checks passed.
// Exercises the real HubConnection/WsChannel code (5 sockets, Hello, framing, epochs, reconnect on host change, hub events,
// /audio/out parsing). It does NOT test microphone, camera, pose sources or audio playback (those need the headset).
using System.Collections.Generic;
using System.Threading;
using Cocktail.Protocol;
using Cocktail.Stream;
using UnityEditor;
using UnityEngine;

namespace Cocktail.Editor
{
    public static class StreamingSelfTest
    {
        public static void Run()
        {
            int code = 1;
            try { code = RunInner() ? 0 : 1; }
            catch (System.Exception e) { Debug.LogException(e); }
            EditorApplication.Exit(code);
        }

        static bool RunInner()
        {
            string host = System.Environment.GetEnvironmentVariable("COCKTAIL_TEST_HOST");
            if (string.IsNullOrEmpty(host)) host = "127.0.0.1";
            var fails = new List<string>();
            void Check(bool ok, string what) { Debug.Log($"[SELFTEST] {(ok ? "PASS" : "FAIL")} {what}"); if (!ok) fails.Add(what); }

            var c = new HubConnection(host, 100);
            int audioOutFrames = 0; long lastAudioOutSeq = -1; int withBearing = 0; int badPayload = 0; long lastTs = 0;
            c.AudioOutReceived += (h, buf, off, bytes) =>
            {
                Interlocked.Increment(ref audioOutFrames);
                lastAudioOutSeq = h.Seq; lastTs = h.DeviceTsUs;
                if (h.TargetBearingDeg.HasValue) Interlocked.Increment(ref withBearing);
                if (bytes != 960 * 4) Interlocked.Increment(ref badPayload);
            };
            c.HelloProvider = ct => System.Threading.Tasks.Task.FromResult(new Hello
            { Device = "mock", MicChannels = 2, MicRaw = true, Camera = true, CameraHfovDeg = 100f });
            c.Start();

            Check(WaitFor(() => System.Array.TrueForAll(c.All, ch => ch.State == LinkState.Connected), 10000), "all 5 sockets connected");
            long poseEpoch1 = c.Pose.Epoch;

            // audio: 100 chunks of 20 ms stereo
            var pcm = new byte[960 * 2 * 2];
            for (int i = 0; i < 100; i++)
            {
                c.SendAudio(new AudioChunk { DeviceTsUs = ClockUs.Now, SampleRate = 48000, Channels = 2, SampleFormat = "s16le", Frames = 960 }, pcm, pcm.Length);
                Thread.Sleep(2);
            }
            // video: 5 small JPEGs
            var tex = new Texture2D(64, 48, TextureFormat.RGB24, false);
            var jpg = tex.EncodeToJPG(60);
            for (int i = 0; i < 5; i++)
            {
                c.SendVideo(new VideoFrame { DeviceTsUs = ClockUs.Now, Width = 64, Height = 48, Codec = "jpeg",
                    CamPos = new[] { 0f, 1.6f, 0f }, CamRotXyzw = new[] { 0f, 0f, 0f, 1f }, HfovDeg = 100f }, jpg, jpg.Length);
                Thread.Sleep(30);
            }
            // pose
            for (int i = 0; i < 50; i++)
            {
                c.SendPose(new PoseSample { DeviceTsUs = ClockUs.Now, Pos = new[] { 0f, 1.6f, 0f }, RotXyzw = new[] { 0f, 0.1f * i / 50f, 0f, 1f } });
                Thread.Sleep(4);
            }
            // input + commands
            var inputSw = System.Diagnostics.Stopwatch.StartNew();
            c.SendInput(new InputEvent { DeviceTsUs = ClockUs.Now, Action = Cocktail.Protocol.Action.Select, Modality = "head",
                RayOrigin = new[] { 0f, 1.6f, 0f }, RayDir = new[] { 0f, 0f, 1f } });
            c.SendCommand(new SetMode { Mode = Mode.Fast });
            c.SendCommand(new SetVolumes { InnerDb = 3f, OuterDb = -15f });

            Check(WaitFor(() => c.Audio.SentMsgs >= 100, 5000), $"audio sent {c.Audio.SentMsgs}/100");
            Check(WaitFor(() => c.Video.SentMsgs >= 3, 5000), $"video sent {c.Video.SentMsgs} (queue keeps newest, drops {c.Video.Queue.Dropped})");
            Check(WaitFor(() => c.Pose.SentMsgs >= 40, 5000), $"pose channel sent {c.Pose.SentMsgs} (hello + pose + input)");

            // events: sources.updated at ~10 Hz and a state change after the select
            int sources = 0, states = 0, targets = 0;
            WaitFor(() =>
            {
                while (c.IncomingEvents.TryDequeue(out var ev))
                {
                    if (ev is SourcesUpdated su && su.Sources != null && su.Sources.Count > 0) sources++;
                    if (ev is StateChanged sc) { states++; Debug.Log($"[SELFTEST] event state.changed {sc.State} target={sc.TargetId}"); }
                    if (ev is TargetUpdated tu) { targets++; Debug.Log($"[SELFTEST] event target.updated {tu.TargetId} bearing={tu.BearingDeg} (select -> echo {inputSw.ElapsedMilliseconds} ms)"); }
                }
                return sources >= 15 && targets > 0;
            }, 6000);
            Check(sources > 0, $"received sources.updated x{sources}");
            Check(states > 0, $"received state.changed x{states}");
            Check(targets > 0, $"received target.updated x{targets} (select echoed by mock hub)");

            Check(WaitFor(() => audioOutFrames >= 20, 4000), $"/audio/out frames received {audioOutFrames}");
            Check(badPayload == 0, "every /audio/out payload is 960 frames x 2 ch x 2 bytes");
            Check(WaitFor(() => withBearing > 0, 3000), $"/audio/out frames carry target_bearing_deg after select ({withBearing})");

            // reconnect on host change: epochs must increase on every socket, Hello re-sent
            long before = c.Audio.Epoch;
            c.SetHost("127.0.0.2"); // any other loopback address: different string, same machine
            Check(WaitFor(() => c.Audio.Epoch > before && c.Pose.Epoch > poseEpoch1 && System.Array.TrueForAll(c.All, ch => ch.State == LinkState.Connected), 10000),
                $"reconnect after host change: audio epoch {before}->{c.Audio.Epoch}, pose epoch {poseEpoch1}->{c.Pose.Epoch}");
            Check(c.Pose.Reconnects >= 1, $"pose reconnect counter {c.Pose.Reconnects}");
            c.SendAudio(new AudioChunk { DeviceTsUs = ClockUs.Now, SampleRate = 48000, Channels = 2, Frames = 960 }, pcm, pcm.Length);
            Check(c.Audio.Seq == 0, $"audio seq restarts at 0 in the new epoch (seq now {c.Audio.Seq})");

            // never blocks the caller: 20000 enqueues must take a few ms even with a full queue
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < 20000; i++) c.SendAudio(new AudioChunk { DeviceTsUs = 0, SampleRate = 48000, Channels = 2, Frames = 960 }, pcm, pcm.Length);
            Check(sw.ElapsedMilliseconds < 500, $"20000 audio enqueues took {sw.ElapsedMilliseconds} ms (producer never blocks)");

            Debug.Log($"[SELFTEST] stats audio tx={c.Audio.SentMsgs} video tx={c.Video.SentMsgs} pose tx={c.Pose.SentMsgs} events rx={c.Events.RecvMsgs} audio_out rx={c.AudioOutCh.RecvMsgs} lastOutSeq={lastAudioOutSeq} lastOutTs={lastTs}");
            c.Stop();
            Debug.Log(fails.Count == 0 ? "[SELFTEST] ALL PASSED" : "[SELFTEST] FAILED: " + string.Join("; ", fails));
            return fails.Count == 0;
        }

        static bool WaitFor(System.Func<bool> cond, int timeoutMs)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                if (cond()) return true;
                Thread.Sleep(20);
            }
            return cond();
        }
    }
}
