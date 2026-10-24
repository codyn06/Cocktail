using System.Collections.Concurrent;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Cocktail.Protocol;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Cocktail.Stream
{
    /// <summary>
    /// Plain C# core of the hub client (no MonoBehaviour), so it can run in the Editor self-test and in
    /// player builds. Five independent sockets: audio in, video in, pose/input in, events (both ways),
    /// audio out. Producers call Send*/Enqueue methods from any thread; nothing here blocks the caller.
    /// </summary>
    public sealed class HubConnection
    {
        public readonly WsChannel Audio, Video, Pose, Events, AudioOutCh;
        public WsChannel[] All => new[] { Audio, Video, Pose, Events, AudioOutCh };

        string _host;
        readonly object _hostLock = new object();
        long _epochCounter;

        /// <summary>Hub events decoded on the network thread (SourcesUpdated, StateChanged, TargetUpdated,
        /// AlertFired, ReplayReady, Metrics). HubClient drains this on the main thread.</summary>
        public readonly ConcurrentQueue<object> IncomingEvents = new ConcurrentQueue<object>();

        /// <summary>Audio-out frame, invoked on the network thread. payload = s16le interleaved (layers: ch0 inner, ch1 outer).
        /// Buffer is reused after the call returns.</summary>
        public delegate void AudioOutHandler(AudioOut header, byte[] buffer, int payloadOffset, int payloadBytes);
        public event AudioOutHandler AudioOutReceived;

        /// <summary>Called on the pose channel after every (re)connect, before queued messages. Must return the Hello to send
        /// (it may wait for mic/camera readiness).</summary>
        public System.Func<CancellationToken, Task<Hello>> HelloProvider;

        public string Host { get { lock (_hostLock) return _host; } }

        public HubConnection(string host, long epochBase, int port = ProtocolInfo.DefaultPort)
        {
            _host = host;
            _epochCounter = epochBase;
            System.Func<string> hostFn = () => Host;
            System.Func<long> epochFn = () => Interlocked.Increment(ref _epochCounter);

            Audio = new WsChannel("audio", ProtocolInfo.WsAudioIn, 150, DropPolicy.DropOldest, hostFn, port, epochFn);        // ~3 s
            Video = new WsChannel("video", ProtocolInfo.WsVideoIn, 2, DropPolicy.DropOldest, hostFn, port, epochFn);         // newest wins
            Pose = new WsChannel("pose", ProtocolInfo.WsPoseIn, 16, DropPolicy.DropOldest, hostFn, port, epochFn);
            Events = new WsChannel("events", ProtocolInfo.WsEvents, 64, DropPolicy.DropOldest, hostFn, port, epochFn) { ClearQueueOnReconnect = false };
            AudioOutCh = new WsChannel("audio_out", ProtocolInfo.WsAudioOut, 1, DropPolicy.DropOldest, hostFn, port, epochFn);

            Pose.OnConnected = async (ch, ct) =>
            {
                if (HelloProvider == null) return;
                var hello = await HelloProvider(ct).ConfigureAwait(false);
                if (hello == null) return;
                hello.Seq = ch.NextSeq();
                hello.DeviceTsUs = ClockUs.Now;
                ch.SendJsonFirst(hello);
            };
            Events.OnText = OnEventText;
            AudioOutCh.OnBinary = OnAudioOutBinary;
        }

        public void Start() { foreach (var c in All) c.Start(); }
        public void Stop() { foreach (var c in All) c.Stop(); }

        public void SetHost(string host)
        {
            lock (_hostLock) { if (_host == host) return; _host = host; }
            foreach (var c in All) c.Kick();
        }

        // ---- senders ----

        // Stream senders drop data while their socket is not connected: stale audio/pose/video is useless after a
        // reconnect, and seq restarts at 0 in every epoch (Hello is the first pose-channel message, seq 0).

        public void SendAudio(AudioChunk header, byte[] pcm, int pcmBytes)
        {
            if (Audio.State != LinkState.Connected) return;
            header.Seq = Audio.NextSeq();
            Audio.SendBinary(header, pcm, pcmBytes);
        }

        public void SendVideo(VideoFrame header, byte[] jpeg, int jpegBytes)
        {
            if (Video.State != LinkState.Connected) return;
            header.Seq = Video.NextSeq();
            Video.SendBinary(header, jpeg, jpegBytes);
        }

        public void SendPose(PoseSample pose)
        {
            if (Pose.State != LinkState.Connected) return;
            pose.Seq = Pose.NextSeq();
            Pose.SendJson(pose);
        }

        public void SendInput(InputEvent input)
        {
            if (Pose.State != LinkState.Connected) return;
            input.Seq = Pose.NextSeq();
            Pose.SendJson(input);
        }

        /// <summary>Client -> hub command (SetMode, SetVolumes, AckAlert) on /events.</summary>
        public void SendCommand(object command) => Events.SendJson(command);

        // ---- receivers (network thread) ----

        void OnEventText(string json)
        {
            var ev = HubEventParser.Parse(json);
            if (ev != null) IncomingEvents.Enqueue(ev);
        }

        void OnAudioOutBinary(byte[] buf, int count)
        {
            if (count < 4) return;
            int n = buf[0] | (buf[1] << 8) | (buf[2] << 16) | (buf[3] << 24);
            if (n <= 0 || 4 + n > count) return;
            AudioOut hdr;
            try { hdr = JsonConvert.DeserializeObject<AudioOut>(Encoding.UTF8.GetString(buf, 4, n)); }
            catch { return; }
            if (hdr == null) return;
            AudioOutReceived?.Invoke(hdr, buf, 4 + n, count - 4 - n);
        }
    }

    public static class HubEventParser
    {
        public static object Parse(string json)
        {
            try
            {
                var o = JObject.Parse(json);
                switch ((string)o["type"])
                {
                    case "sources.updated": return o.ToObject<SourcesUpdated>();
                    case "state.changed": return o.ToObject<StateChanged>();
                    case "target.updated": return o.ToObject<TargetUpdated>();
                    case "alert.fired": return o.ToObject<AlertFired>();
                    case "replay.ready": return o.ToObject<ReplayReady>();
                    case "metrics": return o.ToObject<Metrics>();
                    default: return null;
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("[Hub] bad event: " + e.Message);
                return null;
            }
        }
    }
}
