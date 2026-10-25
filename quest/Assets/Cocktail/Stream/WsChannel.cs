using System.Buffers;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Cocktail.Protocol;
using Newtonsoft.Json;
using UnityEngine;

namespace Cocktail.Stream
{
    public enum LinkState { Idle, Connecting, Connected, Backoff }

    public enum DropPolicy { DropOldest, DropNewest }

    /// <summary>Bounded thread-safe queue of outgoing messages. Producers (any thread) never block.</summary>
    public sealed class OutQueue
    {
        public readonly struct Item
        {
            public readonly object Header;   // protocol object (serialised on the sender thread) or null
            public readonly byte[] Payload;  // binary payload or null
            public readonly int PayloadLen;
            public readonly bool Binary;     // true: 4-byte LE header length + JSON header + payload
            public Item(object header, byte[] payload, int payloadLen, bool binary)
            { Header = header; Payload = payload; PayloadLen = payloadLen; Binary = binary; }
        }

        readonly LinkedList<Item> _items = new LinkedList<Item>();
        readonly SemaphoreSlim _signal = new SemaphoreSlim(0);
        readonly int _capacity;
        readonly DropPolicy _policy;
        long _dropped;

        public OutQueue(int capacity, DropPolicy policy)
        { _capacity = capacity; _policy = policy; }

        public long Dropped => Interlocked.Read(ref _dropped);

        public int Count { get { lock (_items) return _items.Count; } }

        public void Enqueue(Item item)
        {
            bool added = false;
            lock (_items)
            {
                if (_items.Count >= _capacity)
                {
                    Interlocked.Increment(ref _dropped);
                    if (_policy == DropPolicy.DropNewest) return;
                    _items.RemoveFirst();
                }
                else added = true;
                _items.AddLast(item);
            }
            if (added) _signal.Release();
        }

        /// <summary>Put an item ahead of everything queued (used for the Hello that must come first).</summary>
        public void EnqueueFront(Item item)
        {
            lock (_items) _items.AddFirst(item);
            _signal.Release();
        }

        public async Task<Item> DequeueAsync(CancellationToken ct)
        {
            while (true)
            {
                await _signal.WaitAsync(ct).ConfigureAwait(false);
                lock (_items)
                {
                    if (_items.Count > 0)
                    {
                        var it = _items.First.Value;
                        _items.RemoveFirst();
                        return it;
                    }
                }
                // Count was consumed by a DropOldest replacement; wait again.
            }
        }

        public void Clear()
        {
            lock (_items) _items.Clear();
            while (_signal.CurrentCount > 0 && _signal.Wait(0)) { }
        }
    }

    /// <summary>
    /// One reconnecting WebSocket (one endpoint). Runs entirely on thread-pool tasks, so producers and the
    /// Unity main thread never wait on the network. Large frames (video) live on their own channel so they
    /// cannot delay audio.
    /// </summary>
    public sealed class WsChannel
    {
        public readonly string Name;
        public readonly string Path;
        public readonly OutQueue Queue;

        // ---- stats (read from any thread) ----
        public volatile LinkState State = LinkState.Idle;
        public long SentMsgs, SentBytes, RecvMsgs, RecvBytes, Reconnects;
        public volatile string LastError = "";
        long _seq = -1;
        long _epoch;
        long _connects;
        public long Epoch => Interlocked.Read(ref _epoch);
        public long Seq => Interlocked.Read(ref _seq);

        /// <summary>Clear stale queued data after every reconnect (true for streams, false for commands).</summary>
        public bool ClearQueueOnReconnect = true;

        /// <summary>Runs after each (re)connect, before queued messages are sent. Return false to abort the connection.</summary>
        public System.Func<WsChannel, CancellationToken, Task> OnConnected;
        /// <summary>Text frame received (network thread).</summary>
        public System.Action<string> OnText;
        /// <summary>Binary frame received (network thread). Buffer is reused: copy what you need.</summary>
        public System.Action<byte[], int> OnBinary;
        /// <summary>Connection state change (network thread).</summary>
        public System.Action<WsChannel, LinkState> OnState;

        readonly System.Func<string> _hostProvider;
        readonly int _port;
        readonly System.Func<long> _nextEpoch;
        CancellationTokenSource _lifetime;
        CancellationTokenSource _connection;
        readonly object _connLock = new object();
        Task _loop;

        public WsChannel(string name, string path, int capacity, DropPolicy policy,
                         System.Func<string> hostProvider, int port, System.Func<long> nextEpoch)
        {
            Name = name; Path = path;
            Queue = new OutQueue(capacity, policy);
            _hostProvider = hostProvider; _port = port; _nextEpoch = nextEpoch;
        }

        public long NextSeq() => Interlocked.Increment(ref _seq);

        public void Start()
        {
            if (_loop != null) return;
            _lifetime = new CancellationTokenSource();
            var ct = _lifetime.Token;
            _loop = Task.Run(() => RunAsync(ct));
        }

        public void Stop()
        {
            _lifetime?.Cancel();
            Kick();
            _loop = null;
        }

        /// <summary>Drop the current connection so the loop reconnects (used when the host changes).</summary>
        public void Kick()
        {
            lock (_connLock) _connection?.Cancel();
        }

        // ---- enqueue helpers (any thread) ----

        /// <summary>JSON text frame sent before everything already queued.</summary>
        public void SendJsonFirst(object message) =>
            Queue.EnqueueFront(new OutQueue.Item(message, null, 0, false));

        /// <summary>JSON text frame.</summary>
        public void SendJson(object message) =>
            Queue.Enqueue(new OutQueue.Item(message, null, 0, false));

        /// <summary>Binary frame: LE32 header length, JSON header, payload.</summary>
        public void SendBinary(object header, byte[] payload, int payloadLen) =>
            Queue.Enqueue(new OutQueue.Item(header, payload, payloadLen, true));

        void SetState(LinkState s)
        {
            State = s;
            try { OnState?.Invoke(this, s); } catch (System.Exception e) { Debug.LogException(e); }
        }

        async Task RunAsync(CancellationToken life)
        {
            int backoffMs = 250;
            while (!life.IsCancellationRequested)
            {
                string host = _hostProvider();
                var uri = new System.Uri($"ws://{host}:{_port}{Path}");
                using (var ws = new ClientWebSocket())
                {
                    CancellationTokenSource conn;
                    lock (_connLock)
                    {
                        conn = CancellationTokenSource.CreateLinkedTokenSource(life);
                        _connection = conn;
                    }
                    try
                    {
                        SetState(LinkState.Connecting);
                        ws.Options.KeepAliveInterval = System.TimeSpan.FromSeconds(10);
                        using (var connectTimeout = CancellationTokenSource.CreateLinkedTokenSource(conn.Token))
                        {
                            connectTimeout.CancelAfter(4000);
                            await ws.ConnectAsync(uri, connectTimeout.Token).ConfigureAwait(false);
                        }
                        long epoch = _nextEpoch();
                        Interlocked.Exchange(ref _epoch, epoch);
                        Interlocked.Exchange(ref _seq, -1); // seq restarts at 0 in every epoch
                        if (_connects++ > 0) Interlocked.Increment(ref Reconnects);
                        if (ClearQueueOnReconnect) Queue.Clear();
                        LastError = "";
                        backoffMs = 250;
                        Debug.Log($"[Hub:{Name}] connected {uri} epoch={epoch}");
                        if (OnConnected != null) await OnConnected(this, conn.Token).ConfigureAwait(false);
                        SetState(LinkState.Connected);

                        var sendTask = SendLoopAsync(ws, conn.Token);
                        var recvTask = ReceiveLoopAsync(ws, conn.Token);
                        await Task.WhenAny(sendTask, recvTask).ConfigureAwait(false);
                        conn.Cancel();
                        try { await Task.WhenAll(sendTask, recvTask).ConfigureAwait(false); } catch { /* reported below */ }
                    }
                    catch (System.OperationCanceledException) { /* stopped, kicked or timed out */ }
                    catch (System.Exception e)
                    {
                        LastError = e.GetType().Name + ": " + e.Message;
                    }
                    finally
                    {
                        lock (_connLock) { _connection = null; }
                        conn.Dispose();
                        try
                        {
                            // a dead link (wifi gone, hub killed) must not hold up the reconnect: give the goodbye one second, then drop the socket
                            if (ws.State == WebSocketState.Open)
                                using (var closeTimeout = new CancellationTokenSource(1000))
                                    await ws.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "", closeTimeout.Token).ConfigureAwait(false);
                        }
                        catch { try { ws.Abort(); } catch { } }
                    }
                }

                if (life.IsCancellationRequested) break;
                SetState(LinkState.Backoff);
                if (LastError.Length > 0) Debug.LogWarning($"[Hub:{Name}] disconnected: {LastError}; retry in {backoffMs} ms");
                try { await Task.Delay(backoffMs, life).ConfigureAwait(false); } catch (System.OperationCanceledException) { break; }
                backoffMs = System.Math.Min(backoffMs * 2, 3000);
            }
            SetState(LinkState.Idle);
        }

        async Task SendLoopAsync(ClientWebSocket ws, CancellationToken ct)
        {
            var pool = ArrayPool<byte>.Shared;
            while (!ct.IsCancellationRequested)
            {
                var item = await Queue.DequeueAsync(ct).ConfigureAwait(false);
                StampEpoch(item.Header);
                string json = JsonConvert.SerializeObject(item.Header);
                if (!item.Binary)
                {
                    var bytes = Encoding.UTF8.GetBytes(json);
                    await ws.SendAsync(new System.ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, ct).ConfigureAwait(false);
                    Interlocked.Increment(ref SentMsgs);
                    Interlocked.Add(ref SentBytes, bytes.Length);
                }
                else
                {
                    int hlen = Encoding.UTF8.GetByteCount(json);
                    int total = 4 + hlen + item.PayloadLen;
                    var buf = pool.Rent(total);
                    try
                    {
                        buf[0] = (byte)hlen; buf[1] = (byte)(hlen >> 8); buf[2] = (byte)(hlen >> 16); buf[3] = (byte)(hlen >> 24);
                        Encoding.UTF8.GetBytes(json, 0, json.Length, buf, 4);
                        if (item.PayloadLen > 0) System.Buffer.BlockCopy(item.Payload, 0, buf, 4 + hlen, item.PayloadLen);
                        await ws.SendAsync(new System.ArraySegment<byte>(buf, 0, total), WebSocketMessageType.Binary, true, ct).ConfigureAwait(false);
                        Interlocked.Increment(ref SentMsgs);
                        Interlocked.Add(ref SentBytes, total);
                    }
                    finally { pool.Return(buf); }
                }
            }
        }

        async Task ReceiveLoopAsync(ClientWebSocket ws, CancellationToken ct)
        {
            var buffer = new byte[64 * 1024];
            while (!ct.IsCancellationRequested && ws.State == WebSocketState.Open)
            {
                int count = 0;
                WebSocketReceiveResult r;
                do
                {
                    if (count == buffer.Length) System.Array.Resize(ref buffer, buffer.Length * 2);
                    r = await ws.ReceiveAsync(new System.ArraySegment<byte>(buffer, count, buffer.Length - count), ct).ConfigureAwait(false);
                    if (r.MessageType == WebSocketMessageType.Close) return;
                    count += r.Count;
                } while (!r.EndOfMessage);

                Interlocked.Increment(ref RecvMsgs);
                Interlocked.Add(ref RecvBytes, count);
                try
                {
                    if (r.MessageType == WebSocketMessageType.Text) OnText?.Invoke(Encoding.UTF8.GetString(buffer, 0, count));
                    else OnBinary?.Invoke(buffer, count);
                }
                catch (System.Exception e) { Debug.LogException(e); }
            }
        }

        void StampEpoch(object header)
        {
            long e = Epoch;
            switch (header)
            {
                case AudioChunk m: m.Epoch = e; break;
                case VideoFrame m: m.Epoch = e; break;
                case PoseSample m: m.Epoch = e; break;
                case InputEvent m: m.Epoch = e; break;
                case Hello m: m.Epoch = e; break;
            }
        }
    }
}
