# Hub

Laptop-side hub. Right now it does P0 step 1: an **unprocessed audio loopback** with
per-stage timing. Audio that arrives on `/ingest/audio` is converted to 48 kHz stereo,
cut into 20 ms frames and sent straight back on `/audio/out`. No models yet.

## Run

From the repo root (tested with Python 3.14 on Windows):

```
python -m venv .venv
.venv/Scripts/pip install -r hub/requirements.txt      # bin/pip on macOS/Linux
.venv/Scripts/uvicorn hub.main:app --port 8000
```

In a second terminal, the phone stand-in streams a 440 Hz tone and reports the round trip:

```
.venv/Scripts/python -m mock.phone_sim --seconds 10 --save loopback.wav
```

Tests: `.venv/Scripts/python -m pytest hub`

## Endpoints

| Endpoint | Direction | What |
|---|---|---|
| `WS /ingest/audio` | phone -> hub | binary audio frames (below), any of 16 / 44.1 / 48 kHz, int16 or float32, 1+ channels |
| `WS /audio/out` | hub -> phone | binary audio frames, always int16 stereo 48 kHz, 20 ms (960 samples) |
| `GET /stats` | | counters and p50 / p95 / max timings |
| `GET /health` | | `{"ok": true}` |

Only one ingest stream is live at a time: a new connection takes over (that is how a
reconnect works). Each `/audio/out` listener has a 100 ms queue; older frames are
dropped past that rather than letting delay build up.

## Audio frame format (proposed, v1)

Every binary message is a 32-byte little-endian header, then interleaved PCM. Defined in
[protocol.py](protocol.py).

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 4 | magic | `CKAU` |
| 4 | 1 | version | `1` |
| 5 | 1 | sample_format | `1` int16, `2` float32 |
| 6 | 2 | channels | |
| 8 | 8 | pts_us | glasses timestamp of the first sample, µs (signed) |
| 16 | 4 | sample_rate | Hz |
| 20 | 4 | seq | +1 per message |
| 24 | 4 | epoch | sender bumps it on every reconnect |
| 28 | 4 | ref_seq | out frames: newest input chunk in the frame; in: 0 |

Every field is aligned to its own size. Out frames carry the glasses timestamp of their
own first sample, so the phone can match what it plays to when the glasses heard it. A
gap in out `seq` means frames were dropped for that listener.

Any chunk size works; a long chunk (e.g. a 4096-sample iOS tap) arrives as a burst of
back-to-back 20 ms frames, so the phone needs its own small playout buffer. Text
messages on `/ingest/audio` close it with code 1003; malformed binary messages are
dropped and counted in `bad_messages`.

## What `/stats` measures

- `arrival_jitter`: how far each chunk's arrival gap is from the previous chunk's
  duration. Includes the sender's own pacing; on Windows the simulator's timer alone
  adds several ms.
- `process`: chunk received -> resampled and framed.
- `hub_latency`: newest input chunk in a frame received -> frame written to the socket.
  A lower bound on what the hub adds.
- `hub_hold`: chunk holding the frame's first sample received -> frame written. Adds the
  wait for the rest of the 20 ms frame (longer with long input chunks).
- `seq_gaps`, `out_of_order`, `bad_messages`, `frames_dropped`, `frames_unheard`
  (frames made while nobody was listening).

The simulator's `hub round trip` is send -> receive on the same machine. It excludes
the glasses, Bluetooth, the phone and A2DP playback; those need the real devices.

## Known limits

- Linear-interpolation resampling: fine for a loopback, not for quality work.
- Audio runs in the same process as the HTTP endpoints. The plan puts live audio in
  its own process; split it once video and models arrive.
- No video, motion, input or `/events` yet.
