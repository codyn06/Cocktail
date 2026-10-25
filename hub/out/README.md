# /audio/out: the chosen voice plus the room (M2)

`separate/__main__.py` installs this next to `/events`. Feed: `worker.listeners.append(audio_out.on_chunks)`.
`GET /audio/out/status` shows the chosen id, its slots, volumes, queue, drops, underruns and last frame levels.

## Choosing (hub/events/selection.py)

The headset's select / stop / next / prev arrive as `InputEvent` on `/ingest/pose` (not as `/events`
commands). `IngestState.input_hooks` hands each one to `EventsState.on_input`.

- select with `source_id`: must be a published tracker id (`t3`). With a ray: the non-lost source whose
  bearing is nearest the ray yaw (`atan2(x, z)`), within 30 deg, else nothing changes. With neither: the
  head's forward direction is the ray.
- stop: nothing chosen. replay: not handled here.
- next / prev: order is world bearing ascending (left to right); sources without a bearing last, by id
  number; lost sources skipped; wraps. From nothing, next = first, prev = last.

**Continuity.** The choice is a Tracker id. The Tracker keeps one id per sound by world bearing + label
through slot swaps and pauses, and is what the headset's list shows, so `t3` means the same talker when the
separator calls them `s1` then `s2`. Per audio frame `Selection.route()` asks the Tracker which slot (and
duplicate slots folded into the same source) carries the id. Only a `tracking` source is routed: a
`last_seen` source's slot may already belong to another sound, so inner goes silent.
If the Tracker folds the chosen track into another, the choice follows (`reconcile`); if it is forgotten or
the pose epoch restarts, the choice is cleared.
Voice memory (`separate/voice_memory.py`, `--voice-memory`) is complementary, not the key: it only keeps
the two voice slots consistent over pauses with speaker embeddings (needs WavLM, covers no music or
residual, has no ids the headset knows). It reduces slot swaps the Tracker has to absorb.

## Wire format (matches quest AudioOutPlayer.cs)

Binary frame = `protocol.pack(AudioOut header, payload)`: 960 frames, 48 kHz, 2 channels, s16le interleaved,
one frame every 20 ms. Hub audio is 16 kHz and is upsampled 3x with a stateful FIR (about 1 ms delay).

- `layers` (default, what the Quest uses): channel 0 = inner (chosen source), channel 1 = outer (room =
  every other slot, so the chosen voice is not doubled). `target_bearing_deg` = the chosen source's world
  bearing (null when nothing is chosen or it is not audible).
- `stereo` (`/audio/out?layout=stereo`, glasses / simple clients): left/right, inner panned by
  `sin(bearing - head yaw)` with constant power (centred = -3 dB), room on both channels.

**Volumes.** The Quest applies inner/outer dB itself when it writes each frame (`SetVolumesDb`), and only
forwards `SetVolumes` to the hub if `forwardVolumesToHub` is on. So `layers` frames are unity gain and the hub
only stores the values (`EventsState.inner_db / outer_db`); applying them too would double the change.
`stereo` clients cannot apply them, so the hub does (20 ms linear ramp per change; hard clip at full scale).
`--out-gains-in-layers` turns hub gains on for `layers` for a client that does not apply them.

## Timing

Chunks wait `--out-settle-s` (0.3 s) before playing so the Tracker has consumed the same hop (locator
snapshot tick + delivery tick) and the route for that hop is current. Queue is capped at 2.5 s (oldest hops
dropped). A route change is a 20 ms crossfade. With nothing queued the hub sends nothing (the headset
re-primes) and counts an underrun.

## Tests

`uv run pytest tests/test_audio_out.py`: mock headset input, fake separated tones (one frequency per
slot), real ingest + events + tracker + mixer + websocket. No GPU, models or device.
