# Live tracked events

The existing `uv run --group gpu python -m separate` entry point installs this
service on the ingest app. `/sources` remains the raw locator view. `/tracks`
returns the latest tracked Source list. `/events` sends a protocol
`SourcesUpdated` immediately on connection and then every 100 ms, including
empty updates before ingest starts and state transitions after audio stops.

Each connection has one pending update, replaced by the newest tick if its
sender is slow. Sending and receiving run independently. Commands are validated
with the frozen `Command` TypeAdapter and logged by type. Valid values are stored
in `app.state.events.mode`, `inner_db`, `outer_db`, and `acked_alert_ids`.
Malformed JSON, unsupported commands, binary commands and non-finite volumes
increment `bad_commands` without closing the socket. No new error or reply
message is added to the frozen protocol. Volumes and mode are stored; `/audio/out`
(`hub/out`) reads the volumes. The headset's select / stop / next / prev arrive as
`InputEvent` on `/ingest/pose` and are applied by `events/selection.py` (see `hub/out/README.md`).

## Timing and identity

- The events task owns the Tracker. It calls `update()` on every tick, even when
  there are no new locator observations. Raw slot IDs never become public IDs.
- Each slot's `last_observed_us` is consumed at most once. Re-reading a loud
  cached hop cannot keep a source alive or count as another confirmation hit.
- Device time follows pose-channel timestamps and advances using hub monotonic
  elapsed time during gaps. Epoch changes reset tracks without reusing IDs and
  reject pending snapshots from the old epoch. Head yaw comes from the latest
  buffered pose under its lock; without a pose, `in_view` is false.
- `EventsState.max_observation_age_s` defaults to 5 seconds to accommodate
  separator overlap and inference latency independently of track hold time.
  Future and older observations are rejected. Tune this allowance for unusually
  slow processing; such processing is not validated by these tests.
- The unchanged Tracker stamps observations at the delivery tick, not capture
  time. Thus published `last_observed_us` and silence aging include processing
  latency. At the default one-hop-per-second cadence, three distinct bearing
  observations confirm a track; bearingless confirmation requires 15 distinct
  observations and retains the tracker's same-label suppression rules.
- A single background snapshot job copies the locator's hop references under a
  bounded lock acquisition and computes `sources()` on the detached copy. This
  adapter relies on LiveLocator's `latest`, `last_active_us`, and `_pose_epoch`
  fields; keep it aligned if that class changes. No audio arrays are copied.

## Headset-free verification

From `hub/`:

```powershell
uv run --group gpu pytest tests/test_events.py -v -s
uv run --group gpu pytest
```

The event tests cover slot swaps, pause/reacquisition, silence without new hops,
capture latency, clock resets, head yaw, bounded snapshots, blocked senders,
shutdown, command validation, and multiple real WebSocket clients. The replay
test generates a six-second two-voice-plus-music surrogate session, saves and
reloads it, and streams it with `mockclient.replay.replay` at real time. It uses
scripted separation and bearing estimates, the real ingest/worker/LiveLocator/
Tracker pipeline, and asserts 8–12 Hz, schema validity, stable IDs and lost state.
This is simulated/replayed transport evidence with mocked perception, not a
measurement of acoustic accuracy or hardware behavior.
