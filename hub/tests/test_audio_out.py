"""M2: route the chosen voice to /audio/out. Mock headset and fake separated streams; no GPU,
models or device. Voices are told apart by frequency: each source is a pure tone."""

import asyncio
import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import protocol as p
from events import EventsState
from events.selection import cycle_order
from ingest import IngestState, create_app
from mockclient.simulate import quat_yaw_pitch
from out import AudioOutConfig, AudioOutState, ClientFormatter, Upsampler, install_audio_out
from separate.live import StreamChunk
from track import TrackConfig, Tracker

SR16, SR48 = 16000, 48000
FREQ = {"s1": 500, "s2": 1500, "s3": 3000}   # whole-Hz tones: a 0.5 s window holds whole cycles


def pose(ingest, ts, epoch=0, yaw=0):
    ingest.on_pose_channel(p.PoseSample(
        seq=ingest.stats["pose"].messages, device_ts_us=ts, epoch=epoch,
        pos=(0, 0, 0), rot_xyzw=quat_yaw_pitch(yaw, 0)))


def raw(slot, bearing, ts, label="speech"):
    return p.Source(source_id=slot, label=label, label_conf=1, level_db=-20,
                    bearing_deg=bearing, bearing_conf=1 if bearing is not None else 0,
                    last_observed_us=ts)


async def feed(events, mono, sources, epoch=0):
    fut = asyncio.get_running_loop().create_future()
    fut.set_result((epoch, sources))
    events._snapshot_task = fut
    await events.tick(mono)
    return events.latest.sources


def hop_chunks(hop, tones, amp=0.2):
    """One 1 s hop. tones: {slot: freq_hz or None for silence}."""
    n0 = hop * SR16
    t = (n0 + np.arange(SR16)) / SR16
    out = []
    for slot, f in tones.items():
        a = np.zeros(SR16, np.float32) if f is None else (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)
        out.append(StreamChunk(slot, "speech", 1.0, 1_000_000 + hop * 1_000_000, a,
                               np.zeros((1, SR16), np.float32), np.zeros(1), np.zeros((1, 1, 5)), np.ones(1)))
    return out


def tone_db(x, f, sr=SR48):
    """Level in dB of the f Hz component of x (whole cycles)."""
    n = np.arange(len(x))
    amp = 2 / len(x) * abs(np.sum(x * np.exp(-2j * np.pi * f * n / sr)))
    return 20 * math.log10(amp + 1e-12)


def decode(blobs):
    chans = []
    for b in blobs:
        h, payload = p.unpack(b, p.AudioOut)
        assert (h.sample_rate, h.channels, h.frames) == (48000, 2, 960) and len(payload) == 960 * 2 * 2
        chans.append(np.frombuffer(payload, "<i2").reshape(-1, 2).astype(np.float64) / 32768)
    return np.concatenate(chans), h


class Rig:
    """Real ingest + events + tracker + selection + mixer; scripted separator output."""

    def __init__(self, settle=0.0, **cfg):
        self.ingest = IngestState()
        pose(self.ingest, 1_000_000)
        self.events = EventsState(self.ingest, None, Tracker(TrackConfig(confirm_hits=1)))
        self.out = AudioOutState(self.events, AudioOutConfig(settle_s=settle, **cfg))
        self.mono, self.hop = 0.0, 0
        self.fmt = {}

    @property
    def ts(self):
        return 1_000_000 + int(self.mono * 1e6)

    async def observe(self, bearings):
        """bearings: {slot: bearing}. Advance one second of tracker time."""
        srcs = [raw(s, b, self.ts) for s, b in bearings.items()]
        got = await feed(self.events, self.mono, srcs)
        return got

    def ids(self):
        return {round(s.bearing_deg): s.source_id for s in self.events.latest.sources}

    def send(self, action, **kw):
        self.ingest.on_pose_channel(p.InputEvent(
            seq=self.ingest.stats["pose"].messages, device_ts_us=self.ts, epoch=0, action=action, **kw))

    def play(self, tones, layout="layers", frames=50):
        """Push one hop and pull its frames; returns (n,2) float audio and the last header."""
        self.out.mixer.push(hop_chunks(self.hop, tones), now=0.0)
        self.hop += 1
        fmt = self.fmt.setdefault(layout, ClientFormatter(layout, self.out.config))
        blobs = []
        for _ in range(frames):
            f = self.out.tick(now=self.mono + 100)  # far past settle
            assert f is not None
            blobs.append(fmt.format(f, self.events.inner_db, self.events.outer_db, self.events.head_yaw()))
        return decode(blobs)

    def volumes(self, inner_db, outer_db):
        self.events.command(p.SetVolumes(inner_db=inner_db, outer_db=outer_db).model_dump_json())


def steady(x):
    return x[12 * 960:37 * 960]   # 25 whole frames after ramps and filter start-up


async def two_sources(rig, a=-60, b=50, n=3):
    for _ in range(n):
        await rig.observe({"s1": a, "s2": b})
        rig.mono += 1
    return rig.ids()


# ---------------------------------------------------------------- route the chosen voice

def test_chosen_stream_arrives_and_the_others_do_not():
    async def go():
        rig = Rig()
        ids = await two_sources(rig)
        rig.send(p.Action.select, source_id=ids[50])          # s2, 1500 Hz
        tones = {"s1": FREQ["s1"], "s2": FREQ["s2"], "s3": FREQ["s3"]}
        audio, hdr = rig.play(tones)
        inner, outer = steady(audio[:, 0]), steady(audio[:, 1])
        assert hdr.layout == "layers" and hdr.target_bearing_deg == pytest.approx(50, abs=1)
        want = tone_db(inner, 1500)
        assert want == pytest.approx(20 * math.log10(0.2), abs=0.5)      # unity gain, same level
        for other in (500, 3000):
            assert tone_db(inner, other) < want - 40                       # others absent from inner
        assert tone_db(outer, 1500) < want - 40                            # chosen not doubled in the room
        assert tone_db(outer, 500) == pytest.approx(want, abs=0.5)         # room carries the rest
        assert tone_db(outer, 3000) == pytest.approx(want, abs=0.5)
    asyncio.run(go())


def test_choice_survives_slot_swap_and_a_pause():
    async def go():
        rig = Rig()
        ids = await two_sources(rig)
        rig.send(p.Action.select, source_id=ids[-60])         # voice A, on s1 at 500 Hz
        audio, _ = rig.play({"s1": 500, "s2": 1500})
        assert tone_db(steady(audio[:, 0]), 500) > tone_db(steady(audio[:, 0]), 1500) + 40
        # The separator swaps its slots: A (500 Hz, -60 deg) is now s2; B (1500 Hz, 50 deg) is s1.
        for _ in range(2):
            await rig.observe({"s2": -60, "s1": 50})
            rig.mono += 1
        assert rig.events.selection.chosen == ids[-60]
        audio, hdr = rig.play({"s2": 500, "s1": 1500})
        inner = steady(audio[:, 0])
        assert tone_db(inner, 500) > tone_db(inner, 1500) + 40
        assert hdr.target_bearing_deg == pytest.approx(-60, abs=1)
        # A pauses (silent slot), then comes back in yet another slot.
        await rig.observe({"s1": 50})
        rig.mono += 3
        await rig.observe({"s1": 50})
        rig.mono += 1
        await rig.observe({"s3": -60, "s1": 50})
        rig.mono += 1
        audio, _ = rig.play({"s3": 500, "s1": 1500})
        assert rig.events.selection.chosen == ids[-60]
        inner = steady(audio[:, 0])
        assert tone_db(inner, 500) > tone_db(inner, 1500) + 40
    asyncio.run(go())


def test_settle_delay_lets_the_route_catch_up_with_a_swap():
    async def go():
        rig = Rig(settle=0.3)
        ids = await two_sources(rig)
        rig.send(p.Action.select, source_id=ids[-60])
        rig.out.on_chunks(hop_chunks(0, {"s1": 500, "s2": 1500}))
        # the hop is in; before the swap is observed, a pull must wait for the settle delay
        assert rig.out.mixer.pull(now=rig.out.mixer._hops[0].ready - 0.01) is None
        rig.out.mixer.reset()
        # A is on s2 for this hop. The tracker sees that on the next ticks, before the hop plays.
        t0 = 100.0
        rig.out.mixer.push(hop_chunks(1, {"s2": 500, "s1": 1500}), now=t0)
        await rig.observe({"s2": -60, "s1": 50})
        frame = rig.out.mixer.pull(now=t0 + 0.3)
        assert frame is not None
        assert tone_db(frame.inner, 500, SR48) > tone_db(frame.inner, 1500, SR48) + 40
        assert frame.target_id == ids[-60]
    asyncio.run(go())


# ---------------------------------------------------------------- two volumes

def levels(audio, tone_hz):
    return tone_db(steady(audio[:, 0]), tone_hz)


def test_stereo_volume_change_moves_output_by_the_same_db():
    async def go():
        rig = Rig()
        ids = await two_sources(rig)
        rig.send(p.Action.select, source_id=ids[50])          # inner = s2 1500 Hz; room = s1 500 Hz
        pose(rig.ingest, rig.ts, yaw=50)                      # looking at it: centred, constant-power -3 dB
        tones = {"s1": 500, "s2": 1500}
        rig.volumes(0, -12)
        base = rig.play(tones, "stereo")[0]
        base_in, base_out = levels(base, 1500), levels(base, 500)
        assert base_in - base_out == pytest.approx(12 - 3.01, abs=0.5)  # outer 12 dB under, inner pan -3 dB
        for d in (-6.0, -3.0, +4.0):
            rig.volumes(d, -12)
            a = rig.play(tones, "stereo")[0]                             # settle the gain ramp
            a = rig.play(tones, "stereo")[0]
            assert levels(a, 1500) - base_in == pytest.approx(d, abs=0.5)
            assert levels(a, 500) - base_out == pytest.approx(0, abs=0.5)  # room untouched
        rig.volumes(0, -20)
        a = rig.play(tones, "stereo")[0]
        a = rig.play(tones, "stereo")[0]
        assert levels(a, 500) - base_out == pytest.approx(-8, abs=0.5)
        assert levels(a, 1500) - base_in == pytest.approx(0, abs=0.5)
    asyncio.run(go())


def test_room_only_total_rms_follows_outer_volume():
    async def go():
        rig = Rig()
        await two_sources(rig)                                 # nothing chosen: all of it is room
        rig.volumes(0, -10)
        a = rig.play({"s1": 500, "s2": 1500}, "stereo")[0]
        rms = lambda x: 20 * math.log10(np.sqrt(np.mean(steady(x[:, 0]) ** 2)))
        r0 = rms(a)
        rig.volumes(0, -22)
        a = rig.play({"s1": 500, "s2": 1500}, "stereo")[0]
        a = rig.play({"s1": 500, "s2": 1500}, "stereo")[0]
        assert rms(a) - r0 == pytest.approx(-12, abs=0.5)
    asyncio.run(go())


def test_quest_layers_are_unity_gain_because_the_headset_applies_volumes():
    async def go():
        rig = Rig()
        ids = await two_sources(rig)
        rig.send(p.Action.select, source_id=ids[50])
        rig.volumes(0, -12)
        a = rig.play({"s1": 500, "s2": 1500})[0]
        rig.volumes(-20, -30)                                  # hub stores it, must not apply it again
        b = rig.play({"s1": 500, "s2": 1500})[0]
        b = rig.play({"s1": 500, "s2": 1500})[0]
        assert tone_db(steady(b[:, 0]), 1500) == pytest.approx(tone_db(steady(a[:, 0]), 1500), abs=0.1)
        assert tone_db(steady(b[:, 1]), 500) == pytest.approx(tone_db(steady(a[:, 1]), 500), abs=0.1)
        assert (rig.events.inner_db, rig.events.outer_db) == (-20, -30)
    asyncio.run(go())


def test_layers_gain_opt_in_applies_db_per_channel():
    async def go():
        rig = Rig(apply_gains_in_layers=True)
        ids = await two_sources(rig)
        rig.send(p.Action.select, source_id=ids[50])
        rig.volumes(0, 0)
        base = rig.play({"s1": 500, "s2": 1500})[0]
        rig.volumes(-9, -15)
        rig.play({"s1": 500, "s2": 1500})
        a = rig.play({"s1": 500, "s2": 1500})[0]
        assert tone_db(steady(a[:, 0]), 1500) - tone_db(steady(base[:, 0]), 1500) == pytest.approx(-9, abs=0.5)
        assert tone_db(steady(a[:, 1]), 500) - tone_db(steady(base[:, 1]), 500) == pytest.approx(-15, abs=0.5)
    asyncio.run(go())


# ---------------------------------------------------------------- switching

def test_select_a_then_b_then_stop_plays_the_right_stream_each_time():
    async def go():
        rig = Rig()
        ids = await two_sources(rig)
        tones = {"s1": 500, "s2": 1500}

        def inner_tone(audio):
            x = steady(audio[:, 0])
            return {f: tone_db(x, f) for f in (500, 1500)}

        rig.send(p.Action.select, source_id=ids[-60])
        a = rig.play(tones)[0]
        lv = inner_tone(a)
        assert lv[500] > lv[1500] + 40
        rig.send(p.Action.select, source_id=ids[50])
        b = rig.play(tones)[0]
        lv = inner_tone(b)
        assert lv[1500] > lv[500] + 40
        rig.send(p.Action.stop)
        assert rig.events.selection.chosen is None
        c, hdr = rig.play(tones)
        assert np.abs(steady(c[:, 0])).max() < 1e-3                    # inner silent
        assert hdr.target_bearing_deg is None
        assert tone_db(steady(c[:, 1]), 500) > -20 and tone_db(steady(c[:, 1]), 1500) > -20  # room has both
        assert rig.events.selection.version == 3
    asyncio.run(go())


def test_switching_crossfades_without_a_click():
    async def go():
        rig = Rig()
        ids = await two_sources(rig)
        tones = {"s1": 500, "s2": 1500}
        rig.send(p.Action.select, source_id=ids[-60])
        first = rig.play(tones, frames=25)[0][:, 0]
        rig.send(p.Action.select, source_id=ids[50])
        second = rig.play(tones, frames=25)[0][:, 0]
        x = np.concatenate([first, second])
        # two 0.2-amplitude tones: a hard cut would jump by ~0.2 in one sample; a tone alone moves < 0.05
        assert np.abs(np.diff(x[12 * 960:])).max() < 0.1
    asyncio.run(go())


def test_next_and_prev_cycle_left_to_right_and_wrap():
    async def go():
        rig = Rig()
        for _ in range(3):
            await rig.observe({"s1": 50, "s2": -60, "s3": 10})
            rig.mono += 1
        ids = rig.ids()
        order = [ids[-60], ids[10], ids[50]]
        assert [s.source_id for s in cycle_order(rig.events.latest.sources)] == order
        seen = []
        for _ in range(4):
            rig.send(p.Action.next)
            seen.append(rig.events.selection.chosen)
        assert seen == [order[0], order[1], order[2], order[0]]
        seen = []
        for _ in range(3):
            rig.send(p.Action.prev)
            seen.append(rig.events.selection.chosen)
        assert seen == [order[2], order[1], order[0]]
        rig.send(p.Action.stop)
        rig.send(p.Action.prev)
        assert rig.events.selection.chosen == order[2]               # prev from nothing = last
    asyncio.run(go())


def test_next_prev_play_the_stream_of_each_source():
    async def go():
        rig = Rig()
        for _ in range(3):
            await rig.observe({"s1": -60, "s2": 50})
            rig.mono += 1
        tones = {"s1": 500, "s2": 1500}
        rig.send(p.Action.next)                                       # leftmost = s1
        x = steady(rig.play(tones)[0][:, 0])
        assert tone_db(x, 500) > tone_db(x, 1500) + 40
        rig.send(p.Action.next)                                       # then s2
        x = steady(rig.play(tones)[0][:, 0])
        assert tone_db(x, 1500) > tone_db(x, 500) + 40
        rig.send(p.Action.prev)                                       # back to s1
        x = steady(rig.play(tones)[0][:, 0])
        assert tone_db(x, 500) > tone_db(x, 1500) + 40
    asyncio.run(go())


def test_ray_and_forward_select_pick_the_nearest_source_and_unknown_ids_are_ignored():
    async def go():
        rig = Rig()
        ids = await two_sources(rig)
        yaw = math.radians(48)
        rig.send(p.Action.select, ray_dir=(math.sin(yaw), 0.0, math.cos(yaw)), modality="hand")
        assert rig.events.selection.chosen == ids[50]
        yaw = math.radians(-58)
        rig.send(p.Action.select, ray_dir=(math.sin(yaw), 0.0, math.cos(yaw)))
        assert rig.events.selection.chosen == ids[-60]
        rig.send(p.Action.select, ray_dir=(0.0, 0.0, 1.0))             # straight ahead: nothing within 30 deg
        assert rig.events.selection.chosen == ids[-60]
        rig.send(p.Action.select, source_id="t99")
        assert rig.events.selection.chosen == ids[-60]
        pose(rig.ingest, rig.ts, yaw=45)                               # no ray: where the head points
        rig.send(p.Action.select)
        assert rig.events.selection.chosen == ids[50]
    asyncio.run(go())


def test_chosen_source_going_quiet_is_silent_and_reselect_state_is_cleared_on_epoch_change():
    async def go():
        rig = Rig()
        ids = await two_sources(rig)
        rig.send(p.Action.select, source_id=ids[50])
        for _ in range(3):                                            # only s1 talks: chosen -> last_seen
            await rig.observe({"s1": -60})
            rig.mono += 1
        audio, hdr = rig.play({"s1": 500, "s2": 1500})
        assert np.abs(steady(audio[:, 0])).max() < 1e-3               # never the slot another source uses
        assert hdr.target_bearing_deg is None
        pose(rig.ingest, 2_000_000, epoch=1)                          # headset reconnected
        await rig.events.tick(rig.mono + 1)
        assert rig.events.selection.chosen is None
    asyncio.run(go())


def test_forgotten_choice_is_cleared():
    async def go():
        rig = Rig()
        rig.events.tracker = Tracker(TrackConfig(confirm_hits=1, hold_s=0.5, lost_s=1.0, forget_s=2.0))
        ids = await two_sources(rig, n=1)
        rig.send(p.Action.select, source_id=ids[50])
        for _ in range(4):
            await rig.observe({"s1": -60})
            rig.mono += 1
        assert rig.events.selection.chosen is None
    asyncio.run(go())


# ---------------------------------------------------------------- pieces

def test_upsampler_has_no_seams_between_hops_and_unity_passband():
    x = (0.3 * np.sin(2 * np.pi * 1000 * np.arange(SR16 * 2) / SR16)).astype(np.float32)
    whole = Upsampler()(x)
    up = Upsampler()
    split = np.concatenate([up(x[:SR16]), up(x[SR16:])])
    assert np.abs(whole - split).max() < 1e-6
    assert tone_db(whole[SR48 // 2:SR48 // 2 + 24000], 1000) == pytest.approx(20 * math.log10(0.3), abs=0.1)


def test_websocket_wire_format_matches_the_headset_player():
    ingest = IngestState()
    pose(ingest, 1_000_000)
    tracker = Tracker(TrackConfig(confirm_hits=1))
    events = EventsState(ingest, None, tracker)
    tracker.update(1_000_000, [raw("s1", -60, 1_000_000), raw("s2", 50, 1_000_000)])
    events.latest = p.SourcesUpdated(device_ts_us=1_000_000, sources=tracker.sources(1_000_000))
    app = create_app(ingest)
    out = install_audio_out(app, events, AudioOutConfig(settle_s=0.0))
    target = next(s.source_id for s in events.latest.sources if s.bearing_deg == 50)
    ingest.on_pose_channel(p.InputEvent(seq=1, device_ts_us=2, epoch=0, action=p.Action.select, source_id=target))
    with TestClient(app) as c:
        with c.websocket_connect(p.WS_AUDIO_OUT) as ws:
            out.on_chunks(hop_chunks(0, {"s1": 500, "s2": 1500}))
            blobs = [ws.receive_bytes() for _ in range(30)]
        audio, hdr = decode(blobs)
        assert hdr.layout == "layers" and hdr.target_bearing_deg == pytest.approx(50, abs=0.1)
        assert [p.unpack(b, p.AudioOut)[0].seq for b in blobs] == list(range(blobs and p.unpack(blobs[0], p.AudioOut)[0].seq,
                                                                           p.unpack(blobs[0], p.AudioOut)[0].seq + 30))
        x = audio[5 * 960:, 0]
        assert tone_db(x, 1500) > tone_db(x, 500) + 40
        st = json.loads(c.get("/audio/out/status").text)
        assert st["chosen"] == target and st["slots"] == ["s2"] and st["frames_out"] >= 30
        with c.websocket_connect(p.WS_AUDIO_OUT + "?layout=stereo") as ws:
            out.on_chunks(hop_chunks(1, {"s1": 500, "s2": 1500}))
            _, hdr = decode([ws.receive_bytes()])
            assert hdr.layout == "stereo"
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect(p.WS_AUDIO_OUT + "?layout=bogus") as ws:
                ws.receive_bytes()
