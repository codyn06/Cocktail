import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from ingest.buffers import AudioRing, LatestFrame
from refine.contracts import CutUnavailable, LiveView, RefineConfig
from refine.snapshot import SlotRing, TappedFrames, install_frame_tap, make_cut_fn

SR = 16000


def H(ts):
    return SimpleNamespace(device_ts_us=ts)


def chunk(sid, start_us, audio):
    return SimpleNamespace(stream_id=sid, start_ts_us=start_us, audio=audio)


def test_tapped_window_order_bounds_and_drop():
    t = TappedFrames(seconds=2.0)
    for ts in (0, 100_000, 300_000, 200_000):
        t.put(H(ts), str(ts).encode())
    assert t.received == 4 and t.dropped == 3  # 2 unread replaced + 1 out of order
    assert [f[0] for f in t.ring] == [0, 100_000, 300_000]
    assert [f[0] for f in t.window(100_000, 300_000)] == [100_000, 300_000]
    t.put(H(5_000_000), b"x")
    assert [f[0] for f in t.window(0, 10_000_000)] == [5_000_000]  # older than 2 s evicted


def test_tapped_clock_restart_clears():
    t = TappedFrames()
    t.put(H(10_000_000), b"a")
    t.put(H(100_000), b"b")
    assert [f[0] for f in t.window(0, 99_000_000)] == [100_000]


def test_tapped_hard_cap():
    t = TappedFrames(seconds=1000, max_frames=5)
    for i in range(20):
        t.put(H(i * 1000), b"x")
    assert len(t.ring) == 5


def test_install_idempotent_and_counters():
    ing = SimpleNamespace(video=LatestFrame())
    ing.video.put(H(5), b"j")
    ing.video.put(H(1), b"k")
    tap = install_frame_tap(ing)
    assert ing.video is tap and tap.received == 2 and tap.dropped == 1 and tap.jpeg == b"j"
    assert install_frame_tap(ing) is tap


def test_slotring_contiguity_reset_copies():
    r = SlotRing(seconds=3)
    a = np.ones(SR, np.float32)
    for i in range(4):
        r.on_chunks([chunk("s1", i * 1_000_000, a * i), chunk("s2", i * 1_000_000, a)])
    start, w = r.latest_window(10)
    assert start == 1_000_000 and r.end_us == 4_000_000
    assert len(w["s1"]) == len(w["s2"]) == 3 * SR
    w["s1"][:] = 99
    assert r.latest_window(1)[1]["s1"][0] == 3
    r.on_chunks([chunk("s1", 9_000_000, a), chunk("s2", 9_000_000, a)])
    start, w = r.latest_window(10)
    assert start == 9_000_000 and len(w["s1"]) == SR
    r.reset()
    assert r.latest_window(1) is None and r.slot_ids() == []


def rig(video_fps=10, secs=4):
    ing = SimpleNamespace(audio=AudioRing(2, seconds=10), video=TappedFrames())
    ring = SlotRing()
    base = 1_000_000
    ramp = np.arange(secs * SR, dtype=np.float32)
    for i in range(secs):  # audio ring in 0.25 s pieces; device time offset by `base`
        for j in range(4):
            lo = i * SR + j * SR // 4
            x = np.stack([ramp[lo:lo + SR // 4], 3 * ramp[lo:lo + SR // 4]], 1)
            ing.audio.write(x, base + lo * 1_000_000 // SR)
        ring.on_chunks([chunk("s1", base + i * 1_000_000, ramp[i * SR:(i + 1) * SR])])
    for k in range(secs * video_fps):
        ing.video.put(H(base + k * 1_000_000 // video_fps), b"j")
    live = LiveView("t1", 1, 2, ing.audio.resets, ("s1",), True, base + secs * 1_000_000)
    return ing, ring, live


def cutter(ing, ring, label="speech", **kw):
    return make_cut_fn(ing, ring, lambda s: label, RefineConfig(**kw))


def test_reasons():
    ing, ring, live = rig()
    with pytest.raises(CutUnavailable, match="not audible"):
        cutter(ing, ring)(dataclasses.replace(live, slots=()), 1)
    with pytest.raises(CutUnavailable, match="not speech"):
        cutter(ing, ring, "music")(live, 1)
    with pytest.raises(CutUnavailable, match="audio reset"):
        cutter(ing, ring)(dataclasses.replace(live, audio_resets=7), 1)
    with pytest.raises(CutUnavailable, match="window too short"):
        cutter(ing, SlotRing())(live, 1)
    ing3, ring3, live3 = rig(video_fps=3)
    with pytest.raises(CutUnavailable, match=r"insufficient video \(3\.0 fps\)"):
        cutter(ing3, ring3)(live3, 1)
    ing.video = LatestFrame()
    with pytest.raises(CutUnavailable, match="no frame tap"):
        cutter(ing, ring)(live, 1)
    ing.audio = None
    with pytest.raises(CutUnavailable, match="no audio"):
        cutter(ing, ring)(live, 1)


def test_happy_aligned_copies_and_reset():
    ing, ring, live = rig()
    job = cutter(ing, ring)(live, 5)
    n = 4 * SR
    assert len(job.mix_16k) == n and len(job.slot_audio["s1"]) == n
    assert job.capture_end_us - job.capture_start_us == 4_000_000
    # mix = mean(ramp, 3*ramp) = 2*ramp, slot = ramp: sample-exact alignment
    assert np.allclose(job.mix_16k, 2 * job.slot_audio["s1"])
    assert job.key.source_id == "t1" and job.key.selection_version == 2
    assert len(job.frames) == 40 and job.frames[0][0] <= job.frames[-1][0]
    ing.audio.write(np.full((SR, 2), 7, np.float32), 50_000_000)
    ring.on_chunks([chunk("s1", 50_000_000, np.full(SR, 7, np.float32))])
    ing.video.put(H(50_000_000), b"z")
    assert np.allclose(job.mix_16k, 2 * job.slot_audio["s1"]) and job.slot_audio["s1"][0] == 0
    ing.audio.reset()
    with pytest.raises(CutUnavailable, match="audio reset"):
        cutter(ing, ring)(live, 6)


def test_window_floor_clamps_all_streams_consistently():
    ing, ring, live = rig()
    floor = live.now_us - 2_500_000  # base+1.5 s
    job = cutter(ing, ring)(dataclasses.replace(live, window_floor_us=floor), 1)
    assert floor <= job.capture_start_us <= floor + 1
    n = len(job.slot_audio["s1"])
    assert len(job.mix_16k) == n
    assert abs(job.capture_end_us - job.capture_start_us - n * 1_000_000 // SR) <= 1
    assert np.allclose(job.mix_16k, 2 * job.slot_audio["s1"])      # mixture and slot still sample-aligned
    assert job.slot_audio["s1"][0] >= 1.5 * SR - 1                    # ramp value = absolute sample index
    assert all(job.capture_start_us <= f[0] <= job.capture_end_us for f in job.frames)


def test_window_floor_leaving_too_little_is_unavailable():
    ing, ring, live = rig()
    with pytest.raises(CutUnavailable, match="window too short"):
        cutter(ing, ring)(dataclasses.replace(live, window_floor_us=live.now_us - 1_000_000), 1)
