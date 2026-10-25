import numpy as np

import protocol as p
from ingest import IngestState
from locate.live import LiveLocator
from mockclient.simulate import quat_yaw_pitch
from separate.live import BAND_CENTERS_HZ, StreamChunk


def chunk(sid, t0_us, levels, act=1.0):
    n = levels.shape[0]
    return StreamChunk(sid, "speech", 0.9, t0_us, np.full(16000, 0.01, np.float32), np.zeros((levels.shape[1], 16000)),
                       t0_us + 50_000 + np.arange(n) * 50_000, levels, np.full(n, act))


def pose(st, seq, ts, yaw, epoch=0):
    st.on_pose_channel(p.PoseSample(seq=seq, device_ts_us=ts, epoch=epoch, pos=(0, 1.6, 0),
                                    rot_xyzw=quat_yaw_pitch(yaw, 0)))


def test_waits_for_poses_then_publishes_sources():
    st = IngestState()
    loc = LiveLocator(st)
    lv = np.zeros((19, 2, len(BAND_CENTERS_HZ)))
    loc.on_chunks([chunk("s1", 0, lv)])
    assert loc.sources() == []                       # no head pose yet
    for k in range(60):
        pose(st, k, k * 20_000, 30 * np.sin(k / 10))
    loc.on_chunks([chunk("s1", 0, lv), chunk("s2", 0, lv, act=0.0)])
    src = {s.source_id: s for s in loc.sources()}
    assert set(src) == {"s1"} and src["s1"].label == "speech"  # s2 never heard: not a source
    assert src["s1"].last_observed_us == 50_000 + 18 * 50_000
    assert src["s1"].state == p.TrackState.tracking
    assert loc.est._poses[-1][0] == 59 * 0.02         # poses fed in device seconds


def test_reconnect_resets_estimator():
    st = IngestState()
    loc = LiveLocator(st)
    for k in range(10):
        pose(st, k, k * 20_000, 0.0)
    loc.on_chunks([chunk("s1", 0, np.zeros((19, 2, 5)))])
    old = loc.est
    pose(st, 0, 5_000, 0.0, epoch=1)
    loc.on_chunks([chunk("s1", 0, np.zeros((19, 2, 5)))])
    assert loc.est is not old and len(loc.est._poses) == 1


def test_silent_stream_goes_last_seen_then_lost():
    st = IngestState()
    loc = LiveLocator(st)
    for k in range(800):  # 16 s of poses
        pose(st, k, k * 20_000, 30 * np.sin(k / 10))
    lv = np.zeros((19, 2, len(BAND_CENTERS_HZ)))
    loc.on_chunks([chunk("s1", 0, lv, act=1.0)])            # heard in the first second
    first = loc.sources()[0]
    for t in (1, 2, 3):                                      # then silent
        loc.on_chunks([chunk("s1", t * 1_000_000, lv, act=0.0)])
    s = loc.sources()[0]
    assert s.last_observed_us == first.last_observed_us      # does not advance while silent
    assert s.state == p.TrackState.last_seen
    loc.on_chunks([chunk("s1", 12_000_000, lv, act=0.0)])
    assert loc.sources()[0].state == p.TrackState.lost


def test_headset_with_four_mic_channels_uses_the_two_the_locator_models():
    # the Quest app asks for 4 native mic channels first; the locator models 2. Extra channels used to raise
    # "levels shape (4, 5) != (2, 5)" in every hop, so no source was ever located.
    st = IngestState()
    loc = LiveLocator(st)
    for k in range(60):
        pose(st, k, k * 20_000, 30 * np.sin(k / 10))
    lv = np.zeros((19, 4, len(BAND_CENTERS_HZ)))
    lv[:, 2:, :] = 99.0                                # the channels beyond the first two must not be read
    loc.on_chunks([chunk("s1", 0, lv)])
    src = {s.source_id: s for s in loc.sources()}
    assert set(src) == {"s1"} and src["s1"].state == p.TrackState.tracking
