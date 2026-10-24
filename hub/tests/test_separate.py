import numpy as np
import pytest

import protocol as p
from ingest import IngestState
from separate import BAND_CENTERS_HZ, LiveSeparator, SeparationWorker
from separate.score import sisdr

SR = 16000
rng = np.random.default_rng(0)
N = SR * 8
V1 = rng.standard_normal(N).astype(np.float32) * np.repeat(rng.random(N // 1600) > 0.3, 1600)
V2 = np.sin(2 * np.pi * 440 * np.arange(N) / SR).astype(np.float32) * 0.5
MUSIC = np.sin(2 * np.pi * 3000 * np.arange(N) / SR).astype(np.float32) * 0.1
MIX = V1 + V2 + MUSIC


class OracleSeparator:
    """Returns the true voices for the window it is given, in a random order and scale."""
    n_out = 2

    def __init__(self):
        self.pos = 0
        self.flip = np.random.default_rng(1)

    def __call__(self, x):
        i = self.pos
        self.pos += SR  # windows advance by 1 s
        v = np.stack([V1[i : i + len(x)] * 3.0, V2[i : i + len(x)] * 0.5])
        return v[::-1] if self.flip.random() < 0.5 else v


def run(live, x):
    out = {sid: [] for sid in live.ids}
    chunks = []
    for st in range(0, len(x) - live.W + 1, live.H):
        cs = live.step(x[st : st + live.W], st * 1_000_000 // SR)
        chunks.append(cs)
        for c in cs:
            out[c.stream_id].append(c.audio)
    return {k: np.concatenate(v) for k, v in out.items()}, chunks


def test_overlap_add_keeps_order_and_scale():
    x = np.stack([MIX, MIX], axis=1)
    est, chunks = run(LiveSeparator(OracleSeparator()), x)
    n = len(est["s1"])
    assert n == 7 * SR  # 7 windows of 2 s over 8 s, one 1 s hop each
    # Stream identity stays fixed despite random order from the separator.
    first = "s1" if np.dot(est["s1"][:SR], V1[:SR]) > np.dot(est["s2"][:SR], V1[:SR]) else "s2"
    other = "s2" if first == "s1" else "s1"
    assert sisdr(est[first], V1[:n]) > 30 and sisdr(est[other], V2[:n]) > 30
    assert sisdr(est["s3"], MUSIC[:n]) > 30  # residual is the music


def test_chunk_fields_for_locate():
    x = np.stack([MIX, 0.5 * MIX], axis=1)  # right mic 6 dB quieter
    _, chunks = run(LiveSeparator(OracleSeparator()), x)
    c = chunks[3][0]
    assert c.mic_audio.shape == (2, SR) and c.audio.shape == (SR,)
    assert c.levels_db.shape == (len(c.level_ts_us), 2, len(BAND_CENTERS_HZ))
    assert c.start_ts_us == 3_000_000 and np.all(np.diff(c.level_ts_us) == 50_000)
    assert abs(np.median(c.levels_db[:, 0] - c.levels_db[:, 1]) - 6.02) < 0.5
    assert c.activity.shape == c.level_ts_us.shape and 0 <= c.activity.min() <= c.activity.max() <= 1


def test_worker_follows_ingest_ring():
    st = IngestState()
    worker = SeparationWorker(st, LiveSeparator(OracleSeparator()))
    assert not worker.poll()
    frames = SR // 50  # 20 ms chunks at 16 kHz
    got = []
    worker.listeners.append(got.append)
    for k in range(SR * 4 // frames):
        seg = MIX[k * frames : (k + 1) * frames]
        h = p.AudioChunk(seq=k, device_ts_us=k * 20_000, epoch=0, sample_rate=SR, channels=1, frames=frames)
        st.on_audio(h, (seg / 4 * 32767).astype("<i2").tobytes())
        while worker.poll():
            pass
    assert len(got) == 3  # windows ending at 2, 3 and 4 s
    assert [c[0].start_ts_us for c in got] == [0, 1_000_000, 2_000_000]


@pytest.mark.skipif(not __import__("importlib").util.find_spec("transformers"), reason="needs gpu group")
def test_coarse_labels():
    from separate.models import coarse_label
    assert coarse_label("Male speech, man speaking") == "speech"
    assert coarse_label("Smoke detector, smoke alarm") == "alarm"
    assert coarse_label("Violin, fiddle") == "music"


def _feed(st, worker, sig, k0=0):
    frames = SR // 50
    for k in range(len(sig) // frames):
        h = p.AudioChunk(seq=k0 + k, device_ts_us=(k0 + k) * 20_000, epoch=0, sample_rate=SR, channels=1,
                         frames=frames)
        st.on_audio(h, (sig[k * frames : (k + 1) * frames] / 4 * 32767).astype("<i2").tobytes())
        while worker.poll():
            pass


def test_listener_error_does_not_repeat_window():
    st = IngestState()
    worker = SeparationWorker(st, LiveSeparator(OracleSeparator()))
    seen = []

    def bad(chunks):
        seen.append(chunks[0].start_ts_us)
        raise ValueError("boom")
    worker.listeners.append(bad)
    _feed(st, worker, MIX[: SR * 4])
    assert seen == [0, 1_000_000, 2_000_000] and worker.listener_errors == 3


def test_silence_has_no_activity():
    class Quiet:
        n_out = 2

        def __call__(self, x):
            return np.stack([x * 0.5, x * 0.5])
    live = LiveSeparator(Quiet())
    x = (rng.standard_normal((2 * SR, 1)) * 1e-4).astype(np.float32)  # -80 dBFS hiss
    for c in live.step(x, 0):
        assert c.activity.max() == 0


def test_worker_notices_ring_reset():
    st = IngestState()
    worker = SeparationWorker(st, LiveSeparator(OracleSeparator()))
    _feed(st, worker, MIX[: SR * 3])
    st.audio.reset()  # e.g. a reconnect, while the worker holds an old position
    worker.live.sep.pos = 0
    _feed(st, worker, MIX[: SR * 3], k0=1000)
    assert len(worker.chunks) == 4 and worker.chunks[-1][0].start_ts_us == 1000 * 20_000 + 1_000_000
