import numpy as np
import pytest

from hub.audio import FRAME_SAMPLES, Framer, LoopbackPipeline, StreamingResampler
from hub.protocol import FMT_FLOAT32, AudioHeader


def sine(rate, seconds, hz=440.0):
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)[:, None]


@pytest.mark.parametrize("rate", [16_000, 44_100])
def test_chunked_resampling_matches_one_shot(rate):
    x = sine(rate, 0.5)
    whole = StreamingResampler(rate).process(x)
    r = StreamingResampler(rate)
    sizes = [1, 7, 320, 3, 441, 1000]
    parts, i = [], 0
    while i < len(x):
        n = sizes[len(parts) % len(sizes)]
        parts.append(r.process(x[i : i + n]))
        i += n
    np.testing.assert_allclose(np.concatenate(parts), whole, atol=1e-5)
    assert abs(len(whole) - len(x) * 48_000 / rate) <= 48_000 / rate + 1


def test_resampling_keeps_frequency():
    y = StreamingResampler(16_000).process(sine(16_000, 1.0))[:, 0]
    peak_hz = np.argmax(np.abs(np.fft.rfft(y))) * 48_000 / len(y)
    assert abs(peak_hz - 440) < 2


def test_same_rate_passes_through():
    x = sine(48_000, 0.1)
    assert StreamingResampler(48_000).process(x) is x


def test_framer_tags_newest_chunk():
    f = Framer()
    h = [AudioHeader(seq=s, epoch=1, pts_us=s * 1000, sample_rate=48_000, channels=2) for s in range(3)]
    f.push(np.zeros((600, 2), np.float32), h[0], 100)
    f.push(np.zeros((600, 2), np.float32), h[1], 200)  # frame 0 ends inside this chunk
    f.push(np.zeros((600, 2), np.float32), h[2], 300)
    frames = f.pop_frames()
    assert [(fr.ref_seq, fr.ref_recv_ns) for fr in frames] == [(1, 200)]
    assert len(frames[0].pcm) == FRAME_SAMPLES * 2 * 2
    f.push(np.zeros((600, 2), np.float32), AudioHeader(3, 1, 3000, 48_000, 2), 400)
    assert [fr.ref_seq for fr in f.pop_frames()] == [3]


def test_pipeline_mono_float32_to_stereo_int16():
    p = LoopbackPipeline()
    x = sine(48_000, 0.04)[:, 0]
    frames = p.process(AudioHeader(0, 1, 0, 48_000, 1, FMT_FLOAT32), x.astype("<f4").tobytes(), 0)
    assert len(frames) == 2
    pcm = np.frombuffer(frames[0].pcm, "<i2").reshape(-1, 2)
    np.testing.assert_array_equal(pcm[:, 0], pcm[:, 1])
    np.testing.assert_allclose(pcm[:, 0] / 32767, x[:FRAME_SAMPLES], atol=1e-4)


def test_pipeline_resets_on_new_epoch():
    p = LoopbackPipeline()
    half = np.zeros(FRAME_SAMPLES // 2 * 2, "<i2").tobytes()  # half a frame of stereo
    assert p.process(AudioHeader(0, 1, 0, 48_000, 2), half, 0) == []
    # a partial frame from the old epoch is dropped, not joined to the new one
    assert p.process(AudioHeader(0, 2, 0, 48_000, 2), half, 0) == []
