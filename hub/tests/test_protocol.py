import pytest

from hub import protocol
from hub.protocol import AudioHeader, FrameError


def test_round_trip():
    header = AudioHeader(seq=7, epoch=2, pts_us=1_842_330_000, sample_rate=16_000, channels=1, ref_seq=3)
    data = protocol.pack(header, b"\x01\x00\x02\x00")
    assert len(data) == protocol.HEADER_SIZE + 4
    got, payload = protocol.unpack(data)
    assert got == header
    assert bytes(payload) == b"\x01\x00\x02\x00"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: b"XXXX" + d[4:], "magic"),
        (lambda d: d[:4] + b"\x09" + d[5:], "version"),
        (lambda d: d[:5] + b"\x07" + d[6:], "sample_format"),
        (lambda d: d + b"\x00", "whole"),
        (lambda d: d[:10], "shorter"),
    ],
)
def test_rejects_malformed(mutate, message):
    data = protocol.pack(AudioHeader(0, 0, 0, 16_000, 1), b"\x00\x00")
    with pytest.raises(FrameError, match=message):
        protocol.unpack(mutate(data))


def test_rejects_out_of_range_rate():
    with pytest.raises(FrameError, match="sample_rate"):
        protocol.unpack(protocol.pack(AudioHeader(0, 0, 0, 1_000, 1), b""))
