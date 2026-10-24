import numpy as np
from fastapi.testclient import TestClient

from hub import protocol
from hub.main import create_app
from hub.protocol import AudioHeader


def chunk(seq, rate=16_000, ms=20, epoch=1):
    pcm = np.round(0.3 * np.sin(np.arange(rate * ms // 1000)) * 32767).astype("<i2").tobytes()
    return protocol.pack(AudioHeader(seq, epoch, seq * ms * 1000, rate, 1), pcm)


def test_loopback_end_to_end():
    with TestClient(create_app()) as client:
        with client.websocket_connect("/audio/out") as out, client.websocket_connect("/ingest/audio") as ingest:
            for seq in range(10):
                ingest.send_bytes(chunk(seq))
            # 200 ms of 16 kHz input -> 9 full 20 ms frames (the resampler holds one sample)
            headers = []
            for _ in range(9):
                header, payload = protocol.unpack(out.receive_bytes())
                assert (header.sample_rate, header.channels, len(payload)) == (48_000, 2, 960 * 4)
                headers.append(header)
        assert [h.seq for h in headers] == list(range(9))
        assert [h.ref_seq for h in headers] == sorted(h.ref_seq for h in headers)
        assert all(h.pts_us == h.ref_seq * 20_000 for h in headers)
        stats = client.get("/stats").json()
        assert stats["chunks_in"] == 10
        assert stats["seq_gaps"] == 0
        assert stats["frames_out"] == 9
        assert stats["format"]["chunk_ms"] == 20


def test_counts_gaps_and_bad_messages():
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ingest/audio") as ingest:
            ingest.send_bytes(chunk(0))
            ingest.send_bytes(b"garbage")
            ingest.send_bytes(chunk(3))
        stats = client.get("/stats").json()
        assert stats["bad_messages"] == 1
        assert stats["seq_gaps"] == 2
        assert stats["frames_unheard"] == stats["frames_out"] > 0


def test_reconnect_is_not_out_of_order():
    with TestClient(create_app()) as client:
        for _ in range(2):
            with client.websocket_connect("/ingest/audio") as ingest:
                ingest.send_bytes(chunk(0))
                ingest.send_bytes(chunk(1))
        stats = client.get("/stats").json()
        assert (stats["ingest_connections"], stats["out_of_order"], stats["seq_gaps"]) == (2, 0, 0)


def test_text_message_closes_ingest():
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ingest/audio") as ingest:
            ingest.send_text("hello")
            msg = ingest.receive()
            assert msg["type"] == "websocket.close"
            assert msg["code"] == 1003
