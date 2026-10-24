import importlib.util
import json
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import TypeAdapter

import protocol as p

spec = importlib.util.spec_from_file_location(
    "mock_hub", Path(__file__).resolve().parents[2] / "quest" / "tools" / "mock_hub.py"
)
mock_hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock_hub)


def test_select_flow_and_audio_out():
    c = TestClient(mock_hub.app)
    ev = TypeAdapter(p.HubEvent)
    with c.websocket_connect(p.WS_EVENTS) as events, c.websocket_connect(p.WS_POSE_IN) as pose:
        pose.send_text(
            p.Hello(seq=0, device_ts_us=1, epoch=0, device="mock", mic_channels=1,
                    mic_raw=False, camera=False).model_dump_json()
        )
        pose.send_text(
            p.InputEvent(seq=1, device_ts_us=2, epoch=0, action=p.Action.select, source_id="s_2").model_dump_json()
        )
        seen = []
        for _ in range(12):
            m = ev.validate_json(events.receive_text())
            seen.append(m.type)
            if m.type == "state.changed" and m.state == p.HubState.playing:
                assert m.target_id == "t_s_2" and m.selection_version == 1
                break
        assert "sources.updated" in seen
        with c.websocket_connect(p.WS_AUDIO_OUT) as out:
            hdr, payload = p.unpack(out.receive_bytes(), p.AudioOut)
            assert len(payload) == 960 * 2 * 2 and hdr.target_bearing_deg is not None


def test_ingest_counts():
    c = TestClient(mock_hub.app)
    with c.websocket_connect(p.WS_AUDIO_IN) as ws:
        h = p.AudioChunk(seq=0, device_ts_us=5, epoch=0, sample_rate=16000, channels=1, frames=2)
        ws.send_bytes(p.pack(h, b"\x00\x00\x01\x00"))
    assert json.loads(c.get("/health").text)["audio"] >= 1
