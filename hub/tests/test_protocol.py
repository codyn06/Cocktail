import re
from pathlib import Path

from pydantic import TypeAdapter

import protocol as p
from protocol import export


def test_binary_roundtrip():
    h = p.AudioChunk(seq=1, device_ts_us=5, epoch=0, sample_rate=48000, channels=2, frames=480)
    h2, payload = p.unpack(p.pack(h, b"\x00\x01"), p.AudioChunk)
    assert h2 == h and payload == b"\x00\x01"


def test_pose_channel_discriminates():
    ad = TypeAdapter(p.PoseChannelMessage)
    msg = ad.validate_python(
        {"type": "input", "seq": 0, "device_ts_us": 1, "epoch": 0, "action": "select",
         "ray_origin": [0, 1.6, 0], "ray_dir": [0, 0, 1], "modality": "head"}
    )
    assert isinstance(msg, p.InputEvent) and msg.action == p.Action.select


def test_hub_event_discriminates():
    ev = TypeAdapter(p.HubEvent).validate_json(
        '{"type":"target.updated","target_id":"t_7","selection_version":3,"source_id":"s_2",'
        '"kind":"person","assoc_conf":0.81,"state":"tracking","bearing_deg":-35}'
    )
    assert isinstance(ev, p.TargetUpdated) and ev.bearing_deg == -35


def test_generated_cs_is_current():
    """Fails if messages.py changed without re-running `python -m protocol.export`."""
    assert export.CS_PATH.read_text() == export.gen_cs()


def test_cs_has_every_field():
    cs = export.CS_PATH.read_text()
    for model in export.MODELS:
        for name in model.model_fields:
            assert re.search(rf'JsonProperty\("{name}"\)', cs), (model.__name__, name)
