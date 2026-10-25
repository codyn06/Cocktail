"""Phase 2 orchestrator end to end with the two models FAKED (everything else is real: TorchCodec
decode, IoU tracker, polarity-corrected visual prompt, envelope gating, residuals, bundle).
Evidence: mocked models, not a hardware/model test."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("sam_audio")

from av_scene import phase2, preflight  # noqa: E402
from replay import sam_audio as sa  # noqa: E402
from sam_audio import SAMAudioProcessor  # noqa: E402
from vision import sam3_tracker as tr  # noqa: E402

pytestmark = pytest.mark.skipif(preflight.check_decode()["status"] != "ok",
                                reason="TorchCodec cannot decode here (FFmpeg shared DLLs missing)")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("clip") / "clip.mp4"
    assert preflight._synth_clip(p)
    return p


class FakeAudioModel(torch.nn.Module):
    """Returns a loud tone for the speaker's mask and silence when the visual prompt is the bystander's."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def separate(self, batch, **kw):
        n = int(batch.wav_sizes[0])
        vid = batch.masked_video[0]                       # already masked: bright pixels = retained target
        left_kept = float(vid[:, :, :, : vid.shape[-1] // 2].float().sum()) > 0
        self.calls.append("left" if left_kept else "right")
        t = torch.arange(n) / 48_000
        tone = 0.3 * torch.sin(2 * torch.pi * 300 * t) if left_kept else torch.zeros(n)
        return SimpleNamespace(target=[tone], residual=[torch.zeros(n)], noise=None)


@pytest.fixture
def fakes(monkeypatch):
    model = FakeAudioModel()

    def seg(rgb):
        h, w = rgb.shape[:2]
        left, right = np.zeros((h, w), bool), np.zeros((h, w), bool)
        left[h // 4: 3 * h // 4, w // 8: w // 3] = True
        right[h // 4: 3 * h // 4, 2 * w // 3: 7 * w // 8] = True
        return [(left, 0.9), (right, 0.8)]

    monkeypatch.setattr(tr, "load_sam3_image", lambda *a, **k: seg)
    monkeypatch.setattr(sa, "load_sam_audio", lambda **k: sa.SamAudioBundle(
        model, SAMAudioProcessor(audio_hop_length=1920, audio_sampling_rate=48_000), "fake/sam-audio@test"))
    return model


def test_two_pass_flow_then_full_bundle(clip, fakes, tmp_path):
    out = tmp_path / "av"
    # pass 1: no selector -> stops after tracking with a preview
    assert phase2.run(clip, out, run_id="t", device="cpu") is None
    assert (out / "_work" / "tracks_preview.png").is_file() and (out / "_work" / "tracks.json").is_file()
    assert fakes.calls == []                                    # SAM Audio not run before the speaker is chosen

    # pass 2: reuse tracks, speaker = the left person
    m = phase2.run(clip, out, run_id="t", speaker_xy=(0.2, 0.5), reuse_tracks=True, device="cpu", probe_bystanders=1)
    assert fakes.calls == ["left", "right"]                     # speaker first, then the bystander probe

    for f in ("manifest.json", "viewer.html", "metrics.json", "mixture.wav", "stems/source_spk0.wav", "residual.wav"):
        assert (out / f).is_file(), f
    assert not (out / "frames" / "_work").exists() and not (out / "_work" / "frames").exists()

    assert m.evidence_tier == "replayed" and "HARDCODED" in m.evidence_note
    assert [s.source_id for s in m.stems] == ["spk0"] and m.stems[0].sample_rate == 48_000
    assoc = m.associations[0]
    assert assoc.status == "matched" and assoc.components[0].value is None       # a mock, not evidence
    assert len(m.tracks) == 2 and assoc.track_id in {t.track_id for t in m.tracks}

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["speaker"]["active_frames"] > 5
    by = metrics["bystanders"]
    assert by["highlighted_frames"] == 0
    (probe,) = by["visual_prompt_probe"].values()
    assert probe["vs_speaker_stem_db"] < -60                    # the silent person's prompt yields a near-silent stem

    spk_state = json.loads((out / "activity" / f"track_{assoc.track_id}.json").read_text())
    assert spk_state["state"].count("speaking") == metrics["speaker"]["active_frames"]
    assert spk_state["evidence_tier"] == "replayed"
    other = next(t.track_id for t in m.tracks if t.track_id != assoc.track_id)
    assert set(json.loads((out / "activity" / f"track_{other}.json").read_text())["state"]) == {"idle"}

    _serve_and_fetch(out, assoc.track_id)

    names = [s["name"] for s in metrics["stages"]]
    assert names == ["sam3_image"] or names == ["sam_audio_small_tv"]   # vision was reused; only audio ran now


def _serve_and_fetch(out, track_id):
    """The Phase 1 viewer must serve a Phase 2 bundle (mono audio, no array calibration)."""
    import threading
    import urllib.request

    from render.viewer import make_server

    srv = make_server(out, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/"
    try:
        get = lambda rel: urllib.request.urlopen(base + rel, timeout=10).read()
        assert b"<html" in get("viewer.html").lower()
        man = json.loads(get("manifest.json"))
        assert man["assets"]["stem:spk0"] == "stems/source_spk0.wav" and len(get("stems/source_spk0.wav")) > 1000
        frames = json.loads(get("frames.json"))
        assert len(get(frames["files"][0])) > 100 and frames["annotated"] and len(get(frames["annotated"][0])) > 100
        assert get(man["assets"][f"masks:{track_id}"]).strip()
    finally:
        srv.shutdown()


def test_select_speaker_track_by_point_and_id():
    from vision.mask_store import MaskRecord, encode_mask

    def rec(x0, x1):
        m = np.zeros((10, 10), bool)
        m[:, x0:x1] = True
        return [MaskRecord(pts_us=i, frame_idx=i, rle=encode_mask(m)) for i in range(3)]

    records = {"a": rec(0, 4), "b": rec(6, 10)}
    assert phase2.select_speaker_track(records, xy=(0.8, 0.5)) == "b"
    assert phase2.select_speaker_track(records, track_id="a") == "a"
    with pytest.raises(SystemExit):
        phase2.select_speaker_track(records, xy=(0.5, 0.5))      # in the gap between people
    with pytest.raises(SystemExit):
        phase2.select_speaker_track(records, track_id="zzz")
