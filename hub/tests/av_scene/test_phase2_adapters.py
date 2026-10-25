"""Phase 2 adapters with the heavy inference mocked. Labelled: mocked (not hardware-tested)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("sam_audio")

from av_scene.schemas import MaskletTrack, StemResult  # noqa: E402
from replay import sam_audio as sa  # noqa: E402
from sam_audio import SAMAudioProcessor  # noqa: E402
from vision import sam3_tracker as tr  # noqa: E402
from vision.mask_store import FrameTransform, encode_mask, read_track_jsonl  # noqa: E402


# ----------------------------------------------------------------------------- SAM 3
class FakePredictor:
    """Mimics Sam3VideoPredictor's request protocol: two objects over 3 frames."""

    def __init__(self, fail_in_propagation=False):
        self.calls, self.fail = [], fail_in_propagation

    def handle_request(self, req):
        self.calls.append(req["type"])
        if req["type"] == "start_session":
            return {"session_id": "abcdef1234567890"}
        return {"is_success": True}

    def handle_stream_request(self, req):
        assert req["type"] == "propagate_in_video"
        for f in range(3):
            if self.fail and f == 1:
                raise RuntimeError("propagation blew up")
            m = np.zeros((2, 8, 12), bool)
            m[0, 2:5, 2:5] = True
            m[1, 4:7, 6:10] = f > 0  # obj 8 absent-looking (empty) on frame 0
            yield {"frame_index": f, "outputs": {"out_obj_ids": np.array([7, 8]), "out_binary_masks": m,
                                                 "out_probs": np.array([0.9, 0.5])}}


def test_tracker_exports_rle_masklets_and_closes_session(tmp_path):
    p = FakePredictor()
    res = tr.track_concept(p, "clip.mp4", "person", out_dir=tmp_path, fps=25.0)
    assert p.calls == ["start_session", "add_prompt", "close_session"]
    assert {t.track_id for t in res.tracks} == {"sam3_video:abcdef12:7", "sam3_video:abcdef12:8"}
    t7 = next(t for t in res.tracks if t.track_id.endswith(":7"))
    assert isinstance(t7, MaskletTrack) and t7.concept == "person" and t7.n_frames == 3
    assert (t7.image_height, t7.image_width) == (8, 12) and t7.first_pts_us == 0 and t7.last_pts_us == 80_000
    assert t7.mean_confidence == pytest.approx(0.9)
    recs = read_track_jsonl(tmp_path / t7.mask_file)
    assert [r.frame_idx for r in recs] == [0, 1, 2]
    assert recs[0].rle.decode()[3, 3] and not recs[0].rle.decode()[0, 0]
    assert ":" not in t7.mask_file.split("/")[-1]  # Windows-safe filename


def test_tracker_closes_session_even_when_propagation_fails(tmp_path):
    p = FakePredictor(fail_in_propagation=True)
    with pytest.raises(RuntimeError, match="blew up"):
        tr.track_concept(p, "clip.mp4", out_dir=tmp_path, fps=25.0)
    assert p.calls[-1] == "close_session"


def test_tracker_uses_supplied_pts_and_confidence_floor(tmp_path):
    res = tr.track_concept(FakePredictor(), "clip.mp4", out_dir=tmp_path, frame_pts_us=[10, 20, 35],
                           min_confidence=0.6)
    assert [t.track_id.split(":")[-1] for t in res.tracks] == ["7"]  # obj 8 (p=0.5) dropped
    assert (res.tracks[0].first_pts_us, res.tracks[0].last_pts_us) == (10, 35)


def test_windows_guard_blocks_predictor_build(monkeypatch):
    monkeypatch.setattr(tr.sys, "platform", "win32")
    monkeypatch.delenv("COCKTAIL_ALLOW_SAM3_VIDEO", raising=False)
    with pytest.raises(RuntimeError, match="crashed"):
        tr.load_sam3_video_predictor()


# ----------------------------------------------------------------------------- SAM Audio
class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.seen = None

    def separate(self, batch, **kw):
        self.seen = (batch, kw)
        n = int(batch.wav_sizes[0])
        return SimpleNamespace(target=[torch.full((n,), 0.25)], residual=[torch.zeros(n)], noise=None)


def _bundle():
    proc = SAMAudioProcessor(audio_hop_length=1920, audio_sampling_rate=48_000)
    return sa.SamAudioBundle(FakeModel(), proc, "fake/sam-audio-small-tv@test")


def test_extract_stem_prompts_polarity_resample_and_schema(tmp_path):
    from ingest.media_io import read_wav

    b = _bundle()
    T, H, W = 10, 32, 32
    frames = torch.full((T, 3, H, W), 200, dtype=torch.uint8)
    m = np.zeros((H, W), bool)
    m[8:24, 8:24] = True
    masks = [encode_mask(m, FrameTransform.identity(H, W))] * T
    audio = np.random.default_rng(0).standard_normal(16_000).astype(np.float32) * 0.1  # 1 s at 16 kHz

    stem, prompts = sa.extract_stem(b, audio=audio, sample_rate=16_000, frames=frames, masks=masks,
                                    source_id="spk1", out_dir=tmp_path)

    batch, kw = b.model.seen
    assert batch.descriptions == ["person speaking"] and kw["reranking_candidates"] == 1 and not kw["predict_spans"]
    assert batch.audios.shape[-1] == 48_000                      # resampled 16 kHz -> 48 kHz
    assert prompts["anchors"] == [("+", 0.0, pytest.approx(1.0))] and prompts["mask_polarity"] == "retains_zero"
    vid = batch.masked_video[0]
    assert float(vid[:, :, 16, 16].min()) == 200.0 and float(vid[:, :, 0, 0].max()) == 0.0  # target kept, rest black
    assert isinstance(stem, StemResult) and stem.sample_rate == 48_000 and stem.n_samples == 48_000
    assert stem.errors == [] and stem.model_revision.startswith("fake/sam-audio-small-tv")
    data, rate = read_wav(tmp_path / stem.raw_path)
    assert rate == 48_000 and len(data) == 48_000 and np.allclose(data, 0.25)
    json.loads(stem.to_json())


def test_extract_stem_rejects_mask_frame_mismatch(tmp_path):
    frames = torch.zeros((4, 3, 16, 16), dtype=torch.uint8)
    with pytest.raises(ValueError):
        sa.extract_stem(_bundle(), audio=np.zeros(4800, np.float32), sample_rate=48_000, frames=frames,
                        masks=[None] * 3, source_id="x", out_dir=tmp_path)


def test_sanitize_config_drops_phase3_models():
    out = sa._sanitize_config({"span_predictor": "pe-a-frame-large", "visual_ranker": {"a": 1},
                               "text_ranker": {"b": 2}, "num_anchors": 3})
    assert out["span_predictor"] is None and out["visual_ranker"] is None and out["text_ranker"] is None
    assert out["num_anchors"] == 3


def test_resample_audio_mono_and_rate():
    out = sa.resample_audio(np.ones((2, 8000), np.float32), 8000)
    assert out.shape == (1, 48_000)
