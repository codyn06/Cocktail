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


# ----------------------------------------------------------------------------- SAM 3 (image + IoU)
H3, W3 = 40, 60


def _blob(y0, y1, x0, x1):
    m = np.zeros((H3, W3), bool)
    m[y0:y1, x0:x1] = True
    return m


def _scene(n=6):
    """Two people moving slowly; person B is missed on frame 2 (occlusion/detector miss)."""
    frames, per_frame = [], []
    for f in range(n):
        dets = [(_blob(5, 25, 4 + f, 20 + f), 0.9)]
        if f != 2:
            dets.append((_blob(10, 35, 36, 52), 0.6))
        per_frame.append(dets)
        frames.append(np.full((H3, W3, 3), f, np.uint8))
    return frames, per_frame


def _segmenter(per_frame):
    it = iter(per_frame)
    return lambda rgb: next(it)


def test_iou_tracker_keeps_identity_across_motion_and_a_missed_frame(tmp_path):
    frames, per_frame = _scene()
    res = tr.track_concept_images(_segmenter(per_frame), frames, "person", out_dir=tmp_path,
                                  frame_pts_us=[i * 40_000 for i in range(6)], run_tag="t")
    assert res.frames_seen == 6 and len(res.tracks) == 2                      # no identity switch / fragmentation
    by_len = sorted(res.tracks, key=lambda t: t.n_frames)
    assert [t.n_frames for t in by_len] == [5, 6]
    a = by_len[1]
    assert isinstance(a, MaskletTrack) and a.track_id.startswith("sam3_image.t.") and a.concept == "person"
    assert (a.image_height, a.image_width) == (H3, W3) and a.first_pts_us == 0 and a.last_pts_us == 200_000
    recs = read_track_jsonl(tmp_path / a.mask_file)
    assert [r.frame_idx for r in recs] == list(range(6)) and recs[0].rle.decode()[10, 10]
    assert ":" not in a.track_id and a.mask_file == f"masks/{a.track_id}.jsonl"   # Windows-safe


def test_tracker_does_not_merge_disjoint_people(tmp_path):
    frames, per_frame = _scene()
    res = tr.track_concept_images(_segmenter(per_frame), frames, out_dir=tmp_path, frame_pts_us=list(range(6)))
    masks = [read_track_jsonl(tmp_path / t.mask_file)[0].rle.decode() for t in res.tracks]
    assert not np.logical_and(masks[0], masks[1]).any()


def test_tracker_drops_duplicates_flicker_and_low_confidence(tmp_path):
    base = _blob(5, 25, 4, 20)
    per_frame = [[(base, 0.9), (base.copy(), 0.8), (_blob(30, 38, 50, 58), 0.2)], [(base, 0.9)], [(base, 0.9)]]
    res = tr.track_concept_images(_segmenter(per_frame), [np.zeros((H3, W3, 3), np.uint8)] * 3, out_dir=tmp_path,
                                  frame_pts_us=[0, 1, 2], min_confidence=0.3)
    assert len(res.tracks) == 1 and res.tracks[0].n_frames == 3               # duplicate merged, p=0.2 dropped
    res2 = tr.track_concept_images(_segmenter([[(base, 0.9)], [], []]), [np.zeros((H3, W3, 3), np.uint8)] * 3,
                                   out_dir=tmp_path / "b", frame_pts_us=[0, 1, 2], min_track_frames=2)
    assert res2.tracks == []                                                   # single-frame flicker removed


def test_tracker_handles_no_detections(tmp_path):
    res = tr.track_concept_images(lambda rgb: [], [np.zeros((H3, W3, 3), np.uint8)] * 2, out_dir=tmp_path,
                                  frame_pts_us=[0, 1])
    assert res.tracks == [] and res.frames_seen == 2


def test_sam3_image_wrapper_converts_state_to_numpy_masks():
    class P:
        def set_image(self, img):
            return {}

        def set_text_prompt(self, prompt, state):
            assert prompt == "person"
            m = torch.zeros(2, 1, H3, W3, dtype=torch.bool)
            m[0, 0, 5:9, 5:9] = True
            return {"masks": m, "scores": torch.tensor([0.8, 0.4])}

    model = torch.nn.Linear(1, 1)
    dets = tr.Sam3Image(model, P(), "person")(np.zeros((H3, W3, 3), np.uint8))
    assert [round(s, 1) for _, s in dets] == [0.8, 0.4] and dets[0][0].dtype == bool and dets[0][0].shape == (H3, W3)

    class Empty(P):
        def set_text_prompt(self, prompt, state):
            return {"masks": torch.zeros(0, 1, H3, W3, dtype=torch.bool), "scores": torch.zeros(0)}

    assert tr.Sam3Image(model, Empty(), "person")(np.zeros((H3, W3, 3), np.uint8)) == []


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
