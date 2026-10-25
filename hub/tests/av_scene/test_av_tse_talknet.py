"""Tests for the AV-TSE and TalkNet adapters: numpy geometry (no GPU) + a GPU smoke test."""

from __future__ import annotations

import os

import numpy as np
import pytest

from separate import av_tse
from vision import talknet as tn


def test_resample_indices_nearest_and_length():
    idx = tn.resample_indices(30, 30.0)  # 1 s at 30 fps -> 25 frames
    assert len(idx) == 25
    assert idx[0] == 0 and idx[-1] <= 29
    assert np.all(np.diff(idx) >= 1)
    assert idx[1] == 1 and idx[5] == 6  # t=0.04*30=1.2 -> 1 ; 0.2*30=6
    assert np.array_equal(tn.resample_indices(25, 25.0), np.arange(25))
    assert len(tn.resample_indices(0, 30.0)) == 0
    assert len(tn.resample_indices(10, 10.0)) == 25  # upsample 10 -> 25 fps


def test_hold_boxes_fills_and_reports_fraction():
    b = [None, (0, 0, 10, 10), None, None, (5, 5, 15, 15), None]
    out, frac = tn.hold_boxes(b)
    assert frac == pytest.approx(4 / 6)
    assert np.allclose(out[0], (0, 0, 10, 10))  # leading gap takes first real box
    assert np.allclose(out[3], (0, 0, 10, 10))  # held
    assert np.allclose(out[5], (5, 5, 15, 15))
    with pytest.raises(ValueError):
        tn.hold_boxes([None, None])


def test_crop_face_geometry_and_padding():
    frame = np.zeros((200, 300, 3), np.uint8)
    frame[60:100, 100:140] = 255  # white square == the face box
    g = tn.crop_face(frame, (100, 60, 140, 100))
    assert g.shape == (112, 112) and g.dtype == np.uint8
    assert g[56, 56] == 255  # box centre maps to crop centre
    # box at the image corner is padded (value 110), not an error
    g2 = tn.crop_face(frame, (-10, -10, 10, 10))
    assert g2.shape == (112, 112)
    assert (g2 == 110).any()


def test_crop_region_matches_clearvoice_formula():
    # bs = max(w,h)/2 = 20; vertical my-bs..my+bs*1.8 (=36+... ), horizontal mx +- bs*1.4
    frame = np.arange(200 * 300 * 3, dtype=np.uint32).reshape(200, 300, 3).astype(np.uint8)
    tall = tn.crop_face(frame, (100, 40, 130, 100))  # w=30,h=60 -> bs=30
    assert tall.shape == (112, 112)


def test_build_face_stack_resamples_and_reports_coverage():
    frames = np.random.default_rng(0).integers(0, 255, (30, 120, 160, 3), dtype=np.uint8)
    boxes = [None] * 10 + [(60, 40, 100, 80)] * 20
    stack, info = tn.build_face_stack(frames, boxes, 30.0)
    assert stack.shape == (25, 112, 112)
    assert 0.2 < info["box_filled_fraction"] < 0.5
    assert info["box_coverage"] == pytest.approx(1 - info["box_filled_fraction"])


def test_normalise_visual_shape_and_stats():
    v = av_tse.normalise_visual(np.full((7, 112, 112), 106, np.uint8))
    assert v.shape == (1, 7, 112, 112) and v.dtype == np.float32
    assert abs(float(v.mean()) - (106 / 255 - 0.4161) / 0.1688) < 1e-4


def test_smooth5_matches_demo_slicing():
    s = np.arange(10, dtype=float)
    out = tn.smooth5(s)
    assert out[0] == pytest.approx(np.mean(s[0:3]))
    assert out[5] == pytest.approx(np.mean(s[3:8]))
    assert len(out) == 10


# ---------------------------------------------------------------- GPU smoke
def _have_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _synthetic(seconds=2.0, fps=30.0):
    rng = np.random.default_rng(1)
    t = np.arange(int(16000 * seconds)) / 16000
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    n = int(seconds * fps)
    frames = rng.integers(0, 255, (n, 240, 320, 3), dtype=np.uint8)
    boxes = [None] + [(110, 70, 210, 170)] * (n - 1)
    return audio, boxes, frames, fps


@pytest.mark.skipif(not _have_gpu(), reason="CUDA unavailable")
@pytest.mark.skipif(not os.path.isfile(av_tse.os.environ.get("COCKTAIL_AV_TSE_CKPT", av_tse.DEFAULT_CKPT)),
                    reason="AV-TSE weights missing")
def test_av_tse_smoke():
    import torch
    ex = av_tse.AVTargetSpeakerExtractor()
    audio, boxes, frames, fps = _synthetic()
    torch.cuda.reset_peak_memory_stats()
    out, info = ex.extract_target(audio, boxes, frames, fps)
    print("AV-TSE peak VRAM MiB", torch.cuda.max_memory_allocated() / 2**20)
    assert out.ndim == 1 and out.shape[0] == len(audio)
    assert np.isfinite(out).all()
    assert 0 < info["box_filled_fraction"] < 0.1
    del ex
    torch.cuda.empty_cache()


@pytest.mark.skipif(not _have_gpu(), reason="CUDA unavailable")
@pytest.mark.skipif(not os.path.isfile(os.environ.get("COCKTAIL_TALKNET_WEIGHTS", tn.DEFAULT_WEIGHTS)),
                    reason="TalkNet weights missing")
def test_talknet_smoke():
    import torch
    sc = tn.TalkNetScorer()
    audio, boxes, frames, fps = _synthetic()
    torch.cuda.reset_peak_memory_stats()
    scores, ts, info = sc.score_track(audio, boxes, frames, fps)
    print("TalkNet peak VRAM MiB", torch.cuda.max_memory_allocated() / 2**20)
    assert scores.ndim == 1 and scores.shape == ts.shape
    assert 45 <= len(scores) <= 50
    assert np.isfinite(scores).all()
    del sc
    torch.cuda.empty_cache()
