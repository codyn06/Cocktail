"""Mask polarity probe against the REAL upstream `SAMAudioProcessor.mask_videos` (no weights).

Finding recorded in replay/sam_audio.py: upstream computes `video * mask.eq(0)`, so a True mask
BLACKS OUT the target. `apply_visual_prompt_mask` must retain it whatever the helper does.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("sam_audio")

from replay import sam_audio as sa  # noqa: E402
from sam_audio import SAMAudioProcessor  # noqa: E402

T, H, W = 4, 16, 16


@pytest.fixture(scope="module")
def processor():
    return SAMAudioProcessor(audio_hop_length=1920, audio_sampling_rate=48_000)


def _white_and_centre_mask():
    video = torch.ones(T, 3, H, W)
    mask = torch.zeros(T, 1, H, W, dtype=torch.bool)
    mask[:, :, 4:12, 4:12] = True
    return video, mask


def test_upstream_helper_blacks_out_true_pixels(processor):
    """The probe itself: raw upstream behaviour with a True-centre mask."""
    video, mask = _white_and_centre_mask()
    out = processor.mask_videos([video], [mask])[0]
    assert torch.all(out[:, :, 8, 8] == 0.0), "centre (mask=True) is blacked out by the upstream helper"
    assert torch.all(out[:, :, 0, 0] == 1.0), "edge (mask=False) is kept by the upstream helper"
    assert sa.probe_mask_polarity(processor) == "retains_zero"


def test_wrapper_retains_foreground_and_removes_background(processor):
    video, mask = _white_and_centre_mask()
    out = sa.apply_visual_prompt_mask(video, mask, processor)
    assert out.shape == video.shape
    assert torch.all(out[:, :, 4:12, 4:12] == 1.0)           # foreground preserved
    bg = ~mask.expand_as(out)
    assert torch.all(out[bg] == 0.0)                          # background removed


def test_wrapper_accepts_3d_mask_and_uint8_video(processor):
    video = torch.full((T, 3, H, W), 255, dtype=torch.uint8)
    mask = torch.zeros(T, H, W, dtype=torch.bool)
    mask[:, 6:10, 6:10] = True
    out = sa.apply_visual_prompt_mask(video, mask, processor)
    assert out.dtype == torch.uint8
    assert int(out[0, 0, 8, 8]) == 255 and int(out[0, 0, 0, 0]) == 0


def test_wrapper_inverts_only_when_helper_retains_zero():
    """If upstream is ever fixed to retain True pixels, the wrapper must stop inverting."""
    class Fixed:
        def mask_videos(self, videos, masks):
            return [v * m.eq(1) for v, m in zip(videos, masks)]

    video, mask = _white_and_centre_mask()
    sa._POLARITY_CACHE.pop(Fixed, None)
    out = sa.apply_visual_prompt_mask(video, mask, Fixed())
    assert torch.all(out[:, :, 4:12, 4:12] == 1.0) and torch.all(out[:, :, 0, 0] == 0.0)


def test_wrapper_rejects_misaligned_inputs(processor):
    video, mask = _white_and_centre_mask()
    with pytest.raises(ValueError):
        sa.apply_visual_prompt_mask(video, mask[:-1], processor)          # frame count
    with pytest.raises(ValueError):
        sa.apply_visual_prompt_mask(video, mask[..., :-1], processor)     # width
    with pytest.raises(ValueError):
        sa.apply_visual_prompt_mask(video[:, :1], mask, processor)        # channels
