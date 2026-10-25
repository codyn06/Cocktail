"""Perception-models guard.

`import sam_audio` imports `core.audio_visual_encoder.*` at module level (verified), so importing
those modules is allowed. What Phase 2 must never do is INSTANTIATE the Phase 3 discovery /
correspondence models (PEAudioFrame, PEAudioVisual). SAMAudio.__init__ does build PEAudioFrame
when `span_predictor` is configured, which is why replay.sam_audio sanitizes the config.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sam_audio")
core = pytest.importorskip("core.audio_visual_encoder")

HUB = Path(__file__).resolve().parents[2]
FORBIDDEN = ("PEAudioFrame", "PEAudioVisual")


@pytest.fixture
def tripwire(monkeypatch):
    """Make any construction of the Phase 3 models fail the test."""
    def trip(name):
        def _boom(*a, **k):
            raise AssertionError(f"Phase 2 instantiated {name} (Phase 3 model)")
        return _boom

    for name in FORBIDDEN:
        cls = getattr(core, name)  # fails loudly if upstream renames them: the guard must stay real
        monkeypatch.setattr(cls, "__init__", trip(name))
        if hasattr(cls, "from_config"):
            monkeypatch.setattr(cls, "from_config", classmethod(lambda c, *a, _n=name, **k: trip(_n)()))
        if hasattr(cls, "from_pretrained"):
            monkeypatch.setattr(cls, "from_pretrained", classmethod(lambda c, *a, _n=name, **k: trip(_n)()))


def test_phase2_modules_import_without_instantiating_pe_models():
    """Fresh interpreter (no reload: reloading av_scene.runtime would redefine its lease/exception)."""
    script = "\n".join([
        "import core.audio_visual_encoder as c",
        "def boom(*a, **k): raise AssertionError('PE model instantiated')",
        "for n in ('PEAudioFrame', 'PEAudioVisual'):",
        "    getattr(c, n).__init__ = boom",
        "import av_scene.runtime, av_scene.preflight, replay.sam_audio, vision.sam3_tracker",
        "print('ok')",
    ])
    env = {**os.environ, "PYTHONPATH": str(HUB)}
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", script], cwd=HUB, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-800:]


def test_phase2_adapter_path_never_builds_pe_models(tripwire, tmp_path):
    """Run the mocked extraction path (real processor, fake model) with the tripwire armed."""
    from types import SimpleNamespace

    import numpy as np
    import torch

    from replay import sam_audio as sa
    from sam_audio import SAMAudioProcessor

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def separate(self, batch, **kw):
            n = int(batch.wav_sizes[0])
            return SimpleNamespace(target=[torch.zeros(n)], residual=[torch.zeros(n)], noise=None)

    bundle = sa.SamAudioBundle(Model(), SAMAudioProcessor(audio_hop_length=1920, audio_sampling_rate=48_000), "fake")
    sa.extract_stem(bundle, audio=np.zeros(4800, np.float32), sample_rate=48_000,
                    frames=torch.zeros((4, 3, 16, 16), dtype=torch.uint8), masks=[None] * 4,
                    source_id="g", out_dir=tmp_path)


def test_config_sanitiser_is_what_keeps_span_predictor_out():
    from replay import sam_audio as sa

    assert sa._sanitize_config({"span_predictor": "pe-a-frame-large"})["span_predictor"] is None
