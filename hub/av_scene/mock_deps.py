"""Mock providers for `python -m av_scene run --mock-scene SPEC`: synthetic scenes + fake models, CPU only.

Evidence tier is always `mocked/synthetic`. Scenes and fakes live in `eval/av_scene/long_fixtures.py` (a checkout is
needed). Events are not exercised in mock mode (the scenes contain only speech and the time-code tones the fakes
read); everything else, including identity under slot permutation, refusal, the cache, resume and export, runs the
real pipeline code.
"""

from __future__ import annotations

import contextlib

from .run import Providers


def mock_providers(scene, *, n_slots: int = 2, perm="alternate", hop_samples: int = 32000) -> Providers:
    from . import __main__ as cli
    fx = cli._load_eval("long_fixtures")
    models = fx.make_speech_models(scene, n_slots=n_slots, perm=perm, hop_samples=hop_samples)
    visual = fx.FakeVisualFrontEnd(scene)
    return Providers(
        visual=lambda: contextlib.nullcontext((visual, {"fake_visual": "long_fixtures-v1"})),
        speech=lambda: contextlib.nullcontext(models),
        events=None,
        visual_revisions={"fake_visual": "long_fixtures-v1"}, speech_revisions=dict(models.revisions),
        visual_empty_state=getattr(visual, "empty_state", lambda: {}))


def build_mock(spec: str, hop_samples: int):
    from . import __main__ as cli
    fx = cli._load_eval("long_fixtures")
    scene = fx.build_scene(spec)
    return scene, scene.source(), mock_providers(scene, hop_samples=hop_samples)
