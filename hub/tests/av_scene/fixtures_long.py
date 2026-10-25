"""Test import shim for eval/av_scene/long_fixtures.py (the eval dir is not a package).

    from fixtures_long import build_scene, FakeSeparator, ...      # with hub/tests/av_scene on sys.path
Loaded via importlib like av_scene.__main__._load_eval; every public name of long_fixtures is re-exported.
"""

import importlib.util
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[3] / "eval" / "av_scene"
_NAME = "cocktail_eval_av_scene_long_fixtures"


def _load():
    if _NAME in sys.modules:
        return sys.modules[_NAME]
    spec = importlib.util.spec_from_file_location(_NAME, _EVAL / "long_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load()
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
