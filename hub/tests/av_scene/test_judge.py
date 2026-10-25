"""Judge/CLAP ranker: import-safe, availability without loading. Real-model ranking is UNTESTED."""
import sys

import numpy as np
import pytest

from replay import judge as J


def test_import_safe_and_availability_loads_nothing(monkeypatch):
    for heavy in ("torch", "sam_audio", "laion_clap"):
        assert heavy not in J.__dict__                              # not imported at module level
    monkeypatch.setenv("HF_HOME", "/nonexistent-hf-home")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    before = set(sys.modules)
    av = J.availability()
    assert set(av) == {"judge", "clap"}
    assert all(set(v) >= {"installed", "cached", "ok"} for v in av.values())
    assert av["judge"]["cached"] is False and av["judge"]["ok"] is False
    assert not ({"sam_audio", "laion_clap"} & (set(sys.modules) - before))


def test_availability_sees_fake_cache(tmp_path, monkeypatch):
    snap = tmp_path / "models--facebook--sam-audio-judge" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    av = J.availability()
    assert av["judge"]["cached"] is True and av["clap"]["cached"] is False


def test_rank_unavailable_raises_not_loads(monkeypatch):
    monkeypatch.setattr(J, "availability", lambda: {
        "judge": {"installed": False, "cached": False, "ok": False},
        "clap": {"installed": False, "cached": False, "ok": False}})
    r = J.JudgeRanker("judge")
    with pytest.raises(J.RankerUnavailable):
        r.rank([np.zeros(10, np.float32)], "dog", 48000, mix=np.zeros(10, np.float32))
    with pytest.raises(J.RankerUnavailable):
        J.JudgeRanker("auto").rank([np.zeros(10, np.float32)], "dog", 48000)
    assert r.rank([], "dog", 48000) == []


def test_judge_needs_mix(monkeypatch):
    ok = {"installed": True, "cached": True, "ok": True}
    monkeypatch.setattr(J, "availability", lambda: {"judge": ok, "clap": ok})
    with pytest.raises(J.RankerUnavailable, match="mix"):
        J.JudgeRanker("judge").rank([np.zeros(10, np.float32)], "dog", 48000)


@pytest.mark.skip(reason="UNTESTED: real judge/CLAP weights are not loaded in CI (12 GB / 1.8 GB)")
def test_real_models_untested():
    pass
