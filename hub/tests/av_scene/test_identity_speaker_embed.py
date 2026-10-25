"""Status/error paths of the WavLM SV adapter (mocked; real model test skipped = untested)."""
import pytest

from separate import speaker_embed as se


def test_status_packages_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(se.importlib.util, "find_spec", lambda m: None)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    st = se.wavlm_sv_status()
    assert not st["importable"] and not st["weights_cached"] and not st["ready"]
    with pytest.raises(RuntimeError, match="packages"):
        se.load_wavlm_sv()


def test_weights_not_cached_raises_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr(se.importlib.util, "find_spec", lambda m: object())
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert se.wavlm_sv_status()["importable"] and not se.wavlm_sv_status()["weights_cached"]
    with pytest.raises(RuntimeError, match="never downloads"):
        se.load_wavlm_sv()


def test_weights_cached_detected(monkeypatch, tmp_path):
    snap = tmp_path / "models--microsoft--wavlm-base-plus-sv" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert se.wavlm_sv_status()["weights_cached"]


@pytest.mark.skip(reason="untested: needs cached microsoft/wavlm-base-plus-sv weights")
def test_real_model():
    pass
