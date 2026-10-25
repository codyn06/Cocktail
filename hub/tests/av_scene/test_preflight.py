"""Preflight report logic with injected fakes: no GPU, no network, no token in output."""

from __future__ import annotations

import json

from av_scene import preflight


def _ok(**kw):
    return lambda: {"status": "ok", **kw}


def test_report_ok_only_when_every_required_capability_ok():
    fakes = {n: _ok() for n in preflight.REQUIRED}
    assert preflight.run_av_checks(checks=fakes)["ok"] is True
    fakes["decode"] = lambda: {"status": "skipped"}
    assert preflight.run_av_checks(checks=fakes)["ok"] is False  # unverified decode is not a pass


def test_crashing_probe_is_a_failed_capability_not_a_crashed_report():
    def boom():
        raise RuntimeError("kaboom")

    fakes = {n: _ok() for n in preflight.REQUIRED}
    fakes["cuda"] = boom
    rep = preflight.run_av_checks(checks=fakes)
    assert rep["cuda"]["status"] == "fail" and "kaboom" in rep["cuda"]["error"]
    assert rep["ok"] is False


def test_hf_missing_credentials_fails_clearly_without_leaking(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    out = preflight.check_hf_access()
    assert out["status"] == "fail" and out["token_present"] is False
    assert "hf auth login" in out["hint"]


def test_hf_report_never_contains_the_token(monkeypatch):
    import huggingface_hub

    secret = "hf_SECRET_TOKEN_VALUE"
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: secret)

    class FakeApi:
        def auth_check(self, repo_id, repo_type=None):
            if repo_id.endswith("sam3"):
                return None
            raise type("GatedRepoError", (Exception,), {})(f"401 {secret}")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    out = preflight.check_hf_access()
    assert out["repos"]["facebook/sam3"] == "ok"
    assert out["status"] == "fail"
    assert secret not in json.dumps(out)


def test_ffmpeg_dll_dir_from_env(monkeypatch, tmp_path):
    (tmp_path / "avcodec-62.dll").write_bytes(b"")
    (tmp_path / "avformat-62.dll").write_bytes(b"")
    monkeypatch.setenv("COCKTAIL_FFMPEG_BIN", str(tmp_path))
    info = preflight.configure_ffmpeg_dlls()
    assert info["status"] == "registered" and "avcodec-62.dll" in info["dlls"]
