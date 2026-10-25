"""Phase 4 StageCache: checksummed publish, corruption quarantine, state-chained keys, single writer."""
import threading
import time

import pytest

from av_scene.cache import MARKER, CacheBusy, StageCache, code_fingerprint


def _fill(c, stage, k, payload=b"abc", meta=None, required=("a.bin",)):
    with c.begin(stage, k, meta=meta, required=required) as d:
        (d / "a.bin").write_bytes(payload)
        (d / "sub").mkdir()
        (d / "sub" / "b.bin").write_bytes(payload * 2)


def test_lookup_validates_manifest_and_quarantines_corruption(tmp_path):
    c = StageCache(tmp_path)
    _fill(c, "speech_chunk", "k1", meta={"n": 1})
    assert c.lookup("speech_chunk", "k1") is not None and c.meta("speech_chunk", "k1") == {"n": 1}
    (c.path("speech_chunk", "k1") / "a.bin").write_bytes(b"xyz")       # same size, different content
    assert c.lookup("speech_chunk", "k1") is None
    assert not c.path("speech_chunk", "k1").exists() and list((tmp_path / "speech_chunk").glob(".corrupt-k1-*"))
    _fill(c, "speech_chunk", "k1")                                      # recompute works
    (c.path("speech_chunk", "k1") / "sub" / "b.bin").unlink()
    assert c.lookup("speech_chunk", "k1") is None                       # missing file
    _fill(c, "speech_chunk", "k1")
    (c.path("speech_chunk", "k1") / "a.bin").write_bytes(b"truncated!")
    assert c.lookup("speech_chunk", "k1", verify=False) is None         # size mismatch caught without checksums


def test_required_file_missing_never_publishes(tmp_path):
    c = StageCache(tmp_path)
    with pytest.raises(FileNotFoundError):
        with c.begin("speech_chunk", "k", required=("state.json",)) as d:
            (d / "other").write_bytes(b"1")
    assert c.lookup("speech_chunk", "k") is None
    assert not any(p.name.startswith(".") for p in (tmp_path / "speech_chunk").iterdir())


def test_legacy_marker_without_manifest_is_not_a_verified_hit(tmp_path):
    c = StageCache(tmp_path)
    d = c.path("ingest", "legacy")
    d.mkdir(parents=True)
    (d / MARKER).write_text('{"stage": "ingest", "key": "legacy", "meta": {}}')
    assert c.is_complete("ingest", "legacy") and c.lookup("ingest", "legacy") is None


def test_keys_cover_fingerprint_models_seeds_and_incoming_state(tmp_path):
    c = StageCache(tmp_path)
    base = dict(inputs={"v": "h"}, config={"a": 1})
    full = dict(fingerprint="f1", models={"mf2": "r1"}, seeds=[1], state_in="s1")
    k = c.key("speech_chunk", **base, **full)
    assert k == c.key("speech_chunk", **base, **full)
    for kw in ({"fingerprint": "f2"}, {"models": {"mf2": "r2"}}, {"seeds": [2]}, {"state_in": "s2"}):
        assert c.key("speech_chunk", **base, **{**full, **kw}) != k


def test_code_fingerprint_changes_with_source(tmp_path):
    import importlib.util
    p = tmp_path / "m.py"
    p.write_text("x = 1\n")
    spec = importlib.util.spec_from_file_location("m_fp", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = code_fingerprint(m)
    p.write_text("x = 2\n")
    assert code_fingerprint(m) != a


def test_single_writer_lock_waiting_reader_and_dead_lock_theft(tmp_path):
    c = StageCache(tmp_path)
    started, release = threading.Event(), threading.Event()

    def writer():
        with c.begin("speech_chunk", "kk", required=("a.bin",)) as d:
            (d / "a.bin").write_bytes(b"1")
            started.set()
            release.wait(5)
    t = threading.Thread(target=writer)
    t.start()
    assert started.wait(5)
    with pytest.raises(CacheBusy):
        with c.begin("speech_chunk", "kk"):
            pass
    out = []

    def second():
        _, computed = c.get_or_compute("speech_chunk", "kk", lambda d, m: out.append("ran"), wait_s=5)
        out.append(computed)
    t2 = threading.Thread(target=second)
    t2.start()
    time.sleep(0.2)
    release.set()
    t.join(5)
    t2.join(5)
    assert out == [False]                         # waited for the writer, reused its entry, produced nothing itself
    lp = tmp_path / "speech_chunk" / ".lock-dead"
    lp.write_text('{"pid": 99999999, "t": 0}')    # owner process does not exist
    _fill(c, "speech_chunk", "dead")
    assert c.lookup("speech_chunk", "dead") is not None and not lp.exists()


def test_get_or_compute_skips_producer_on_valid_hit_and_reruns_after_corruption(tmp_path):
    c = StageCache(tmp_path)
    n = []

    def prod(d, m):
        n.append(1)
        (d / "a.bin").write_bytes(b"data")
        m["samples"] = 7
    p, computed = c.get_or_compute("speech_chunk", "g", prod, required=("a.bin",))
    assert computed and c.meta("speech_chunk", "g") == {"samples": 7}
    c.get_or_compute("speech_chunk", "g", prod, required=("a.bin",))
    assert len(n) == 1
    (p / "a.bin").write_bytes(b"dat!")
    c.get_or_compute("speech_chunk", "g", prod, required=("a.bin",))
    assert len(n) == 2


def test_new_chunk_stages_in_dag_without_changing_old_invalidation(tmp_path):
    c = StageCache(tmp_path)
    assert "speech_chunk" in c.downstream("visual_chunk") and "export" in c.downstream("event_window")
    assert set(c.downstream("vision")) == {"evidence", "association", "stems", "residual", "render"}


def test_long_run_extension_is_optional_and_old_manifests_still_load():
    from av_scene.schemas import LongRunInfo, RunManifest, TimelineSegment
    m = RunManifest(run_id="r", config_hash="h")
    assert m.long_run is None and RunManifest.from_json(m.to_json()).long_run is None
    m.long_run = LongRunInfo(switches={"peel": False})
    assert RunManifest.from_json(m.to_json()).long_run.switches == {"peel": False}
    seg = TimelineSegment(source_id="src_0", start_sample=0, end_sample=10, start_us=0, end_us=1,
                          association_status="matched", visibility="visible", stem_availability="unavailable",
                          refusal_reason="ambiguous identity")
    assert (seg.association_status, seg.stem_availability) == ("matched", "unavailable")   # independent fields
