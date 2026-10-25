"""End to end on CPU fakes: visual -> speech -> events -> residual -> long bundle, plus warm cache and crash resume.

Evidence: mocked/synthetic. No model, GPU or device is involved; nothing here validates real-model behavior.
"""
import contextlib
import json
import shutil
from pathlib import Path

import fakes_chunked as fk
import numpy as np
import pytest

from av_scene.config import load_config
from av_scene.event_scope import EventAdapters
from av_scene.run import Providers, run_pipeline
from av_scene.schemas import AudioIndex, RunManifest, TimelineIndex, TimelinePageDoc
from replay.chunked import InjectedCrash


def providers(ef=None):
    ef = ef or fk.EventFakes()
    ad = EventAdapters(sam_audio=ef.sam_audio, judge=ef.judge, clap=None, tracker=ef.tracker)
    return Providers(
        visual=lambda: contextlib.nullcontext((fk.FakeVisual(), {"fake": "v1"})),
        speech=lambda: contextlib.nullcontext(fk.speech_models()),
        events=lambda: contextlib.nullcontext((ad, fk.fake_classifier, {"ast": "fake1"}))), ef


def go(tmp, name="out", fault=None, ef=None, cache=None, cfg=None):
    prov, ef = providers(ef)
    return run_pipeline(fk.make_source(), prov, cfg or load_config(), tmp / name, media_id="syn-14.5s", run_id="e2e",
                        cache_dir=cache or tmp / "cache", fault=fault,
                        evidence_note="mocked/synthetic CPU fakes"), ef


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    res, ef = go(d)
    return d, res, ef


def test_bundle_is_a_valid_long_run_bundle_with_all_source_kinds(base):
    d, res, _ = base
    out = d / "out"
    man = RunManifest.from_json((out / "manifest.json").read_text(encoding="utf-8"))
    assert man.long_run is not None and man.long_run.chunking["n_chunks"] == 7 and man.long_run.switches["peel"] is False
    assert man.long_run.diarization["status"] == "skipped"                     # intentional skip, recorded
    audio = AudioIndex.model_validate_json((out / "audio" / "index.json").read_text(encoding="utf-8"))
    kinds = {sid: e.kind for sid, e in audio.sources.items()}
    assert {"src_0", "src_1", "mixture", "residual"} <= set(kinds)
    assert any(k.startswith("tse:") and v == "tse" for k, v in kinds.items())
    assert any(k.startswith("event_") and v == "event" for k, v in kinds.items())
    assert audio.sources["src_0"].nyquist_hz == 8000.0                        # 16 kHz speech stem
    tl = TimelineIndex.model_validate_json((out / "timeline" / "index.json").read_text(encoding="utf-8"))
    assert tl.sample_rate == 16000 and tl.n_samples == res.speech.n_samples
    segs = [s for p in tl.pages for s in TimelinePageDoc.model_validate_json((out / p.path).read_text()).segments]
    assert segs and all(s.association_status and s.visibility and s.stem_availability for s in segs)
    assert (out / "frames" / "index.json").is_file() and (out / "viewer.html").is_file()
    # audio segments of one source never overlap and stay inside the clip
    for e in audio.sources.values():
        spans = sorted((s.start_sample, s.start_sample + s.n_samples) for s in e.segments)
        assert all(a1 <= b0 for (_, a1), (b0, _) in zip(spans, spans[1:]))


def test_event_window_is_sparse_attribution_stays_unmatched_and_gap_is_silent(base):
    d, res, ef = base
    audio = AudioIndex.model_validate_json((d / "out" / "audio" / "index.json").read_text())
    ev = {k: v for k, v in audio.sources.items() if k.startswith("event_")}
    assert len(ev) == 1
    (sid, e), = ev.items()
    total = sum(s.n_samples for s in e.segments)
    assert total < 4 * 48000                       # one short window, not the 14.5 s clip
    assert e.n_samples >= 0 and ef.sam_calls and max(len(c) for c in ef.sam_calls) <= load_config().memory.candidate_batch
    assert sum(len(c) for c in ef.sam_calls) <= load_config().memory.max_candidates
    meta = next(json.loads(p.read_text()) for p in (d / "cache" / "event_window").glob("*/event.json"))
    assert meta["attribution"]["status"] != "matched"                          # no tracked noun, no control, no spatial
    assert meta["control_available"] in (True, False) and meta["spatial_available"] is False
    assert meta["seeds"] == [1000 + i for i in range(load_config().memory.max_candidates)][: len(meta["seeds"])]


def test_residual_uses_real_stem_bandwidth(base):
    d, res, _ = base
    metas = [json.loads(p.read_text()) for p in (d / "cache" / "residual_chunk").glob("*/residual.json")]
    assert metas
    saw_speech = saw_event = False
    for m in metas:
        for sid, nyq in zip(m["stems"], m["stem_nyquist_hz"]):
            if sid.startswith("src_"):
                assert nyq == 8000.0               # 16 kHz speech upsampled to the 48 kHz master: nothing above 8 kHz
                saw_speech = True
            if sid.startswith("event_"):
                assert nyq is None                 # 48 kHz stem at a 48 kHz master is full band
                saw_event = True
    assert saw_speech and saw_event


def test_warm_rerun_invokes_no_model_and_reproduces_the_bundle(base, tmp_path):
    d, res, _ = base
    res2, ef2 = go(d, name="out_warm", cache=d / "cache")
    assert not res2.counters.calls and not res2.counters.chunks_computed and ef2.sam_calls == []
    a = AudioIndex.model_validate_json((d / "out" / "audio" / "index.json").read_text())
    b = AudioIndex.model_validate_json((d / "out_warm" / "audio" / "index.json").read_text())
    assert {k: [(s.start_sample, s.n_samples, s.sha256) for s in v.segments] for k, v in a.sources.items()} == \
           {k: [(s.start_sample, s.n_samples, s.sha256) for s in v.segments] for k, v in b.sources.items()}


@pytest.mark.parametrize("boundary", [("event_detect:mid", 0), ("event_window:mid", 0), ("residual:mid", 3),
                                      ("residual:after", 1)])
def test_interrupt_after_speech_resumes_without_rerunning_speech(base, tmp_path, boundary):
    d, ref, _ = base
    # reuse the completed speech/visual cache, drop everything downstream of it, then crash mid-way
    cache = tmp_path / "cache"
    shutil.copytree(d / "cache", cache)
    for stage in ("event_detect", "event_window", "residual_chunk"):
        shutil.rmtree(cache / stage, ignore_errors=True)

    def fault(name, k):
        if (name, k) == boundary:
            raise InjectedCrash(name)
    with pytest.raises(InjectedCrash):
        go(tmp_path, name="a", fault=fault, cache=cache)
    res, _ = go(tmp_path, name="b", cache=cache)
    assert not [c for c in res.counters.chunks_computed if c[0] in ("visual", "speech")]
    assert res.counters.calls == {}                                             # no speech/visual model call at all
    A = AudioIndex.model_validate_json((d / "out" / "audio" / "index.json").read_text())
    B = AudioIndex.model_validate_json((tmp_path / "b" / "audio" / "index.json").read_text())
    for sid in A.sources:
        assert [(s.start_sample, s.n_samples, s.sha256) for s in A.sources[sid].segments] == \
               [(s.start_sample, s.n_samples, s.sha256) for s in B.sources[sid].segments], sid


def test_event_stage_change_leaves_speech_cache_valid(base, tmp_path):
    d, _, _ = base
    cache = tmp_path / "cache"
    shutil.copytree(d / "cache", cache)
    cfg = load_config()
    cfg.memory.max_candidates = 2                    # changes the event stage only
    res, _ = go(tmp_path, name="c", cache=cache, cfg=cfg)
    kinds = {c[0] for c in res.counters.chunks_computed}
    assert "speech" not in kinds and "visual" not in kinds and "event_window" in kinds


def test_inspect_cli_summarises_a_long_bundle(base, capsys):
    from av_scene.__main__ import main
    d, _, _ = base
    assert main(["inspect", str(d / "out" / "manifest.json")]) == 0
    out = capsys.readouterr().out
    assert "long run: 7 chunks" in out and "timeline (association / visibility / stem availability kept separate)" in out
    assert "tse:p0" in out and "refused x" in out


def test_sustained_event_is_split_into_bounded_chained_windows(tmp_path):
    cfg = load_config()
    cfg.memory.event_max_window_s = 4.0
    prov, ef = providers()
    src = fk.make_source(chirp=(0.0, 14.5))                    # the sound never stops: a full-clip event
    res = run_pipeline(src, prov, cfg, tmp_path / "o", media_id="sustained", run_id="s", cache_dir=tmp_path / "c")
    metas = [json.loads(p.read_text()) for p in sorted((tmp_path / "c" / "event_window").glob("*/event.json"))]
    wins = sorted(m["window"] for m in metas)
    assert len(wins) >= 4 and all(hi - lo <= 4.0 + 1e-6 for lo, hi in wins)
    assert all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(wins, wins[1:]))        # contiguous: no gap, no duplicate
    assert all(m["continues_prev"] for m in sorted(metas, key=lambda m: m["window"][0])[1:])
    assert max(len(c) for c in ef.sam_calls) <= cfg.memory.candidate_batch
    assert res.counters.max_frames_resident <= cfg.memory.max_decoded_frames
    # the RAM used by event stems is bounded by one window: every stem file is at most window-length
    stems = list((tmp_path / "c" / "event_window").glob("*/stem.npy"))
    assert stems and all(np.load(p, mmap_mode="r").shape[0] <= int(4.0 * 48000) + 2 for p in stems)


def test_residual_proxy_is_stitched_without_gap_or_duplicate(base):
    d, res, _ = base
    audio = AudioIndex.model_validate_json((d / "out" / "audio" / "index.json").read_text())
    spans = sorted((s.start_sample, s.start_sample + s.n_samples) for s in audio.sources["residual"].segments)
    assert spans[0][0] == 0 and all(a1 == b0 for (_, a1), (b0, _) in zip(spans, spans[1:]))     # contiguous
    mix = sorted((s.start_sample, s.start_sample + s.n_samples) for s in audio.sources["mixture"].segments)
    assert spans[-1][1] == mix[-1][1]                                                           # whole clip covered


def test_diarization_load_failure_is_failed_not_not_attempted(tmp_path, monkeypatch):
    from av_scene import diar_windows
    monkeypatch.setattr(diar_windows, "availability", lambda **k: {"package": True, "cache": True, "token_present": True,
                                                                  "gated_access": "ok", "status": "ok"})
    prov, ef = providers()

    def boom():
        raise RuntimeError("terms not accepted")
    prov.diarization = boom
    cfg = load_config()
    cfg.switches.diarization = "on"
    r = run_pipeline(fk.make_source(), prov, cfg, tmp_path / "o", media_id="syn-14.5s", run_id="d", cache_dir=tmp_path / "c")
    assert r.manifest.long_run.diarization["status"] == "failed"
    assert "terms not accepted" in r.manifest.long_run.diarization["reason"]
    assert r.manifest.speakers                                              # the core pipeline still produced anchors
    prov2, _ = providers()                                                  # no pipeline provider at all: intentional skip
    cfg2 = load_config()
    cfg2.switches.diarization = "on"
    monkeypatch.setattr(diar_windows, "availability", lambda **k: {"package": False, "cache": False, "token_present": False,
                                                                  "gated_access": "not_attempted", "status": "not_attempted"})
    r2 = run_pipeline(fk.make_source(), prov2, cfg2, tmp_path / "o2", media_id="syn-14.5s", run_id="d2",
                      cache_dir=tmp_path / "c")
    assert r2.manifest.long_run.diarization["status"] == "not_attempted"


def test_diarization_activity_is_clip_time_even_with_an_audio_offset():
    from av_scene.run import identity_activity
    metas = [{"segments": [{"source_id": "src_0", "stem_availability": "ok", "start_us": 5_000_000, "end_us": 7_000_000}]}]
    assert identity_activity(metas, audio_start_us=1_500_000)["src_0"] == [(3_500_000, 5_500_000)]
