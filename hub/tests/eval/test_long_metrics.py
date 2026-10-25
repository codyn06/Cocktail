"""Long-clip metrics, annotations and the memory-matrix runner (evidence tier: mocked/synthetic).

Metric expectations are computed by hand in the comments; the scenes from long_fixtures supply the oracle runs."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

EVAL = Path(__file__).resolve().parents[3] / "eval" / "av_scene"
sys.path.insert(0, str(EVAL))
import long_annotations as A  # noqa: E402
import long_fixtures as F  # noqa: E402
import long_metrics as M  # noqa: E402
import long_run_matrix as RM  # noqa: E402

S = 16_000


def sec(a, b):
    return (int(a * S), int(b * S))


def seg(src, a, b, track=None, status=None, refusal=None):
    status = status or ("matched" if track else "unmatched")
    return M.Seg(src, int(a * S), int(b * S), track, status, "visible", "ok", refusal)


# ----------------------------------------------------------------------------------------- continuity
def test_identity_switches_hand_computed():
    gt = {"A": [sec(0, 10)]}
    pred = [seg("src_0", 0, 4), seg("src_1", 4, 6), seg("src_0", 6, 10)]       # src_0 -> src_1 -> src_0 = 2 switches
    r = M.identity_switches(pred, gt)
    assert r["total"] == 2 and r["id_sequence"]["A"] == ["src_0", "src_1", "src_0"]
    assert M.switches_per_minute(r, 10 * S) == pytest.approx(12.0)              # 2 switches in 10 s = 12 / min
    assert M.fragmentation(pred, gt)["total"] == 1                              # two distinct ids -> 1 beyond the first


def test_gaps_are_not_switches_but_a_new_id_after_a_gap_is():
    gt = {"A": [sec(0, 10)]}
    assert M.identity_switches([seg("src_0", 0, 4), seg("src_0", 6, 10)], gt)["total"] == 0
    assert M.identity_switches([seg("src_0", 0, 4), seg("src_1", 6, 10)], gt)["total"] == 1


def test_crosstalk_with_two_concurrent_ids_has_no_switches_and_slivers_are_not_fragmentation():
    gt = {"A": [sec(0, 10)], "B": [sec(4, 8)]}
    pred = [seg("src_0", 0, 10), seg("src_1", 4, 8)]
    assert M.identity_switches(pred, gt)["total"] == 0 and M.fragmentation(pred, gt)["total"] == 0
    sliver = [seg("src_0", 0, 9.9), seg("src_2", 9.9, 10)]                      # 0.1 s boundary sliver
    assert M.fragmentation(sliver, {"A": [sec(0, 10)]})["total"] == 0           # below the 0.5 s floor
    assert M.fragmentation(sliver, {"A": [sec(0, 10)]}, min_overlap_samples=1000)["total"] == 1
    assert M.identity_switches(sliver, {"A": [sec(0, 10)]}, min_run_samples=S // 5)["total"] == 0


# ---------------------------------------------------------------------------------------- attribution
def attribution_case():
    # A speaks [0,10) as person pA (track tA); B speaks [10,20) as pB (tB); both on screen
    ctx = dict(onscreen={"A": [sec(0, 10)], "B": [sec(10, 20)]}, track_person={"tA": "pA", "tB": "pB"},
               person_speaker={"pA": "A", "pB": "B"})
    pred = [seg("src_0", 0, 10, "tA"), seg("src_0", 10, 15, "tA")]              # second span is the WRONG person
    return pred, ctx


def test_coverage_recall_and_wrong_person_hand_computed():
    pred, ctx = attribution_case()
    assert M.attribution_coverage(pred, ctx["onscreen"]) == pytest.approx(0.75)       # [0,15) of 20 s attributed
    assert M.active_speaker_recall(pred, **ctx) == pytest.approx(0.5)                 # only A's 10 s are right
    w = M.wrong_person_assignments(pred, {"A": [sec(0, 10)], "B": [sec(10, 20)]}, ctx["track_person"], ctx["person_speaker"])
    assert (w["wrong_samples"], w["correct_samples"], w["wrong_segments"]) == (5 * S, 10 * S, 1)
    assert w["wrong_rate"] == pytest.approx(5 / 15)
    assert M.source_coverage([seg("src_0", 0, 5)], {"A": [sec(0, 10)]}) == pytest.approx(0.5)


def test_refused_or_unmatched_time_is_not_attribution():
    pred = [seg("src_0", 0, 10, "tA", status="ambiguous"), seg("src_1", 10, 20, "tB", refusal="off-screen")]
    assert M.attribution_coverage(pred, {"A": [sec(0, 20)]}) == 0.0
    assert M.is_refused(pred[0]) and M.is_refused(pred[1]) and not M.is_attributed(pred[1])


def refusal_case():
    # A on screen [0,10); O is off screen [10,20) and must be refused
    gt = {"A": [sec(0, 10)], "O": [sec(10, 20)]}
    return gt, [sec(10, 20)]


def test_refusal_precision_recall_hand_computed():
    gt, should = refusal_case()
    perfect = [seg("src_0", 0, 10, "tA"), seg("src_1", 10, 20, None, refusal="outside_view")]
    r = M.refusal_precision_recall(perfect, gt, should)
    assert (r["precision"], r["recall"]) == (1.0, 1.0)
    leaky = [seg("src_0", 0, 10, "tA"), seg("src_1", 10, 14, "tA")]                   # attributes 4 s of the stranger
    r = M.refusal_precision_recall(leaky, gt, should)
    assert r["recall"] == pytest.approx(6 / 10) and r["precision"] == 1.0             # refused [14,20) = 6 s, all correct
    assert M.refusal_precision_recall([], {"A": []}, [])["recall"] is None            # nothing to refuse: n/a, not 0


def test_refuse_everything_fails_the_gate_via_recall_and_precision():
    gt, should = refusal_case()
    ctx = dict(onscreen={"A": [sec(0, 10)]}, track_person={"tA": "pA"}, person_speaker={"pA": "A"})
    nothing = []                                                                      # no attribution anywhere
    r = M.refusal_precision_recall(nothing, gt, should)
    assert r["recall"] == 1.0 and r["precision"] == pytest.approx(0.5)                # recall alone would pass
    m = {"active_speaker_recall": M.active_speaker_recall(nothing, **ctx), "refusal_precision": r["precision"],
         "refusal_recall": r["recall"], "silent_bystander_rate": 0.0}
    g = M.gate_pass(m)
    assert m["active_speaker_recall"] == 0.0 and not g["passed"]
    assert g["checks"]["active_speaker_recall"]["status"] == "fail" and g["checks"]["refusal_precision"]["status"] == "fail"
    assert g["checks"]["refusal_recall"]["status"] == "pass"
    perfect = {"active_speaker_recall": 0.95, "refusal_precision": 0.9, "refusal_recall": 0.85, "silent_bystander_rate": 0.0}
    assert M.gate_pass(perfect)["passed"]
    assert not M.gate_pass({k: v for k, v in perfect.items() if k != "refusal_recall"})["passed"]   # unmeasured fails
    assert M.gate_pass({**perfect, "refusal_recall": None})["checks"]["refusal_recall"]["status"] == "n/a"


def test_silent_bystander_rate_arithmetic_and_gate_boundary():
    silent = {"tB": [sec(0, 10)], "tC": [sec(0, 10)]}                                 # 20 person-seconds
    pred = [seg("src_0", 0, 0.2, "tB"), seg("src_0", 5, 5.2, "tB", status="ambiguous"), seg("src_1", 3, 3.1, "tA")]
    r = M.silent_bystander_highlight_rate(pred, silent)
    assert r == pytest.approx(0.2 / 20)                                               # ambiguous and other tracks ignored
    assert M.gate_pass({"silent_bystander_rate": r, "active_speaker_recall": 1, "refusal_precision": 1,
                        "refusal_recall": 1})["passed"]
    worse = M.silent_bystander_highlight_rate([seg("src_0", 0, 0.5, "tB")], silent)
    assert worse == pytest.approx(0.025)
    assert not M.gate_pass({"silent_bystander_rate": worse, "active_speaker_recall": 1, "refusal_precision": 1,
                            "refusal_recall": 1})["passed"]
    assert M.silent_bystander_highlight_rate([seg("src_0", 0, 0.5, "tB")], silent, guard_samples=S // 2) == 0.0
    grouped = M.silent_bystander_highlight_rate([seg("src_0", 0, 1, "tB.r1")], {"p": [sec(0, 10)]}, groups={"p": {"tB", "tB.r1"}})
    assert grouped == pytest.approx(0.1)
    assert M.silent_bystander_highlight_rate([], {}) is None


# ------------------------------------------------------------------------------------------ memory growth
def series(rate_per_min, base=1000.0, dur=300.0, step=2.0, noise=2.0, warm_ramp=0.0):
    rng = np.random.default_rng(0)
    out = []
    for t in np.arange(step, dur + 1e-9, step):
        v = base + (min(t, 12.0) / 12.0) * warm_ramp + max(0.0, t - 12.0) / 60.0 * rate_per_min + noise * rng.standard_normal()
        out.append((float(t), float(v)))
    return out


def test_memory_growth_flat_passes_and_growing_fails():
    flat = [{"duration_s": 30, "series": series(0, dur=30, warm_ramp=800.0), "gpu_peak_mib": "unavailable"},
            {"duration_s": 300, "series": series(0, dur=300, warm_ramp=800.0), "gpu_peak_mib": "unavailable"}]
    r = M.memory_growth(flat)
    c = r["checks"]
    assert r["warmup_end_audio_s"] == 12.0                       # 4 s window + (5 - 1) * 2 s hop
    assert c["rss_slope_mib_per_audio_min"]["status"] == "pass" and abs(c["rss_slope_mib_per_audio_min"]["value"]) < 0.5
    assert c["rss_peak_growth_mib"]["status"] == "pass" and c["rss_peak_mib"]["status"] == "pass"
    assert c["gpu_peak_mib"]["status"] == "unavailable" and r["passed"] and not r["complete"]     # honest, not a pass
    grow = [flat[0], {"duration_s": 300, "series": series(100.0, dur=300, warm_ramp=800.0), "gpu_peak_mib": 8000.0}]
    g = M.memory_growth(grow)
    assert g["checks"]["rss_slope_mib_per_audio_min"]["status"] == "fail"
    assert g["checks"]["rss_slope_mib_per_audio_min"]["value"] == pytest.approx(100.0, abs=3.0)
    assert g["checks"]["rss_peak_growth_mib"]["status"] == "fail"                 # +~480 MiB over 288 s
    assert not g["passed"] and g["complete"]


def test_memory_growth_peak_and_gpu_budgets_use_the_contract_numbers():
    b = M.load_hub_module("contracts").MemoryBudgets()
    w = [{"duration_s": 30, "series": series(0, dur=30), "rss_peak_mib": 1000.0, "gpu_peak_mib": 9000.0},
         {"duration_s": 300, "series": series(0, dur=300), "rss_peak_mib": 1000.0 + b.rss_peak_growth_mib_30_to_300s + 1,
          "gpu_peak_mib": b.gpu_peak_mib + 1}]
    c = M.memory_growth(w, b)["checks"]
    assert c["rss_peak_growth_mib"]["status"] == "fail" and c["rss_peak_growth_mib"]["limit"] == 256.0
    assert c["gpu_peak_mib"]["status"] == "fail" and c["gpu_peak_mib"]["limit"] == 9216.0
    assert M.memory_growth([{"duration_s": 300, "series": series(0, dur=300), "rss_peak_mib": 7000.0}], b)["checks"][
        "rss_peak_mib"]["status"] == "fail"


# ------------------------------------------------------------------------------------------ compare_runs
def snapshot(n=1000, n_samples=1000, segs=None):
    rng = np.random.default_rng(1)
    segs = segs or [M.Seg("src_0", 0, 500, "t", "matched", "visible", "ok").__dict__]
    return {"timeline_samples": n_samples,
            "sources": {"src_0": {"n_samples": n_samples, "segments": segs, "audio": rng.standard_normal(n).astype(np.float32)},
                        "src_1": {"n_samples": n_samples, "segments": [], "audio": np.zeros(n, np.float32)}}}


def test_compare_runs_identical_and_detects_each_kind_of_difference():
    a, b = snapshot(), snapshot()
    r = M.compare_runs(a, b, tol=1e-6)
    assert r["identical"] and r["audio_max_abs_diff"] == 0.0
    longer = snapshot(n_samples=1001)                                       # 1-sample logical length change
    r = M.compare_runs(a, longer)
    assert not r["identical"] and {d["source_id"] for d in r["sample_count_diffs"]} >= {"src_0", "<timeline>"}
    arr = snapshot()
    arr["sources"]["src_0"]["audio"] = np.append(arr["sources"]["src_0"]["audio"], np.float32(0))   # 1-sample array change
    r = M.compare_runs(a, arr)
    assert not r["identical"] and r["audio_length_diffs"] == [{"source_id": "src_0", "a": 1000, "b": 1001}]
    tweak = snapshot()
    tweak["sources"]["src_0"]["audio"] = tweak["sources"]["src_0"]["audio"].copy()
    tweak["sources"]["src_0"]["audio"][10] += 2e-3
    r = M.compare_runs(a, tweak, tol=1e-3)
    assert not r["identical"] and r["audio_max_abs_diff"] == pytest.approx(2e-3, rel=0.05) and r["audio_over_tol"]
    assert M.compare_runs(a, tweak, tol=1e-2)["identical"]                  # within the declared tolerance
    seg2 = snapshot(segs=[M.Seg("src_0", 0, 501, "t", "matched", "visible", "ok").__dict__])
    assert M.compare_runs(a, seg2)["timeline_diffs"]
    gone = snapshot()
    del gone["sources"]["src_1"]
    assert M.compare_runs(a, gone)["missing_in_b"] == ["src_1"] and not M.compare_runs(a, gone)["identical"]


# ------------------------------------------------------------------------------------ bundle round trip
def write_bundle(root: Path, shorten=0):
    sch = M.load_hub_module("schemas")
    (root / "timeline").mkdir(parents=True)
    (root / "audio").mkdir()
    segs = [sch.TimelineSegment(source_id="src_0", start_sample=0, end_sample=24_000, start_us=0, end_us=1_500_000,
                                track_id="t0", association_status="matched", visibility="visible", stem_availability="ok"),
            sch.TimelineSegment(source_id="src_0", start_sample=24_000, end_sample=40_000, start_us=1_500_000, end_us=2_500_000,
                                association_status="unmatched", visibility="outside_view", refusal_reason="off-screen"),
            sch.TimelineSegment(source_id="tse:t0", start_sample=0, end_sample=24_000, start_us=0, end_us=1_500_000,
                                track_id="t0")]
    for k, part in enumerate((segs[:1], segs[1:])):                         # src_0 is split across two pages
        (root / "timeline" / f"page_{k}.json").write_text(sch.TimelinePageDoc(segments=part).to_json(), encoding="utf-8")
    idx = sch.TimelineIndex(sample_rate=16_000, n_samples=40_000, pages=[
        sch.TimelinePage(path="timeline/page_0.json", start_us=0, end_us=1_500_000),
        sch.TimelinePage(path="timeline/page_1.json", start_us=1_500_000, end_us=2_500_000)],
        sources=[sch.TimelineSource(source_id="src_0", kind="speech", role="default"),
                 sch.TimelineSource(source_id="tse:t0", kind="tse", role="default")])
    (root / "timeline" / "index.json").write_text(idx.to_json(), encoding="utf-8")
    x = np.sin(np.arange(40_000) / 7.0).astype(np.float32)[: 40_000 - shorten]
    np.save(root / "audio" / "src_0_0.npy", x[:24_000])
    np.save(root / "audio" / "src_0_1.npy", x[24_000:])
    ai = sch.AudioIndex(sources={"src_0": sch.AudioSourceEntry(source_id="src_0", kind="speech", role="default", rate_hz=16_000,
                        n_samples=40_000 - shorten, segments=[
                            sch.AudioSegmentRef(start_sample=0, n_samples=24_000, path="audio/src_0_0.npy"),
                            sch.AudioSegmentRef(start_sample=24_000, n_samples=16_000 - shorten, path="audio/src_0_1.npy")])})
    (root / "audio" / "index.json").write_text(ai.to_json(), encoding="utf-8")


def test_load_timeline_and_snapshot_bundle_roundtrip(tmp_path):
    write_bundle(tmp_path / "a")
    tl = M.load_timeline(tmp_path / "a")
    assert tl.sample_rate == 16_000 and sorted(tl.segments) == ["src_0", "tse:t0"]
    assert [(s.start, s.end) for s in tl.segments["src_0"]] == [(0, 24_000), (24_000, 40_000)]       # pages merged in order
    assert [s.source_id for s in tl.speech_segments()] == ["src_0", "src_0"]                         # tse stems excluded
    assert tl.segments["src_0"][1].refusal == "off-screen" and M.is_attributed(tl.segments["src_0"][0])
    assert [s.start for s in tl.segments_at(8_000)] == [0, 12_000]
    write_bundle(tmp_path / "b")
    sa, sb = M.snapshot_bundle(tmp_path / "a"), M.snapshot_bundle(tmp_path / "b")
    assert M.compare_runs(sa, sb)["identical"] and len(sa["sources"]["src_0"]["audio"]) == 40_000
    write_bundle(tmp_path / "c", shorten=1)                                                          # 1 sample shorter
    r = M.compare_runs(sa, M.snapshot_bundle(tmp_path / "c"))
    assert not r["identical"] and r["sample_count_diffs"][0]["a"] == 40_000 and r["sample_count_diffs"][0]["b"] == 39_999
    (tmp_path / "a" / "audio" / "src_0_0.npy").unlink()
    np.save(tmp_path / "a" / "audio" / "src_0_0.npy", np.zeros(24_000, np.float32))                   # content change
    assert M.compare_runs(M.snapshot_bundle(tmp_path / "a"), sb, tol=1e-4)["audio_over_tol"]


# ----------------------------------------------------------------------------------- scenes x metrics
def score(scene, pred):
    gt = scene.ground_truth
    return M.score_all(pred, duration_samples=gt.n_samples, **gt.metric_inputs())


@pytest.mark.parametrize("name", ["pause_return", "crossing", "reentry", "silent_overlap", "slot_swap_every_chunk",
                                  "final_partial", "silent_bystander"])
def test_oracle_prediction_passes_every_scene(name):
    scene = F.build_scene(name)
    r = score(scene, F.oracle_segments(scene))
    m = r["metrics"]
    assert r["gate"]["passed"], r["gate"]
    assert m["identity_switches"] == 0 and m["fragmentation"] == 0 and m["source_coverage"] == 1.0
    assert m["silent_bystander_rate"] in (0.0, None) and m["wrong_person_rate"] in (0.0, None)
    assert m["active_speaker_recall"] in (1.0, None)


def test_scene_failures_are_caught():
    scene = F.build_scene("silent_bystander")
    oracle = F.oracle_segments(scene)
    refuse_all = [M.Seg(s.source_id, s.start, s.end, None, "unmatched", "unknown", "unavailable", "refused") for s in oracle]
    r = score(scene, refuse_all)
    assert not r["gate"]["passed"] and r["metrics"]["active_speaker_recall"] == 0.0
    assert r["metrics"]["refusal_recall"] == 1.0 and r["metrics"]["refusal_precision"] < 0.8
    n = scene.n16
    highlight = oracle + [M.Seg("src_9", 0, n, "trk_pB", "matched", "visible", "ok")]                # bystander lit all clip
    r = score(scene, highlight)
    assert r["metrics"]["silent_bystander_rate"] == pytest.approx(24 / 37) and not r["gate"]["passed"]    # 24 s of 37 silent s
    assert r["metrics"]["wrong_person_rate"] > 0.3
    swap = F.build_scene("slot_swap_every_chunk")                                                      # ids flip every 4 s
    pieces = [(s, a, min(s.end, a + 64_000)) for s in F.oracle_segments(swap) for a in range(s.start, s.end, 64_000)]
    r = score(swap, [M.Seg(f"src_{(a // 64_000) % 2}", a, b, s.track_id, s.status, s.visibility, s.stem) for s, a, b in pieces])
    assert r["metrics"]["identity_switches"] > 3 and r["metrics"]["fragmentation"] >= 1 and r["metrics"]["switches_per_minute"] > 5


def test_reentry_new_track_wrongly_given_to_other_person_is_a_wrong_assignment():
    scene = F.build_scene("reentry")
    oracle = F.oracle_segments(scene)
    bad = [M.Seg(s.source_id, s.start, s.end, "trk_pB" if s.track_id == "trk_pA.r1" else s.track_id, s.status, s.visibility,
                 s.stem, s.refusal) for s in oracle]
    r = score(scene, bad)
    assert r["metrics"]["wrong_person_rate"] > 0.3 and r["metrics"]["active_speaker_recall"] < 0.7


# ------------------------------------------------------------------------------------------ annotations
def doc_for_bystander_scene():
    us = lambda s: int(s * 1_000_000)
    return {"version": 1, "clip_id": "synthetic_bystander", "duration_us": us(24),
            "people": [{"person_id": "pA", "speaker_id": "A", "track_ids": ["trk_pA"], "visible_us": [[0, us(24)]],
                        "speaking_us": [[us(1), us(12)]]},
                       {"person_id": "pB", "speaker_id": None, "track_ids": ["trk_pB"], "visible_us": [[0, us(24)]]}],
            "off_screen_speakers": [{"speaker_id": "C", "speaking_us": [[us(14), us(20)]]}]}


def test_annotation_validator_rejects_malformed_documents():
    good = doc_for_bystander_scene()
    ann = A.parse_annotations(good)
    assert ann.people[1].silent_visible_us == [(0, 24_000_000)] and ann.people[0].silent_visible_us == [(0, 1_000_000), (12_000_000, 24_000_000)]
    for mutate in (lambda d: d["people"][0].update(speaking_us=[[0, 25_000_000]]),                                  # past the end
                   lambda d: d["people"][0].update(speaking_us=[[5_000_000, 6_000_000], [1_000_000, 2_000_000]]),   # unsorted
                   lambda d: d["people"][0].update(speaking_us=[[1.5, 2]]),                                            # float
                   lambda d: d["people"][0].update(speaking_us=[[3, 3]]),                                              # empty span
                   lambda d: d["people"][0].update(visible_us=[[0, 1_000_000]]),                                       # speaking not visible
                   lambda d: d["people"][1].update(speaker_id="A"),                                                    # duplicate speaker
                   lambda d: d["people"][1].update(track_ids=["trk_pA"]),                                              # duplicate track
                   lambda d: d["people"][0].update(silent_visible_us=[[2_000_000, 3_000_000]]),                        # overlaps speaking
                   lambda d: d.update(version=2)):
        d = json.loads(json.dumps(good))
        mutate(d)
        with pytest.raises(A.AnnotationError):
            A.parse_annotations(d)
    assert A.us_to_sample(1_000_000) == 16_000 and A.us_to_sample(62) == 1                                             # half-up


def test_score_annotations_on_oracle_and_on_a_highlighted_bystander():
    scene = F.build_scene("silent_bystander")
    ann = A.parse_annotations(doc_for_bystander_scene())
    oracle = F.oracle_segments(scene)
    r = A.score_annotations(oracle, ann)
    assert r["gate"]["passed"] and r["metrics"]["silent_bystander_rate"] == 0.0 and r["unmapped_tracks"] == []
    assert r["metrics"]["refusal_recall"] == 1.0 and r["duration_s"] == 24.0
    lit = oracle + [M.Seg("src_9", 16_000 * 13, 16_000 * 14, "trk_pB", "matched", "visible", "ok")]       # 1 s of 24 s
    r = A.score_annotations(lit, ann)
    assert r["metrics"]["silent_bystander_rate"] == pytest.approx(1 / 37) and not r["gate"]["passed"]    # 1 s of 37 silent-visible person-seconds
    stray = oracle + [M.Seg("src_8", 0, 16_000, "trk_new", "matched", "visible", "ok")]
    assert A.score_annotations(stray, ann)["unmapped_tracks"] == ["trk_new"]
    mapped = A.score_annotations(stray, ann, track_map={"trk_new": "pB"})
    assert mapped["unmapped_tracks"] == [] and mapped["metrics"]["silent_bystander_rate"] == pytest.approx(1 / 37)
    with pytest.raises(A.AnnotationError):
        A.score_annotations(stray, ann, track_map={"trk_pA": "pB"})                                         # conflicts with the file


def test_score_annotations_reads_a_loaded_timeline(tmp_path):
    write_bundle(tmp_path / "b")
    tl = M.load_timeline(tmp_path / "b")
    doc = {"version": 1, "clip_id": "c", "duration_us": 2_500_000,
           "people": [{"person_id": "p", "speaker_id": "s", "track_ids": ["t0"], "visible_us": [[0, 2_500_000]],
                       "speaking_us": [[0, 1_500_000]]}],
           "off_screen_speakers": [{"speaker_id": "o", "speaking_us": [[1_500_000, 2_500_000]]}]}
    r = A.score_annotations(tl, doc)
    assert r["gate"]["passed"] and r["metrics"]["active_speaker_recall"] == 1.0 and r["metrics"]["refusal_recall"] == 1.0


# ---------------------------------------------------------------------------------------- run matrix
STANDIN = textwrap.dedent("""
    import argparse, subprocess, sys, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--duration", type=float); ap.add_argument("--grow-mib", type=float, default=0.0)
    ap.add_argument("--mock-scene", default=None); ap.add_argument("--fail", action="store_true")
    a = ap.parse_args()
    child = subprocess.Popen([sys.executable, "-c", "import time; x = bytearray(80 * 2**20); x[::4096] = b'1' * len(x[::4096]); time.sleep(30)"])
    held = []
    n = int(a.duration // 2)
    for i in range(1, n + 1):
        if a.grow_mib:
            b = bytearray(int(a.grow_mib * 2**20)); b[::4096] = b'1' * len(b[::4096]); held.append(b)
        print(f"chunk {i}/{n}", flush=True)
        time.sleep(0.04)
    child.kill()
    sys.exit(3 if a.fail else 0)
""")


@pytest.fixture()
def standin(tmp_path):
    p = tmp_path / "standin.py"
    p.write_text(STANDIN, encoding="utf-8")
    return p.as_posix()


LOOSE = {"rss_slope_mib_per_audio_min": 60.0, "rss_peak_growth_mib_30_to_300s": 100.0}


def test_build_argv_drops_unset_flag_pairs():
    t = "{python} -m av_scene run --media {media} --out {out} --config {config} --mock-scene {mock_scene}"
    v = {"python": "py", "media": None, "out": "o", "config": None, "mock_scene": "matrix:30"}
    assert RM.build_argv(t, v) == ["py", "-m", "av_scene", "run", "--out", "o", "--mock-scene", "matrix:30"]
    assert RM.build_argv("{python} {media}", {"python": "p", "media": "m.mp4"}) == ["p", "m.mp4"]
    with pytest.raises(ValueError):
        RM.build_argv("{python} {media}", {"python": "p", "media": None})


def test_matrix_measures_the_process_tree_and_passes_a_flat_workload(tmp_path, standin):
    gpu = iter(range(1000, 10_000, 7))
    res = RM.run_matrix(tmp_path / "m", durations=[20, 40], mock=True, argv_template=f"{{python}} {standin} --out {{out}} --duration {{duration_s}}",
                        cwd=tmp_path, interval_s=0.03, gpu_reader=lambda: float(next(gpu)), budgets=LOOSE)
    assert (tmp_path / "m" / "matrix.json").is_file() and json.loads((tmp_path / "m" / "matrix.json").read_text())["mock"]
    for w in res["workloads"]:
        assert w["returncode"] == 0 and w["audio_time_mapping"] == "progress_lines" and len(w["rss_series"]) >= 5
        assert w["rss_peak_mib"] > 70, "the 80 MiB grandchild must be counted (whole process tree)"
        assert w["gpu"]["status"] == "ok" and w["gpu"]["baseline_mib"] >= 1000.0 and w["gpu"]["peak_mib"] >= w["gpu"]["baseline_mib"]
    assert res["memory_growth"]["warmup_end_audio_s"] == 12.0
    assert res["passed"], RM.format_table(res)


def test_matrix_flags_a_growing_workload_and_reports_missing_gpu(tmp_path, standin):
    res = RM.run_matrix(tmp_path / "m", durations=[20, 40], mock=True, gpu_reader=lambda: None, cwd=tmp_path, interval_s=0.03,
                        argv_template=f"{{python}} {standin} --out {{out}} --duration {{duration_s}} --grow-mib 6", budgets=LOOSE)
    assert not res["passed"] and res["memory_growth"]["checks"]["rss_slope_mib_per_audio_min"]["status"] == "fail"
    assert res["workloads"][0]["gpu"]["status"] == "unavailable" and "gpu_peak_mib" in res["memory_growth"]["unavailable"]
    assert "FAIL" in RM.format_table(res)


def test_matrix_records_a_failed_child_and_main_exit_code(tmp_path, standin):
    rc = RM.main(["--out", str(tmp_path / "m"), "--mock", "--durations", "20", "--gpu", "off", "--cwd", str(tmp_path),
                  "--interval-ms", "30", "--argv-template", f"{{python}} {standin} --duration {{duration_s}} --fail"])
    assert rc == 1
    assert json.loads((tmp_path / "m" / "matrix.json").read_text())["failed_workloads"] == [20.0]


def test_run_matrix_module_does_not_import_torch():
    code = ("import sys; sys.path.insert(0, %r); import long_run_matrix, long_metrics, long_annotations, long_fixtures; "
            "assert 'torch' not in sys.modules, 'torch imported'" % str(EVAL))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_abstention_is_not_an_identity_switch_but_still_costs_recall():
    gt = {"A": [(0, 3000)]}
    pred = [M.Seg("unresolved", 0, 1000, None, "unmatched", "visible", "unavailable", "short utterance"),
            M.Seg("src_0", 1000, 3000, "tA", "matched", "visible", "ok", None)]
    assert M.identity_switches(pred, gt)["total"] == 0 and M.fragmentation(pred, gt, min_overlap_samples=1)["total"] == 0
    assert M.source_coverage([p for p in pred if p.source_id != "unresolved"], gt) < 1.0     # abstaining is not free
