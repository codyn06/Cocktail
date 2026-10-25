"""Phase 4 CPU/mocked acceptance gates: T4's deterministic scenes through the REAL chunked pipeline.

Evidence: mocked/synthetic. The separator, scorer, TSE, embedder and visual front end are fakes
(`eval/av_scene/long_fixtures.py`); the pipeline, identity store, stitching, cache and export are the production code.
These gates say the plumbing preserves identity and refuses correctly when the models are perfect; they say nothing
about real-model quality.
"""
import json
from pathlib import Path

import pytest

from av_scene import __main__ as cli
from av_scene.config import load_config
from av_scene.mock_deps import build_mock
from av_scene.run import run_pipeline

LM = cli._load_eval("long_metrics")
FX = cli._load_eval("long_fixtures")

SCENES = ["pause_return", "crossing", "reentry", "silent_overlap", "slot_swap_every_chunk", "final_partial",
          "silent_bystander"]
_cache: dict = {}


def scored(spec, tmp_root):
    if spec not in _cache:
        cfg = load_config()
        cfg.switches.events = False
        scene, source, prov = build_mock(spec, 32000)
        out = tmp_root / spec
        res = run_pipeline(source, prov, cfg, out, media_id=f"mock:{spec}", run_id=spec,
                           evidence_tier="mocked/synthetic", cache_dir=tmp_root / (spec + "_cache"))
        tl = LM.load_timeline(out)
        sc = LM.score_all(tl.segments_at(16000), duration_samples=scene.n16, **scene.ground_truth.metric_inputs())
        _cache[spec] = (scene, res, tl, sc)
    return _cache[spec]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return tmp_path_factory.mktemp("gates")


@pytest.mark.parametrize("spec", SCENES)
def test_identity_survives_pauses_crossings_reentry_and_slot_permutation_without_swaps(spec, root):
    scene, res, tl, sc = scored(spec, root)
    m = sc["metrics"]
    assert m["identity_switches"] == 0, sc["detail"]["identity_switches"]            # zero on deterministic fixtures
    assert m["fragmentation"] == 0
    assert m["wrong_person_rate"] == 0.0, sc["detail"]["wrong_person"]


@pytest.mark.parametrize("spec", SCENES)
def test_silent_bystanders_are_not_highlighted_and_speakers_are_not_missed(spec, root):
    scene, res, tl, sc = scored(spec, root)
    m = sc["metrics"]
    if m["silent_bystander_rate"] is not None:
        assert m["silent_bystander_rate"] < 0.02, m                                  # the spec's gate
    assert m["source_coverage"] >= 0.9                                               # refusing everything cannot pass:
    assert m["active_speaker_recall"] is None or m["active_speaker_recall"] >= 0.9    # recall is gated alongside


def test_reentry_keeps_the_acoustic_id_across_absence_and_refuses_the_absent_speaker(root):
    scene, res, tl, sc = scored("reentry", root)
    seq = sc["detail"]["identity_switches"]["id_sequence"]["A"]
    assert seq == ["src_0"]                         # same person, absent for 16 s and back under a NEW visual track id
    ref = sc["detail"]["refusal"]
    assert ref["recall"] == 1.0 and ref["precision"] >= 0.8                          # off-view speech was refused
    segs = [s for s in tl.speech_segments() if s.source_id.startswith("src_")]
    tracks = {s.track_id for s in segs if s.track_id}
    assert {"trk_pA", "trk_pA.r1"} <= tracks        # visual ids differ; acoustic id is what persisted


def test_short_first_utterance_abstains_with_a_reason_instead_of_minting_an_id(root):
    scene, res, tl, sc = scored("reentry", root)
    unresolved = [s for s in tl.all_segments() if s.source_id == "unresolved"]
    assert unresolved and all(s.stem == "unavailable" and s.refusal for s in unresolved)
    assert any("identity_not_established" in s.refusal for s in unresolved)


def test_silent_bystander_scene_refuses_offscreen_voice_and_reports_four_separate_fields(root):
    scene, res, tl, sc = scored("silent_bystander", root)
    assert sc["metrics"]["refusal_recall"] == 1.0 and sc["metrics"]["silent_bystander_rate"] == 0.0
    tse = [s for s in tl.all_segments() if s.source_id.startswith("tse:")]
    assert any(s.status == "unmatched" and s.stem == "unavailable" and s.refusal for s in tse)   # reason kept per segment


def test_final_partial_window_is_covered_to_the_last_sample(root):
    scene, res, tl, sc = scored("final_partial", root)
    assert tl.n_samples == scene.n16 and res.speech.records[-1].spec.own_end == scene.n16
    assert res.speech.records[-1].spec.win_len < 4 * 16000                           # the last window really is partial
    audio = json.loads((root / "final_partial" / "audio" / "index.json").read_text())
    for sid in ("src_0", "src_1"):
        end = max(s["start_sample"] + s["n_samples"] for s in audio["sources"][sid]["segments"])
        assert end <= scene.n16


def test_refusing_everything_cannot_pass_the_gate(root):
    """Guard on the metric itself: a prediction that refuses every segment fails on recall."""
    scene, res, tl, sc = scored("pause_return", root)
    refused = [LM.Seg(s.source_id, s.start, s.end, None, "unmatched", "unknown", "unavailable", "refused", s.kind)
               for s in tl.segments_at(16000)]
    bad = LM.score_all(refused, duration_samples=scene.n16, **scene.ground_truth.metric_inputs())
    assert bad["metrics"]["active_speaker_recall"] == 0.0 and not bad["gate"]["passed"]
