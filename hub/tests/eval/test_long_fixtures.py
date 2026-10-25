"""Synthetic long-clip scenes and fake models (evidence tier: mocked/synthetic)."""

from __future__ import annotations

import json
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "av_scene"))
import fixtures_long as F  # noqa: E402  (shim over eval/av_scene/long_fixtures.py)
import av_scene  # noqa: E402,F401
from av_scene import contracts as C  # noqa: E402
from av_scene.speech import run_speech_branch  # noqa: E402

LM = F.LM


def chunk_inputs(scene, start16, end16, tracker=None):
    """What the pipeline hands the models for window [start16, end16): audio, frames, tracker output."""
    src = scene.source()
    audio = src.read_audio(start16 * 3, end16 * 3, 16000).mean(axis=0)
    pts = src.video_pts_us()
    i0, i1 = C.video_index_range(pts, C.sample_to_us(start16, 16000), C.sample_to_us(end16, 16000))
    frames = src.read_frames(i0, i1)
    vc = (tracker or F.FakeVisualFrontEnd(scene)).step(frames, pts[i0:i1], {})
    return audio, frames, vc


def corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.mark.parametrize("name", list(F.SCENARIOS))
def test_scenarios_are_deterministic(name):
    a, b = F.build_scene(name), F.build_scene(name)
    assert a.video_pts_us() == b.video_pts_us()
    assert a.ground_truth == b.ground_truth
    xa = a.source().read_audio(48_000, 48_000 + 24_000, 48_000)
    xb = b.source().read_audio(48_000, 48_000 + 24_000, 48_000)
    assert np.array_equal(xa, xb) and np.abs(xa).max() > 0
    assert np.array_equal(a.render_frame(a.video_pts_us()[40]), b.render_frame(b.video_pts_us()[40]))


def test_read_audio_is_span_consistent_and_rate_consistent():
    src = F.pause_return().source()
    whole = src.read_audio(3 * 48_000, 6 * 48_000, 48_000)
    assert whole.shape == (2, 3 * 48_000) and whole.dtype == np.float32
    part = src.read_audio(3 * 48_000 + 12_345, 3 * 48_000 + 99_999, 48_000)
    assert np.array_equal(whole[:, 12_345:99_999], part)
    assert np.allclose(whole[1], 0.9 * whole[0], atol=1e-6)
    w16 = src.read_audio(3 * 48_000, 6 * 48_000, 16_000)
    assert w16.shape == (2, 3 * 16_000)
    assert np.allclose(w16[0], whole[0, ::3], atol=2e-6)       # 16 kHz view == 48 kHz master at the same instants
    with pytest.raises(ValueError):
        src.read_audio(0, 10, 44_100)


def test_300s_scene_is_never_allocated_whole():
    tracemalloc.start()
    scene = F.matrix(300.0)
    src = scene.source()
    assert src.n_audio_samples == 300 * 48_000
    x = src.read_audio(100 * 48_000, 104 * 48_000, 48_000)
    f = src.read_frames(2500, 2600)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert x.shape == (2, 192_000) and f.shape == (100, 90, 160, 3)
    assert peak < 60 * 2**20, f"peak {peak / 2**20:.1f} MiB (a whole 300 s master would be ~110 MiB as float32)"
    assert src.stats["max_audio_samples_per_read"] == 192_000 and src.stats["max_frames_per_read"] == 100


def test_read_frames_enforces_the_frame_cap():
    src = F.pause_return().source()
    with pytest.raises(ValueError):
        src.read_frames(0, 151)
    assert src.read_frames(0, 150).shape[0] == 150


def test_video_pts_are_variable_rate_and_sorted():
    pts = F.pause_return().video_pts_us()
    d = np.diff(pts)
    assert (d > 0).all() and len(set(d.tolist())) > 3
    assert d.max() >= 79_000                                   # a dropped frame leaves a ~2-frame gap
    assert len(pts) < 60 * 25


def test_timecode_decodes_every_aligned_window_and_rejects_misaligned():
    scene = F.slot_swap_every_chunk()
    src = scene.source()
    for start in (0, 32_000, 64_000, 8 * 32_000, 640):
        a = src.read_audio(start * 3, (start + 64_000) * 3, 16_000).mean(axis=0)
        assert F.decode_window_start(a) == start
    for off in (1, 100, 320, 639):                             # never a silent wrong answer
        a = src.read_audio((32_000 + off) * 3, (96_000 + off) * 3, 16_000).mean(axis=0)
        with pytest.raises(F.TimecodeError):
            F.decode_window_start(a)
    with pytest.raises(F.TimecodeError):
        F.decode_window_start(np.random.default_rng(0).standard_normal(4000) * 0.1)


def test_separator_really_permutes_and_stems_are_oracle():
    scene = F.slot_swap_every_chunk()
    sep = F.FakeSeparator(scene, n_slots=2, perm="alternate")
    slots = {}
    for k in (2, 3):                                           # both A and B are active in chunks 2 and 3
        s = k * 32_000
        audio, _, _ = chunk_inputs(scene, s, s + 64_000)
        out = sep(audio)
        assert out.shape == (2, 64_000)
        call = sep.calls[-1]
        assert call["chunk"] == k and set(call["active"]) == {"A", "B"}
        for spk in "AB":
            ref = scene.speaker_signal(spk, s, s + 64_000, 16_000)
            j = call["slot_of"][spk]
            assert corr(out[j], ref) > 0.99 and corr(out[1 - j], ref) < 0.3
        slots[k] = call["slot_of"]["A"]
    assert slots[2] != slots[3], "slot order must flip between consecutive chunks"
    assert sep.calls[0]["perm"] == [0, 1] and sep.calls[1]["perm"] == [1, 0]


def test_separator_capacity_fewer_slots_than_speakers_and_random_schedule():
    scene = F.slot_swap_every_chunk()
    audio, _, _ = chunk_inputs(scene, 3 * 32_000, 3 * 32_000 + 64_000)
    one = F.FakeSeparator(scene, n_slots=1)(audio)
    ra = scene.speaker_signal("A", 96_000, 160_000, 16_000)
    rb = scene.speaker_signal("B", 96_000, 160_000, 16_000)
    assert one.shape == (1, 64_000) and 0.2 < corr(one[0], ra) < 0.95 and 0.2 < corr(one[0], rb) < 0.95    # degraded
    rnd = F.FakeSeparator(scene, perm="random")
    perms = set()
    for k in range(8):
        a, _, _ = chunk_inputs(scene, k * 32_000, k * 32_000 + 64_000)
        rnd(a)
        perms.add(tuple(rnd.calls[-1]["perm"]))
    assert perms == {(0, 1), (1, 0)}
    with pytest.raises(ValueError):
        F.FakeSeparator(scene, perm=lambda c, n: [0, 0])(audio)


def test_scorer_logits_follow_ground_truth_with_configurable_misses():
    scene = F.pause_return()                                   # A speaks [1, 7)
    audio, frames, vc = chunk_inputs(scene, 0, 64_000)
    boxes = vc.boxes_by_track["trk_pA"]
    raw, _, info = F.FakeScorer(scene).score_track(audio, boxes, frames, 25.0)
    assert len(raw) == 100 and info["fake"] == 1.0
    assert (raw[:24] < 0).all() and (raw[27:98] > 0).all()     # on at 1 s = frame 25 through the window end
    raw_b, *_ = F.FakeScorer(scene).score_track(audio, vc.boxes_by_track["trk_pB"], frames, 25.0)
    assert (raw_b < 0).all()                                   # B silent in [0, 4)
    miss, *_ = F.FakeScorer(scene, misses={"pA": [(2, 3)]}).score_track(audio, boxes, frames, 25.0)
    assert (miss[52:73] < 0).all() and (miss[27:48] > 0).all() and (miss[78:98] > 0).all()
    fa, *_ = F.FakeScorer(scene, false_alarms={"pB": [(0, 1)]}).score_track(audio, vc.boxes_by_track["trk_pB"], frames, 25.0)
    assert (fa[2:23] > 0).all() and (fa[30:] < 0).all()


def test_tse_returns_the_target_stem_and_silence_for_bystanders():
    scene = F.silent_bystander()
    audio, frames, vc = chunk_inputs(scene, 32_000, 96_000)    # 2 s to 6 s: A speaking
    a, _ = F.FakeTSE(scene).extract_target(audio, vc.boxes_by_track["trk_pA"], frames, 25.0)
    b, _ = F.FakeTSE(scene).extract_target(audio, vc.boxes_by_track["trk_pB"], frames, 25.0)
    assert corr(a, scene.speaker_signal("A", 32_000, 96_000, 16_000)) > 0.99
    assert np.sqrt(np.mean(b ** 2)) < 0.01 < np.sqrt(np.mean(a ** 2))


def test_fake_embed_separates_voices_and_twins_are_ambiguous():
    bank = F.VOICE_BANK
    emb = {k: F.fake_embed(F.voice_signal(v, 0, 32_000, 16_000)) for k, v in bank.items()}
    keys = list(emb)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert float(emb[a] @ emb[b]) < 0.9, (a, b)
    again = F.fake_embed(F.voice_signal(bank["A"], 48_000, 80_000, 16_000) * 0.3)      # other time, other level
    assert float(emb["A"] @ again) > 0.97
    twin = F.fake_embed(F.voice_signal(F.twin_of(bank["A"], "A2"), 0, 32_000, 16_000))
    assert float(emb["A"] @ twin) > 0.99
    assert not F.fake_embed(np.zeros(16_000)).any()


def test_visual_front_end_reentry_track_ids_and_json_state():
    scene = F.reentry()                                        # A visible [0,12) and [24,40), new track id on return
    fe = F.FakeVisualFrontEnd(scene)
    src, pts, state, seen = scene.source(), scene.video_pts_us(), {}, set()
    for k in range(0, 20):                                     # 2 s steps
        i0, i1 = C.video_index_range(pts, k * 2_000_000, (k + 2) * 2_000_000)
        vc = fe.step(src.read_frames(i0, i1), pts[i0:i1], state)
        state = json.loads(json.dumps(vc.state))               # must be JSON-safe
        assert all(len(v) == i1 - i0 for v in vc.boxes_by_track.values())
        assert all(m is None for ms in vc.masks_by_track.values() for m in ms)
        seen |= set(vc.boxes_by_track)
    assert seen == {"trk_pA", "trk_pA.r1", "trk_pB"}
    gt = scene.ground_truth
    assert gt.track_person == {"trk_pA": "pA", "trk_pA.r1": "pA", "trk_pB": "pB"}
    assert gt.track_visible["trk_pA"] == [(0, 12 * 16000)] and gt.track_visible["trk_pA.r1"] == [(24 * 16000, 40 * 16000)]
    assert state["tracks"]["trk_pA.r1"]["frames"] > 300
    same = F.reentry(new_track=False).ground_truth
    assert set(same.track_visible) == {"trk_pA", "trk_pB"} and len(same.track_visible["trk_pA"]) == 2
    drop = F.FakeVisualFrontEnd(scene, dropouts={"pB": [(0, 100)]}).step(src.read_frames(0, 50), pts[:50], {})
    assert all(b is None for b in drop.boxes_by_track["trk_pB"])


def test_ground_truth_per_scenario():
    gt = F.pause_return().ground_truth
    assert gt.speaker_active["A"] == [(16_000, 7 * 16_000), (33 * 16_000, 41 * 16_000)]
    assert gt.speaker_active["A"][1][0] - gt.speaker_active["A"][0][1] >= 10 * 32_000       # 10+ chunk hops of silence
    gt = F.silent_bystander().ground_truth
    assert gt.silent_bystander == {"pA": False, "pB": True}
    assert gt.offscreen_speakers == ["C"] and gt.should_refuse == [(14 * 16_000, 20 * 16_000)]
    assert gt.silent_visible_by_track["trk_pB"] == [(0, 24 * 16_000)]
    assert gt.silent_visible_by_track["trk_pA"] == [(0, 16_000), (12 * 16_000, 24 * 16_000)]
    gt = F.reentry().ground_truth                              # A speaks [14, 19) while absent -> refuse
    assert gt.should_refuse == [(14 * 16_000, 19 * 16_000)]
    assert gt.onscreen["A"] == [(2 * 16_000, 10 * 16_000), (26 * 16_000, 36 * 16_000)]
    gt = F.silent_overlap().ground_truth
    assert LM.intersect(gt.speaker_active["A"], gt.speaker_active["B"]) == [(12 * 16_000, 16 * 16_000)]
    assert gt.silent_bystander["pC"] and not gt.should_refuse


def test_silent_overlap_boxes_overlap_but_colours_stay_identifiable():
    scene = F.silent_overlap()
    audio, frames, vc = chunk_inputs(scene, 0, 64_000)
    a, c = vc.boxes_by_track["trk_pA"][0], vc.boxes_by_track["trk_pC"][0]
    assert a[2] > c[0], "boxes should overlap"
    assert scene.person_at_pixel(frames[0], a) == "pA" and scene.person_at_pixel(frames[0], c) == "pC"


def test_final_partial_is_not_a_hop_multiple_and_last_window_decodes():
    scene = F.final_partial()
    plan = C.plan_chunks(scene.n16)
    assert scene.n16 % 32_000 != 0 and plan[-1].win_len < 64_000 and plan[-1].win_start % 640 == 0
    assert scene.n48 == 3 * scene.n16
    src = scene.source()
    last = src.read_audio(plan[-1].win_start * 3, plan[-1].win_end * 3, 16_000).mean(axis=0)
    assert F.decode_window_start(last) == plan[-1].win_start
    assert src.read_audio(0, scene.n48, 48_000).shape == (2, scene.n48)


def test_build_scene_specs():
    assert F.build_scene("matrix:120").duration_s == 120.0
    assert F.build_scene("matrix", duration_s=30).duration_s == 30.0
    assert F.build_scene("crossing::3").seed == 3
    with pytest.raises(ValueError):
        F.build_scene("crossing:99")
    with pytest.raises(ValueError):
        F.build_scene("nope")
    a, b = F.build_scene("matrix:30"), F.build_scene("matrix:300")
    assert len(a.people) == len(b.people) and (a.frame_h, a.frame_w) == (b.frame_h, b.frame_w)


def test_fakes_drive_the_real_speech_branch_end_to_end():
    """Protocol compatibility: run_speech_branch (real code) over fake models, on slot-swapped windows."""
    scene = F.slot_swap_every_chunk()
    for k in (2, 3):
        s = k * 32_000
        audio, frames, vc = chunk_inputs(scene, s, s + 64_000)
        m = F.make_speech_models(scene)
        res = run_speech_branch(audio.astype(np.float32), frames, 25.0, vc.boxes_by_track, separator=m.separator,
                                scorer=m.scorer, tse=m.tse)
        assert set(res.selection) == {"trk_pA", "trk_pB"}
        for tid, (stream, ext) in res.selection.items():
            assert ext.status in ("ok", "degraded"), (k, tid, ext)
            spk = "A" if tid.endswith("pA") else "B"
            assert corr(res.streams[stream][: 64_000], scene.speaker_signal(spk, s, s + 64_000, 16_000)) > 0.95


def test_oracle_segments_are_self_consistent():
    for name in ("reentry", "silent_bystander", "pause_return"):
        scene = F.build_scene(name)
        segs = F.oracle_segments(scene)
        assert LM.total(LM.merge((s.start, s.end) for s in segs)) == LM.total(LM.union_of(scene.ground_truth.speaker_active))


def run_branch(scene, start, **kw):
    audio, frames, vc = chunk_inputs(scene, start, start + 64_000)
    m = F.make_speech_models(scene)
    return run_speech_branch(audio.astype(np.float32), frames, 25.0, vc.boxes_by_track, separator=m.separator,
                             scorer=m.scorer, tse=m.tse, **kw)


def test_empty_slots_read_as_silent_not_as_phantom_voices():
    scene = F.pause_return()                                   # nobody speaks in [24, 28); only A in [0, 4)
    res = run_branch(scene, 24 * 16_000)
    assert res.info["n_suspected_talkers"] == 0
    assert all(np.sqrt(np.mean(v ** 2)) < 1e-3 for v in res.streams.values()) and set(res.streams) == {"voice_0", "voice_1"}
    notes = [c.note for a in res.associations for c in a.components]
    assert notes.count("voice stream near-silent") == 2
    assert all(a.visibility == "unknown" and a.status == "unmatched" for a in res.associations)
    one = run_branch(scene, 0)
    assert one.info["n_suspected_talkers"] == 1
    st = {t: e.status for t, (_, e) in one.selection.items()}
    assert st["trk_pA"] == "ok" and st["trk_pB"] == "unavailable"
    notes = sorted(c.note for a in one.associations for c in a.components if a.status == "unmatched")
    assert "voice stream near-silent" in " ".join(notes)       # the empty slot, whichever index the permutation put it at
    two = run_branch(F.slot_swap_every_chunk(), 3 * 32_000)
    assert two.info["n_suspected_talkers"] == 2


def test_silent_bystander_stays_inactive_with_the_activity_gate():
    scene = F.silent_bystander()
    res = run_branch(scene, 32_000, activity_gate=True)        # A speaking [2, 6); B never speaks
    sel = {t: (s, e.status, e.reason) for t, (s, e) in res.selection.items()}
    assert sel["trk_pA"][1] == "ok", sel
    assert sel["trk_pB"][0] is None and sel["trk_pB"][1] == "unavailable", sel
    assert "trk_pB" not in res.tse and res.info["n_suspected_talkers"] == 1


def test_contract_classes_are_the_pipelines_own():
    assert F.AvMap is C.AvMap and F.SpeechModels is C.SpeechModels and F.VisualChunk is C.VisualChunk
    assert F.make_speech_models(F.pause_return()).revisions == {"fake_models": "long_fixtures-v1"}
    assert isinstance(F.pause_return().source().av_map, C.AvMap)
