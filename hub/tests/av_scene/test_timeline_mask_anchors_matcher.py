"""Timeline, mask codec, anchors, matcher and off-axis learning gate (Phase 1 contract 5)."""

import json
from pathlib import Path

import numpy as np
import pytest

from ingest.media_io import (MediaTimeline, build_media_index, detect_dropped_frames, to_common_us, from_common_us,
                             write_audio_stream)
from av_scene.schemas import GapInterval, StreamTransform
from separate import diarization as dz
from separate.voice_memory import SR, VoiceMemory, VoiceMemoryConfig
from vision.mask_store import (FrameTransform, MaskRecord, decode_mask, encode_mask, read_track_jsonl,
                               write_track_jsonl, RleMask)
from who.matcher import (EligibilityGatedVoiceMemory, Evidence, FaceObservation, LearningEligibility,
                         MatcherConfig, ScoreResult, assign, causal_lag_score, combine, lag_evidence,
                         lip_asd_available)

GOLDEN = json.loads((Path(__file__).parent / "golden_rle.json").read_text())


# ------------------------------------------------------------------ timeline
def vfr_pts():
    rng = np.random.default_rng(3)
    pts = np.cumsum(np.full(100, 40_000) + rng.integers(-3000, 3000, 100))
    pts = np.delete(pts, 50)                       # one dropped frame
    return [int(p) for p in pts]


def test_vfr_timeline_never_uses_nominal_fps_and_flags_dropped_frames(tmp_path):
    pts = vfr_pts()
    audio = write_audio_stream(tmp_path, "mic", np.zeros((48000, 2), np.float32), 48000, rel_to=tmp_path)
    idx = build_media_index("t", video_pts_us=pts, width=320, height=240, audio=[audio], nominal_fps=25.0)
    tl = MediaTimeline(idx)
    for k in (0, 10, 49, 50, 70, 98):
        assert tl.frame_at(pts[k]) == k and tl.frame_at(pts[k] + 1) == k
    assert tl.frame_at(pts[0] - 1) == -1
    naive = [int(i * 40_000) for i in range(len(pts))]
    assert max(abs(a - b) for a, b in zip(pts, naive)) > 50_000        # timing is NOT idx / nominal_fps
    k = 70
    assert tl.quantization_ms_at(pts[k] + 5) == pytest.approx((pts[k + 1] - pts[k]) / 2000.0)
    drops = [g for g in idx.gaps if g.kind == "missing_frames"]
    assert len(drops) == 1 and drops[0].start_us == pts[49]
    assert tl.has_missing_data(pts[49], pts[50])
    with pytest.raises(ValueError):
        build_media_index("t", video_pts_us=[5, 5, 9], width=1, height=1, audio=[])


def test_common_clock_transform_and_audio_sample_mapping(tmp_path):
    tf = StreamTransform(a=1.0001, b_us=2500.0, confidence=0.9, method="drift_fit")
    for t in (0, 1_000_000, 7_654_321):
        assert from_common_us(to_common_us(t, tf), tf) == pytest.approx(t, abs=1.0)
    audio = write_audio_stream(tmp_path, "mic", np.zeros((48000, 1), np.float32), 48000, start_us=10_000,
                               transform=tf, rel_to=tmp_path)
    idx = build_media_index("t", video_pts_us=[0, 40_000], width=2, height=2, audio=[audio])
    tl = MediaTimeline(idx)
    for s in (0, 12345, 47999):
        assert tl.common_us_to_sample(tl.sample_to_common_us(s)) == pytest.approx(s, abs=0.1)
    assert audio.views["16k"].n_samples == 16000 and audio.views["48k"].n_samples == 48000
    assert audio.sample_rate == 48000 and audio.channels == 1       # master preserved at capture rate


def test_gaps_and_padding_are_explicit_not_silence():
    idx = build_media_index("t", video_pts_us=[0, 40_000, 80_000], width=2, height=2, audio=[],
                            extra_gaps=[GapInterval(stream_id="mic", start_us=100, end_us=900, kind="padding")])
    tl = MediaTimeline(idx)
    assert tl.has_missing_data(0, 500, "mic") and not tl.has_missing_data(1000, 2000, "mic")
    assert detect_dropped_frames([0, 40_000]) == []


# ------------------------------------------------------------------ RLE
@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_golden_rle_vectors(case):
    m = np.array([[ch == "1" for ch in row] for row in case["rows"]])
    r = encode_mask(m)
    assert list(r.counts) == case["counts"] and r.order == "row-major" and r.foreground is True
    assert np.array_equal(decode_mask(r), m)


def test_rle_round_trips_random_empty_full_and_nonsquare():
    rng = np.random.default_rng(0)
    shapes = [(1, 1), (7, 13), (240, 320), (13, 7)]
    for h, w in shapes:
        for m in (rng.random((h, w)) > 0.5, np.zeros((h, w), bool), np.ones((h, w), bool),
                  np.triu(np.ones((h, w), bool))):
            r = encode_mask(m)
            assert sum(r.counts) == h * w and np.array_equal(r.decode(), m)
            r2 = RleMask.from_dict(json.loads(json.dumps(r.to_dict())))
            assert r2 == r
    with pytest.raises(ValueError):
        decode_mask(RleMask(2, 2, (1, 1), FrameTransform.identity(2, 2), None))


def test_bbox_is_normalised_to_the_original_frame_through_the_transform(tmp_path):
    m = np.zeros((60, 80), bool)
    m[10:30, 20:60] = True                                    # model-resolution mask
    tf = FrameTransform.resize(480, 640, 60, 80)               # original frame is 8x larger
    r = encode_mask(m, tf)
    assert r.bbox_norm == pytest.approx((20 / 80, 10 / 60, 60 / 80, 30 / 60))
    crop = FrameTransform.crop_resize(480, 640, 100, 50, 320, 240, 60, 80)
    assert encode_mask(m, crop).bbox_norm == pytest.approx(((100 + 20 * 4) / 640, (50 + 10 * 4) / 480,
                                                            (100 + 60 * 4) / 640, (50 + 30 * 4) / 480))
    m2 = np.zeros((60, 80), bool)
    m2[10:30, 10:30] = True
    flipped = FrameTransform.identity(60, 80).mirrored(80)      # mask was mirrored before encoding
    assert encode_mask(m2, flipped).bbox_norm == pytest.approx((50 / 80, 10 / 60, 70 / 80, 30 / 60))
    assert encode_mask(np.zeros((4, 4), bool)).bbox_norm is None
    recs = [MaskRecord(1_000, 0, r, 0.9), MaskRecord(41_000, 1, None, 0.0, "occluded", "propagated", 3)]
    p = tmp_path / "t.jsonl"
    write_track_jsonl(p, recs)
    back = read_track_jsonl(p)
    assert back[0].rle == r and back[1].rle is None and back[1].visibility == "occluded" and back[1].age_frames == 3


# ------------------------------------------------------------------ interval algebra and anchors
def bitmap(iv, n=200):
    b = np.zeros(n, bool)
    for s, e in iv:
        b[s:e] = True
    return b


def test_half_open_algebra_matches_bitmaps():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = [tuple(sorted(rng.integers(0, 200, 2))) for _ in range(4)]
        b = [tuple(sorted(rng.integers(0, 200, 2))) for _ in range(4)]
        A, B = bitmap(a), bitmap(b)
        assert np.array_equal(bitmap(dz.union(a, b)), A | B)
        assert np.array_equal(bitmap(dz.subtract(a, b)), A & ~B)
        assert np.array_equal(bitmap(dz.intersect(a, b)), A & B)
    assert dz.normalize([(0, 5), (5, 9)]) == [(0, 9)]                 # touching half-open intervals merge
    assert dz.normalize([(3, 3), (4, 2)]) == []


def test_anchor_positive_and_negative_candidates_with_provenance():
    s = 1_000_000
    acts = {"A": [(0, 4 * s)], "B": [(3 * s, 7 * s)], "C": [(6_500_000, 9 * s)]}
    noise = [(8 * s, 8_400_000)]
    out = dz.compute_anchors(acts, noise)
    assert out["A"].positive == [(50_000, 2_950_000)] and out["A"].empty_reason is None
    assert out["B"].positive == [(4_050_000, 6_450_000)]
    assert out["C"].positive == [(7_050_000, 7_950_000), (8_450_000, 8_950_000)]
    assert out["A"].provenance["overlapped_by_other_speakers_us"] == 1_000_000
    for a in out.values():                                       # negative never covers positive or own speech
        assert dz.intersect(a.negative, a.positive) == [] and dz.intersect(a.negative, acts[a.speaker_id]) == []
    assert out["A"].negative and out["A"].negative[0][0] >= 4 * s


def test_empty_anchor_sets_are_valid_with_a_reason():
    s = 1_000_000
    out = dz.compute_anchors({"A": [(0, 2 * s)], "B": [(0, 2 * s)], "D": [(5 * s, 8 * s)],
                              "E": [(10 * s, 12_200_000)], "F": [(10 * s, 12 * s)], "G": []})
    assert out["A"].positive == [] and out["A"].empty_reason == "total_overlap"
    assert out["B"].positive == [] and out["B"].empty_reason == "total_overlap"
    assert out["E"].positive == [] and out["E"].empty_reason == "too_short"   # 0.2 s solo after trimming
    assert out["G"].empty_reason == "no_activity"
    assert out["D"].positive and out["D"].empty_reason is None
    assert dz.intersect(out["A"].negative, [(0, 2 * s)]) == []                # nothing manufactured for total overlap


def test_gap_merging_never_bridges_into_other_speakers():
    s = 100_000
    out = dz.compute_anchors({"A": [(0, 10 * s), (12 * s, 30 * s)], "B": [(10 * s, 12 * s)]},
                             cfg=dz.AnchorConfig(merge_gap_us=3 * s, trim_us=0, min_duration_us=s))
    assert dz.intersect(out["A"].positive, [(10 * s, 12 * s)]) == []


# ------------------------------------------------------------------ matcher
CFG = MatcherConfig()


def test_causal_tof_window_is_free_and_audio_lead_is_penalised():
    for lag in (0.0, 10.0, 29.0, 35.0, -3.0):
        assert causal_lag_score(lag, CFG) == 1.0
    assert causal_lag_score(-10.0, CFG) == pytest.approx(0.75)
    assert causal_lag_score(-25.0, CFG) == 0.0
    assert causal_lag_score(36.0, CFG) < 1.0 and causal_lag_score(80.0, CFG) == 0.0
    assert causal_lag_score(-20.0, CFG, quantization_ms=20.0) == 1.0   # frame quantization widens uncertainty
    assert causal_lag_score(-10.0, CFG) > causal_lag_score(-15.0, CFG) > causal_lag_score(-20.0, CFG)
    ev = lag_evidence([5.0, 12.0, 20.0], CFG)
    assert ev.value == 1.0 and lag_evidence([], CFG).value is None


def test_unavailable_evidence_is_not_a_zero():
    good = [Evidence("sync", 0.9), Evidence("spatial", 0.9)]
    with_none = combine(good + [Evidence("asd", None, note="off-axis lips")], CFG)
    with_zero = combine(good + [Evidence("asd", 0.0)], CFG)
    assert with_none.score == pytest.approx(0.9) and with_zero.score == pytest.approx(0.6)
    assert combine([Evidence("a", None), Evidence("b", None)], CFG).score is None
    assert combine([Evidence("a", 0.9, quality=0.1)], CFG).score is None   # too little reliable evidence


def sr(score):
    return ScoreResult(score, [Evidence("x", score)], 1.0)


def test_assignment_matches_abstains_and_flags_ambiguity():
    d = assign(["s1", "s2"], ["alice", "bob", "silent"], {("s1", "alice"): sr(0.9), ("s1", "bob"): sr(0.2),
                                                         ("s1", "silent"): sr(0.1), ("s2", "bob"): sr(0.85),
                                                         ("s2", "alice"): sr(0.3), ("s2", "silent"): sr(0.1)}, CFG)
    assert (d["s1"].track_id, d["s1"].status) == ("alice", "matched")
    assert (d["s2"].track_id, d["s2"].status) == ("bob", "matched")
    assert "silent" not in {x.track_id for x in d.values()}                     # silent bystander never matched
    assert d["s1"].margin == pytest.approx(0.9 - 0.2) and d["s1"].runner_up_id == "bob"

    off = assign(["offscreen"], ["cat1", "cat2"], {("offscreen", "cat1"): sr(0.4), ("offscreen", "cat2"): sr(0.3)}, CFG)
    assert off["offscreen"].status == "unmatched" and off["offscreen"].track_id is None   # stays unmatched
    none = assign(["x"], ["cat1"], {("x", "cat1"): ScoreResult(None, [], 0.0)}, CFG)
    assert none["x"].status == "unmatched" and "no available evidence" in none["x"].reason

    amb = assign(["chirp"], ["cat1", "cat2"], {("chirp", "cat1"): sr(0.80), ("chirp", "cat2"): sr(0.78)}, CFG)
    assert amb["chirp"].status == "ambiguous" and amb["chirp"].track_id == "cat1"
    assert amb["chirp"].candidates["cat2"]["score"] == pytest.approx(0.78)

    tie = assign(["s1", "s2"], ["A", "B"], {("s1", "A"): sr(0.9), ("s1", "B"): sr(0.1), ("s2", "A"): sr(0.85),
                                           ("s2", "B"): sr(0.8)}, CFG)
    assert tie["s1"].track_id == "A" and tie["s2"].track_id == "B"          # one-to-one

    tv = assign(["voice1", "voice2"], ["tv"], {("voice1", "tv"): sr(0.9), ("voice2", "tv"): sr(0.8)}, CFG,
                shared_emitter_tracks={"tv"})
    assert tv["voice1"].track_id == "tv" and tv["voice2"].track_id == "tv"  # one emitter, two voices
    solo = assign(["voice1", "voice2"], ["tv"], {("voice1", "tv"): sr(0.9), ("voice2", "tv"): sr(0.8)}, CFG)
    assert sum(x.track_id == "tv" for x in solo.values()) == 1

    prev = assign(["s"], ["A", "B"], {("s", "A"): sr(0.70), ("s", "B"): sr(0.72)}, CFG, previous={"s": "A"})
    assert prev["s"].track_id == "A"                                        # continuity bonus
    assert not ({"matched", "ambiguous", "unmatched", "rejected"} & {"visible", "occluded", "outside_view", "unknown"})


# ------------------------------------------------------------------ off-axis voice-profile learning
def make_encoder():
    edges = np.geomspace(100, 6000, 9)

    def enc(audios):
        out = []
        for a in audios:
            a = np.asarray(a, float)[-16000:]
            sp = np.abs(np.fft.rfft(a))
            f = np.fft.rfftfreq(len(a), 1 / SR)
            v = np.array([sp[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])]) + 1e-6
            out.append(v / np.linalg.norm(v))
        return np.array(out)
    return enc


def voice(f0, secs=2.0, amp=0.2):
    t = np.arange(int(secs * SR)) / SR
    return amp * sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6))


SIL = np.zeros(2 * SR)
CFGM = VoiceMemoryConfig(profile_min_s=1.0, profile_update_s=1.0, pause_s=1.0)
S1 = 1_000_000


def run(mem, windows, t0=0):
    out = []
    for k, w in enumerate(windows):
        out.append(mem.reorder(np.stack(w), t0 + k * S1))
    return out


def snapshot(m, slot):
    return (None if m.profiles[slot] is None else m.profiles[slot].copy(), len(m._history[slot]),
            m._history_samples[slot], m._profile_active_s[slot], m._next_update_s[slot])


def test_off_axis_face_blocks_enrollment_history_and_counters():
    seq = [[voice(300), SIL]] * 3
    free = VoiceMemory(make_encoder(), CFGM)
    run(free, seq)
    assert free.profiles[0] is not None and len(free._history[0]) > 0 and free._profile_active_s[0] > 0   # control learns

    elig = LearningEligibility()
    assert elig.observe(0, FaceObservation(0, 10 * S1, yaw_deg=75.0)).reason.startswith("off_axis")
    gated = VoiceMemory(make_encoder(), CFGM)
    init_next = list(gated._next_update_s)
    run(EligibilityGatedVoiceMemory(gated, elig), seq)
    assert gated.profiles[0] is None and len(gated._history[0]) == 0
    assert gated._history_samples[0] == 0 and gated._profile_active_s[0] == 0.0
    assert gated._next_update_s[0] == init_next[0]
    assert gated._last_active_us[0] is not None                      # recognition bookkeeping still ran


def test_occlusion_and_unknown_pose_policy():
    e = LearningEligibility()
    assert e.observe(0, FaceObservation(0, S1, yaw_deg=10.0, occluded=True)).reason == "mouth_occluded"
    assert e.observe(1, FaceObservation(0, S1, yaw_deg=None)).eligible                # explicit 'allow' default
    assert not LearningEligibility("block").observe(1, FaceObservation(0, S1, yaw_deg=None)).eligible
    assert e.observe(1, FaceObservation(0, S1, yaw_deg=-61.0)).eligible is False
    assert not lip_asd_available(FaceObservation(0, S1, yaw_deg=61.0)) and not lip_asd_available(None)
    assert lip_asd_available(FaceObservation(0, S1, yaw_deg=-59.0))
    assert not lip_asd_available(FaceObservation(0, S1, yaw_deg=0.0, occluded=True))


def test_off_axis_blocks_adaptation_but_not_clean_intervals():
    clean = [[voice(300), SIL]] * 3
    drift = [[voice(1500), SIL]] * 3                                     # would pull the profile if learned
    free = VoiceMemory(make_encoder(), CFGM)
    run(free, clean + drift)

    elig = LearningEligibility()
    elig.observe(0, FaceObservation(0, 3 * S1, yaw_deg=5.0))              # forward during the clean part
    elig.observe(0, FaceObservation(3 * S1, 9 * S1, yaw_deg=80.0))        # turned away afterwards
    gated = VoiceMemory(make_encoder(), CFGM)
    g = EligibilityGatedVoiceMemory(gated, elig)
    run(g, clean)
    after_clean = snapshot(gated, 0)
    assert after_clean[0] is not None and after_clean[2] > 0              # clean intervals still teach the profile
    run(g, drift, t0=3 * S1)
    after_drift = snapshot(gated, 0)
    assert np.array_equal(after_clean[0], after_drift[0]) and after_clean[1:] == after_drift[1:]
    assert float(np.dot(free.profiles[0], after_clean[0])) < 0.999        # control profile did move
    # a later clear face cannot approve the contaminated interval
    elig.observe(0, FaceObservation(0, 20 * S1, yaw_deg=0.0))
    assert elig.blocked(0, 3 * S1, 4 * S1)[0] and not elig.blocked(0, 12 * S1, 13 * S1)[0]


@pytest.mark.parametrize("limit,yaw,eligible", [(30.0, 45.0, False), (30.0, 25.0, True),
                                                (80.0, 70.0, True), (80.0, 85.0, False),
                                                (60.0, 70.0, False)])
def test_off_axis_limit_is_configurable(limit, yaw, eligible):
    e = LearningEligibility(yaw_limit_deg=limit)
    assert e.observe(0, FaceObservation(0, S1, yaw_deg=yaw)).eligible is eligible
    assert e.observe(0, FaceObservation(0, S1, yaw_deg=-yaw)).eligible is eligible
    assert lip_asd_available(FaceObservation(0, S1, yaw_deg=yaw), limit) is eligible


def test_off_axis_with_custom_limit_never_adapts_but_still_recognises():
    clean = [[voice(300), SIL]] * 3
    drift = [[voice(1500), SIL]] * 3
    for limit, yaw in ((30.0, 40.0), (80.0, 85.0)):
        elig = LearningEligibility(yaw_limit_deg=limit)
        elig.observe(0, FaceObservation(0, 3 * S1, yaw_deg=0.0))
        elig.observe(0, FaceObservation(3 * S1, 9 * S1, yaw_deg=yaw))
        gated = VoiceMemory(make_encoder(), CFGM)
        g = EligibilityGatedVoiceMemory(gated, elig)
        run(g, clean)
        a = snapshot(gated, 0)
        assert a[0] is not None and a[2] > 0
        run(g, drift, t0=3 * S1)
        b = snapshot(gated, 0)
        assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]


def test_return_window_creation_is_blocked_but_recognition_continues():
    # slot 1 is silent for a second, then a new voice returns: VoiceMemory would create its profile directly
    seq = [[voice(300), SIL], [voice(300), voice(1500)], [voice(300), voice(1500)]]
    free = VoiceMemory(make_encoder(), CFGM)
    run(free, seq)
    assert free.profiles[1] is not None                                   # control: return-window creation happens
    elig = LearningEligibility()
    elig.observe(1, FaceObservation(0, 10 * S1, occluded=True))
    gated = VoiceMemory(make_encoder(), CFGM)
    run(EligibilityGatedVoiceMemory(gated, elig), seq)
    assert gated.profiles[1] is None and gated._profile_active_s[1] == 0.0 and len(gated._history[1]) == 0
    assert gated.profiles[0] is not None                                  # the eligible slot still learns

    # recognition: after a long pause the returning voices are mapped back using the existing clean profiles
    base = [[voice(300), voice(1500)]] * 3
    pause = [[SIL, SIL]] * 2
    swapped = [[voice(1500), voice(300)]]
    mem = VoiceMemory(make_encoder(), CFGM)
    e2 = LearningEligibility()
    gm = EligibilityGatedVoiceMemory(mem, e2)
    run(gm, base + pause)
    before = [None if p is None else p.copy() for p in mem.profiles]
    e2.observe(0, FaceObservation(5 * S1, 9 * S1, yaw_deg=90.0))
    e2.observe(1, FaceObservation(5 * S1, 9 * S1, yaw_deg=90.0))
    res, diag = gm.reorder(np.stack(swapped[0]), 5 * S1)
    assert diag["slot_mapping"] == {0: 1, 1: 0} and diag["returning"]
    assert np.allclose(res[1], voice(1500)) and np.allclose(res[0], voice(300))
    assert all(np.array_equal(a, b) for a, b in zip(before, mem.profiles))
    assert set(diag["learning_blocked"]) == {"s1", "s2"}


def test_png_roundtrip_with_every_row_filter(tmp_path):
    """read_png must decode all five PNG row filters, including values that wrap past 255."""
    import struct
    import zlib

    from ingest.media_io import read_png, write_png

    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, (9, 7, 3), dtype=np.uint8)
    write_png(tmp_path / "plain.png", img)
    assert np.array_equal(read_png(tmp_path / "plain.png"), img)
    h, w, bpp = img.shape[0], img.shape[1], 3
    flat = img.reshape(h, w * 3).astype(np.int32)
    rows = []
    for y in range(h):
        f = [1, 2, 3, 4][y % 4]
        cur = flat[y]
        prev = flat[y - 1] if y else np.zeros(w * 3, np.int32)
        out = np.zeros(w * 3, np.int32)
        for x in range(w * 3):
            a = cur[x - bpp] if x >= bpp else 0
            c = prev[x - bpp] if x >= bpp else 0
            if f == 1:
                pred = a
            elif f == 2:
                pred = prev[x]
            elif f == 3:
                pred = (a + prev[x]) // 2
            else:
                p0 = a + prev[x] - c
                pa, pb, pc = abs(p0 - a), abs(p0 - prev[x]), abs(p0 - c)
                pred = a if pa <= pb and pa <= pc else (prev[x] if pb <= pc else c)
            out[x] = (cur[x] - pred) & 0xFF
        rows.append(bytes([f]) + out.astype(np.uint8).tobytes())

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    png = (bytes([0x89]) + b"PNG" + bytes([0x0D, 0x0A, 0x1A, 0x0A]) + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))
    (tmp_path / "filtered.png").write_bytes(png)
    assert np.array_equal(read_png(tmp_path / "filtered.png"), img)
