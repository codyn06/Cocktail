"""IdentityStore tests (mocked: deterministic FFT band-energy embedder, synthetic voices)."""
import json
import shutil

import numpy as np
import pytest

from av_scene.associate import continuity
from av_scene.continuity_windows import (IdentityStore, SlotInput, stream_sims_for_continuity)
from av_scene.schemas import Association

SR = 16000
WIN = 4 * SR
OWN = (2 * SR, 4 * SR)          # chunk owns its trailing 2 s (overlap is owned by the previous chunk)
F0 = {"A": 200.0, "B": 330.0, "C": 470.0, "D": 610.0, "A2": 203.0}


def voice(name, t0, n=WIN, amp=0.1):
    t = (t0 + np.arange(n)) / SR
    f = F0[name]
    return (amp * (np.sin(2 * np.pi * f * t) + 0.6 * np.sin(2 * np.pi * 2 * f * t)
                   + 0.3 * np.sin(2 * np.pi * 3 * f * t + 0.4))).astype(np.float32)


def embed(a):
    a = np.asarray(a, dtype=np.float64)
    if len(a) == 0:
        return np.zeros(128)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    edges = np.linspace(0, len(sp), 129).astype(int)
    v = np.array([sp[edges[i]:edges[i + 1]].sum() for i in range(128)])
    v = np.sqrt(v)
    if v.sum() == 0:
        return np.zeros(128)
    return v - v.mean()


def chunk_slots(speakers, k, perm=None, **kw):
    """speakers: list of names or None (silent) per slot; chunk k's window starts at k*2 s."""
    t0 = k * 2 * SR
    order = list(range(len(speakers))) if perm is None else perm
    slots = {}
    for slot, s in zip(order, speakers):
        a = np.zeros(WIN, np.float32) if s is None else voice(s, t0)
        slots[slot] = SlotInput(a, OWN if k else (0, WIN), **kw)
    return slots


def new_store(**kw):
    return IdentityStore(embed, **kw)


def run(store, plan, permute=False, start=0):
    outs, ids = [], {}
    rng = np.random.default_rng(1)
    for k, speakers in enumerate(plan, start=start):
        perm = list(rng.permutation(len(speakers))) if permute else None
        out = store.assign_chunk(k, chunk_slots(speakers, k, perm))
        order = perm if perm is not None else range(len(speakers))
        for slot, s in zip(order, speakers):
            if s is not None and slot in out.mapping:
                ids.setdefault(s, set()).add(out.mapping[slot])
        outs.append(out)
    return outs, ids


def test_permutation_every_chunk_zero_switches():
    outs, ids = run(new_store(), [["A", "B", "C"]] * 8, permute=True)
    assert all(len(v) == 1 for v in ids.values()), ids
    assert len({next(iter(v)) for v in ids.values()}) == 3
    assert all(s == "resolved" for o in outs[1:] for s in o.status.values())


def test_pause_then_return_same_id():
    plan = [["A", "B"]] + [["A", None]] * 12 + [["A", "B"]]
    outs, ids = run(new_store(), plan)
    assert len(ids["B"]) == 1 and len(ids["A"]) == 1
    assert all(o.status[1] == "none" and o.reasons[1] == "insufficient_audio" for o in outs[1:13])
    assert sum(len(o.new_ids) for o in outs) == 2


def test_crossings_and_reentry():
    plan = [["A", "B"], ["B", "A"], ["A", "B"], ["A", None], ["C", "A"], ["A", "C"], ["B", "A"]]
    outs, ids = run(new_store(), plan)
    assert all(len(v) == 1 for v in ids.values()), ids
    assert len({next(iter(v)) for v in ids.values()}) == 3


def test_silent_slots_mint_nothing():
    st = new_store()
    out = st.assign_chunk(0, {0: SlotInput(np.zeros(WIN, np.float32), (0, WIN)),
                              1: SlotInput(np.zeros(WIN, np.float32), (0, WIN))})
    assert out.status == {0: "none", 1: "none"} and not out.new_ids and not out.mapping
    assert st.to_dict()["active"] == {}


def test_ambiguity_abstains_without_blending():
    st = new_store()
    run(st, [["A"]] * 3)
    before = json.dumps(st.to_dict(), sort_keys=True)
    t0 = 3 * 2 * SR
    out = st.assign_chunk(3, {0: SlotInput(voice("A", t0), OWN), 1: SlotInput(voice("A2", t0), OWN)})
    assert set(out.status.values()) == {"ambiguous"}
    assert not out.mapping and not out.new_ids and not out.enrolled_s
    assert all(r for r in out.reasons.values())
    after = st.to_dict()
    assert json.dumps(after, sort_keys=True).replace('"last_chunk": 2', '"last_chunk": 3') == \
        before.replace('"last_chunk": 2', '"last_chunk": 3')


def test_offaxis_excerpt_does_not_change_centroid():
    st = new_store()
    run(st, [["A"]] * 3)
    c0 = st.to_dict()["active"]["src_0"]["sum"]
    t0 = 3 * 2 * SR
    out = st.assign_chunk(3, {0: SlotInput(voice("A", t0), OWN, eligible=[])})
    assert out.status[0] == "resolved" and out.enrolled_s == {}
    assert st.to_dict()["active"]["src_0"]["sum"] == c0
    # only 0.5 s eligible inside own -> below min_enroll_s -> still no change
    out = st.assign_chunk(4, {0: SlotInput(voice("A", 4 * 2 * SR), OWN, eligible=[(OWN[0], OWN[0] + SR // 2)])})
    assert out.enrolled_s == {} and st.to_dict()["active"]["src_0"]["sum"] == c0
    # eligible only part of own: enrolled seconds equal the eligible excerpt length
    out = st.assign_chunk(5, {0: SlotInput(voice("A", 5 * 2 * SR), OWN, eligible=[(OWN[0], OWN[0] + SR + SR // 2)])})
    assert out.enrolled_s["src_0"] == pytest.approx(1.5)


def test_new_id_needs_eligible_audio():
    out = new_store().assign_chunk(0, {0: SlotInput(voice("A", 0), (0, WIN), eligible=[])})
    assert out.status[0] == "none" and out.reasons[0] == "insufficient_eligible_audio"


def test_short_utterance_abstains_without_blaming_eligibility():
    st = new_store()
    short = SlotInput(voice("A", 0), (0, WIN), eligible=None)
    short.audio = short.audio.copy()
    short.audio[SR // 2 + SR // 4:] = 0.0                      # 0.75 s of voiced owned audio < min_enroll_s
    out = st.assign_chunk(0, {0: short})
    assert out.status[0] == "none" and out.reasons[0] == "short_utterance" and not out.mapping and not out.new_ids


def test_enrolled_seconds_not_duplicated():
    outs, _ = run(new_store(), [["A", "B"]] * 6)
    total = sum(sum(o.enrolled_s.values()) for o in outs)
    owned = 4.0 + 5 * 2.0            # first chunk owns full window, later chunks own 2 s
    assert total == pytest.approx(2 * owned)           # 2 speakers
    assert total < 2 * 6 * 4.0                         # not window seconds


def test_resume_equals_uninterrupted(tmp_path):
    plan = [["A", "B", "C", "D"], ["B", "A", "D", "C"]] * 5
    plan2 = plan + [["D", "B", "A", "C"]] * 3

    def mk(d):
        return new_store(max_active=2, spill_dir=d)

    full = mk(tmp_path / "full")
    full_outs, _ = run(full, plan2)
    part = mk(tmp_path / "part")
    run(part, plan)
    snap = json.loads(json.dumps(part.to_dict()))
    shutil.copytree(tmp_path / "part", tmp_path / "resumed")
    resumed = IdentityStore.from_dict(snap, embed, spill_dir=tmp_path / "resumed")
    r_outs, _ = run(resumed, plan2[len(plan):], start=len(plan))
    for a, b in zip(full_outs[len(plan):], r_outs):
        assert a.mapping == b.mapping and a.status == b.status and a.enrolled_s == b.enrolled_s
    assert json.dumps(full.to_dict(), sort_keys=True) == json.dumps(resumed.to_dict(), sort_keys=True)
    assert json.loads(json.dumps(full.to_dict())) == full.to_dict()


def test_spill_cap_and_recall(tmp_path):
    st = new_store(max_active=2, spill_dir=tmp_path)
    plan = [["A", "B"], ["A", "B"], ["C", "D"], ["C", "D"], ["A", "B"], ["C", "A"], ["D", "B"]]
    seen = {}
    for k, speakers in enumerate(plan):
        out = st.assign_chunk(k, chunk_slots(speakers, k))
        assert st.resident_count <= 2 and st.peak_resident <= 3
        for slot, s in enumerate(speakers):
            seen.setdefault(s, set()).add(out.mapping[slot])
    assert all(len(v) == 1 for v in seen.values()), seen
    assert len(st.to_dict()["spilled"]) == 2
    assert any(tmp_path.glob("*.npz"))


def test_cap_without_spill_dir_raises():
    with pytest.raises(RuntimeError, match="spill_dir"):
        run(new_store(max_active=1), [["A", "B"]])


def test_short_final_window():
    st = new_store()
    run(st, [["A"]] * 3)
    c0 = st.to_dict()["active"]["src_0"]["sum"]
    short = voice("A", 6 * 2 * SR, n=SR // 2)
    out = st.assign_chunk(3, {0: SlotInput(short, (0, len(short)))})
    assert out.status[0] == "resolved" and out.enrolled_s == {}
    assert st.to_dict()["active"]["src_0"]["sum"] == c0
    tiny = voice("A", 0, n=SR // 10)
    out = st.assign_chunk(4, {0: SlotInput(tiny, (0, len(tiny)))})
    assert out.status[0] == "none"
    alone = new_store().assign_chunk(0, {0: SlotInput(short, (0, len(short)))})
    assert alone.status[0] == "none" and not alone.new_ids


def test_deterministic():
    plan = [["A", "B", "C"]] * 4
    a, _ = run(new_store(), plan, permute=True)
    b, _ = run(new_store(), plan, permute=True)
    assert [o.mapping for o in a] == [o.mapping for o in b]


def test_adapter_feeds_associate_continuity():
    st = new_store()
    outs, _ = run(st, [["A", "B"], ["B", "A"]])
    sims = stream_sims_for_continuity(outs[0], outs[1], st, name=lambda s: f"s{s}")
    same = [k for k, v in sims.items() if v == 1.0]
    assert sorted(same) == [("s0", "s1"), ("s1", "s0")]
    assert sims[("s0", "s0")] < 0.99

    def assoc(i, src, trk):
        return Association(association_id=i, source_id=src, track_id=trk, valid_start_us=0,
                           valid_end_us=1, status="matched", visibility="visible", runner_up_margin=0.0)
    prev = [assoc("p1", "s0", "T1"), assoc("p2", "s1", "T2")]
    curr = [assoc("c1", "s1", "T1"), assoc("c2", "s0", "T2")]   # slots swapped, same voices per track
    got = continuity(prev, curr, sims)
    assert [a.status for a in got] == ["matched", "matched"]
    got2 = continuity(prev, curr, {})
    assert [a.status for a in got2] == ["ambiguous", "ambiguous"]
