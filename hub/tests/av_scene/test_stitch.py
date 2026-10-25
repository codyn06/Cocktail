"""Chunk stitching: coverage, gapless crossfade, permutation safety, mismatch guard."""
import numpy as np

from av_scene.contracts import plan_chunks
from av_scene.stitch import blend, crossfade_samples, emit_chunk, final_region

SR = 16000


def _run(n, streams_fn, f=800):
    specs = plan_chunks(n)
    prev = {}
    out = {}
    flags = {}
    for sp in specs:
        win = {sid: fn(sp.win_start, sp.win_end) for sid, fn in streams_fn.items() if fn(sp.win_start, sp.win_end) is not None}
        finals, tails, fl = emit_chunk(sp, n, f, win, prev)
        for sid, (s, a) in finals.items():
            out.setdefault(sid, []).append((s, a))
        prev = tails
        flags.update(fl)
    return specs, out, flags


def _tone(f0):
    return lambda a, b: np.sin(2 * np.pi * f0 * np.arange(a, b) / SR).astype(np.float32) * 0.5


def test_final_spans_tile_exactly_and_a_coherent_signal_is_reproduced():
    for secs in (3.0, 4.0, 4.3, 9.7, 30.0):
        n = int(secs * SR)
        specs, out, flags = _run(n, {"src_0": _tone(220.0)})
        pos = 0
        parts = []
        for s, a in out["src_0"]:
            assert s == pos                                  # no gap, no duplicate
            parts.append(a)
            pos += len(a)
        assert pos == n and not flags
        y = np.concatenate(parts)
        assert np.max(np.abs(y - _tone(220.0)(0, n))) < 1e-5     # coherent chunks stitch to the original


def test_last_region_shorter_than_crossfade_is_handled():
    n = 4 * SR + 5                                            # final partial chunk owns a few samples past the boundary
    specs, out, _ = _run(n, {"src_0": _tone(300.0)}, f=800)
    assert sum(len(a) for _, a in out["src_0"]) == n


def test_source_absent_on_one_side_fades_instead_of_borrowing_another_stream():
    n = 12 * SR
    specs = plan_chunks(n)
    tone_a, tone_b = _tone(200.0), _tone(900.0)
    f = 800
    prev = {}
    seen = {}
    for sp in specs:
        win = {"src_0": tone_a(sp.win_start, sp.win_end)}
        if sp.index >= 2:
            win = {"src_1": tone_b(sp.win_start, sp.win_end)}   # src_0 vanishes (e.g. ambiguous identity)
        finals, prev, _ = emit_chunk(sp, n, f, win, prev)
        seen[sp.index] = finals
    # chunk 2 only emits a fade-out of src_0's tail and a fade-in for src_1; src_1 never contains src_0's tone
    s0, a0 = seen[2]["src_0"]
    assert len(a0) == f and abs(a0[-1]) <= abs(a0[0]) + 0.05 and np.max(np.abs(a0)) <= 0.5
    s1, a1 = seen[2]["src_1"]
    assert abs(a1[0]) < 0.05                                    # faded in from silence
    freqs = np.fft.rfftfreq(len(a1), 1 / SR)
    assert abs(freqs[np.argmax(np.abs(np.fft.rfft(a1)))] - 900.0) < 30


def test_mismatching_streams_are_not_blended():
    m = 800
    t = np.arange(m)
    a = np.sin(2 * np.pi * 200 * t / SR).astype(np.float32) * 0.5
    b = np.sin(2 * np.pi * 530 * t / SR + 1.0).astype(np.float32) * 0.5
    out, why = blend(a, b)
    assert why and why.startswith("boundary_mismatch")
    assert np.array_equal(out[: m // 2], a[: m // 2]) and np.array_equal(out[m // 2:], b[m // 2:])   # hard switch
    out2, why2 = blend(a, a)
    assert why2 is None and np.allclose(out2, a, atol=1e-6)


def test_region_and_crossfade_helpers():
    assert crossfade_samples(50.0, SR) == 800 and crossfade_samples(51.0, SR) % 2 == 0
    sp = plan_chunks(10 * SR)
    r = [final_region(c, 10 * SR, 800) for c in sp]
    assert r[0][0] == 0 and r[-1][1] == 10 * SR
    assert all(r[i][1] + 800 == r[i + 1][0] + 800 + 0 or r[i][1] == r[i + 1][0] + 800 for i in range(len(r) - 1))
