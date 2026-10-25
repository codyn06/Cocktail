import sys

import numpy as np
import pytest

from separate import diarization as dz
from separate import pyannote_diar as pd

TURNS = [("A", 0.0, 3.0), ("B", 2.0, 5.0), ("B", 5.0, 6.0), ("C", 0.0, 0.3)]


def test_diarize_with_fake_pipeline():
    p = pd.FakePipeline(TURNS)
    out = pd.diarize(np.zeros(16000, np.float32), 16000, p)
    assert p.calls == 1 and ("A", 0.0, 3.0) in out
    assert out == sorted(out, key=lambda t: (t[1], t[2], t[0]))


def test_timeline_us_and_merge():
    tl = pd.turns_to_timeline(TURNS)
    assert tl["A"] == [(0, 3_000_000)]
    assert tl["B"] == [(2_000_000, 6_000_000)]  # touching turns merged


def test_solo_and_competing_with_overlap():
    acts, noise = pd.turns_to_anchor_inputs(TURNS)
    res = dz.compute_anchors(acts, noise)
    t = dz.AnchorConfig().trim_us
    # A solo: 0.3..2s (C owns 0..0.3, B owns 2..3 overlap)
    assert res["A"].positive == [(300_000 + t, 2_000_000 - t)]
    assert res["B"].positive == [(3_000_000 + t, 6_000_000 - t)]
    # B is competing for A only outside A's own speech; the overlap stays out of every negative
    assert res["A"].negative and all(s >= 3_000_000 for s, _ in res["A"].negative)
    for sid in ("A", "B"):
        for iv in res[sid].negative:
            assert not dz.intersect([iv], acts[sid])


def test_full_overlap_gives_no_solo():
    acts, noise = pd.turns_to_anchor_inputs([("A", 0, 2), ("B", 0, 2)])
    res = dz.compute_anchors(acts, noise)
    assert res["A"].positive == [] and res["A"].empty_reason == "total_overlap"
    assert res["B"].empty_reason == "total_overlap"


def test_short_spans_filtered():
    acts, noise = pd.turns_to_anchor_inputs([("A", 0.0, 0.4), ("B", 5.0, 8.0)])
    res = dz.compute_anchors(acts, noise)
    assert res["A"].positive == [] and res["A"].empty_reason == "too_short"
    assert res["B"].positive


def test_load_pipeline_raises_without_pyannote(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyannote", None)
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    with pytest.raises(RuntimeError, match="gated"):
        pd.load_pipeline()
