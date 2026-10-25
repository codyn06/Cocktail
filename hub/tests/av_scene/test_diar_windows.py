import numpy as np

from av_scene import diar_windows as dw
from separate import diarization as dz
from separate import pyannote_diar as pd

S = 1_000_000
SR = dw.SR


class SeqPipeline(pd.FakePipeline):
    """Returns a different canned turn list per call; an entry that is an Exception is raised."""

    def __init__(self, seq):
        super().__init__([])
        self.seq = list(seq)

    def __call__(self, data):
        item = self.seq[self.calls]
        if isinstance(item, Exception):
            self.calls += 1
            raise item
        self._turns = item
        return super().__call__(data)


def reader(log):
    def read(s, e):
        log.append(e - s)
        return np.zeros(e - s, np.float32)
    return read


IDENT = {"src_0": [(0, 10 * S), (30 * S, 40 * S)], "src_1": [(10 * S, 20 * S), (40 * S, 50 * S)]}


def test_default_is_off_and_core_untouched():
    assert dw.DiarConfig().window_s == 30.0
    res = dw.run_windows(reader([]), 55 * SR, IDENT)  # mode default off
    assert res.status.status == "skipped" and res.activities == {}
    acts = {"a": [(0, 3 * S)], "b": [(2 * S, 5 * S)]}
    base = dz.compute_anchors(acts)
    wrapped = dz.compute_anchors_with_diarization(acts, diarization=res)
    assert {k: v.to_dict() for k, v in base.items()} == {k: v.to_dict() for k, v in wrapped.items()}
    assert {k: v.to_dict() for k, v in dz.compute_anchors_with_diarization(acts).items()} == \
        {k: v.to_dict() for k, v in base.items()}


def test_label_permutation_maps_by_identity():
    seq = [[("SPEAKER_00", 10, 20), ("SPEAKER_01", 0, 10)],
           [("SPEAKER_00", 5, 15), ("SPEAKER_01", 15, 25)]]  # window 1 starts at 25 s
    log = []
    res = dw.run_windows(reader(log), 55 * SR, IDENT, SeqPipeline(seq), mode="on")
    assert res.status.status == "ok"
    assert res.activities["src_1"] == [(10 * S, 20 * S), (40 * S, 50 * S)]
    assert res.activities["src_0"] == [(0, 10 * S), (30 * S, 40 * S)]
    assert res.unmapped_s == 0 and res.mapped_s == 40.0


def test_low_overlap_unmapped_namespaced():
    ident = {"src_0": [(0, 10 * S)]}
    turns = [("SPEAKER_00", 0, 3), ("SPEAKER_00", 20, 29)]  # 3 of 12 s overlap -> below threshold
    mp = dw.map_labels_to_sources(turns, ident, window_index=2)
    assert mp == {"SPEAKER_00": "diar:w2:SPEAKER_00"}
    mp = dw.map_labels_to_sources([("SPEAKER_00", 0, 0.4)], ident)  # below min_overlap_s
    assert mp["SPEAKER_00"].startswith("diar:w0:")


def test_tie_leaves_both_unmapped():
    ident = {"src_0": [(0, 10 * S)]}
    mp = dw.map_labels_to_sources([("A", 0, 5), ("B", 5, 10)], ident)
    assert mp == {"A": "diar:w0:A", "B": "diar:w0:B"}


def test_overlapped_speech_preserved():
    ident = {"src_0": [(0, 10 * S)], "src_1": [(5 * S, 15 * S)]}
    turns = [("X", 0, 10), ("Y", 5, 15)]
    mp = dw.map_labels_to_sources(turns, ident)
    assert mp == {"X": "src_0", "Y": "src_1"}
    res = dw.run_windows(reader([]), 20 * SR, ident, SeqPipeline([turns]), mode="on")
    anchors = dz.compute_anchors(res.activities)
    assert dz.intersect(res.activities["src_0"], res.activities["src_1"]) == [(5 * S, 10 * S)]
    assert anchors["src_0"].provenance["overlapped_by_other_speakers_us"] == 5 * S


def test_window_boundaries_tile_exactly():
    cfg = dw.DiarConfig()
    total = 100 * SR
    spans = dw.window_spans(total, cfg)
    assert spans[0][2] == 0 and spans[-1][3] == total
    for a, b in zip(spans, spans[1:]):
        assert a[3] == b[2]  # no gap, no duplicate
    for s, e, o0, o1 in spans:
        assert s <= o0 < o1 <= e and e - s <= int(cfg.window_s * SR)
    # a speaker talking continuously is reported once with no duplicated time
    ident = {"src_0": [(0, 100 * S)]}
    seq = [[("L", 0, 30)], [("L", 0, 30)], [("L", 0, 30)], [("L", 0, 25)]]
    n = len(spans)
    res = dw.run_windows(reader([]), total, ident, SeqPipeline(seq[:n] + [[("L", 0, 5)]] * (n - 4)), mode="on")
    assert res.activities["src_0"] == [(0, 100 * S)]


def test_only_one_window_resident():
    log = []
    ident = {"src_0": [(0, 100 * S)]}
    n = len(dw.window_spans(100 * SR, dw.DiarConfig()))
    dw.run_windows(reader(log), 100 * SR, ident, SeqPipeline([[]] * n), mode="on")
    assert len(log) == n and max(log) <= 30 * SR


def test_statuses_distinct_and_failure_degrades_one_window():
    off = dw.run_windows(reader([]), SR, IDENT, mode="off")
    na = dw.run_windows(reader([]), SR, IDENT, pipeline=None, mode="auto", not_attempted_reason="no token")
    assert (off.status.status, na.status.status) == ("skipped", "not_attempted")
    assert na.status.reason == "no token"
    seq = [[("A", 0, 10)], RuntimeError("boom"), [("A", 0, 5)]]
    ident = {"src_0": [(0, 80 * S)]}
    res = dw.run_windows(reader([]), 60 * SR, ident, SeqPipeline(seq), mode="on")
    assert [w.status for w in res.windows] == ["ok", "failed", "ok"]
    assert res.status.status == "ok" and "src_0" in res.activities
    assert res.activities["src_0"][0][0] == 0  # window 0 evidence survives, window 1 range absent
    allfail = dw.run_windows(reader([]), 20 * SR, ident, SeqPipeline([RuntimeError("x")]), mode="on")
    assert allfail.status.status == "failed"


def test_diarization_only_anchor_marked():
    res = dw.DiarResult(dw.DiarStatus("ok"), [], {"diar:w0:A": [(0, 3 * S)]}, 0.0, 3.0)
    out = dz.compute_anchors_with_diarization({"a": [(5 * S, 8 * S)]}, diarization=res)
    assert out["diar:w0:A"].provenance["diarization"]["diarization_only"] is True
    assert out["a"].provenance["diarization"]["diarization_only"] is False


def test_availability_never_leaks_token(tmp_path):
    env = {"HF_TOKEN": "hf_SECRET123", "HF_HUB_CACHE": str(tmp_path)}
    a = dw.availability(env, tmp_path)
    assert "hf_SECRET123" not in repr(a) and a["token_present"] is True
    assert a["gated_access"] == "not_attempted" and a["cache"] is False
    (tmp_path / "models--pyannote--speaker-diarization-community-1").mkdir()
    assert dw.availability({}, tmp_path)["cache"] is True
