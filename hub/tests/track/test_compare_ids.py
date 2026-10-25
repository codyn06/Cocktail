"""track.compare_ids on hand-made Phase 4 timelines and tracker logs (FIXTURES only, no models)."""

import json

from track.compare_ids import compare, load_phase4, load_tracker_log

RATE = 16000
START_US = 1_000_000


def write_run(d, spans):
    """spans: src -> [(t0_s, t1_s)] -> a Phase 4 run folder with one timeline page."""
    segs = [{"source_id": s, "start_sample": int(a * RATE), "end_sample": int(b * RATE),
             "start_us": int(a * 1e6), "end_us": int(b * 1e6)} for s, v in spans.items() for a, b in v]
    (d / "timeline").mkdir()
    (d / "timeline" / "page_0000.json").write_text(json.dumps({"segments": segs}))
    (d / "timeline" / "index.json").write_text(json.dumps({
        "version": 1, "sample_rate": RATE, "n_samples": 15 * RATE,
        "pages": [{"path": "timeline/page_0000.json", "start_us": 0, "end_us": 15_000_000}]}))
    return d


def write_log(path, active, secs=15.0):
    """active: tid -> [(t0_s, t1_s)] spans in state tracking (the id stays listed as last_seen after)."""
    lines = []
    for k in range(int(secs * 10)):
        t = k / 10
        srcs = [{"source_id": tid, "state": "tracking" if any(a <= t < b for a, b in v) else "last_seen"}
                for tid, v in active.items() if t >= v[0][0]]
        lines.append(json.dumps({"device_ts_us": START_US + k * 100_000, "sources": srcs}))
    path.write_text("\n".join(lines))
    return path


def run(tmp_path, spans, active):
    rate, sp = load_phase4(write_run(tmp_path, spans))
    return compare(rate, sp, load_tracker_log(write_log(tmp_path / "log.jsonl", active)), START_US)


def test_alternating_talkers_pair_one_to_one(tmp_path):
    r = run(tmp_path, {"src_1": [(0, 5), (10, 15)], "src_2": [(5, 10)]},
            {"t1": [(0, 5), (10, 15)], "t2": [(5, 10)]})
    assert r["pairing"]["src_1"]["tid"] == "t1" and r["pairing"]["src_1"]["status"] == "paired"
    assert r["pairing"]["src_2"]["tid"] == "t2" and r["pairing"]["src_2"]["status"] == "paired"
    assert not r["splits"] and not r["merges"] and not r["disagreements"]


def test_tracker_new_id_on_return_is_a_split(tmp_path):
    r = run(tmp_path, {"src_1": [(0, 5), (10, 15)], "src_2": [(5, 10)]},
            {"t1": [(0, 5)], "t2": [(5, 10)], "t3": [(10, 15)]})
    assert set([r["pairing"]["src_1"]["tid"], *r["splits"]["src_1"]]) == {"t1", "t3"}
    assert r["disagreements"], "src_1 alone while its paired id is not tracking"


def test_tracker_merging_two_talkers_is_a_merge(tmp_path):
    r = run(tmp_path, {"src_1": [(0, 5), (10, 15)], "src_2": [(5, 10)]}, {"t1": [(0, 15)]})
    assert r["merges"] == {"t1": ["src_1", "src_2"]}


def test_always_overlapping_talkers_stay_unpaired(tmp_path):
    r = run(tmp_path, {"src_1": [(0, 15)], "src_2": [(0, 15)]}, {"t1": [(0, 15)], "t2": [(0, 15)]})
    assert all(pr["status"] == "unpaired" for pr in r["pairing"].values())


def test_id_on_the_whole_time_does_not_collect_every_src(tmp_path):
    # t1 announced at once and never stops; the second talker's id t2 only appears at 6 s and is on
    # while src_2 talks; src_2 is never alone, so the pairing falls back to lift
    r = run(tmp_path, {"src_1": [(0, 15)], "src_2": [(6, 9), (11, 13)]}, {"t1": [(0, 15)], "t2": [(6, 9), (11, 13)]})
    assert r["pairing"]["src_2"]["tid"] == "t2" and r["pairing"]["src_2"]["evidence"] == "lift"
    assert not r["merges"]


def test_alignment_uses_audio_start(tmp_path):
    # the same alternating scene, but Phase 4 sample 0 is 2 s before the tracker's first message
    rate, sp = load_phase4(write_run(tmp_path, {"src_1": [(2, 7), (12, 17)], "src_2": [(7, 12)]}))
    log = load_tracker_log(write_log(tmp_path / "log.jsonl", {"t1": [(0, 5), (10, 15)], "t2": [(5, 10)]}))
    r = compare(rate, sp, log, START_US - 2_000_000)
    assert r["pairing"]["src_1"]["tid"] == "t1" and r["pairing"]["src_2"]["tid"] == "t2"
    assert not r["disagreements"]
