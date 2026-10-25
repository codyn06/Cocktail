from refine.tools import live_check as lc


def test_parse_gpu():
    assert lc.parse_gpu_total("1000\n2000\n") == 3000
    assert lc.parse_gpu_total("") is None
    assert lc.parse_gpu_total("N/A") is None
    txt = "100, 500\n200, 700\nbad\n"
    assert lc.parse_gpu_apps(txt, {100, 5}) == 500
    assert lc.parse_gpu_apps(txt, {1}) is None


def test_peak():
    assert lc.peak([None, 3, 7.5]) == 7.5
    assert lc.peak([None]) is None


def test_continuity():
    ok = [(0.0, 0, 0), (0.02, 1, 20000), (0.05, 2, 40000)]
    r = lc.check_frame_continuity(ok)
    assert r["ok"] and r["count"] == 3 and r["longest_gap_s"] == 0.03
    bad = lc.check_frame_continuity([(0, 0, 10), (1, 2, 5)])
    assert not bad["ok"] and bad["seq_breaks"] == 1 and bad["ts_regressions"] == 1
    assert not lc.check_frame_continuity([])["ok"]


def test_timeline_and_ttfa():
    s = [{"t": 1, "current": None, "counts": {}},
         {"t": 3, "current": {"selection_version": 1, "state": "running"}, "counts": {"running": 1}},
         {"t": 4, "current": {"selection_version": 1, "state": "running"}, "counts": {"running": 1}},
         {"t": 6, "current": {"selection_version": 1, "state": "accepted"}, "counts": {"accepted": 1}}]
    tl = lc.reduce_status_timeline(s)
    assert tl["first_seen"]["1"] == {"running": 3, "accepted": 6}
    assert tl["final_counts"] == {"accepted": 1}
    assert lc.time_to_first_admitted(tl["first_seen"], [{"t": 2.0, "source_id": "a"}]) == 4.0
    assert lc.time_to_first_admitted({}, []) is None


def test_stale():
    sel = [{"t": 1, "source_id": "a"}, {"t": 10, "source_id": "b"}]
    good = [{"t": 5, "selection_version": 1, "source_id": "a", "accepted": True},
            {"t": 12, "selection_version": 2, "source_id": "b", "accepted": True}]
    assert lc.check_stale(good, sel)["ok"]
    late_a = good + [{"t": 11, "selection_version": 1, "source_id": "a", "accepted": True}]
    r = lc.check_stale(late_a, sel)
    assert not r["ok"] and r["violations"][0]["why"] == "source_not_selected"
    old = [{"t": 3, "selection_version": 2, "source_id": "a", "accepted": False},
           {"t": 4, "selection_version": 1, "source_id": "a", "accepted": True}]
    assert lc.check_stale(old, sel)["violations"][0]["why"] == "older_version"


def _report(frames, baseline=None, rss=100.0):
    return lc.build_report(
        config={}, hub_pid=1, wall_s=10, replay_sent=None, selects=[{"t": 1, "source_id": "a"}],
        status_samples=[{"t": 2, "current": {"selection_version": 1, "source_id": "a", "state": "accepted"},
                         "counts": {"accepted": 1}}],
        audio_samples=[{"t": 0, "underruns": 0, "hops_dropped": 0}, {"t": 9, "underruns": 2, "hops_dropped": 1}],
        separate_samples=[{"lag_s": 0.4, "skips": 1}, {"lag_s": 0.9, "skips": 3}],
        target_updates=[{"t": 2, "selection_version": 1, "source_id": "a", "kind": "person", "state": "tracking"}],
        sources_updated_count=7, audio_frames=frames,
        resource_samples=[{"t": 0, "rss_mib": rss, "gpu_total_mib": 3000, "gpu_hub_mib": None}],
        gpu_baseline_mib=1000, baseline=baseline)


def test_build_report():
    frames = [(0.0, 0, 0), (0.02, 1, 1)]
    r = _report(frames)
    assert r["separate"] == {"lag_s_max": 0.9, "skips_max": 3}
    assert r["audio_out"]["underruns_end"] == 2 and r["audio_out"]["hops_dropped_start"] == 0
    assert r["resources"]["gpu_total_delta_mib"] == 2000
    assert r["status"]["time_to_first_admitted_s"] == 1.0
    v = r["verdicts"]
    assert v["stale_never_current"] and v["fallback_audio_ok"] and v["rss_within_6gib"]
    worse = {"audio_out": {"underruns_end": 0, "hops_dropped_end": 0}}
    assert not _report(frames, worse)["verdicts"]["fallback_audio_ok"]
    assert not _report(frames, rss=7000)["verdicts"]["rss_within_6gib"]
    assert not _report([(0, 0, 0), (1, 5, 0)])["verdicts"]["fallback_audio_ok"]


def test_pick_loudest():
    assert lc.pick_loudest([{"source_id": "a", "level_db": -30}, {"source_id": "b", "level_db": -10}]) == "b"
    assert lc.pick_loudest([]) is None
