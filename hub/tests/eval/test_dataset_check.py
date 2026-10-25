"""Tests for eval/dataset_check.py (the recordings data check).

These live under hub/tests because that is where pytest runs (hub/pyproject.toml testpaths =
["tests"]); the tool itself is in eval/, which is not a package, so it is put on sys.path here.
Every take below is synthetic (numpy noise + a sinusoidal head yaw): evidence label "simulated".
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
import dataset_check as dc  # noqa: E402

CUE = {"noise_until": 4.0, "go": 8.0, "stop": 24.0}
RATE = 48000
BASE_US = 1_000_000
GOOD = {"snr_db": 20.0, "yaw_std": 25.0, "log": ""}


def make_take(path: Path, truth=90.0, secs=26.0, cue=CUE, source_gain=1.0, room=0.005, yaw_amp=40.0,
              chunks=True, meta_extra=None, pose_hz=50.0, drop_truth=False) -> Path:
    """A synthetic take in the record_session.py format (meta, audio.wav, pose.csv, audio_chunks.csv)."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    n = int(RATE * secs)
    t = np.arange(n) / RATE
    yaw = yaw_amp * np.sin(2 * math.pi * t / 10)
    gain = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(np.radians(yaw - truth)))
    x = rng.standard_normal(n) * 0.05 * gain * source_gain
    if cue:
        x = np.where(t < cue["noise_until"], 0.0, x) + rng.standard_normal(n) * room
    pcm = np.clip(np.stack([x, x], axis=1) * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path / "audio.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    pt = np.arange(0, secs, 1 / pose_hz)
    rows = []
    for ti in pt:
        h = math.radians(yaw_amp * math.sin(2 * math.pi * ti / 10)) / 2
        rows.append((BASE_US + int(ti * 1e6), 0.0, math.sin(h), 0.0, math.cos(h)))
    np.savetxt(path / "pose.csv", rows, delimiter=",", header="t_us,qx,qy,qz,qw", comments="")
    meta = {"truth_deg": truth, "sample_rate": RATE, "channels": 2, "first_audio_ts_us": BASE_US,
            "audio_frames": n, "pose_samples": len(rows), "cue": cue}
    if drop_truth:
        del meta["truth_deg"]
    meta.update(meta_extra or {})
    (path / "meta.json").write_text(json.dumps(meta))
    if chunks:
        k = n // 960
        with open(path / "audio_chunks.csv", "w") as f:
            f.write("seq,device_ts_us,frames,epoch\n")
            for i in range(k):
                f.write(f"{i},{BASE_US + i * 20000},960,0\n")
    return path


def check(take: Path, measure=GOOD, **kw) -> dc.TakeResult:
    return dc.check_take(take, measure_fn=lambda _t: dict(measure), **kw)


def failed(r: dc.TakeResult) -> dict[str, str]:
    return {c.name: c.detail for c in r.checks if not c.ok}


def rewrite_pose(take: Path, fn) -> None:
    rows = np.loadtxt(take / "pose.csv", delimiter=",", skiprows=1)
    rows = fn(rows)
    np.savetxt(take / "pose.csv", rows, delimiter=",", header="t_us,qx,qy,qz,qw", comments="")


def edit_meta(take: Path, **kv) -> None:
    m = json.loads((take / "meta.json").read_text())
    m.update(kv)
    (take / "meta.json").write_text(json.dumps(m))


def edit_chunks(take: Path, fn) -> None:
    rows = np.loadtxt(take / "audio_chunks.csv", delimiter=",", skiprows=1)
    rows = fn(rows)
    with open(take / "audio_chunks.csv", "w") as f:
        f.write("seq,device_ts_us,frames,epoch\n")
        for r in rows:
            f.write(",".join(str(int(v)) for v in r) + "\n")


@pytest.fixture(scope="module")
def good_take(tmp_path_factory) -> Path:
    return make_take(tmp_path_factory.mktemp("good") / "t90", meta_extra={"distance_m": 1.0, "motion": "normal", "stimulus": "pink"})


@pytest.fixture
def take(good_take, tmp_path) -> Path:
    """A fresh copy of the good take for tests that damage it."""
    dst = tmp_path / "t"
    shutil.copytree(good_take, dst)
    return dst


# ---------------------------------------------------------------- real measurement, end to end


def test_good_take_passes_with_the_real_measurement(good_take):
    r = dc.check_take(good_take)
    assert r.passed, failed(r)
    assert r.snr_db > dc.MIN_SNR_DB and r.yaw_std >= dc.YAW_STD_MIN_DEG
    assert (r.truth_deg, r.distance_m, r.motion, r.stimulus) == (90.0, 1.0, "normal", "pink")


def test_quiet_source_fails_the_snr_check(tmp_path):
    t = make_take(tmp_path / "q", source_gain=0.1, room=0.01)
    r = dc.check_take(t)
    bad = failed(r)
    assert any("dB over the room" in k for k in bad), bad
    assert r.snr_db < dc.MIN_SNR_DB
    assert not any("head turning" in k for k in bad)


def test_still_head_fails_the_head_turning_check(tmp_path):
    t = make_take(tmp_path / "s", yaw_amp=1.0)
    r = dc.check_take(t)
    bad = failed(r)
    assert any("head turning" in k for k in bad), bad
    assert r.yaw_std < dc.YAW_STD_MIN_DEG


def test_take_without_room_noise_segment_cannot_prove_the_snr(tmp_path):
    t = make_take(tmp_path / "nocue", cue=None)
    bad = failed(dc.check_take(t))
    assert any("dB over the room" in k and "cannot measure" in v for k, v in bad.items()), bad


# ---------------------------------------------------------------- (c) angle label


@pytest.mark.parametrize("bad_truth", [None, "ninety", True, float("nan")])
def test_angle_must_be_a_number(take, bad_truth):
    m = json.loads((take / "meta.json").read_text())
    m["truth_deg"] = bad_truth
    (take / "meta.json").write_text(json.dumps(m))
    bad = failed(check(take))
    assert list(bad) == ["angle labeled (truth_deg is a number)"], bad


def test_truth_key_missing_entirely(tmp_path):
    t = make_take(tmp_path / "x", drop_truth=True)
    r = check(t)
    assert "truth_deg missing" in failed(r)["angle labeled (truth_deg is a number)"]
    assert dc.bin_angle(r.truth_deg) is None


# ---------------------------------------------------------------- (d) clock glitches, one reason each


def clock_detail(r: dc.TakeResult) -> str:
    return failed(r).get("no clock glitches", "")


def test_good_clock_passes_and_chunk_data_is_used(take):
    r = check(take)
    assert r.passed and r.has_chunks
    assert any(n.startswith("chunk timestamps ok") for n in r.notes)


def test_pose_gap_over_half_a_second(take):
    rewrite_pose(take, lambda a: a[(a[:, 0] < BASE_US + 12e6) | (a[:, 0] > BASE_US + 13e6)])
    assert "pose gap" in clock_detail(check(take))


def test_pose_time_goes_backwards(take):
    def swap(a):
        a = a.copy()
        a[[300, 301]] = a[[301, 300]]
        return a
    rewrite_pose(take, swap)
    assert "backwards" in clock_detail(check(take))


def test_pose_ends_before_the_turn_phase_ends(take):
    rewrite_pose(take, lambda a: a[a[:, 0] < BASE_US + 18e6])
    d = clock_detail(check(take))
    assert "pose ends at" in d and "before the turn phase ends (24 s)" in d


def test_pose_ending_after_the_turn_phase_but_before_the_audio_end_is_fine(take):
    rewrite_pose(take, lambda a: a[a[:, 0] < BASE_US + 24.8e6])
    assert check(take).passed


def test_pose_that_starts_early_costs_nothing_and_is_only_noted(take):
    """The scorer drops poses before the audio; a reconnect backoff can start one channel seconds early."""
    def early(a):
        extra = np.repeat(a[:1], 300, axis=0)
        extra[:, 0] = BASE_US - 6_000_000 + np.arange(300) * 20000  # 50 Hz from 6 s before the audio, no gap
        return np.vstack([extra, a])
    rewrite_pose(take, early)
    r = check(take)
    assert r.passed, failed(r)
    assert any("before audio start (ignored by the scorer)" in n for n in r.notes)


def test_first_pose_slightly_before_audio_start_is_tolerated(take):
    def shift_first(a):
        a = a.copy()
        a[0, 0] = BASE_US - 300_000
        return a
    rewrite_pose(take, shift_first)
    assert check(take).passed


def test_pose_that_starts_a_few_seconds_late_but_before_the_head_still_window_is_fine(take):
    rewrite_pose(take, lambda a: a[a[:, 0] > BASE_US + 3e6])
    assert check(take).passed


def test_pose_missing_in_the_head_still_window_fails(take):
    rewrite_pose(take, lambda a: a[a[:, 0] > BASE_US + 7.9e6])  # starts 0.1 s before the turn cue
    d = clock_detail(check(take))
    assert "head-still window" in d and "pose starts at 7.9 s" in d


def test_no_pose_inside_the_turn_phase(take):
    rewrite_pose(take, lambda a: a[(a[:, 0] < BASE_US + 8e6) | (a[:, 0] > BASE_US + 24.4e6)])
    assert "inside the turn phase" in clock_detail(check(take))


def test_recorder_restart_marker(take):
    edit_meta(take, clock_restarts=1)
    assert "clock restarts" in clock_detail(check(take))


def test_pose_restart_marker(take):
    edit_meta(take, pose_restarts=2)
    assert "pose restarts" in clock_detail(check(take))


def test_meta_audio_frames_disagree_with_the_wav(take):
    edit_meta(take, audio_frames=1_000_000)
    assert "audio_frames" in clock_detail(check(take))


def test_dropped_audio_chunk_shows_as_timestamp_drift(take):
    """A dropped 20 ms chunk: every later timestamp is 20 ms ahead of samples/rate."""
    def drop(a):
        a = a.copy()
        a[600:, 1] += 20000
        a[600:, 0] += 1
        return a
    edit_chunks(take, drop)
    d = clock_detail(check(take))
    assert "dropped" in d and "20.0 ms" in d  # the 20 ms step shows against the median offset


def test_lost_chunks_show_as_seq_jump_and_sample_shortfall(take):
    edit_chunks(take, lambda a: np.delete(a, np.s_[100:103], axis=0))
    d = clock_detail(check(take))
    assert "seq jumps" in d and "frames sum" in d


def test_clock_jump_in_chunk_timestamps(take):
    def jump(a):
        a = a.copy()
        a[400:, 1] += 3_000_000
        return a
    edit_chunks(take, jump)
    assert "drift" in clock_detail(check(take))


def test_chunk_timestamps_backwards(take):
    def back(a):
        a = a.copy()
        a[500, 1] -= 100000
        return a
    edit_chunks(take, back)
    assert "backwards" in clock_detail(check(take))


def test_epoch_change_means_a_reconnect(take):
    def reconnect(a):
        a = a.copy()
        a[700:, 3] = 1
        return a
    edit_chunks(take, reconnect)
    assert "epoch" in clock_detail(check(take))


def test_first_chunk_timestamp_must_match_meta(take):
    edit_meta(take, first_audio_ts_us=BASE_US + 40_000)
    assert f"first chunk timestamp {BASE_US} != meta first_audio_ts_us {BASE_US + 40_000}" in clock_detail(check(take))


def test_jittery_chunk_timestamps_and_a_noisy_first_chunk_are_not_a_glitch(take):
    """Native Quest mic path: per-chunk extrapolated timestamps jitter by several ms."""
    rng = np.random.default_rng(5)

    def jitter(a):
        a = a.copy()
        a[:, 1] += np.round(rng.uniform(-3000, 3000, len(a)))
        a[0, 1] += 12000  # first chunk 12 ms off: must not become an offset on every chunk
        return a
    edit_chunks(take, jitter)
    first_ts = int(np.loadtxt(take / "audio_chunks.csv", delimiter=",", skiprows=1)[0, 1])
    edit_meta(take, first_audio_ts_us=first_ts)
    r = check(take)
    assert r.passed, failed(r)
    assert any("jitter p95" in n and "median" in n for n in r.notes)


def test_slow_clock_drift_is_fine_but_a_big_one_fails_with_the_numbers(take):
    def drift(ppm):
        def f(a):
            a = a.copy()
            a[:, 1] += np.round((a[:, 1] - a[0, 1]) * ppm * 1e-6)
            return a
        return f
    edit_chunks(take, drift(100))      # 100 ppm = 2.6 ms over the 26 s take
    assert check(take).passed
    edit_chunks(take, drift(3000))     # on top: ~80 ms
    d = clock_detail(check(take))
    assert "drift +" in d and "ppm" in d and "jitter p95" in d


def test_chunk_tolerance_is_adjustable(take):
    def small(a):
        a = a.copy()
        a[600:, 1] += 8000  # 8 ms ahead
        return a
    edit_chunks(take, small)
    assert check(take).passed
    assert not check(take, chunk_tol_ms=5.0).passed


# ---------------------------------------------------------------- takes recorded before the metadata change


def test_old_take_without_labels_or_chunks_does_not_crash(tmp_path):
    t = make_take(tmp_path / "old", chunks=False)
    r = check(t)
    assert r.passed, failed(r)
    assert any("audio_chunks.csv" in n for n in r.notes)
    assert not r.has_chunks
    assert (r.distance_m, r.motion, r.stimulus) == (None, None, None)
    cells = dc.build_grid([r])
    assert cells == {(90, None): [1, 1]}
    assert dc.filled_cells(cells) == set()  # unlabeled distance fills no cell


def test_missing_files_and_unreadable_meta_are_reported_not_raised(tmp_path):
    t = make_take(tmp_path / "nopose")
    (t / "pose.csv").unlink()
    bad = failed(check(t))
    assert bad and all("missing pose.csv" in v for k, v in bad.items() if k != "angle labeled (truth_deg is a number)")
    j = tmp_path / "junk"
    j.mkdir()
    (j / "meta.json").write_text("{not json")
    assert not dc.check_take(j).passed


def test_measurement_crash_becomes_a_fail(take):
    def boom(_t):
        raise RuntimeError("bad audio")
    r = dc.check_take(take, measure_fn=boom)
    assert any("could not analyze" in v for v in failed(r).values())


# ---------------------------------------------------------------- grid


def test_angle_and_distance_binning():
    assert dc.bin_angle(0) == 0 and dc.bin_angle(-90) == -90 and dc.bin_angle(270) == -90
    assert dc.bin_angle(180) == 180 and dc.bin_angle(-180) == 180 and dc.bin_angle(330) == -30
    assert dc.bin_angle(34) == 30
    assert dc.bin_angle(45) is None and dc.bin_angle(None) is None
    assert dc.bin_distance(0.5) == 0.5 and dc.bin_distance(1.1) == 1.0 and dc.bin_distance(2.0) == 2.0
    assert dc.bin_distance(1.5) is None and dc.bin_distance(None) is None
    assert len(dc.ANGLES) == 12 and dc.GRID_CELLS == 36


def tiny_take(path: Path, truth, dist, motion="normal", stim="pink", clock_restarts=0) -> Path:
    t = make_take(path, truth=truth, secs=2.0, cue=None,
                  meta_extra={"distance_m": dist, "motion": motion, "stimulus": stim, "clock_restarts": clock_restarts})
    return t


@pytest.fixture
def stub_measure(monkeypatch):
    monkeypatch.setattr(dc, "measure", lambda _t: dict(GOOD))


def full_grid(root: Path, skip=None) -> None:
    for a in dc.ANGLES:
        for d in dc.DISTANCES:
            if (a, d) != skip:
                tiny_take(root / f"a{a}-d{d}", a, d)


def test_full_grid_and_all_pass_exits_zero_and_prints_final_line(tmp_path, stub_measure, capsys):
    full_grid(tmp_path)
    rc = dc.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip().splitlines()[-1] == "36 of 36 takes PASS (100%); grid 36 of 36 cells filled"
    assert "READY" in out and "empty cells" not in out


def test_empty_cell_is_visible_and_blocks_exit_zero(tmp_path, stub_measure, capsys):
    full_grid(tmp_path, skip=(120, 2.0))
    rc = dc.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "+120: 2 m" in out and "NOT READY: grid has 1 empty cells" in out
    assert out.strip().splitlines()[-1] == "35 of 35 takes PASS (100%); grid 35 of 36 cells filled"
    row = next(line for line in out.splitlines() if line.startswith("   +120 |"))
    assert row.split("|")[3].strip() == "."


def test_failed_take_does_not_fill_its_cell(tmp_path, stub_measure, capsys):
    full_grid(tmp_path, skip=(0, 1.0))
    tiny_take(tmp_path / "bad0", 0, 1.0, clock_restarts=1)  # fails the clock check
    rc = dc.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "+0: 1 m" in out
    assert "FAIL  bad0" in out and "clock restarts" in out
    assert out.strip().splitlines()[-1] == "35 of 36 takes PASS (97%); grid 35 of 36 cells filled"


def test_pass_rate_below_90_percent_blocks_exit_zero(tmp_path, stub_measure, capsys):
    full_grid(tmp_path)
    for i in range(5):  # 5 more failing takes in already-filled cells: 36 of 41 = 88%
        tiny_take(tmp_path / f"extra{i}", 0, 1.0, clock_restarts=1)
    rc = dc.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1 and "pass rate 88% < 90%" in out
    assert out.strip().splitlines()[-1] == "36 of 41 takes PASS (88%); grid 36 of 36 cells filled"


def test_exactly_90_percent_passes(tmp_path, stub_measure, capsys):
    full_grid(tmp_path)
    for i in range(4):  # 36 of 40 = 90%
        tiny_take(tmp_path / f"extra{i}", 0, 1.0, clock_restarts=1)
    assert dc.main([str(tmp_path)]) == 0


def test_motion_and_stimulus_breakdown_when_labels_exist(tmp_path, stub_measure, capsys):
    tiny_take(tmp_path / "a", 0, 1.0, motion="small", stim="pink")
    tiny_take(tmp_path / "b", 90, 1.0, motion="large", stim="speech")
    dc.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "small 1/1 pass, 1 cells" in out and "large 1/1 pass, 1 cells" in out
    assert "normal 0/0 pass, 0 cells" in out  # a missing size is visible
    assert "speech 1/1 pass, 1 cells" in out


def test_unlabeled_takes_go_in_the_question_mark_column(tmp_path, stub_measure, capsys):
    make_take(tmp_path / "old", truth=0.0, secs=2.0, cue=None, chunks=False)
    rc = dc.main([str(tmp_path)])
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("     +0 |"))
    assert row.split("|")[4].strip() == "1/1"
    assert rc == 1 and "grid 0 of 36" in out
    assert "no take has this label" in out


def test_discovery_finds_nested_takes_and_a_single_take(tmp_path):
    a = make_take(tmp_path / "day1" / "ta", secs=1.0, cue=None)
    b = make_take(tmp_path / "day2" / "tb", secs=1.0, cue=None)
    assert dc.find_takes([tmp_path]) == sorted([a, b])
    assert dc.find_takes([a]) == [a]
    assert dc.find_takes([tmp_path / "nothing"]) == []


def test_no_takes_found_exit_code_2(tmp_path, capsys):
    assert dc.main([str(tmp_path)]) == 2


def test_failure_reasons_are_printed_and_json_written(tmp_path, stub_measure, capsys):
    tiny_take(tmp_path / "x", 0, 1.0, clock_restarts=1)
    out_json = tmp_path / "res.json"
    dc.main([str(tmp_path / "x"), "--json", str(out_json)])
    out = capsys.readouterr().out
    assert "FAIL  x" in out and "FAIL no clock glitches: recorder saw 1 clock restarts" in out
    rows = json.loads(out_json.read_text())
    assert rows[0]["pass"] is False and rows[0]["distance_m"] == 1.0


def test_takes_without_chunk_timestamps_are_flagged_everywhere_but_do_not_block_ready(tmp_path, stub_measure, capsys):
    full_grid(tmp_path)
    for a in dc.ANGLES[:2]:  # two takes of the grid recorded before audio_chunks.csv existed
        make_take(tmp_path / f"old{a}", truth=float(a), secs=2.0, cue=None, chunks=False,
                  meta_extra={"distance_m": 1.0, "motion": "normal", "stimulus": "pink"})
    rc = dc.main([str(tmp_path)])  # no -v: the warning must show anyway
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("WARNING: no audio_chunks.csv") == 2
    assert "WARNING: 2 of 38 takes have no chunk timestamps" in out
    assert "READY: grid full and pass rate >= 90% (but 2 take(s) were not checked for dropped chunks)" in out
    assert out.strip().splitlines()[-1] == "38 of 38 takes PASS (100%); grid 36 of 36 cells filled; 2 without chunk timestamps"
