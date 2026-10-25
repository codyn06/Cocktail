"""Tests for eval/results_table.py, on the hand-made example JSON in eval/examples (not measurements)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[3] / "eval"
sys.path.insert(0, str(EVAL))
import results_table as rt  # noqa: E402

BENCH = json.loads((EVAL / "examples" / "bench_example.json").read_text())
DELAY = json.loads((EVAL / "examples" / "delay_example.json").read_text())


def table_rows(text: str) -> dict[str, list[str]]:
    rows = [ln for ln in text.splitlines() if ln.startswith("| ") and not ln.startswith("| Metric")]
    return {r.split("|")[1].strip(): [c.strip() for c in r.split("|")[2:-1]] for r in rows}


def test_example_table_has_one_table_and_the_example_banner():
    t = rt.build_table(BENCH, DELAY)
    assert t.startswith("**EXAMPLE")
    assert t.count("| Metric |") == 1
    assert "Direction method: `tdoa-loo`, 11 valid take(s), 1 invalid" in t


def test_status_is_judged_against_the_targets():
    rows = table_rows(rt.build_table(BENCH, DELAY))
    assert rows["Error in front, median / p90"][1:] == ["11.0 / 16.8 deg (n=5)", "met"]
    assert rows["Error behind, median / p90"][2] == "NOT MET"          # median 26.5 > 25
    assert rows["Front/back right"][1:] == ["89% (n=9)", "NOT MET"]    # 0.89 < 0.95
    lock = rows["Seconds to lock"]
    assert lock[2] == "met" and "within 3 s: 8/11" in lock[1] and "never: 1" in lock[1]
    assert rows["Confidence honesty"][2] == "no target" and "ECE 0.12" in rows["Confidence honesty"][1]
    assert rows["Two speakers within 25 deg"][2] == "no data"
    rt_row = rows["Round trip median / p90 MOCK"]
    assert rt_row[1] == "128 / 151 ms (n=38/40)" and rt_row[2] == "met"
    assert rows["Round trip worst case MOCK"][1] == "190 ms"


def test_method_choice():
    assert "Direction method: `level`" in rt.build_table(BENCH, None, "level")
    assert table_rows(rt.build_table(BENCH, None, "level"))["Front/back right"][1] == "75% (n=4)"
    with pytest.raises(SystemExit):
        rt.build_table(BENCH, None, "nope")


def test_never_locking_takes_count_against_the_median():
    b = json.loads(json.dumps(BENCH))
    b["summaries"]["tdoa-loo"]["lock_s"] = [None] * 6 + [1.0] * 5
    assert table_rows(rt.build_table(b, None))["Seconds to lock"][2] == "NOT MET"


def test_missing_inputs_say_no_data_instead_of_crashing():
    only_delay = table_rows(rt.build_table(None, DELAY))
    assert only_delay["Round trip median / p90 MOCK"][2] == "met"
    only_bench = table_rows(rt.build_table(BENCH, None))
    assert only_bench["Round trip median / p90"][2] == "no data"
    nothing = rt.build_table({"summaries": {}}, {})
    assert "no bench data" in nothing and nothing.count("no data") >= 8


def test_failed_delay_run_is_not_met():
    d = dict(DELAY, **{"pass": False, "fail_reasons": ["median 180 ms > 150 ms"], "median_ms": 180.0})
    d.pop("example")
    rows = table_rows(rt.build_table(None, d))
    assert rows["Round trip median / p90 MOCK"][2] == "NOT MET"
    assert not rt.build_table(None, d).startswith("**EXAMPLE")


def test_cli_prints_and_writes(tmp_path, capsys):
    out = tmp_path / "t.md"
    rc = rt.main(["--bench", str(EVAL / "examples" / "bench_example.json"),
                  "--delay", str(EVAL / "examples" / "delay_example.json"), "--out", str(out)])
    assert rc == 0 and out.read_text() == capsys.readouterr().out
    assert rt.main(["--bench", str(tmp_path / "missing.json")]) == 2
