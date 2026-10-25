"""locate.scenes: the two-talker scoreboard on one short simulated scene (pairsim: simulated data only)."""

import warnings

from locate import pairsim, real_eval, scenes
from mockclient import Session


def _calib16(tmp_path):
    takes = []
    for i, ang in enumerate((0.0, 90.0, 180.0, -90.0)):
        pairsim.write_take(tmp_path / f"t{i}", ang, seed=i, secs=15.0)
        takes.append(tmp_path / f"t{i}")
    return real_eval.calibrate(takes, tmp_path / "cal16.json", rate_hz=16000)


def test_scene_scoreboard_rows_and_summary(tmp_path, capsys):
    cal = _calib16(tmp_path)
    warnings.simplefilter("ignore")
    d = scenes.ensure_scene(tmp_path / "scenes", 0, (-30, 40), seed=100, secs=15.0)
    assert scenes.ensure_scene(tmp_path / "scenes", 0, (-30, 40), seed=100, secs=15.0) == d  # reused
    raw = []
    rows = scenes.score_scene(d, cal, None, raw)
    assert raw and raw[-1]["sources"] and {r["stream"] for r in rows} <= {x["source_id"] for x in raw[-1]["sources"]}
    assert [r["truth"] for r in rows] == [-30.0, 40.0]
    assert all(r["timeline"] and r["conf_points"] for r in rows)
    assert all(0.0 <= c <= 1.0 for r in rows for c, _ in r["conf_points"])
    s = scenes.summarize(rows)
    assert s["n"] == 2 and s["n_scenes"] == 1 and 0 <= s["both_within_25"] <= 1
    assert s["ece"] is not None
    assert sum(n for *_, n in s["calibration"]) == sum(len(r["conf_points"]) for r in rows)
    scenes.print_summary(s, "fixed: ")
    out = capsys.readouterr().out
    assert "both within 25 deg" in out and "ECE" in out


def test_a_new_seed_rewrites_the_scene(tmp_path):
    root = tmp_path / "scenes"
    scenes.ensure_scene(root, 0, (-30, 40), seed=100, secs=6.0)
    d = scenes.ensure_scene(root, 0, (-30, 40), seed=101, secs=6.0)
    assert Session.load(d).meta["seed"] == 101
