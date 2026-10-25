"""Locating each separated stream on the live path (score_live --oracle) in simulated 3S scenes,
and calibrations that carry the rate they were fitted at (pairsim: simulated data only)."""

import warnings

from locate import pairsim, real_eval, score_live
from locate.live import LiveLocator
from locate.pair import PairCalibration
from ingest import IngestState
from mockclient import Session
from separate.live import LiveSeparator


def _calib16(tmp_path):
    takes = []
    for i, ang in enumerate((0.0, 90.0, 180.0, -90.0)):
        pairsim.write_take(tmp_path / f"t{i}", ang, seed=i)
        takes.append(tmp_path / f"t{i}")
    return real_eval.calibrate(takes, tmp_path / "cal16.json", rate_hz=16000)


def test_calibration_records_the_live_rate_and_live_warns_otherwise(tmp_path):
    cal = _calib16(tmp_path)
    assert (cal.rate_hz, cal.win) == (16000, 1024)
    assert PairCalibration.load(tmp_path / "cal16.json").rate_hz == 16000
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LiveLocator(IngestState(), method="tdoa", calib=cal)  # matches the live framing: no warning
    old = PairCalibration.default()
    assert (old.rate_hz, old.win) == (48000, 4096)  # what quest3s.json was fitted at
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        LiveLocator(IngestState(), method="tdoa", calib=old)
    assert any("16000" in str(x.message) for x in w)
    r = real_eval.run(tmp_path / "t1", 90.0, 12.0, verbose=False, method="tdoa", calib=cal)
    assert r["valid"] and abs(real_eval.circ_diff(r["world_deg"], 90.0)) <= 30


def test_oracle_locates_a_separated_talker_on_the_live_path(tmp_path):
    cal = _calib16(tmp_path)
    pairsim.write_session(tmp_path / "scene", [-30.0], seed=0, secs=20.0)
    s = Session.load(tmp_path / "scene")
    res = score_live.score(s, LiveSeparator(score_live.oracle_separator(s), None), 25.0, "tdoa", cal)
    assert res["talker1"]["error_deg"] <= 25, res


def test_oracle_keeps_two_talkers_apart_and_refuses_a_wrong_window(tmp_path):
    import numpy as np
    import pytest
    cal = _calib16(tmp_path)
    pairsim.write_session(tmp_path / "scene2", [-60.0, 60.0], seed=3, secs=24.0)
    s = Session.load(tmp_path / "scene2")
    res = score_live.score(s, LiveSeparator(score_live.oracle_separator(s), None), 35.0, "tdoa", cal)
    # each stream sits nearer its own talker than the other's: a swapped or shifted stem would not
    for name, other in (("talker1", 60.0), ("talker2", -60.0)):
        r = res[name]
        assert abs(r["world_deg"] - r["truth_deg"]) < abs(r["world_deg"] - other), res
    sep = score_live.oracle_separator(s)
    with pytest.raises(AssertionError):
        sep(np.zeros(1000, np.float32))
