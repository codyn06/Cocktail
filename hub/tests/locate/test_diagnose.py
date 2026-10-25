"""Delay-method diagnostics (locate.diagnose) on simulated takes (locate.pairsim) with known faults."""

import json

from locate import diagnose, pairsim, real_eval
from locate.pair import fit_calibration


def test_finds_a_pose_timing_offset(tmp_path):
    pairsim.write_take(tmp_path / "ok", 90.0, seed=3)
    pairsim.write_take(tmp_path / "late", 90.0, seed=3, pose_lag_s=0.5)
    ok, late = diagnose.diagnose([tmp_path / "ok", tmp_path / "late"])
    assert abs(ok["shift_s"]) <= 0.06
    assert abs(late["shift_s"] - 0.5) <= 0.06 and late["res_shift_us"] < 0.6 * late["res_us"]
    assert abs(late["audio_s"] - late["pose_span_s"]) < 0.1
    assert "res_shared_us" in ok and json.dumps([ok, late])


def test_real_eval_scores_a_take_with_a_phase_calibration(tmp_path):
    obs = []
    for i, ang in enumerate((0.0, 90.0, 180.0, -90.0)):
        pairsim.write_take(tmp_path / f"t{i}", ang, seed=i)
        (y, tau, sig), _ = real_eval.observations(tmp_path / f"t{i}", "phase")
        obs.append((y, tau, ang, sig))
    cal = fit_calibration(obs[1:])
    assert cal.delay_method == "phase" and cal.model_s is not None
    r = real_eval.run(tmp_path / "t0", 0.0, 12.0, verbose=False, method="tdoa", calib=cal)
    assert r["valid"] and abs(real_eval.circ_diff(r["world_deg"], 0.0)) <= 30
