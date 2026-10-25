"""ModelLifecycleManager: lease exclusivity, ordering, OOM reporting. CPU-only, no models."""

from __future__ import annotations

import pytest
import torch

from av_scene.runtime import Fallback, GpuLeaseBusy, ModelLifecycleManager, run_stage


class Model:
    def __init__(self, log):
        self.log = log

    def to(self, device):
        self.log.append(f"to:{device}")
        return self


def test_yields_loaded_model_and_cleans_up_in_order():
    log, reports = [], []

    def loader():
        log.append("load")
        return Model(log)

    mgr = ModelLifecycleManager("s", loader, device="cpu", reports=reports, cleanup=lambda m: log.append("cleanup"))
    with mgr as model:
        assert isinstance(model, Model)
        with mgr.phase("inference"):
            log.append("infer")
    assert log == ["load", "infer", "cleanup"]  # `.to("cpu")` is skipped for a non-CUDA device
    assert reports == [mgr.report] and mgr.report.error is None and "inference" in mgr.report.phases


def test_lease_is_exclusive_and_released_after_exit():
    a = ModelLifecycleManager("a", lambda: object(), device="cpu")
    with a:
        with pytest.raises(GpuLeaseBusy):
            with ModelLifecycleManager("b", lambda: object(), device="cpu"):
                pass
    with ModelLifecycleManager("c", lambda: object(), device="cpu"):  # free again
        pass


def test_loader_failure_releases_lease_and_is_reported():
    reports = []

    def boom():
        raise ValueError("no weights")

    with pytest.raises(ValueError):
        with ModelLifecycleManager("x", boom, device="cpu", reports=reports):
            pass
    assert "ValueError" in reports[0].error
    with ModelLifecycleManager("y", lambda: 1, device="cpu"):
        pass


def test_oom_is_flagged_and_not_swallowed():
    reports = []
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with ModelLifecycleManager("oom", lambda: object(), device="cpu", reports=reports):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert reports[0].oom is True


def test_run_stage_retries_once_with_fallback_and_reports_both():
    reports = []

    def big(mgr, model):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    out = run_stage("sam3", lambda: object(), big, reports=reports, device="cpu", workload={"frames": 100},
                    fallback=Fallback(fn=lambda mgr, m: "ok-small", workload={"frames": 25}, note="smaller window"))
    assert out == "ok-small"
    assert [r.name for r in reports] == ["sam3", "sam3:fallback"]
    assert reports[0].oom and reports[1].fallback_of == "sam3" and reports[1].workload["frames"] == 25


def test_run_stage_oom_without_fallback_raises():
    def big(mgr, model):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_stage("s", lambda: object(), big, reports=[], device="cpu")
