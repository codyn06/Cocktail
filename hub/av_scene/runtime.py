"""Sequential GPU model leasing (spec Section 9).

One inference lease at a time: load -> run -> `.to("cpu")` / `del` -> `empty_cache()` -> release.
Each stage records load/phase timings and peak allocated/reserved memory, and an OOM is reported
(never swallowed). `empty_cache()` cannot free tensors something else still references, so the
report also carries the memory still allocated after release: a non-trivial number means the
caller kept a reference to the model (or a session) and should be treated as a leak.

Works on CPU (`device="cpu"`) so the lease/ordering logic is unit-testable without a GPU.
"""

from __future__ import annotations

import gc
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch

_MIB = 1024 * 1024
_LEASE = threading.Lock()


class GpuLeaseBusy(RuntimeError):
    """Another stage already holds the (single) GPU inference lease."""


@dataclass
class StageReport:
    name: str
    device: str
    workload: dict = field(default_factory=dict)
    load_s: float = 0.0
    phases: dict[str, float] = field(default_factory=dict)   # e.g. {"inference": 1.2}
    release_s: float = 0.0
    peak_allocated_mib: float | None = None
    peak_reserved_mib: float | None = None
    host_rss_mib: float | None = None
    allocated_after_release_mib: float | None = None
    output_seconds: float | None = None
    oom: bool = False
    error: str | None = None
    leak_warning: str | None = None
    fallback_of: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def _rss_mib() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / _MIB, 1)
    except Exception:
        return None


class ModelLifecycleManager:
    """`with ModelLifecycleManager("sam3", loader, device="cuda") as model: ...`

    `loader` is a zero-arg callable returning the loaded model/predictor. On exit the model is
    moved to CPU (when it supports `.to`), deleted, and the CUDA cache emptied before the lease
    is released. Do not keep a reference to the yielded object past the `with` block.
    """

    def __init__(self, name: str, loader: Callable[[], Any], *, device: str = "cuda",
                 workload: dict | None = None, reports: list[StageReport] | None = None,
                 leak_tolerance_mib: float = 64.0, cleanup: Callable[[Any], None] | None = None):
        self.report = StageReport(name=name, device=device, workload=dict(workload or {}))
        self._loader, self._device, self._reports = loader, device, reports
        self._leak_tol, self._cleanup = leak_tolerance_mib, cleanup
        self._model: Any = None
        self._held = False

    def __enter__(self):
        if not _LEASE.acquire(blocking=False):
            raise GpuLeaseBusy(f"cannot start stage {self.report.name!r}: the GPU lease is held by another stage")
        self._held = True
        cuda = _is_cuda(self._device)
        try:
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            self._model = self._loader()
            self.report.load_s = round(time.perf_counter() - t0, 3)
        except BaseException as e:
            self._finish(e)
            raise
        return self._model

    @contextmanager
    def phase(self, name: str):
        """Time a named phase (preprocess / inference / ...) inside the lease."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.report.phases[name] = round(self.report.phases.get(name, 0.0) + time.perf_counter() - t0, 3)

    def __exit__(self, exc_type, exc, tb):
        self._finish(exc)
        return False  # never swallow

    def _finish(self, exc: BaseException | None) -> None:
        r = self.report
        cuda = _is_cuda(self._device)
        if cuda:
            torch.cuda.synchronize()
            r.peak_allocated_mib = round(torch.cuda.max_memory_allocated() / _MIB, 1)
            r.peak_reserved_mib = round(torch.cuda.max_memory_reserved() / _MIB, 1)
        r.host_rss_mib = _rss_mib()
        if exc is not None:
            r.oom = isinstance(exc, torch.cuda.OutOfMemoryError)
            r.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
        t0 = time.perf_counter()
        model, self._model = self._model, None
        try:
            if model is not None:
                if self._cleanup is not None:
                    self._cleanup(model)  # e.g. close predictor sessions / terminate workers
                to = getattr(model, "to", None)
                if callable(to) and _is_cuda(self._device):
                    try:
                        to("cpu")
                    except Exception:
                        pass
            del model
            gc.collect()
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                r.allocated_after_release_mib = round(torch.cuda.memory_allocated() / _MIB, 1)
                if r.allocated_after_release_mib > self._leak_tol:
                    r.leak_warning = (f"{r.allocated_after_release_mib} MiB still allocated after release: "
                                      "something still references the model or its session")
        finally:
            r.release_s = round(time.perf_counter() - t0, 3)
            if self._held:
                self._held = False
                _LEASE.release()
            if self._reports is not None:
                self._reports.append(r)


@dataclass
class Fallback:
    """One smaller window/backend to try after an OOM. Reported, never silent."""

    fn: Callable[[Any], Any]
    loader: Callable[[], Any] | None = None   # default: same loader as the primary attempt
    workload: dict = field(default_factory=dict)
    note: str = ""


def run_stage(name: str, loader: Callable[[], Any], fn: Callable[[ModelLifecycleManager, Any], Any], *,
              reports: list[StageReport], device: str = "cuda", workload: dict | None = None,
              fallback: Fallback | None = None, cleanup: Callable[[Any], None] | None = None) -> Any:
    """Run `fn(manager, model)` under a lease. On OOM, discard the stage and try `fallback` once."""
    mgr = ModelLifecycleManager(name, loader, device=device, workload=workload, reports=reports, cleanup=cleanup)
    try:
        with mgr as model:
            return fn(mgr, model)
    except torch.cuda.OutOfMemoryError:
        if fallback is None:
            raise
    fb_name = f"{name}:fallback"
    mgr = ModelLifecycleManager(fb_name, fallback.loader or loader, device=device, workload=fallback.workload,
                                reports=reports, cleanup=cleanup)
    mgr.report.fallback_of = name
    if fallback.note:
        mgr.report.workload = {**mgr.report.workload, "note": fallback.note}
    with mgr as model:
        return fallback.fn(mgr, model)
