"""Matched 30 / 120 / 300 s memory matrix for the Phase 4 chunked pipeline.

Runs the pipeline CLI as a SUBPROCESS (so the measurement is the real whole-process footprint, children included),
samples RSS of the whole process tree every 250 ms with psutil, polls GPU memory with `nvidia-smi`, writes
`matrix.json` and prints a table, then judges the numbers against `contracts.MemoryBudgets` with
`long_metrics.memory_growth`. Never imports torch.

    python eval/av_scene/long_run_matrix.py --out runs/matrix --mock
    python eval/av_scene/long_run_matrix.py --out runs/matrix --media "clips/cafe_{duration_s}s.mp4" --config cfg.toml

Workloads must be MATCHED: same resolution, same source count, same models/config; only the duration differs. With
`--mock` the synthetic `matrix` scene (long_fixtures) is used at each duration. With real media, pre-cut one clip per
duration (`{duration_s}` in `--media` is substituted) from the same footage.

ARGV TEMPLATE (what is executed per workload; the orchestrator's `python -m av_scene run` entry point):

    {python} -m av_scene run --media {media} --out {out} --cache {cache} --config {config} --mock-scene {mock_scene}

Placeholders: {python} {media} {out} {cache} {config} {mock_scene} {duration_s}. A `--flag value` pair whose value
placeholder is unset (for example `--config` when not given, `--media` in mock mode) is dropped. Override with
`--argv-template "..."` (shell-style split; use forward slashes). The child runs with cwd = hub/ and a fresh
`--cache` per workload (a warm cache would skip the work being measured). If the child prints progress lines such as
`chunk 17/149`, RSS samples are mapped to audio time from them; otherwise linearly from wall time (recorded in
`audio_time_mapping`). GPU readings are DEVICE-WIDE (`memory.used`, other processes included); the baseline before
launch is recorded and the budget is judged on the absolute peak, the conservative choice. No nvidia-smi: recorded
as "unavailable", never as zero.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent)) if str(Path(__file__).resolve().parent) not in sys.path else None
import long_metrics as LM  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARGV = ("{python} -m av_scene run --media {media} --out {out} --cache {cache} --config {config} "
                "--mock-scene {mock_scene}")
DEFAULT_DURATIONS = (30, 120, 300)
PROGRESS_RE = re.compile(r"chunk[ _:=]*(\d+)\s*/\s*(\d+)", re.I)
MIB = 2**20


# ------------------------------------------------------------------------------------------------ plumbing
def build_argv(template: str | Sequence[str], values: dict[str, str | None]) -> list[str]:
    toks = list(template) if not isinstance(template, str) else shlex.split(template)
    pat = re.compile(r"\{(\w+)\}")
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        unset = [n for n in pat.findall(t) if values.get(n) is None]
        if unset:
            if t.startswith("-"):
                pass                                                # a flag whose own token has an unset placeholder
            elif out and out[-1].startswith("-") and not pat.findall(out[-1]):
                out.pop()                                           # drop the flag that owned this value
            else:
                raise ValueError(f"template placeholder {{{unset[0]}}} has no value")
            i += 1
            continue
        out.append(pat.sub(lambda m: str(values[m.group(1)]), t))
        i += 1
    return out


def tree_rss_mib(pid: int) -> float | None:
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
    except psutil.Error:
        return None
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except psutil.Error:
            pass
    return total / MIB


def read_nvidia_smi_mib() -> float | None:
    """Device-wide memory.used of GPU 0 in MiB, or None when nvidia-smi is absent/failing."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=5)
        return float(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


class _GpuPoller(threading.Thread):
    def __init__(self, reader: Callable[[], float | None] | None, period_s: float):
        super().__init__(daemon=True)
        self.reader, self.period, self.series, self._stop_evt = reader, period_s, [], threading.Event()
        self.t0 = time.monotonic()

    def run(self):
        while self.reader is not None and not self._stop_evt.is_set():
            v = self.reader()
            if v is None:
                self.reader = None                                   # absent or failing: stop, report unavailable
                break
            self.series.append((round(time.monotonic() - self.t0, 3), v))
            self._stop_evt.wait(self.period)

    def stop(self):
        self._stop_evt.set()


# ------------------------------------------------------------------------------------------------ one run
def run_workload(argv: list[str], *, duration_s: float, cwd: str | Path | None = None, interval_s: float = 0.25,
                 gpu_reader: Callable[[], float | None] | None = None, gpu_period_s: float = 1.0,
                 timeout_s: float | None = None, env: dict | None = None) -> dict:
    """Run `argv`, sampling process-tree RSS every `interval_s`. Returns plain data (JSON-safe)."""
    base_gpu = gpu_reader() if gpu_reader else None
    poller = _GpuPoller(gpu_reader if base_gpu is not None else None, gpu_period_s)
    tail: list[str] = []
    progress: list[tuple[float, float]] = []                         # (t_wall, fraction)
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, cwd=str(cwd) if cwd else None, env={**os.environ, **(env or {})},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace")

    def pump():
        for line in proc.stdout:
            tail.append(line.rstrip())
            del tail[:-40]
            m = PROGRESS_RE.search(line)
            if m and int(m.group(2)) > 0:
                progress.append((time.monotonic() - t0, int(m.group(1)) / int(m.group(2))))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    poller.start()
    series, timed_out = [], False
    while proc.poll() is None:
        v = tree_rss_mib(proc.pid)
        if v is not None:
            series.append((round(time.monotonic() - t0, 3), round(v, 2)))
        if timeout_s and time.monotonic() - t0 > timeout_s:
            timed_out = True
            for q in psutil.Process(proc.pid).children(recursive=True) + [psutil.Process(proc.pid)]:
                try:
                    q.kill()
                except psutil.Error:
                    pass
            break
        time.sleep(interval_s)
    rc = proc.wait()
    reader.join(timeout=5)
    poller.stop()
    wall = time.monotonic() - t0
    if len(progress) >= 3:
        pt, pf = zip(*progress)
        audio = [(float(np.interp(t, pt, pf)) * duration_s, r) for t, r in series]
        mapping = "progress_lines"
    else:
        audio = [(duration_s * t / wall if wall > 0 else 0.0, r) for t, r in series]
        mapping = "linear_wall_time"
    peak = max((r for _, r in series), default=None)
    gpu = {"status": "ok" if poller.series else "unavailable", "scope": "device-wide memory.used",
           "baseline_mib": base_gpu, "peak_mib": max((v for _, v in poller.series), default=None),
           "series": poller.series}
    return {"returncode": rc, "timed_out": timed_out, "wall_s": round(wall, 3), "duration_s": duration_s,
            "rss_peak_mib": peak, "rss_series": series, "audio_series": [(round(a, 3), r) for a, r in audio],
            "audio_time_mapping": mapping, "n_progress_lines": len(progress), "gpu": gpu, "stdout_tail": tail[-15:],
            "argv": argv}


# ----------------------------------------------------------------------------------------------- matrix
def run_matrix(out_dir: str | Path, *, durations: Sequence[float] = DEFAULT_DURATIONS, mock: bool = False,
               media: str | None = None, config: str | None = None, argv_template: str | Sequence[str] = DEFAULT_ARGV,
               cwd: str | Path | None = None, interval_s: float = 0.25, gpu_reader="auto", budgets: dict | None = None,
               warmup_chunks: int | None = None, timeout_s: float | None = None, python: str | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not mock and not media and "{media}" in (argv_template if isinstance(argv_template, str) else " ".join(argv_template)):
        raise SystemExit("need --mock or --media")
    reader = read_nvidia_smi_mib if gpu_reader == "auto" else gpu_reader
    contracts = LM.load_hub_module("contracts")
    bud = dataclasses.replace(contracts.MemoryBudgets(), **(budgets or {}))
    runs = []
    for d in sorted(durations):
        wd = out / f"w{int(d)}s"
        (wd / "cache").mkdir(parents=True, exist_ok=True)
        vals = {"python": python or sys.executable, "media": media.replace("{duration_s}", str(int(d))) if media else None,
                "out": str(wd / "bundle"), "cache": str(wd / "cache"), "config": config,
                "mock_scene": f"matrix:{int(d)}" if mock else None, "duration_s": str(int(d))}
        argv = build_argv(argv_template, vals)
        print(f"[matrix] {d:g} s: {' '.join(argv)}", flush=True)
        runs.append(run_workload(argv, duration_s=float(d), cwd=cwd or (REPO / "hub"), interval_s=interval_s,
                                 gpu_reader=reader, timeout_s=timeout_s))
    verdict = LM.memory_growth(
        [{"duration_s": r["duration_s"], "series": r["audio_series"], "rss_peak_mib": r["rss_peak_mib"],
          "gpu_peak_mib": r["gpu"]["peak_mib"] if r["gpu"]["status"] == "ok" else "unavailable"} for r in runs],
        bud, warmup_chunks)
    failed = [r["duration_s"] for r in runs if r["returncode"] != 0]
    result = {"evidence": "mocked/synthetic" if mock else "measured (whole process tree, this machine)",
              "mock": mock, "sample_interval_s": interval_s, "budgets": dataclasses.asdict(bud),
              "workloads": runs, "memory_growth": verdict, "failed_workloads": failed,
              "passed": bool(not failed and verdict["passed"]), "complete": bool(verdict["complete"] and not failed)}
    (out / "matrix.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(format_table(result))
    return result


def format_table(res: dict) -> str:
    g = res["memory_growth"]["per_workload"]
    rows = [f"{'audio_s':>8} {'rc':>3} {'wall_s':>8} {'rss_peak_MiB':>13} {'slope_MiB/min':>14} {'gpu_peak_MiB':>13} mapping"]
    for r in res["workloads"]:
        p = g[str(r["duration_s"])]
        f = lambda v, w, nd=1: (f"{v:{w}.{nd}f}" if isinstance(v, (int, float)) else f"{'n/a':>{w}}")
        rows.append(f"{r['duration_s']:>8g} {r['returncode']:>3} {r['wall_s']:>8.1f} {f(r['rss_peak_mib'], 13)} "
                    f"{f(p['slope_mib_per_audio_min'], 14, 2)} {f(r['gpu']['peak_mib'], 13)} {r['audio_time_mapping']}")
    rows.append("")
    for name, c in res["memory_growth"]["checks"].items():
        v = "n/a" if c["value"] is None else f"{c['value']:.2f}"
        rows.append(f"  {name:<30} {v:>10}  limit {c['limit']:<8g} {c['status'].upper()}")
    rows.append(f"  warm-up ends at {res['memory_growth']['warmup_end_audio_s']:g} audio s; RSS = process tree, GPU = device-wide")
    if res["failed_workloads"]:
        rows.append(f"  FAILED workloads (nonzero exit): {res['failed_workloads']}")
    rows.append(f"  RESULT: {'PASS' if res['passed'] else 'FAIL'}{'' if res['complete'] else ' (incomplete: some checks unavailable)'}")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (matrix.json + one subdirectory per workload)")
    ap.add_argument("--mock", action="store_true", help="synthetic 'matrix' scene + fake models via --mock-scene")
    ap.add_argument("--media", help="real media path; {duration_s} is substituted (pre-cut one clip per duration)")
    ap.add_argument("--config", help="pipeline config file passed through")
    ap.add_argument("--durations", type=float, nargs="+", default=list(DEFAULT_DURATIONS))
    ap.add_argument("--argv-template", default=DEFAULT_ARGV)
    ap.add_argument("--interval-ms", type=float, default=250.0)
    ap.add_argument("--gpu", choices=["auto", "off"], default="auto")
    ap.add_argument("--budgets-json", help="JSON object overriding MemoryBudgets fields (for dry runs only)")
    ap.add_argument("--warmup-chunks", type=int)
    ap.add_argument("--timeout-s", type=float)
    ap.add_argument("--cwd", help="working directory for the child (default: hub/)")
    a = ap.parse_args(argv)
    res = run_matrix(a.out, durations=a.durations, mock=a.mock, media=a.media, config=a.config,
                     argv_template=a.argv_template, cwd=a.cwd, interval_s=a.interval_ms / 1000.0,
                     gpu_reader="auto" if a.gpu == "auto" else None,
                     budgets=json.loads(a.budgets_json) if a.budgets_json else None, warmup_chunks=a.warmup_chunks,
                     timeout_s=a.timeout_s)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
