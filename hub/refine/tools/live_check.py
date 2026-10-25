"""Live check harness (Phase 5 T6b): run the refine hub as a subprocess, replay a session, select a
source, and record RSS / GPU memory / refinement status / audio-out continuity into a JSON report.

  python -m refine.tools.live_check --session sessions/p5/stkM --label stkM_refine --refine --out r.json
  python -m refine.tools.live_check --session sessions/p5/stkM --label base --no-refine --out base.json
  python -m refine.tools.live_check ... --baseline base.json

Select transport: a SECOND /ingest/pose connection with the replay's epoch (0, so the hub does not reset
poses) and its own seq counter starting at SELECT_SEQ_BASE. The hub's per-channel ChannelStats will count
that seq jump as seq_gaps (statistics only; nothing rejects on seq). Not run with real models by tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SELECT_SEQ_BASE = 1_000_000
ACCEPTED_STATES = {"accepted", "admitted", "applied"}
HUB_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- pure helpers

def parse_gpu_total(text: str) -> int | None:
    """`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` -> total MiB over GPUs."""
    vals = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln:
            try:
                vals.append(int(float(ln)))
            except ValueError:
                return None
    return sum(vals) if vals else None


def parse_gpu_apps(text: str, pids: set[int]) -> int | None:
    """`--query-compute-apps=pid,used_memory` lines "pid, MiB" -> MiB summed over `pids` (None if no data)."""
    total, seen = 0, False
    for ln in (text or "").splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, mem = int(parts[0]), int(float(parts[1]))
        except ValueError:
            continue
        if pid in pids:
            total += mem
            seen = True
    return total if seen else None


def peak(values) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def check_frame_continuity(frames: list[tuple[float, int, int]]) -> dict:
    """frames = [(recv_time_s, seq, device_ts_us)] in arrival order."""
    out = {"count": len(frames), "seq_breaks": 0, "ts_regressions": 0, "longest_gap_s": 0.0, "ok": True}
    for (t0, s0, d0), (t1, s1, d1) in zip(frames, frames[1:]):
        if s1 != s0 + 1:
            out["seq_breaks"] += 1
        if d1 < d0:
            out["ts_regressions"] += 1
        out["longest_gap_s"] = max(out["longest_gap_s"], t1 - t0)
    out["longest_gap_s"] = round(out["longest_gap_s"], 4)
    out["ok"] = bool(frames) and out["seq_breaks"] == 0 and out["ts_regressions"] == 0
    return out


def reduce_status_timeline(samples: list[dict]) -> dict:
    """samples = [{t, current: {...}|None, counts: {...}}]. First time each state is seen per selection_version."""
    first: dict[str, dict[str, float]] = {}
    for s in samples:
        c = s.get("current")
        if not c:
            continue
        v = str(c.get("selection_version"))
        first.setdefault(v, {}).setdefault(str(c.get("state")), s["t"])
    last_counts = next((s["counts"] for s in reversed(samples) if s.get("counts")), {})
    return {"first_seen": first, "final_counts": last_counts}


def time_to_first_admitted(first_seen: dict, select_times: list[dict]) -> float | None:
    """Seconds from the earliest select to the first accepted-state sighting (any version)."""
    if not select_times:
        return None
    t_sel = min(s["t"] for s in select_times)
    ts = [t for st in first_seen.values() for k, t in st.items() if k in ACCEPTED_STATES and t >= t_sel]
    return round(min(ts) - t_sel, 3) if ts else None


def check_stale(records: list[dict], selects: list[dict]) -> dict:
    """records: [{t, source, selection_version, source_id, accepted}], selects: [{t, source_id}].
    Violation: an accepted record whose source_id is not the source selected at its time, or whose
    selection_version is lower than one already seen for that source."""
    selects = sorted(selects, key=lambda s: s["t"])
    viol, maxv = [], {}
    for r in sorted(records, key=lambda r: r["t"]):
        active = None
        for s in selects:
            if s["t"] <= r["t"]:
                active = s["source_id"]
        v = r.get("selection_version")
        if r.get("accepted"):
            if active is not None and r.get("source_id") not in (None, active):
                viol.append({**r, "why": "source_not_selected"})
            elif v is not None and v < maxv.get(r.get("source_id"), -1):
                viol.append({**r, "why": "older_version"})
        if v is not None and r.get("source_id") is not None:
            maxv[r["source_id"]] = max(maxv.get(r["source_id"], -1), v)
    return {"ok": not viol, "violations": viol}


def build_report(*, config: dict, hub_pid: int | None, wall_s: float, replay_sent: dict | None, selects: list[dict],
                 status_samples: list[dict], audio_samples: list[dict], separate_samples: list[dict],
                 target_updates: list[dict], sources_updated_count: int, audio_frames: list[tuple[float, int, int]],
                 resource_samples: list[dict], gpu_baseline_mib: int | None, baseline: dict | None = None,
                 errors: list[str] | None = None) -> dict:
    tl = reduce_status_timeline(status_samples)
    cont = check_frame_continuity(audio_frames)
    peak_rss = peak(s.get("rss_mib") for s in resource_samples)
    gpu_total_peak = peak(s.get("gpu_total_mib") for s in resource_samples)
    gpu_hub_peak = peak(s.get("gpu_hub_mib") for s in resource_samples)

    def edge(samples, key):
        vals = [s.get(key) for s in samples if s.get(key) is not None]
        return (vals[0], vals[-1]) if vals else (None, None)

    ur0, ur1 = edge(audio_samples, "underruns")
    hd0, hd1 = edge(audio_samples, "hops_dropped")
    lag = peak(s.get("lag_s") for s in separate_samples)
    skips = peak(s.get("skips") for s in separate_samples)
    records = [{"t": t["t"], "source": "target", "selection_version": t.get("selection_version"),
                "source_id": t.get("source_id"),
                "accepted": t.get("kind") == "person" or t.get("state") in ACCEPTED_STATES} for t in target_updates]
    for s in status_samples:
        c = s.get("current")
        if c:
            records.append({"t": s["t"], "source": "status", "selection_version": c.get("selection_version"),
                            "source_id": c.get("source_id"), "accepted": c.get("state") in ACCEPTED_STATES})
    stale = check_stale(records, selects)

    audio_ok = cont["ok"]
    base_ok = True
    if baseline:
        b = baseline.get("audio_out", {})
        for cur, key in ((ur1, "underruns_end"), (hd1, "hops_dropped_end")):
            if cur is not None and b.get(key) is not None and cur > b[key]:
                base_ok = False
    rss_ok = None if peak_rss is None else peak_rss < 6 * 1024
    return {
        "config": config, "hub_pid": hub_pid, "wall_s": round(wall_s, 2), "replay_sent": replay_sent,
        "selects": selects, "errors": errors or [],
        "status": {"timeline": tl, "samples": status_samples,
                   "time_to_first_admitted_s": time_to_first_admitted(tl["first_seen"], selects)},
        "target_updates": target_updates, "sources_updated_count": sources_updated_count,
        "audio_out": {"continuity": cont, "underruns_start": ur0, "underruns_end": ur1,
                      "hops_dropped_start": hd0, "hops_dropped_end": hd1},
        "separate": {"lag_s_max": lag, "skips_max": skips},
        "resources": {"peak_rss_mib": peak_rss, "peak_rss_gib": None if peak_rss is None else round(peak_rss / 1024, 3),
                      "gpu_baseline_mib": gpu_baseline_mib, "gpu_total_peak_mib": gpu_total_peak,
                      "gpu_total_delta_mib": None if gpu_total_peak is None or gpu_baseline_mib is None
                      else gpu_total_peak - gpu_baseline_mib,
                      "gpu_hub_peak_mib": gpu_hub_peak, "samples": resource_samples},
        "verdicts": {"stale_never_current": stale["ok"], "stale_violations": stale["violations"],
                     "fallback_audio_ok": bool(audio_ok and base_ok), "rss_within_6gib": rss_ok},
    }


def pick_loudest(tracks: list[dict]) -> str | None:
    tracks = [t for t in tracks if t.get("source_id")]
    return max(tracks, key=lambda t: t.get("level_db", -1e9))["source_id"] if tracks else None


# ---------------------------------------------------------------- I/O (not unit-tested)

def _get_json(url: str, timeout: float = 2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _nvsmi(args: list[str]) -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=5)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def gpu_total_mib() -> int | None:
    return parse_gpu_total(_nvsmi(["--query-gpu=memory.used"]) or "")


class Sampler(threading.Thread):
    def __init__(self, pid: int, t0: float, period: float = 0.5):
        super().__init__(daemon=True)
        self.pid, self.t0, self.period = pid, t0, period
        self.samples: list[dict] = []
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()
        self.join(5)

    def run(self):
        import psutil
        while not self._stop_evt.is_set():
            rss, pids = None, {self.pid}
            try:
                proc = psutil.Process(self.pid)
                procs = [proc, *proc.children(recursive=True)]
                pids = {p.pid for p in procs}
                rss = 0.0
                for p in procs:
                    try:
                        rss += p.memory_info().rss
                    except Exception:
                        pass
                rss /= 1024 * 1024
            except Exception:
                pass
            apps = _nvsmi(["--query-compute-apps=pid,used_memory"])
            self.samples.append({"t": round(time.perf_counter() - self.t0, 2), "rss_mib": rss,
                                 "gpu_total_mib": gpu_total_mib(),
                                 "gpu_hub_mib": parse_gpu_apps(apps, pids) if apps is not None else None})
            self._stop_evt.wait(self.period)


async def _poll(base: str, t0: float, out: dict, stop: asyncio.Event):
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        t = round(time.perf_counter() - t0, 2)
        for key, path in (("status", "/refinement/status"), ("audio", "/audio/out/status"),
                          ("separate", "/separate/status")):
            d = await loop.run_in_executor(None, _get_json, base + path)
            if isinstance(d, dict):
                out[key].append({"t": t, **d})
        try:
            await asyncio.wait_for(stop.wait(), 1.0)
        except asyncio.TimeoutError:
            pass


async def _events(ws_base: str, t0: float, out: dict, stop: asyncio.Event):
    import websockets
    try:
        async with websockets.connect(ws_base + "/events") as ws:
            while not stop.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), 0.5)
                except asyncio.TimeoutError:
                    continue
                if isinstance(msg, bytes):
                    continue
                d = json.loads(msg)
                if d.get("type") == "target.updated":
                    out["targets"].append({"t": round(time.perf_counter() - t0, 3), **d})
                elif d.get("type") == "sources.updated":
                    out["sources_n"] += 1
    except Exception as e:
        out["errors"].append(f"events: {e!r}")


async def _audio_out(ws_base: str, t0: float, out: dict, stop: asyncio.Event):
    import websockets
    try:
        async with websockets.connect(ws_base + "/audio/out?layout=layers") as ws:
            while not stop.is_set():
                try:
                    data = await asyncio.wait_for(ws.recv(), 0.5)
                except asyncio.TimeoutError:
                    continue
                if isinstance(data, bytes) and len(data) > 4:
                    (n,) = struct.unpack_from("<I", data)
                    h = json.loads(data[4:4 + n])
                    out["frames"].append((time.perf_counter() - t0, int(h["seq"]), int(h["device_ts_us"])))
    except Exception as e:
        out["errors"].append(f"audio_out: {e!r}")


async def _send_selects(ws_base: str, t0: float, sess, args, sink: dict, tracks_url: str):
    import websockets
    import protocol as p
    await asyncio.sleep(args.select_after_s / args.speed if args.speed > 0 else 0)
    loop = asyncio.get_running_loop()
    tracks = await loop.run_in_executor(None, _get_json, tracks_url) or []
    ids = [pick_loudest(tracks)] if args.select == "loudest" else \
        [t["source_id"] for t in sorted(tracks, key=lambda t: -t.get("level_db", -1e9))]
    ids = [i for i in ids if i]
    if not ids:
        sink["errors"].append("no sources at select time")
        return
    seq = SELECT_SEQ_BASE
    async with websockets.connect(ws_base + p.WS_POSE_IN) as ws:
        for i, sid in enumerate(ids):
            dev_ts = sess.start_ts_us + int((time.perf_counter() - t0) * 1e6 * max(args.speed, 1e-9))
            m = p.InputEvent(seq=seq, device_ts_us=dev_ts, epoch=0, action=p.Action.select, source_id=sid)
            seq += 1
            await ws.send(m.model_dump_json())
            sink["selects"].append({"t": round(time.perf_counter() - t0, 3), "source_id": sid})
            if i < len(ids) - 1:
                await asyncio.sleep(args.hold_s)


def _wait_ready(base: str, proc: subprocess.Popen, timeout: float = 240.0) -> bool:
    t = time.time()
    while time.time() - t < timeout and proc.poll() is None:
        if _get_json(base + "/tracks", 1.0) is not None:
            return True
        time.sleep(0.5)
    return False


async def _run(args, base, ws_base, t0, sess) -> dict:
    from mockclient.replay import replay
    out = {"status": [], "audio": [], "separate": [], "targets": [], "sources_n": 0, "frames": [],
           "selects": [], "errors": []}
    stop = asyncio.Event()
    bg = [asyncio.create_task(c) for c in (
        _poll(base, t0, out, stop), _events(ws_base, t0, out, stop), _audio_out(ws_base, t0, out, stop))]
    await asyncio.sleep(0.5)
    sel = asyncio.create_task(_send_selects(ws_base, t0, sess, args, out, base + "/tracks"))
    sent = await replay(sess, ws_base, args.speed, 0)
    await sel
    await asyncio.sleep(args.tail_s)
    stop.set()
    await asyncio.gather(*bg, return_exceptions=True)
    out["sent"] = sent
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--label", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--refine", dest="refine", action="store_true", default=True)
    g.add_argument("--no-refine", dest="refine", action="store_false")
    ap.add_argument("--fake", default="none", choices=["none", "accept", "refuse", "slow"])
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--select", choices=["loudest", "all"], default="loudest")
    ap.add_argument("--select-after-s", type=float, default=2.0)
    ap.add_argument("--hold-s", type=float, default=4.0, help="seconds to hold each source for --select all")
    ap.add_argument("--tail-s", type=float, default=5.0, help="keep observing after replay ends")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    from mockclient.session import Session
    sess = Session.load(a.session)
    cmd = [sys.executable, "-m", "refine", "--port", str(a.port)]
    if not a.refine:
        cmd.append("--no-refine")
    if a.fake != "none":
        cmd += ["--refine-fake", a.fake]
    base, ws_base = f"http://127.0.0.1:{a.port}", f"ws://127.0.0.1:{a.port}"
    gpu0 = gpu_total_mib()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=HUB_DIR)
    sampler, res, errors = None, None, []
    try:
        if not _wait_ready(base, proc):
            errors.append("hub not ready (exited or timed out)")
        else:
            sampler = Sampler(proc.pid, t0)
            sampler.start()
            t0 = time.perf_counter()  # replay-relative clock
            res = asyncio.run(_run(a, base, ws_base, t0, sess))
    except Exception as e:
        errors.append(repr(e))
    finally:
        if sampler:
            sampler.stop()
        try:
            import psutil
            kids = psutil.Process(proc.pid).children(recursive=True)
        except Exception:
            kids = []
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
        for k in kids:
            try:
                k.kill()
            except Exception:
                pass
    res = res or {"status": [], "audio": [], "separate": [], "targets": [], "sources_n": 0, "frames": [],
                  "selects": [], "errors": []}
    baseline = json.loads(a.baseline.read_text()) if a.baseline else None
    rep = build_report(
        config={"label": a.label, "session": str(a.session), "refine": a.refine, "fake": a.fake,
                "select": a.select, "select_after_s": a.select_after_s, "speed": a.speed, "port": a.port,
                "cmd": cmd}, hub_pid=proc.pid, wall_s=time.perf_counter() - t0, replay_sent=res.get("sent"),
        selects=res["selects"], status_samples=res["status"], audio_samples=res["audio"],
        separate_samples=res["separate"], target_updates=res["targets"], sources_updated_count=res["sources_n"],
        audio_frames=res["frames"], resource_samples=sampler.samples if sampler else [], gpu_baseline_mib=gpu0,
        baseline=baseline, errors=errors + res["errors"])
    text = json.dumps(rep, indent=2, default=str)
    if a.out:
        a.out.write_text(text)
    print(json.dumps({"label": a.label, "verdicts": rep["verdicts"], "peak_rss_gib": rep["resources"]["peak_rss_gib"],
                      "time_to_first_admitted_s": rep["status"]["time_to_first_admitted_s"],
                      "errors": rep["errors"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
