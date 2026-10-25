"""Round-trip delay test: hub -> headset speaker -> headset mic -> hub, 30+ trials.

    # With a Quest (a person does this, see "Run it with the headset" below):
    uv run --project hub python eval/delay_test.py --trials 40 --json delay.json
    # No hardware: a mock headset in this process with a known, simulated delay:
    uv run --project hub python eval/delay_test.py --mock --trials 30

What is measured, and why it needs no headset-app change
--------------------------------------------------------
Nothing on main echoes the mic back: `hub/ingest` has no `/audio/out`, and only
`quest/tools/mock_hub.py` (a tone) and `record_session.py` (silence + cue beeps) serve it.
So this test is its own tiny hub (the same five sockets the recorder serves) and uses the
protocol that exists. Per trial:

  1. the hub puts a 12 ms chirp (1.5-7 kHz) into one `/audio/out` frame (`AudioOut`, 48 kHz,
     960 frames, `layers`: inner = the chosen-voice layer by default) and notes the monotonic
     time `t_send` just before sending it; an idle silence stream keeps the headset's jitter
     buffer primed, as it is during live use;
  2. the headset's `AudioOutPlayer` plays it (jitter buffer + output device), the sound
     reaches the headset microphones through the air;
  3. the mic stream arrives on `/ingest/audio`; the hub finds the chirp by matched filter and
     takes its time as chunk arrival minus the sample's offset from the end of the chunk;
  4. delay = that time - `t_send`.

Everything is on the hub clock, so no clock sync between headset and laptop is needed. The loop
holds the same pieces as live mic -> hub -> headset audio: capture batching and uplink, hub,
downlink, jitter buffer, output device (plus ~1 ms of air, and no separation model). A hub that
also runs a model adds its compute time on top of this number.

Limits (read before quoting a number)
  * The mic must hear the speakers: volume up, headset on the head or the speakers next to the
    mics. A trial whose chirp is not found is reported as "not heard", never as a delay.
  * Chunks that reach the hub in a burst (TCP batching) are timed at their arrival, so this
    includes network jitter, which is what a live pipeline experiences.
  * The pass budget (default median <= 150 ms, worst <= 300 ms) is an assumption: the team has
    not stated one. Change it with --budget-median-ms / --budget-worst-ms.
  * Evidence: --mock is MOCKED (software loop on this machine, validated against its known
    simulated delay in hub/tests/eval/test_delay_test.py). Without --mock the number is only
    as good as the headset that was connected; this file has NOT been run against a Quest.

Run it with the headset (what a person must do; nothing here touches adb or the device)
  1. Close anything else using port 8765 (the recorder, the mock hub, a running hub).
  2. Start this script (it waits up to --wait-s for the headset). Then point the Quest app at
     the laptop exactly as quest/tools/run_locate_take.ps1 does: `adb reverse tcp:8765 tcp:8765`,
     then launch the app with `-e hub_host 127.0.0.1 -e micsrc unprocessed -e micch 2`
     (or set hub_host to the laptop's LAN address).
  3. Wear the headset, set the headset volume high enough that you hear a short chirp every
     ~1.3 s, keep the room quiet, do not talk during the run (~1 minute for 40 trials).
  4. Read the table. "not heard" trials mean the mics did not pick up the speakers: raise the
     volume or move the speakers closer. Keep the JSON (--json) for eval/results_table.py.
  The number from the app's own overlay ("mic->hub->headset age") is a different quantity (it
  needs a hub that echoes the mic and excludes the mic-side capture batching).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hub"))

import protocol as p  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

OUT_RATE = 48000
FRAME = 960                      # AudioOut: 20 ms at 48 kHz
FRAME_S = FRAME / OUT_RATE
CLICK_S = 0.012
MIN_TRIALS = 30
BUDGET_MEDIAN_MS = 150.0
BUDGET_WORST_MS = 300.0
MIN_NCC = 0.30                   # matched-filter peak (normalized) needed to call the chirp heard


# ---------------------------------------------------------------- signal and statistics (pure)


def click_template(rate: int) -> np.ndarray:
    """12 ms Hann-windowed chirp, 1.5 -> 7 kHz, peak 1.0: sharp autocorrelation, survives a speaker and a mic."""
    n = int(rate * CLICK_S)
    t = np.arange(n) / rate
    f0, f1 = 1500.0, 7000.0
    k = (f1 - f0) / CLICK_S
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t)) * np.hanning(n)


def find_click(x: np.ndarray, template: np.ndarray, min_ncc: float = MIN_NCC) -> tuple[int, float] | None:
    """Index of the template's first sample in x, and its normalized correlation, or None if not heard."""
    n, m = len(x), len(template)
    if n < m:
        return None
    size = 1 << (n + m).bit_length()
    corr = np.fft.irfft(np.fft.rfft(x, size) * np.conj(np.fft.rfft(template, size)), size)[: n - m + 1]
    c2 = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    energy = c2[m:] - c2[:-m]
    ncc = corr / np.sqrt(np.maximum(energy, 1e-12) * float(template @ template))
    k = int(np.argmax(ncc))
    return (k, float(ncc[k])) if ncc[k] >= min_ncc else None


def summarize(delays_ms: list[float], n_trials: int, budget_median_ms: float = BUDGET_MEDIAN_MS,
              budget_worst_ms: float = BUDGET_WORST_MS) -> dict:
    d = np.array(delays_ms, dtype=float)
    out = {"n_trials": n_trials, "n_valid": len(d), "n_missed": n_trials - len(d),
           "budget_median_ms": budget_median_ms, "budget_worst_ms": budget_worst_ms}
    if len(d):
        out.update(median_ms=float(np.median(d)), p90_ms=float(np.percentile(d, 90)), worst_ms=float(d.max()),
                   min_ms=float(d.min()), jitter_ms=float(statistics.pstdev(d.tolist())))
    why = []
    if len(d) < MIN_TRIALS:
        why.append(f"only {len(d)} valid trials (need >= {MIN_TRIALS})")
    if len(d) and out["median_ms"] > budget_median_ms:
        why.append(f"median {out['median_ms']:.0f} ms > {budget_median_ms:g} ms")
    if len(d) and out["worst_ms"] > budget_worst_ms:
        why.append(f"worst {out['worst_ms']:.0f} ms > {budget_worst_ms:g} ms")
    out["pass"] = not why and len(d) > 0
    out["fail_reasons"] = why
    return out


# ---------------------------------------------------------------- the hub side


@dataclass
class Trial:
    index: int
    delay_ms: float | None
    ncc: float | None


class DelayHub:
    """The five sockets a headset connects to, enough to time a chirp through the headset."""

    def __init__(self, layer: str = "inner", layout: str = "layers", silence: bool = True):
        self.layer, self.layout, self.silence = layer, layout, silence
        self.rate: int | None = None
        self.template: np.ndarray | None = None
        self.hello: dict | None = None
        self.chunks: list[tuple[float, np.ndarray]] = []   # (arrival monotonic s, mono float32)
        self.audio_in_connected = False
        self.out_connected = asyncio.Event()
        self._arm: asyncio.Future | None = None
        self.frames_out = 0
        self.chunks_in = 0
        app = self.app = FastAPI(title="Cocktail delay test hub")

        @app.websocket(p.WS_AUDIO_IN)
        async def ws_audio(ws: WebSocket) -> None:
            await ws.accept()
            self.audio_in_connected = True
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        return
                    if msg.get("bytes") is not None:
                        self._on_audio(time.perf_counter(), msg["bytes"])
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                self.audio_in_connected = False

        @app.websocket(p.WS_VIDEO_IN)
        async def ws_video(ws: WebSocket) -> None:
            await ws.accept()
            try:
                while (await ws.receive())["type"] != "websocket.disconnect":
                    pass
            except (WebSocketDisconnect, RuntimeError):
                pass

        @app.websocket(p.WS_POSE_IN)
        async def ws_pose(ws: WebSocket) -> None:
            await ws.accept()
            ad = TypeAdapter(p.PoseChannelMessage)
            try:
                while True:
                    text = await ws.receive_text()
                    try:
                        m = ad.validate_json(text)
                    except ValueError:
                        continue
                    if isinstance(m, p.Hello):
                        self.hello = m.model_dump()
            except (WebSocketDisconnect, RuntimeError):
                pass

        @app.websocket(p.WS_EVENTS)
        async def ws_events(ws: WebSocket) -> None:
            await ws.accept()
            try:
                await ws.send_text(p.StateChanged(state=p.HubState.listening).model_dump_json())
                while True:
                    await ws.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                pass

        @app.websocket(p.WS_AUDIO_OUT)
        async def ws_audio_out(ws: WebSocket) -> None:
            await ws.accept()
            self.out_connected.set()
            try:
                await self._out_loop(ws)
            except (WebSocketDisconnect, RuntimeError, OSError):
                pass
            finally:
                self.out_connected.clear()

    # -- mic side
    def _on_audio(self, arrival: float, data: bytes) -> None:
        hdr, pcm = p.unpack(data, p.AudioChunk)
        dtype = "<i2" if hdr.sample_format == "s16le" else "<f4"
        x = np.frombuffer(pcm, dtype=dtype).reshape(-1, hdr.channels).astype(np.float32)
        if hdr.sample_format == "s16le":
            x /= 32768.0
        if self.rate != hdr.sample_rate:
            self.rate, self.template = hdr.sample_rate, click_template(hdr.sample_rate)
        self.chunks.append((arrival, x.mean(axis=1)))
        self.chunks_in += 1
        if len(self.chunks) > 600:          # ~12 s of 20 ms chunks
            self.chunks = self.chunks[-400:]

    def analyze(self, t_send: float) -> tuple[float, float] | None:
        """(delay s, ncc) of the chirp sent at t_send, from the chunks that arrived after it."""
        got = [c for c in self.chunks if c[0] >= t_send]
        if not got or self.template is None:
            return None
        x = np.concatenate([c[1] for c in got])
        hit = find_click(x, self.template)
        if hit is None:
            return None
        idx, ncc = hit
        edges = np.cumsum([len(c[1]) for c in got])
        ci = int(np.searchsorted(edges, idx, side="right"))
        start = edges[ci - 1] if ci else 0
        j = idx - start
        arrival, mono = got[ci]
        heard_at = arrival - (len(mono) - 1 - j) / self.rate
        return heard_at - t_send, ncc

    # -- speaker side
    async def _out_loop(self, ws) -> None:
        seq, nxt = 0, time.perf_counter()
        click = (click_template(OUT_RATE) * 0.5).astype(np.float32)
        while True:
            now = time.perf_counter()
            if now - nxt > 0.2:
                nxt = now                       # fell behind (blocked loop): resync instead of bursting
            frame = np.zeros((FRAME, 2), dtype=np.float32)
            armed = self._arm is not None and not self._arm.done()
            if armed:
                if self.layout == "stereo" or self.layer in ("inner", "both"):
                    frame[: len(click), 0] = click
                if self.layout == "stereo" or self.layer in ("outer", "both"):
                    frame[: len(click), 1] = click
            if armed or self.silence:
                pcm = (frame * 32767).astype("<i2")
                msg = p.pack(p.AudioOut(seq=seq, device_ts_us=int(time.perf_counter() * 1e6), layout=self.layout),
                             pcm.tobytes())
                t = time.perf_counter()
                await ws.send_bytes(msg)
                seq += 1
                self.frames_out += 1
                if armed:
                    self._arm.set_result(t)
            nxt += FRAME_S
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))

    async def send_click(self, timeout: float = 5.0) -> float:
        """Put the chirp into the next /audio/out frame; returns the monotonic time it was sent."""
        self._arm = asyncio.get_running_loop().create_future()
        return await asyncio.wait_for(self._arm, timeout)

    async def wait_ready(self, wait_s: float) -> bool:
        end = time.perf_counter() + wait_s
        while time.perf_counter() < end:
            if self.out_connected.is_set() and self.audio_in_connected and self.chunks_in > 5:
                return True
            await asyncio.sleep(0.2)
        return False


# ---------------------------------------------------------------- the mock headset


class MockHeadset:
    """Stands in for the Quest: streams a quiet mic, plays /audio/out after a simulated delay into its own mic."""

    def __init__(self, port: int, up_ms: float = 15.0, down_ms: float = 15.0, prefill_frames: int = 3,
                 device_ms: float = 30.0, jitter_ms: float = 4.0, seed: int = 1, rate: int = 48000, coupling: float = 0.5):
        self.port, self.rate, self.coupling = port, rate, coupling
        self.up_ms, self.down_ms, self.prefill_frames = up_ms, down_ms, prefill_frames
        self.device_ms, self.jitter_ms = device_ms, jitter_ms
        self.rng = np.random.default_rng(seed)
        self.mic_t0 = 0.0
        self.playback: list[tuple[int, np.ndarray]] = []   # (mic sample index, mono samples)
        self.tasks: list[asyncio.Task] = []

    def expected_ms(self) -> float:
        """Simulated delay a perfect measurement returns (jitter adds jitter_ms/2 on average)."""
        return self.down_ms + self.prefill_frames * FRAME_S * 1000 + self.device_ms + self.up_ms + self.jitter_ms / 2

    async def start(self) -> None:
        from websockets.asyncio.client import connect

        base = f"ws://127.0.0.1:{self.port}"
        self.audio_ws = await connect(base + p.WS_AUDIO_IN, max_size=None)
        self.out_ws = await connect(base + p.WS_AUDIO_OUT, max_size=None)
        self.pose_ws = await connect(base + p.WS_POSE_IN)
        self.mic_t0 = time.perf_counter()
        self.tasks = [asyncio.create_task(self._mic()), asyncio.create_task(self._speaker())]

    async def stop(self) -> None:
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for name in ("audio_ws", "out_ws", "pose_ws"):
            if hasattr(self, name):
                await getattr(self, name).close()

    async def _speaker(self) -> None:
        async for data in self.out_ws:
            hdr, pcm = p.unpack(data, p.AudioOut)
            x = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768.0
            if not x.any():
                continue
            play_at = time.perf_counter() + (self.down_ms + self.prefill_frames * FRAME_S * 1000 + self.device_ms) / 1000
            self.playback.append((int(round((play_at - self.mic_t0) * self.rate)), x.mean(axis=1)))

    async def _mic(self) -> None:
        n = int(self.rate * 0.02)
        queue: asyncio.Queue = asyncio.Queue()
        sender = asyncio.create_task(self._uplink(queue))
        try:
            k = 0
            while True:
                due = self.mic_t0 + (k + 1) * 0.02
                await asyncio.sleep(max(0.0, due - time.perf_counter()))
                m0, m1 = k * n, (k + 1) * n
                mono = self.rng.standard_normal(n).astype(np.float32) * 0.003
                keep = []
                for start, s in self.playback:
                    a, b = max(start, m0), min(start + len(s), m1)
                    if b > a:
                        mono[a - m0:b - m0] += s[a - start:b - start] * self.coupling
                    if start + len(s) > m1:
                        keep.append((start, s))
                self.playback = keep
                pcm = (np.clip(np.stack([mono, mono], axis=1), -1, 1) * 32767).astype("<i2")
                hdr = p.AudioChunk(seq=k, device_ts_us=int(m0 / self.rate * 1e6), epoch=0, sample_rate=self.rate,
                                   channels=2, frames=n)
                queue.put_nowait((due, p.pack(hdr, pcm.tobytes())))  # stamped with the capture clock, not when we got to it
                k += 1
        finally:
            sender.cancel()

    async def _uplink(self, queue: asyncio.Queue) -> None:
        last_due = 0.0
        while True:
            made, msg = await queue.get()
            due = max(last_due, made + (self.up_ms + self.rng.uniform(0, self.jitter_ms)) / 1000)
            last_due = due
            await asyncio.sleep(max(0.0, due - time.perf_counter()))
            await self.audio_ws.send(msg)


# ---------------------------------------------------------------- the run


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def run_trials(hub: DelayHub, n: int, gap_s: tuple[float, float], timeout_s: float, seed: int,
                     quiet: bool = False) -> list[Trial]:
    rng = np.random.default_rng(seed)
    trials = []
    for i in range(n):
        await asyncio.sleep(float(rng.uniform(*gap_s)))
        t_send = await hub.send_click()
        await asyncio.sleep(timeout_s)
        hit = hub.analyze(t_send)
        tr = Trial(i + 1, None if hit is None else hit[0] * 1000, None if hit is None else hit[1])
        trials.append(tr)
        if not quiet:
            print(f"  trial {tr.index:3d}: " + (f"{tr.delay_ms:7.1f} ms  (match {tr.ncc:.2f})" if hit else "not heard"),
                  flush=True)
    return trials


def _timer_resolution(on: bool) -> None:
    """Windows sleeps in ~15.6 ms steps unless asked for 1 ms; that would blur a delay measured in ms."""
    if sys.platform == "win32":
        import ctypes
        (ctypes.windll.winmm.timeBeginPeriod if on else ctypes.windll.winmm.timeEndPeriod)(1)


async def run(args) -> dict:
    import uvicorn

    _timer_resolution(True)

    hub = DelayHub(args.layer, args.layout, silence=not args.no_silence)
    port = free_port() if args.mock else args.port
    host = "127.0.0.1" if args.mock else "0.0.0.0"
    server = uvicorn.Server(uvicorn.Config(hub.app, host=host, port=port, log_level="warning"))
    serve = asyncio.create_task(server.serve())
    while not server.started and not serve.done():
        await asyncio.sleep(0.05)
    if serve.done():
        serve.result()
    mock = None
    result: dict = {}
    try:
        if args.mock:
            mock = MockHeadset(port, args.sim_up_ms, args.sim_down_ms, args.sim_prefill_frames, args.sim_device_ms,
                               args.sim_jitter_ms, args.seed, coupling=args.sim_coupling)
            await mock.start()
            print(f"MOCK headset connected (simulated delay {mock.expected_ms():.0f} ms)")
        else:
            print(f"delay-test hub on ws://{host}:{port}. Start the headset app now (see the top of "
                  f"eval/delay_test.py); waiting up to {args.wait_s:g} s ...")
        if not await hub.wait_ready(args.wait_s):
            print("No headset audio connection (mic stream and /audio/out) within the wait time.")
            return {"error": "no headset connected", "pass": False}
        if hub.hello:
            print(f"headset hello: {hub.hello}")
        print(f"running {args.trials} trials (chirp on the {args.layer} layer, {args.layout} layout) ...")
        trials = await run_trials(hub, args.trials, (args.gap_min_s, args.gap_max_s), args.timeout_s, args.seed,
                                  args.quiet)
        s = summarize([t.delay_ms for t in trials if t.delay_ms is not None], len(trials),
                      args.budget_median_ms, args.budget_worst_ms)
        s["evidence"] = ("MOCKED: software loop on this machine, simulated delay "
                         f"{mock.expected_ms():.0f} ms" if mock else
                         "headset acoustic loop on the hub clock (hardware-tested only if a real Quest was connected)")
        s["mode"] = "mock" if mock else "headset"
        s["layer"], s["layout"] = args.layer, args.layout
        s["trials"] = [{"i": t.index, "delay_ms": t.delay_ms, "ncc": t.ncc} for t in trials]
        if mock:
            s["simulated_expected_ms"] = mock.expected_ms()
        result = s
    finally:
        if mock:
            await mock.stop()
        server.should_exit = True
        await asyncio.gather(serve, return_exceptions=True)
        _timer_resolution(False)
    return result


def print_report(s: dict) -> None:
    print()
    if "median_ms" in s:
        print(f"round trip over {s['n_valid']} of {s['n_trials']} trials ({s['n_missed']} not heard):")
        print(f"  median {s['median_ms']:.0f} ms   p90 {s['p90_ms']:.0f} ms   worst {s['worst_ms']:.0f} ms   "
              f"best {s['min_ms']:.0f} ms   jitter (std) {s['jitter_ms']:.1f} ms")
    else:
        print(f"no trial was heard ({s['n_trials']} sent): the mics did not pick up the speaker")
    print(f"budget (assumed, not a team target): median <= {s['budget_median_ms']:g} ms, worst <= {s['budget_worst_ms']:g} ms")
    print(f"evidence: {s['evidence']}")
    print("RESULT: " + ("PASS" if s["pass"] else "FAIL (" + "; ".join(s["fail_reasons"]) + ")"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mock", action="store_true", help="use a simulated headset in this process (no hardware)")
    ap.add_argument("--trials", type=int, default=40, help=f"chirps to send; >= {MIN_TRIALS} valid ones are needed to pass")
    ap.add_argument("--port", type=int, default=p.DEFAULT_PORT)
    ap.add_argument("--wait-s", type=float, default=120.0, help="how long to wait for the headset to connect")
    ap.add_argument("--timeout-s", type=float, default=0.8, help="how long to listen for each chirp")
    ap.add_argument("--gap-min-s", type=float, default=1.0)
    ap.add_argument("--gap-max-s", type=float, default=1.6)
    ap.add_argument("--layer", choices=["inner", "outer", "both"], default="inner",
                    help="/audio/out layer carrying the chirp (layers layout)")
    ap.add_argument("--layout", choices=["layers", "stereo"], default="layers")
    ap.add_argument("--no-silence", action="store_true", help="send frames only for the chirp (headset must re-prime each time)")
    ap.add_argument("--budget-median-ms", type=float, default=BUDGET_MEDIAN_MS)
    ap.add_argument("--budget-worst-ms", type=float, default=BUDGET_WORST_MS)
    ap.add_argument("--json", type=Path, default=None, help="write the summary and every trial here")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quiet", action="store_true", help="no per-trial lines")
    sim = ap.add_argument_group("mock headset (only with --mock)")
    sim.add_argument("--sim-up-ms", type=float, default=15.0, help="mic chunk uplink delay")
    sim.add_argument("--sim-down-ms", type=float, default=15.0, help="/audio/out downlink delay")
    sim.add_argument("--sim-prefill-frames", type=int, default=3, help="20 ms frames held in the jitter buffer")
    sim.add_argument("--sim-device-ms", type=float, default=30.0, help="output device latency")
    sim.add_argument("--sim-coupling", type=float, default=0.5,
                     help="how loudly the mock mic hears its speaker (0 = not at all: every trial 'not heard')")
    sim.add_argument("--sim-jitter-ms", type=float, default=4.0, help="uniform extra uplink jitter, 0..this")
    a = ap.parse_args(argv)
    if a.trials < MIN_TRIALS:
        print(f"note: {a.trials} < {MIN_TRIALS} trials cannot give a PASS (debug run)", file=sys.stderr)
    s = asyncio.run(run(a))
    if "error" in s:
        return 2
    print_report(s)
    if a.json:
        a.json.write_text(json.dumps(s, indent=1))
        print(f"wrote {a.json}")
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
