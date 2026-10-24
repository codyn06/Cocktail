"""Stand-in for the phone app: streams a test tone into the hub and listens to the loopback.

Measures the hub round trip (send to /ingest/audio -> receive on /audio/out). This
excludes the glasses, Bluetooth and phone playback, which need the real devices.

    python -m mock.phone_sim --seconds 10 --save loopback.wav
"""

import argparse
import asyncio
import json
import time
import urllib.request
import wave

import numpy as np
import websockets

from hub import protocol


def tone(rate: int, seconds: float, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def encode(x: np.ndarray, fmt: int) -> bytes:
    if fmt == protocol.FMT_INT16:
        return np.round(x * 32767).astype("<i2").tobytes()
    return x.astype("<f4").tobytes()


async def run(args) -> None:
    signal = tone(args.rate, args.seconds)
    chunk = args.rate * args.chunk_ms // 1000
    fmt = protocol.FMT_FLOAT32 if args.float32 else protocol.FMT_INT16
    sent_ns: dict[int, int] = {}
    rtt_ms: list[float] = []
    received = bytearray()

    async with websockets.connect(f"{args.url}/audio/out") as out_ws, websockets.connect(
        f"{args.url}/ingest/audio"
    ) as in_ws:

        async def listen():
            async for msg in out_ws:
                header, payload = protocol.unpack(msg)
                if header.ref_seq in sent_ns:
                    rtt_ms.append((time.perf_counter_ns() - sent_ns[header.ref_seq]) / 1e6)
                received.extend(payload)

        listener = asyncio.create_task(listen())
        start = time.perf_counter()
        for seq, i in enumerate(range(0, len(signal) - chunk + 1, chunk)):
            await asyncio.sleep(max(0.0, start + seq * args.chunk_ms / 1000 - time.perf_counter()))
            header = protocol.AudioHeader(
                seq=seq,
                epoch=1,
                pts_us=seq * args.chunk_ms * 1000,
                sample_rate=args.rate,
                channels=1,
                sample_format=fmt,
            )
            sent_ns[seq] = time.perf_counter_ns()
            await in_ws.send(protocol.pack(header, encode(signal[i : i + chunk], fmt)))
        await asyncio.sleep(0.5)  # let the last frames arrive
        listener.cancel()

    expected = int(args.seconds * 50)
    got = len(received) // (protocol.SAMPLE_BYTES[protocol.FMT_INT16] * 2 * 960)
    print(f"sent {len(sent_ns)} chunks of {args.chunk_ms} ms at {args.rate} Hz")
    print(f"received {got} of ~{expected} 20 ms frames")
    if rtt_ms:
        a = np.asarray(rtt_ms)
        print(
            f"hub round trip ms: p50 {np.percentile(a, 50):.2f}  p95 {np.percentile(a, 95):.2f}"
            f"  max {a.max():.2f}"
        )
    if args.save:
        with wave.open(args.save, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(48_000)
            w.writeframes(bytes(received))
        print(f"wrote {args.save}")
    http = args.url.replace("ws://", "http://", 1)
    with urllib.request.urlopen(f"{http}/stats") as r:
        print("hub /stats:", json.dumps(json.load(r), indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="ws://127.0.0.1:8000")
    p.add_argument("--rate", type=int, default=16_000, choices=[16_000, 44_100, 48_000])
    p.add_argument("--chunk-ms", type=int, default=20)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--float32", action="store_true", help="send float32 instead of int16")
    p.add_argument("--save", help="write the received loopback audio to this WAV file")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
