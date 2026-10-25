"""Deterministic, truth-bearing synthetic AV fixture for Phase 1 (evidence tier: mocked/synthetic).

Run from `hub/`:  uv run python ../eval/av_scene/generate_fixture.py --out sessions/phase1_fixture/fixture

Scene (8 s, 48 kHz two-microphone pair 14 cm apart, 320x240 variable-frame-rate video):
  alice   visible person, left, talks 0.6-2.4 s and 4.2-6.0 s; turns away (|yaw| 75 deg) 4.0-6.2 s
  bob     visible person, centre, talks 3.4-5.0 s (overlaps alice 4.2-5.0 s) and 6.8-7.6 s;
          briefly occluded 5.2-5.4 s
  carol   visible SILENT bystander on the right who moves her mouth 1.0-1.8 s (silent-mouth decoy)
  cat1    visible cat, chirps 2.7-3.05 s (3->5 kHz up) and 6.3-6.65 s (5->3.2 kHz down)
  knock   OFF-SCREEN broadband knock 5.7-6.3 s (bearing 60 deg, outside the 56 deg field of view):
          it must stay unattributed
Sound reaches the microphones after its acoustic time of flight (range / c) and with the exact
inter-microphone delay of its true ray, so all timing and spatial evidence is physically
consistent. Nothing here is learned perception, a room recording or a device measurement.

On-disk fixture:
  fixture.json            truth (activity, t_visual_truth, source<->track, visibility, face yaw, geometry)
  master.wav              float32 (n, 2) mixture the microphones hear
  truth_stems/<src>.wav   each source alone in the declared mono downmix (mean over the two mics)
  frames/NNNNNN.png       video frames at variable presentation timestamps
  masks_truth/<track>.jsonl   hand-authored masks (RLE, identity transform)
  visual_motion.json      per-frame noisy mouth/body motion measurement per track (what vision would see)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

HUB = Path(__file__).resolve().parents[2] / "hub"
if str(HUB) not in sys.path:
    sys.path.insert(0, str(HUB))

from ingest.media_io import write_png, write_wav  # noqa: E402
from locate.av_spatial import c_sound, direction_from_az_el, ray_from_pixel, tau_geom  # noqa: E402
from vision.mask_store import FrameTransform, MaskRecord, encode_mask, write_track_jsonl  # noqa: E402

FS = 48000
DUR = 8.0
W, H = 320, 240
CAM = dict(fx=300.0, fy=300.0, cx=160.0, cy=120.0)
MICS = [[-0.07, 0.0, 0.0], [0.07, 0.0, 0.0]]
TEMP_C = 20.0
EVIDENCE_TIER = "mocked/synthetic"

TRACKS = [
    dict(id="alice", concept="person", center=(70, 125), rx=26, ry=58, color=(235, 130, 90)),
    dict(id="bob", concept="person", center=(170, 125), rx=28, ry=60, color=(90, 160, 235)),
    dict(id="carol", concept="person", center=(270, 130), rx=26, ry=56, color=(120, 205, 130)),
    dict(id="cat1", concept="cat", center=(228, 196), rx=22, ry=16, color=(240, 205, 70)),
]
SOURCES = [
    dict(id="spk_alice", kind="speech", label="speech (alice)", track="alice", range_m=2.0, level_dbfs=-24.0,
         turns=[(0.6, 2.4), (4.2, 6.0)], f0=190.0, formants=(600.0, 1700.0)),
    dict(id="spk_bob", kind="speech", label="speech (bob)", track="bob", range_m=3.0, level_dbfs=-24.0,
         turns=[(3.4, 5.0), (6.8, 7.6)], f0=110.0, formants=(450.0, 1200.0)),
    dict(id="cat_chirp", kind="non_speech", label="cat chirp", track="cat1", range_m=1.5, level_dbfs=-26.0,
         turns=[(2.7, 3.05), (6.3, 6.65)], chirps=[(3000.0, 5000.0), (5000.0, 3200.0)]),
    dict(id="knock_offscreen", kind="non_speech", label="off-screen knock", track=None, range_m=4.0,
         level_dbfs=-26.0, turns=[(5.7, 6.3)], bearing=(60.0, 5.0)),
]
OCCLUSIONS = {"bob": [(5.2, 5.4)]}
FACE_YAW = {"alice": [(0.0, 4.0, 8.0), (4.0, 6.2, 75.0), (6.2, 8.0, 8.0)], "bob": [(0.0, 8.0, 0.0)],
            "carol": [(0.0, 8.0, 15.0)]}
DECOY = {"carol": [(1.0, 1.8)]}                  # silent mouth movement, no sound
NOISE_DBFS = -70.0


def frac_delay(x: np.ndarray, delay_s: float) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * delay_s), len(x))


def scale_to(x: np.ndarray, dbfs: float) -> np.ndarray:
    act = x[np.abs(x) > 1e-9]
    return x * (10 ** (dbfs / 20.0) / np.sqrt(np.mean(act ** 2)))


def voiced_turn(n: int, t0: float, t1: float, f0: float, formants, rng) -> np.ndarray:
    t = np.arange(n) / FS
    out = np.zeros(n)
    a, b = int(t0 * FS), int(t1 * FS)
    tt = t[a:b]
    f = f0 * (1 + 0.012 * np.sin(2 * np.pi * 5.0 * tt))
    phase = 2 * np.pi * np.cumsum(f) / FS
    sig = np.zeros(len(tt))
    for h in range(1, 28):
        fh = h * f0
        if fh > 3800:
            break
        amp = (1.0 / h) * (1 + sum(2.0 * np.exp(-((fh - F) / 280.0) ** 2) for F in formants))
        sig += amp * np.sin(h * phase)
    syl = 0.65 + 0.35 * np.sin(2 * np.pi * 3.3 * (tt - t0)) ** 2
    ramp = np.minimum(1.0, np.minimum((tt - t0) / 0.008, (t1 - tt) / 0.015))
    breath = 0.03 * signal.sosfilt(signal.butter(2, [800, 4000], "bandpass", fs=FS, output="sos"),
                                   rng.standard_normal(len(tt)))
    out[a:b] = (sig * syl + breath) * ramp
    return out


def chirp_turn(n: int, t0: float, t1: float, f_from: float, f_to: float) -> np.ndarray:
    out = np.zeros(n)
    a, b = int(t0 * FS), int(t1 * FS)
    tt = np.arange(b - a) / FS
    out[a:b] = signal.chirp(tt, f_from, tt[-1], f_to) * np.hanning(b - a) ** 0.3
    return out


def burst_turn(n: int, t0: float, t1: float, rng) -> np.ndarray:
    out = np.zeros(n)
    a, b = int(t0 * FS), int(t1 * FS)
    tt = np.arange(b - a) / FS
    body = signal.sosfilt(signal.butter(4, [500, 2500], "bandpass", fs=FS, output="sos"), rng.standard_normal(b - a))
    out[a:b] = body * np.exp(-tt / 0.18) * np.minimum(1.0, tt / 0.002)
    return out


def source_ray(src: dict) -> np.ndarray:
    if src.get("track"):
        tr = next(t for t in TRACKS if t["id"] == src["track"])
        return ray_from_pixel(*tr["center"], **CAM)
    return direction_from_az_el(*src["bearing"])


def synth_audio(rng):
    n = int(DUR * FS)
    c = c_sound(TEMP_C)
    p1, p2 = np.array(MICS[0]), np.array(MICS[1])
    stems, ch0, ch1, truth_sources = {}, np.zeros(n), np.zeros(n), []
    for src in SOURCES:
        raw = np.zeros(n)
        for k, (t0, t1) in enumerate(src["turns"]):
            if src["kind"] == "speech":
                raw += voiced_turn(n, t0, t1, src["f0"], src["formants"], rng)
            elif src["id"] == "cat_chirp":
                raw += chirp_turn(n, t0, t1, *src["chirps"][k])
            else:
                raw += burst_turn(n, t0, t1, rng)
        raw = scale_to(raw, src["level_dbfs"])
        r = source_ray(src)
        tof = src["range_m"] / c
        tau = float(tau_geom(r, p1, p2, TEMP_C))               # positive = channel 0 later
        s1 = frac_delay(raw, tof)
        s0 = frac_delay(raw, tof + tau)
        ch1 += s1
        ch0 += s0
        stems[src["id"]] = 0.5 * (s0 + s1)       # what the declared mean-over-channels downmix hears
        truth_sources.append(dict(
            id=src["id"], kind=src["kind"], label=src["label"], track_id=src["track"], range_m=src["range_m"],
            tof_ms=tof * 1000.0, ray=[float(v) for v in r], tau_s=tau, level_dbfs=src["level_dbfs"],
            activity_us=[[int(a * 1e6), int(b * 1e6)] for a, b in src["turns"]],
            t_visual_truth_us=[int(a * 1e6) for a, _ in src["turns"]], stem_file=f"truth_stems/{src['id']}.wav"))
    amp = 10 ** (NOISE_DBFS / 20.0)
    ch0 += amp * rng.standard_normal(n)
    ch1 += amp * rng.standard_normal(n)
    return np.stack([ch0, ch1], 1).astype(np.float32), stems, truth_sources


def make_pts(rng) -> list[int]:
    pts, t = [], 0
    while t < DUR * 1e6:
        pts.append(int(t))
        t += 40_000 + int(rng.integers(-4000, 4001))
    del pts[100]                                              # one dropped frame (an 80 ms hole)
    return pts


def openness(track_id: str, t: float) -> float:
    on = 0.0
    spans = [turn for s in SOURCES if s["track"] == track_id for turn in s["turns"]] + DECOY.get(track_id, [])
    for a, b in spans:
        if a <= t < b:
            on = max(on, min(1.0, (t - a) / 0.02))
    return on


def occluded(track_id: str, t: float) -> bool:
    return any(a <= t < b for a, b in OCCLUSIONS.get(track_id, []))


def ellipse(cx, cy, rx, ry) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def render_frame(t: float) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (28, 32, 44)
    img[170:] = (44, 48, 60)
    for tr in TRACKS:
        cx, cy = tr["center"]
        cx = cx + 3.0 * np.sin(2 * np.pi * 0.4 * t + len(tr["id"]))
        m = ellipse(cx, cy, tr["rx"], tr["ry"])
        img[m] = tr["color"]
        if tr["concept"] == "person":
            head = ellipse(cx, cy - tr["ry"] - 6, 15, 15)
            img[head] = tuple(int(v * 0.9) for v in tr["color"])
            my = int(cy - tr["ry"] - 1)
            open_ = openness(tr["id"], t) > 0.5
            h = 7 if open_ else 2
            img[my:my + h, int(cx) - 5:int(cx) + 6] = (200, 30, 30) if open_ else (20, 20, 24)
        else:
            my = int(cy + 2)
            h = 6 if openness(tr["id"], t) > 0.5 else 1
            img[my:my + h, int(cx) - 4:int(cx) + 5] = (200, 30, 30) if h > 1 else (20, 20, 24)
        if occluded(tr["id"], t):
            img[int(cy - tr["ry"] - 22):int(cy + tr["ry"] + 4), int(cx - tr["rx"] - 4):int(cx + tr["rx"] + 4)] = (90, 90, 90)
    return img


def track_mask(tr: dict, t: float) -> np.ndarray:
    cx = tr["center"][0] + 3.0 * np.sin(2 * np.pi * 0.4 * t + len(tr["id"]))
    return ellipse(cx, tr["center"][1], tr["rx"], tr["ry"])


def generate(out_dir: str | Path, seed: int = 0) -> Path:
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    master, stems, truth_sources = synth_audio(rng)
    write_wav(out / "master.wav", master, FS, "float32")
    for sid, s in stems.items():
        write_wav(out / "truth_stems" / f"{sid}.wav", s.astype(np.float32), FS, "float32")

    pts = make_pts(rng)
    motion = {tr["id"]: [] for tr in TRACKS}
    tf = FrameTransform.identity(H, W)
    records = {tr["id"]: [] for tr in TRACKS}
    for i, p in enumerate(pts):
        t = p / 1e6
        write_png(out / "frames" / f"{i:06d}.png", render_frame(t))
        for tr in TRACKS:
            vis = "occluded" if occluded(tr["id"], t) else "visible"
            records[tr["id"]].append(MaskRecord(p, i, None if vis == "occluded" else encode_mask(track_mask(tr, t), tf),
                                                confidence=1.0, visibility=vis))
            val = openness(tr["id"], t)
            yaw = next((y for a, b, y in FACE_YAW.get(tr["id"], []) if a <= t < b), 0.0)
            if abs(yaw) > 60.0 or vis == "occluded":          # mouth not visible: the measurement is noise
                val = 0.25
            motion[tr["id"]].append(float(np.clip(val + rng.normal(0, 0.05), 0, 1)))
    for tid, recs in records.items():
        write_track_jsonl(out / "masks_truth" / f"{tid}.jsonl", recs)
    (out / "visual_motion.json").write_text(json.dumps({"frame_pts_us": pts, "motion": motion}))

    truth = dict(
        evidence_tier=EVIDENCE_TIER, seed=seed, sample_rate=FS, duration_s=DUR, nominal_fps=25.0, image=dict(w=W, h=H),
        frame_pts_us=pts, temperature_c=TEMP_C, camera=CAM, mic_positions_m=MICS,
        tracks=[dict(id=t["id"], concept=t["concept"], center_px=list(t["center"]),
                     silent=not any(s["track"] == t["id"] for s in SOURCES),
                     occlusions_us=[[int(a * 1e6), int(b * 1e6)] for a, b in OCCLUSIONS.get(t["id"], [])],
                     face_yaw=[[int(a * 1e6), int(b * 1e6), y] for a, b, y in FACE_YAW.get(t["id"], [])],
                     decoy_motion_us=[[int(a * 1e6), int(b * 1e6)] for a, b in DECOY.get(t["id"], [])])
                for t in TRACKS],
        sources=truth_sources,
        noise_floor_dbfs=NOISE_DBFS,
        notes=("Synthetic: sources are scripted signals; the visual measurement is a scripted mouth/body "
               "openness with Gaussian noise. Not learned perception."))
    (out / "fixture.json").write_text(json.dumps(truth, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="sessions/phase1_fixture/fixture")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(generate(a.out, a.seed))


if __name__ == "__main__":
    main()
