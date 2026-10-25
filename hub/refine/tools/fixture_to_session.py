"""Convert an AV fixture into a mock-client Session folder (Phase 5 replay input).

Output (readable by mockclient.session.Session.load, streamable by mockclient.replay.replay):
  session.json   {"device":"mock","sample_rate":16000,"start_ts_us":1000000,"camera_hfov_deg":90.0,
                  "source":..., "ground_truth":{...}}
  audio.wav      16 kHz mono int16
  pose.jsonl     identity rotation (0,0,0,1) at 10 Hz covering the clip
  video/<device_ts_us>.jpg + video/frames.jsonl   frames at --fps, device_ts_us = start + i*1e6/fps
  truth.json     (--case only) ground truth known from the eval Case, see case_truth().

Usage (from hub/):
  python -m refine.tools.fixture_to_session --mp4 PATH --out DIR [--fps 25] [--seconds N]
  python -m refine.tools.fixture_to_session --case overlap_stack_AB --out DIR [--sessions-dir DIR]

Heavy imports (torch, eval mixtures) are lazy; --mp4 needs only PyAV, cv2, numpy, scipy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

START_TS_US = 1_000_000
RATE = 16_000
POSE_HZ = 10
HFOV = 90.0


def decode_mp4(path, seconds=None):
    """-> (frames [(pts_s, HxWx3 BGR uint8)], mono float32 audio @16 kHz). Both cut to `seconds`."""
    import av

    frames, chunks = [], []
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        astream = c.streams.audio[0] if c.streams.audio else None
        res = av.AudioResampler(format="s16", layout="mono", rate=RATE) if astream else None
        t0 = None
        for pkt in c.demux(*([vs] + ([astream] if astream else []))):
            for f in pkt.decode():
                if isinstance(f, av.VideoFrame):
                    t = float(f.pts * vs.time_base) if f.pts is not None else len(frames) / float(vs.average_rate)
                    t0 = t if t0 is None else t0
                    if seconds is None or t - t0 < seconds:
                        frames.append((t - t0, f.to_ndarray(format="bgr24")))
                else:
                    for r in res.resample(f):
                        chunks.append(r.to_ndarray().reshape(-1))
        for r in (res.resample(None) if res else []):
            chunks.append(r.to_ndarray().reshape(-1))
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0 if chunks else np.zeros(0, np.float32)
    if seconds is not None:
        audio = audio[: int(seconds * RATE)]
    return frames, audio


def resample_frames(frames, fps, duration):
    """Pick for output index i the latest decoded frame with pts <= i/fps. -> list of BGR arrays."""
    pts = np.array([t for t, _ in frames])
    n = int(np.floor(duration * fps + 1e-6))
    idx = np.clip(np.searchsorted(pts, np.arange(n) / fps + 1e-6, side="right") - 1, 0, len(frames) - 1)
    return [frames[i][1] for i in idx]


def write_session(out, frames_bgr, audio16k, fps, meta_extra, truth=None):
    import cv2
    from scipy.io import wavfile

    out = Path(out)
    (out / "video").mkdir(parents=True, exist_ok=True)
    meta = {"device": "mock", "sample_rate": RATE, "start_ts_us": START_TS_US,
            "camera_hfov_deg": HFOV, **meta_extra}
    (out / "session.json").write_text(json.dumps(meta, indent=1))
    wavfile.write(out / "audio.wav", RATE, (np.clip(audio16k, -1, 1) * 32767).astype(np.int16))
    dur = len(audio16k) / RATE
    n_pose = int(dur * POSE_HZ) + 1
    (out / "pose.jsonl").write_text("".join(
        json.dumps({"device_ts_us": START_TS_US + round(i * 1e6 / POSE_HZ), "pos": [0.0, 0.0, 0.0],
                    "rot_xyzw": [0.0, 0.0, 0.0, 1.0]}) + "\n" for i in range(n_pose)))
    rows = []
    for i, bgr in enumerate(frames_bgr):
        ts = START_TS_US + round(i * 1e6 / fps)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        (out / "video" / f"{ts}.jpg").write_bytes(buf.tobytes())
        h, w = bgr.shape[:2]
        rows.append({"device_ts_us": ts, "width": w, "height": h, "cam_pos": None,
                     "cam_rot_xyzw": None, "hfov_deg": HFOV})
    (out / "video" / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if truth is not None:
        (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return {"duration_s": dur, "frames": len(frames_bgr), "fps": fps, "out": str(out)}


def convert_mp4(mp4, out, fps=25.0, seconds=None):
    frames, audio = decode_mp4(mp4, seconds)
    if not frames:
        raise ValueError(f"no video frames in {mp4}")
    dur = len(audio) / RATE if len(audio) else frames[-1][0] + 1.0 / fps
    return write_session(out, resample_frames(frames, fps, dur), audio, fps,
                         {"source": str(mp4), "ground_truth": {"kind": "mp4", "note": "no per-face truth"}})


def _bbox(mask_t):
    ys, xs = np.nonzero(mask_t)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


def case_truth(case):
    """What the eval Case knows. Per target: name, ref key (None = target absent from the audio),
    strongest competitor, tile bbox [x0,y0,x1,y1] in output-frame pixels (from the mask),
    and its active speech spans in seconds. Also layout, frame size, and scalar case meta."""
    targets = []
    for t in case.targets:
        spans = case.spans.get(t.ref or t.name, [])
        targets.append({"name": t.name, "ref": t.ref, "other": t.other, "label": t.label, "text": t.text,
                        "face_tile_bbox_xyxy": _bbox(case.masks[t.mask][0].numpy()),
                        "speaks": t.ref is not None, "active_spans_s": [[float(a), float(b)] for a, b in spans]})
    h, w = case.frames.shape[-2:]
    return {"case_id": case.case_id, "description": case.description, "layout": case.layout,
            "frame_size_wh": [int(w), int(h)], "targets": targets,
            "all_spans_s": {k: [[float(a), float(b)] for a, b in v] for k, v in case.spans.items()},
            "case_meta": {k: v for k, v in case.meta.items() if isinstance(v, (int, float, str, bool))},
            "note": "Correct match for the voice of target X = the face inside X's face_tile_bbox; "
                    "targets with speaks=false are visible but silent (distractors)."}


def convert_case(case_id, out, sessions_dir=None):
    import av_scene  # noqa: F401  registers FFmpeg DLLs before torchcodec (lazy: pulls torch)
    import torch
    import torchaudio.functional as AF

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval" / "av_scene"))
    import mixtures as X

    sess = Path(sessions_dir) if sessions_dir else X.SESSIONS
    case = next((c for c in X.build_cases(sess, include=[case_id]) if c.case_id == case_id), None)
    if case is None:
        raise SystemExit(f"case {case_id!r} not found (or its assets are missing) under {sess}")
    fps = round(case.frames.shape[0] / case.duration, 3)
    a = torch.from_numpy(np.ascontiguousarray(case.audio, dtype=np.float32))
    a16 = AF.resample(a, X.RATE, RATE).numpy()
    bgr = [f.permute(1, 2, 0).numpy()[:, :, ::-1].copy() for f in case.frames]
    return write_session(out, bgr, a16, fps,
                         {"source": f"case:{case_id}", "ground_truth": {"kind": "case", "case_id": case_id}},
                         truth=case_truth(case))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mp4")
    g.add_argument("--case")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--seconds", type=float)
    p.add_argument("--sessions-dir")
    a = p.parse_args(argv)
    r = convert_mp4(a.mp4, a.out, a.fps, a.seconds) if a.mp4 else convert_case(a.case, a.out, a.sessions_dir)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
