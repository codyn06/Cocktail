"""Score the Phase 2 speech branch (MossFormer2_SS + TalkNet association, optional AV-TSE) on the
controlled mixtures. Evidence: SYNTHETIC mixtures of recorded clips, models hardware-run.

    uv run python ../eval/av_scene/run_speech.py --out sessions/av_matrix/speech [--tse]

Per case and per prompted person this records what the deployment path returns (a stream or a
refusal), SI-SDRi against the person's clean reference when it is non-silent, output level when the
person is silent, and whether the output is closer to the target than to the strongest competitor.
Face boxes are S3FD detections assigned to the case's person masks, as in the pipeline; no clean
reference is an input. SI-SDRi is meaningless when the mixture equals the reference (solo cases).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
HUB = HERE.parents[1] / "hub"
for p in (str(HUB), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import metrics as M  # noqa: E402
import mixtures as X  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from av_scene.speech import run_speech_branch  # noqa: E402
from vision.face_detect import FaceDetector, face_boxes_by_track  # noqa: E402
from ingest.media_io import write_wav  # noqa: E402

SR = 16000
CASES = ("overlap_stack_AB", "overlap_stack_BA", "overlap_side_AB", "overlap_side_BA", "A_alone_stack_AB",
         "B_alone_stack_AB", "silent_stack_AB", "offscreen3_stack_AB", "offscreen_only_stack_AB",
         "speech_music_stack_AB", "overlap3_side_ABC")


def _score(est, case, t, mix16):
    refs = {k: resample_poly(v, 1, 3).astype(np.float32) for k, v in case.refs.items()}
    row = {"out_dbfs": round(M.output_dbfs(est), 1)}
    if t.ref and not M.is_silent(refs[t.ref]):
        row["si_sdri"] = (M.si_sdri(est, refs[t.ref], mix16, SR) or {}).get("si_sdri")
    else:
        row["target_silent"] = True
        row["out_minus_mix_db"] = round(M.output_dbfs(est) - M.output_dbfs(mix16), 1)
    if t.ref and t.other:
        sc = M.selection_correct(est, refs[t.ref], refs[t.other], SR)
        row["selection"] = sc and {"correct": sc["correct"], "margin_db": round(sc["margin_db"], 1)}
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tse", action="store_true")
    ap.add_argument("--gate", action="store_true", help="Phase 3 slice 1: TalkNet + TSE activity gate (needs --tse)")
    ap.add_argument("--offscreen", action="store_true", help="Phase 3 slice 2: upgraded outside_view rule")
    ap.add_argument("--peel", action="store_true", help="Phase 3 slice 3b: residual speech peeling for a 3rd voice")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from separate.models import VoiceSeparator
    from vision.talknet import TalkNetScorer
    sep, scorer, det = VoiceSeparator("cuda"), TalkNetScorer(device="cuda"), FaceDetector("cuda")
    tse = None
    if a.tse:
        from separate.av_tse import AVTargetSpeakerExtractor
        tse = AVTargetSpeakerExtractor(device="cuda")

    rows = []
    for case in X.build_cases(include=CASES):
        if case.case_id not in CASES:
            continue
        mix16 = resample_poly(case.audio, 1, 3).astype(np.float32)
        rgb = case.frames.permute(0, 2, 3, 1).contiguous().numpy()
        fps = rgb.shape[0] / (len(case.audio) / X.RATE)
        boxes = face_boxes_by_track(det, rgb, {t.name: case.masks[t.mask].numpy() for t in case.targets})
        cover = {k: round(float(np.mean([b is not None for b in v])), 3) for k, v in boxes.items()}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = run_speech_branch(mix16, rgb, fps, boxes, separator=sep, scorer=scorer, tse=tse,
                              activity_gate=a.gate, offscreen_upgrade=a.offscreen, peel=a.peel)
        wall = time.perf_counter() - t0
        d = out / case.case_id
        for k, x in r.streams.items():
            write_wav(d / f"{k}.wav", x, SR)
        for t in case.targets:
            stream, ext = r.selection[t.name]
            est = r.streams[stream] if stream else np.zeros_like(mix16)
            row = {"case": case.case_id, "target": t.name, "stream": stream, "status": ext.status,
                   "reason": ext.reason, "face_coverage": cover[t.name], **_score(est, case, t, mix16)}
            # the oracle-best stream says whether separation or attribution failed (DIAGNOSTIC)
            if t.ref and not M.is_silent(case.refs[t.ref]):
                ref = resample_poly(case.refs[t.ref], 1, 3).astype(np.float32)
                row["oracle_best_stream_si_sdri"] = round(max(
                    (M.si_sdri(s, ref, mix16, SR) or {}).get("si_sdri", -99) for s in r.streams.values()), 1)
            if t.name in r.tse:
                write_wav(d / f"tse_{t.name}.wav", r.tse[t.name], SR)
                row["tse"] = _score(r.tse[t.name], case, t, mix16)
            act = np.nan_to_num(r.track_scores[t.name], nan=0)
            row["talknet_mean_p"] = round(float(act.mean()), 3)
            row["activity_s"] = round(sum(b - a for a, b in r.activity_us[t.name]) / 1e6, 2)
            rows.append(row)
            print(json.dumps(row, default=float), flush=True)
        rows.append({"case": case.case_id, "wall_s": round(wall, 2),
                     "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                     "associations": [(x.source_id, x.track_id, x.status, x.visibility,
                                       None if x.score is None else round(x.score, 3)) for x in r.associations]})
        print(json.dumps(rows[-1], default=float), flush=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
