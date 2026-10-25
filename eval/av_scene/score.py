"""Scorer for the Phase 1 synthetic fixture bundle (evidence tier: mocked/synthetic).

Reads a bundle written by `av_scene.phase1.run` plus the fixture truth and reports:
  alignment    signed delta_audio_ms / delta_mask_ms vs t_visual_truth and delta_skew_ms between them,
               against [-15-q, +45+q] and |skew| <= 20+q (q = half the local bracketing frame interval)
  attribution  precision, coverage, abstention, wrong-object and silent-bystander highlight
  leakage      per-stem SI-SDR vs truth, cross-source leakage, duplicate-stem similarity
  remainder    false unexplained events, omitted-source recall (re-run with a stem removed)

Everything here is a deterministic-contract check of an oracle-derived scene. It says nothing about
learned perception, rooms or devices.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np

HUB = Path(__file__).resolve().parents[2] / "hub"
if str(HUB) not in sys.path:
    sys.path.insert(0, str(HUB))

from av_scene.timing import check_sync  # noqa: E402
from ingest.media_io import read_wav  # noqa: E402
from replay.residual import ResidualConfig, find_unexplained_events  # noqa: E402

TIER = "mocked/synthetic"


def _q_ms(pts, t_us):
    i = min(max(bisect.bisect_right(pts, t_us) - 1, 0), len(pts) - 2)
    return (pts[i + 1] - pts[i]) / 2000.0


def _mid(pts):
    p = np.asarray(pts, np.int64)
    return p + (np.concatenate([p[1:], [p[-1] + (p[-1] - p[-2])]]) - p) // 2


def si_sdr(est, ref):
    ref, est = np.asarray(ref, float), np.asarray(est, float)
    a = float(est @ ref) / float(ref @ ref + 1e-20)
    t = a * ref
    return float(10 * np.log10((t @ t + 1e-20) / (((est - t) @ (est - t)) + 1e-20)))


def score(bundle_dir: str | Path, fixture_dir: str | Path) -> dict:
    b, fx = Path(bundle_dir), Path(fixture_dir)
    truth = json.loads((fx / "fixture.json").read_text())
    man = json.loads((b / "manifest.json").read_text())
    pts = truth["frame_pts_us"]
    mid = _mid(pts)
    cfg_t = man["timing"]["gates"]
    act = {s["source_id"]: json.loads((b / man["assets"][f"activity:{s['source_id']}"]).read_text()) for s in man["sources"]}
    tstate = {t["track_id"]: json.loads((b / man["assets"][f"track_state:{t['track_id']}"]).read_text())["state"]
              for t in man["tracks"]}
    assoc = {a["source_id"]: a for a in man["associations"]}

    # ---- alignment
    turns = []
    for s in truth["sources"]:
        if not s["track_id"]:
            continue
        a = act[s["id"]]
        for k, (t0, t1) in enumerate(s["activity_us"]):
            tv = s["t_visual_truth_us"][k]
            seg = next((sg for sg in a["segments_us"] if sg[0] < t1 and sg[1] > t0), None)
            if seg is None:
                turns.append(dict(source=s["id"], turn=k, detected=False, ok=False))
                continue
            idx = [i for i in range(len(pts)) if a["active"][i] and seg[0] <= mid[i] < seg[1]]
            if not idx:
                turns.append(dict(source=s["id"], turn=k, detected=True, highlighted=False, ok=False))
                continue
            sync = check_sync(seg[0], pts[idx[0]], tv, _q_ms(pts, tv), cfg_t["gate_early_ms"], cfg_t["gate_late_ms"],
                              cfg_t["skew_base_ms"])
            turns.append(dict(source=s["id"], turn=k, detected=True, highlighted=True, **sync.to_dict()))
    align = dict(turns=turns, all_ok=bool(turns) and all(t["ok"] for t in turns),
                 max_abs_skew_ms=max((abs(t["delta_skew_ms"]) for t in turns if "delta_skew_ms" in t), default=None))

    # ---- attribution / highlight
    supported = [s for s in truth["sources"] if s["track_id"]]
    unsupported = [s for s in truth["sources"] if not s["track_id"]]
    pred = {s["id"]: assoc[s["id"]] for s in truth["sources"]}
    attributed = [sid for sid, a in pred.items() if a["status"] == "matched" and a["track_id"]]
    correct = [sid for sid in attributed if pred[sid]["track_id"] == next(s["track_id"] for s in truth["sources"] if s["id"] == sid)]
    truth_active = {t["id"]: np.zeros(len(pts), bool) for t in truth["tracks"]}
    for s in supported:
        for a0, b0 in s["activity_us"]:
            truth_active[s["track_id"]] |= (mid >= a0) & (mid < b0)
    frame_s = np.diff(pts, append=pts[-1] + 40000) / 1e6
    visible = {}
    for t in man["tracks"]:
        lines = [json.loads(l) for l in (b / t["mask_file"]).read_text().splitlines() if l.strip()]
        visible[t["track_id"]] = np.array([r["rle"] is not None for r in lines])
    outside_s, wrong_s, silent = 0.0, 0.0, {}
    hl_tp = hl_fp = hl_fn = 0
    for t in truth["tracks"]:
        hl = np.array([st != "idle" for st in tstate[t["id"]]])
        outside_s += float(frame_s[hl & ~truth_active[t["id"]]].sum())
        if t["silent"]:
            wrong_s += float(frame_s[hl].sum())
        hl_tp += int((hl & truth_active[t["id"]]).sum()); hl_fp += int((hl & ~truth_active[t["id"]]).sum())
        hl_fn += int((~hl & truth_active[t["id"]]).sum())
        if t["silent"]:
            vis = visible[t["id"]]
            silent[t["id"]] = dict(highlight_pct_of_visible_time=float(100.0 * frame_s[hl & vis].sum() / max(frame_s[vis].sum(), 1e-9)))
    attr = dict(
        precision=len(correct) / len(attributed) if attributed else None,
        coverage=len(correct) / len(supported) if supported else None,
        wrong_attributions=len(attributed) - len(correct),
        abstention_correct={s["id"]: pred[s["id"]]["status"] == "unmatched" and pred[s["id"]]["track_id"] is None
                            for s in unsupported},
        wrong_object_highlight_s=round(wrong_s, 3), highlight_outside_truth_s=round(outside_s, 3),
        frame_highlight_precision=hl_tp / max(hl_tp + hl_fp, 1), frame_highlight_recall=hl_tp / max(hl_tp + hl_fn, 1),
        silent_bystanders=silent, predicted={sid: dict(track=a["track_id"], status=a["status"]) for sid, a in pred.items()})

    # ---- leakage
    est = {s["source_id"]: read_wav(b / man["assets"][f"stem:{s['source_id']}"])[0][:, 0].astype(np.float64) for s in man["sources"]}
    ref = {s["id"]: read_wav(fx / s["stem_file"])[0][:, 0].astype(np.float64) for s in truth["sources"]}
    leak, ids = {}, list(est)
    for i in ids:
        others = [j for j in ids if j != i]
        cross = max(((float(est[i] @ ref[j]) / float(ref[j] @ ref[j] + 1e-20)) ** 2 * float(ref[j] @ ref[j]) for j in others), default=0.0)
        leak[i] = dict(si_sdr_db=si_sdr(est[i], ref[i]), max_cross_leakage_db=float(10 * np.log10(cross / float(est[i] @ est[i] + 1e-20) + 1e-30)))
    dup = max((abs(float(est[i] @ est[j])) / (np.linalg.norm(est[i]) * np.linalg.norm(est[j]) + 1e-20)
               for n, i in enumerate(ids) for j in ids[n + 1:]), default=0.0)

    # ---- remainder
    mix = read_wav(b / man["assets"]["mixture"])[0][:, 0].astype(np.float64)
    rate = man["media"]["audio"][0]["sample_rate"]
    rem = {r["kind"]: r for r in man["remainders"]}
    events = rem["unexplained_power"]["stft"].get("events", [])
    rc = ResidualConfig()
    omitted = {}
    for s in truth["sources"]:
        if s["kind"] == "speech":
            continue
        ev, _ = find_unexplained_events(mix, [est[i] for i in ids if i != s["id"]], rate, rc)
        hit = [any(e["start_s"] < b1 / 1e6 and e["end_s"] > a1 / 1e6 for e in ev) for a1, b1 in s["activity_us"]]
        omitted[s["id"]] = dict(turns=len(hit), detected=int(sum(hit)), recall=float(np.mean(hit)))
    remainder = dict(false_unexplained_events_full_inventory=len(events),
                     false_events_power_subtraction=len(rem["unexplained_power"]["stft"].get("events_power_subtraction", [])),
                     projection_error_db=rem["phase_projected_stft"]["projection_error_db"],
                     omitted_source_recall=omitted)
    return dict(evidence_tier=TIER, fixture=truth["notes"], alignment=align, attribution=attr,
                leakage=dict(per_stem=leak, max_duplicate_similarity=float(dup)), remainder=remainder)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("fixture")
    ap.add_argument("--write", action="store_true", help="write metrics.json into the bundle")
    a = ap.parse_args()
    m = score(a.bundle, a.fixture)
    if a.write:
        (Path(a.bundle) / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
