"""Bounded SAM Audio candidate matrix + MossFormer2 baseline on labeled SYNTHETIC mixtures.

Run from hub/ (GPU; HF_HUB_OFFLINE=1 keeps it from touching the network):
  uv run python ../eval/av_scene/run_matrix.py selftest --out sessions/av_matrix/run1
  uv run python ../eval/av_scene/run_matrix.py sam      --out sessions/av_matrix/run1
  uv run python ../eval/av_scene/run_matrix.py judge    --out sessions/av_matrix/run1
  uv run python ../eval/av_scene/run_matrix.py moss     --out sessions/av_matrix/run1
  uv run python ../eval/av_scene/run_matrix.py report   --out sessions/av_matrix/run1

Stages are resumable. Everything lands under --out (hub/sessions/ is gitignored; never commit it).
All candidate waveforms are retained as float32 WAV under <out>/cands/<trial>/<condition>/.

Conditions (overlap trials): prompts {person speaking | speech, single speaker | speech, one person |
gender text (woman/man speaking, matched to the prompted person: ORACLE-ish)} x anchors {none + predicted
spans | full-clip "+" | oracle "+" spans from the clean ref (DIAGNOSTIC) | oracle "+/-" spans (DIAGNOSTIC)}
plus a visual-off control. Seeds are identical across conditions (matched start noise).
"""

from __future__ import annotations

import argparse
import json
import os
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

SEEDS = list(range(100, 108))
NEUTRAL = {"p1": "person speaking", "p2": "speech, single speaker", "p3": "speech, one person"}
GENDER = {"A": "woman speaking", "B": "man speaking"}
OVERLAP = ("overlap_stack_AB", "overlap_stack_BA")
OTHERS = ("A_alone_stack_AB", "B_alone_stack_AB", "silent_stack_AB", "offscreen3_stack_AB",
          "offscreen_only_stack_AB", "speech_music_stack_AB", "nonspeech_two")


def anchors_for(mode: str, case: X.Case, target: X.Target):
    """-> (anchors, predict_spans) or None when the mode is undefined for this target."""
    if mode == "none":
        return None, True
    if mode == "full":
        return "full_clip", False
    if mode in ("oracle_pos", "oracle_posneg"):
        a = X.oracle_anchors(case, target, negatives=(mode == "oracle_posneg"))
        return (a, False) if a else None
    raise ValueError(mode)


def conditions(case: X.Case, target: X.Target):
    """Yield dicts describing every condition run for this trial."""
    if case.case_id in OVERLAP:
        prompts = {**NEUTRAL, "gender": GENDER[target.name]}
        for pk, text in prompts.items():
            for am in ("none", "full", "oracle_pos", "oracle_posneg"):
                yield {"prompt_key": pk, "text": text, "anchor": am, "visual": True, "n": 8}
        yield {"prompt_key": "p1", "text": NEUTRAL["p1"], "anchor": "full", "visual": False, "n": 8}
    else:
        for am in ("full", "none"):
            yield {"prompt_key": "p1", "text": target.text, "anchor": am, "visual": True, "n": 4}


def cid(c: dict) -> str:
    return f"{c['prompt_key']}_{c['anchor']}_{'vis' if c['visual'] else 'novis'}"


def trials(cases):
    for c in cases:
        for t in c.targets:
            yield c, t


def _read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["key"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


# --------------------------------------------------------------------------- metrics for one candidate set
def score_candidates(case: X.Case, target: X.Target, cands: list[np.ndarray], labeler=None) -> dict:
    mix = case.audio
    ref = case.refs[target.ref] if target.ref else None
    other = case.refs.get(target.other) if target.other else None
    rows = []
    for k, out in enumerate(cands):
        n = min(len(out), len(mix))
        out = out[:n]
        r = {"k": k, "out_dbfs": M.output_dbfs(out), "mix_dbfs": M.rms_dbfs(mix[:n]),
             "suppression_db": M.output_dbfs(out) - M.rms_dbfs(mix[:n])}
        if ref is not None:
            r["target"] = M.si_sdri(out, ref[:n], mix[:n], X.RATE)
            if other is not None and not M.is_silent(other):
                r["other"] = M.si_sdri(out, other[:n], mix[:n], X.RATE)
                r["selection"] = M.selection_correct(out, ref[:n], other[:n], X.RATE)
        rows.append(r)
    out = {"cands": rows}
    if ref is not None:
        s = np.array([r["target"]["si_sdri"] for r in rows])
        best = int(s.argmax())
        out["summary"] = {"cand0_sdri": float(s[0]), "best_idx": best, "best_sdri": float(s[best]), "mean_sdri": float(s.mean()),
                          "std_sdri": float(s.std()), "range_db": float(s.max() - s.min()),
                          "n_ge_3db": int((s >= 3.0).sum()), "n": len(s),
                          "cand0_selection_correct": (rows[0].get("selection") or {}).get("correct"),
                          "frac_selection_correct": (float(np.mean([bool(r["selection"]["correct"]) for r in rows]))
                                                     if "selection" in rows[0] and rows[0]["selection"] else None)}
    else:
        o = np.array([r["out_dbfs"] for r in rows])
        out["summary"] = {"cand0_out_dbfs": float(o[0]), "mean_out_dbfs": float(o.mean()), "max_out_dbfs": float(o.max()),
                          "mix_dbfs": rows[0]["mix_dbfs"], "cand0_suppression_db": rows[0]["suppression_db"]}
    if labeler is not None and ref is not None:
        out["label"] = {"truth": target.label, "pred": labeler(cands[0]), "correct": None}
        out["label"]["correct"] = out["label"]["truth"] == out["label"]["pred"]
    return out


class Labeler:
    """AST coarse label of a 48 kHz waveform (separate.models.SoundLabeler on 16 kHz)."""

    def __init__(self, device="cuda"):
        from separate.models import SoundLabeler

        self.m = SoundLabeler(device)

    def __call__(self, wav48: np.ndarray) -> str:
        from scipy.signal import resample_poly

        p = self.m.probs([resample_poly(wav48, 1, 3).astype(np.float32)])[0]
        sc = self.m.coarse(p)
        return max(sc, key=sc.get)


def _save(out_dir: Path, tid: str, c: str, cands: list[np.ndarray]) -> list[str]:
    from ingest.media_io import write_wav

    paths = []
    for k, w in enumerate(cands):
        rel = f"cands/{tid}/{c}/cand{k}.wav"
        write_wav(out_dir / rel, np.nan_to_num(w), X.RATE)
        paths.append(rel)
    return paths


# --------------------------------------------------------------------------- stages
def load_bundle(span_predictor=True):
    from replay import sam_audio as sa

    return sa.load_sam_audio_diagnostic(span_predictor=span_predictor)


def stage_selftest(a, cases):
    """extract_candidates candidate k must equal upstream separate() fed the same noise."""
    from replay import sam_audio as sa

    bundle = load_bundle(span_predictor=False)
    case = next(c for c in cases if c.case_id == "overlap_stack_AB")
    t = case.targets[0]
    mask = case.masks[t.mask]
    cs = sa.extract_candidates(bundle, audio=case.audio, sample_rate=X.RATE, frames=case.frames, masks=mask, text="person speaking",
                               anchors="full_clip", n_candidates=2, seeds=[SEEDS[0], SEEDS[1]])
    batch, _, _ = sa.build_batch(bundle, audio=case.audio, sample_rate=X.RATE, frames=case.frames, masks=mask,
                                 text="person speaking", anchors="full_clip")
    dev = next(bundle.model.parameters()).device
    batch = batch.to(dev)
    res = {}
    for k in range(2):
        with torch.inference_mode():
            feats_shape = tuple(bundle.model._get_forward_args(batch, candidates=1)["audio_features"].shape)
        gen = torch.Generator(device=dev).manual_seed(SEEDS[k])
        noise = torch.randn(feats_shape, generator=gen, device=dev)
        with torch.inference_mode():
            up = bundle.model.separate(batch, noise=noise, reranking_candidates=1, predict_spans=False)
        up_wav = sa._first_waveform(up.target)
        d = np.abs(up_wav[: len(cs.targets[k])] - cs.targets[k][: len(up_wav)])
        res[f"cand{k}"] = {"max_abs_diff": float(d.max()), "si_sdr_vs_upstream_db": M.si_sdr(cs.targets[k][: len(up_wav)], up_wav)}
    res["noise_feature_shape"] = feats_shape
    # two different seeds must differ; same seed twice must match
    res["cand0_vs_cand1_si_sdr_db"] = M.si_sdr(cs.targets[0], cs.targets[1])
    cs2 = sa.extract_candidates(bundle, audio=case.audio, sample_rate=X.RATE, frames=case.frames, masks=mask, text="person speaking",
                                anchors="full_clip", n_candidates=1, seeds=[SEEDS[0]])
    res["same_seed_repeat_max_abs_diff"] = float(np.abs(cs2.targets[0] - cs.targets[0]).max())
    print(json.dumps(res, indent=1))
    (Path(a.out) / "selftest.json").parent.mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "selftest.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def stage_sam(a, cases):
    from replay import sam_audio as sa

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    resf = out / "results.jsonl"
    done = _read_done(resf)
    bundle = load_bundle(span_predictor=True)
    labeler = Labeler() if a.labels else None
    span_cache: dict = {}
    plan = [(c, t, cond) for c, t in trials(cases) for cond in conditions(c, t)
            if (c.case_id in OVERLAP or c.case_id in OTHERS)]
    if a.only:
        import re
        plan = [(c, t, cond) for c, t, cond in plan if re.search(a.only, f"{c.case_id}__{t.name}__{cid(cond)}")]
    if a.limit:
        plan = plan[: a.limit]
    print(f"{len(plan)} conditions, {len(done)} already done", flush=True)
    for i, (case, t, cond) in enumerate(plan):
        tid = f"{case.case_id}__{t.name}"
        key = f"{tid}__{cid(cond)}"
        if key in done:
            continue
        am = anchors_for(cond["anchor"], case, t)
        if am is None:
            continue
        anchors, pred = am
        span_key = (hash(case.audio.tobytes()), cond["text"])
        if cond["anchor"] == "none" and span_key in span_cache:        # predictor depends on audio+text only
            anchors, pred = span_cache[span_key], False
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        cs = sa.extract_candidates(bundle, audio=case.audio, sample_rate=X.RATE,
                                   frames=case.frames if cond["visual"] else None, masks=case.masks[t.mask],
                                   text=cond["text"], anchors=anchors, predict_spans=pred, n_candidates=cond["n"],
                                   seeds=SEEDS[: cond["n"]])
        wall = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        if cond["anchor"] == "none" and cs.spans is not None:
            span_cache[span_key] = [tuple(x) for x in cs.spans]
        if peak > 10.5 * 1024:
            raise SystemExit(f"peak VRAM {peak:.0f} MiB too close to the 12 GiB card limit; stopping")
        sc = score_candidates(case, t, cs.targets, labeler)
        paths = _save(out, tid, cid(cond), cs.targets)
        row = {"key": key, "trial": tid, "case": case.case_id, "target": t.name, "present": t.ref is not None,
               "layout": case.layout, "cond": cond, "anchors": cs.prompts["anchors"], "spans_predicted": cs.spans,
               "timings": cs.timings, "wall_s": round(wall, 2), "peak_vram_mib": round(peak, 1), "wavs": paths, **sc}
        with resf.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")
        s = sc["summary"]
        brief = (f"cand0 {s['cand0_sdri']:+.1f} best {s['best_sdri']:+.1f}@{s['best_idx']} mean {s['mean_sdri']:+.1f}"
                 if "cand0_sdri" in s else f"absent: cand0 {s['cand0_out_dbfs']:.1f} dBFS (mix {s['mix_dbfs']:.1f})")
        print(f"[{i + 1}/{len(plan)}] {key}  {brief}  wall {wall:.1f}s peak {peak:.0f} MiB", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stage_judge(a, cases):
    """Score every retained candidate set with the SAM-Audio judge (text-conditioned, no visual input) and
    report the SI-SDRi of its pick. ImageBind (the upstream visual ranker) is not cached: not run."""
    from sam_audio.model.config import JudgeRankerConfig
    from sam_audio.ranking.judge import JudgeRanker
    import soundfile as sf

    out = Path(a.out)
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    byid = {c.case_id: c for c in cases}
    ranker = JudgeRanker(JudgeRankerConfig("facebook/sam-audio-judge")).to("cuda").eval()
    jf = out / "judge.jsonl"
    done = _read_done(jf)
    for r in rows:
        if r["key"] in done or not r["present"]:
            continue
        case = byid[r["case"]]
        t = next(t for t in case.targets if t.name == r["target"])
        cands = [torch.from_numpy(sf.read(out / p, dtype="float32")[0]) for p in r["wavs"]]
        n = min(len(c) for c in cands)
        ex = torch.stack([c[:n] for c in cands]).cuda()
        mix = torch.from_numpy(case.audio[:n]).cuda()[None].expand(len(cands), -1)
        with torch.inference_mode():
            sc = ranker(input_audio=[mix], extracted_audio=[ex], descriptions=[r["cond"]["text"]], sample_rate=X.RATE)[0].float().cpu().numpy()
        sd = np.array([c["target"]["si_sdri"] for c in r["cands"]])
        pick = int(sc.argmax())
        rec = {"key": r["key"], "scores": sc.tolist(), "pick": pick, "pick_sdri": float(sd[pick]), "cand0_sdri": float(sd[0]),
               "best_sdri": float(sd.max()), "best_idx": int(sd.argmax()),
               "pick_selection_correct": (r["cands"][pick].get("selection") or {}).get("correct")}
        with jf.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{r['key']}: pick {pick} sdri {rec['pick_sdri']:+.1f} (cand0 {rec['cand0_sdri']:+.1f}, best {rec['best_sdri']:+.1f})", flush=True)
        torch.cuda.empty_cache()


def stage_moss(a, cases):
    """MossFormer2_SS_16K (2 outputs) on the mixtures: per-output SI-SDRi vs every reference, best permutation."""
    from scipy.signal import resample_poly
    from separate.models import VoiceSeparator
    import itertools

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sep = VoiceSeparator("cuda")
    SR = 16000
    res = []
    seen = set()
    for case in cases:
        if case.case_id not in ("overlap_stack_AB",) + tuple(o for o in OTHERS if o != "nonspeech_two"):
            continue
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        mix16 = resample_poly(case.audio, 1, 3).astype(np.float32)
        est = sep(mix16)
        wall = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        refs = {k: resample_poly(v, 1, 3).astype(np.float32) for k, v in case.refs.items() if not M.is_silent(v)}
        row = {"case": case.case_id, "wall_s": round(wall, 2), "peak_vram_mib": round(peak, 1), "refs": list(refs), "outputs": []}
        for j in range(est.shape[0]):
            row["outputs"].append({"out": j, **{f"sdri_vs_{k}": (M.si_sdri(est[j], r, mix16[: len(r)], SR) or {}).get("si_sdri") for k, r in refs.items()}})
        names = list(refs)
        if len(names) >= 2:
            best = None
            for perm in itertools.permutations(range(est.shape[0]), min(len(names), est.shape[0])):
                s = [M.si_sdri(est[j], refs[nm], mix16[: len(refs[nm])], SR)["si_sdri"] for j, nm in zip(perm, names)]
                if best is None or np.mean(s) > np.mean(best[1]):
                    best = (perm, s)
            row["best_permutation"] = {nm: {"output": int(j), "si_sdri": float(s)} for nm, j, s in zip(names, best[0], best[1])}
            row["mean_best_sdri"] = float(np.mean(best[1]))
        else:
            row["note"] = "fewer than two non-silent references: per-output scores only"
        for j in range(est.shape[0]):
            from ingest.media_io import write_wav
            write_wav(out / "moss" / case.case_id / f"out{j}.wav", est[j], SR)
        res.append(row)
        print(json.dumps(row, default=float), flush=True)
    (out / "moss.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


def _pairs(rows):
    """matched-noise mask-swap pairs: A-masked vs B-masked outputs, same ordering/prompt/anchor/seed."""
    import soundfile as sf

    idx = {r["key"]: r for r in rows}
    return idx


def stage_report(a, cases):
    import soundfile as sf

    out = Path(a.out)
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    judge = {}
    if (out / "judge.jsonl").exists():
        judge = {j["key"]: j for j in (json.loads(l) for l in (out / "judge.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    lines = []
    P = lines.append
    P("## Overlap trials (stack 480x540, 8 s, both speakers ~7 s overlap). SI-SDRi (dB) of the PROMPTED person vs the mixture")
    P("trial | prompt | anchors | vis | cand0 | best(idx) | mean+-std | n>=3dB | cand0 picks right person | frac right | judge pick")
    for r in rows:
        if r["case"] not in OVERLAP or not r["present"]:
            continue
        s, c = r["summary"], r["cond"]
        j = judge.get(r["key"])
        P(f"{r['trial']} | {c['prompt_key']} | {c['anchor']} | {'y' if c['visual'] else 'n'} | {s['cand0_sdri']:+.1f} | "
          f"{s['best_sdri']:+.1f}({s['best_idx']}) | {s['mean_sdri']:+.1f}+-{s['std_sdri']:.1f} | {s['n_ge_3db']}/{s['n']} | "
          f"{s['cand0_selection_correct']} | {s['frac_selection_correct']:.2f} | "
          + (f"{j['pick']} ({j['pick_sdri']:+.1f})" if j else "-"))
    # mask-swap sensitivity (matched noise): do the A-masked and B-masked outputs differ?
    P("\n## Mask-swap sensitivity (matched noise): SI-SDR between output(mask=A) and output(mask=B), per candidate mean; "
      "~+20 dB or more means the mask is ignored, ~0 or less means outputs differ")
    P("ordering | prompt | anchors | mean SI-SDR(A-out vs B-out) | mean SI-SDR(A-out vs A-ref) | mean SI-SDR(B-out vs B-ref)")
    by = {r["key"]: r for r in rows}
    for r in rows:
        if r["case"] not in OVERLAP or r["target"] != "A" or not r["cond"]["visual"]:
            continue
        kb = r["key"].replace("__A__", "__B__")
        if "gender" in r["key"] or kb not in by:
            continue
        rb = by[kb]
        sims = []
        for pa, pb in zip(r["wavs"], rb["wavs"]):
            xa, xb = sf.read(out / pa, dtype="float32")[0], sf.read(out / pb, dtype="float32")[0]
            sims.append(M.si_sdr(xa, xb))
        P(f"{r['case']} | {r['cond']['prompt_key']} | {r['cond']['anchor']} | {np.mean(sims):+.1f} | "
          f"{np.mean([c['target']['si_sdr'] for c in r['cands']]):+.1f} | {np.mean([c['target']['si_sdr'] for c in rb['cands']]):+.1f}")
    P("\n## Target-absent and non-overlap trials (cand0)")
    P("trial | anchors | present | cand0 SI-SDRi or out dBFS | mix dBFS | label ok")
    for r in rows:
        if r["case"] in OVERLAP:
            continue
        s = r["summary"]
        v = f"{s['cand0_sdri']:+.1f} dB (best {s['best_sdri']:+.1f})" if r["present"] else f"{s['cand0_out_dbfs']:.1f} dBFS (suppression {s['cand0_suppression_db']:+.1f} dB)"
        lab = (r.get("label") or {}).get("correct")
        P(f"{r['trial']} | {r['cond']['anchor']} | {r['present']} | {v} | {s.get('mix_dbfs', r['cands'][0]['mix_dbfs']):.1f} | {lab}")
    P("\n## Latency / VRAM")
    for r in rows[:1] + [x for x in rows if x['cond']['anchor'] == 'none'][:1]:
        P(f"{r['key']}: encode {r['timings']['encode_s']} s, per candidate {r['timings']['per_candidate_s']} s, wall {r['wall_s']} s, peak {r['peak_vram_mib']} MiB")
    walls = [r["wall_s"] / r["cond"]["n"] for r in rows]
    P(f"mean wall per candidate {np.mean(walls):.2f} s; max peak VRAM {max(r['peak_vram_mib'] for r in rows):.0f} MiB")
    if (out / "moss.json").exists():
        P("\n## MossFormer2_SS_16K baseline")
        for m in json.loads((out / "moss.json").read_text(encoding="utf-8")):
            P(json.dumps(m))
    txt = "\n".join(lines)
    (out / "report.txt").write_text(txt, encoding="utf-8")
    print(txt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["selftest", "sam", "judge", "moss", "report"])
    ap.add_argument("--out", default="sessions/av_matrix/run1")
    ap.add_argument("--sessions", default=str(X.SESSIONS))
    ap.add_argument("--limit", type=int, default=0, help="run only the first N conditions (smoke test)")
    ap.add_argument("--only", default="", help="regex on case__target__condition ids to run")
    ap.add_argument("--no-labels", dest="labels", action="store_false")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cases = X.build_cases(Path(a.sessions))
    {"selftest": stage_selftest, "sam": stage_sam, "judge": stage_judge, "moss": stage_moss, "report": stage_report}[a.stage](a, cases)


if __name__ == "__main__":
    main()
