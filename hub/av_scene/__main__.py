"""`python -m av_scene` (run from hub/): preflight | inspect | serve | fixture | run.

`run` is the Phase 4 chunked offline pipeline (recorded media in, long-clip bundle out). It is NOT real time and
makes no device claim. The other commands are Phase 1 and need no model weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "eval" / "av_scene"


def _load_eval(name: str):
    path = EVAL_DIR / f"{name}.py"
    if not path.is_file():
        raise SystemExit(f"{path} not found: the fixture commands need a repository checkout with eval/av_scene/")
    spec = importlib.util.spec_from_file_location(f"cocktail_eval_av_scene_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def cmd_preflight(args) -> int:
    import numpy as np
    import scipy

    from ingest.media_io import locate_ffmpeg
    from render.envelope import k_weighting_sos

    ff = locate_ffmpeg(args.ffmpeg)
    k_weighting_sos(16000), k_weighting_sos(48000)
    report = {
        "evidence_tier": "mocked/synthetic",
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "capabilities": {
            "cpu_math": True, "wav_writer": True, "png_writer": True, "k_weighting": True,
            "ffmpeg": ff.to_dict(),
            "mp4_output": ff.can_mux_mp4,
            "sam3_image": "not checked (Phase 2)", "sam3_video": "not checked (Phase 2)",
            "sam_audio": "not checked (Phase 2)",
            "diarization": "not checked (use --av; reports not_attempted/ok/fail, never a token)",
        },
        "notes": [] if ff.ffmpeg else ["ffmpeg not found: bundles use PNG frame sequences; set COCKTAIL_FFMPEG "
                                       "or --ffmpeg to enable annotated.mp4/source_video.mp4"],
    }
    if args.av:
        from av_scene import preflight

        report["evidence_tier"] = "hardware-tested (this laptop) when checks pass"
        report["av"] = preflight.run_av_checks(args.media)
    print(json.dumps(report, indent=2))
    return 0 if report.get("av", {"ok": True})["ok"] else 1


def cmd_inspect(args) -> int:
    p = Path(args.path)
    mp = p / "manifest.json" if p.is_dir() else p
    man = json.loads(mp.read_text(encoding="utf-8"))
    root = mp.parent
    print(f"run {man['run_id']}   evidence_tier = {man['evidence_tier']}")
    print(man["evidence_note"])
    print(f"code {man.get('code_revision')}  config {man['config_hash'][:12]}  schema v{man['schema_version']}")
    print("\nsources / associations")
    assoc = {a["source_id"]: a for a in man["associations"]}
    for s in man["sources"]:
        a = assoc[s["source_id"]]
        sc = "n/a" if a["score"] is None else f"{a['score']:.2f}"
        mg = "n/a" if a["runner_up_margin"] is None else f"{a['runner_up_margin']:.2f}"
        print(f"  {s['source_id']:<18} {s['label']:<18} {a['status']:<10} track={a['track_id'] or '-':<7} score={sc} margin={mg}"
              f" aliasing={a['spatial']['aliasing_multimodal_flag'] if a['spatial'] else '-'}")
        for c in a["components"]:
            v = "unavailable" if c["value"] is None else f"{c['value']:.2f}"
            print(f"      {c['name']:<18} {v:<11} q={c['quality']:.2f} w={c['weight']:.1f}  {c['note']}")
    print("\nstems:", ", ".join(s["raw_path"] for s in man["stems"]))
    print("remainders:", "; ".join(f"{r['kind']}({r['use']})" for r in man["remainders"]))
    print("capabilities:", json.dumps(man["capabilities"]))
    print("stages:", ", ".join(f"{s['name']} {s['seconds']}s" for s in man["stages"]))
    missing = [k for k, v in man["assets"].items() if not (root / v).exists() and k != "source_video"]
    print("assets:", f"{len(man['assets'])} listed, {len(missing)} missing" + (f": {missing}" if missing else ""))
    if man.get("long_run"):
        _inspect_long(man, root)
    mpth = root / (man.get("metrics_path") or "metrics.json")
    if mpth.exists():
        m = json.loads(mpth.read_text())
        al, at, rm = m["alignment"], m["attribution"], m["remainder"]
        print(f"\nmetrics ({m['evidence_tier']}): sync all_ok={al['all_ok']} max|skew|={al['max_abs_skew_ms']} ms; "
              f"precision={at['precision']} coverage={at['coverage']}; abstention={at['abstention_correct']}; "
              f"silent bystanders={at['silent_bystanders']}; false unexplained events={rm['false_unexplained_events_full_inventory']}")
    return 0


def _inspect_long(man: dict, root: Path) -> None:
    lr = man["long_run"]
    ch = lr["chunking"]
    print(f"\nlong run: {ch['n_chunks']} chunks of {ch['window_s']} s / overlap {ch['overlap_s']} s, "
          f"{ch['n_samples'] / ch['sample_rate']:.1f} s of audio, crossfade {ch['crossfade_ms']} ms")
    print("switches:", json.dumps(lr["switches"]))
    print("diarization:", json.dumps(lr["diarization"]))
    print("resume:", json.dumps({k: (v if not isinstance(v, list) else f"{len(v)} units") for k, v in lr["resume"].items()}))
    idx = json.loads((root / lr["timeline_index"]).read_text(encoding="utf-8"))
    from collections import Counter
    rows: dict = {}
    for pg in idx["pages"]:
        for seg in json.loads((root / pg["path"]).read_text(encoding="utf-8"))["segments"]:
            c = rows.setdefault(seg["source_id"], Counter())
            c[(seg["association_status"], seg["visibility"], seg["stem_availability"])] += 1
    print("timeline (association / visibility / stem availability kept separate):")
    for sid, c in sorted(rows.items()):
        print(f"  {sid:<24}", "; ".join(f"{a}/{v}/{st} x{n}" for (a, v, st), n in c.most_common(4)))
        refusals = Counter()
        for pg in idx["pages"]:
            for seg in json.loads((root / pg["path"]).read_text(encoding="utf-8"))["segments"]:
                if seg["source_id"] == sid and seg["refusal_reason"]:
                    refusals[seg["refusal_reason"][:60]] += 1
        for r, n in refusals.most_common(2):
            print(f"      refused x{n}: {r}")


def cmd_run(args) -> int:
    from ingest.media_io import locate_ffmpeg

    from . import preflight
    from .cache import hash_file
    from .config import load_config
    from .phase1 import _git_rev
    from .real_deps import real_providers
    from .run import run_pipeline

    cfg = load_config(args.config)
    sw = cfg.switches
    for name in ("activity_gate", "offscreen_upgrade", "peel"):
        if getattr(args, name):
            setattr(sw, name, True)
    if args.no_tse:
        sw.use_tse = False
    if args.no_events:
        sw.events = False
    if args.no_residual:
        sw.residual = False
    if args.no_identity:
        sw.identity = False
    if args.diarization:
        sw.diarization = args.diarization
    if args.window_s is not None:
        cfg.chunking.window_s = args.window_s
    if args.overlap_s is not None:
        cfg.chunking.overlap_s = args.overlap_s
    cfg.chunking.validate()
    media = Path(args.media)
    if not media.is_file():
        raise SystemExit(f"{media} not found")
    from replay.chunked import TorchcodecSource
    ff = locate_ffmpeg(args.ffmpeg)
    source = TorchcodecSource(media, max_side=args.max_side, ffmpeg_info=ff)
    media_id = hash_file(media)
    out = Path(args.out)
    hw: dict = {}
    try:
        import torch
        if torch.cuda.is_available():
            hw = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    except Exception:
        pass
    prov = real_providers(cfg, device=args.device, run_tag=media_id[:8])
    res = run_pipeline(source, prov, cfg, out, media_id=media_id, run_id=args.run_id or out.name,
                       cache_dir=args.cache, evidence_tier="replayed",
                       evidence_note="Recorded clip, chunked offline run. Models are hardware-run only where the run "
                                     "report says so; not real time; no device validation.",
                       code_revision=_git_rev(), hardware=hw)
    r = res.report
    print(f"bundle ready: {out.resolve()}")
    print(f"  chunks computed this run: {len(r['resume']['computed'])}, restored from cache: {r['resume']['restored']}")
    print(f"  model calls: {r['resume']['model_calls']}")
    print(f"  peak RSS {r['memory']['peak_rss_mib']} MiB, peak GPU {r['memory']['peak_gpu_mib']}")
    print(f"inspect:  uv run python -m av_scene inspect {out / 'manifest.json'}")
    print(f"listen:   uv run python -m av_scene serve {out}")
    return 0


def cmd_serve(args) -> int:
    from render.viewer import make_server

    root = Path(args.bundle)
    if not (root / "viewer.html").is_file():
        raise SystemExit(f"{root} is not a bundle (viewer.html missing)")
    srv = make_server(root, args.host, args.port)
    print(f"serving {root.resolve()} at http://{args.host}:{srv.server_address[1]}/viewer.html  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_fixture(args) -> int:
    from . import phase1

    gen, scorer = _load_eval("generate_fixture"), _load_eval("score")
    out = Path(args.out)
    fx = Path(args.fixture_dir) if args.fixture_dir else out.parent / "fixture"
    print(f"generating fixture (seed {args.seed}) -> {fx}")
    gen.generate(fx, args.seed)
    print(f"running Phase 1 pipeline -> {out}")
    phase1.run(fx, out, ffmpeg_path=args.ffmpeg, run_id=out.parent.name or "phase1_fixture")
    metrics = scorer.score(out, fx)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"bundle ready: {out.resolve()}   [evidence_tier = {metrics['evidence_tier']}]")
    print(f"inspect:  uv run python -m av_scene inspect {out / 'manifest.json'}")
    print(f"listen:   uv run python -m av_scene serve {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="av_scene", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight", help="report Phase 1 capabilities (no models)")
    p.add_argument("--ffmpeg", default=None, help="explicit ffmpeg path (else COCKTAIL_FFMPEG / PATH)")
    p.add_argument("--av", action="store_true", help="also check Phase 2: CUDA, torchcodec decode, HF gated access")
    p.add_argument("--media", default=None, help="media file for the --av decode check (else a synthetic clip)")
    p.set_defaults(fn=cmd_preflight)
    p = sub.add_parser("inspect", help="summarise a manifest.json or bundle folder")
    p.add_argument("path")
    p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("serve", help="serve a bundle on localhost")
    p.add_argument("bundle")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("fixture", help="generate the synthetic fixture, run the pipeline, write the bundle and metrics")
    p.add_argument("--out", default="sessions/phase1_fixture/av")
    p.add_argument("--fixture-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ffmpeg", default=None)
    p.set_defaults(fn=cmd_fixture)
    p = sub.add_parser("run", help="Phase 4 chunked offline pipeline on a recorded clip (GPU; not real time)")
    p.add_argument("media")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=None, help="stage cache dir (default OUT/_cache); reuse it to resume")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-id", default=None)
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--window-s", type=float, default=None)
    p.add_argument("--overlap-s", type=float, default=None)
    p.add_argument("--activity-gate", dest="activity_gate", action="store_true", help="Phase 3 slice 1")
    p.add_argument("--offscreen", dest="offscreen_upgrade", action="store_true", help="Phase 3 slice 2")
    p.add_argument("--peel", action="store_true", help="Phase 3 slice 3b")
    p.add_argument("--no-tse", action="store_true")
    p.add_argument("--no-events", action="store_true")
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--no-identity", action="store_true", help="baseline: slot order not persistent across chunks")
    p.add_argument("--diarization", choices=["off", "auto", "on"], default=None)
    p.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
