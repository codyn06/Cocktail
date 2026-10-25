"""End-to-end Phase 1 fixture -> pipeline -> bundle -> score (evidence tier: mocked/synthetic).

Fixture results validate the deterministic output contract only; they are not learned perception,
room validation or device validation.
"""

import http.client
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from av_scene import __main__ as cli
from av_scene import phase1
from ingest.media_io import locate_ffmpeg, read_wav
from render.envelope import compute_envelope
from render.viewer import serve_in_thread

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "eval" / "av_scene"
ASSETS = Path(__file__).resolve().parents[2] / "render" / "assets"


def load_eval(name):
    spec = importlib.util.spec_from_file_location(f"t_eval_{name}", EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixture")
    gen, scorer = load_eval("generate_fixture"), load_eval("score")
    fx = gen.generate(root / "fixture", seed=0)
    out = root / "av"
    manifest = phase1.run(fx, out, ffmpeg_path=None, run_id="test_fixture")
    metrics = scorer.score(out, fx)
    (out / "metrics.json").write_text(json.dumps(metrics))
    return dict(fx=fx, out=out, manifest=manifest, metrics=metrics, truth=json.loads((fx / "fixture.json").read_text()),
                gen=gen, scorer=scorer)


def test_fixture_is_deterministic():
    gen = load_eval("generate_fixture")
    a = gen.synth_audio(np.random.default_rng(0))
    b = gen.synth_audio(np.random.default_rng(0))
    assert np.array_equal(a[0], b[0]) and a[2] == b[2]


def test_bundle_has_every_listed_asset_and_evidence_labels(scene):
    out, man = scene["out"], json.loads((scene["out"] / "manifest.json").read_text())
    assert man["evidence_tier"] == "mocked/synthetic" and scene["metrics"]["evidence_tier"] == "mocked/synthetic"
    for key, rel in man["assets"].items():
        if key in ("source_video", "annotated_video"):
            continue
        assert (out / rel).is_file(), key
    for rel in ("viewer.html", "assets/viewer.js", "assets/viewer.css", "frames.json", "diagnostics/residual_waveform.wav",
                "residual.wav", "residuals/unexplained_power.npz", "metrics.json", "mixture.wav"):
        assert (out / rel).is_file(), rel
    assert len(list((out / "stems").glob("*.wav"))) == 4 and len(list((out / "masks").glob("*.jsonl"))) == 4
    for f in (out / "activity").glob("*.json"):
        assert json.loads(f.read_text())["evidence_tier"] == "mocked/synthetic"
    assert len((out / "manifest.json").read_text()) < 200_000                   # no dense arrays in the manifest
    assert man["capabilities"]["learned_models"] == "none"
    assert man["capabilities"]["mp4"]["status"] in ("skipped", "written", "failed")
    assert man["model_revisions"]["separator"].startswith("oracle")
    kinds = {r["kind"]: r["use"] for r in man["remainders"]}
    assert kinds == {"raw_waveform_diagnostic": "diagnostic", "phase_projected_stft": "playback_proxy",
                     "unexplained_power": "discovery"}


def test_associations_unmatched_stays_unmatched_and_bystander_never_glows(scene):
    a = {x["source_id"]: x for x in scene["manifest"].model_dump(mode="json")["associations"]}
    assert (a["spk_alice"]["status"], a["spk_alice"]["track_id"]) == ("matched", "alice")
    assert (a["spk_bob"]["status"], a["spk_bob"]["track_id"]) == ("matched", "bob")
    assert (a["cat_chirp"]["status"], a["cat_chirp"]["track_id"]) == ("matched", "cat1")
    assert (a["knock_offscreen"]["status"], a["knock_offscreen"]["track_id"]) == ("unmatched", None)
    assert a["knock_offscreen"]["visibility"] == "unknown"                        # no match != proven off-screen
    m = scene["metrics"]["attribution"]
    assert m["precision"] == 1.0 and m["coverage"] == 1.0 and m["wrong_attributions"] == 0
    assert m["silent_bystanders"]["carol"]["highlight_pct_of_visible_time"] == 0.0     # under known truth
    assert m["wrong_object_highlight_s"] == 0.0 and m["abstention_correct"] == {"knock_offscreen": True}
    state = json.loads((scene["out"] / "activity" / "track_carol.json").read_text())["state"]
    assert set(state) == {"idle"}
    assert not any(json.loads((scene["out"] / "activity" / "knock_offscreen.json").read_text())["active"])


def test_evidence_unavailable_off_axis_and_chirp_aliasing_are_recorded(scene):
    a = {x["source_id"]: x for x in scene["manifest"].model_dump(mode="json")["associations"]}
    comp = {c["name"]: c for c in a["spk_alice"]["components"]}
    assert comp["onset_synchrony"]["quality"] == 0.5                              # only the forward-facing turn counted
    assert "1 onset(s)" in comp["onset_synchrony"]["note"]
    el = a["spk_alice"]["learning_eligibility"]
    assert any(not e["eligible"] and e["reason"].startswith("off_axis") for e in el)
    assert any(e["eligible"] for e in el)
    assert a["cat_chirp"]["spatial"]["aliasing_multimodal_flag"] is True and len(a["cat_chirp"]["spatial"]["peaks_s"]) >= 2
    assert a["spk_bob"]["spatial"]["aliasing_multimodal_flag"] is False
    kc = {c["name"]: c for c in a["knock_offscreen"]["components"]}
    assert kc["spatial_delay"]["value"] is not None and kc["spatial_delay"]["value"] < 0.3


def test_overlapping_speech_anchors_exclude_the_overlap(scene):
    sp = {s["speaker_id"]: s for s in scene["manifest"].model_dump(mode="json")["speakers"]}
    overlap = (4_200_000, 5_000_000)
    for sid in ("spk_alice", "spk_bob"):
        assert sp[sid]["anchors"]["empty_reason"] is None
        for s, e in sp[sid]["anchors"]["positive"]:
            assert e <= overlap[0] or s >= overlap[1], (sid, s, e)
        assert sp[sid]["anchors"]["negative"], sid


def test_signed_sync_gates_hold_with_measured_quantization(scene):
    al = scene["metrics"]["alignment"]
    assert al["all_ok"] and len(al["turns"]) == 6
    pts = scene["truth"]["frame_pts_us"]
    assert len(set(np.diff(pts))) > 10 and max(np.diff(pts)) > 70_000          # VFR with a dropped frame
    for t in al["turns"]:
        q = t["q_ms"]
        assert 14.0 < q < 45.0                                                  # per-turn local half frame interval
        assert -15 - q <= t["delta_audio_ms"] <= 45 + q
        assert -15 - q <= t["delta_mask_ms"] <= 45 + q
        assert abs(t["delta_skew_ms"]) <= 20 + q
        assert t["delta_skew_ms"] == pytest.approx(t["delta_mask_ms"] - t["delta_audio_ms"], abs=0.01)
        assert t["delta_audio_ms"] > 0                                          # natural ToF is kept, not removed
    assert max(abs(t["delta_skew_ms"]) for t in al["turns"]) < 40.0


def test_stem_leakage_remainder_and_omitted_source_metrics(scene):
    m = scene["metrics"]
    assert all(v["si_sdr_db"] > 6.0 for v in m["leakage"]["per_stem"].values())
    assert m["leakage"]["max_duplicate_similarity"] < 0.3
    assert m["remainder"]["false_unexplained_events_full_inventory"] == 0
    assert all(v["recall"] == 1.0 for v in m["remainder"]["omitted_source_recall"].values())
    assert m["remainder"]["projection_error_db"] < -20.0


def test_exported_stem_bytes_are_untouched_by_the_envelope_branch(scene):
    stem = scene["out"] / "stems" / "source_spk_alice.wav"
    before = stem.read_bytes()
    x, rate = read_wav(stem)
    pts = scene["truth"]["frame_pts_us"]
    compute_envelope(x[:, 0].astype(np.float64), rate, pts)
    assert stem.read_bytes() == before


# ------------------------------------------------------------------ viewer and server
def test_viewer_assets_have_no_network_dependencies():
    for name in ("viewer.html", "viewer.js", "viewer.css"):
        text = (ASSETS / name).read_text(encoding="utf-8")
        for bad in ("http://", "https://", "//cdn", "unpkg", "cdnjs", "googleapis", "@import url"):
            assert bad not in text, (name, bad)
    html = (ASSETS / "viewer.html").read_text()
    assert "AudioContext" in (ASSETS / "viewer.js").read_text() and 'id="stage"' in html
    for needle in ("Solo", "Mute", "residual", "mixture", "seek", "candidates"):
        assert needle.lower() in (html + (ASSETS / "viewer.js").read_text()).lower(), needle


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_viewer_js_parses_and_decodes_the_golden_rle_vectors(tmp_path):
    golden = json.loads((Path(__file__).parent / "golden_rle.json").read_text())
    script = tmp_path / "check.js"
    script.write_text(
        "const fs=require('fs'),vm=require('vm');\n"
        f"const code=fs.readFileSync({json.dumps(str(ASSETS / 'viewer.js'))},'utf8');\n"
        "vm.runInThisContext(code);\n"
        f"const g=JSON.parse(fs.readFileSync({json.dumps(str(Path(__file__).parent / 'golden_rle.json'))},'utf8'));\n"
        "for(const c of g.cases){const out=globalThis.__decodeRle({h:c.h,w:c.w,counts:c.counts,foreground:true});\n"
        "const want=c.rows.join('').split('').map(Number);\n"
        "if(JSON.stringify(Array.from(out))!==JSON.stringify(want)){console.error('MISMATCH',c.name);process.exit(1);}}\n"
        "console.log('ok',g.cases.length);\n")
    r = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and f"ok {len(golden['cases'])}" in r.stdout, r.stderr


def test_local_server_serves_the_bundle_with_http_range(scene):
    srv, _ = serve_in_thread(scene["out"])
    port = srv.server_address[1]
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", "/viewer.html")
        r = c.getresponse()
        assert r.status == 200 and b"Cocktail AV inspector" in r.read()
        size = (scene["out"] / "stems" / "source_spk_bob.wav").stat().st_size
        c.request("GET", "/stems/source_spk_bob.wav", headers={"Range": "bytes=100-199"})
        r = c.getresponse()
        body = r.read()
        assert r.status == 206 and len(body) == 100 and r.getheader("Content-Range") == f"bytes 100-199/{size}"
        assert body == (scene["out"] / "stems" / "source_spk_bob.wav").read_bytes()[100:200]
        c.request("GET", "/stems/source_spk_bob.wav", headers={"Range": f"bytes={size + 5}-"})
        r = c.getresponse()
        r.read()
        assert r.status == 416
        c.request("GET", "/stems/source_spk_bob.wav", headers={"Range": "bytes=-50"})
        r = c.getresponse()
        assert r.status == 206 and len(r.read()) == 50
        c.request("GET", "/nope.txt")
        r = c.getresponse()
        r.read()
        assert r.status == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_inspect_and_preflight(scene, capsys):
    assert cli.main(["inspect", str(scene["out"] / "manifest.json")]) == 0
    text = capsys.readouterr().out
    assert "mocked/synthetic" in text and "knock_offscreen" in text and "unmatched" in text and "0 missing" in text
    assert cli.main(["preflight"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["evidence_tier"] == "mocked/synthetic" and "ffmpeg" in rep["capabilities"]


@pytest.mark.skipif(not locate_ffmpeg().can_mux_mp4, reason="no ffmpeg with an MP4 encoder discovered (set COCKTAIL_FFMPEG)")
def test_mp4_output_when_ffmpeg_is_available(scene, tmp_path):
    out = tmp_path / "av"
    man = phase1.run(scene["fx"], out, run_id="mp4")
    assert man.capabilities["mp4"]["status"] == "written"
    assert (out / "source_video.mp4").stat().st_size > 1000 and (out / "annotated.mp4").stat().st_size > 1000
