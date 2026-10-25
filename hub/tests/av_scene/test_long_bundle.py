"""LongBundleWriter: paged, incremental, bounded-memory bundle (evidence tier: mocked/synthetic)."""

import hashlib
import http.client
import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from av_scene.schemas import (AudioIndex, LongRunInfo, RunManifest, TimelineIndex, TimelinePageDoc, TimelineSegment)
from ingest.media_io import read_wav
from render.long_bundle import LongBundleWriter, StreamingWavWriter, safe_name
from render.viewer import serve_in_thread
from vision.mask_store import MaskRecord, encode_mask

RATE = 16000


def _manifest(**lr):
    return RunManifest(run_id="long_t", config_hash="x" * 8,
                       long_run=LongRunInfo(chunking={"sample_rate": RATE, **lr}))


def _seg(sid, a, b, **kw):
    return TimelineSegment(source_id=sid, start_sample=a, end_sample=b, start_us=a * 1_000_000 // RATE,
                           end_us=b * 1_000_000 // RATE, **kw)


def test_safe_name_is_filesystem_safe_and_collision_free():
    a, b = safe_name("tse:track_1"), safe_name("tse_track_1")
    assert a != b and ":" not in a and "/" not in safe_name("../../x") and not safe_name(".hidden").startswith(".")
    assert safe_name("src_3") == "src_3"


def test_index_validates_checksums_and_streaming_export(tmp_path):
    w = LongBundleWriter(tmp_path / "b", _manifest(), page_span_s=2.0)
    rng = np.random.default_rng(0)
    seg1, seg2 = rng.standard_normal(RATE).astype(np.float32) * .1, rng.standard_normal(RATE // 2).astype(np.float32) * .1
    r1 = w.add_audio_segment("tse:track_1", "tse", "diagnostic", RATE, 8000.0, 0, seg1)
    # gap of 0.5 s, then a second segment, then a 1 s tail of silence declared via the logical length
    r2 = w.add_audio_segment("tse:track_1", "tse", "diagnostic", RATE, 8000.0, RATE + RATE // 2, seg2)
    w.set_source_length("tse:track_1", 4 * RATE)
    w.add_audio_segment("src_1", "speech", "default", RATE, 8000.0, 0, seg1)
    for sid, a, b, kw in [("src_1", 0, RATE, dict(association_status="matched", visibility="visible", stem_availability="ok", track_id="t1")),
                          ("tse:track_1", 0, RATE, dict(stem_availability="degraded", refusal_reason="overlap")),
                          ("tse:track_1", RATE + RATE // 2, 2 * RATE, dict(stem_availability="unavailable", refusal_reason="no face")),
                          ("src_1", 5 * RATE, 6 * RATE, dict(stem_availability="unavailable", refusal_reason="gap"))]:
        w.add_timeline_segment(_seg(sid, a, b, **kw))
    out = tmp_path / "b"
    # flat export before finalize: zeros for gaps, equals the naive concatenation
    n = w.export_flat_wav("tse:track_1", tmp_path / "flat.wav")
    flat, rate = read_wav(tmp_path / "flat.wav")
    want = np.zeros(4 * RATE, np.float32)
    want[:RATE] = seg1
    want[RATE + RATE // 2:RATE + RATE // 2 + RATE // 2] = seg2
    assert n == 4 * RATE and rate == RATE and np.array_equal(flat[:, 0], want)

    man = w.finalize()
    for rel in ("audio/index.json", "timeline/index.json", "manifest.json", "viewer.html", "assets/viewer.js",
                "assets/viewer.css", "frames/index.json"):
        assert (out / rel).is_file(), rel
    ai = AudioIndex.model_validate_json((out / "audio/index.json").read_text())
    ti = TimelineIndex.model_validate_json((out / "timeline/index.json").read_text())
    assert set(ai.sources) == {"tse:track_1", "src_1"} and ai.sources["tse:track_1"].n_samples == 4 * RATE
    assert ai.sources["tse:track_1"].nyquist_hz == 8000.0 and ai.sources["tse:track_1"].role == "diagnostic"
    for ref in ai.sources["tse:track_1"].segments:
        data = (out / ref.path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == ref.sha256 and ":" not in ref.path
    assert [s.start_sample for s in ai.sources["tse:track_1"].segments] == [0, RATE + RATE // 2]
    assert ti.sample_rate == RATE and {s.source_id for s in ti.sources} == {"src_1", "tse:track_1"}
    # four independent status fields survive paging, with the refusal reason
    segs = [s for p in ti.pages for s in TimelinePageDoc.model_validate_json((out / p.path).read_text()).segments]
    assert len(segs) == 4 and len(ti.pages) >= 2
    got = {(s.source_id, s.start_sample): s for s in segs}
    s = got[("tse:track_1", RATE + RATE // 2)]
    assert (s.stem_availability, s.refusal_reason, s.association_status, s.visibility) == ("unavailable", "no face", "unmatched", "unknown")
    m = json.loads((out / "manifest.json").read_text())
    assert m["long_run"]["audio_index"] == "audio/index.json" and m["assets"]["frames_index"] == "frames/index.json"
    assert man.long_run.timeline_index == "timeline/index.json"


def test_overlapping_audio_segment_is_rejected(tmp_path):
    w = LongBundleWriter(tmp_path / "b", _manifest())
    w.add_audio_segment("s", "speech", "default", RATE, None, 0, np.zeros(100, np.float32))
    with pytest.raises(ValueError, match="overlaps"):
        w.add_audio_segment("s", "speech", "default", RATE, None, 50, np.zeros(100, np.float32))
    w.add_audio_segment("s", "speech", "default", RATE, None, 100, np.zeros(100, np.float32))   # adjacent is fine
    with pytest.raises(ValueError, match="unknown source"):
        w.add_timeline_segment(_seg("nope", 0, 10))


def test_streaming_wav_writer_header_and_atomic_replace(tmp_path):
    p = tmp_path / "x.wav"
    with StreamingWavWriter(p, 8000) as sw:
        sw.write(np.array([0.5, -0.5], np.float32))
        sw.write_zeros(3)
        assert not p.exists()                      # still writing to .tmp
    x, r = read_wav(p)
    assert r == 8000 and x[:, 0].tolist() == [0.5, -0.5, 0, 0, 0] and not (tmp_path / "x.wav.tmp").exists()


def test_frames_and_masks_are_paged_and_keep_frame_transform(tmp_path):
    man = _manifest()
    w = LongBundleWriter(tmp_path / "b", man, page_span_s=1.0)
    frames = [np.full((12, 16, 3), i, np.uint8) for i in range(6)]
    pts = [0, 400_000, 900_000, 1_100_000, 2_500_000, 3_200_000]
    w.add_frames(frames[:3], pts[:3])
    w.add_frames(frames[3:], pts[3:])
    rle = encode_mask(np.ones((6, 8), bool))
    recs = [MaskRecord(pts_us=p, frame_idx=i, rle=rle) for i, p in enumerate(pts)]
    w.add_masks("t1", recs[:3])
    w.add_masks("t1", recs[3:])
    w.finalize()
    out = tmp_path / "b"
    idx = json.loads((out / "frames/index.json").read_text())
    ft = man.capabilities["frame_transform"]
    assert idx["transform"] == ft == {"version": 1, "frame_w": 16, "frame_h": 12, "mask_w": 8, "mask_h": 6}
    assert idx["n_frames"] == 6 and [p["n"] for p in idx["pages"]] == [3, 1, 1, 1]
    page0 = json.loads((out / idx["pages"][0]["path"]).read_text())
    assert page0["first_index"] == 0 and page0["pts_us"] == pts[:3] and (out / page0["files"][0]).is_file()
    mp = idx["mask_pages"]["t1"]
    assert sum(p["n"] for p in mp) == 6
    lines = (out / mp[0]["path"]).read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["pts_us"] == 0
    with pytest.raises(RuntimeError):
        w.add_frames([frames[0]], [0])


def test_peak_memory_is_independent_of_run_length(tmp_path):
    n_seg, seg_len = 120, RATE * 5                         # 120 x 5 s x 16 kHz float32 = 38.4 MB written
    rng = np.random.default_rng(1)
    w = LongBundleWriter(tmp_path / "b", _manifest(), page_span_s=30.0)
    tracemalloc.start()
    for i in range(n_seg):
        x = (rng.standard_normal(seg_len) * 0.05).astype(np.float32)
        w.add_audio_segment("src_1", "speech", "default", RATE, 8000.0, i * seg_len, x)
        w.add_timeline_segment(_seg("src_1", i * seg_len, (i + 1) * seg_len, stem_availability="ok"))
        del x
    w.export_flat_wav("src_1", tmp_path / "flat.wav")
    w.finalize()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total = n_seg * seg_len * 4
    assert (tmp_path / "flat.wav").stat().st_size > total
    assert peak < total / 8, (peak, total)


def test_range_server_serves_long_bundle_files_and_viewer(tmp_path):
    out = tmp_path / "b"
    w = LongBundleWriter(out, _manifest())
    w.add_audio_segment("src_1", "speech", "default", RATE, None, 0, np.zeros(RATE, np.float32))
    w.finalize()
    srv, _ = serve_in_thread(out)
    try:
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        ref = AudioIndex.model_validate_json((out / "audio/index.json").read_text()).sources["src_1"].segments[0]
        size = (out / ref.path).stat().st_size
        c.request("GET", "/" + ref.path, headers={"Range": "bytes=10-19"})
        r = c.getresponse()
        assert r.status == 206 and r.getheader("Content-Range") == f"bytes 10-19/{size}" and len(r.read()) == 10
        c.request("GET", "/assets/viewer.js")
        r = c.getresponse()
        assert r.status == 200 and b"__viewerHelpers" in r.read()
    finally:
        srv.shutdown()
        srv.server_close()


def test_server_never_serves_run_internals_inside_the_output_folder(tmp_path):
    from render.viewer import serve_in_thread
    (tmp_path / "_cache").mkdir()
    (tmp_path / "_cache" / "secret.bin").write_bytes(b"x")
    (tmp_path / "ok.txt").write_text("fine")
    srv, _ = serve_in_thread(tmp_path)
    try:
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        for path, want in (("/ok.txt", 200), ("/_cache/secret.bin", 404), ("/%5Fcache/secret.bin", 404)):
            c.request("GET", path)
            r = c.getresponse()
            r.read()
            assert r.status == want, path
    finally:
        srv.shutdown()
