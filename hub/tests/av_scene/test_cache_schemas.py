"""Versioned contracts, atomic stage cache, TOML config and the signed sync gates."""

import json

import pytest
from pydantic import ValidationError

from av_scene.cache import MARKER, StageCache, hash_bytes, hash_config, hash_file
from av_scene.config import load_array_calibration, load_config
from av_scene.schemas import (SCHEMA_VERSION, Association, ComponentScore, MaskletTrack, MediaIndex, RemainderResult,
                              RunManifest, SceneSource, SpatialEvidence, StemResult)
from av_scene.timing import check_sync


def test_schemas_round_trip_and_stay_json_only():
    m = RunManifest(run_id="r", config_hash="abc", media=MediaIndex(run_id="r"),
                    associations=[Association(association_id="a1", source_id="s1", track_id=None, valid_start_us=0,
                                              valid_end_us=10, status="unmatched", visibility="unknown",
                                              components=[ComponentScore(name="asd", value=None, note="off-axis")],
                                              spatial=SpatialEvidence(aliasing_multimodal_flag=True,
                                                                      aliasing_pairs=["0-2"], parallax_sigma_rad=0.1))],
                    stems=[StemResult(source_id="s1", model_revision="oracle", sample_rate=48000, n_samples=10,
                                      raw_path="stems/s1.wav")],
                    remainders=[RemainderResult(remainder_id="r1", kind="phase_projected_stft", source_stem_ids=["s1"],
                                                sample_rate=48000, n_samples=10, use="playback_proxy")])
    back = RunManifest.from_json(m.to_json())
    assert back == m and back.evidence_tier == "mocked/synthetic" and back.schema_version == SCHEMA_VERSION
    assert back.associations[0].components[0].value is None            # unavailable survives, not coerced to 0
    assert len(m.to_json()) < 5000                                      # no embedded arrays


def test_schemas_reject_unknown_fields_and_bad_enums():
    with pytest.raises(ValidationError):
        SceneSource(source_id="s", surprise=1)
    with pytest.raises(ValidationError):
        Association(association_id="a", source_id="s", track_id=None, valid_start_us=0, valid_end_us=1,
                    status="maybe", visibility="unknown")
    with pytest.raises(ValidationError):
        MaskletTrack(track_id="t", concept="c", mask_file="m", image_height=1, image_width=1, n_frames=1,
                     dense_mask=[[1]])
    with pytest.raises(ValidationError):
        MediaIndex(run_id="r", evidence_tier="trust-me")


def key(c, **kw):
    base = dict(inputs={"video": "h1"}, config={"x": 1}, transform=None, upstream={})
    base.update(kw)
    return c.key("ingest", **base)


def test_cache_keys_track_content_config_transform_and_upstream(tmp_path):
    c = StageCache(tmp_path)
    k = key(c)
    assert k == key(c) and len(k) == 32
    assert key(c, inputs={"video": "h2"}) != k and key(c, config={"x": 2}) != k
    assert key(c, transform={"resample": 16000}) != k
    up1 = c.key("evidence", upstream={"ingest": k}, config=1)
    up2 = c.key("evidence", upstream={"ingest": key(c, config={"x": 2})}, config=1)
    assert up1 != up2                                                   # upstream change invalidates downstream keys
    with pytest.raises(KeyError):
        c.key("nope")
    assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})


def test_crashed_or_markerless_directories_are_never_hits(tmp_path):
    c = StageCache(tmp_path)
    k = key(c)
    with pytest.raises(RuntimeError):
        with c.begin("ingest", k) as d:
            (d / "partial.bin").write_bytes(b"x")
            raise RuntimeError("crash mid-stage")
    assert not c.is_complete("ingest", k) and c.lookup("ingest", k) is None
    assert not [p for p in (tmp_path / "ingest").iterdir()]             # scratch removed
    # a leftover directory with the right name but no marker (hard crash) is not a hit either
    (tmp_path / "ingest" / k).mkdir(parents=True)
    (tmp_path / "ingest" / k / "data.bin").write_bytes(b"y")
    assert not c.is_complete("ingest", k)
    (tmp_path / "ingest" / k / MARKER).write_text("{not json")
    assert not c.is_complete("ingest", k)
    with c.begin("ingest", k, meta={"n": 1}) as d:                      # a good run replaces the stale directory
        (d / "ok.bin").write_bytes(b"z")
    assert c.is_complete("ingest", k) and (c.lookup("ingest", k) / "ok.bin").read_bytes() == b"z"
    assert not (c.lookup("ingest", k) / "data.bin").exists()


def test_downstream_invalidation(tmp_path):
    c = StageCache(tmp_path)
    for stage in ("ingest", "discovery", "vision", "evidence", "association", "stems", "residual", "render"):
        with c.begin(stage, "k" + stage) as d:
            (d / "f").write_text(stage)
    removed = c.invalidate("vision")
    assert set(removed) == {"vision", "evidence", "association", "stems", "residual", "render"}
    assert c.is_complete("ingest", "kingest") and c.is_complete("discovery", "kdiscovery")
    assert not c.is_complete("stems", "kstems")


def test_file_hashes_are_content_based(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same" * 100000)
    b.write_bytes(b"same" * 100000)
    assert hash_file(a) == hash_file(b) == hash_bytes(b"same" * 100000)
    b.write_bytes(b"diff")
    assert hash_file(a) != hash_file(b)


def test_default_config_and_array_example_load_with_the_spec_values():
    cfg = load_config()
    assert cfg.acoustics.ambient_temperature_c == 20.0 and cfg.acoustics.temperature_provenance == "assumed"
    assert cfg.acoustics.r_min_m == 0.5
    assert (cfg.matcher.tof_min_ms, cfg.matcher.tof_max_ms) == (0.0, 35.0)
    assert (cfg.timing.gate_early_ms, cfg.timing.gate_late_ms, cfg.timing.skew_base_ms) == (15.0, 45.0, 20.0)
    assert cfg.spatial.wdo_min_weight == 0.1 and cfg.off_axis.yaw_limit_deg == 60.0
    assert cfg.envelope.attack_ms == 30.0 and cfg.envelope.release_ms == 180.0
    assert len(cfg.hash()) == 64
    cal = load_array_calibration()
    assert cal.baffle_scale == 1.0 and len(cal.mic_positions_m) == 2 and cal.camera.fx == 300.0


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[timing]\ngate_early_ms = 15.0\nbogus = 1\n")
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text("[nonsense]\nx = 1\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_signed_sync_gates_with_symmetric_skew():
    q = 20.0                                                            # 25 fps
    # audio 40 ms late, mask 40 ms late -> within [-35, +65], skew 0
    ok = check_sync(1_040_000, 1_040_000, 1_000_000, q)
    assert ok.ok and ok.delta_skew_ms == 0
    # +65 + epsilon late fails; -35 early passes, -36 fails (asymmetric gate)
    assert check_sync(1_065_000, 1_065_000, 1_000_000, q).audio_ok
    assert not check_sync(1_066_000, 1_066_000, 1_000_000, q).audio_ok
    assert check_sync(965_000, 965_000, 1_000_000, q).audio_ok and not check_sync(964_000, 964_000, 1_000_000, q).audio_ok
    # skew is symmetric: +-(20 + q) = +-40 ms regardless of sign, even though each endpoint is individually in-gate
    assert check_sync(1_000_000, 1_040_000, 1_000_000, q).skew_ok and check_sync(1_040_000, 1_000_000, 1_000_000, q).skew_ok
    late_mask = check_sync(1_000_000, 1_041_000, 1_000_000, q)
    early_mask = check_sync(1_041_000, 1_000_000, 1_000_000, q)
    assert late_mask.delta_skew_ms == 41.0 and not late_mask.skew_ok and late_mask.audio_ok and late_mask.mask_ok
    assert early_mask.delta_skew_ms == -41.0 and not early_mask.skew_ok          # opposite endpoint errors cannot hide
    assert json.dumps(ok.to_dict())
    # variable frame rate: a longer local interval widens q
    assert not check_sync(1_000_000, 1_050_000, 1_000_000, 20.0).skew_ok
    assert check_sync(1_000_000, 1_050_000, 1_000_000, 30.0).skew_ok
