"""Phase 4 contracts: chunk coverage, time conversion, state hashing, config registration."""
import numpy as np
import pytest

from av_scene.config import load_config
from av_scene.contracts import (AvMap, PipelineState, check_coverage, plan_chunks, sample_to_us, us_to_sample,
                                video_index_range)

SR = 16000


@pytest.mark.parametrize("seconds", [0.5, 3.99, 4.0, 4.0 + 1 / SR, 4.5, 5.0, 6.0, 7.3, 30.0, 61.37, 300.0])
def test_owned_regions_tile_exactly_including_final_partial(seconds):
    n = int(round(seconds * SR))
    specs = plan_chunks(n)
    check_coverage(specs, n)
    owned = np.zeros(n, int)
    for c in specs:
        owned[c.own_start:c.own_end] += 1
    assert (owned == 1).all()                              # no missing, no duplicated sample
    assert specs[0].is_first and specs[-1].is_last and sum(c.own_len for c in specs) == n
    for c in specs[:-1]:
        assert c.win_len == 4 * SR                         # only the last window may be partial


def test_boundaries_sit_mid_overlap_so_owned_audio_is_away_from_window_edges():
    specs = plan_chunks(20 * SR)
    for a, b in zip(specs, specs[1:]):
        assert a.own_end == b.own_start
        assert a.own_end - b.win_start == 1 * SR and a.win_end - a.own_end == 1 * SR   # 1 s margin each side


def test_other_window_overlap_and_degenerate_inputs():
    n = 10 * SR
    specs = plan_chunks(n, window_s=3.0, overlap_s=0.0)
    check_coverage(specs, n)
    assert plan_chunks(0) == []
    with pytest.raises(ValueError):
        plan_chunks(n, window_s=2.0, overlap_s=2.0)
    with pytest.raises(AssertionError):
        check_coverage(plan_chunks(n)[:-1], n)


def test_time_conversion_is_integer_and_documents_non_bijection():
    assert sample_to_us(16000, SR) == 1_000_000 and sample_to_us(1, SR) == 63            # 62.5 rounds half up
    assert us_to_sample(1_000_000, SR) == 16000 and isinstance(sample_to_us(5, 48000), int)
    assert sample_to_us(48000 * 3, 48000) == 3_000_000
    m = AvMap(SR, audio_start_us=250_000)
    assert m.sample_to_common_us(16000) == 1_250_000 and m.common_us_to_sample(1_250_000) == 16000
    pts = [0, 40_000, 80_000, 130_000, 160_000]                                          # variable rate
    assert video_index_range(pts, 40_000, 130_000) == (1, 3)                              # half-open
    assert video_index_range(pts, 0, 10**9) == (0, 5) and video_index_range(pts, 200, 300) == (1, 1)


def test_pipeline_state_hash_chains_and_round_trips(tmp_path):
    a = PipelineState(next_chunk=2, emitted_samples=48000, identity={"n": 1})
    b = PipelineState.from_dict(a.to_dict())
    assert a.hash() == b.hash()
    b.identity = {"n": 2}
    assert a.hash() != b.hash()
    p = tmp_path / "s.json"
    a.write(p)
    assert PipelineState.read(p) == a and not list(tmp_path.glob("*.tmp"))


def test_config_exposes_chunking_memory_budgets_and_switches():
    cfg = load_config()
    assert (cfg.chunking.window_s, cfg.chunking.overlap_s) == (4.0, 2.0)
    assert cfg.memory.max_resident_models == 2 and cfg.budgets.rss_peak_mib > 0
    assert cfg.switches.diarization == "off" and not cfg.switches.peel
    assert "switches" in cfg.to_dict()
