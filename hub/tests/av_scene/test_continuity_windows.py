import numpy as np

from av_scene.continuity_windows import EmbeddingCache, track_identities

SR = 16000
DIM = 64
rng = np.random.default_rng(0)


def _spk():
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


VOICES = {1: _spk(), 2: _spk(), 3: _spk()}


def fake_embed(seg):
    """Fake extractor: speaker tag stored in sample 0 of the (4 s) window."""
    return VOICES[int(seg[0]) or int(seg.max())] + rng.normal(scale=0.05, size=DIM)


def _build(plan):
    """plan: list of 2 s chunks, each a list of per-slot speaker tags (0 = silent).
    Windows are 4 s / hop 2 s so each window starts on a chunk boundary."""
    streams = {}
    for k in range(len(plan[0])):
        chunks = []
        for t in plan:
            c = np.zeros(2 * SR, dtype=np.float32)
            if t[k]:
                c[:] = 0.1
                c[0] = t[k]
            chunks.append(c)
        streams[f"s{k}"] = np.concatenate(chunks)
    return streams


def _ids(res, sid):
    return [w.mapping.get(sid) for w in res]


def test_same_speaker_after_pause_keeps_id():
    s = _build([[1], [1], [0], [0], [0], [0], [1], [1], [1]])
    res = track_identities(s, fake_embed)
    ids = [i for i in _ids(res, "s0") if i is not None]
    assert any(w.skipped for w in res)  # pause windows are skipped
    assert len(ids) >= 2 and len(set(ids)) == 1


def test_swap_between_slots_keeps_ids():
    s = _build([[1, 2], [1, 2], [1, 2], [2, 1], [2, 1], [2, 1]])
    res = track_identities(s, fake_embed)
    first = res[0].mapping
    assert first["s0"] != first["s1"]
    last = res[-1].mapping
    assert last["s0"] == first["s1"] and last["s1"] == first["s0"]
    assert {i for w in res for i in w.mapping.values()} == set(first.values())


def test_new_speaker_gets_new_id():
    s = _build([[1], [1], [1], [3], [3], [3]])
    res = track_identities(s, fake_embed)
    ids = [i for i in _ids(res, "s0") if i is not None]
    assert ids[0] != ids[-1] and len(set(ids)) == 2
    assert any(w.new_ids for w in res[1:])


def test_silent_windows_create_no_ids():
    s = _build([[0], [0], [0], [0]])
    res = track_identities(s, fake_embed)
    assert all(not w.mapping and not w.new_ids and w.skipped == ["s0"] for w in res)


def test_embedding_cache_hits_and_confidence():
    s = _build([[1], [1], [1], [1]])
    cache = EmbeddingCache(fake_embed)
    r1 = track_identities(s, cache)
    m = cache.misses
    track_identities(s, cache)
    assert cache.misses == m and cache.hits >= m
    assert all(0.0 <= c <= 1.0 for w in r1 for c in w.confidence.values())
