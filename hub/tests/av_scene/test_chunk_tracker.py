import json

import numpy as np

from vision.chunk_tracker import ChunkTracker, empty_state

H, W = 40, 80
FPS = 25


def rect(x, y=10, w=10, h=20):
    m = np.zeros((H, W), bool)
    m[y:y + h, max(0, x):max(0, x + w)] = True
    return m


class FakeWorld:
    """Frames carry their time in ms in pixel [0,0] (3 bytes); people are rectangles by time."""

    def __init__(self):
        self.calls = []

    def frame(self, t_us):
        ms = t_us // 1000
        f = np.zeros((H, W, 3), np.uint8)
        f[0, 0] = [ms % 256, (ms // 256) % 256, (ms // 65536) % 256]
        return f

    @staticmethod
    def t_of(f):
        return int(f[0, 0, 0]) + int(f[0, 0, 1]) * 256 + int(f[0, 0, 2]) * 65536

    def __call__(self, rgb):
        t = self.t_of(rgb)
        self.calls.append(t)
        s = t / 1000
        out = [(rect(int(5 + 10 * s)), 0.9),          # A moves right
               (rect(int(60 - 10 * s)), 0.9)]         # B moves left, crosses A
        if not (2.0 <= s < 3.4):                      # C leaves 2.0-3.4 s (gap > max_gap), re-enters
            out.append((rect(35, y=2, w=6, h=10), 0.8))
        return out


def face_fn(frames, masks_by_track):
    res = {}
    for t, ms in masks_by_track.items():
        res[t] = [None if m is None else (float(np.nonzero(m)[1].min()), float(np.nonzero(m)[0].min()),
                                          float(np.nonzero(m)[1].min() + 3), float(np.nonzero(m)[0].min() + 3))
                  for m in ms]
    return res


def make(world):
    return ChunkTracker(world, face_fn, run_tag="r", max_gap=5)


def windows(world, pts_all, size=100, hop=50):
    i = 0
    while i < len(pts_all):
        p = pts_all[i:i + size]
        yield np.stack([world.frame(x) for x in p]), p
        if i + size >= len(pts_all):
            break
        i += hop


def run(world, pts_all, roundtrip, **kw):
    tr = make(world)
    state, outs = empty_state(), []
    for fr, p in windows(world, pts_all, **kw):
        ch = tr.step(fr, p, state)
        state = json.loads(json.dumps(ch.state)) if roundtrip else ch.state
        outs.append((p, ch))
    return outs


def digest(outs):
    return json.dumps([[p, {t: [None if r is None else r.to_dict() for r in rs] for t, rs in c.masks_by_track.items()},
                        c.boxes_by_track, c.state] for p, c in outs], sort_keys=True, default=list)


PTS = [int(i * 1e6 / FPS) for i in range(300)]       # 12 s


def _spans(outs):
    seen = {}
    for _, c in outs:
        for t, rs in c.masks_by_track.items():
            for r in rs:
                if r is not None:
                    seen.setdefault(t, set()).add(r.pts_us)
    return seen


def test_ids_stable_across_boundaries_and_namespaced():
    outs = run(FakeWorld(), PTS, True)
    ids0, ids1 = set(outs[0][1].masks_by_track), set(outs[1][1].masks_by_track)
    assert all(t.startswith("sam3_image:r:") for t in ids0 | ids1)
    shared = ids0 & ids1
    assert len(shared) >= 2
    for t in shared:
        a = outs[0][1].masks_by_track[t][50:100]
        b = outs[1][1].masks_by_track[t][0:50]
        assert [r and r.rle.counts for r in a] == [r and r.rle.counts for r in b]


def test_overlap_not_resegmented():
    w = FakeWorld()
    run(w, PTS, True)
    assert len(w.calls) == len(PTS) and len(set(w.calls)) == len(PTS)


def test_reentry_gets_new_visual_id():
    seen = _spans(run(FakeWorld(), PTS, True))
    # C: present 0-2.0 s, gone 2.0-3.4 s (gap 35 frames > max_gap), back after 3.4 s
    early = {t for t, s in seen.items() if min(s) == 0 and 1.9e6 < max(s) < 2.1e6}
    late = {t for t, s in seen.items() if 3.3e6 < min(s) < 3.5e6}
    assert early and late and not (early & late)


def test_resume_bit_identical():
    assert digest(run(FakeWorld(), PTS, False)) == digest(run(FakeWorld(), PTS, True))


def test_ring_and_state_bounded():
    w = FakeWorld()
    tr = make(w)
    state = empty_state()
    sizes = []
    for fr, p in windows(w, PTS):
        ch = tr.step(fr, p, state)
        state = json.loads(json.dumps(ch.state))
        assert len(fr) <= 150                       # MemoryCaps.max_decoded_frames
        assert len(state["ring"]) <= int(2.5 * FPS) + 1
        sizes.append(len(json.dumps(state)))
    assert sizes[-1] < 1.5 * max(sizes[:4])         # no growth with stream length


def test_zero_detections_window():
    tr = ChunkTracker(lambda rgb: [], face_fn, run_tag="r")
    p = PTS[:100]
    fr = np.stack([FakeWorld().frame(x) for x in p])
    ch = tr.step(fr, p, empty_state())
    assert ch.boxes_by_track == {} and ch.masks_by_track == {}
    json.dumps(ch.state)
    assert tr.step(fr, p, ch.state).masks_by_track == {}


def test_partial_first_window_and_boxes():
    w = FakeWorld()
    tr = make(w)
    p = PTS[:30]                                    # fewer frames than the 2.5 s overlap
    ch = tr.step(np.stack([w.frame(x) for x in p]), p, empty_state())
    assert len(ch.state["ring"]) == 30
    for t, bs in ch.boxes_by_track.items():
        assert len(bs) == 30 and len(ch.masks_by_track[t]) == 30
        assert all(b is not None for b in bs)
    n0 = len(w.calls)
    p2 = PTS[10:40]
    ch2 = tr.step(np.stack([w.frame(x) for x in p2]), p2, json.loads(json.dumps(ch.state)))
    assert len(w.calls) - n0 == 10
    assert set(ch2.masks_by_track) == set(ch.masks_by_track)


def test_variable_rate_pts_and_skip():
    pts = [0]
    rng = np.random.default_rng(0)
    while len(pts) < 150:
        pts.append(pts[-1] + int(rng.choice([20000, 40000, 40000, 120000])))
    w = FakeWorld()
    a = digest(run(w, pts, False, size=60, hop=30))
    b = digest(run(FakeWorld(), pts, True, size=60, hop=30))
    assert a == b
    assert len(set(w.calls)) == len(w.calls)
