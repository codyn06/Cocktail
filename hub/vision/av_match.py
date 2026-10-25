"""Voice-to-face association by audio/visual contrast (pure logic: no models, no I/O).

Pieces
- FaceTracker: tiny IoU tracker over normalized face boxes (stable ids f1, f2, ...; <= 4 faces; a track is
  dropped after 2 s without a detection; between detections the box is carried by linear extrapolation).
- mouth_motion(): mean absolute grayscale frame difference over the lower third of a face box.
- AVMatch: collects 100 ms bins labelled by who is speaking ALONE (speaker id, or 'nobody'), with the
  per-face mean mouth motion of that bin, and scores every (speaker, face) pair by CONTRAST:

      score = (mean motion while S speaks alone - mean motion while others speak alone)
              / (mean_S + mean_others + 1)                       (bounded to (-1, 1), unit-free)

  Speakers are assigned to faces jointly (Hungarian), then gated by evidence, an absolute floor, and a
  margin (best >= 1.5x runner-up face for that speaker AND no other speaker scores >= best/1.5 on that
  face). Anything that fails a gate yields NO association (the caller publishes no box).

Clock: speaker intervals (hub/speakers recorder) and camera frames (VideoFrame.device_ts_us) are both in
the Quest device microsecond clock. Bins tolerate +-200 ms skew: a bin counts as "S alone" only if S's
interval overlaps it and no OTHER speaker's interval, dilated by 200 ms each side, touches it; 'nobody'
bins need no interval (dilated) to touch. Overlapping speech therefore adds no evidence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

BIN_US = 100_000
SLACK_US = 200_000
FINALIZE_LAG_US = 350_000        # wait this long behind the newest frame so intervals have caught up
BIN_WINDOW = 400                 # keep ~40 s of labelled bins
MIN_ALONE_S = 1.5
MIN_SINGLE_S = 2.5
SCORE_FLOOR = 0.08
MARGIN_X = 1.5
SWITCH_S = 3.0
MAX_FACES = 4
DROP_US = 2_000_000
IOU_MIN = 0.2
HISTORY_US = 6_000_000
NOBODY = "_nobody"


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


@dataclass
class Face:
    id: str
    box: tuple            # last detected box (x0,y0,x1,y1), normalized
    det_ts: int
    vx: float = 0.0       # normalized units / second
    vy: float = 0.0
    series: deque = field(default_factory=deque)   # (ts_us, motion)


def predict_box(f: Face, ts: int) -> tuple:
    dt = min(max((ts - f.det_ts) / 1e6, 0.0), 0.6)
    dx, dy = f.vx * dt, f.vy * dt
    b = f.box
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


class FaceTracker:
    def __init__(self):
        self.faces: dict[str, Face] = {}
        self._n = 0

    def reset(self):
        self.faces.clear()

    def update(self, dets: list, ts: int) -> None:
        pairs = []
        for fid, f in self.faces.items():
            pb = predict_box(f, ts)
            for j, d in enumerate(dets):
                v = iou(pb, d)
                if v >= IOU_MIN:
                    pairs.append((v, fid, j))
        pairs.sort(reverse=True)
        used_f, used_d = set(), set()
        for _, fid, j in pairs:
            if fid in used_f or j in used_d:
                continue
            used_f.add(fid)
            used_d.add(j)
            f, d = self.faces[fid], dets[j]
            dt = (ts - f.det_ts) / 1e6
            if 0.05 < dt < 1.0:
                ncx, ocx = (d[0] + d[2]) / 2, (f.box[0] + f.box[2]) / 2
                ncy, ocy = (d[1] + d[3]) / 2, (f.box[1] + f.box[3]) / 2
                f.vx = 0.5 * f.vx + 0.5 * (ncx - ocx) / dt
                f.vy = 0.5 * f.vy + 0.5 * (ncy - ocy) / dt
            elif dt >= 1.0:
                f.vx = f.vy = 0.0
            f.box, f.det_ts = tuple(d), ts
        for fid in [k for k, f in self.faces.items() if ts - f.det_ts > DROP_US]:
            del self.faces[fid]
        for j, d in enumerate(dets):
            if j in used_d or len(self.faces) >= MAX_FACES:
                continue
            self._n += 1
            self.faces[f"f{self._n}"] = Face(f"f{self._n}", tuple(d), ts)

    def prune(self, ts: int) -> None:
        for fid in [k for k, f in self.faces.items() if ts - f.det_ts > DROP_US]:
            del self.faces[fid]


def mouth_motion(prev: np.ndarray, cur: np.ndarray, box) -> float | None:
    """Mean |cur - prev| over the lower third of a normalized face box (same pixels in both frames)."""
    h, w = cur.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0, x1 = int((box[0] + 0.15 * bw) * w), int((box[2] - 0.15 * bw) * w)
    y0, y1 = int((box[1] + 0.62 * bh) * h), int(box[3] * h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3 or prev.shape != cur.shape:
        return None
    a = prev[y0:y1, x0:x1].astype(np.int16)
    b = cur[y0:y1, x0:x1].astype(np.int16)
    return float(np.abs(b - a).mean())


@dataclass
class Assoc:
    face: str
    score: float
    margin: float
    evidence_s: float


class AVMatch:
    def __init__(self):
        self.bins: deque = deque(maxlen=BIN_WINDOW)     # (bin_idx, label, {fid: mean_motion})
        self._pend: dict[int, dict[str, list]] = {}
        self._done_to = None            # first bin index not yet finalized
        self.assoc: dict[str, Assoc] = {}
        self._contra: dict[str, tuple[str, int]] = {}   # sid -> (face, since_ts)
        self.reasons: dict[str, str] = {}
        self.latest: dict[str, Assoc | None] = {}
        self.last_ts = 0

    def reset(self):
        self.bins.clear()
        self._pend.clear()
        self._done_to = None
        self.assoc.clear()
        self._contra.clear()
        self.reasons.clear()
        self.latest.clear()

    # ---- evidence collection
    def add_motion(self, fid: str, ts: int, m: float) -> None:
        b = ts // BIN_US
        s = self._pend.setdefault(b, {}).setdefault(fid, [0.0, 0])
        s[0] += m
        s[1] += 1
        self.last_ts = max(self.last_ts, ts)

    def finalize(self, intervals: dict[str, list]) -> None:
        """intervals: sid -> [(start_us, end_us)]. Label every bin older than newest-frame minus lag."""
        if not self._pend:
            return
        cutoff = (self.last_ts - FINALIZE_LAG_US) // BIN_US
        for b in sorted(k for k in self._pend if k < cutoff):
            faces = {fid: s[0] / s[1] for fid, s in self._pend.pop(b).items() if s[1]}
            lo, hi = b * BIN_US, (b + 1) * BIN_US
            core, near = [], []
            for sid, ivs in intervals.items():
                if any(s < hi and e > lo for s, e in ivs):
                    core.append(sid)
                if any(s - SLACK_US < hi and e + SLACK_US > lo for s, e in ivs):
                    near.append(sid)
            if not near:
                label = NOBODY
            elif len(core) == 1 and len(near) == 1:
                label = core[0]
            else:
                continue            # overlap / boundary: no evidence
            self.bins.append((b, label, faces))
        # drop leftover ancient pending bins (no frames for them will come)
        for k in [k for k in self._pend if k < cutoff - 50]:
            del self._pend[k]

    # ---- scoring
    def _stats(self, face_ids: list[str]):
        """label -> fid -> (sum, n)."""
        st: dict[str, dict[str, list]] = {}
        for _, label, fm in self.bins:
            for fid, m in fm.items():
                if fid in face_ids:
                    x = st.setdefault(label, {}).setdefault(fid, [0.0, 0])
                    x[0] += m
                    x[1] += 1
        return st

    def scores(self, sids: list[str], face_ids: list[str]):
        """-> (score[sid][fid] or None, evidence_s[sid]); None = insufficient evidence."""
        st = self._stats(face_ids)
        speaking = [s for s in sids if s in st]
        single = len(sids) <= 1
        need_oth = MIN_SINGLE_S if single else MIN_ALONE_S
        need_own = MIN_SINGLE_S if single else MIN_ALONE_S
        out: dict[str, dict[str, float | None]] = {}
        ev: dict[str, float] = {}
        for s in sids:
            out[s] = {}
            own = st.get(s, {})
            ev[s] = max((v[1] for v in own.values()), default=0) * BIN_US / 1e6
            for fid in face_ids:
                o = own.get(fid)
                if single:
                    base = st.get(NOBODY, {}).get(fid)
                    nb, sb = (base[1], base[0]) if base else (0, 0.0)
                else:
                    nb = sum(st.get(t, {}).get(fid, (0, 0))[1] for t in sids if t != s)
                    sb = sum(st.get(t, {}).get(fid, (0.0, 0))[0] for t in sids if t != s)
                if not o or o[1] * BIN_US / 1e6 < need_own or nb * BIN_US / 1e6 < need_oth:
                    out[s][fid] = None
                    continue
                ms, mo = o[0] / o[1], sb / nb
                out[s][fid] = (ms - mo) / (ms + mo + 1.0)
        return out, ev

    def update(self, sids: list[str], face_ids: list[str], now_ts: int) -> None:
        """Recompute candidates and apply the sticky policy. Never raises on empty input."""
        from scipy.optimize import linear_sum_assignment

        for s in [s for s in self.assoc if s not in sids]:
            self.assoc.pop(s)
            self._contra.pop(s, None)
        # sticky: drop associations whose face track vanished
        for s in [s for s, a in self.assoc.items() if a.face not in face_ids]:
            self.assoc.pop(s)
            self._contra.pop(s, None)
            self.reasons[s] = "face_unseen"
        if not sids or not face_ids:
            for s in sids:
                self.reasons.setdefault(s, "no_association")
            return
        sc, ev = self.scores(sids, face_ids)
        rows = [s for s in sids if any(v is not None for v in sc[s].values())]
        cands: dict[str, Assoc] = {}
        for s in sids:
            if s not in rows:
                self.reasons[s] = "insufficient_evidence"
        if rows:
            M = np.full((len(rows), len(face_ids)), -1.0)
            for i, s in enumerate(rows):
                for j, f in enumerate(face_ids):
                    v = sc[s][f]
                    if v is not None:
                        M[i, j] = v
            ri, cj = linear_sum_assignment(M, maximize=True)
            for i, j in zip(ri, cj):
                s, f = rows[i], face_ids[j]
                v = sc[s][f]
                if v is None or v < SCORE_FLOOR:
                    self.reasons[s] = "no_association"
                    continue
                rival_f = max([x for g, x in sc[s].items() if g != f and x is not None] + [0.0])
                rival_s = max([sc[t][f] for t in sids if t != s and sc[t].get(f) is not None] + [0.0])
                if v < MARGIN_X * rival_f or v < MARGIN_X * rival_s:
                    self.reasons[s] = "ambiguous"
                    continue
                margin = v / max(rival_f, rival_s, 1e-6)
                cands[s] = Assoc(f, v, min(margin, 99.0), ev[s])
            for s in rows:
                if s not in cands:
                    self.reasons.setdefault(s, "no_association")
        self.latest = {s: cands.get(s) for s in sids}
        held = {a.face: s for s, a in self.assoc.items()}
        for s in sids:
            c, cur = cands.get(s), self.assoc.get(s)
            if cur is not None:
                if c is None or c.face == cur.face:
                    self._contra.pop(s, None)
                    if c is not None:
                        self.assoc[s] = c
                    continue
                f0, since = self._contra.get(s, (None, 0))
                if f0 != c.face:
                    self._contra[s] = (c.face, now_ts)
                elif now_ts - since > SWITCH_S * 1e6 and held.get(c.face, s) == s:
                    self.assoc[s] = c
                    held = {a.face: x for x, a in self.assoc.items()}
                    self._contra.pop(s, None)
            elif c is not None and held.get(c.face, s) == s:
                self.assoc[s] = c
                held[c.face] = s
                self.reasons.pop(s, None)
        for s in self.assoc:
            self.reasons.pop(s, None)

    # ---- debug
    def bins_tail(self, n: int = 60):
        return [[b * BIN_US, lab, {k: round(v, 2) for k, v in fm.items()}] for b, lab, fm in list(self.bins)[-n:]]
