"""Model-free stream<->track association and selection (Phase 2 speech branch).

Pure numpy logic. Inputs are produced elsewhere: per-stream energy envelopes
(25 fps) from separation, and per-track TalkNet speaking scores (25 fps, NaN
where the face is not visible). Nothing here loads a model.

Invariants: unavailable evidence is None, never 0.0; selection never falls back
to the loudest/dominant stream.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np
from pydantic import BaseModel

from .schemas import Association, ComponentScore

FPS = 25
MIN_CORR = 0.35
MIN_MARGIN = 0.15
MIN_ACTIVE_FRAC = 0.05
MIN_VALID_FRAMES = 10          # ~0.4 s of jointly valid frames
SILENT_THR = 0.3               # track score below this == confidently silent
ACT_REL = 0.1                  # stream active when env >= ACT_REL * peak
ACT_FLOOR = 1e-3               # ...and peak must exceed this absolute floor
SWAP_MARGIN = 0.3              # evidence needed to accept a mapping change
EMBED_SAME = 0.7               # stream embedding sim that supports identity
MAX_SEPARABLE_TALKERS = 2      # MossFormer2_SS limit


class Extraction(BaseModel):
    """Local mirror of the spec'd Extraction (not yet in schemas.py)."""
    status: Literal["ok", "degraded", "unavailable", "not_attempted"]
    method: str = "track_stream_association"
    stem_source_id: str | None = None
    reason: str = ""


def _us(frame: float, fps: int) -> int:
    return int(round(frame * 1_000_000 / fps))


def _activity(env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (continuous activity in [0,1], boolean active mask)."""
    env = np.nan_to_num(np.asarray(env, dtype=float), nan=0.0)
    peak = float(env.max()) if env.size else 0.0
    if peak < ACT_FLOOR:
        return np.zeros_like(env), np.zeros(env.shape, bool)
    mask = env >= ACT_REL * peak
    return np.where(mask, np.clip(env / peak, 0, 1), 0.0), mask


def _corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray, min_valid: int):
    """Pearson r over valid frames, or None if undefined."""
    if int(valid.sum()) < min_valid:
        return None
    x, y = a[valid], b[valid]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def associate_streams(stream_env: dict[str, np.ndarray],
                      track_scores: dict[str, np.ndarray],
                      fps: int = FPS,
                      min_corr: float = MIN_CORR,
                      min_margin: float = MIN_MARGIN,
                      min_active_frac: float = MIN_ACTIVE_FRAC,
                      one_to_one: bool = True,
                      min_valid_frames: int = MIN_VALID_FRAMES,
                      silent_thr: float = SILENT_THR,
                      t0_us: int = 0) -> list[Association]:
    """Associate each separated stream with at most one visual track.

    one_to_one=False lets several streams map to the same track.
    """
    sids, tids = list(stream_env), list(track_scores)
    acts = {s: _activity(stream_env[s]) for s in sids}
    scores = {t: np.asarray(track_scores[t], dtype=float) for t in tids}

    corr: dict[tuple[str, str], float | None] = {}
    nvalid: dict[tuple[str, str], int] = {}
    for s in sids:
        a = acts[s][0]
        for t in tids:
            n = min(len(a), len(scores[t]))
            v = ~np.isnan(scores[t][:n])
            nvalid[(s, t)] = int(v.sum())
            corr[(s, t)] = _corr(a[:n], np.nan_to_num(scores[t][:n]), v, min_valid_frames)

    active_frac = {s: float(acts[s][1].mean()) if acts[s][1].size else 0.0 for s in sids}
    eligible = {s for s in sids if active_frac[s] >= min_active_frac}

    # greedy assignment, strongest correlation first
    pairs = sorted(((c, s, t) for (s, t), c in corr.items()
                    if c is not None and c >= min_corr and s in eligible),
                   key=lambda x: -x[0])
    chosen: dict[str, str] = {}
    used_tracks: set[str] = set()
    for c, s, t in pairs:
        if s in chosen or (one_to_one and t in used_tracks):
            continue
        chosen[s] = t
        used_tracks.add(t)

    def margin(s: str, t: str) -> float | None:
        c = corr[(s, t)]
        others = [corr[(s, u)] for u in tids if u != t and corr[(s, u)] is not None]
        others += [corr[(r, t)] for r in sids if r != s and corr[(r, t)] is not None]
        return None if not others else float(c - max(others))

    out: list[Association] = []
    for s in sids:
        n = len(acts[s][0])
        start, end = t0_us, t0_us + _us(n, fps)
        comps = [ComponentScore(name="stream_active_frac", value=active_frac[s],
                                quality=1.0, weight=0.0)]
        t = chosen.get(s)
        if t is not None:
            c, m = corr[(s, t)], margin(s, t)
            comps.insert(0, ComponentScore(
                name="envelope_speaking_corr", value=c,
                quality=min(1.0, nvalid[(s, t)] / max(1, n)), weight=1.0))
            amb = m is not None and m < min_margin
            out.append(Association(
                association_id=f"assoc:{s}:{t}", source_id=s, track_id=t,
                valid_start_us=start, valid_end_us=end,
                status="ambiguous" if amb else "matched", visibility="visible",
                score=c, components=comps, runner_up_margin=m))
            continue
        vis = "unknown"
        note = "stream inactive" if s not in eligible else "no track above threshold"
        mask = acts[s][1]
        if s in eligible:
            seen = silent = 0
            for u in tids:
                n2 = min(len(mask), len(scores[u]))
                m2 = mask[:n2] & ~np.isnan(scores[u][:n2])
                if m2.sum() >= min_valid_frames:
                    seen += 1
                    if float(np.mean(scores[u][:n2][m2])) < silent_thr:
                        silent += 1
            if seen > 0 and seen == silent:
                vis, note = "outside_view", "stream active while all visible tracks silent"
        best = max((c for (ss, _), c in corr.items() if ss == s and c is not None),
                   default=None)
        comps.append(ComponentScore(name="best_track_corr", value=best,
                                    quality=1.0, weight=0.0, note=note))
        out.append(Association(
            association_id=f"assoc:{s}:none", source_id=s, track_id=None,
            valid_start_us=start, valid_end_us=end, status="unmatched",
            visibility=vis, score=None, components=comps, runner_up_margin=None))
    return out


def track_activity(track_scores: dict[str, np.ndarray], threshold: float = 0.5,
                   min_dur_frames: int = 5, fps: int = FPS
                   ) -> dict[str, list[tuple[int, int]]]:
    """Speaking intervals (start_us, end_us) per track; NaN counts as inactive."""
    res: dict[str, list[tuple[int, int]]] = {}
    for t, sc in track_scores.items():
        on = np.nan_to_num(np.asarray(sc, dtype=float), nan=0.0) >= threshold
        runs, i, n = [], 0, len(on)
        while i < n:
            if on[i]:
                j = i
                while j < n and on[j]:
                    j += 1
                if j - i >= min_dur_frames:
                    runs.append((_us(i, fps), _us(j, fps)))
                i = j
            else:
                i += 1
        res[t] = runs
    return res


def select_for_track(track_id: str, associations: list[Association],
                     streams: Mapping[str, Any],
                     target_active: bool | None = None,
                     n_suspected_talkers: int | None = None
                     ) -> tuple[str | None, Extraction]:
    """Return the stream associated with a track, only if status == matched.

    Never falls back to the loudest/dominant stream.
    """
    def no(reason: str):
        return None, Extraction(status="unavailable", reason=reason)

    if target_active is False:
        return no("target inactive")
    mine = [a for a in associations if a.track_id == track_id]
    ok = [a for a in mine if a.status == "matched" and a.source_id in streams]
    if not ok:
        if any(a.status == "ambiguous" for a in mine):
            return no("ambiguous")
        return no("no associated stream")
    best = max(ok, key=lambda a: a.score if a.score is not None else -1.0)
    if n_suspected_talkers is not None and n_suspected_talkers > MAX_SEPARABLE_TALKERS:
        return best.source_id, Extraction(
            status="degraded", stem_source_id=best.source_id,
            reason=f"{n_suspected_talkers} talkers suspected; separator supports "
                   f"{MAX_SEPARABLE_TALKERS}")
    return best.source_id, Extraction(status="ok", stem_source_id=best.source_id)


def continuity(prev: list[Association], curr: list[Association],
               stream_embed_sim: dict | None = None,
               swap_margin: float = SWAP_MARGIN,
               embed_same: float = EMBED_SAME) -> list[Association]:
    """Downgrade unsupported mapping changes to ambiguous instead of swapping.

    stream_embed_sim maps (prev_stream_id, curr_stream_id) -> cosine similarity.
    A change is supported by runner_up_margin >= swap_margin or by a high
    embedding similarity between the streams involved.
    """
    sim = stream_embed_sim or {}
    p_by_track = {a.track_id: a for a in prev if a.status == "matched" and a.track_id}
    p_by_stream = {a.source_id: a for a in prev if a.status == "matched" and a.track_id}
    out = []
    for a in curr:
        if a.status != "matched" or a.track_id is None:
            out.append(a)
            continue
        changed_from = []
        pt = p_by_track.get(a.track_id)
        if pt is not None and pt.source_id != a.source_id:
            changed_from.append(pt.source_id)
        ps = p_by_stream.get(a.source_id)
        if ps is not None and ps.track_id != a.track_id:
            changed_from.append(a.source_id)
        if not changed_from:
            out.append(a)
            continue
        supported = a.runner_up_margin is not None and a.runner_up_margin >= swap_margin
        if not supported:
            supported = any(sim.get((p, a.source_id), sim.get((a.source_id, p), -1.0))
                            >= embed_same for p in changed_from)
        if supported:
            out.append(a)
        else:
            note = ComponentScore(name="continuity", value=None, quality=0.0, weight=0.0,
                                  note="mapping changed from prior window without "
                                       "sufficient evidence")
            out.append(a.model_copy(update={"status": "ambiguous",
                                            "components": [*a.components, note]}))
    return out
