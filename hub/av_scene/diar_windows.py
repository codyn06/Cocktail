"""Optional, bounded-window diarization evidence for `separate.diarization.compute_anchors`.

Samples are 16 kHz and windows are half-open `[start, end)`; intervals handed to `compute_anchors`
are integer microseconds, half-open (see `av_scene.contracts`). Audio is read lazily through an
injected `read_audio(start_sample, end_sample)`, one window at a time (never two resident).

Pyannote labels (`SPEAKER_00`, ...) are window-local and NOT stable across windows. Cross-window
continuity is never assumed: `map_labels_to_sources` links a label to a persistent source id only
through overlap with identity-derived activity (Hungarian assignment). Anything below threshold, or
tied, stays unmapped as `diar:w<window_index>:<label>` and is never merged into a source.

Consecutive windows overlap; each window owns the first half of each overlap (boundary at the
midpoint), so no time is duplicated or dropped.

Statuses (per window and overall): "ok", "not_attempted" (package / cache / credentials missing,
NOT an error), "skipped" (disabled by config), "failed" (pipeline raised; only that window loses
evidence, the run continues). Default is OFF (`mode="off"`).

Limitations
- 4 s pipeline chunks are not diarized directly: pyannote needs long context to cluster speakers;
  on 4 s inputs speaker counts and labels are unreliable and inconsistent chunk to chunk.
- NOT demonstrated: no real pyannote run was performed (the model is gated, no token/package here).
  Everything is tested with `FakePipeline` (mocked).
- Turn on only when (a) pyannote + the gated model are available (see `availability`), and (b) bounded,
  consistent diarization has been shown on real long clips (label->source mapping stable, unmapped
  fraction acceptable). Otherwise leave it off; it must never weaken the identity memory guarantee.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import permutations

from separate.diarization import intersect, normalize, total_us
from separate.pyannote_diar import MODEL_ID, diarize

SR = 16000
STATUSES = ("ok", "not_attempted", "skipped", "failed")


@dataclass
class DiarConfig:
    window_s: float = 30.0
    hop_s: float = 25.0
    min_turn_s: float = 0.3
    min_overlap_s: float = 0.5
    map_threshold: float = 0.5


@dataclass
class DiarStatus:
    status: str
    reason: str = ""
    window: int | None = None

    def to_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "window": self.window}


@dataclass
class DiarResult:
    status: DiarStatus
    windows: list[DiarStatus] = field(default_factory=list)
    activities: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    mapped_s: float = 0.0
    unmapped_s: float = 0.0

    def to_dict(self) -> dict:
        return {"status": self.status.status, "reason": self.status.reason,
                "windows": [w.to_dict() for w in self.windows],
                "mapped_s": self.mapped_s, "unmapped_s": self.unmapped_s}


def window_spans(total_samples: int, cfg: DiarConfig) -> list[tuple[int, int, int, int]]:
    """[(start, end, own_start, own_end)] in samples; owned ranges tile [0, total) exactly."""
    w, h = int(round(cfg.window_s * SR)), int(round(cfg.hop_s * SR))
    if w <= 0 or h <= 0 or h > w:
        raise ValueError("need 0 < hop_s <= window_s")
    starts = [0]
    while starts[-1] + w < total_samples:
        starts.append(starts[-1] + h)
    spans = []
    for i, s in enumerate(starts):
        e = min(s + w, total_samples)
        own_s = 0 if i == 0 else (s + (starts[i - 1] + w)) // 2
        own_e = total_samples if i == len(starts) - 1 else (starts[i + 1] + e) // 2
        spans.append((s, e, own_s, own_e))
    return spans


def _best_assignment(m: list[list[float]]) -> list[tuple[int, int]]:
    """Maximise total overlap; rows=labels, cols=sources. Hungarian via scipy, brute-force fallback."""
    nr, nc = len(m), (len(m[0]) if m else 0)
    if not nr or not nc:
        return []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-np.asarray(m, dtype=float))
        return [(int(a), int(b)) for a, b in zip(r, c)]
    except ImportError:
        pass
    best, best_v = [], -1.0
    if nr <= nc:
        for cols in permutations(range(nc), nr):
            v = sum(m[i][cols[i]] for i in range(nr))
            if v > best_v:
                best_v, best = v, list(enumerate(cols))
    else:
        for rows in permutations(range(nr), nc):
            v = sum(m[rows[j]][j] for j in range(nc))
            if v > best_v:
                best_v, best = v, [(rows[j], j) for j in range(nc)]
    return best


def map_labels_to_sources(turns, identity_activity, cfg: DiarConfig | None = None,
                          window_index: int = 0, offset_us: int = 0) -> dict[str, str]:
    """Map window-local labels to persistent source ids; unmapped labels get `diar:w<i>:<label>`.

    `turns`: [(label, start_s, end_s)] window-relative. `identity_activity`: {source_id:
    [(start_us, end_us)]} in clip time; `offset_us` shifts turns to clip time.
    """
    cfg = cfg or DiarConfig()
    by_label: dict[str, list[tuple[int, int]]] = {}
    all_labels = set()
    for lab, s, e in turns:
        all_labels.add(str(lab))
        if e - s >= cfg.min_turn_s:
            by_label.setdefault(str(lab), []).append(
                (offset_us + int(round(s * 1e6)), offset_us + int(round(e * 1e6))))
    by_label = {k: normalize(v) for k, v in by_label.items()}
    out = {lab: f"diar:w{window_index}:{lab}" for lab in sorted(all_labels)}
    labels, sources = sorted(by_label), sorted(identity_activity)
    if not labels or not sources:
        return out
    m = [[total_us(intersect(by_label[lb], identity_activity[s])) / 1e6 for s in sources] for lb in labels]
    for i, j in _best_assignment(m):
        ov = m[i][j]
        tot = total_us(by_label[labels[i]]) / 1e6
        if ov < cfg.min_overlap_s or tot <= 0 or ov / tot < cfg.map_threshold:
            continue
        if any(k != i and abs(m[k][j] - ov) < 1e-9 for k in range(len(labels))):
            continue  # tie for this source: ambiguous, leave unmapped
        out[labels[i]] = sources[j]
    return out


def diarization_activities(window_turns, mappings, spans, min_turn_s: float = 0.0
                           ) -> dict[str, list[tuple[int, int]]]:
    """Combine per-window turns into {source_or_diar_id: normalized intervals} (us), clipped to owned time.

    `window_turns[i]`: [(label, start_s, end_s)] window-relative or None (no evidence);
    `mappings[i]`: label -> id; `spans[i]`: (start, end, own_start, own_end) samples.
    """
    acc: dict[str, list[tuple[int, int]]] = {}
    for turns, mp, (s0, _e, o0, o1) in zip(window_turns, mappings, spans):
        if turns is None:
            continue
        off = s0 * 1_000_000 // SR
        own = [(o0 * 1_000_000 // SR, o1 * 1_000_000 // SR)]
        for lab, s, e in turns:
            if e - s < min_turn_s:
                continue
            iv = [(off + int(round(s * 1e6)), off + int(round(e * 1e6)))]
            clipped = intersect(iv, own)
            if clipped:
                acc.setdefault(mp.get(str(lab), f"diar:{lab}"), []).extend(clipped)
    return {k: normalize(v) for k, v in acc.items()}


def run_windows(read_audio, total_samples: int, identity_activity, pipeline=None,
                cfg: DiarConfig | None = None, mode: str = "off",
                not_attempted_reason: str = "") -> DiarResult:
    """Diarize bounded windows lazily. `mode` is Switches.diarization: off|auto|on."""
    cfg = cfg or DiarConfig()
    if mode == "off":
        return DiarResult(DiarStatus("skipped", "disabled by config"))
    if pipeline is None:
        return DiarResult(DiarStatus(
            "not_attempted", not_attempted_reason or "no pipeline (package/cache/credentials missing)"))
    spans = window_spans(total_samples, cfg)
    wstat, wturns, maps = [], [], []
    for i, (s, e, _o0, _o1) in enumerate(spans):
        try:
            turns = diarize(read_audio(s, e), SR, pipeline)
            mp = map_labels_to_sources(turns, identity_activity, cfg, i, s * 1_000_000 // SR)
        except Exception as exc:  # one window failing never aborts the run
            wstat.append(DiarStatus("failed", type(exc).__name__, i))
            wturns.append(None)
            maps.append({})
            continue
        wstat.append(DiarStatus("ok", "", i))
        wturns.append(turns)
        maps.append(mp)
    acts = diarization_activities(wturns, maps, spans, cfg.min_turn_s)
    unmapped_us = sum(total_us(v) for k, v in acts.items() if k.startswith("diar:"))
    mapped_us = sum(total_us(v) for k, v in acts.items() if not k.startswith("diar:"))
    oks = [w for w in wstat if w.status == "ok"]
    overall = (DiarStatus("ok", f"{len(oks)}/{len(wstat)} windows ok") if oks
               else DiarStatus("failed", "all windows failed"))
    return DiarResult(overall, wstat, acts, mapped_us / 1e6, unmapped_us / 1e6)


def availability(env=None, cache_dir=None, online: bool = False) -> dict:
    """Capability report without loading models; no network unless `online=True`.

    Never returns token values, only whether one is present. `gated_access` stays "not_attempted"
    unless `online` (then "ok"/"denied"/"unknown" via a metadata-only HF call).
    """
    env = os.environ if env is None else env
    import importlib.util
    try:
        package = importlib.util.find_spec("pyannote.audio") is not None
    except (ImportError, ValueError):
        package = False
    if cache_dir is None:
        hub = env.get("HF_HUB_CACHE") or os.path.join(
            env.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "hub")
    else:
        hub = str(cache_dir)
    cache = os.path.isdir(os.path.join(hub, "models--" + MODEL_ID.replace("/", "--")))
    token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    gated = "not_attempted"
    if online:
        if not token:
            gated = "denied"
        else:
            try:
                from huggingface_hub import HfApi
                HfApi().model_info(MODEL_ID, token=token)
                gated = "ok"
            except Exception as exc:
                gated = "denied" if type(exc).__name__ in ("GatedRepoError", "RepositoryNotFoundError") else "unknown"
    ready = package and (cache or gated == "ok")
    return {"package": package, "cache": cache, "token_present": bool(token),
            "gated_access": gated, "status": "ok" if ready else "not_attempted"}
