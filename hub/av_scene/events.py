"""Non-speech sound-event labelling for the Phase 2 pipeline.

Labels what is audible over time (AudioSet classes from AST). It says nothing about WHICH visible
object made a sound: `visual_noun_candidates` is only a hand-written candidate list for a later
association stage, not attribution. Speech-class windows are flagged (`provenance["is_speech"]`)
so the speech branch can take them; `to_scene_source` refuses them.
"""

from __future__ import annotations

from typing import Callable, Iterator, Sequence

import numpy as np

from .schemas import AcousticEvent, LabelScore, SceneSource

MODEL_SR = 16000
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SILENCE_RMS = 1e-3          # windows quieter than this (~-60 dBFS) are not classified
DEFAULT_MIN_SCORE = 0.3     # AST sigmoid probability

# Classifier: list of mono 16 kHz windows -> per window, [(label, score), ...] sorted descending.
Classifier = Callable[[Sequence[np.ndarray]], "list[list[tuple[str, float]]]"]

SPEECH_LABELS = frozenset({
    "speech", "male speech, man speaking", "female speech, woman speaking",
    "child speech, kid speaking", "conversation", "narration, monologue", "babbling",
    "speech synthesizer", "whispering", "shout", "yell",
})
IGNORED_LABELS = frozenset({"silence"})

# AudioSet label (lowercase) -> SAM Audio text prompt. Others fall back to the first lowercased name.
_PROMPTS = {
    "dog": "dog barking", "bark": "dog barking", "bow-wow": "dog barking",
    "cat": "cat meowing", "meow": "cat meowing", "purr": "cat purring",
    "speech": "person speaking", "doorbell": "doorbell", "ding-dong": "doorbell", "knock": "knocking",
    "telephone bell ringing": "phone ringing", "ringtone": "phone ringing", "telephone": "phone ringing",
    "smoke detector, smoke alarm": "smoke alarm", "typing": "keyboard typing",
    "computer keyboard": "keyboard typing", "bird vocalization, bird call, bird song": "bird chirping",
    "bird": "bird chirping", "water": "running water",
}
# Candidate visible nouns only; keys are substrings of the lowercase AudioSet label.
_NOUNS = {
    "dog": ["dog"], "bark": ["dog"], "bow-wow": ["dog"], "cat": ["cat"], "meow": ["cat"], "purr": ["cat"],
    "bird": ["bird"], "speech": ["person"], "conversation": ["person"], "laughter": ["person"],
    "clapping": ["person"], "applause": ["person"], "footsteps": ["person"],
    "doorbell": ["door"], "knock": ["door"], "door": ["door"],
    "telephone": ["phone"], "ringtone": ["phone"], "typing": ["keyboard", "laptop", "person"],
    "keyboard": ["keyboard", "laptop"], "guitar": ["guitar"], "piano": ["piano"], "drum": ["drum"],
    "violin": ["violin"], "music": ["speaker", "guitar", "piano"], "car": ["car"], "vehicle": ["car"],
    "engine": ["car"], "siren": ["car"], "alarm": ["clock", "phone"], "water": ["faucet"],
}


def audio_prompt_for(label: str) -> str:
    key = label.lower()
    return _PROMPTS.get(key) or key.split(",")[0].strip()


def visual_nouns_for(label: str) -> list[str]:
    key = label.lower()
    out: list[str] = []
    for k, nouns in _NOUNS.items():
        if k in key:
            out += [n for n in nouns if n not in out]
    return out


def is_speech_label(label: str) -> bool:
    return label.lower() in SPEECH_LABELS


def _to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:  # (samples, channels) or (channels, samples): the smaller axis is channels
        x = x.mean(axis=1 if x.shape[1] < x.shape[0] else 0)
    if sr != MODEL_SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), MODEL_SR)
        x = resample_poly(x, MODEL_SR // g, int(sr) // g).astype(np.float32)
    return x


def ast_classifier(device: str = "cpu", top_k: int = 3) -> Classifier:
    """AST (AudioSet) via the existing `separate.models.SoundLabeler`; loads from the HF cache."""
    from separate.models import SoundLabeler

    lab = SoundLabeler(device=device)

    def run(windows):
        out = []
        for p in lab.probs(list(windows)):
            idx = np.argsort(p)[::-1][:top_k + 1]
            out.append([(lab.names[i], float(p[i])) for i in idx])
        return out

    return run


def detect_events(audio: np.ndarray, sr: int, window_s: float = 1.0, hop_s: float = 0.5, top_k: int = 3,
                  min_score: float = DEFAULT_MIN_SCORE, classifier: Classifier | None = None,
                  device: str = "cpu", batch: int = 16) -> list[AcousticEvent]:
    """Windowed classification; consecutive windows with the same top label merge into one event.

    Times are microseconds from sample 0 of `audio`. Silent windows and windows whose top score is
    below `min_score` break a run. `classifier` defaults to AST (lazy load)."""
    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window_s and hop_s must be positive")
    x = _to_mono_16k(audio, sr)
    win, hop = int(round(window_s * MODEL_SR)), int(round(hop_s * MODEL_SR))
    if len(x) == 0:
        return []
    starts = list(range(0, len(x) - win + 1, hop)) if len(x) >= win else [0]
    keep = [s for s in starts if float(np.sqrt(np.mean(x[s:s + win] ** 2))) >= SILENCE_RMS]
    if not keep:
        return []
    clf = classifier or ast_classifier(device, top_k)
    per: dict[int, list[tuple[str, float]]] = {}
    for i in range(0, len(keep), batch):
        chunk = keep[i:i + batch]
        res = clf([x[s:s + win] for s in chunk])
        for s, r in zip(chunk, res):
            per[s] = [(l, float(sc)) for l, sc in r if l.lower() not in IGNORED_LABELS][:top_k]

    runs: list[dict | None] = []
    for s in starts:
        r = per.get(s)
        if not r or r[0][1] < min_score:
            runs.append(None)
            continue
        cur = runs[-1] if runs else None
        if cur and cur["label"] == r[0][0]:
            cur["end"] = s + win
            cur["wins"].append(r)
        else:
            runs.append({"label": r[0][0], "start": s, "end": s + win, "wins": [r]})

    events: list[AcousticEvent] = []
    for run in (r for r in runs if r):
        label = run["label"]
        agg: dict[str, list[float]] = {}
        for w in run["wins"]:
            for l, sc in w:
                agg.setdefault(l, []).append(sc)
        labels = sorted(((l, float(np.mean(v))) for l, v in agg.items()), key=lambda t: -t[1])[:top_k]
        events.append(AcousticEvent(
            event_id=f"ev{len(events):03d}",
            labels=[LabelScore(label=l, score=s) for l, s in labels],
            start_us=int(round(run["start"] / MODEL_SR * 1e6)),
            end_us=int(round(min(run["end"], len(x)) / MODEL_SR * 1e6)),
            activity_score=float(np.mean([w[0][1] for w in run["wins"]])),
            audio_prompt=audio_prompt_for(label),
            visual_noun_candidates=visual_nouns_for(label),
            provenance={"model": MODEL_ID if classifier is None else "custom",
                        "is_speech": is_speech_label(label), "n_windows": len(run["wins"]),
                        "window_s": window_s, "hop_s": hop_s, "attribution": "none: candidates only"},
        ))
    return events


def _event_from_run(run: dict, index: int, n_samples: int, top_k: int, window_s: float, hop_s: float,
                    custom: bool) -> AcousticEvent:
    label = run["label"]
    agg: dict[str, list[float]] = {}
    for w in run["wins"]:
        for l, sc in w:
            agg.setdefault(l, []).append(sc)
    labels = sorted(((l, float(np.mean(v))) for l, v in agg.items()), key=lambda t: -t[1])[:top_k]
    return AcousticEvent(
        event_id=f"ev{index:03d}",
        labels=[LabelScore(label=l, score=s) for l, s in labels],
        start_us=int(round(run["start"] / MODEL_SR * 1e6)),
        end_us=int(round(min(run["end"], n_samples) / MODEL_SR * 1e6)),
        activity_score=float(np.mean([w[0][1] for w in run["wins"]])),
        audio_prompt=audio_prompt_for(label),
        visual_noun_candidates=visual_nouns_for(label),
        provenance={"model": "custom" if custom else MODEL_ID,
                    "is_speech": is_speech_label(label), "n_windows": len(run["wins"]),
                    "window_s": window_s, "hop_s": hop_s, "attribution": "none: candidates only"},
    )


def iter_detect_events(read_block: Callable[[float, float], np.ndarray], duration_s: float,
                       classifier: Classifier | None = None, *, block_s: float = 30.0, window_s: float = 1.0,
                       hop_s: float = 0.5, top_k: int = 3, min_score: float = DEFAULT_MIN_SCORE,
                       device: str = "cpu", batch: int = 16) -> Iterator[AcousticEvent]:
    """Streaming `detect_events`: yields the SAME events as `detect_events(full_array)` without ever holding
    the full clip.

    `read_block(start_s, end_s)` returns mono 16 kHz float32 for that span and is called lazily, in order,
    for consecutive blocks of `block_s`. Resident audio is at most one block plus the unconsumed tail
    (< one window) carried across the block edge, so an event (or a classifier window) that straddles a
    block boundary is classified from contiguous samples. Only the single open run is kept between
    windows; a run is emitted when a silent/low-score window or a different label closes it. The window
    grid is anchored at sample 0 of the clip (starts at k * hop), exactly as in `detect_events`."""
    if window_s <= 0 or hop_s <= 0 or block_s <= 0:
        raise ValueError("window_s, hop_s and block_s must be positive")
    n = int(round(duration_s * MODEL_SR))
    if n <= 0:
        return
    win, hop = int(round(window_s * MODEL_SR)), int(round(hop_s * MODEL_SR))
    blk = max(1, int(round(block_s * MODEL_SR)))
    starts_total = list(range(0, n - win + 1, hop)) if n >= win else [0]
    clf = classifier or ast_classifier(device, top_k)
    custom = classifier is not None

    buf = np.zeros(0, np.float32)
    buf_off = 0           # absolute sample index of buf[0]
    read_to = 0           # absolute sample index read so far

    def ensure(end: int) -> None:
        nonlocal buf, read_to
        end = min(end, n)
        while read_to < end:
            hi = min(read_to + blk, n)
            x = np.asarray(read_block(read_to / MODEL_SR, hi / MODEL_SR), np.float32).reshape(-1)
            if x.size != hi - read_to:                 # tolerate off-by-rounding, never silently mis-align
                x = np.pad(x[:hi - read_to], (0, max(0, hi - read_to - x.size)))
            buf = np.concatenate([buf, x])
            read_to = hi

    def window(s: int) -> np.ndarray:
        nonlocal buf, buf_off
        ensure(s + win)
        if s > buf_off:                                # drop what is behind the window grid
            buf = buf[s - buf_off:]
            buf_off = s
        return buf[:min(win, n - s)] if n >= win else buf[:n]

    cur: dict | None = None
    n_events = 0
    pending: list[tuple[int, np.ndarray]] = []

    def classify(items: list[tuple[int, np.ndarray]]) -> dict[int, list[tuple[str, float]]]:
        res = clf([w for _, w in items])
        return {s: [(l, float(sc)) for l, sc in r if l.lower() not in IGNORED_LABELS][:top_k]
                for (s, _), r in zip(items, res)}

    def step(s: int, per: list[tuple[str, float]] | None):
        nonlocal cur, n_events
        if not per or per[0][1] < min_score:
            done, cur = cur, None
            return done
        if cur and cur["label"] == per[0][0]:
            cur["end"] = s + win
            cur["wins"].append(per)
            return None
        done = cur
        cur = {"label": per[0][0], "start": s, "end": s + win, "wins": [per]}
        return done

    def flush(items, order):
        """Classify buffered windows, then advance the run state machine over `order` (starts, in order)."""
        nonlocal n_events
        per = classify(items) if items else {}
        for s in order:
            done = step(s, per.get(s))
            if done:
                yield _event_from_run(done, n_events, n, top_k, window_s, hop_s, custom)
                n_events += 1

    order: list[int] = []
    for s in starts_total:
        w = window(s)
        order.append(s)
        if float(np.sqrt(np.mean(w ** 2))) >= SILENCE_RMS:
            pending.append((s, w.copy()))
        if len(pending) >= batch:
            yield from flush(pending, order)
            pending, order = [], []
    yield from flush(pending, order)
    if cur:
        yield _event_from_run(cur, n_events, n, top_k, window_s, hop_s, custom)


def speech_events(events: Sequence[AcousticEvent]) -> list[AcousticEvent]:
    return [e for e in events if e.provenance.get("is_speech")]


def non_speech_events(events: Sequence[AcousticEvent]) -> list[AcousticEvent]:
    return [e for e in events if not e.provenance.get("is_speech")]


def to_scene_source(event: AcousticEvent) -> SceneSource:
    """Non-speech event -> candidate SceneSource: no association, visibility unknown, extraction not attempted."""
    if event.provenance.get("is_speech"):
        raise ValueError(f"{event.event_id} is a speech event; route it to the speech branch")
    top = event.labels[0] if event.labels else None
    kw: dict = dict(source_id=f"src_{event.event_id}", kind="non_speech",
                    label=event.audio_prompt or (top.label if top else ""),
                    event_ids=[event.event_id], visibility="unknown")
    fields = SceneSource.model_fields
    # Newer schema fields (label_confidence / activity_us / extraction) may not exist on older bases.
    if "label_confidence" in fields:
        kw["label_confidence"] = top.score if top else None
    if "activity_us" in fields:
        kw["activity_us"] = [(event.start_us, event.end_us)]
    if "extraction" in fields:
        from .schemas import Extraction
        kw["extraction"] = Extraction(status="not_attempted", reason="label only; no separation run")
    return SceneSource(**kw)
