"""Non-speech event labelling: merge logic with a stub classifier; one real AST smoke test (CPU)."""

from pathlib import Path

import numpy as np
import pytest

from av_scene import events as ev

SR = 16000


def _stub(script):
    """script: label per call index by window start second -> classifier returning [(label, 0.9)]."""
    def clf(windows):
        out = []
        for w in windows:
            # encode the window's identity in its amplitude: label index = round(mean|w| * 10)
            k = int(round(float(np.abs(w).mean()) * 10))
            out.append([(script[k], 0.9), ("Other", 0.1)])
        return out
    return clf


def _tone(seconds, k):
    """Constant-|amplitude| signal that the stub decodes as label index k."""
    return np.full(int(seconds * SR), k / 10.0, dtype=np.float32)


def test_silence_gives_no_events():
    assert ev.detect_events(np.zeros(SR * 3, np.float32), SR, classifier=_stub({})) == []


def test_empty_audio_gives_no_events():
    assert ev.detect_events(np.zeros(0, np.float32), SR, classifier=_stub({})) == []


def test_merge_consecutive_same_label_and_split_on_change():
    x = np.concatenate([_tone(2, 1), _tone(2, 2)])
    events = ev.detect_events(x, SR, window_s=1.0, hop_s=0.5, classifier=_stub({1: "Dog", 2: "Doorbell"}))
    assert [e.labels[0].label for e in events] == ["Dog", "Doorbell"]
    assert events[0].start_us == 0
    assert events[0].audio_prompt == "dog barking"
    assert events[0].visual_noun_candidates == ["dog"]
    assert events[1].audio_prompt == "doorbell"
    assert events[0].end_us <= 2_500_000 and events[1].start_us >= 1_500_000


def test_gap_breaks_run_into_two_events():
    x = np.concatenate([_tone(2, 1), np.zeros(2 * SR, np.float32), _tone(2, 1)])
    events = ev.detect_events(x, SR, classifier=_stub({1: "Dog"}))
    assert len(events) == 2
    assert events[0].end_us <= 2_500_000 and events[1].start_us >= 2_500_000  # window-resolution bleed


def test_low_score_windows_dropped():
    clf = lambda ws: [[("Dog", 0.1)] for _ in ws]
    assert ev.detect_events(_tone(3, 1), SR, classifier=clf, min_score=0.3) == []


def test_speech_flagged_and_split_and_refused_as_source():
    x = np.concatenate([_tone(2, 1), _tone(2, 2)])
    events = ev.detect_events(x, SR, classifier=_stub({1: "Speech", 2: "Dog"}))
    assert [e.provenance["is_speech"] for e in events] == [True, False]
    assert [e.event_id for e in ev.speech_events(events)] == [events[0].event_id]
    assert [e.event_id for e in ev.non_speech_events(events)] == [events[1].event_id]
    with pytest.raises(ValueError):
        ev.to_scene_source(events[0])


def test_to_scene_source_is_unattributed():
    (e,) = ev.detect_events(_tone(2, 1), SR, classifier=_stub({1: "Dog"}))
    s = ev.to_scene_source(e)
    assert s.kind == "non_speech" and s.visibility == "unknown"
    assert s.event_ids == [e.event_id] and s.current_association_id is None
    if "extraction" in type(s).model_fields:
        assert s.extraction.status == "not_attempted"
        assert s.activity_us == [(e.start_us, e.end_us)]


def test_resamples_other_rates():
    x = _tone(2, 1)[:: 2]  # 8 kHz
    (e,) = ev.detect_events(x, 8000, classifier=_stub({1: "Dog"}))
    assert abs(e.end_us - 2_000_000) < 20_000


_AST = Path.home() / ".cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593"


@pytest.mark.skipif(not _AST.exists(), reason="AST not in HF cache")
def test_real_ast_smoke_cpu(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    rng = np.random.default_rng(0)
    x = np.zeros(SR * 6, np.float32)
    x[2 * SR:4 * SR] = (0.3 * rng.standard_normal(2 * SR)).astype(np.float32)
    events = ev.detect_events(x, SR, device="cpu", min_score=0.05)
    assert events, "noise burst should produce at least one event"
    assert all(0 <= e.start_us < e.end_us <= 6_000_000 for e in events)
    assert all(e.start_us >= 1_000_000 and e.end_us <= 5_000_000 for e in events)
    assert all(e.audio_prompt and e.audio_prompt == e.audio_prompt.lower() for e in events)
