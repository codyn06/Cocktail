import numpy as np
import pytest

import protocol as p
from ingest import IngestState
from separate import BAND_CENTERS_HZ, LiveSeparator, SeparationWorker
from separate.score import sisdr
from separate.voice_memory import VoiceMemory, VoiceMemoryConfig

SR = 16000
rng = np.random.default_rng(0)
N = SR * 8
V1 = rng.standard_normal(N).astype(np.float32) * np.repeat(rng.random(N // 1600) > 0.3, 1600)
V2 = np.sin(2 * np.pi * 440 * np.arange(N) / SR).astype(np.float32) * 0.5
MUSIC = np.sin(2 * np.pi * 3000 * np.arange(N) / SR).astype(np.float32) * 0.1
MIX = V1 + V2 + MUSIC


class OracleSeparator:
    """Returns the true voices for the window it is given, in a random order and scale."""
    n_out = 2

    def __init__(self):
        self.pos = 0
        self.flip = np.random.default_rng(1)

    def __call__(self, x):
        i = self.pos
        self.pos += SR  # windows advance by 1 s
        v = np.stack([V1[i : i + len(x)] * 3.0, V2[i : i + len(x)] * 0.5])
        return v[::-1] if self.flip.random() < 0.5 else v


def run(live, x):
    out = {sid: [] for sid in live.ids}
    chunks = []
    for st in range(0, len(x) - live.W + 1, live.H):
        cs = live.step(x[st : st + live.W], st * 1_000_000 // SR)
        chunks.append(cs)
        for c in cs:
            out[c.stream_id].append(c.audio)
    return {k: np.concatenate(v) for k, v in out.items()}, chunks


def test_overlap_add_keeps_order_and_scale():
    x = np.stack([MIX, MIX], axis=1)  # no direction information: direction gate off
    est, chunks = run(LiveSeparator(OracleSeparator(), merge_ild_db=0), x)
    n = len(est["s1"])
    assert n == 7 * SR  # 7 windows of 2 s over 8 s, one 1 s hop each
    # Stream identity stays fixed despite random order from the separator.
    first = "s1" if np.dot(est["s1"][:SR], V1[:SR]) > np.dot(est["s2"][:SR], V1[:SR]) else "s2"
    other = "s2" if first == "s1" else "s1"
    assert sisdr(est[first], V1[:n]) > 30 and sisdr(est[other], V2[:n]) > 30
    assert sisdr(est["s3"], MUSIC[:n]) > 30  # residual is the music


def test_chunk_fields_for_locate():
    x = np.stack([MIX, 0.5 * MIX], axis=1)  # right mic 6 dB quieter
    _, chunks = run(LiveSeparator(OracleSeparator()), x)
    c = chunks[3][0]
    assert c.mic_audio.shape == (2, SR) and c.audio.shape == (SR,)
    assert c.levels_db.shape == (len(c.level_ts_us), 2, len(BAND_CENTERS_HZ))
    assert c.start_ts_us == 3_000_000 and np.all(np.diff(c.level_ts_us) == 50_000)
    assert abs(np.median(c.levels_db[:, 0] - c.levels_db[:, 1]) - 6.02) < 0.5
    assert c.activity.shape == c.level_ts_us.shape and 0 <= c.activity.min() <= c.activity.max() <= 1


def test_worker_follows_ingest_ring():
    st = IngestState()
    worker = SeparationWorker(st, LiveSeparator(OracleSeparator()))
    assert not worker.poll()
    frames = SR // 50  # 20 ms chunks at 16 kHz
    got = []
    worker.listeners.append(got.append)
    for k in range(SR * 4 // frames):
        seg = MIX[k * frames : (k + 1) * frames]
        h = p.AudioChunk(seq=k, device_ts_us=k * 20_000, epoch=0, sample_rate=SR, channels=1, frames=frames)
        st.on_audio(h, (seg / 4 * 32767).astype("<i2").tobytes())
        while worker.poll():
            pass
    assert len(got) == 3  # windows ending at 2, 3 and 4 s
    assert [c[0].start_ts_us for c in got] == [0, 1_000_000, 2_000_000]


@pytest.mark.skipif(not __import__("importlib").util.find_spec("transformers"), reason="needs gpu group")
def test_coarse_labels():
    from separate.models import coarse_label
    assert coarse_label("Male speech, man speaking") == "speech"
    assert coarse_label("Smoke detector, smoke alarm") == "alarm"
    assert coarse_label("Violin, fiddle") == "music"


def _feed(st, worker, sig, k0=0):
    frames = SR // 50
    for k in range(len(sig) // frames):
        h = p.AudioChunk(seq=k0 + k, device_ts_us=(k0 + k) * 20_000, epoch=0, sample_rate=SR, channels=1,
                         frames=frames)
        st.on_audio(h, (sig[k * frames : (k + 1) * frames] / 4 * 32767).astype("<i2").tobytes())
        while worker.poll():
            pass


def test_listener_error_does_not_repeat_window():
    st = IngestState()
    worker = SeparationWorker(st, LiveSeparator(OracleSeparator()))
    seen = []

    def bad(chunks):
        seen.append(chunks[0].start_ts_us)
        raise ValueError("boom")
    worker.listeners.append(bad)
    _feed(st, worker, MIX[: SR * 4])
    assert seen == [0, 1_000_000, 2_000_000] and worker.listener_errors == 3


def test_silence_has_no_activity():
    class Quiet:
        n_out = 2

        def __call__(self, x):
            return np.stack([x * 0.5, x * 0.5])
    live = LiveSeparator(Quiet())
    x = (rng.standard_normal((2 * SR, 1)) * 1e-4).astype(np.float32)  # -80 dBFS hiss
    for c in live.step(x, 0):
        assert c.activity.max() == 0


def test_worker_notices_ring_reset():
    st = IngestState()
    worker = SeparationWorker(st, LiveSeparator(OracleSeparator()))
    _feed(st, worker, MIX[: SR * 3])
    st.audio.reset()  # e.g. a reconnect, while the worker holds an old position
    worker.live.sep.pos = 0
    _feed(st, worker, MIX[: SR * 3], k0=1000)
    assert len(worker.chunks) == 4 and worker.chunks[-1][0].start_ts_us == 1000 * 20_000 + 1_000_000


class SplitOracle:
    """Splits one source into two 'voices' by alternating 100 ms pieces, like MossFormer2 does
    with a single talker."""
    n_out = 2

    def __call__(self, x):
        mask = (np.arange(len(x)) // 1600) % 2 == 0
        return np.stack([x * mask, x * ~mask])


def test_direction_gate_merges_one_source_split_in_two():
    rng2 = np.random.default_rng(3)
    v = rng2.standard_normal(SR * 4).astype(np.float32) * 0.1
    x = np.stack([v, 0.3 * v], axis=1)  # one source on the left: same level difference everywhere
    live = LiveSeparator(SplitOracle(), merge_ild_db=4.5)
    est, _ = run(live, x)
    loud, quiet = sorted([est["s1"], est["s2"]], key=lambda e: -np.sum(e**2))
    assert live.last_gates["merged"] and np.sum(quiet**2) == 0
    assert sisdr(loud, v[: len(loud)] * 0.65) > 20  # the whole voice is back in one stream


def test_direction_gate_keeps_two_sources_from_different_directions():
    rng2 = np.random.default_rng(4)
    a = rng2.standard_normal(SR * 4).astype(np.float32) * 0.1
    b = rng2.standard_normal(SR * 4).astype(np.float32) * 0.1

    class TwoOracle:
        n_out = 2
        pos = 0

        def __call__(self, x):
            i = self.pos
            self.pos += SR
            return np.stack([0.65 * a[i : i + len(x)], 0.65 * b[i : i + len(x)]])

    x = np.stack([a + 0.3 * b, 0.3 * a + b], axis=1)  # a on the left, b on the right
    live = LiveSeparator(TwoOracle(), merge_ild_db=4.5)
    run(live, x)
    assert not live.last_gates["merged"] and live.last_gates["ild_diff_db"] > live.merge_ild_db



def test_merged_voice_stays_in_one_stream_across_a_pause():
    """One talker, a pause, then the same direction again: every window merges, and the voice
    must stay in one stream (identity across pauses with different people is D1's job)."""
    rng2 = np.random.default_rng(5)
    a = rng2.standard_normal(SR * 3).astype(np.float32) * 0.1
    left = np.concatenate([a, np.zeros(SR, np.float32), a])
    x = np.stack([left, 0.3 * left], axis=1)
    est, _ = run(LiveSeparator(SplitOracle(), merge_ild_db=4.5), x)
    e = {k: np.sum(est[k] ** 2) for k in ("s1", "s2")}
    assert min(e.values()) == 0


def test_ild_diff_is_per_band_average_not_sum():
    """Same 6 dB level difference in every band gives 6 dB (an average, not 3 x 6). Two halves
    of one talker in different bands, each with the same 3 dB ILD, give about 0."""
    f = np.linspace(0, 8000, 257)
    T = 10
    E = np.ones((2, 2, len(f), T))
    E[0, 1] *= 10 ** (6 / 10)                      # voice 1: right mic +6 dB in every band
    assert abs(LiveSeparator._ild_diff(E, f) - 6.0) < 1e-6
    E = np.full((2, 2, len(f), T), 1e-12)
    lo, hi = (f >= 1000) & (f < 2000), (f >= 4000) & (f < 8000)
    for v, band in ((0, lo), (1, hi)):             # one talker split by frequency region
        E[v, 0, band] = 1.0
        E[v, 1, band] = 10 ** (3 / 10)
    E[:, 0, lo | hi] += 1e-3                       # each half leaks a little into the other band
    E[:, 1, lo | hi] += 1e-3 * 10 ** (3 / 10)      # with the same 3 dB ILD
    assert LiveSeparator._ild_diff(E, f) < 1.0


def test_gate_is_off_by_default():
    assert LiveSeparator(OracleSeparator(), win_s=2.0).merge_ild_db == 0


def test_faint_leftover_does_not_pull_a_voice_into_the_empty_slot():
    """Live bug: one talker alone (slot 2 holds only a faint copy of them), then a second talker
    starts. The first talker must stay in their slot."""
    rng2 = np.random.default_rng(7)
    you = rng2.standard_normal(SR * 6).astype(np.float32) * 0.1
    other = rng2.standard_normal(SR * 6).astype(np.float32) * 0.03
    other[: SR * 3] = 0

    class Live:
        n_out = 2
        pos = 0

        def __call__(self, x):
            i = self.pos
            self.pos += SR
            y, o = you[i : i + len(x)], other[i : i + len(x)]
            if not o.any():                      # alone: slot 1 = you, slot 2 = faint copy of you
                return np.stack([y, 1e-3 * y])
            return np.stack([o, y])              # the separator now puts you second

    x = np.stack([you + other, you + other], axis=1)
    est, _ = run(LiveSeparator(Live(), win_s=2.0), x)
    n = len(est["s1"])
    assert sisdr(est["s1"][: 3 * SR], you[: 3 * SR]) > 20
    assert sisdr(est["s1"][3 * SR : n], you[3 * SR : n]) > 10   # still you after the onset


def test_voice_memory_splits_turn_taking_speakers_after_pause():
    """Baseline bug: a two-output separator can put both turn-taking speakers in slot one."""
    n = 18 * SR
    a = np.sin(2 * np.pi * 300 * np.arange(n) / SR).astype(np.float32) * 0.1
    b = np.sin(2 * np.pi * 700 * np.arange(n) / SR).astype(np.float32) * 0.1
    a[5 * SR : 13 * SR] = 0
    b[:7 * SR] = 0
    b[12 * SR :] = 0

    class TurnTakingOracle:
        n_out = 2
        pos = 0

        def __call__(self, _mix):
            i = self.pos
            self.pos += SR
            # MossFormer can return one turn-taking person in the same raw
            # output slot. Voice memory must assign identities across the gap.
            return np.stack([a[i : i + 2 * SR] + b[i : i + 2 * SR],
                             np.zeros(2 * SR, np.float32)])

    class ToneEncoder:
        def __call__(self, audios):
            vectors = []
            for audio in audios:
                freq = np.fft.rfftfreq(len(audio), 1 / SR)
                power = np.abs(np.fft.rfft(audio))
                peak = freq[np.argmax(power)]
                vectors.append([1.0, 0.0] if peak < 500 else [0.0, 1.0])
            return np.asarray(vectors, np.float32)

    mix = a + b
    audio = np.stack([mix, mix], axis=1)
    live = LiveSeparator(TurnTakingOracle(), voice_memory=True, speaker_encoder=ToneEncoder())
    outputs, _ = run(live, audio)
    assignment = {}
    for name, reference, intervals in [('man', a, [(0, 5), (13, 18)]),
                                       ('woman', b, [(7, 12)])]:
        scores = {sid: sum(np.sum(outputs[sid][start * SR : end * SR] ** 2)
                           for start, end in intervals) for sid in outputs}
        assignment[name] = max(scores, key=scores.get)
    assert assignment['man'] != assignment['woman'], assignment
    for name, intervals in [('man', [(0, 5), (13, 18)]), ('woman', [(7, 12)])]:
        expected = assignment[name]
        for start, end in intervals:
            for sec in range(start, end):
                hop_scores = {sid: np.sum(outputs[sid][sec * SR : (sec + 1) * SR] ** 2)
                              for sid in outputs}
                assert max(hop_scores, key=hop_scores.get) == expected, (name, sec, hop_scores)
    assert live.last_gates.get('voice_memory', {}).get('mapping')


def test_voice_memory_encoder_failure_does_not_stop_separation():
    class FailingEncoder:
        def __call__(self, _audios):
            raise RuntimeError("model unavailable")

    class SpeechOracle:
        n_out = 2

        def __call__(self, _mix):
            x = np.ones(2 * SR, np.float32) * 0.05
            return np.stack([x, np.zeros_like(x)])

    live = LiveSeparator(SpeechOracle(), voice_memory=True, speaker_encoder=FailingEncoder())
    audio = np.ones((live.W + 3 * live.H, 2), np.float32) * 0.05
    outputs, _ = run(live, audio)
    assert len(outputs["s1"]) > 0
    assert "RuntimeError: model unavailable" in live.last_gates["voice_memory"].get("embedding_error", "")


def test_voice_memory_does_not_learn_uncertain_voice_into_confident_profile():
    class ProfileEncoder:
        def __call__(self, audios):
            result = []
            for audio in audios:
                rms = np.sqrt(np.mean(np.asarray(audio) ** 2))
                result.append([1.0, 0.0] if rms < 0.15 else [0.7, 0.71414286])
            return np.asarray(result, np.float32)

    memory = VoiceMemory(ProfileEncoder(), VoiceMemoryConfig(profile_min_s=0.1,
                                                              profile_update_s=0.1), hop_samples=SR)
    memory.profiles = [np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)]
    memory._raw_to_slot = [1, 0]
    memory._silent_s = [2.0, 2.0]
    memory._profile_active_s = [1.0, 1.0]
    memory._next_update_s = [0.1, 0.1]
    voices = np.stack([np.full(2 * SR, 0.1, np.float32),
                       np.full(2 * SR, 0.2, np.float32)])

    _, diagnostic = memory.reorder(voices, 0)

    assert diagnostic["slot_mapping"] == {0: 0, 1: 1}
    assert memory._learning_blocked == [False, True]
    assert np.allclose(memory.profiles[1], [0.0, 1.0])
