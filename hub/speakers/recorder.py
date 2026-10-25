"""Per-speaker recorder: tracker id -> timeline-aligned audio segments + speaking intervals.

Attribution rules (slot = separator voice stream "s1"/"s2"; residual "s3" is never recorded):
  1. A voice hop (stream in voice_slots, label "speech") is attributed to track T only if T is
     confirmed and `tracking`, T is the slot's current stream (or `_dup_of[slot] == T`), and no
     other track was heard in that slot more recently (same rule as events.selection.Selection.route).
  2. Exactly one candidate -> attribute. Several -> ambiguous: drop the hop and clear the slot's pending.
  3. No candidate (track not yet confirmed, paused/lost, slot claimed by someone else) -> the hop is held
     in a per-slot pending buffer (<= 5 hops). When the slot later resolves to T, pending hops are
     flushed to T only if hop.start >= T.first_us - 4 s AND the slot did not previously resolve to a
     different track. Otherwise they are dropped. Recorded audio is never moved between speakers.
  4. Epoch / ring reset clears everything (tracker ids restart history there). Ids themselves are never
     reused by the tracker, and speaker numbers are 1-based by first attributed audio, never reassigned.

Intervals come from the same chunks' `activity` frames (>ACTIVE_MIN active), gaps < 300 ms merged,
islands < 150 ms dropped. Memory: <= MAX_SPEAKERS speakers x RING_S seconds of audio (and a span cap).
"""

from __future__ import annotations

import io
import logging
import struct
import threading
from collections import deque

import numpy as np

log = logging.getLogger("speakers")

SR = 16000
ACTIVE_MIN = 0.2          # same constant as locate/live.py ACTIVE_MIN
MERGE_GAP_US = 300_000
MIN_ISLAND_US = 150_000
SPEAKING_US = 600_000
FRAME_US = 50_000
MAX_SPEAKERS = 8
RING_S = 120
MAX_SPAN_S = 300          # WAV time axis cap (gaps are zero-filled)
MAX_PENDING = 5
MAX_INTERVALS = 200
PRE_ROLL_US = 4_000_000  # the tracker stamps a track at delivery, ~2.5 s after a hop's capture (window + hop + compute)


class _Hop:
    __slots__ = ("start_us", "audio", "ts", "act")

    def __init__(self, start_us, audio, ts, act):
        self.start_us, self.audio, self.ts, self.act = start_us, audio, ts, act


class _Speaker:
    def __init__(self, sid: str, number: int):
        self.sid, self.number = sid, number
        self.segs: list[list] = []        # [start_us, [arrays], nsamples]
        self.samples = 0
        self.end_us: int | None = None    # end of the newest stored hop
        self.intervals: deque = deque(maxlen=MAX_INTERVALS)   # (start, end) finalized
        self.cur: list[int] | None = None  # [start, end] run being built
        self.dropped = 0

    # audio ---------------------------------------------------------------
    def add_audio(self, start_us: int, audio: np.ndarray) -> bool:
        if self.end_us is not None and start_us < self.end_us - 125:
            self.dropped += 1                    # overlapping/out-of-order: never rewrite the past
            return False
        n = len(audio)
        end = start_us + n * 1_000_000 // SR
        if self.segs and abs(start_us - self.end_us) <= 125:
            self.segs[-1][1].append(audio)
            self.segs[-1][2] += n
        else:
            self.segs.append([start_us, [audio], n])
        self.samples += n
        self.end_us = end
        self._evict()
        return True

    def _evict(self):
        while self.segs and (self.samples > RING_S * SR or
                             (self.end_us - self.segs[0][0]) > MAX_SPAN_S * 1_000_000):
            seg = self.segs[0]
            first = seg[1].pop(0)
            seg[2] -= len(first)
            self.samples -= len(first)
            seg[0] += len(first) * 1_000_000 // SR
            if not seg[1]:
                self.segs.pop(0)

    def audio_start_us(self):
        return self.segs[0][0] if self.segs else None

    def wav(self) -> bytes:
        if not self.segs:
            pcm = np.zeros(0, np.int16)
        else:
            t0 = self.segs[0][0]
            total = int(round((self.end_us - t0) * SR / 1e6))
            buf = np.zeros(max(total, 0), np.float32)
            for start, arrs, _n in self.segs:
                off = int(round((start - t0) * SR / 1e6))
                a = np.concatenate(arrs)
                piece = a[:max(len(buf) - off, 0)]
                buf[off:off + len(piece)] = piece
            pcm = (np.clip(buf, -1, 1) * 32767).astype(np.int16)
        data = pcm.tobytes()
        out = io.BytesIO()
        out.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt ")
        out.write(struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16))
        out.write(b"data" + struct.pack("<I", len(data)) + data)
        return out.getvalue()

    # intervals -----------------------------------------------------------
    def add_activity(self, ts: np.ndarray, act: np.ndarray):
        for t, a in zip(ts, act):
            if a <= ACTIVE_MIN:
                continue
            s, e = int(t) - FRAME_US // 2, int(t) + FRAME_US // 2
            if self.cur is not None and s - self.cur[1] < MERGE_GAP_US:
                self.cur[1] = max(self.cur[1], e)
            else:
                self._finalize()
                self.cur = [s, e]

    def _finalize(self):
        if self.cur is not None and self.cur[1] - self.cur[0] >= MIN_ISLAND_US:
            self.intervals.append((self.cur[0], self.cur[1]))
        self.cur = None

    def advance(self, now_us: int):
        if self.cur is not None and now_us - self.cur[1] >= SPEAKING_US:
            self._finalize()

    def speaking(self, now_us: int) -> bool:
        c = self.cur
        return c is not None and c[1] - c[0] >= MIN_ISLAND_US and now_us - c[1] <= SPEAKING_US

    def interval_list(self, last: int | None = None) -> list[dict]:
        out = [{"start_us": s, "end_us": e, "open": False} for s, e in self.intervals]
        if self.cur is not None and self.cur[1] - self.cur[0] >= MIN_ISLAND_US:
            out.append({"start_us": self.cur[0], "end_us": self.cur[1], "open": True})
        return out[-last:] if last else out


class SpeakerRecorder:
    def __init__(self, tracker_fn, epoch_fn=None, voice_slots=("s1", "s2")):
        self._tracker_fn = tracker_fn
        self._epoch_fn = epoch_fn
        self.voice_slots = tuple(voice_slots)
        self._lock = threading.Lock()
        self._epoch = None
        self.errors = 0
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def _reset_locked(self):
        self.speakers: dict[str, _Speaker] = {}
        self._pending: dict[str, list[_Hop]] = {s: [] for s in self.voice_slots}
        self._last_tid: dict[str, str | None] = {s: None for s in self.voice_slots}
        self.now_us = 0
        self.hops_seen = self.hops_attributed = self.hops_dropped = 0
        self._next_number = 1

    # ------------------------------------------------------------------ attribution
    @staticmethod
    def _candidates(tracker, slot: str) -> list:
        tracks = tracker.tracks

        def ok(t):
            return (t.confirmed and str(getattr(t.state, "value", t.state)) == "tracking"
                    and not any(o is not t and o.stream == slot and o.last_obs_us > t.last_obs_us
                                for o in tracks.values()))
        cands = [t for t in tracks.values() if t.stream == slot and ok(t)]
        dup = tracks.get(getattr(tracker, "_dup_of", {}).get(slot, ""))
        if dup is not None and dup not in cands and ok(dup):
            cands.append(dup)
        return cands

    def on_chunks(self, chunks) -> None:
        try:
            with self._lock:
                self._on_chunks(chunks)
        except Exception:
            self.errors += 1
            log.exception("speaker recorder failed")

    def _on_chunks(self, chunks):
        if self._epoch_fn is not None:
            ep = self._epoch_fn()
            if self._epoch is not None and ep != self._epoch:
                self._reset_locked()
            self._epoch = ep
        tracker = self._tracker_fn()
        for ch in chunks:
            if ch.stream_id not in self.voice_slots or ch.label != "speech":
                continue
            self.hops_seen += 1
            n = len(ch.audio)
            self.now_us = max(self.now_us, int(ch.start_ts_us) + n * 1_000_000 // SR)
            hop = _Hop(int(ch.start_ts_us), np.asarray(ch.audio, np.float32).copy(),
                       np.asarray(ch.level_ts_us).copy(), np.asarray(ch.activity).copy())
            slot = ch.stream_id
            cands = self._candidates(tracker, slot)
            if len(cands) > 1:
                self.hops_dropped += len(self._pending[slot]) + 1
                self._pending[slot] = []
            elif not cands:
                pend = self._pending[slot]
                pend.append(hop)
                if len(pend) > MAX_PENDING:
                    pend.pop(0)
                    self.hops_dropped += 1
            else:
                self._attribute(slot, cands[0], hop)
        for sp in self.speakers.values():
            sp.advance(self.now_us)

    def _attribute(self, slot, t, hop):
        tid = t.tid
        pend, self._pending[slot] = self._pending[slot], []
        prev = self._last_tid[slot]
        self._last_tid[slot] = tid
        sp = self.speakers.get(tid)
        if sp is None:
            if len(self.speakers) >= MAX_SPEAKERS:
                self.hops_dropped += len(pend) + 1
                return
            sp = self.speakers[tid] = _Speaker(tid, self._next_number)
            self._next_number += 1
        flush_ok = prev in (None, tid)
        for h in pend:
            if flush_ok and h.start_us >= t.first_us - PRE_ROLL_US:
                self._store(sp, h)
            else:
                self.hops_dropped += 1
                sp.dropped += 1
        self._store(sp, hop)

    def _store(self, sp, h):
        if sp.add_audio(h.start_us, h.audio):
            sp.add_activity(h.ts, h.act)
            self.hops_attributed += 1
        else:
            self.hops_dropped += 1

    # ------------------------------------------------------------------ queries
    def info(self, sid: str) -> dict | None:
        with self._lock:
            sp = self.speakers.get(sid)
            if sp is None:
                return None
            return {"number": sp.number, "speaking": sp.speaking(self.now_us),
                    "audio_start_ts_us": sp.audio_start_us(), "audio_s": round(sp.samples / SR, 3),
                    "intervals": sp.interval_list(20), "dropped_hops": sp.dropped}

    def ids(self) -> list[str]:
        with self._lock:
            return sorted(self.speakers, key=lambda k: self.speakers[k].number)

    def intervals(self, sid: str):
        with self._lock:
            sp = self.speakers.get(sid)
            return None if sp is None else sp.interval_list()

    def wav_bytes(self, sid: str) -> bytes | None:
        with self._lock:
            sp = self.speakers.get(sid)
            return None if sp is None else sp.wav()

    def status(self) -> dict:
        with self._lock:
            return {"hops_seen": self.hops_seen, "hops_attributed": self.hops_attributed,
                    "hops_dropped": self.hops_dropped,
                    "pending": sum(len(v) for v in self._pending.values()),
                    "errors": self.errors, "speakers": len(self.speakers)}
