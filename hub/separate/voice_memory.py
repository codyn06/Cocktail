"""Optional speaker identity memory for the two MossFormer voice streams.

The separator's stream order is still decided by overlap first. After a long
quiet interval, this module uses normalized speaker embeddings to map returning
voices back to their previous stream. Low-confidence matches preserve the
separator's order instead of forcing a speculative swap.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

SR = 16000


class SpeakerEncoder(Protocol):
    def __call__(self, audios: list[np.ndarray]) -> np.ndarray:
        """Return one L2-normalized speaker vector per 16 kHz mono waveform."""


@dataclass
class VoiceMemoryConfig:
    pause_s: float = 1.0
    profile_min_s: float = 2.0
    profile_update_s: float = 3.0
    history_s: float = 8.0
    min_active_dbfs: float = -55.0
    match_threshold: float = 0.86
    match_margin: float = 0.08
    profile_alpha: float = 0.15


class VoiceMemory:
    """Keep a small running speaker profile per logical voice slot.

    `reorder` receives the two separated 16 kHz mono waveforms for a full
    2-second separator window. It returns a permutation from raw output slots
    to the stable logical slots. Embeddings run only on initial/update samples
    and after a >=pause_s interruption; quiet frames are omitted.
    """

    def __init__(self, encoder: SpeakerEncoder, config: VoiceMemoryConfig | None = None,
                 hop_samples: int = SR):
        self.encoder = encoder
        self.config = config or VoiceMemoryConfig()
        self.hop_samples = hop_samples
        self.profiles: list[np.ndarray | None] = [None, None]
        self._history: list[deque[np.ndarray]] = [deque(), deque()]
        self._history_samples = [0, 0]
        self._last_active_us: list[int | None] = [None, None]
        self._silent_s = [0.0, 0.0]
        self._learning_blocked = [False, False]
        self._raw_to_slot = [0, 1]
        self._profile_active_s = [0.0, 0.0]
        self._next_update_s = [self.config.profile_min_s, self.config.profile_min_s]
        self.last: dict = {}
        self.trace: deque[dict] = deque(maxlen=600)

    def reset(self) -> None:
        self.profiles = [None, None]
        self._history = [deque(), deque()]
        self._history_samples = [0, 0]
        self._last_active_us = [None, None]
        self._silent_s = [0.0, 0.0]
        self._learning_blocked = [False, False]
        self._raw_to_slot = [0, 1]
        self._profile_active_s = [0.0, 0.0]
        self._next_update_s = [self.config.profile_min_s, self.config.profile_min_s]
        self.last = {}
        self.trace.clear()

    def _active_excerpt(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        frame = SR // 10
        audio = np.asarray(audio, np.float32).reshape(-1)
        n = len(audio) // frame
        if n == 0:
            return np.empty(0, np.float32), 0.0
        frames = audio[: n * frame].reshape(n, frame)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        peak = float(rms.max(initial=0))
        floor = max(10 ** (self.config.min_active_dbfs / 20), peak * 0.01)
        active = rms >= floor
        excerpt = frames[active].reshape(-1).copy()
        return excerpt, float(active.sum()) / 10.0

    def _encode(self, waveforms: list[np.ndarray]) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        vectors = np.asarray(self.encoder(waveforms), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        return vectors, (time.perf_counter() - start) * 1000

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def reorder(self, voices: np.ndarray, start_ts_us: int) -> tuple[np.ndarray, dict]:
        """Assign active raw outputs to stable speaker slots and return reordered voices."""
        if voices.ndim not in (2, 3) or voices.shape[0] != 2:
            raise ValueError("voice memory expects two mono outputs or two multi-channel outputs")
        mono = voices if voices.ndim == 2 else voices[:, 0]
        now_us = int(start_ts_us + self.hop_samples * 1_000_000 // SR)
        hop = mono[:, : self.hop_samples]
        rms = np.sqrt(np.mean(hop * hop, axis=1) + 1e-12)
        floor = max(10 ** (self.config.min_active_dbfs / 20), float(rms.max(initial=0)) * 0.0316)
        active = [i for i in range(2) if rms[i] >= floor]
        # Silence belongs to raw separator slots. A previously unused slot also
        # has enough silence when a returning source lands there after a swap.
        paused = any(self._silent_s[i] >= self.config.pause_s for i in active)
        returning = bool(active and paused)
        diagnostic = {"device_ts_us": now_us, "rms_dbfs": [round(float(20 * np.log10(x + 1e-12)), 1)
                       for x in rms], "returning": returning, "scores": {}, "mapping": {},
                      "embedding_ms": 0.0, "reason": "overlap"}
        old_map = self._raw_to_slot.copy()
        mapping = {i: old_map[i] for i in range(2)}
        fresh_vectors: dict[int, np.ndarray] = {}
        assigned_inputs: set[int] = set()

        if returning:
            excerpts = {}
            for i in active:
                excerpt, duration = self._active_excerpt(mono[i])
                if duration >= 0.5:
                    excerpts[i] = excerpt
            if excerpts:
                candidate_ids = list(excerpts)
                try:
                    vectors, ms = self._encode([excerpts[i] for i in candidate_ids])
                    diagnostic["embedding_ms"] = round(ms, 2)
                    fresh_vectors = {i: vectors[k] for k, i in enumerate(candidate_ids)}
                except Exception as exc:  # optional identity must not stop separation
                    diagnostic["embedding_error"] = f"{type(exc).__name__}: {exc}"
                scores = {i: {slot: round(self._cosine(vec, profile), 4)
                              for slot, profile in enumerate(self.profiles) if profile is not None}
                          for i, vec in fresh_vectors.items()}
                diagnostic["scores"] = scores
                used: set[int] = set()
                # Assign only distinctive matches. Similarity thresholds depend
                # on speakers, microphones and utterance length, so uncertainty
                # falls back to current overlap ordering.
                confident = []
                for i, row in scores.items():
                    rank = sorted(row.items(), key=lambda item: item[1], reverse=True)
                    if rank and rank[0][1] >= self.config.match_threshold and (
                            len(rank) == 1 or rank[0][1] - rank[1][1] >= self.config.match_margin):
                        confident.append((rank[0][1], i, rank[0][0]))
                for _, i, slot in sorted(confident, reverse=True):
                    if slot not in used:
                        mapping[i] = slot
                        used.add(slot)
                        assigned_inputs.add(i)
                # A clearly new speaker takes an unused, unprofiled slot. Never
                # overwrite a known speaker just because the match was weak.
                for i in active:
                    if i in fresh_vectors and i not in assigned_inputs:
                        free = next((slot for slot, profile in enumerate(self.profiles)
                                     if profile is None and slot not in used), None)
                        if free is not None:
                            mapping[i] = free
                            used.add(free)
                            assigned_inputs.add(i)
                            self.profiles[free] = fresh_vectors[i]
                            self._next_update_s[free] = self.config.profile_update_s
                            diagnostic["reason"] = "new-speaker"
                if confident:
                    diagnostic["reason"] = "matched"
        # An uncertain match can collide with a confident match when both raw
        # inputs are active. Keep the confident assignment and put the other
        # input in the remaining slot without changing either profile.
        reserved = set()
        for i in active:
            if i in assigned_inputs:
                reserved.add(mapping[i])
        for i in active:
            if i in assigned_inputs:
                continue
            if mapping[i] in reserved:
                mapping[i] = next(slot for slot in range(2) if slot not in reserved)
            reserved.add(mapping[i])

        if returning:
            for i in active:
                # Apply this only after collision repair so an uncertain input
                # cannot block or learn into another input's confident profile.
                self._learning_blocked[mapping[i]] = i not in assigned_inputs

        # Complete a one-to-one permutation for inactive channels too.
        targets = set(mapping[i] for i in active)
        free_targets = [slot for slot in range(2) if slot not in targets]
        for i in range(2):
            if i not in active:
                mapping[i] = free_targets.pop(0) if free_targets else i
        result = np.empty_like(voices)
        for source, target in mapping.items():
            result[target] = voices[source]

        for source in range(2):
            self._silent_s[source] = 0.0 if source in active else \
                self._silent_s[source] + self.hop_samples / SR

        # Learn/update profiles from distinct emitted hops only (not the
        # overlapping lookahead window) and only from active speech samples.
        pending: list[tuple[int, np.ndarray]] = []
        for slot in range(2):
            src = next((i for i, target in mapping.items() if target == slot), None)
            if src is None or src not in active:
                continue
            first = (result[slot, : self.hop_samples] if result.ndim == 2 else
                     result[slot, 0, : self.hop_samples]).astype(np.float32, copy=True)
            excerpt, duration = self._active_excerpt(first)
            if duration <= 0:
                continue
            self._last_active_us[slot] = now_us
            if self._learning_blocked[slot]:
                continue
            self._profile_active_s[slot] += duration
            self._history[slot].append(excerpt)
            self._history_samples[slot] += len(excerpt)
            while self._history_samples[slot] > int(self.config.history_s * SR):
                old = self._history[slot].popleft()
                self._history_samples[slot] -= len(old)
            if self.profiles[slot] is None and self._profile_active_s[slot] >= self.config.profile_min_s:
                pending.append((slot, np.concatenate(self._history[slot])))
            elif self.profiles[slot] is not None and self._profile_active_s[slot] >= self._next_update_s[slot]:
                pending.append((slot, np.concatenate(self._history[slot])))

        if pending:
            try:
                vectors, ms = self._encode([wave for _, wave in pending])
                diagnostic["embedding_ms"] = round(diagnostic["embedding_ms"] + ms, 2)
                for k, (slot, _) in enumerate(pending):
                    if self.profiles[slot] is None:
                        self.profiles[slot] = vectors[k]
                        self._next_update_s[slot] = self._profile_active_s[slot] + self.config.profile_update_s
                    else:
                        mixed = (1 - self.config.profile_alpha) * self.profiles[slot] + \
                                self.config.profile_alpha * vectors[k]
                        self.profiles[slot] = mixed / (np.linalg.norm(mixed) + 1e-12)
                        self._next_update_s[slot] = self._profile_active_s[slot] + self.config.profile_update_s
            except Exception as exc:  # optional identity must not stop separation
                diagnostic["embedding_error"] = f"{type(exc).__name__}: {exc}"

        diagnostic["mapping"] = {f"s{src + 1}": f"s{dst + 1}" for src, dst in mapping.items()}
        diagnostic["slot_mapping"] = mapping.copy()
        diagnostic["logical_permutation"] = {old_map[src]: mapping[src] for src in range(2)}
        self._raw_to_slot = [mapping[i] for i in range(2)]
        diagnostic["profile_pair_similarity"] = (None if any(p is None for p in self.profiles) else
                                                 round(self._cosine(self.profiles[0], self.profiles[1]), 4))
        self.last = diagnostic
        self.trace.append(diagnostic.copy())
        return result, diagnostic


class WavLMSpeakerEncoder:
    """Microsoft WavLM Base Plus SV encoder; requires the optional GPU group.

    The checkpoint is loaded only when voice memory is explicitly enabled. The
    Hugging Face card currently links to a license whose scope for weights is
    unclear; do not redistribute the checkpoint with this project.
    """

    def __init__(self, device: str = "cuda", model_id: str = "microsoft/wavlm-base-plus-sv",
                 use_bf16: bool = True):
        import torch
        from transformers import AutoModelForAudioXVector, Wav2Vec2FeatureExtractor

        self.device = torch.device(device)
        self.dtype = (torch.bfloat16 if use_bf16 and self.device.type == "cuda" and
                      torch.cuda.is_bf16_supported() else torch.float32)
        # The model card's detailed speaker-vector example uses this feature
        # extractor. AutoProcessor in newer Transformers also probes optional
        # chat-template metadata, which is unrelated to audio and can stall
        # offline runs even when the model and feature config are cached.
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioXVector.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(self.device).eval()
        self._torch = torch

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, np.float32).reshape(-1)
        rms = np.sqrt(np.mean(x * x) + 1e-12)
        if rms > 1e-6:
            x = x * min(0.1 / rms, 20.0)
        return np.clip(x, -1, 1).astype(np.float32, copy=False)

    def __call__(self, audios: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        batch = self.processor([self._normalize(x) for x in audios], sampling_rate=SR,
                               padding=True, return_tensors="pt")
        batch = {k: v.to(self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                 for k, v in batch.items()}
        with torch.inference_mode():
            vectors = self.model(**batch).embeddings.float()
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        return vectors.cpu().numpy()
