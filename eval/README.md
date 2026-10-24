# eval/ - separation scoring harness

Evidence label for everything produced here: **replayed** (offline mixtures streamed
chunk-by-chunk through the model on the local GPU). Not live audio, not on-device.

## Run (from `hub/`)

```
uv sync                                   # project-local venv (Python 3.12, torch cu124)
# one-time: LibriSpeech dev-clean (public, 337 MB) -> eval/data/raw/
curl -L -o ../eval/data/raw/dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz
uv run python ../eval/prepare_data.py     # builds 9 controlled 20 s mixtures in eval/data/
uv run python ../eval/bench.py --models convtasnet-libri2mix-16k sepformer-whamr16k
```

Mixtures: two LibriSpeech speakers at 0 dB and +5 dB relative level, and the 0 dB pair plus
a GTZAN music clip 5 dB below the speech. Three speaker pairs. Output: `eval/results.json`.

## What is measured

- SI-SDRi (permutation resolved once over the whole stream, first window skipped as warm-up).
- Per-hop compute (ms, mean/p95/max, CUDA-synchronised), RTF = mean compute / hop.
- Algorithmic latency = hop + lookahead; e2e p95 = that + p95 compute.
- Peak VRAM of the process (PyTorch allocator, excludes CUDA context).

Modes (`hub/separate/streaming.py:MODES`): fast (1.5 s window, 20 ms hop, 0 look),
balanced (2 s, 50 ms hop, 100 ms look), accurate (6 s, 1 s hop, 2 s look).

Add a model: register a loader returning `(fn, n_spk, info)` in `hub/separate/models.py`.
