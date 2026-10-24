# SAM Audio notes

Source: https://github.com/facebookresearch/sam-audio

Status: as-of-research claims. Verify every item against the installed version and its
README/examples before relying on it.

- Purpose: prompted audio source separation. Prompts: text description, visual
  (video frames/masks), or time spans.
- Requirements (claimed): Python >= 3.11; GPU recommended.
- Checkpoints: gated on Hugging Face (e.g. `facebook/sam-audio-large`). Access request
  and download need user approval.
- API (claimed): `SAMAudio.from_pretrained(...)`, `SAMAudioProcessor`,
  `model.separate(...)` returning a result with `.target` and `.residual`.
- Sample rate: read `processor.audio_sampling_rate`; resample input to it and record it
  with outputs.

Check: channel handling (mono vs multi-channel), maximum clip length, alignment of
visual prompts to audio timestamps, and latency per second of audio on available
hardware. Separation output is not localization: it does not tell you where a source is.
