---
name: cocktail-perception
description: Use when working on Cocktail perception - SAM3 visual segmentation, SAM Audio source separation, sound localization, audio cleanup, audio/visual timing alignment, or associating sounds with visual objects. Covers verification of model APIs, checkpoints, hardware, and data formats.
---

# Cocktail Perception

Project-authored skill, not an official vendor skill.

## Name the task precisely

These are different problems; do not conflate them:
- **Visual segmentation** (SAM3): masks for objects in images/video from text, exemplar, point, or box prompts.
- **Audio separation** (SAM Audio): splitting a mixture into a target and residual from a text, visual, or time-span prompt.
- **Sound localization**: where a sound comes from (direction/position). Needs multi-channel audio or a dedicated method; neither model above provides it by itself.
- **Cleanup**: denoising/enhancement of a single stream.

A visual mask alone never proves which object made a sound, or where it is in 3D. Say what evidence supports any association (visual prompt to SAM Audio, timing correlation, multi-mic data) and how strong it is.

## Before writing code

1. Check what is installed (package versions, commit) and read the real example code for that version. Treat `references/*.md` as starting points that may be stale.
2. Check checkpoint access: the checkpoints are gated on Hugging Face. Do not request access, log in, or download checkpoints without the user's explicit approval.
3. Check hardware: GPU/CUDA availability, memory, and Python/PyTorch versions against the model's stated requirements.

## Track data formats explicitly

- Audio: sample rate (resample to the processor's expected rate), channel count and layout, dtype/range, and clip length.
- Time: timestamps and clock source for every stream; offsets between glasses, phone, and any server.
- Images: dimensions, orientation, color order, and every coordinate transform between capture, model input, and display.

## Measure and label

- Measure latency on the hardware actually available; state the hardware and input size.
- Label each result: hardware-tested/live (real device stream), replayed (recorded data), mocked (synthetic or stubbed inputs), or untested.
- Report measured behavior separately from expectations.

## References (load on demand)

- `references/sam3.md` - SAM3 setup and API notes.
- `references/sam-audio.md` - SAM Audio setup and API notes.
