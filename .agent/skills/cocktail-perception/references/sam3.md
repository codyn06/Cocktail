# SAM3 notes

Source: https://github.com/facebookresearch/sam3

Status: as-of-research claims. Verify every item against the installed version and its
README/examples before relying on it.

- Purpose: promptable concept segmentation for images and video. Prompts: text,
  image exemplars, points, boxes.
- Requirements (claimed): Python >= 3.12, PyTorch >= 2.7, CUDA-capable GPU.
- Checkpoints: gated on Hugging Face. Access request and download need user approval.
- Image API (claimed): `build_sam3_image_model()`, then `Sam3Processor` with
  `set_image(...)` and `set_text_prompt(...)` to get masks/boxes/scores.
- Video API (claimed): `build_sam3_video_predictor()` and `handle_request(...)`
  for session-based tracking.

Check: output mask resolution vs input image size, coordinate conventions for boxes and
points, frame indexing/timestamps for video, and per-frame latency on available hardware.
