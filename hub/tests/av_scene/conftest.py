"""Import `av_scene` first: it registers the FFmpeg shared DLLs before torchcodec/sam_audio are imported."""

import av_scene  # noqa: F401
