"""Autonomous audio-visual sound discovery: contracts, cache and the Phase 1 deterministic pipeline."""

# torchcodec binds its native FFmpeg libraries when it is first imported, and `import sam_audio` imports it.
# The shared-DLL directory must therefore be registered before anything else touches either package.
from .preflight import configure_ffmpeg_dlls as _configure_ffmpeg_dlls

_configure_ffmpeg_dlls()
