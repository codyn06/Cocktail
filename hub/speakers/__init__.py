"""Per-speaker identity, speaking intervals and isolated audio, keyed by tracker id."""

from .recorder import SpeakerRecorder
from .server import install_speakers

__all__ = ["SpeakerRecorder", "install_speakers"]
