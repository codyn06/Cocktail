"""Mock headset client (task B4): simulated sessions and real-time replay into the hub."""

from .session import Session
from .talkers import TalkerSpan, active_at, frame_truth, load_talkers

__all__ = ["Session", "TalkerSpan", "active_at", "frame_truth", "load_talkers"]
