"""Live tracker events and command handling, without model or device dependencies."""

from .server import EventsState, install_events

__all__ = ["EventsState", "install_events"]
