"""Drop audio-derived bearings from the published sources (demo mode).

On the Quest 3S the two microphones are centimetres apart and the audio direction can land tens of degrees off
even through the fit gates (locate/TUNING.md), so a box drawn at that bearing sits next to the person, not on them.
A position the headset shows must come from the camera: this overlay clears `bearing_deg`, `bearing_conf`,
`bearing_source` and `in_view` on every source whose bearing did not come from the camera, and leaves camera-sourced
bearings (set later by vision.person_box) alone. Register it BEFORE the person-box overlay.
"""

from __future__ import annotations

import protocol as p


def strip_audio_bearing(sources: list[p.Source], now_us: int = 0) -> list[p.Source]:
    out = []
    for s in sources:
        if s.bearing_source == p.BearingSource.camera or s.bearing_deg is None:
            out.append(s)
        else:
            out.append(s.model_copy(update={"bearing_deg": None, "bearing_conf": 0.0,
                                            "bearing_source": None, "in_view": False, "bbox": None}))
    return out
