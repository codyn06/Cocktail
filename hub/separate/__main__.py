"""Hub ingest + live separation. Run from hub/: `uv run --group gpu python -m separate`.

GET /separate/status shows the newest hop per stream (label, level, activity) and compute time.
"""

import argparse
import logging

import numpy as np
import uvicorn

import protocol as p
from ingest import IngestState, create_app
from events import install_events
from out import AudioOutConfig, install_audio_out

from .live import LiveSeparator
from .models import SoundLabeler, VoiceSeparator
from .worker import SeparationWorker

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=p.DEFAULT_PORT)
ap.add_argument("--no-labels", action="store_true")
ap.add_argument("--voice-memory", action="store_true",
                help="use optional WavLM speaker embeddings to retain voice slots across pauses")
ap.add_argument("--locate-method", choices=["level", "tdoa"], default="level",
                help="direction finding: level (head shadow) or tdoa (inter-mic delay, calibrated; Quest 3S)")
ap.add_argument("--out-settle-s", type=float, default=0.3,
                help="hold separated audio this long before /audio/out so the tracker has caught up")
ap.add_argument("--out-gains-in-layers", action="store_true",
                help="hub applies inner/outer dB in the Quest 'layers' layout (default: the Quest applies them)")
ap.add_argument("--merge-ild-db", type=float, default=0.0,
                help="one-talker merge threshold in dB; 0 = off. Calibrate per device (4.5 = simulator)")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)

state = IngestState()
worker = SeparationWorker(state, LiveSeparator(VoiceSeparator(), None if a.no_labels else SoundLabeler(),
                                               merge_ild_db=a.merge_ild_db,
                                               voice_memory=a.voice_memory))
try:  # direction finding (C4) if hub/locate is present
    from locate.live import LiveLocator

    locator = LiveLocator(state, method=a.locate_method)
    worker.listeners.append(locator.on_chunks)
except ImportError:
    logging.getLogger("separate").exception("direction finding disabled: hub/locate failed to import")
    locator = None
worker.start()
app = create_app(state)
events_state = install_events(app, locator)
audio_out = install_audio_out(app, events_state, AudioOutConfig(
    settle_s=a.out_settle_s, apply_gains_in_layers=a.out_gains_in_layers))
worker.listeners.append(audio_out.on_chunks)


@app.get("/sources")
def sources():
    """Separated, labeled sources with head-movement bearings (protocol Source objects)."""
    if locator is None:
        return {"error": "direction finding disabled (see hub log)", "sources": []}
    return [s.model_dump(mode="json") for s in locator.sources()]


@app.get("/separate/status")
def status():
    last = worker.chunks[-1] if worker.chunks else []
    ms = list(worker.compute_ms)
    return {
        "hops_done": len(ms),
        "lag_s": round(worker.lag_s, 2),
        "skips": worker.skips,
        "listener_errors": worker.listener_errors,
        "gates": worker.live.last_gates,  # one-voice merge and speech gate on the latest window
        "compute_ms_p50": float(np.median(ms)) if ms else None,
        "compute_ms_max": float(np.max(ms)) if ms else None,
        "streams": [{"id": c.stream_id, "label": c.label, "label_conf": round(c.label_conf, 3),
                     "top_scores": {k: round(v, 3) for k, v in c.scores.items()},
                     "start_ts_us": c.start_ts_us,
                     "level_db": round(float(10 * np.log10(np.mean(c.audio ** 2) + 1e-10)), 1),
                     "activity_mean": round(float(c.activity.mean()), 2)} for c in last],
    }


uvicorn.run(app, host=a.host, port=a.port)
