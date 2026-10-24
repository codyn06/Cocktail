"""Hub ingest + live separation. Run from hub/: `uv run --group gpu python -m separate`.

GET /separate/status shows the newest hop per stream (label, level, activity) and compute time.
"""

import argparse
import logging

import numpy as np
import uvicorn

import protocol as p
from ingest import IngestState, create_app

from .live import LiveSeparator
from .models import SoundLabeler, VoiceSeparator
from .worker import SeparationWorker

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=p.DEFAULT_PORT)
ap.add_argument("--no-labels", action="store_true")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)

state = IngestState()
worker = SeparationWorker(state, LiveSeparator(VoiceSeparator(), None if a.no_labels else SoundLabeler())).start()
app = create_app(state)


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
