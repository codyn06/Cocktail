"""Live hub with background speech-refinement evidence. Run from hub/: `uv run --group gpu python -m refine`.

Same composition as `python -m separate` (flags included) plus refinement: GET /refinement/status and
target.updated on /events. Refinement never changes selection, route or /audio/out audio.

  --refine                    enable selected-voice refinement (default OFF: measured to add output underruns, see docs/STATUS.md)
  --no-refine                 explicit baseline (the default)
  --refine-device cuda        device for the refinement models (loaded in a background thread)
  --refine-window-s 4.0       audio/video window per job
  --refine-refresh-s 4.0      re-run for the same selection after this long
  --refine-fake {none,accept,refuse,slow}   built-in fake runner, no models (slow sleeps 20 s)
"""

import argparse
import logging
import time

import numpy as np
import uvicorn

import protocol as p
from ingest import IngestState, create_app
from events import install_events
from out import AudioOutConfig, install_audio_out
from separate.live import LiveSeparator
from separate.models import SoundLabeler, VoiceSeparator
from separate.worker import SeparationWorker

from .contracts import RefineConfig, RefineOutcome, RefineState
from .install import install_refinement

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
ap.add_argument("--refine", action="store_true", help="enable refinement (default off)")
ap.add_argument("--no-refine", action="store_true", help="baseline; this is the default")
ap.add_argument("--refine-device", default="cuda")
ap.add_argument("--refine-window-s", type=float, default=4.0)
ap.add_argument("--refine-refresh-s", type=float, default=4.0)
ap.add_argument("--refine-fake", choices=["none", "accept", "refuse", "slow"], default="none",
                help="built-in fake runner for wiring smoke tests without models (slow sleeps 20 s)")
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
    logging.getLogger("refine").exception("direction finding disabled: hub/locate failed to import")
    locator = None
worker.start()
app = create_app(state)
events_state = install_events(app, locator)
audio_out = install_audio_out(app, events_state, AudioOutConfig(
    settle_s=a.out_settle_s, apply_gains_in_layers=a.out_gains_in_layers))
worker.listeners.append(audio_out.on_chunks)

if a.refine and not a.no_refine:
    cfg = RefineConfig(window_s=a.refine_window_s, refresh_s=a.refine_refresh_s)

    def fake_runner(job):
        if a.refine_fake == "slow":
            time.sleep(20)
        if a.refine_fake == "accept":
            return RefineOutcome(RefineState.accepted, "fake accept", track_id="fake", score=10.0, assoc_conf=0.5)
        return RefineOutcome(RefineState.refused, "fake refuse")

    def runner_factory():
        if a.refine_fake != "none":
            return fake_runner
        from .speech_refiner import load_speech_refiner
        return load_speech_refiner(a.refine_device, cfg)

    install_refinement(app, events_state, state, worker, runner_factory=runner_factory, config=cfg)


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
