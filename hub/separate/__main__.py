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
from track import TrackConfig, Tracker

from .live import LiveSeparator
from .models import SoundLabeler, VoiceSeparator
from .worker import SeparationWorker

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=p.DEFAULT_PORT)
ap.add_argument("--no-labels", action="store_true")
ap.add_argument("--no-speakers", action="store_true", help="disable /speakers (per-speaker audio + intervals)")
ap.add_argument("--voice-memory", action="store_true",
                help="use optional WavLM speaker embeddings to retain voice slots across pauses")
ap.add_argument("--locate-method", choices=["level", "tdoa"], default="level",
                help="direction finding: level (head shadow) or tdoa (inter-mic delay, calibrated; Quest 3S)")
ap.add_argument("--locate-ungated", action="store_true",
                help="level method without the fit gates (can report a confident wrong bearing on the Quest 3S; see locate/TUNING.md)")
ap.add_argument("--active-floor-db", type=float, default=-88.0,
                help="activity floor in band-level dB (not dBFS: the band level reads ~29 dB below the stream rms). "
                     "-88 = zero activity at a ~-59 dBFS stream, full at ~-49; the Quest mic's speech sits at -43 to -50 dBFS "
                     "(the class default -60 only suits louder simulated signals)")
ap.add_argument("--out-settle-s", type=float, default=0.3,
                help="hold separated audio this long before /audio/out so the tracker has caught up")
ap.add_argument("--out-gains-in-layers", action="store_true",
                help="hub applies inner/outer dB in the Quest 'layers' layout (default: the Quest applies them)")
ap.add_argument("--merge-ild-db", type=float, default=0.0,
                help="one-talker merge threshold in dB; 0 = off. Calibrate per device (4.5 = simulator)")
ap.add_argument("--no-person-box", action="store_true",
                help="disable the camera face box for the selected non-wearer speaker")
ap.add_argument("--camera-flipped", action="store_true",
                help="camera frames are upside-down: mirror the person-box bbox vertically")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)

state = IngestState()
worker = SeparationWorker(state, LiveSeparator(VoiceSeparator(), None if a.no_labels else SoundLabeler(),
                                               merge_ild_db=a.merge_ild_db,
                                               voice_memory=a.voice_memory,
                                               floor_db=a.active_floor_db))
try:  # direction finding (C4) if hub/locate is present
    from locate.estimator import LocateConfig
    from locate.live import LiveLocator

    # Gated by default: on the real 4-channel Quest the plain level model "locks" on a bearing it does not
    # fit (locate/TUNING.md), so withhold the bearing (unknown direction) unless the fit is good.
    gates = None if a.locate_ungated else LocateConfig(profile_diff=True, max_resid_chi2=3.0, min_fit_gain=0.05)
    locator = LiveLocator(state, cfg=gates, method=a.locate_method, mic_channels=(0, 1))
    worker.listeners.append(locator.on_chunks)
except ImportError:
    logging.getLogger("separate").exception("direction finding disabled: hub/locate failed to import")
    locator = None
worker.start()
app = create_app(state)
# Direction is usually unknown on the Quest (gated locator), so speech must be announced without a bearing.
events_state = install_events(app, locator, Tracker(TrackConfig(bearingless_speech_hits=3, bearingless_speech_per_slot=True)))
audio_out = install_audio_out(app, events_state, AudioOutConfig(
    settle_s=a.out_settle_s, apply_gains_in_layers=a.out_gains_in_layers))
worker.listeners.append(audio_out.on_chunks)
if not a.no_speakers:
    from speakers import install_speakers

    install_speakers(app, events_state, worker)

from vision.person_box import PersonBox  # noqa: E402

person_box = PersonBox(state, events_state.selection, camera_flipped=a.camera_flipped,
                       enabled=not a.no_person_box,
                       recorder_fn=lambda: getattr(app.state, "speakers", None))
from events.no_audio_bearing import strip_audio_bearing  # noqa: E402

# Demo: no audio-derived direction is ever shown (it lands tens of degrees off on the Quest 3S); a box only from the camera.
events_state.source_overlays.insert(0, strip_audio_bearing)
events_state.source_overlays.append(person_box.overlay)
person_box.start()


@app.get("/person-box/status")
def person_box_status():
    return person_box.status()


@app.get("/person-box/debug")
def person_box_debug():
    return person_box.debug()


@app.get("/sources")
def sources():
    """Separated, labeled sources with head-movement bearings (protocol Source objects)."""
    if locator is None:
        return {"error": "direction finding disabled (see hub log)", "sources": []}
    return [s.model_dump(mode="json") for s in locator.sources()]


@app.get("/ingest/levels")
def ingest_levels():
    """Per-channel level of the last 2 s of raw headset audio (dBFS), and of the plain mean the separator uses."""
    ring = state.audio
    if ring is None or ring.written < 32000:
        return {"error": "not enough audio yet"}
    got = ring.read(ring.written - 32000, 32000)
    if got is None:
        return {"error": "audio not readable"}
    x = np.asarray(got[0], dtype=np.float64)
    db = lambda v: round(float(20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-9)), 1)
    return {"channels": [db(x[:, i]) for i in range(x.shape[1])], "mean_of_all": db(x.mean(axis=1))}


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
