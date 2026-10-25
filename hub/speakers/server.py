"""HTTP routes for per-speaker data on the hub app (not part of the wire protocol).

Reset: the recorder clears itself when the pose epoch (events.ingest.pose_epoch) or the audio ring
reset count changes, matching EventsState._sync_epoch -> tracker.reset (tracker ids restart history).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from .recorder import SpeakerRecorder

BEARING_MIN_CONF = 0.25


def install_speakers(app: FastAPI, events, worker) -> SpeakerRecorder:
    def epoch():
        return (events.ingest.pose_epoch, getattr(worker, "_resets", None))

    rec = SpeakerRecorder(lambda: events.tracker, epoch)
    app.state.speakers = rec
    worker.listeners.append(rec.on_chunks)

    @app.get("/speakers/status")
    def speakers_status():
        return rec.status()

    @app.get("/speakers")
    def speakers():
        srcs = {s.source_id: s for s in events.latest.sources}
        tracks = events.tracker.tracks
        out = []
        for sid in rec.ids():
            info = rec.info(sid)
            if info is None:
                continue
            t = tracks.get(sid)
            state = str(getattr(t.state, "value", t.state)) if t is not None else "unknown"
            s = srcs.get(sid)
            bearing = conf = None
            if (s is not None and s.bearing_deg is not None and s.bearing_conf >= BEARING_MIN_CONF
                    and str(getattr(s.state, "value", s.state)) in ("tracking", "last_seen")
                    and state in ("tracking", "last_seen")):
                bearing, conf = s.bearing_deg, s.bearing_conf
            out.append({"speaker_id": sid, "number": info["number"], "label": f"Speaker {info['number']}",
                        "state": state, "speaking": info["speaking"], "bearing_deg": bearing,
                        "bearing_conf": conf, "selected": events.selection.chosen == sid,
                        "audio_start_ts_us": info["audio_start_ts_us"], "audio_s": info["audio_s"],
                        "intervals": info["intervals"], "dropped_hops": info["dropped_hops"]})
        return out

    @app.get("/speakers/{speaker_id}/audio.wav")
    def speaker_audio(speaker_id: str):
        data = rec.wav_bytes(speaker_id)
        if data is None:
            raise HTTPException(404, "unknown speaker")
        return Response(data, media_type="audio/wav")

    @app.get("/speakers/{speaker_id}/intervals")
    def speaker_intervals(speaker_id: str):
        iv = rec.intervals(speaker_id)
        if iv is None:
            raise HTTPException(404, "unknown speaker")
        return iv

    return rec
