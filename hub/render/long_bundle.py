"""Incremental, paged bundle writer for long recordings (Phase 4).

The Phase 1-3 `render.viewer.build_bundle` assembles one file per stem/mask track from full-run arrays. For
30-300 s (and longer) runs the browser must not load that, and the writer must not hold it. `LongBundleWriter`
takes one piece at a time (an audio segment, a timeline segment, a frame, a mask record) and keeps only
bounded buffers, so peak memory is independent of run length.

Layout (all paths relative to `out_dir`, UTF-8 JSON, written atomically via tmp + replace):

    manifest.json                       RunManifest; `long_run` set, `assets` lists the index files
    viewer.html  assets/viewer.js  assets/viewer.css
    audio/index.json                    AudioIndex (true source ids; sanitized dir names are only paths)
    audio/<safe_id>/seg_<start_sample:012d>.wav     float32 mono, one file per segment
    timeline/index.json                 TimelineIndex (sources + pages)
    timeline/page_NNNN.json             TimelinePageDoc, segments grouped by start_us // page span
    frames/index.json                   frame pages, per-track mask pages, frame_transform
    frames/page_NNNN.json               {"first_index", "pts_us": [...], "files": [...]}
    frames/<name>.png
    masks/<safe_track>/page_NNNN.jsonl  one MaskRecord per line

Spans of a source without an audio segment are SILENT: nothing is stored for them and the viewer zero-fills.
Audio segment `start_sample` is in that source's own `rate_hz`; timeline segment samples are at the timeline
sample rate. Both are convertible to time without ambiguity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
from pathlib import Path

import numpy as np

from av_scene.schemas import (LONG_RUN_SCHEMA_VERSION, AudioIndex, AudioSegmentRef, AudioSourceEntry, LongRunInfo,
                              TimelineIndex, TimelinePage, TimelinePageDoc, TimelineSegment, TimelineSource)
from ingest.media_io import read_wav, write_png, write_wav
from vision.mask_store import MaskRecord

ASSET_DIR = Path(__file__).parent / "assets"
_ZERO_CHUNK = 1 << 16


# ---------------------------------------------------------------- helpers
def safe_name(identifier: str) -> str:
    """File-system-safe, collision-resistant directory name for an arbitrary id such as `tse:track_1`.

    The true id lives only in the index; when sanitizing changed anything a short hash of the original is
    appended so `a:b` and `a_b` cannot collide. Windows-reserved names and leading dots are neutralised.
    """
    s = re.sub(r"[^A-Za-z0-9._-]", "_", identifier).lstrip(".") or "src"
    s = s[:80]
    if s != identifier or s.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "LPT1", "LPT2"}:
        s = f"{s}-{hashlib.sha1(identifier.encode('utf-8')).hexdigest()[:8]}"
    return s


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class StreamingWavWriter:
    """Mono/multichannel float32 WAV written block by block; the header sizes are patched on close.

    Writes to `<path>.tmp` and replaces `path` on a clean close, so a crash never leaves a truncated wav under
    the final name. RIFF sizes are 32-bit: raises if the data would exceed 4 GiB.
    """

    def __init__(self, path: str | Path, rate: int, channels: int = 1):
        self.path = Path(path)
        self.rate, self.channels = int(rate), int(channels)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self._f = open(self._tmp, "wb")
        self._f.write(b"\0" * 44)
        self.frames = 0
        self._closed = False

    def write(self, block: np.ndarray) -> None:
        a = np.ascontiguousarray(block, dtype="<f4")
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {a.shape[1]}")
        if 44 + (self.frames + a.shape[0]) * self.channels * 4 >= 2 ** 32:
            raise ValueError("StreamingWavWriter: data exceeds the 4 GiB RIFF limit")
        self._f.write(a.tobytes())
        self.frames += a.shape[0]

    def write_zeros(self, n: int) -> None:
        left = int(n)
        while left > 0:
            m = min(left, _ZERO_CHUNK)
            self.write(np.zeros((m, self.channels), dtype="<f4"))
            left -= m

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        data_bytes = self.frames * self.channels * 4
        block_align = self.channels * 4
        header = (b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE" + b"fmt " +
                  struct.pack("<IHHIIHH", 16, 3, self.channels, self.rate, self.rate * block_align, block_align, 32) +
                  b"data" + struct.pack("<I", data_bytes))
        self._f.seek(0)
        self._f.write(header)
        self._f.close()
        os.replace(self._tmp, self.path)

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            self._f.close()
            self._tmp.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.abort() if exc_type else self.close()


# ---------------------------------------------------------------- writer
class LongBundleWriter:
    def __init__(self, out_dir: str | Path, manifest, *, page_span_s: float = 30.0):
        if page_span_s <= 0:
            raise ValueError("page_span_s must be positive")
        self.out = Path(out_dir)
        self.manifest = manifest
        self.page_span_us = int(round(page_span_s * 1e6))
        self.out.mkdir(parents=True, exist_ok=True)
        self._audio: dict[str, AudioSourceEntry] = {}
        self._declared_len: dict[str, int] = {}
        self._sources: dict[str, TimelineSource] = {}
        # open timeline pages: page_no -> list[TimelineSegment]; flushed when two pages behind the newest
        self._tl_open: dict[int, list[TimelineSegment]] = {}
        self._tl_max_end: dict[int, int] = {}
        self._tl_pages: dict[int, TimelinePage] = {}
        self._src_span: dict[str, tuple[int, int]] = {}
        self._fr_open: dict[int, tuple[list[int], list[str]]] = {}
        self._fr_pages: dict[int, dict] = {}
        self._fr_count = 0
        self._fr_first: dict[int, int] = {}
        self._frame_size: tuple[int, int] | None = None
        self._mask_open: dict[str, dict[int, list[MaskRecord]]] = {}
        self._mask_pages: dict[str, dict[int, dict]] = {}
        self._mask_size: tuple[int, int] | None = None
        self._finalized = False

    # ------------------------------------------------------------ registry
    def register_source(self, source_id: str, kind: str, role: str, label: str = "") -> None:
        cur = self._sources.get(source_id)
        if cur is not None and (cur.kind, cur.role) != (kind, role):
            raise ValueError(f"source {source_id!r} already registered as {cur.kind}/{cur.role}")
        self._sources[source_id] = TimelineSource(source_id=source_id, kind=kind, role=role,
                                                  label=label or (cur.label if cur else ""))

    def set_source_length(self, source_id: str, n_samples: int) -> None:
        """Declare the full logical length (in the source's own samples); the default is the last segment end."""
        self._declared_len[source_id] = int(n_samples)

    # ------------------------------------------------------------ audio
    def add_audio_segment(self, source_id: str, kind: str, role: str, rate_hz: int, nyquist_hz: float | None,
                          start_sample: int, samples: np.ndarray) -> AudioSegmentRef:
        self._check_open()
        x = np.asarray(samples)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1 or x.shape[0] == 0:
            raise ValueError("audio segments are non-empty mono arrays")
        start_sample, n = int(start_sample), int(x.shape[0])
        if start_sample < 0:
            raise ValueError("start_sample must be >= 0")
        self.register_source(source_id, kind, role)
        entry = self._audio.get(source_id)
        if entry is None:
            entry = self._audio[source_id] = AudioSourceEntry(source_id=source_id, kind=kind, role=role,
                                                              rate_hz=int(rate_hz), n_samples=0, nyquist_hz=nyquist_hz)
        elif entry.rate_hz != int(rate_hz):
            raise ValueError(f"{source_id}: rate changed from {entry.rate_hz} to {rate_hz}")
        # a segment may replace the one with the identical start; any other overlap would double-play
        kept = [s for s in entry.segments if s.start_sample != start_sample]
        for s in kept:
            if start_sample < s.start_sample + s.n_samples and s.start_sample < start_sample + n:
                raise ValueError(f"{source_id}: segment [{start_sample},{start_sample + n}) overlaps "
                                 f"[{s.start_sample},{s.start_sample + s.n_samples})")
        rel = f"audio/{safe_name(source_id)}/seg_{start_sample:012d}.wav"
        dst = self.out / rel
        tmp = dst.with_name(dst.name + ".tmp")
        write_wav(tmp, x, int(rate_hz), "float32")
        os.replace(tmp, dst)
        ref = AudioSegmentRef(start_sample=start_sample, n_samples=n, path=rel, sha256=_sha256_file(dst))
        kept.append(ref)
        kept.sort(key=lambda s: s.start_sample)
        entry.segments = kept
        entry.n_samples = max(entry.n_samples, start_sample + n)
        return ref

    def export_flat_wav(self, source_id: str, path: str | Path) -> int:
        """Stream a full-length mono float32 wav (zeros for gaps), one segment in memory at a time."""
        entry = self._audio[source_id]
        total = max(entry.n_samples, self._declared_len.get(source_id, 0))
        pos = 0
        with StreamingWavWriter(path, entry.rate_hz) as w:
            for seg in entry.segments:
                if seg.start_sample > pos:
                    w.write_zeros(seg.start_sample - pos)
                x, rate = read_wav(self.out / seg.path)
                if rate != entry.rate_hz or x.shape[0] != seg.n_samples:
                    raise ValueError(f"{seg.path}: stored segment disagrees with the index")
                w.write(x[:, 0])
                pos = seg.start_sample + seg.n_samples
                del x
            if total > pos:
                w.write_zeros(total - pos)
        return total

    # ------------------------------------------------------------ timeline
    def add_timeline_segment(self, seg: TimelineSegment) -> None:
        self._check_open()
        if seg.source_id not in self._sources:
            raise ValueError(f"unknown source {seg.source_id!r}: call register_source or add_audio_segment first")
        if seg.end_sample < seg.start_sample or seg.end_us < seg.start_us:
            raise ValueError("segment ends before it starts")
        page = max(seg.start_us, 0) // self.page_span_us
        if page in self._tl_pages:
            raise ValueError(f"timeline page {page} is already flushed; add segments in roughly time order")
        self._tl_open.setdefault(page, []).append(seg)
        self._tl_max_end[page] = max(self._tl_max_end.get(page, 0), seg.end_us)
        lo, hi = self._src_span.get(seg.source_id, (seg.start_sample, seg.end_sample))
        self._src_span[seg.source_id] = (min(lo, seg.start_sample), max(hi, seg.end_sample))
        self._flush_old(self._tl_open, page, self._flush_tl_page)

    def _flush_tl_page(self, page: int) -> None:
        segs = sorted(self._tl_open.pop(page), key=lambda s: (s.start_us, s.source_id))
        rel = f"timeline/page_{page:04d}.json"
        _atomic_text(self.out / rel, TimelinePageDoc(segments=segs).to_json())
        end = max(self._tl_max_end.pop(page), (page + 1) * self.page_span_us)
        self._tl_pages[page] = TimelinePage(path=rel, start_us=page * self.page_span_us, end_us=end)

    @staticmethod
    def _flush_old(open_pages: dict, newest: int, flush) -> None:
        for p in [p for p in open_pages if p < newest - 1]:
            flush(p)

    # ------------------------------------------------------------ frames
    def set_frame_transform(self, frame_w: int, frame_h: int, mask_w: int | None = None, mask_h: int | None = None):
        self.manifest.capabilities["frame_transform"] = {
            "version": 1, "frame_w": int(frame_w), "frame_h": int(frame_h),
            "mask_w": int(mask_w or frame_w), "mask_h": int(mask_h or frame_h)}

    def add_frames(self, frames, pts_us) -> None:
        """`frames`: sequence of PNG paths (copied) or (h, w[, c]) uint8 arrays (encoded to PNG)."""
        self._check_open()
        frames = list(frames)
        pts_us = [int(p) for p in pts_us]
        if len(frames) != len(pts_us):
            raise ValueError("frames and pts_us differ in length")
        for fr, pts in zip(frames, pts_us):
            idx = self._fr_count
            name = f"{idx:06d}.png"
            dst = self.out / "frames" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            if isinstance(fr, (str, Path)):
                shutil.copyfile(fr, tmp)
                if self._frame_size is None:
                    self._frame_size = _png_size(tmp)
            else:
                arr = np.asarray(fr)
                write_png(tmp, arr)
                if self._frame_size is None:
                    self._frame_size = (int(arr.shape[1]), int(arr.shape[0]))
            os.replace(tmp, dst)
            page = max(pts, 0) // self.page_span_us
            if page in self._fr_pages:
                raise ValueError(f"frame page {page} is already flushed; add frames in time order")
            ptsl, files = self._fr_open.setdefault(page, ([], []))
            if not ptsl:
                self._fr_first[page] = idx
            ptsl.append(pts)
            files.append(f"frames/{name}")
            self._fr_count += 1
            self._flush_old(self._fr_open, page, self._flush_frame_page)

    def _flush_frame_page(self, page: int) -> None:
        ptsl, files = self._fr_open.pop(page)
        rel = f"frames/page_{page:04d}.json"
        _atomic_text(self.out / rel, json.dumps({"first_index": self._fr_first[page], "pts_us": ptsl, "files": files},
                                                separators=(",", ":")))
        self._fr_pages[page] = {"path": rel, "start_us": page * self.page_span_us,
                                "end_us": (page + 1) * self.page_span_us, "n": len(ptsl)}

    # ------------------------------------------------------------ masks
    def add_masks(self, track_id: str, records: list[MaskRecord]) -> None:
        self._check_open()
        open_pages = self._mask_open.setdefault(track_id, {})
        flushed = self._mask_pages.setdefault(track_id, {})
        for r in records:
            if self._mask_size is None and r.rle is not None:
                self._mask_size = (int(r.rle.width), int(r.rle.height))
            page = max(int(r.pts_us), 0) // self.page_span_us
            if page in flushed:
                raise ValueError(f"mask page {page} of {track_id} is already flushed; add records in time order")
            open_pages.setdefault(page, []).append(r)
            for p in [p for p in open_pages if p < page - 1]:
                self._flush_mask_page(track_id, p)

    def _flush_mask_page(self, track_id: str, page: int) -> None:
        recs = self._mask_open[track_id].pop(page)
        rel = f"masks/{safe_name(track_id)}/page_{page:04d}.jsonl"
        _atomic_text(self.out / rel, "".join(json.dumps(r.to_dict(), separators=(",", ":")) + "\n" for r in recs))
        self._mask_pages[track_id][page] = {"path": rel, "start_us": page * self.page_span_us,
                                            "end_us": (page + 1) * self.page_span_us, "n": len(recs)}

    # ------------------------------------------------------------ finalize
    def finalize(self, *, sample_rate: int | None = None, n_samples: int | None = None):
        """Flush every open buffer, write the indexes and the manifest, copy the viewer; return the manifest."""
        self._check_open()
        for p in list(self._tl_open):
            self._flush_tl_page(p)
        for p in list(self._fr_open):
            self._flush_frame_page(p)
        for tid in list(self._mask_open):
            for p in list(self._mask_open[tid]):
                self._flush_mask_page(tid, p)

        # audio index
        for sid, entry in self._audio.items():
            entry.n_samples = max(entry.n_samples, self._declared_len.get(sid, 0))
        audio = AudioIndex(sources=dict(self._audio))
        _atomic_text(self.out / "audio" / "index.json", audio.to_json())

        # timeline index
        chunking = self.manifest.long_run.chunking if self.manifest.long_run else {}
        rate = sample_rate or chunking.get("sample_rate") or max((e.rate_hz for e in self._audio.values()), default=None)
        if rate is None:
            raise ValueError("sample_rate is unknown: pass it to finalize() or set long_run.chunking['sample_rate']")
        total = n_samples or chunking.get("n_samples") or max((s[1] for s in self._src_span.values()), default=0)
        sources = []
        for sid, src in self._sources.items():
            lo_hi = self._src_span.get(sid)
            sources.append(src.model_copy(update={"first_sample": lo_hi[0] if lo_hi else None,
                                                  "last_sample": lo_hi[1] if lo_hi else None}))
        tl = TimelineIndex(sample_rate=int(rate), n_samples=int(total), sources=sources,
                           pages=[self._tl_pages[p] for p in sorted(self._tl_pages)])
        _atomic_text(self.out / "timeline" / "index.json", tl.to_json())

        # frame transform + frame/mask index
        caps = self.manifest.capabilities
        if "frame_transform" not in caps:
            fw, fh = self._frame_size or (None, None)
            media = getattr(self.manifest, "media", None)
            if media is not None and media.video:
                fw, fh = media.video[0].width, media.video[0].height
            if fw is not None:
                mw, mh = self._mask_size or (fw, fh)
                self.set_frame_transform(fw, fh, mw, mh)
        ft = caps.get("frame_transform")
        frames_index = {
            "version": LONG_RUN_SCHEMA_VERSION, "n_frames": self._fr_count, "page_span_us": self.page_span_us,
            "width": ft["frame_w"] if ft else None, "height": ft["frame_h"] if ft else None, "transform": ft,
            "pages": [self._fr_pages[p] for p in sorted(self._fr_pages)],
            "mask_pages": {tid: [pages[p] for p in sorted(pages)] for tid, pages in self._mask_pages.items()}}
        _atomic_text(self.out / "frames" / "index.json", json.dumps(frames_index, separators=(",", ":")))

        # viewer + manifest
        for name, dest in (("viewer.html", "viewer.html"), ("viewer.js", "assets/viewer.js"),
                           ("viewer.css", "assets/viewer.css")):
            _atomic_bytes(self.out / dest, (ASSET_DIR / name).read_bytes())
        lr = self.manifest.long_run or LongRunInfo()
        lr.timeline_index, lr.audio_index = "timeline/index.json", "audio/index.json"
        self.manifest.long_run = lr
        caps.setdefault("mp4", {"status": "skipped", "detail": "long-run bundles play paged PNG frames"})
        self.manifest.assets.update({"viewer": "viewer.html", "manifest": "manifest.json",
                                     "audio_index": lr.audio_index, "timeline_index": lr.timeline_index,
                                     "frames_index": "frames/index.json"})
        _atomic_text(self.out / "manifest.json", self.manifest.to_json(indent=1))
        self._finalized = True
        return self.manifest

    def _check_open(self) -> None:
        if self._finalized:
            raise RuntimeError("LongBundleWriter is finalized")


def _png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)
