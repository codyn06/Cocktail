"""Content-addressed stage cache with atomic completion markers.

A stage's key hashes its inputs' CONTENT, its relevant config, its transforms, the schema
version and the keys of the stages it depends on, so changing anything upstream changes every
downstream key. Outputs are written into a `.partial-*` directory and only become a cache hit
after a `COMPLETE.json` marker is written and the directory is atomically renamed into place.
A directory left by a crashed job (no marker, or still `.partial-*`) is never a hit.

Phase 4 additions (long clips, resume):
  * COMPLETE.json carries a `files` manifest (size + sha256 per file) and the `required` file list;
    `lookup(..., verify=True)` re-checks sizes and checksums and quarantines a corrupt entry
    (`.corrupt-*`) instead of serving it.
  * Keys additionally hash the implementation fingerprint, model/checkpoint revisions, seeds and the
    key of the INCOMING STATE, so a stateful chunk chain re-keys everything after a changed chunk.
  * Single writer per key: `begin()` takes `<stage>/.lock-<key>` (O_EXCL); a live holder makes a second
    writer wait (`get_or_compute`) or raise `CacheBusy` (`begin`). A lock whose owner process is gone
    is stolen. Ownership is per process+thread-safe because `begin` always releases in `finally`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .schemas import SCHEMA_VERSION, canonical_json

MARKER = "COMPLETE.json"
LOCK_STALE_S = 600.0       # an UNREADABLE lock older than this is stolen; a readable one only when its owner is gone
LOCK_POLL_S = 0.05

# stage -> stages it consumes. Source-specific artifacts hang off the stage that produced them.
DEFAULT_DAG: dict[str, tuple[str, ...]] = {
    "ingest": (),
    "discovery": ("ingest",),
    "vision": ("ingest",),
    "evidence": ("ingest", "discovery", "vision"),
    "association": ("evidence",),
    "stems": ("association",),
    "residual": ("stems",),
    "render": ("association", "stems", "residual"),
    # ---- Phase 4 chunked pipeline. One cache entry per chunk (or event window); the chunk index and the
    # incoming-state key are part of each entry's key, so these stage-level edges only drive invalidate().
    "visual_chunk": ("ingest",),
    "speech_chunk": ("ingest", "visual_chunk"),
    "diar_window": ("ingest",),
    "event_detect": ("ingest",),
    "event_window": ("ingest", "event_detect"),
    "residual_chunk": ("speech_chunk", "event_window"),
    "export": ("speech_chunk", "event_window", "residual_chunk"),
}


class CacheBusy(RuntimeError):
    """Another live writer holds the lock for this (stage, key)."""


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def hash_config(obj) -> str:
    return hash_bytes(canonical_json(obj).encode("utf-8"))


def code_fingerprint(*objs) -> str:
    """Implementation fingerprint: sha256 over the source files of the given modules/classes/functions.
    Editing any of them re-keys every stage that lists the fingerprint."""
    h = hashlib.sha256()
    seen: set[str] = set()
    for o in objs:
        f = inspect.getsourcefile(o)
        if f is None or f in seen:
            continue
        seen.add(f)
        h.update(Path(f).name.encode())
        h.update(hash_file(f).encode())
    return h.hexdigest()[:32]


def _pid_alive(pid: int) -> bool:
    """Liveness probe that never signals the target (os.kill(pid, 0) TERMINATES the process on Windows)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = wintypes.HANDLE
        h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return k32.GetLastError() == 5               # access denied == exists
        try:
            code = wintypes.DWORD()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class StageCache:
    def __init__(self, root: str | Path, dag: dict[str, tuple[str, ...]] | None = None):
        self.root = Path(root)
        self.dag = dict(dag or DEFAULT_DAG)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- keys
    def key(self, stage: str, *, inputs: dict[str, str] | None = None, config=None, transform=None,
            upstream: dict[str, str] | None = None, fingerprint: str | None = None,
            models: dict[str, str] | None = None, seeds=None, state_in: str | None = None) -> str:
        """`fingerprint` = implementation hash (`code_fingerprint`); `models` = model/checkpoint revisions;
        `seeds` = every RNG seed that influences the output; `state_in` = hash of the incoming state for
        stateful chunk chains. All are part of the key so none can be silently stale."""
        if stage not in self.dag:
            raise KeyError(f"unknown stage {stage!r}")
        payload = {"stage": stage, "schema": SCHEMA_VERSION, "inputs": dict(sorted((inputs or {}).items())),
                   "config": config, "transform": transform,
                   "upstream": dict(sorted((upstream or {}).items())),
                   "fingerprint": fingerprint, "models": dict(sorted((models or {}).items())),
                   "seeds": seeds, "state_in": state_in}
        return hash_bytes(canonical_json(payload).encode("utf-8"))[:32]

    def path(self, stage: str, key: str) -> Path:
        return self.root / stage / key

    # ---- lookup
    def _read_marker(self, stage: str, key: str) -> dict | None:
        marker = self.path(stage, key) / MARKER
        if not marker.is_file():
            return None
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(info, dict) or info.get("stage") != stage or info.get("key") != key:
            return None
        return info

    def is_complete(self, stage: str, key: str) -> bool:
        """Cheap check: a valid marker exists. `lookup` is the one that validates content."""
        return self._read_marker(stage, key) is not None

    def validate(self, stage: str, key: str, *, checksums: bool = True) -> str | None:
        """None if the entry is intact, else the reason it is not (marker, manifest, file, size, checksum)."""
        info = self._read_marker(stage, key)
        if info is None:
            return "no valid marker"
        files = info.get("files")
        if not isinstance(files, dict):
            return "marker has no file manifest"
        for rel in info.get("required", []):
            if rel not in files:
                return f"required file {rel!r} not in manifest"
        d = self.path(stage, key)
        for rel, rec in files.items():
            f = d / rel
            if not f.is_file():
                return f"missing file {rel!r}"
            if f.stat().st_size != rec.get("size"):
                return f"size mismatch {rel!r}"
            if checksums and hash_file(f) != rec.get("sha256"):
                return f"checksum mismatch {rel!r}"
        return None

    def quarantine(self, stage: str, key: str) -> Path | None:
        d = self.path(stage, key)
        if not d.exists():
            return None
        dest = d.parent / f".corrupt-{key}-{uuid.uuid4().hex[:6]}"
        os.replace(d, dest)
        return dest

    def lookup(self, stage: str, key: str, *, verify: bool = True) -> Path | None:
        """The entry directory, or None. With `verify` (default) sizes AND checksums are re-checked and a
        damaged entry is quarantined, so a corrupt cache can only cost a recompute, never a wrong result."""
        d = self.path(stage, key)
        if not d.exists():
            return None
        if self.validate(stage, key, checksums=verify) is None:
            return d
        if self._read_marker(stage, key) is not None:         # was published, now damaged
            self.quarantine(stage, key)
        return None

    def meta(self, stage: str, key: str) -> dict | None:
        info = self._read_marker(stage, key)
        return None if info is None else info.get("meta", {})

    # ---- locking (single writer per key)
    def _lock_path(self, stage: str, key: str) -> Path:
        return self.root / stage / f".lock-{key}"

    def _try_lock(self, stage: str, key: str, stale_s: float) -> bool:
        lp = self._lock_path(stage, key)
        lp.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(json.loads(lp.read_text(encoding="utf-8")).get("pid", 0))
                    steal = not _pid_alive(pid)
                except (OSError, ValueError, TypeError):
                    try:                                           # unreadable/half-written: stale only when old
                        steal = time.time() - lp.stat().st_mtime > stale_s
                    except OSError:
                        steal = False
                if not steal:
                    return False
                gone = lp.with_name(lp.name + f".stolen-{uuid.uuid4().hex[:6]}")
                try:
                    os.replace(lp, gone)                           # atomic: only one thief wins
                    gone.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "t": time.time()}, f)
            return True
        return False

    def _unlock(self, stage: str, key: str) -> None:
        self._lock_path(stage, key).unlink(missing_ok=True)

    # ---- write
    @contextmanager
    def begin(self, stage: str, key: str, meta: dict | None = None, required: tuple[str, ...] = (),
              stale_s: float = LOCK_STALE_S) -> Iterator[Path]:
        """Yield a scratch directory; on clean exit publish it atomically as the cache entry.

        Holds the per-key lock for the whole write (`CacheBusy` if a live writer has it). `required` files
        must exist at publish. `meta` is read at publish time, so the body may add to it. A valid entry
        already present for this key is kept (the scratch is discarded): a lost race can never replace
        good bytes with different ones."""
        final = self.path(stage, key)
        final.parent.mkdir(parents=True, exist_ok=True)
        if not self._try_lock(stage, key, stale_s):
            raise CacheBusy(f"{stage}/{key} is being written by another process")
        work = final.parent / f".partial-{key}-{uuid.uuid4().hex[:8]}"
        try:
            work.mkdir()
            yield work
            files = {}
            for f in sorted(p for p in work.rglob("*") if p.is_file()):
                files[f.relative_to(work).as_posix()] = {"size": f.stat().st_size, "sha256": hash_file(f)}
            for rel in required:
                if rel not in files:
                    raise FileNotFoundError(f"{stage}/{key}: required output {rel!r} was not written")
            tmp_marker = work / (MARKER + ".tmp")
            tmp_marker.write_text(json.dumps({"stage": stage, "key": key, "meta": meta or {}, "files": files,
                                              "required": list(required)}, sort_keys=True), encoding="utf-8")
            os.replace(tmp_marker, work / MARKER)
            if final.exists():
                if self.validate(stage, key, checksums=False) is None:
                    shutil.rmtree(work, ignore_errors=True)
                    return
                shutil.rmtree(final)
            os.replace(work, final)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        finally:
            self._unlock(stage, key)

    def get_or_compute(self, stage: str, key: str, producer: Callable[[Path, dict], None], *,
                       meta: dict | None = None, required: tuple[str, ...] = (), wait_s: float = 120.0,
                       stale_s: float = LOCK_STALE_S) -> tuple[Path, bool]:
        """(entry dir, computed). A valid hit never calls `producer`. A concurrent live writer is waited for
        (up to `wait_s`) and its result reused. `producer(scratch_dir, meta)` fills the dir and may update `meta`."""
        deadline = time.monotonic() + wait_s
        while True:
            hit = self.lookup(stage, key)
            if hit is not None:
                return hit, False
            m = dict(meta or {})
            try:
                with self.begin(stage, key, meta=m, required=required, stale_s=stale_s) as d:
                    producer(d, m)
            except CacheBusy:
                if time.monotonic() > deadline:
                    raise
                time.sleep(LOCK_POLL_S)
                continue
            hit = self.lookup(stage, key)
            if hit is None:
                raise RuntimeError(f"{stage}/{key} could not be read back after publish")
            return hit, True

    # ---- invalidation
    def downstream(self, stage: str) -> list[str]:
        out: list[str] = []
        todo = [stage]
        while todo:
            cur = todo.pop()
            for s, deps in self.dag.items():
                if cur in deps and s not in out:
                    out.append(s)
                    todo.append(s)
        return out

    def invalidate(self, stage: str) -> list[str]:
        """Drop a stage and everything that consumes it. Returns the stages removed."""
        removed = []
        for s in [stage, *self.downstream(stage)]:
            d = self.root / s
            if d.exists():
                shutil.rmtree(d)
                removed.append(s)
        return removed
