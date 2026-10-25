"""Content-addressed stage cache with atomic completion markers.

A stage's key hashes its inputs' CONTENT, its relevant config, its transforms, the schema
version and the keys of the stages it depends on, so changing anything upstream changes every
downstream key. Outputs are written into a `.partial-*` directory and only become a cache hit
after a `COMPLETE.json` marker is written and the directory is atomically renamed into place.
A directory left by a crashed job (no marker, or still `.partial-*`) is never a hit.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .schemas import SCHEMA_VERSION, canonical_json

MARKER = "COMPLETE.json"

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
}


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


class StageCache:
    def __init__(self, root: str | Path, dag: dict[str, tuple[str, ...]] | None = None):
        self.root = Path(root)
        self.dag = dict(dag or DEFAULT_DAG)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- keys
    def key(self, stage: str, *, inputs: dict[str, str] | None = None, config=None, transform=None,
            upstream: dict[str, str] | None = None) -> str:
        if stage not in self.dag:
            raise KeyError(f"unknown stage {stage!r}")
        payload = {"stage": stage, "schema": SCHEMA_VERSION, "inputs": dict(sorted((inputs or {}).items())),
                   "config": config, "transform": transform,
                   "upstream": dict(sorted((upstream or {}).items()))}
        return hash_bytes(canonical_json(payload).encode("utf-8"))[:32]

    def path(self, stage: str, key: str) -> Path:
        return self.root / stage / key

    # ---- lookup
    def is_complete(self, stage: str, key: str) -> bool:
        d = self.path(stage, key)
        marker = d / MARKER
        if not marker.is_file():
            return False
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return info.get("stage") == stage and info.get("key") == key

    def lookup(self, stage: str, key: str) -> Path | None:
        return self.path(stage, key) if self.is_complete(stage, key) else None

    # ---- write
    @contextmanager
    def begin(self, stage: str, key: str, meta: dict | None = None) -> Iterator[Path]:
        """Yield a scratch directory; on clean exit publish it atomically as the cache entry."""
        final = self.path(stage, key)
        final.parent.mkdir(parents=True, exist_ok=True)
        work = final.parent / f".partial-{key}-{uuid.uuid4().hex[:8]}"
        work.mkdir()
        try:
            yield work
            tmp_marker = work / (MARKER + ".tmp")
            tmp_marker.write_text(json.dumps({"stage": stage, "key": key, "meta": meta or {}},
                                             sort_keys=True), encoding="utf-8")
            os.replace(tmp_marker, work / MARKER)
            if final.exists():
                shutil.rmtree(final)
            os.replace(work, final)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise

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
