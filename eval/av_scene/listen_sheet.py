"""Write each mixture case's mixture + references as 48 kHz 16-bit WAV plus listen_sheet.md for manual listening.

    python listen_sheet.py --out DIR [--sessions PATH] [--include PREFIX ...]

No GPU or model use. Evidence tier is per case (see the table): all SYNTHETIC / replayed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

import mixtures as X


def _write(path: Path, x: np.ndarray) -> None:
    sf.write(str(path), np.clip(np.asarray(x, np.float32), -1.0, 1.0), X.RATE, subtype="PCM_16")


def write_sheet(out: Path, sessions: Path = X.SESSIONS, include=None) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    rows = ["# Listen sheet", "",
            "| case_id | description | duration_s | overlap_s | evidence tier | files |", "|---|---|---|---|---|---|"]
    ids = []
    for c in X.build_cases(sessions, include=include):
        d = out / c.case_id
        d.mkdir(parents=True, exist_ok=True)
        _write(d / "mixture.wav", c.audio)
        links = [f"[mixture]({c.case_id}/mixture.wav)"]
        for k, ref in c.refs.items():
            _write(d / f"ref_{k}.wav", ref)
            links.append(f"[ref {k}]({c.case_id}/ref_{k}.wav)")
        if "overlap_s" in c.meta:
            ov = float(c.meta["overlap_s"])
        else:
            keys = [k for k, v in c.spans.items() if v]
            ov = sum(X.M.spans_overlap_s(c.spans[a], c.spans[b]) for i, a in enumerate(keys) for b in keys[i + 1:])
        tier = str(c.meta.get("tier", X.TIER)).replace("|", "/")
        desc = c.description.replace("|", "/")
        rows.append(f"| {c.case_id} | {desc} | {c.duration:.2f} | {ov:.2f} | {tier} | {' '.join(links)} |")
        ids.append(c.case_id)
    (out / "listen_sheet.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sessions", type=Path, default=X.SESSIONS)
    ap.add_argument("--include", nargs="*", default=None)
    a = ap.parse_args()
    ids = write_sheet(a.out, a.sessions, a.include)
    print(f"wrote {len(ids)} cases to {a.out}")


if __name__ == "__main__":
    main()
