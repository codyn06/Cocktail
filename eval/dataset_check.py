"""Data check for the labeled real recordings: coverage grid + PASS/FAIL per take.

    uv run --project hub python eval/dataset_check.py hub/recordings          # a folder of takes
    uv run --project hub python eval/dataset_check.py hub/recordings/t90-1024 # or one take
    ... -v                      # also list every check of every take
    ... --json check.json       # also write the per-take results

A take is a folder from quest/tools/record_session.py (audio.wav, pose.csv, meta.json,
and audio_chunks.csv for takes recorded after the metadata change). A folder of takes is
searched below the given path; a folder that has a meta.json is a take.

Prints
  1. the coverage grid: 12 angles (-150 .. 180 deg, + = right, 180 = behind) x 3 distances
     (0.5, 1, 2 m). Each cell shows "passing/total" takes; "." is an empty cell. A cell counts
     as FILLED when it holds at least one passing take (a failed recording covers nothing).
     Takes without a distance (or an angle) label are counted in the "?" column (row), not in
     a cell. Counts by head-movement size and stimulus follow when the labels exist.
  2. PASS or FAIL for each take, with the reason for every failed check:
       a. source at least MIN_SNR_DB (6 dB) above the room   (hub/locate/real_eval.py's measure)
       b. enough head turning: yaw std >= 8 deg in the turn phase (same measure)
       c. angle labeled: truth_deg present and a finite number
       d. no clock glitches: pose gaps > 0.5 s / non-monotonic pose time, pose present in the
          head-still window, the turn phase and until the turn phase ends (a pose that starts
          EARLY costs nothing and is only noted), chunk timestamps vs samples/rate (their
          median offset is removed, so a noisy first chunk or a constant offset does not matter;
          jitter, drift or a dropped-chunk step beyond --chunk-tol-ms fails and the observed
          numbers are printed), no recorder clock restart
  3. a final line "X of Y takes PASS (Z%); grid N of 36 cells filled".

Takes without audio_chunks.csv (recorded before it existed) cannot be checked for dropped chunks
or audio clock drift. They are checked for everything else, a WARNING is printed under each one and
in the summary, and the last line ends with "; N without chunk timestamps". They do NOT block READY
(old recordings cannot be re-recorded; the READY line names how many were not fully checked).

Exit code 0 only if the grid is full and at least 90% of the takes PASS.

Evidence: analysis code. Its tests (hub/tests/eval/test_dataset_check.py) use synthetic numpy
takes written by the test itself; the only other run was a smoke run on Sadie's simulated pairsim
sessions. It has never been run on a recording from a headset.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hub"))
from locate import real_eval  # noqa: E402  (eval/ is outside the hub package)

ANGLES = tuple(range(-150, 181, 30))        # 12 angles; 180 = behind, -90 = left
DISTANCES = (0.5, 1.0, 2.0)                 # meters, the tape marks on the floor
GRID_CELLS = len(ANGLES) * len(DISTANCES)   # 36
ANGLE_TOL_DEG = 10.0                        # a label within this of a grid angle belongs to it
DISTANCE_TOL_FRAC = 0.2                     # likewise for distance, relative
PASS_RATE_MIN = 0.90

MIN_SNR_DB = real_eval.MIN_SNR_DB           # source over the room
YAW_STD_MIN_DEG = 8.0                       # real_eval hard-codes the same number
POSE_GAP_MAX_S = 0.5
POSE_EARLY_NOTE_S = 3.0                     # a pose starting earlier than this before the audio is noted
TURN_POSE_MIN = 5                           # pose samples needed inside the turn phase
CHUNK_TOL_MS = 15.0                         # chunk timestamp vs samples/rate; below one 20 ms chunk


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class TakeResult:
    take: Path
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    truth_deg: float | None = None
    distance_m: float | None = None
    motion: str | None = None
    stimulus: str | None = None
    snr_db: float | None = None
    has_chunks: bool = False   # audio_chunks.csv present: the dropped-chunk / timestamp-drift check ran
    yaw_std: float | None = None

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def name(self) -> str:
        return self.take.name


# ---------------------------------------------------------------- discovery and labels


def find_takes(paths: list[Path]) -> list[Path]:
    takes: list[Path] = []
    for p in paths:
        if (p / "meta.json").exists():
            takes.append(p)
        elif p.is_dir():
            for d in sorted(p.rglob("meta.json")):
                takes.append(d.parent)
    seen, out = set(), []
    for t in takes:
        k = t.resolve()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def bin_angle(truth: float | None) -> int | None:
    """Nearest grid angle (-150 .. 180), or None when unlabeled or off-grid."""
    if truth is None:
        return None
    a = (truth + 180.0) % 360.0 - 180.0
    if a == -180.0:
        a = 180.0
    nearest = int(round(a / 30.0)) * 30
    if nearest == -180:
        nearest = 180
    return nearest if abs((a - nearest + 180) % 360 - 180) <= ANGLE_TOL_DEG else None


def bin_distance(d: float | None) -> float | None:
    if d is None:
        return None
    best = min(DISTANCES, key=lambda g: abs(g - d))
    return best if abs(best - d) <= DISTANCE_TOL_FRAC * best else None


# ---------------------------------------------------------------- measurements


def measure(take: Path) -> dict:
    """Source-over-room dB and turn-phase yaw std, from the scorer's own measurement."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = real_eval.run(take, None, 12.0, verbose=False)
    return {"snr_db": res["snr_db"], "yaw_std": res["yaw_std"], "log": buf.getvalue().strip()}


def _read_pose_t_us(path: Path) -> np.ndarray:
    try:
        a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (ValueError, OSError, UserWarning):
        return np.array([])
    return a[:, 0] if a.size else np.array([])


def _read_chunks(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (ValueError, OSError, UserWarning):
        return np.zeros((0, 4))
    return a if a.size else np.zeros((0, 4))


def clock_problems(meta: dict, wav_frames: int, rate: int, pose_t_us: np.ndarray,
                   chunks: np.ndarray | None, chunk_tol_ms: float = CHUNK_TOL_MS) -> tuple[list[str], list[str]]:
    """(problems, notes). A problem fails check d; a note is information (e.g. a check skipped)."""
    bad: list[str] = []
    notes: list[str] = []
    dur = wav_frames / rate
    start_us = meta.get("first_audio_ts_us")

    for key in ("clock_restarts", "pose_restarts"):
        n = meta.get(key)
        if isinstance(n, (int, float)) and n > 0:
            bad.append(f"recorder saw {int(n)} {key.replace('_', ' ')} (headset clock restarted; earlier data discarded)")
    if meta.get("audio_frames") is not None and meta["audio_frames"] != wav_frames:
        bad.append(f"meta.json audio_frames {meta['audio_frames']} != {wav_frames} samples in audio.wav")
    if meta.get("sample_rate") not in (None, rate):
        bad.append(f"meta.json sample_rate {meta['sample_rate']} != audio.wav {rate}")
    if _number(start_us) is None:
        bad.append("meta.json has no first_audio_ts_us: cannot align audio and pose")
        start_us = None

    if len(pose_t_us) < 2:
        bad.append(f"only {len(pose_t_us)} pose sample(s)")
    else:
        d = np.diff(pose_t_us) / 1e6
        back = int((d < 0).sum())
        if back:
            bad.append(f"pose time goes backwards {back}x (worst {d.min():.3f} s): clock glitch")
        gaps = d[d > POSE_GAP_MAX_S]
        if len(gaps):
            k = int(np.argmax(d))
            at = (pose_t_us[k] - (start_us if start_us is not None else pose_t_us[0])) / 1e6
            bad.append(f"{len(gaps)} pose gap(s) > {POSE_GAP_MAX_S} s (longest {d.max():.2f} s at t={at:.1f} s)")
        if start_us is not None:
            # The scorer drops poses before the audio starts, and each channel reconnects with its
            # own backoff (up to ~3 s), so an EARLY or slightly late pose costs nothing. It only
            # matters if pose is missing where the scorer needs it: the head-still window before
            # the turn cue, the turn phase, or the end of the turn phase.
            t = (pose_t_us - start_us) / 1e6
            first, last = float(t[0]), float(t[-1])
            if first < -POSE_EARLY_NOTE_S:
                notes.append(f"first pose is {-first:.1f} s before audio start (ignored by the scorer)")
            cue = meta.get("cue") or {}
            nu, go, stop = cue.get("noise_until") or 0.0, cue.get("go"), cue.get("stop")
            if go and stop:
                lo, hi = nu, go - 0.5  # real_eval's head-still window (yaw zero point)
                nb = int(((t >= lo) & (t < hi)).sum())
                if nb < TURN_POSE_MIN:
                    bad.append(f"{nb} pose sample(s) in the head-still window before the turn cue "
                               f"({lo:g}-{hi:g} s); pose starts at {first:.1f} s")
                n = int(((t >= go) & (t <= stop)).sum())
                if n < TURN_POSE_MIN:
                    bad.append(f"{n} pose sample(s) inside the turn phase ({go:g}-{stop:g} s)")
                if last < stop - POSE_GAP_MAX_S:
                    bad.append(f"pose ends at {last:.2f} s, before the turn phase ends ({stop:g} s)")
            elif last < 0 or first > dur:
                bad.append(f"pose [{first:.1f}, {last:.1f}] s does not overlap the audio [0, {dur:.1f}] s")

    if chunks is None:
        notes.append("no audio_chunks.csv (recorded before chunk timestamps existed): dropped-chunk check skipped")
    elif len(chunks) == 0:
        bad.append("audio_chunks.csv has no chunks")
    else:
        seq, ts, frames = chunks[:, 0], chunks[:, 1], chunks[:, 2]
        epoch = chunks[:, 3] if chunks.shape[1] > 3 else np.zeros(len(chunks))
        if int(frames.sum()) != wav_frames:
            bad.append(f"audio_chunks.csv frames sum {int(frames.sum())} != {wav_frames} samples in audio.wav")
        ds = np.diff(seq)
        if (ds != 1).any():
            k = int(np.argmax(ds != 1))
            bad.append(f"chunk seq jumps ({int((ds != 1).sum())}x; first {int(seq[k])} -> {int(seq[k + 1])}): "
                       "chunks lost or reordered")
        if (epoch != epoch[0]).any():
            bad.append("chunk epoch changed mid-take (the headset reconnected)")
        if (np.diff(ts) < 0).any():
            bad.append("chunk device timestamps go backwards")
        # Offset of each chunk's timestamp from where samples/rate puts it. The native Quest mic
        # path stamps each chunk from an extrapolated AudioRecord timestamp and jitters by a few
        # ms, so the constant part (the median offset, which also absorbs a noisy first chunk) is
        # removed; what is left is jitter, a slow clock drift, or a step from a dropped chunk.
        pos_ms = np.concatenate([[0], np.cumsum(frames)[:-1]]) / rate * 1e3
        off_ms = ts / 1e3 - pos_ms
        med = float(np.median(off_ms))
        res = off_ms - med
        worst = int(np.argmax(np.abs(res)))
        p95 = float(np.percentile(np.abs(res), 95))
        slope_ms = float(np.polyfit(pos_ms / 1e3, off_ms, 1)[0]) if len(off_ms) > 2 else 0.0  # ms per s = ppm / 1000
        summary = (f"offset from samples/rate: median {med:+.1f} ms, jitter p95 {p95:.1f} ms, "
                   f"worst {res[worst]:+.1f} ms at chunk {worst}, drift {slope_ms * 1e3:+.0f} ppm")
        if abs(res[worst]) > chunk_tol_ms:
            bad.append(f"chunk timestamps {summary} (> {chunk_tol_ms:g} ms from the median: "
                       "a chunk was dropped, the clock jumped, or the timestamps drift)")
        else:
            notes.append("chunk timestamps ok: " + summary)
        if start_us is not None and abs(ts[0] - start_us) > 1:
            bad.append(f"first chunk timestamp {int(ts[0])} != meta first_audio_ts_us {int(start_us)}")
    return bad, notes


def check_take(take: Path, measure_fn=None, chunk_tol_ms: float = CHUNK_TOL_MS) -> TakeResult:
    measure_fn = measure_fn or measure  # looked up at call time so tests can replace measure
    r = TakeResult(take)
    try:
        meta = json.loads((take / "meta.json").read_text())
        if not isinstance(meta, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:
        r.checks.append(Check("take readable", False, f"meta.json unreadable: {e}"))
        return r
    r.truth_deg = _number(meta.get("truth_deg"))
    r.distance_m = _number(meta.get("distance_m"))
    r.motion = meta.get("motion") if isinstance(meta.get("motion"), str) else None
    r.stimulus = meta.get("stimulus") if isinstance(meta.get("stimulus"), str) else None

    # (c) angle labeled
    raw = meta.get("truth_deg")
    r.checks.append(Check("angle labeled (truth_deg is a number)", r.truth_deg is not None,
                          f"{r.truth_deg:+.0f} deg" if r.truth_deg is not None
                          else ("truth_deg missing" if raw is None else f"truth_deg is {raw!r}")))

    missing = [f for f in ("audio.wav", "pose.csv") if not (take / f).exists()]
    if missing:
        why = "missing " + ", ".join(missing)
        for n in (f"source >= {MIN_SNR_DB:g} dB over the room", f"head turning (yaw std >= {YAW_STD_MIN_DEG:g} deg)",
                  "no clock glitches"):
            r.checks.append(Check(n, False, why))
        return r

    # (a) + (b) from the scorer's own measurement
    try:
        m = measure_fn(take)
        r.snr_db, r.yaw_std = m["snr_db"], m["yaw_std"]
        if m.get("log"):
            r.notes.append(m["log"])
        if r.snr_db is None:
            r.checks.append(Check(f"source >= {MIN_SNR_DB:g} dB over the room", False,
                                  "cannot measure: no room-noise segment (meta.json cue.noise_until / go missing; "
                                  "record with --noise-until --go-at --stop-at)"))
        else:
            r.checks.append(Check(f"source >= {MIN_SNR_DB:g} dB over the room", r.snr_db >= MIN_SNR_DB,
                                  f"{r.snr_db:+.1f} dB"))
        r.checks.append(Check(f"head turning (yaw std >= {YAW_STD_MIN_DEG:g} deg in the turn phase)",
                              r.yaw_std >= YAW_STD_MIN_DEG, f"std {r.yaw_std:.1f} deg"))
    except Exception as e:  # noqa: BLE001  (a corrupt take must be reported, not crash the batch)
        why = f"could not analyze the take: {type(e).__name__}: {e}"
        r.checks.append(Check(f"source >= {MIN_SNR_DB:g} dB over the room", False, why))
        r.checks.append(Check(f"head turning (yaw std >= {YAW_STD_MIN_DEG:g} deg in the turn phase)", False, why))

    # (d) clock glitches
    try:
        with wave.open(str(take / "audio.wav")) as w:
            wav_frames, rate = w.getnframes(), w.getframerate()
        chunks = _read_chunks(take / "audio_chunks.csv")
        r.has_chunks = chunks is not None
        bad, notes = clock_problems(meta, wav_frames, rate, _read_pose_t_us(take / "pose.csv"), chunks, chunk_tol_ms)
        r.notes += notes
        r.checks.append(Check("no clock glitches", not bad, "; ".join(bad) if bad else "timestamps consistent"))
    except Exception as e:  # noqa: BLE001
        r.checks.append(Check("no clock glitches", False, f"could not read audio.wav: {type(e).__name__}: {e}"))
    return r


# ---------------------------------------------------------------- report


def build_grid(results: list[TakeResult]):
    """cells[(angle|None, dist|None)] = [passing, total]."""
    cells: dict[tuple[int | None, float | None], list[int]] = {}
    for r in results:
        key = (bin_angle(r.truth_deg), bin_distance(r.distance_m))
        c = cells.setdefault(key, [0, 0])
        c[0] += r.passed
        c[1] += 1
    return cells


def filled_cells(cells) -> set[tuple[int, float]]:
    return {k for k, (p, _) in cells.items() if p > 0 and k[0] is not None and k[1] is not None}


def print_grid(results: list[TakeResult]) -> set[tuple[int, float]]:
    cells = build_grid(results)
    filled = filled_cells(cells)
    print("Coverage grid (passing/total takes per cell; . = empty; ? = unlabeled or off-grid)")
    cols_h = [f"{d:g} m" for d in DISTANCES] + ["?"]
    print("  angle |" + "".join(f" {c:>6} |" for c in cols_h))
    print("  ------+" + "--------+" * len(cols_h))

    def cell(key):
        p, t = cells.get(key, (0, 0))
        return f"{p}/{t}" if t else "."

    for a in list(ANGLES) + [None]:
        label = "?" if a is None else f"{a:+d}"
        cols = [cell((a, d)) for d in DISTANCES] + [cell((a, None))]
        print(f"  {label:>5} |" + "".join(f" {c:>6} |" for c in cols))
    empty = [(a, d) for a in ANGLES for d in DISTANCES if (a, d) not in filled]
    if empty:
        parts = []
        for a in ANGLES:
            ds = [d for d in DISTANCES if (a, d) not in filled]
            if ds:
                parts.append(f"{a:+d}: " + ("all" if len(ds) == len(DISTANCES) else ",".join(f"{d:g}" for d in ds) + " m"))
        print(f"  {len(empty)} empty cells (no passing take), by angle: " + "; ".join(parts))
    for lab, vals in (("motion", ("small", "normal", "large")), ("stimulus", ("pink", "speech"))):
        parts = []
        if not any(getattr(r, lab) in vals for r in results):
            print(f"  by {lab}: no take has this label ({len(results)} unlabeled)")
            continue
        for v in list(vals) + [None]:
            rs = [r for r in results if (getattr(r, lab) if getattr(r, lab) in vals else None) == v]
            if not rs and v is None:
                continue
            cov = {(bin_angle(r.truth_deg), bin_distance(r.distance_m)) for r in rs if r.passed}
            n_cov = len([k for k in cov if k[0] is not None and k[1] is not None])
            parts.append(f"{v or 'unlabeled'} {sum(r.passed for r in rs)}/{len(rs)} pass, {n_cov} cells")
        print(f"  by {lab}: " + "; ".join(parts))
    return filled


def print_takes(results: list[TakeResult], verbose: bool) -> None:
    print()
    for r in results:
        lab = " ".join([
            f"truth {r.truth_deg:+.0f} deg" if r.truth_deg is not None else "truth ?",
            f"{r.distance_m:g} m" if r.distance_m is not None else "? m",
            r.motion or "?", r.stimulus or "?"])
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name}  ({lab})")
        for c in r.checks:
            if verbose or not c.ok:
                print(f"      {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}")
        if not r.has_chunks:
            print("      WARNING: no audio_chunks.csv, so the dropped-chunk / timestamp-drift check did NOT run")
        if verbose:
            for n in r.notes:
                print(f"      note: {n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="take folders, or folders of takes")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every check and note of every take")
    ap.add_argument("--json", type=Path, default=None, help="also write the per-take results here")
    ap.add_argument("--chunk-tol-ms", type=float, default=CHUNK_TOL_MS,
                    help="allowed chunk timestamp drift from samples/rate (default %(default)s)")
    a = ap.parse_args(argv)

    takes = find_takes(a.paths)
    if not takes:
        print("no takes found (a take is a folder with meta.json, audio.wav, pose.csv)")
        return 2
    results = []
    for i, tk in enumerate(takes, 1):
        print(f"checking {i}/{len(takes)}: {tk.name}", file=sys.stderr)
        results.append(check_take(tk, chunk_tol_ms=a.chunk_tol_ms))

    filled = print_grid(results)
    print_takes(results, a.verbose)
    n_pass = sum(r.passed for r in results)
    n_nochunks = sum(not r.has_chunks for r in results)
    rate = n_pass / len(results)
    ready = len(filled) == GRID_CELLS and rate >= PASS_RATE_MIN
    print()
    if not ready:
        why = []
        if len(filled) < GRID_CELLS:
            why.append(f"grid has {GRID_CELLS - len(filled)} empty cells")
        if rate < PASS_RATE_MIN:
            why.append(f"pass rate {100 * rate:.0f}% < {100 * PASS_RATE_MIN:.0f}%")
        print("NOT READY: " + "; ".join(why))
    else:
        print("READY: grid full and pass rate >= 90%"
              + (f" (but {n_nochunks} take(s) were not checked for dropped chunks)" if n_nochunks else ""))
    if n_nochunks:
        print(f"WARNING: {n_nochunks} of {len(results)} takes have no chunk timestamps (recorded before "
              "audio_chunks.csv existed); their PASS does not cover dropped chunks or audio clock drift")
    if a.json:
        a.json.write_text(json.dumps([{
            "take": str(r.take), "pass": r.passed, "truth_deg": r.truth_deg, "distance_m": r.distance_m,
            "motion": r.motion, "stimulus": r.stimulus, "snr_db": r.snr_db, "yaw_std": r.yaw_std,
            "chunk_timestamps": r.has_chunks,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in r.checks],
        } for r in results], indent=1))
    print(f"{n_pass} of {len(results)} takes PASS ({100 * rate:.0f}%); grid {len(filled)} of {GRID_CELLS} cells filled"
          + (f"; {n_nochunks} without chunk timestamps" if n_nochunks else ""))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
