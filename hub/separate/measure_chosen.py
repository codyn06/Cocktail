"""How many dB louder is the chosen voice than the others, and how soon after selecting?

  uv run --group gpu python -m separate.measure_chosen sessions/suite/scene01/session [--target man]
      [--t0 4] [--threshold-db 10] [--window-ms 250] [--json out.json]

The session needs ground-truth stems (every simulated session has them; real recordings do not,
and this script refuses them rather than guessing). The live separator is run over it exactly as
in `separate.score`. For each target voice (default: every source) the chosen stream is the one
that carries it best after the selection time t0 (oracle choice, like score.py; a UI tap would
pick the stream currently showing that voice). Then, in 250 ms windows from t0:

  margin = 10 log10( energy of the target in the stream / energy of every other source in it )

Each window's stream samples are projected onto the ground-truth stems (least squares, a few
samples of ridge); a stem's contribution is its fitted coefficient times the stem. This is an
SIR-style measure: it ignores sound that matches none of the stems (reported separately as
`unexplained_db`). Windows where the target is silent in the reference are skipped (a margin
there says nothing), so a pause in the target neither passes nor fails the check.

Reported per target: the first window whose margin reaches the threshold, the time from which
every later active window stays at or above it ("stable"), the steady-state median margin, and
the same median in the raw mix for comparison. Times are the END of the window, measured from t0
in signal time (the stream runs continuously for all voices, so selecting does not delay the
separation itself; the pipeline lag of up to 2 hops before audio reaches the output is NOT included
and is reported on its own line).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
CAP_DB = 60.0  # margins are clipped to +-this (a silent "other" would otherwise be infinite)


def _db(p: float) -> float:
    return float(10 * np.log10(max(p, 1e-12)))


def sisdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Same definition as separate.score.sisdr (kept here so unit tests need no model stack)."""
    est, ref = est - est.mean(), ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    return float(10 * np.log10(np.sum((a * ref) ** 2) / (np.sum((a * ref - est) ** 2) + 1e-12)))


@dataclass
class Margins:
    """Per-window result for one stream and one target. All arrays have one entry per window."""
    t_start_s: np.ndarray    # window start, seconds from t0
    margin_db: np.ndarray    # target vs all other stems, clipped to +-CAP_DB; NaN where inactive
    target_db: np.ndarray    # energy of the target component (dB re 1.0 power per sample)
    others_db: np.ndarray    # energy of the other stems' components
    unexplained_db: np.ndarray  # energy matching no stem, relative to the target (dB)
    active: np.ndarray       # bool: target present in the reference during this window
    window_s: float


def window_margins(x: np.ndarray, refs: dict[str, np.ndarray], target: str, t0_s: float = 0.0,
                   window_s: float = 0.25, min_active_db: float = -14.0, sr: int = SR) -> Margins:
    """Target-vs-others margin of signal `x` in consecutive windows starting at t0_s.

    x: (n,) the stream (or the mix). refs: stem name -> (n,) ground truth, same sample alignment.
    A window is active when the target stem's RMS there is within `min_active_db` of its RMS over
    the observed span's non-silent part (so pauses are excluded without an absolute threshold)."""
    names = list(refs)
    R = np.stack([refs[k] for k in names], axis=1).astype(np.float64)  # (n, k)
    ti = names.index(target)
    n = min(len(x), len(R))
    w, a0 = int(round(window_s * sr)), int(round(t0_s * sr))
    starts = list(range(a0, n - w + 1, w))
    tgt_rms = np.array([np.sqrt(np.mean(R[s : s + w, ti] ** 2)) for s in starts])
    loud = tgt_rms[tgt_rms > 1e-4]  # relative to when it is actually speaking
    ref_rms = float(np.sqrt(np.mean(loud**2))) if len(loud) else 0.0
    out = {k: np.full(len(starts), np.nan) for k in ("margin", "tgt", "oth", "unx")}
    active = np.zeros(len(starts), bool)
    for i, s in enumerate(starts):
        e, Rw = x[s : s + w].astype(np.float64), R[s : s + w]
        if ref_rms == 0.0 or tgt_rms[i] < ref_rms * 10 ** (min_active_db / 20):
            continue
        active[i] = True
        G = Rw.T @ Rw
        lam = 1e-6 * np.trace(G) / len(G) + 1e-12
        coef = np.linalg.solve(G + lam * np.eye(len(G)), Rw.T @ e)
        comp = Rw * coef  # (w, k): each stem as it appears in x
        p_t = float(np.mean(comp[:, ti] ** 2))
        p_o = float(np.mean(np.sum(np.delete(comp, ti, axis=1) ** 2, axis=1))) if len(names) > 1 else 0.0
        p_u = float(np.mean((e - comp.sum(1)) ** 2))
        out["tgt"][i], out["oth"][i] = _db(p_t), _db(p_o)
        out["margin"][i] = np.clip(_db(p_t) - _db(p_o), -CAP_DB, CAP_DB)
        out["unx"][i] = np.clip(_db(p_u) - _db(p_t), -CAP_DB, CAP_DB)
    return Margins(np.arange(len(starts)) * window_s, out["margin"], out["tgt"], out["oth"], out["unx"],
                   active, window_s)


def reach_times(m: Margins, threshold_db: float) -> dict:
    """First and stable crossing of the threshold, seconds from t0 to the END of the window."""
    act = np.flatnonzero(m.active)
    if not len(act):
        return {"first_reach_s": None, "stable_s": None, "frac_windows_ok": None}
    ok = m.margin_db[act] >= threshold_db
    first = act[np.argmax(ok)] if ok.any() else None
    # stable: earliest active window from which every later active window is >= threshold
    bad = np.flatnonzero(~ok)
    stable = (act[0] if not len(bad) else (act[bad[-1] + 1] if bad[-1] + 1 < len(act) else None))
    end = lambda i: None if i is None else round(float(m.t_start_s[i] + m.window_s), 3)
    return {"first_reach_s": end(first), "stable_s": end(stable), "frac_windows_ok": round(float(ok.mean()), 3)}


def choose_stream(est: dict[str, np.ndarray], ref: np.ndarray, t0_s: float, sr: int = SR) -> str:
    """Stream that carries the target best from t0 on (highest SI-SDR). Oracle, like score.py."""
    a = int(round(t0_s * sr))
    n = min(len(ref), *(len(v) for v in est.values()))
    return max(est, key=lambda sid: sisdr(est[sid][a:n], ref[a:n]))


def measure(est: dict[str, np.ndarray], mix: np.ndarray, refs: dict[str, np.ndarray], target: str,
            t0_s: float = 4.0, threshold_db: float = 10.0, window_s: float = 0.25,
            steady_after_s: float = 1.0, budget_s: float = 1.0, stream: str | None = None) -> dict:
    sid = stream or choose_stream(est, refs[target], t0_s)
    m = window_margins(est[sid], refs, target, t0_s, window_s)
    mm = window_margins(mix, refs, target, t0_s, window_s)
    act = m.active
    steady = act & (m.t_start_s >= steady_after_s)
    med = lambda v, sel: None if not sel.any() else round(float(np.median(v[sel])), 2)
    r = reach_times(m, threshold_db)
    # Which stream holds the target loudest in each active window: if it is not the chosen one,
    # the voice moved (swap / sent to the residual) and the chosen stream is the wrong place to listen.
    per_stream = np.stack([window_margins(v, refs, target, t0_s, window_s).target_db for v in est.values()])
    top = np.array(list(est))[np.nanargmax(per_stream[:, act], axis=0)] if act.any() else []
    carried_by = {s: round(float(np.mean(top == s)), 3) for s in est if np.any(top == s)}
    return {
        "target": target, "stream": sid, "t0_s": t0_s, "threshold_db": threshold_db,
        "n_windows": int(len(act)), "n_active": int(act.sum()),
        **r,
        "met_within_budget": r["stable_s"] is not None and r["stable_s"] <= budget_s,
        "budget_s": budget_s,
        "margin_first_window_db": None if not act.any() else round(float(m.margin_db[np.flatnonzero(act)[0]]), 2),
        "steady_margin_db": med(m.margin_db, steady),
        "steady_min_db": None if not steady.any() else round(float(np.nanmin(m.margin_db[steady])), 2),
        "mix_margin_db": med(mm.margin_db, act),
        "unexplained_db": med(m.unexplained_db, steady),
        "carried_by": carried_by,
        "margins_db": [None if np.isnan(v) else round(float(v), 1) for v in m.margin_db],
    }


def load_refs(sess) -> dict[str, np.ndarray]:
    from scipy.io import wavfile

    srcs = sess.meta.get("sources") or []
    if not srcs:
        raise SystemExit(f"{sess.path}: no ground-truth sources in session.json; nothing to measure against "
                         "(real recordings have no stems; this measurement needs them)")
    refs = {}
    for q in srcs:
        p = sess.path / q["stem"]
        if not p.exists():
            raise SystemExit(f"{p}: stem file missing")
        sr, r = wavfile.read(p)
        assert sr == SR, f"{p}: expected {SR} Hz stems, got {sr}"
        refs[q["name"]] = r.astype(np.float32)
    return refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--target", action="append", help="source name(s) to treat as chosen (default: all)")
    ap.add_argument("--t0", default="4", help="selection time(s) in s from session start, comma separated")
    ap.add_argument("--threshold-db", type=float, default=10.0)
    ap.add_argument("--window-ms", type=float, default=250.0)
    ap.add_argument("--budget-s", type=float, default=1.0, help="must be stable within this long after t0")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--voice-memory", action="store_true")
    ap.add_argument("--merge-ild-db", type=float,
                    help="one-talker merge threshold (default: 4.5 for simulated sessions, else 0 = off)")
    ap.add_argument("--json", type=Path, help="write all results here")
    a = ap.parse_args()

    from mockclient import Session

    from .live import LiveSeparator
    from .models import SoundLabeler, VoiceSeparator
    from .score import run_session
    from .suite import SIM_MERGE_ILD_DB

    sess = Session.load(a.session)
    refs = load_refs(sess)
    merge = a.merge_ild_db if a.merge_ild_db is not None else (SIM_MERGE_ILD_DB if sess.meta.get("simulated") else 0.0)
    live = LiveSeparator(VoiceSeparator(), None if a.no_labels else SoundLabeler(),
                         merge_ild_db=merge, voice_memory=a.voice_memory)
    mix, est, _ = run_session(sess, live)
    n = min(len(mix), *(len(v) for v in est.values()), *(len(r) for r in refs.values()))
    est, mix, refs = {k: v[:n] for k, v in est.items()}, mix[:n], {k: v[:n] for k, v in refs.items()}
    targets = a.target or list(refs)
    unknown = [t for t in targets if t not in refs]
    if unknown:
        raise SystemExit(f"unknown target(s) {unknown}; session sources: {list(refs)}")

    results = []
    print(f"{a.session}  (replayed, simulated={bool(sess.meta.get('simulated'))}, merge_ild_db={merge}, "
          f"{len(refs)} sources, signal {n / SR:.1f} s)")
    print(f"pipeline lag not included below: a hop is final {live.H / SR:.1f} s after it ends "
          f"(window end), plus compute mean {np.mean(live.compute_ms):.0f} ms/step")
    for t0 in (float(x) for x in a.t0.split(",")):
        for tg in targets:
            r = measure(est, mix, refs, tg, t0, a.threshold_db, a.window_ms / 1000, budget_s=a.budget_s)
            results.append(r)
            f = lambda v, u="s": "never" if v is None else f"{v:g} {u}"
            verdict = "MET" if r["met_within_budget"] else "NOT MET"
            print(f"t0={t0:g}s target={tg:6s} stream={r['stream']}  first>={a.threshold_db:g}dB {f(r['first_reach_s'])}  "
                  f"stable {f(r['stable_s'])}  steady {f(r['steady_margin_db'], 'dB')} "
                  f"(min {f(r['steady_min_db'], 'dB')}, mix {f(r['mix_margin_db'], 'dB')})  "
                  f"ok {r['frac_windows_ok']}  unexplained {f(r['unexplained_db'], 'dB')} re target  "
                  f"carried_by {r['carried_by']}  [{verdict} within {a.budget_s:g} s]")
            print(f"    margin per {a.window_ms:g} ms window from t0 (dB): {r['margins_db']}")
    if a.json:
        a.json.write_text(json.dumps({"session": str(a.session), "evidence": "replayed",
                                      "merge_ild_db": merge, "results": results}, indent=1))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
