"""Evaluate the alert rules (eval/alerts_rules.json) without a headset.

    # sound rules on a label timeline (what the hub's sound labeler outputs), no GPU needed:
    uv run --project hub python eval/alerts_check.py --labels labels.csv
    # wearer's name on speech text with times:
    uv run --project hub python eval/alerts_check.py --transcript speech.csv --name Sadie
    # a recording (needs the gpu group: torch + transformers, and the AST checkpoint download):
    uv run --project hub python eval/alerts_check.py --wav laptop_mic.wav --device cpu
    ... --expect smoke_alarm,siren      # exit 1 unless exactly these rules fire (use --expect none)

Input files (CSV with a header, '#' lines ignored)
  labels:     t_s,label,score[,bearing_deg]   one row per label per hop; t_s = start of the 1 s hop
              the score describes; label = a coarse label (alarm, siren, speech, ...) or an exact
              AudioSet class name; score 0..1 as the labeler gives it (after its smoothing).
  transcript: t_s,text[,bearing_deg]

Rules (see the notes in alerts_rules.json): a sound rule is active in a hop when its best matching
score >= min_conf; active hops separated by at most max_gap_s make one event; the rule fires when
the event has lasted min_duration_s, at the end of the hop that completes it, and again only after
cooldown_s. A keyword rule fires on a transcript line containing a keyword as a whole word, then
waits cooldown_s. Fires are printed in the shape of protocol.AlertFired.

Laptop mic: there is no --mic option because no audio-capture package (sounddevice) is a hub
dependency and none may be added here. Record the laptop mic to a WAV with any recorder (for
example Windows Voice Recorder), then use --wav.

Evidence: the rule engine is exercised by hub/tests/eval/test_alerts_check.py on hand-made label
timelines (simulated). --wav runs the real AST labeler and has NOT been run in this environment
(no torch installed here). No rule threshold has been tuned on a real recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RULES = HERE / "alerts_rules.json"
LABEL_SMOOTH = 0.6          # hub/separate/live.py LiveSeparator.label_smooth default
WIN_S, HOP_S = 2.0, 1.0     # the labeler window and hop in live.py
SR = 16000


@dataclass
class Fire:
    t_s: float
    rule: str
    label: str
    critical: bool
    spoken_text: str
    bearing_deg: float | None = None

    def as_alert(self) -> dict:
        """Shape of protocol.AlertFired plus the firing time and rule id."""
        return {"type": "alert.fired", "alert_id": f"{self.rule}@{self.t_s:g}", "label": self.label,
                "critical": self.critical, "spoken_text": self.spoken_text, "bearing_deg": self.bearing_deg,
                "t_s": self.t_s, "rule": self.rule}


@dataclass
class Obs:
    t_s: float
    label: str
    score: float
    bearing_deg: float | None = None


@dataclass
class Rules:
    hop_s: float
    rules: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- loading


def load_rules(path: Path, extra_names: list[str] | None = None) -> Rules:
    d = json.loads(Path(path).read_text())
    if d.get("schema") != "cocktail-alert-rules/1":
        raise ValueError(f"{path}: unknown schema {d.get('schema')!r}")
    rules = d.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"{path}: no rules")
    ids = set()
    for r in rules:
        rid = r.get("id")
        if not rid or rid in ids:
            raise ValueError(f"{path}: rule without a unique id: {r}")
        ids.add(rid)
        det = r.get("detector")
        need = {"sound_label": ("min_conf", "min_duration_s", "max_gap_s", "cooldown_s"),
                "keyword": ("keywords", "cooldown_s")}.get(det)
        if need is None:
            raise ValueError(f"rule {rid}: detector must be sound_label or keyword, got {det!r}")
        for k in ("label", "critical", "spoken_text") + need:
            if k not in r:
                raise ValueError(f"rule {rid}: missing {k!r}")
        if det == "sound_label":
            if not (r.get("labels") or r.get("classes")):
                raise ValueError(f"rule {rid}: needs labels or classes")
            if not 0 < r["min_conf"] <= 1:
                raise ValueError(f"rule {rid}: min_conf must be in (0, 1]")
        if det == "keyword" and extra_names:
            r["keywords"] = list(r["keywords"]) + list(extra_names)
    return Rules(float(d.get("hop_s", HOP_S)), rules)


def _rows(path: Path):
    with open(path, newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def _opt_float(v) -> float | None:
    return float(v) if v not in (None, "") else None


def read_labels(path: Path) -> list[Obs]:
    out = []
    for i, r in enumerate(_rows(path), 2):
        try:
            out.append(Obs(float(r["t_s"]), r["label"].strip(), float(r["score"]), _opt_float(r.get("bearing_deg"))))
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"{path} row {i}: need t_s,label,score[,bearing_deg] ({e})") from e
    return sorted(out, key=lambda o: o.t_s)


def read_transcript(path: Path) -> list[tuple[float, str, float | None]]:
    out = []
    for i, r in enumerate(_rows(path), 2):
        try:
            out.append((float(r["t_s"]), r["text"], _opt_float(r.get("bearing_deg"))))
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"{path} row {i}: need t_s,text[,bearing_deg] ({e})") from e
    return sorted(out)


# ---------------------------------------------------------------- the rule engine


def _fire(r: dict, t: float, bearing: float | None) -> Fire:
    return Fire(t, r["id"], r["label"], bool(r["critical"]), r["spoken_text"], bearing)


def eval_sound_rule(r: dict, obs: list[Obs], hop_s: float) -> list[Fire]:
    names = {n.lower() for n in list(r.get("labels", [])) + list(r.get("classes", []))}
    best: dict[float, Obs] = {}
    for o in obs:
        if o.label.lower() in names and (o.t_s not in best or o.score > best[o.t_s].score):
            best[o.t_s] = o
    fires: list[Fire] = []
    start = prev_end = last_fire = None
    for t in sorted(best):
        o = best[t]
        if o.score < r["min_conf"]:
            continue
        if prev_end is None or t - prev_end > r["max_gap_s"] + 1e-9:
            start = t                       # a new event
        prev_end = t + hop_s
        if prev_end - start + 1e-9 >= r["min_duration_s"] and (last_fire is None or prev_end - last_fire + 1e-9 >= r["cooldown_s"]):
            fires.append(_fire(r, prev_end, o.bearing_deg))
            last_fire = prev_end
    return fires


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return " ".join(re.findall(r"\w+", "".join(c for c in s if not unicodedata.combining(c))))


def eval_keyword_rule(r: dict, transcript: list[tuple[float, str, float | None]]) -> list[Fire]:
    kws = [k for k in (_norm(k) for k in r["keywords"]) if k]
    fires, last = [], None
    for t, text, brg in transcript:
        padded = f" {_norm(text)} "
        if any(f" {k} " in padded for k in kws) and (last is None or t - last + 1e-9 >= r["cooldown_s"]):
            fires.append(_fire(r, t, brg))
            last = t
    return fires


def evaluate(rules: Rules, obs: list[Obs], transcript: list[tuple[float, str, float | None]]) -> list[Fire]:
    fires: list[Fire] = []
    for r in rules.rules:
        if r["detector"] == "sound_label":
            fires += eval_sound_rule(r, obs, rules.hop_s)
        else:
            fires += eval_keyword_rule(r, transcript)
    return sorted(fires, key=lambda f: (f.t_s, f.rule))


# ---------------------------------------------------------------- WAV through the real labeler (needs the gpu group)


def wav_to_obs(path: Path, rules: Rules, device: str, smooth: float = LABEL_SMOOTH) -> list[Obs]:
    import wave

    import numpy as np
    from scipy.signal import resample_poly

    sys.path.insert(0, str(HERE.parent / "hub"))
    try:
        from separate.models import SoundLabeler
    except ImportError as e:  # torch / transformers not installed
        raise SystemExit(f"--wav needs the sound labeler (torch + transformers): uv sync --group gpu ({e})") from e
    with wave.open(str(path)) as w:
        ch, rate, width = w.getnchannels(), w.getframerate(), w.getsampwidth()
        if width != 2:
            raise SystemExit(f"{path}: need 16-bit PCM WAV")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, ch).astype(np.float32).mean(1) / 32768
    if rate != SR:
        from math import gcd
        g = gcd(rate, SR)
        x = resample_poly(x, SR // g, rate // g).astype(np.float32)
    lab = SoundLabeler(device)
    wanted = {c for r in rules.rules if r["detector"] == "sound_label" for c in r.get("classes", [])}
    idx = {n: i for i, n in enumerate(lab.names) if n in wanted}
    missing = wanted - set(idx)
    if missing:
        raise SystemExit(f"rule classes not in the labeler's label set: {sorted(missing)}")
    W, H = int(WIN_S * SR), int(HOP_S * SR)
    out: list[Obs] = []
    state: dict[str, float] = {}
    for s in range(0, max(1, len(x) - W + 1), H):
        clip = x[s:s + W]
        if len(clip) < W:
            clip = np.pad(clip, (0, W - len(clip)))
        p = lab.probs([clip])[0]
        new = dict(lab.coarse(p)) | {n: float(p[i]) for n, i in idx.items()}
        for k, v in new.items():
            state[k] = smooth * state.get(k, v) + (1 - smooth) * v
            out.append(Obs(s / SR, k, state[k]))
    return out


# ---------------------------------------------------------------- CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    ap.add_argument("--labels", type=Path, help="CSV t_s,label,score[,bearing_deg]")
    ap.add_argument("--transcript", type=Path, help="CSV t_s,text[,bearing_deg]")
    ap.add_argument("--wav", type=Path, help="run the real sound labeler on this WAV (gpu group)")
    ap.add_argument("--device", default="cpu", help="device for --wav (default cpu)")
    ap.add_argument("--name", action="append", default=[], help="wearer's name (repeatable; adds to the keyword rule)")
    ap.add_argument("--expect", help="comma list of rule ids that must fire (and no others); 'none' for no fires")
    ap.add_argument("--json", type=Path, help="write the fired alerts here")
    a = ap.parse_args(argv)
    if not (a.labels or a.transcript or a.wav):
        ap.error("give --labels, --transcript or --wav")
    try:
        rules = load_rules(a.rules, a.name)
        obs = (read_labels(a.labels) if a.labels else []) + (wav_to_obs(a.wav, rules, a.device) if a.wav else [])
        transcript = read_transcript(a.transcript) if a.transcript else []
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}")
        return 2

    print("rules:")
    for r in rules.rules:
        if r["detector"] == "sound_label":
            print(f"  {r['id']:12s} sound  {sorted(set(r.get('labels', [])))} score >= {r['min_conf']:g} for "
                  f">= {r['min_duration_s']:g} s (gap <= {r['max_gap_s']:g} s), cooldown {r['cooldown_s']:g} s")
        else:
            kws = r["keywords"]
            print(f"  {r['id']:12s} words  {kws if kws else 'NONE SET (give --name or fill keywords)'}, cooldown {r['cooldown_s']:g} s")
        if r["detector"] == "keyword" and not r["keywords"] and a.transcript:
            print(f"  warning: rule {r['id']} has no keywords, it cannot fire")
    fires = evaluate(rules, obs, transcript)
    print(f"\n{len(fires)} alert(s) fired:")
    for f in fires:
        b = f" bearing {f.bearing_deg:+.0f}" if f.bearing_deg is not None else ""
        print(f"  t={f.t_s:7.1f} s  {f.rule:12s} \"{f.spoken_text}\"{' CRITICAL' if f.critical else ''}{b}")
    if a.json:
        a.json.write_text(json.dumps([f.as_alert() for f in fires], indent=1))
    if a.expect is not None:
        want = set() if a.expect.strip().lower() == "none" else {s.strip() for s in a.expect.split(",") if s.strip()}
        got = {f.rule for f in fires}
        ok = want == got
        print(f"\nexpected {sorted(want) or 'none'}, got {sorted(got) or 'none'}: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
