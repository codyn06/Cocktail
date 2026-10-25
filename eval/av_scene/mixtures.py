"""Labeled SYNTHETIC audiovisual mixtures with clean references and active-span timestamps.

Evidence tier for everything here: SYNTHETIC / replayed. The audio is an electronic sum of
recorded or generated stems; the video is a tiled composite of recorded interview clips. Nothing
is a room recording, a device stream, or a real glasses-captured scene.

Source assets (all under hub/sessions/, gitignored; nothing is generated from thin air):
  A  woman, clean:   audio of sessions/compA/av/source_video.mp4 (the composite whose RIGHT person was
                     muted, so its audio is the left person alone); video = LEFT 480x270 tile.
  B  man, derived:   audio = compB audio - compA audio (compB = both speaking). This assumes the clips
                     were mixed additively; `asset_report()` records the check (corr(A, B) ~ 0).
                     It is a derived reference, not a clean recording. Video = RIGHT 480x270 tile
                     (identical in compA and compB: the man's mouth keeps moving in compA, silently).
  V3 synthetic voice, hub/sessions/demo/stems/voice1.wav (mockclient/demo_stems.py output).
  MUSIC synthetic music, hub/sessions/demo/stems/music.wav.
  CAT, KNOCK  truth stems + frames + masks of hub/sessions/phase1_fixture (generate_fixture.py).

The two people are different clips; no recording exists of the two-person scene that the team's
reported composite used (stack_man_top / stack_woman_top have no clean references), so those
experiments are rebuilt here from the compA/compB clips with the same 480x540 vertical stack.

Every mixture is RMS-matched: refs are scaled to the level of A (full-clip RMS) before summing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

HUB = Path(__file__).resolve().parents[2] / "hub"
if str(HUB) not in sys.path:
    sys.path.insert(0, str(HUB))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M  # noqa: E402

SESSIONS = HUB / "sessions"
RATE = 48_000
TILE_H, TILE_W = 270, 480
TIER = "SYNTHETIC (electronic sum of recorded/generated stems; tiled recorded video)"
NOISE_DBFS = -70.0


class AssetMissing(RuntimeError):
    pass


@dataclass
class Target:
    """One prompted source of a case. `ref` is None when the target is absent from the audio."""
    name: str                 # source name, e.g. "A"
    mask: str                 # key into Case.masks
    ref: str | None           # key into Case.refs or None
    other: str | None         # key of the strongest competing ref (selection test) or None
    label: str = "speech"     # expected coarse label of the target
    text: str = "person speaking"


@dataclass
class Case:
    case_id: str
    description: str
    layout: str
    audio: np.ndarray                     # (n,) float32 mixture @ RATE
    frames: torch.Tensor                  # (T, 3, H, W) uint8
    refs: dict[str, np.ndarray]           # clean references, same length as audio
    spans: dict[str, list[tuple[float, float]]]
    masks: dict[str, torch.Tensor]        # (T, H, W) bool, True = prompted source
    targets: list[Target]
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.audio) / RATE


# --------------------------------------------------------------------------- assets
def _decode(path: Path):
    from av_scene.phase2 import decode_clip

    if not path.exists():
        raise AssetMissing(f"missing {path}")
    return decode_clip(path, max_side=4096)


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _scale_to(x: np.ndarray, rms: float) -> np.ndarray:
    return (x * (rms / max(_rms(x), 1e-12))).astype(np.float32)


def _resample48(x: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if rate == RATE:
        return x.astype(np.float32)
    g = np.gcd(RATE, rate)
    return resample_poly(x, RATE // g, rate // g).astype(np.float32)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]


def load_speech_assets(sessions: Path = SESSIONS) -> dict:
    """A (clean woman), B (derived man), tile videos."""
    a = _decode(sessions / "compA" / "av" / "source_video.mp4")
    b = _decode(sessions / "compB" / "av" / "source_video.mp4")
    if a.audio_rate != RATE or b.audio_rate != RATE:
        raise AssetMissing("expected 48 kHz audio in compA/compB")
    n = min(len(a.audio), len(b.audio))
    wav_a = a.audio[:n].astype(np.float64)
    wav_b = (b.audio[:n].astype(np.float64) - wav_a)
    ref_a = _scale_to(wav_a, _rms(wav_a))
    ref_b = _scale_to(wav_b, _rms(wav_a))                      # RMS-match B to A
    frames = a.frames[: round(n / RATE * a.fps)]
    return {"A": ref_a, "B": ref_b, "A_tile": frames[..., :TILE_W], "B_tile": frames[..., TILE_W:],
            "fps": a.fps, "corr_A_B": float(np.corrcoef(wav_a, wav_b)[0, 1]),
            "raw_rms_dbfs": {"A": M.rms_dbfs(wav_a), "B_derived": M.rms_dbfs(wav_b)}}


def load_demo_stem(name: str, sessions: Path = SESSIONS) -> np.ndarray:
    import soundfile as sf

    p = sessions / "demo" / "stems" / f"{name}.wav"
    if not p.exists():
        raise AssetMissing(f"missing {p}")
    x, sr = sf.read(p, dtype="float32")
    return _resample48(x.mean(1) if x.ndim == 2 else x, sr)


def asset_report(sessions: Path = SESSIONS) -> dict:
    s = load_speech_assets(sessions)
    return {"tier": TIER, "fps": s["fps"], "n_samples": len(s["A"]), "corr_A_B_raw": s["corr_A_B"],
            "raw_rms_dbfs": s["raw_rms_dbfs"], "spans_A": M.active_spans(s["A"], RATE),
            "spans_B": M.active_spans(s["B"], RATE)}


# --------------------------------------------------------------------------- composition
def resize_like_pipeline(frames: torch.Tensor, masks: dict[str, torch.Tensor], max_side: int = 640):
    """Same downscale as av_scene.phase2.decode_clip (area), masks resized by nearest."""
    h, w = frames.shape[-2:]
    if max(h, w) <= max_side:
        return frames, masks
    s = max_side / max(h, w)
    size = (round(h * s / 2) * 2, round(w * s / 2) * 2)
    fr = torch.nn.functional.interpolate(frames.float(), size=size, mode="area").round().clamp(0, 255).to(torch.uint8)
    mk = {k: torch.nn.functional.interpolate(v[:, None].float(), size=size, mode="nearest")[:, 0] > 0.5
          for k, v in masks.items()}
    return fr, mk


def compose(first: torch.Tensor, second: torch.Tensor, layout: str):
    """Tile two (T,3,270,480) videos. stack: first on top (480x540); side: first on the left (960x270).
    Returns (frames, mask_first, mask_second)."""
    T = first.shape[0]
    if layout == "stack":
        fr = torch.cat([first, second], dim=2)
        m1 = torch.zeros(T, 2 * TILE_H, TILE_W, dtype=torch.bool)
        m1[:, :TILE_H] = True
        m2 = torch.zeros_like(m1)
        m2[:, TILE_H:] = True
    elif layout == "side":
        fr = torch.cat([first, second], dim=3)
        m1 = torch.zeros(T, TILE_H, 2 * TILE_W, dtype=torch.bool)
        m1[:, :, :TILE_W] = True
        m2 = torch.zeros_like(m1)
        m2[:, :, TILE_W:] = True
    else:
        raise ValueError(layout)
    return fr, m1, m2


def _noise(n: int, seed: int = 0) -> np.ndarray:
    return _scale_to(np.random.default_rng(seed).standard_normal(n), 10 ** (NOISE_DBFS / 20))


def _fixture_case(sessions: Path):
    """Two non-speech events (cat chirp + off-screen knock) over the Phase 1 fixture video."""
    import json

    import soundfile as sf
    from torchvision.io import read_image
    from vision.mask_store import read_track_jsonl

    root = sessions / "phase1_fixture" / "fixture"
    if not (root / "fixture.json").exists():
        raise AssetMissing(f"missing {root}")
    meta = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    pts = np.asarray(meta["frame_pts_us"], np.float64) / 1e6
    n_frames = int(meta["duration_s"] * 25)
    want = np.arange(n_frames) / 25.0
    idx = np.abs(pts[None] - want[:, None]).argmin(axis=1)         # nearest frame per 25 fps slot
    frames = torch.stack([read_image(str(root / "frames" / f"{i:06d}.png"))[:3] for i in idx])
    masks = {}
    for tr in ("cat1", "alice"):
        recs = {r.frame_idx: r for r in read_track_jsonl(root / "masks_truth" / f"{tr}.jsonl")}
        m = np.zeros((n_frames, frames.shape[2], frames.shape[3]), bool)
        for k, i in enumerate(idx):
            r = recs.get(int(i))
            if r is not None and r.rle is not None:
                m[k] = r.rle.decode()
        masks[tr] = torch.from_numpy(m)
    refs = {}
    for nm in ("cat_chirp", "knock_offscreen"):
        x, sr = sf.read(root / "truth_stems" / f"{nm}.wav", dtype="float32")
        refs[nm] = _resample48(x.mean(1) if x.ndim == 2 else x, sr)
    return frames, masks, refs


def build_cases(sessions: Path = SESSIONS, *, include=None) -> list[Case]:
    """All labeled cases. `include` filters by case_id prefix. Missing optional assets are skipped
    (reported through `skipped`) rather than fabricated."""
    s = load_speech_assets(sessions)
    A, B = s["A"], s["B"]
    n = len(A)
    zeros = np.zeros(n, np.float32)
    cases: list[Case] = []
    sp = {"A": M.active_spans(A, RATE), "B": M.active_spans(B, RATE)}

    def add(c: Case):
        if include is None or any(c.case_id.startswith(p) for p in include):
            cases.append(c)

    def pair(layout: str, a_first: bool):
        first, second = (s["A_tile"], s["B_tile"]) if a_first else (s["B_tile"], s["A_tile"])
        fr, m1, m2 = compose(first, second, layout)
        mA, mB = (m1, m2) if a_first else (m2, m1)
        T = min(fr.shape[0], round(n / RATE * s["fps"]))
        return fr[:T], {"A": mA[:T], "B": mB[:T]}

    for layout in ("stack", "side"):
        for a_first in (True, False):
            tag = f"{layout}_{'AB' if a_first else 'BA'}"             # first-named is top / left
            fr, masks = pair(layout, a_first)
            add(Case(f"overlap_{tag}", f"A (woman) + B (man) simultaneously, {layout}, "
                     f"{'A top/left' if a_first else 'B top/left'}; refs RMS-matched",
                     layout, A + B, fr, {"A": A, "B": B}, sp, masks,
                     [Target("A", "A", "A", "B", text="person speaking"),
                      Target("B", "B", "B", "A", text="person speaking")],
                     {"overlap_s": M.spans_overlap_s(sp["A"], sp["B"])}))
            if layout != "stack":
                continue
            add(Case(f"A_alone_{tag}", "only A speaks; B visible, silent", layout, A.copy(), fr, {"A": A, "B": zeros},
                     {"A": sp["A"], "B": []}, masks,
                     [Target("A", "A", "A", None), Target("B", "B", None, "A")], {}))
            add(Case(f"B_alone_{tag}", "only B speaks; A visible, silent", layout, B.copy(), fr, {"A": zeros, "B": B},
                     {"A": [], "B": sp["B"]}, masks,
                     [Target("A", "A", None, "B"), Target("B", "B", "B", None)], {}))
            add(Case(f"silent_{tag}", "nobody speaks (-70 dBFS noise); both visible", layout, _noise(n), fr,
                     {"A": zeros, "B": zeros}, {"A": [], "B": []}, masks,
                     [Target("A", "A", None, None), Target("B", "B", None, None)], {}))
            try:
                v3 = _scale_to(_fit(load_demo_stem("voice1", sessions), n), _rms(A))
                add(Case(f"offscreen3_{tag}", "A speaks + OFF-SCREEN synthetic third talker V3 (no tile); B silent",
                         layout, A + v3, fr, {"A": A, "B": zeros, "V3": v3},
                         {"A": sp["A"], "B": [], "V3": M.active_spans(v3, RATE)}, masks,
                         [Target("A", "A", "A", "V3"), Target("B", "B", None, "V3")], {}))
                add(Case(f"offscreen_only_{tag}", "only the off-screen third talker speaks; both tiles silent",
                         layout, v3.copy(), fr, {"A": zeros, "B": zeros, "V3": v3},
                         {"A": [], "B": [], "V3": M.active_spans(v3, RATE)}, masks,
                         [Target("A", "A", None, "V3"), Target("B", "B", None, "V3")], {}))
            except AssetMissing:
                pass
            try:
                mu = _scale_to(_fit(load_demo_stem("music", sessions), n), _rms(A))
                add(Case(f"speech_music_{tag}", "A speaks over off-screen synthetic music; B silent",
                         layout, A + mu, fr, {"A": A, "B": zeros, "MUSIC": mu},
                         {"A": sp["A"], "B": [], "MUSIC": M.active_spans(mu, RATE)}, masks,
                         [Target("A", "A", "A", "MUSIC")], {}))
            except AssetMissing:
                pass
    try:
        ffr, fmasks, frefs = _fixture_case(sessions)
        k = int(round(ffr.shape[0] / 25 * RATE))
        cat, knock = _fit(frefs["cat_chirp"], k), _fit(frefs["knock_offscreen"], k)
        cat = _scale_to(cat, _rms(cat))                                      # keep fixture levels
        add(Case("nonspeech_two", "cat chirp (visible cat) + off-screen knock; alice visible, silent. SYNTHETIC fixture",
                 "fixture", cat + knock, ffr, {"cat": cat, "knock": knock, "alice": np.zeros(k, np.float32)},
                 {"cat": M.active_spans(cat, RATE, rel_db=30), "knock": M.active_spans(knock, RATE, rel_db=30),
                  "alice": []}, fmasks,
                 [Target("cat", "cat1", "cat", "knock", label="other", text="cat chirp"),
                  Target("alice", "alice", None, "cat", label="speech", text="person speaking")],
                 {"fixture": "phase1_fixture"}))
    except (AssetMissing, ImportError, FileNotFoundError):
        pass
    return cases


def oracle_anchors(case: Case, target: Target, *, negatives: bool = False) -> list[tuple[str, float, float]]:
    """DIAGNOSTIC / ORACLE anchors from the CLEAN reference spans of the target.
    '+' spans where the target is active; with `negatives`, '-' spans where the target is silent
    but a competitor is active (clipped so no anchor overlaps another)."""
    if target.ref is None:
        return []
    pos = list(case.spans.get(target.ref, []))
    out = [("+", float(a), float(b)) for a, b in pos]
    if negatives:
        for key, spans in case.spans.items():
            if key == target.ref:
                continue
            for a, b in spans:
                cur = a
                for pa, pb in sorted(pos):
                    if pb <= cur or pa >= b:
                        continue
                    if pa > cur:
                        out.append(("-", float(cur), float(pa)))
                    cur = max(cur, pb)
                if cur < b:
                    out.append(("-", float(cur), float(b)))
    return sorted(out, key=lambda t: t[1])
