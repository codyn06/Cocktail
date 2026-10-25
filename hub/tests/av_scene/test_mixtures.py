"""overlap3_ / pair2_ mixture cases and the listen sheet. SYNTHETIC assets; skipped if sessions are missing."""

import sys
from pathlib import Path

import numpy as np
import pytest

import av_scene  # noqa: F401  (registers FFmpeg DLLs before torchcodec)

EVAL = Path(__file__).resolve().parents[3] / "eval" / "av_scene"
sys.path.insert(0, str(EVAL))
import mixtures as X  # noqa: E402

# hub/sessions is gitignored; fall back to the main checkout when running from a worktree.
SESSIONS = X.SESSIONS
if not (SESSIONS / "compA" / "av" / "source_video.mp4").exists():
    SESSIONS = Path(r"C:\Users\nguye\repos\Cocktail\hub\sessions")


def _cases(prefix):
    if not (SESSIONS / "compA" / "av" / "source_video.mp4").exists():
        pytest.skip("session assets missing")
    cs = X.build_cases(SESSIONS, include=[prefix])
    if not cs:
        pytest.skip(f"no {prefix} cases (assets missing)")
    return cs


def test_overlap3_structure():
    (c,) = _cases("overlap3_")
    assert set(c.refs) == set(c.masks) == {"A", "B", "C"} and len(c.targets) == 3
    assert all(len(r) == len(c.audio) for r in c.refs.values())
    T = c.frames.shape[0]
    assert all(m.shape == (T, c.frames.shape[2], c.frames.shape[3]) for m in c.masks.values())
    assert int(sum(m.int() for m in c.masks.values()).max()) == 1       # disjoint
    rms = [float(np.sqrt(np.mean(r.astype(np.float64) ** 2))) for r in c.refs.values()]
    assert max(rms) / min(rms) < 1.01
    for t in c.targets:
        assert t.ref in c.refs and t.other in c.refs and t.other != t.ref
    assert all(c.spans[k] for k in "ABC")


def test_pair2_audio_only_cases():
    cs = {c.case_id: c for c in _cases("pair2_")}
    c = cs["pair2_overlap"]
    assert c.meta["video"] is False and len(c.audio) == len(c.refs["V1"]) == len(c.refs["V2"])
    assert np.allclose(c.audio, c.refs["V1"] + c.refs["V2"], atol=1e-6)
    assert int((c.masks["V1"] & c.masks["V2"]).sum()) == 0


def test_listen_sheet(tmp_path):
    _cases("overlap3_")
    import listen_sheet as L
    import soundfile as sf

    ids = L.write_sheet(tmp_path, SESSIONS, ["overlap3_"])
    assert ids == ["overlap3_side_ABC"]
    d = tmp_path / "overlap3_side_ABC"
    for f in ("mixture.wav", "ref_A.wav", "ref_B.wav", "ref_C.wav"):
        assert (d / f).exists()
    info = sf.info(d / "mixture.wav")
    assert info.samplerate == 48000 and info.subtype == "PCM_16"
    assert "overlap3_side_ABC" in (tmp_path / "listen_sheet.md").read_text(encoding="utf-8")
