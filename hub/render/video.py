"""Timestamped mask compositor (numpy only; no PIL/OpenCV dependency).

Three visually distinct states, encoded by outline style and text as well as colour:
  speaking / sounding   solid 2 px contour, translucent fill, "<NAME> SPEAKING" (or SOUNDING)
  possible match        dashed contour, lighter fill, "POSSIBLE: <NAME>"
  unattributed sound    a caption bar "UNATTRIBUTED SOUND: <LABEL>" (no mask is highlighted: a
                        failure to match is not evidence of where the sound is)
Silent or idle objects are never filled or outlined. The motion-reduced style draws constant
fill with no glow modulation. Each overlay belongs to ONE frame (its presentation timestamp).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ingest.media_io import read_png, write_png

STATE_COLORS = {"speaking": (255, 214, 10), "sounding": (255, 214, 10), "possible": (80, 220, 255),
                "unattributed": (255, 120, 220)}

# 5x7 bitmap font, one string of 7 five-bit rows per glyph ("." = off, "#" = on).
_FONT_SRC = {
    "A": ".###.#...##...######...##...#", "B": "####.#...##...#####.#...##...#####.",
    "C": ".###.#...##....#....#....#...#.###.", "D": "####.#...##...##...##...##...#####.",
    "E": "#####.#....#....####.#....#....#####", "F": "#####.#....#....####.#....#....#....",
    "G": ".###.#...##....#.####...##...#.###.", "H": "#...##...##...#######...##...##...#",
    "I": ".###...#....#....#....#....#...###.", "J": "..###...#....#....#....#.#..#..##..",
    "K": "#...##..#.#.#..##...#.#..#..#.#...#", "L": "#....#....#....#....#....#....#####",
    "M": "#...###.###.#.##...##...##...##...#", "N": "#...###..##.#.##..###...##...##...#",
    "O": ".###.#...##...##...##...##...#.###.", "P": "####.#...##...#####.#....#....#....",
    "Q": ".###.#...##...##...##.#.##..#..##.#", "R": "####.#...##...#####.#.#..#..#.#...#",
    "S": ".####.....#.....###.....#.....#####.", "T": "#####..#....#....#....#....#....#...",
    "U": "#...##...##...##...##...##...#.###.", "V": "#...##...##...##...#.#.#..#.#...#...",
    "W": "#...##...##...##.#.##.#.###.###...#", "X": "#...##...#.#.#...#...#.#.#...##...#",
    "Y": "#...##...#.#.#...#....#....#....#...", "Z": "#####....#...#...#...#...#....#####",
    "0": ".###.#...##..###.#.###..##...#.###.", "1": "..#...##....#....#....#....#...###.",
    "2": ".###.#...#....#...#...#...#...#####", "3": "#####...#...#.....#.....##...#.###.",
    "4": "...#...##..#.#.#..#####...#....#...", "5": "#####.#....####.....#.....##...#.###.",
    "6": "..##..#...#....####.#...##...#.###.", "7": "#####....#...#...#....#....#....#...",
    "8": ".###.#...##...#.###.#...##...#.###.", "9": ".###.#...##...#.####....#...#..##...",
    ":": ".......##.....##.......", " ": "", "-": ".........................###........",
    "(": "...#...#...#....#....#.....#.....#", ")": "#.....#.....#....#....#...#...#...",
    "?": ".###.#...#....#...#....#.......#...", ".": "...............................##.##",
}

# Hand-checked rows override the packed strings above for glyphs that rendered poorly.
_ROWS = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    ":": [".....", "..#..", "..#..", ".....", "..#..", "..#..", "....."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    ".": [".....", ".....", ".....", ".....", ".....", "..##.", "..##."],
    "(": ["...#.", "..#..", "..#..", "..#..", "..#..", "..#..", "...#."],
    ")": [".#...", "..#..", "..#..", "..#..", "..#..", "..#..", ".#..."],
    "?": [".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."],
}
_FONT_SRC.update({k: "".join(v) for k, v in _ROWS.items()})


def _glyph(ch: str) -> np.ndarray:
    s = _FONT_SRC.get(ch.upper(), _FONT_SRC["?"])
    if not s:
        return np.zeros((7, 5), bool)
    s = (s + "." * 35)[:35]
    return np.array([[c == "#" for c in s[r * 5:(r + 1) * 5]] for r in range(7)])


def text_width(text: str, scale: int = 1) -> int:
    return len(text) * 6 * scale


def draw_text(img: np.ndarray, x: int, y: int, text: str, color, scale: int = 1, bg=None) -> None:
    H, W = img.shape[:2]
    if bg is not None:
        x0, x1, y0, y1 = max(x - 2, 0), min(x + text_width(text, scale) + 1, W), max(y - 2, 0), min(y + 7 * scale + 2, H)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = (0.25 * img[y0:y1, x0:x1] + 0.75 * np.array(bg)).astype(np.uint8)
    for i, ch in enumerate(text):
        g = np.kron(_glyph(ch), np.ones((scale, scale), bool))
        gx, gy = x + i * 6 * scale, y
        h, w = g.shape
        if gx >= W or gy >= H or gx + w <= 0 or gy + h <= 0:
            continue
        sx0, sy0 = max(0, -gx), max(0, -gy)
        ex, ey = min(w, W - gx), min(h, H - gy)
        sub = g[sy0:ey, sx0:ex]
        img[gy + sy0:gy + ey, gx + sx0:gx + ex][sub] = color


def contour(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    er = mask.copy()
    for _ in range(thickness):
        p = np.pad(er, 1)
        er = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~er


@dataclass
class OverlayItem:
    track_id: str
    label: str
    state: str                     # speaking | sounding | possible | idle
    mask: np.ndarray | None
    glow: float = 1.0


def overlay_frame(img: np.ndarray, items: list[OverlayItem], caption: str | None = None,
                  motion_reduced: bool = False) -> np.ndarray:
    out = img.copy()
    for it in items:
        if it.state == "idle" or it.mask is None or not it.mask.any():
            continue                                   # silent bystanders never glow
        col = np.array(STATE_COLORS["possible" if it.state == "possible" else "speaking"], np.float32)
        base = 0.18 if it.state == "possible" else 0.28
        alpha = base + (0.0 if motion_reduced else 0.30 * float(np.clip(it.glow, 0, 1)))
        out[it.mask] = ((1 - alpha) * out[it.mask] + alpha * col).astype(np.uint8)
        edge = contour(it.mask)
        if it.state == "possible":                     # dashed
            ys, xs = np.nonzero(edge)
            keep = ((xs // 4 + ys // 4) % 2) == 0
            out[ys[keep], xs[keep]] = col.astype(np.uint8)
        else:
            out[edge] = col.astype(np.uint8)
        ys, xs = np.nonzero(it.mask)
        tag = f"POSSIBLE: {it.label}" if it.state == "possible" else (
            f"{it.label} {'SPEAKING' if it.state == 'speaking' else 'SOUNDING'}")
        tx = int(np.clip(xs.min(), 2, max(2, out.shape[1] - text_width(tag) - 2)))
        draw_text(out, tx, max(int(ys.min()) - 10, 2), tag, col.astype(np.uint8), bg=(10, 10, 16))
    if caption:
        col = STATE_COLORS["unattributed"]
        h = 15
        out[-h:] = (0.2 * out[-h:] + 0.8 * np.array((20, 10, 24))).astype(np.uint8)
        out[-h:-h + 1:, ::6] = col
        draw_text(out, 6, out.shape[0] - 11, caption, col)
    return out


def render_sequence(frame_paths: list[Path], overlays: list[tuple[list[OverlayItem], str | None]],
                    out_dir: str | Path, motion_reduced: bool = False) -> list[Path]:
    """Read PNG frames, composite each frame's own overlays and write an annotated PNG sequence."""
    out_dir = Path(out_dir)
    written = []
    for src, (items, caption) in zip(frame_paths, overlays):
        dst = out_dir / Path(src).name
        write_png(dst, overlay_frame(read_png(src), items, caption, motion_reduced))
        written.append(dst)
    return written
