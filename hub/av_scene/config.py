"""Typed loader for the packaged TOML defaults (tomllib) and the calibrated array example."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from render.envelope import EnvelopeConfig
from replay.residual import ResidualConfig
from separate.diarization import AnchorConfig
from who.matcher import MatcherConfig

from .cache import hash_config
from .contracts import ChunkConfig, MemoryBudgets, MemoryCaps, Switches
from .schemas import ArrayCalibration, CameraModel

CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass
class AcousticsConfig:
    ambient_temperature_c: float = 20.0
    temperature_provenance: str = "assumed"
    r_min_m: float = 0.5


@dataclass
class TimingConfig:
    gate_early_ms: float = 15.0
    gate_late_ms: float = 45.0
    skew_base_ms: float = 20.0


@dataclass
class SpatialConfig:
    wdo_min_weight: float = 0.1
    nfft: int = 1024
    hop: int = 512
    interp: int = 16
    narrowband_ratio: float = 1.0
    onset_window_ms: float = 20.0
    onset_sustain_weight: float = 0.2


@dataclass
class OffAxisConfig:
    yaw_limit_deg: float = 60.0
    unknown_pose: str = "allow"


@dataclass
class AvConfig:
    ffmpeg: str = ""
    acoustics: AcousticsConfig = field(default_factory=AcousticsConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    off_axis: OffAxisConfig = field(default_factory=OffAxisConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    memory: MemoryCaps = field(default_factory=MemoryCaps)
    budgets: MemoryBudgets = field(default_factory=MemoryBudgets)
    switches: Switches = field(default_factory=Switches)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hash_config(self.to_dict())


_SECTIONS = {"acoustics": AcousticsConfig, "timing": TimingConfig, "spatial": SpatialConfig,
             "envelope": EnvelopeConfig, "residual": ResidualConfig, "anchors": AnchorConfig,
             "matcher": MatcherConfig, "off_axis": OffAxisConfig, "chunking": ChunkConfig,
             "memory": MemoryCaps, "budgets": MemoryBudgets, "switches": Switches}


def _build(cls, data: dict, where: str):
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown key(s) in [{where}]: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None = None) -> AvConfig:
    """Load `default.toml` (or `path`). Unknown sections/keys are errors, not silently ignored."""
    p = Path(path) if path else CONFIG_DIR / "default.toml"
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    kwargs = {}
    for name, data in raw.items():
        if name == "media":
            unknown = set(data) - {"ffmpeg"}
            if unknown:
                raise ValueError(f"unknown key(s) in [media]: {sorted(unknown)}")
            kwargs["ffmpeg"] = data.get("ffmpeg", "")
        elif name in _SECTIONS:
            kwargs[name] = _build(_SECTIONS[name], data, name)
        else:
            raise ValueError(f"unknown config section [{name}]")
    cfg = AvConfig(**kwargs)
    if cfg.acoustics.temperature_provenance not in ("assumed", "measured"):
        raise ValueError("acoustics.temperature_provenance must be 'assumed' or 'measured'")
    if cfg.off_axis.unknown_pose not in ("allow", "block"):
        raise ValueError("off_axis.unknown_pose must be 'allow' or 'block'")
    if cfg.switches.diarization not in ('off', 'auto', 'on'):
        raise ValueError("switches.diarization must be 'off', 'auto' or 'on'")
    cfg.chunking.validate()
    return cfg


def load_array_calibration(path: str | Path | None = None) -> ArrayCalibration:
    p = Path(path) if path else CONFIG_DIR / "array_example.toml"
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    cal = dict(raw["calibration"])
    cam = raw.get("camera")
    if cam:
        cam = dict(cam)
        r, t = cam.pop("r_cam_to_array", None), cam.pop("t_cam_to_array_m", None)
        if r is not None:
            cal["r_cam_to_array"] = r
        if t is not None:
            cal["t_cam_to_array_m"] = t
        cal["camera"] = CameraModel(**cam)
    return ArrayCalibration(**cal)
