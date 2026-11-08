"""Configuration for the controlled Gen2 direction-normalization study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import STUDY_SEEDS
from .config_v2 import ExperimentV2Config, preset_v2


@dataclass
class NormalizationStudyConfig:
    """Wrap the unchanged V2 configuration with diagnostic-only settings."""

    name: str = "normalization_main"
    base: ExperimentV2Config = field(default_factory=ExperimentV2Config)
    alpha_values: list[float] = field(
        default_factory=lambda: [
            0.0,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0,
        ]
    )
    equivariance_trials: int = 3
    translation_magnitude: float = 2.0
    normalization_eps: float = 1e-8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_normalization(name: str) -> NormalizationStudyConfig:
    if name == "main":
        config = NormalizationStudyConfig(base=preset_v2("main"))
        config.base.run_ablations = False
        config.base.run_generalization = False
        return config
    if name == "smoke":
        config = NormalizationStudyConfig(
            name="normalization_smoke", base=preset_v2("smoke")
        )
        config.base.seeds = list(STUDY_SEEDS[:2])
        config.base.run_ablations = False
        config.base.run_generalization = False
        config.equivariance_trials = 1
        return config
    raise ValueError(
        f"Unknown normalization preset {name!r}; choose 'main' or 'smoke'."
    )
