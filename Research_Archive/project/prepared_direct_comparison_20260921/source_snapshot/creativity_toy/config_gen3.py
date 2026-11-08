"""Configuration for Generation-3 learned-magnitude experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import STUDY_SEEDS
from .config_v2 import ExperimentV2Config, preset_v2


@dataclass
class Gen3StudyConfig:
    name: str = "gen3_learned_alpha"
    base: ExperimentV2Config = field(default_factory=ExperimentV2Config)
    alpha_max: float = 6.0
    alpha_initial: float = 3.0
    alpha_hidden_dimension: int = 32
    alpha_modes: list[str] = field(
        default_factory=lambda: [
            "global",
            "contextual_linear",
            "contextual_mlp",
        ]
    )
    unnormalized_results: str = "results/v2_normalization_main"
    fixed_alpha_results: str = "results/v2_normalization_alpha_3"
    old_norm_results: str = "results/v2_normalization_main"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_gen3(name: str) -> Gen3StudyConfig:
    if name == "main":
        config = Gen3StudyConfig(base=preset_v2("main"))
        config.base.training.alpha = 3.0
        config.base.run_ablations = False
        config.base.run_generalization = False
        return config
    if name == "smoke":
        config = Gen3StudyConfig(name="gen3_smoke", base=preset_v2("smoke"))
        config.base.seeds = list(STUDY_SEEDS[:2])
        config.base.training.alpha = 3.0
        config.base.run_ablations = False
        config.base.run_generalization = False
        return config
    raise ValueError(f"Unknown Gen3 preset {name!r}; choose 'main' or 'smoke'.")
