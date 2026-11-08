"""Predeclared configuration for the isolated non-oracle 32D experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from creativity_toy.config import STUDY_SEEDS


@dataclass
class LandscapeRepresentationConfig:
    dimension: int = 32
    candidates: int = 8
    latent_dimension: int = 3
    training_geometries: int = 80
    starts_per_task: int = 32
    supervised_steps: int = 1_500
    batch_size: int = 960
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    lambda_alpha: float = 0.10
    alpha_max: float = 6.0
    task_token_dimension: int = 32
    component_hidden_dimension: int = 64
    component_token_dimension: int = 32
    proposal_hidden_dimension: int = 16
    seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))
    curve_steps: list[int] = field(
        default_factory=lambda: [0, 25, 50, 100, 250, 500, 1_000, 1_500]
    )
    conditions: list[str] = field(
        default_factory=lambda: [
            "TaskOnly",
            "TaskOnlyParameterMatched",
            "LandscapeDeepSets",
            "LandscapeAttention",
        ]
    )
    evaluation_conditions: list[str] = field(
        default_factory=lambda: [
            "TaskOnly",
            "TaskOnlyParameterMatched",
            "LandscapeDeepSets",
            "LandscapeAttention",
            "LandscapeAttentionShuffled",
            "OracleTargetDirection",
        ]
    )
    shifted_variants: list[str] = field(
        default_factory=lambda: [
            "rotated",
            "translated",
            "width_shift",
            "more_peaks",
            "more_decoys",
            "new_goals",
            "unseen",
        ]
    )


@dataclass
class ComponentProbeConfig:
    hidden_dimension: int = 32
    steps: int = 750
    batch_size: int = 2_048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class LandscapeRepresentationStudyConfig:
    name: str = "d32_nonoracle_landscape_representation"
    representation: LandscapeRepresentationConfig = field(
        default_factory=LandscapeRepresentationConfig
    )
    component_probe: ComponentProbeConfig = field(default_factory=ComponentProbeConfig)
    paired_cosine_effect_threshold: float = 0.03
    paired_recall_effect_threshold: float = 0.10
    component_probe_chance_auroc_band: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_landscape_representation_config() -> LandscapeRepresentationStudyConfig:
    config = LandscapeRepresentationStudyConfig(
        name="d32_nonoracle_landscape_representation_smoke"
    )
    config.representation.training_geometries = 10
    config.representation.starts_per_task = 4
    config.representation.supervised_steps = 150
    config.representation.batch_size = 128
    config.representation.seeds = list(STUDY_SEEDS[:3])
    config.representation.curve_steps = [0, 25, 50, 100, 150]
    config.component_probe.steps = 100
    config.component_probe.batch_size = 256
    return config
