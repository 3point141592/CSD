"""Configuration for black-box group-relative REINFORCE training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RLRewardConfig:
    lambda_novelty: float = 0.65
    quality_threshold: float = 0.62
    quality_gate_beta: float = 14.285714285714286
    diversity_scale: float = 5.0
    novelty_clip: float = 1.5


@dataclass
class RLTrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 12
    max_candidate_evaluations: int = 100_000
    checkpoint_budgets: list[int] = field(
        default_factory=lambda: [0, 10_000, 25_000, 50_000, 100_000]
    )
    direction_std_init: float = 0.20
    alpha_std_init: float = 0.35
    direction_std_min: float = 0.01
    direction_std_max: float = 1.0
    alpha_std_min: float = 0.01
    alpha_std_max: float = 2.0
    entropy_coefficient: float = 1e-3
    gradient_clip: float = 1.0
    advantage_epsilon: float = 1e-8
    nearly_deterministic_direction_cosine: float = 0.999
    nearly_deterministic_alpha_delta: float = 0.01
    log_every_steps: int = 10
    reward: RLRewardConfig = field(default_factory=RLRewardConfig)


@dataclass
class RLSweepConfig:
    learning_rates: list[float] = field(
        default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3]
    )
    weight_decays: list[float] = field(default_factory=lambda: [0.0, 1e-4, 1e-3])
    seeds: int = 3
    candidate_evaluations: int = 25_000
    stochastic_validation_rollouts: int = 3


@dataclass
class RLStudyConfig:
    name: str = "rl_adamw_main"
    primary_alpha_mode: str = "contextual_linear"
    alpha_max: float = 6.0
    alpha_hidden_dimension: int = 32
    validation_geometries: int = 5
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    sweep: RLSweepConfig = field(default_factory=RLSweepConfig)
    generalization_variants: list[str] = field(
        default_factory=lambda: [
            "iid",
            "rotated",
            "translated",
            "width_shift",
            "more_peaks",
            "more_decoys",
            "new_goals",
            "unseen",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_rl_config() -> RLStudyConfig:
    config = RLStudyConfig(name="rl_adamw_smoke")
    config.training.max_candidate_evaluations = 5_000
    config.training.checkpoint_budgets = [0, 1_000, 2_500, 5_000]
    config.sweep.learning_rates = [3e-4]
    config.sweep.weight_decays = [1e-4]
    config.sweep.seeds = 2
    config.sweep.candidate_evaluations = 2_500
    config.sweep.stochastic_validation_rollouts = 1
    config.validation_geometries = 2
    return config
