"""Predeclared configuration for representability and gated anchor escape."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from creativity_toy.config import STUDY_SEEDS


@dataclass
class SupervisedCapacityConfig:
    primary_dimension: int = 16
    dimensions: list[int] = field(default_factory=lambda: [2, 4, 8, 16])
    d16_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))
    dimension_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS[:10]))
    smoke_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS[:3]))
    starts_per_training_task: int = 32
    single_steps: int = 1_000
    multimode_steps: int = 1_500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    lambda_alpha: float = 0.10
    log_steps: list[int] = field(
        default_factory=lambda: [0, 25, 50, 100, 250, 500, 1_000, 1_500]
    )
    batch_sizes: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 960])
    throughput_warmup_steps: int = 5
    throughput_timing_steps: int = 20
    plateau_fraction: float = 0.95
    best_cosine_success: float = 0.90
    proposal_recall_success: float = 0.75


@dataclass
class AnchorEscapeConfig:
    lambda_values: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.03, 0.10, 0.30, 1.0]
    )
    validation_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS[:3]))
    main_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))
    candidate_budget: int = 100_000
    capture_distance: float = 3.0
    validation_decoy_tolerance: float = 0.10
    portfolio_escape_enabled: bool = False
    portfolio_min_escape: int = 6


@dataclass
class CapacityDiagnosticConfig:
    name: str = "supervised_representability_anchor_escape"
    supervised: SupervisedCapacityConfig = field(
        default_factory=SupervisedCapacityConfig
    )
    anchor_escape: AnchorEscapeConfig = field(default_factory=AnchorEscapeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_capacity_config() -> CapacityDiagnosticConfig:
    config = CapacityDiagnosticConfig(name="capacity_diagnostic_smoke")
    config.supervised.d16_seeds = list(config.supervised.smoke_seeds)
    config.supervised.dimension_seeds = list(config.supervised.smoke_seeds)
    config.supervised.starts_per_training_task = 4
    config.supervised.single_steps = 120
    config.supervised.multimode_steps = 180
    config.supervised.log_steps = [0, 25, 50, 100, 120, 180]
    config.supervised.batch_sizes = [32, 64, 128]
    config.supervised.throughput_warmup_steps = 2
    config.supervised.throughput_timing_steps = 5
    config.anchor_escape.validation_seeds = list(config.supervised.smoke_seeds[:2])
    config.anchor_escape.main_seeds = list(config.supervised.smoke_seeds)
    config.anchor_escape.candidate_budget = 5_000
    return config
