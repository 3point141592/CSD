"""Predeclared configuration for diagnosing dimensional failure."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from creativity_toy.config import STUDY_SEEDS


@dataclass
class FailureDiagnosticConfig:
    name: str = "csd_failure_diagnostics"
    dimensions: list[int] = field(default_factory=lambda: [2, 4, 8, 16])
    gradient_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))
    dimension_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS[:10]))
    budget_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS[:5]))
    smoke_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS[:3]))
    candidate_budgets: list[int] = field(
        default_factory=lambda: [100_000, 300_000, 1_000_000]
    )
    encounter_budgets: list[int] = field(
        default_factory=lambda: [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    )
    quality_thresholds: list[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5, 0.7]
    )
    quality_quantiles: list[float] = field(
        default_factory=lambda: [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    )
    batch_sizes: list[int] = field(
        default_factory=lambda: [
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16_384,
            32_768,
            65_536,
            131_072,
            262_144,
        ]
    )
    batch_warmup_steps: int = 3
    batch_timing_steps: int = 8
    batch_plateau_fraction: float = 0.95
    random_hit_outcomes: int = 100_000
    direct_gradient_budget: int = 100_000
    direct_gradient_learning_rate: float = 3e-4
    direct_gradient_weight_decay: float = 1e-4
    direct_gradient_batch_size: int = 12
    direct_gradient_clip: float = 1.0
    contrastive_alpha: float = 3.0
    geometry_seed_count: int = 10
    capture_distance_bins: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 1e9]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_failure_config() -> FailureDiagnosticConfig:
    config = FailureDiagnosticConfig(name="csd_failure_diagnostics_smoke")
    config.gradient_seeds = list(config.smoke_seeds)
    config.dimension_seeds = list(config.smoke_seeds)
    config.budget_seeds = list(config.smoke_seeds[:2])
    config.candidate_budgets = [5_000, 10_000]
    config.encounter_budgets = [1_000, 2_500, 5_000]
    config.direct_gradient_budget = 5_000
    config.batch_sizes = [128, 512, 2048]
    config.batch_warmup_steps = 1
    config.batch_timing_steps = 2
    config.random_hit_outcomes = 5_000
    config.geometry_seed_count = 3
    return config
