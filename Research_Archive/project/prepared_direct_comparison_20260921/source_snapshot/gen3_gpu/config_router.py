"""Predeclared configuration for the frozen-generator router experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RouterSweepConfig:
    lambda_novelty: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 1.0]
    )
    quality_thresholds: list[float] = field(
        default_factory=lambda: [0.50, 0.60, 0.70, 0.80]
    )
    quality_gate_betas: list[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])
    selection_seeds: int = 3
    useful_weight: float = 0.25
    decoy_penalty: float = 0.25


@dataclass
class RouterExperimentConfig:
    name: str = "router_useful_novelty"
    ridge: float = 0.01
    diversity_scale: float = 5.0
    novelty_clip: float = 1.5
    novelty_normalization: str = "divide_by_diversity_scale_then_clip"
    target_epsilon: float = 1e-8
    pairwise_steps: int = 200
    pairwise_learning_rate: float = 0.03
    pairwise_weight_decay: float = 1e-4
    inference_repetitions: int = 50
    sweep: RouterSweepConfig = field(default_factory=RouterSweepConfig)
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


def smoke_router_config() -> RouterExperimentConfig:
    config = RouterExperimentConfig(name="router_useful_novelty_smoke")
    config.sweep.lambda_novelty = [0.10, 0.50]
    config.sweep.quality_thresholds = [0.60, 0.70]
    config.sweep.quality_gate_betas = [10.0]
    config.sweep.selection_seeds = 2
    config.pairwise_steps = 30
    config.inference_repetitions = 5
    config.generalization_variants = ["iid", "rotated"]
    return config
