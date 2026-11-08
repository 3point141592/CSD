"""Configuration for the controlled semantic divergence experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

STUDY_SEEDS = [
    7,
    19,
    43,
    71,
    101,
    131,
    163,
    197,
    229,
    263,
    293,
    331,
    367,
    401,
    439,
    479,
    521,
    563,
    607,
    653,
    701,
    751,
    809,
    863,
    919,
    977,
    1031,
    1091,
    1153,
    1217,
]


@dataclass
class LandscapeConfig:
    dimension: int = 32
    intrinsic_dimension: int = 6
    train_peaks: int = 5
    peak_radius: float = 4.0
    peak_width: float = 0.75
    initial_noise: float = 0.20
    family_seed: int = 1729
    max_family_peaks: int = 10


@dataclass
class TrainingConfig:
    alpha_train: float = 1.0
    single_iterations: int = 55
    multi_iterations: int = 80
    population: int = 16
    batch_size: int = 8
    learning_rate: float = 0.030
    sigma_start: float = 0.28
    sigma_end: float = 0.050
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    gradient_clip: float = 8.0
    refinement_steps: int = 14
    # Width-squared scaling keeps this rate comparable across peak-width variants.
    refinement_rate: float = 0.32
    quality_weight: float = 1.0
    raw_quality_weight: float = 0.05
    novelty_weight: float = 1.00
    diversity_weight: float = 1.20
    semantic_weight: float = 0.05
    regularization_weight: float = 0.002
    quality_gate: float = 0.65
    gate_temperature: float = 0.08
    semantic_scale: float = 5.0
    diversity_scale: float = 5.0


@dataclass
class ExperimentConfig:
    name: str = "quick"
    seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))
    train_landscapes: int = 18
    test_landscapes: int = 10
    trials_per_landscape: int = 3
    heads: int = 5
    alpha_values: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    )
    quality_threshold: float = 0.75
    low_ranks: list[int] = field(default_factory=lambda: [4, 8, 16])
    run_low_rank: bool = True
    run_2d_demo: bool = True
    landscape: LandscapeConfig = field(default_factory=LandscapeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset(name: str) -> ExperimentConfig:
    """Return a named, reproducible experiment preset."""
    if name == "quick":
        return ExperimentConfig()
    if name == "large":
        cfg = ExperimentConfig(name="large")
        cfg.train_landscapes = 64
        cfg.test_landscapes = 30
        cfg.trials_per_landscape = 6
        cfg.training.single_iterations = 120
        cfg.training.multi_iterations = 180
        cfg.training.population = 32
        cfg.training.batch_size = 24
        return cfg
    raise ValueError(f"Unknown preset {name!r}; choose 'quick' or 'large'.")
