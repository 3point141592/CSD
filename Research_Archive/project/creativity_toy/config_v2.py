"""Configuration for Generation-2 goal-directed divergence experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import STUDY_SEEDS


@dataclass
class LandscapeV2Config:
    dimension: int = 2
    useful_peaks: int = 3
    decoys: int = 5
    alternative_radius: float = 4.5
    decoy_radius: float = 7.0
    anchor_quality: float = 0.88
    target_quality: float = 0.98
    secondary_quality: float = 0.68
    decoy_quality_min: float = 0.24
    decoy_quality_max: float = 0.38
    useful_radial_width: float = 0.70
    useful_lateral_width: float = 0.42
    corridor_radial_width: float = 2.9
    corridor_lateral_width: float = 0.28
    corridor_target_quality: float = 0.30
    corridor_secondary_quality: float = 0.18
    decoy_radial_width: float = 1.55
    decoy_lateral_width: float = 1.20
    anchor_width: float = 0.65
    train_translation_scale: float = 0.8
    train_rotation_range: float = 0.30
    family_seed: int = 27_041
    useful_threshold: float = 0.62
    high_quality_threshold: float = 0.78
    region_mahalanobis_threshold: float = 2.5


@dataclass
class TrainingV2Config:
    alpha: float = 1.0
    candidates: int = 8
    latent_dimension: int = 3
    hidden_dimension: int = 16
    gen1_iterations: int = 45
    gen2_iterations: int = 65
    population: int = 14
    batch_size: int = 12
    learning_rate: float = 0.035
    sigma_start: float = 0.24
    sigma_end: float = 0.045
    gradient_clip: float = 8.0
    refinement_steps: int = 9
    refinement_rate: float = 0.28
    quality_weight: float = 1.0
    best_quality_weight: float = 0.20
    raw_quality_weight: float = 0.08
    novelty_weight: float = 0.25
    diversity_weight: float = 0.65
    step_weight: float = 0.08
    low_quality_distance_weight: float = 0.18
    regularization_weight: float = 0.002
    quality_gate: float = 0.62
    gate_temperature: float = 0.07
    diversity_beta: float = 7.0
    semantic_scale: float = 4.5
    diversity_scale: float = 5.0
    random_radius: float = 4.8
    pure_divergence_radius: float = 7.0
    critic_ridge: float = 0.01
    critic_temperature: float = 0.12
    alpha_regularization_weight: float = 0.0
    alpha_reference: float = 3.0


@dataclass
class ExperimentV2Config:
    name: str = "main"
    seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))
    train_geometries: int = 10
    test_geometries: int = 5
    trials_per_task: int = 2
    ablation_seeds: int = 6
    alpha_values: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25]
    )
    k_values: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    run_ablations: bool = True
    run_generalization: bool = True
    landscape: LandscapeV2Config = field(default_factory=LandscapeV2Config)
    training: TrainingV2Config = field(default_factory=TrainingV2Config)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_v2(name: str) -> ExperimentV2Config:
    if name == "main":
        return ExperimentV2Config()
    if name == "smoke":
        cfg = ExperimentV2Config(name="smoke")
        cfg.seeds = list(STUDY_SEEDS[:2])
        cfg.train_geometries = 5
        cfg.test_geometries = 2
        cfg.trials_per_task = 1
        cfg.ablation_seeds = 1
        cfg.training.gen1_iterations = 18
        cfg.training.gen2_iterations = 24
        cfg.training.population = 10
        cfg.training.batch_size = 8
        return cfg
    raise ValueError(f"Unknown V2 preset {name!r}; choose 'main' or 'smoke'.")
