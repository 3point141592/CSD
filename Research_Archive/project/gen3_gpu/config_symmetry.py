"""Predeclared configuration for the isolated d=32 symmetry experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from creativity_toy.config import STUDY_SEEDS


@dataclass
class SymmetryAugmentationConfig:
    rotation_pool_size: int = 4_096
    translation_radius_max: float = 6.0
    distribution: str = "Haar_SO32_pool_x_uniform_radial_translation"


@dataclass
class SymmetryStudyConfig:
    name: str = "d32_landscape_attention_symmetry"
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
    invariant_component_hidden_dimension: int = 313
    invariant_component_token_dimension: int = 64
    seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))
    curve_steps: list[int] = field(
        default_factory=lambda: [0, 25, 50, 100, 250, 500, 1_000, 1_500]
    )
    conditions: list[str] = field(
        default_factory=lambda: [
            "CurrentAttention",
            "CurrentAttentionAugmented",
            "EquivariantAttention",
            "EquivariantAttentionAugmented",
        ]
    )
    primary_splits: list[str] = field(
        default_factory=lambda: [
            "iid",
            "exact_translated",
            "exact_rotated",
            "exact_rotation_translation",
        ]
    )
    secondary_splits: list[str] = field(
        default_factory=lambda: [
            "width_shift",
            "more_peaks",
            "more_decoys",
            "new_goals",
            "unseen",
            "family_translated",
            "family_rotated",
        ]
    )
    exact_translation_radius: float = 4.5
    coordinate_transfer_success_recall: float = 0.75
    exact_equivariance_cosine_threshold: float = 0.99999
    augmentation: SymmetryAugmentationConfig = field(
        default_factory=SymmetryAugmentationConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_symmetry_config() -> SymmetryStudyConfig:
    config = SymmetryStudyConfig(name="d32_landscape_attention_symmetry_smoke")
    config.training_geometries = 10
    config.starts_per_task = 4
    config.supervised_steps = 150
    config.batch_size = 128
    config.seeds = list(STUDY_SEEDS[:3])
    config.curve_steps = [0, 25, 50, 100, 150]
    config.augmentation.rotation_pool_size = 256
    return config
