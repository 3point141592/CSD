"""Predeclared configuration for the isolated 32D generalization diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from creativity_toy.config import STUDY_SEEDS


@dataclass
class D32DataScalingConfig:
    dimension: int = 32
    geometry_counts: list[int] = field(default_factory=lambda: [10, 20, 40, 80])
    grid_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS[:10]))
    endpoint_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))
    starts_per_task: int = 32
    single_steps: int = 1_000
    multimode_steps: int = 1_500
    batch_size: int = 960
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    lambda_alpha: float = 0.10
    curve_steps: list[int] = field(
        default_factory=lambda: [0, 25, 50, 100, 250, 500, 1_000, 1_500]
    )


@dataclass
class D32ProbeConfig:
    training_geometries: int = 80
    steps: int = 1_000
    batch_size: int = 960
    hidden_dimension: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))


@dataclass
class D32IdentifiabilityConfig:
    nearest_neighbors: int = 5
    task_starts: int = 1
    local_variance_neighbors: int = 5


@dataclass
class D32SlotConfig:
    training_geometries: int = 80
    steps: int = 1_500
    batch_size: int = 960
    hidden_dimension: int = 16
    seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))


@dataclass
class D32EndpointAnchorConfig:
    lambda_values: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.03, 0.10, 0.30, 1.0]
    )
    validation_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS[:3]))
    main_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))
    candidate_budget: int = 100_000
    anchor_radius: float = 1.625
    capture_distance: float = 3.0
    decoy_tolerance: float = 0.10


@dataclass
class D32GeneralizationConfig:
    name: str = "d32_supervised_generalization_diagnostic"
    scaling: D32DataScalingConfig = field(default_factory=D32DataScalingConfig)
    probes: D32ProbeConfig = field(default_factory=D32ProbeConfig)
    identifiability: D32IdentifiabilityConfig = field(
        default_factory=D32IdentifiabilityConfig
    )
    slots: D32SlotConfig = field(default_factory=D32SlotConfig)
    endpoint_anchor: D32EndpointAnchorConfig = field(
        default_factory=D32EndpointAnchorConfig
    )
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
    batch_sizes: list[int] = field(
        default_factory=lambda: [128, 256, 512, 960, 1_920, 3_840, 7_680]
    )
    batch_warmup_steps: int = 3
    batch_timing_steps: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_d32_generalization_config() -> D32GeneralizationConfig:
    config = D32GeneralizationConfig(name="d32_generalization_smoke")
    smoke_seeds = list(STUDY_SEEDS[:3])
    config.scaling.geometry_counts = [10, 20]
    config.scaling.grid_seeds = smoke_seeds
    config.scaling.endpoint_seeds = smoke_seeds
    config.scaling.starts_per_task = 4
    config.scaling.single_steps = 120
    config.scaling.multimode_steps = 180
    config.scaling.batch_size = 128
    config.scaling.curve_steps = [0, 25, 50, 100, 120, 180]
    config.probes.training_geometries = 20
    config.probes.steps = 120
    config.probes.batch_size = 128
    config.probes.seeds = smoke_seeds
    config.slots.training_geometries = 20
    config.slots.steps = 180
    config.slots.batch_size = 128
    config.slots.seeds = smoke_seeds
    config.endpoint_anchor.validation_seeds = smoke_seeds[:2]
    config.endpoint_anchor.main_seeds = smoke_seeds
    config.endpoint_anchor.candidate_budget = 5_000
    config.batch_sizes = [64, 128, 256]
    config.batch_warmup_steps = 1
    config.batch_timing_steps = 3
    return config
