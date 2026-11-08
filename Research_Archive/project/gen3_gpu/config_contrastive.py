"""Configuration for the isolated 2D contrastive-steering benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ContrastiveDataConfig:
    samples_per_region: int = 48
    sample_scale: float = 0.45
    background_radius_scale: float = 1.15
    rejection_batch: int = 256
    max_rejection_batches: int = 100


@dataclass
class ContrastiveSweepConfig:
    alphas: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    )
    seeds: int = 6
    quality_tolerance: float = 0.05
    decoy_tolerance: float = 0.05


@dataclass
class ContrastiveExperimentConfig:
    name: str = "contrastive_steering"
    dimension: int = 2
    candidate_count: int = 8
    direction_kinds: list[str] = field(
        default_factory=lambda: ["global", "goal", "target_anchor", "useful_decoy"]
    )
    proposal_forms: list[str] = field(default_factory=lambda: ["single", "k_radius"])
    data: ContrastiveDataConfig = field(default_factory=ContrastiveDataConfig)
    sweep: ContrastiveSweepConfig = field(default_factory=ContrastiveSweepConfig)
    representative_seeds: list[int] = field(default_factory=lambda: [7, 19])
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


def smoke_contrastive_config() -> ContrastiveExperimentConfig:
    config = ContrastiveExperimentConfig(name="contrastive_steering_smoke")
    config.data.samples_per_region = 12
    config.sweep.alphas = [0.0, 1.0, 3.0, 6.0]
    config.sweep.seeds = 2
    config.generalization_variants = ["iid", "rotated"]
    config.representative_seeds = [7]
    return config
