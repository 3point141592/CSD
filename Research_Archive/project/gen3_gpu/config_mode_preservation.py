"""Predeclared configuration for generator mode-preservation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ModeAllocation = Literal["shared_joint", "candidate_specific"]


@dataclass
class ModeRewardConfig:
    lambda_mode: float = 0.0
    allocation: ModeAllocation = "shared_joint"
    quality_gated: bool = True
    epsilon: float = 1e-8


@dataclass
class ModeSweepConfig:
    lambda_modes: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.03, 0.10, 0.30, 1.0]
    )
    seeds: int = 3
    candidate_evaluations: int = 25_000
    stochastic_validation_rollouts: int = 3
    quality_tolerance: float = 0.02
    decoy_tolerance: float = 0.02


@dataclass
class ModePreservationConfig:
    name: str = "mode_preservation"
    reward: ModeRewardConfig = field(default_factory=ModeRewardConfig)
    sweep: ModeSweepConfig = field(default_factory=ModeSweepConfig)
    ablation_seeds: int = 6
    representative_seed: int = 7
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


def smoke_mode_config() -> ModePreservationConfig:
    config = ModePreservationConfig(name="mode_preservation_smoke")
    config.sweep.lambda_modes = [0.0, 0.10]
    config.sweep.seeds = 2
    config.sweep.candidate_evaluations = 5_000
    config.sweep.stochastic_validation_rollouts = 1
    config.ablation_seeds = 2
    config.generalization_variants = ["iid", "rotated"]
    return config
