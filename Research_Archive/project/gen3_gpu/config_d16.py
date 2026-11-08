"""Fixed protocol for the isolated, full-rank d=16 benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from creativity_toy.config import STUDY_SEEDS
from creativity_toy.config_gen3 import Gen3StudyConfig, preset_gen3

from .config_contrastive import ContrastiveExperimentConfig
from .config_mode_preservation import ModeRewardConfig
from .config_rl import RLStudyConfig
from .config_router import RouterExperimentConfig


@dataclass
class D16BenchmarkConfig:
    name: str = "highdim_d16"
    dimension: int = 16
    seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS))
    smoke_seeds: list[int] = field(default_factory=lambda: list(STUDY_SEEDS[:3]))
    contrastive_validation_seeds: int = 6
    mode_lambda: float = 0.10
    router_lambda_novelty: float = 0.10
    router_quality_threshold: float = 0.50
    router_quality_gate_beta: float = 10.0
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
    base: Gen3StudyConfig = field(default_factory=lambda: preset_gen3("main"))
    rl: RLStudyConfig = field(default_factory=RLStudyConfig)
    mode: ModeRewardConfig = field(
        default_factory=lambda: ModeRewardConfig(
            lambda_mode=0.10,
            allocation="shared_joint",
            quality_gated=True,
        )
    )
    contrastive: ContrastiveExperimentConfig = field(
        default_factory=ContrastiveExperimentConfig
    )
    router: RouterExperimentConfig = field(default_factory=RouterExperimentConfig)

    def __post_init__(self) -> None:
        self.base.name = self.name
        self.base.base.name = self.name
        self.base.base.landscape.dimension = self.dimension
        self.base.base.training.candidates = 8
        self.base.base.training.refinement_steps = 9
        self.base.base.run_ablations = False
        self.base.base.run_generalization = False
        self.contrastive.name = self.name
        self.contrastive.dimension = self.dimension
        self.contrastive.candidate_count = 8
        self.contrastive.direction_kinds = ["goal"]
        self.contrastive.proposal_forms = ["single", "k_radius"]
        self.contrastive.sweep.seeds = self.contrastive_validation_seeds
        self.contrastive.generalization_variants = list(self.generalization_variants)
        self.mode.lambda_mode = self.mode_lambda

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_d16_config() -> D16BenchmarkConfig:
    config = D16BenchmarkConfig(name="highdim_d16_smoke")
    config.seeds = list(config.smoke_seeds)
    config.base.base.train_geometries = 5
    config.base.base.test_geometries = 2
    config.base.base.trials_per_task = 1
    config.rl.validation_geometries = 2
    config.rl.training.max_candidate_evaluations = 5_000
    config.rl.training.checkpoint_budgets = [0, 1_000, 2_500, 5_000]
    config.contrastive.data.samples_per_region = 12
    config.contrastive.sweep.seeds = min(2, len(config.seeds))
    config.router.pairwise_steps = 30
    return config
