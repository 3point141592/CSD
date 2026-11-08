"""Detached simulator boundary for scalar useful-novelty rewards."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from creativity_toy.config_v2 import TrainingV2Config

from .config_rl import RLRewardConfig
from .task_batch import TorchTaskBatch


@dataclass
class BlackBoxOutcome:
    refined: torch.Tensor
    quality: torch.Tensor
    novelty: torch.Tensor
    quality_gate: torch.Tensor
    reward: torch.Tensor
    candidate_evaluations: int
    q_equivalent_evaluations: int


class BlackBoxLandscapeEnvironment:
    """Treat Q/refinement as an external simulator with no autograd contract."""

    def __init__(
        self,
        *,
        refinement_steps: int,
        refinement_rate: float,
        reward: RLRewardConfig,
    ) -> None:
        self.refinement_steps = int(refinement_steps)
        self.refinement_rate = float(refinement_rate)
        self.reward_config = reward

    @classmethod
    def from_configs(
        cls, training: TrainingV2Config, reward: RLRewardConfig
    ) -> BlackBoxLandscapeEnvironment:
        return cls(
            refinement_steps=training.refinement_steps,
            refinement_rate=training.refinement_rate,
            reward=reward,
        )

    def evaluate(
        self, candidates: torch.Tensor, tasks: TorchTaskBatch
    ) -> BlackBoxOutcome:
        """Return only detached outcomes; never expose dQ/dcandidate."""
        with torch.no_grad():
            action = candidates.detach()
            refined = tasks.refine(
                action,
                steps=self.refinement_steps,
                rate=self.refinement_rate,
            ).detach()
            quality = tasks.quality(refined).detach()
            pairwise = torch.cdist(refined, refined, p=2)
            count = refined.shape[1]
            novelty = (
                torch.sum(pairwise, dim=-1) / max(count - 1, 1)
            ) / self.reward_config.diversity_scale
            novelty = torch.clamp(novelty, max=self.reward_config.novelty_clip).detach()
            gate = torch.sigmoid(
                self.reward_config.quality_gate_beta
                * (quality - self.reward_config.quality_threshold)
            ).detach()
            scalar_reward = (
                quality + self.reward_config.lambda_novelty * gate * novelty
            ).detach()
        candidate_evaluations = int(candidates.shape[0] * candidates.shape[1])
        return BlackBoxOutcome(
            refined=refined,
            quality=quality,
            novelty=novelty,
            quality_gate=gate,
            reward=scalar_reward,
            candidate_evaluations=candidate_evaluations,
            q_equivalent_evaluations=candidate_evaluations
            * (self.refinement_steps + 1),
        )
