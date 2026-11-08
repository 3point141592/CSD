"""Vectorized quality-gated repulsion for fixed-size proposal portfolios."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModeRewardBatch:
    set_reward: torch.Tensor
    candidate_reward: torch.Tensor
    quality_gate: torch.Tensor
    cosine: torch.Tensor
    distance: torch.Tensor
    pair_weight: torch.Tensor


@torch.no_grad()
def quality_gated_directional_repulsion(
    direction: torch.Tensor,
    quality: torch.Tensor,
    *,
    quality_threshold: float,
    quality_gate_beta: float,
    quality_gated: bool,
    epsilon: float = 1e-8,
) -> ModeRewardBatch:
    """Return detached set and candidate-specific rewards for [B,K,d]."""

    if direction.ndim != 3 or quality.shape != direction.shape[:2]:
        raise ValueError("direction must be [B,K,d] and quality must be [B,K]")
    count = direction.shape[1]
    cosine = torch.matmul(direction.detach(), direction.detach().transpose(-1, -2))
    distance = torch.clamp(1.0 - cosine, min=0.0, max=2.0)
    if quality_gated:
        gate = torch.sigmoid(quality_gate_beta * (quality.detach() - quality_threshold))
    else:
        gate = torch.ones_like(quality)
    off_diagonal = ~torch.eye(
        count, dtype=torch.bool, device=direction.device
    ).unsqueeze(0)
    pair_weight = gate.unsqueeze(-1) * gate.unsqueeze(-2)
    pair_weight = torch.where(off_diagonal, pair_weight, torch.zeros_like(pair_weight))
    weighted_distance = pair_weight * distance
    set_reward = torch.sum(weighted_distance, dim=(-2, -1)) / (
        torch.sum(pair_weight, dim=(-2, -1)) + epsilon
    )
    candidate_reward = torch.sum(weighted_distance, dim=-1) / (
        torch.sum(pair_weight, dim=-1) + epsilon
    )
    return ModeRewardBatch(
        set_reward=set_reward.detach(),
        candidate_reward=candidate_reward.detach(),
        quality_gate=gate.detach(),
        cosine=cosine.detach(),
        distance=distance.detach(),
        pair_weight=pair_weight.detach(),
    )


def batch_relative_advantage(
    reward: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """Normalize one joint portfolio reward across independent task groups."""

    with torch.no_grad():
        centered = reward - torch.mean(reward)
        scale = torch.std(reward, correction=0)
        return (centered / (scale + epsilon)).detach()


def broadcast_set_reward(
    base_reward: torch.Tensor, set_reward: torch.Tensor, lambda_mode: float
) -> torch.Tensor:
    return (
        base_reward.detach() + float(lambda_mode) * set_reward.detach().unsqueeze(-1)
    ).detach()
