"""Batched copy of the unchanged goal-conditional ridge critic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from creativity_toy.config_v2 import TrainingV2Config

from .operator import BatchedCreativeOperator
from .task_batch import TorchTaskBatch


def critic_features(
    h: torch.Tensor, candidates: torch.Tensor, goal: torch.Tensor
) -> torch.Tensor:
    """Exact critic features for [B,N,d] candidate tensors."""
    batch, count, dimension = candidates.shape
    delta = candidates - h[:, None, :]
    candidate_norm = torch.linalg.vector_norm(candidates, dim=-1)
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    alignment = torch.sum(delta * goal[:, None, :], dim=-1)
    cosine = alignment / torch.clamp(
        delta_norm * torch.linalg.vector_norm(goal, dim=-1)[:, None], min=1e-12
    )
    return torch.cat(
        [
            torch.ones(
                (batch, count, 1), dtype=candidates.dtype, device=candidates.device
            ),
            h[:, None, :].expand(batch, count, dimension),
            candidates,
            goal[:, None, :].expand(batch, count, dimension),
            delta,
            candidates * goal[:, None, :],
            delta * goal[:, None, :],
            candidates * candidates,
            delta * delta,
            candidate_norm.unsqueeze(-1),
            (candidate_norm * candidate_norm).unsqueeze(-1),
            delta_norm.unsqueeze(-1),
            (delta_norm * delta_norm).unsqueeze(-1),
            alignment.unsqueeze(-1),
            (alignment * alignment).unsqueeze(-1),
            (alignment * delta_norm).unsqueeze(-1),
            cosine.unsqueeze(-1),
        ],
        dim=-1,
    )


@dataclass
class BatchedRidgeCritic:
    coefficients: torch.Tensor
    ridge: float
    temperature: float

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return torch.matmul(features, self.coefficients)

    @property
    def parameter_count(self) -> int:
        return int(self.coefficients.numel())


def make_numpy_broad_candidates(
    starts: np.ndarray,
    *,
    candidates: int,
    pure_divergence_radius: float,
    seed: int,
) -> np.ndarray:
    """Generate broad critic inputs in the reference loop/RNG call order."""
    rng = np.random.default_rng(seed)
    broad_count = max(2 * candidates, 8)
    rows: list[np.ndarray] = []
    for h in starts:
        directions = rng.normal(size=(broad_count, h.size))
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
        )
        radii = rng.uniform(0.0, pure_divergence_radius, size=broad_count)
        rows.append(h[None, :] + radii[:, None] * directions)
    return np.stack(rows)


def make_torch_native_broad_candidates(
    starts: torch.Tensor,
    *,
    candidates: int,
    pure_divergence_radius: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=starts.device)
    generator.manual_seed(seed)
    broad_count = max(2 * candidates, 8)
    rows: list[torch.Tensor] = []
    # Keep per-task draw order so vectorization does not change seeded inputs.
    # This preparation step is outside the candidate-scoring hot path.
    for h in starts:
        directions = torch.randn(
            (broad_count, h.numel()),
            generator=generator,
            device=starts.device,
            dtype=starts.dtype,
        )
        directions = directions / torch.clamp(
            torch.linalg.vector_norm(directions, dim=-1, keepdim=True), min=1e-12
        )
        radii = (
            torch.rand(
                broad_count,
                generator=generator,
                device=starts.device,
                dtype=starts.dtype,
            )
            * pure_divergence_radius
        )
        rows.append(h[None, :] + radii[:, None] * directions)
    return torch.stack(rows)


def fit_critic_batched(
    operator: BatchedCreativeOperator,
    theta: torch.Tensor,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    config: TrainingV2Config,
    *,
    seed: int,
    broad_candidates: np.ndarray | torch.Tensor | None = None,
) -> tuple[BatchedRidgeCritic, dict[str, torch.Tensor]]:
    learned = operator.forward(theta, starts, tasks.goals).candidates.squeeze(0)
    if broad_candidates is None:
        broad_candidates = make_numpy_broad_candidates(
            starts.detach().cpu().to(torch.float64).numpy(),
            candidates=operator.candidates,
            pure_divergence_radius=config.pure_divergence_radius,
            seed=seed,
        )
    broad = (
        broad_candidates.to(device=starts.device, dtype=starts.dtype)
        if isinstance(broad_candidates, torch.Tensor)
        else torch.as_tensor(
            np.ascontiguousarray(broad_candidates),
            dtype=starts.dtype,
            device=starts.device,
        )
    )
    candidates = torch.cat([learned, broad], dim=1)
    refined = tasks.refine(
        candidates,
        steps=config.refinement_steps,
        rate=config.refinement_rate,
    )
    features = critic_features(starts, candidates, tasks.goals)
    targets = tasks.quality(refined)
    x = features.reshape(-1, features.shape[-1])
    y = targets.reshape(-1)
    identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    identity[0, 0] = 0.0
    coefficients = torch.linalg.solve(
        x.T @ x + config.critic_ridge * identity,
        x.T @ y,
    )
    critic = BatchedRidgeCritic(
        coefficients=coefficients,
        ridge=config.critic_ridge,
        temperature=config.critic_temperature,
    )
    return critic, {
        "features": features,
        "targets": targets,
        "candidates": candidates,
        "refined": refined,
    }
