"""Frozen candidate batches and tiny quality/useful-novelty routers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from creativity_toy.normalization_diagnostics import spearman_rank_correlation

from .config_router import RouterExperimentConfig
from .critic import critic_features
from .runtime import synchronize

FeatureMode = Literal["local", "set_aware"]
TargetMode = Literal["quality", "novelty_only", "ungated", "gated"]


@dataclass
class FrozenCandidateBatch:
    starts: torch.Tensor
    goals: torch.Tensor
    candidates: torch.Tensor
    refined: torch.Tensor
    raw_quality: torch.Tensor
    quality: torch.Tensor
    regions: torch.Tensor
    alphas: torch.Tensor
    task_ids: torch.Tensor
    trials: torch.Tensor
    n_useful: torch.Tensor
    target_regions: torch.Tensor
    novelty: torch.Tensor
    source: str

    @property
    def batch_size(self) -> int:
        return int(self.candidates.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.candidates.shape[1])


@dataclass
class LinearRouter:
    coefficients: torch.Tensor
    feature_mode: FeatureMode
    target_mode: str
    ridge: float
    training_seconds: float
    peak_allocated_bytes: int

    @property
    def parameter_count(self) -> int:
        return int(self.coefficients.numel())

    def scores(
        self, batch: FrozenCandidateBatch, config: RouterExperimentConfig
    ) -> torch.Tensor:
        return torch.matmul(
            router_features(batch, self.feature_mode, config), self.coefficients
        )


def normalized_candidate_novelty(
    candidates: torch.Tensor,
    *,
    diversity_scale: float,
    novelty_clip: float,
) -> torch.Tensor:
    """Mean sibling Euclidean distance, divided by scale and clipped."""
    count = candidates.shape[-2]
    pairwise = torch.cdist(candidates, candidates, p=2)
    novelty = torch.sum(pairwise, dim=-1) / max(count - 1, 1)
    return torch.clamp(novelty / diversity_scale, max=novelty_clip)


def router_features(
    batch: FrozenCandidateBatch,
    mode: FeatureMode,
    config: RouterExperimentConfig,
) -> torch.Tensor:
    local = critic_features(batch.starts, batch.candidates, batch.goals)
    if mode == "local":
        return local
    centroid = torch.mean(batch.candidates, dim=1)
    centered = batch.candidates - centroid[:, None, :]
    centroid_distance = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
    set_mean_novelty = torch.mean(batch.novelty, dim=1, keepdim=True)
    relational = torch.cat(
        [
            centroid[:, None, :].expand(-1, batch.candidate_count, -1),
            centered,
            centroid_distance / config.diversity_scale,
            batch.novelty.unsqueeze(-1),
            set_mean_novelty[:, :, None].expand(-1, batch.candidate_count, -1),
        ],
        dim=-1,
    )
    return torch.cat([local, relational], dim=-1)


def routing_target(
    batch: FrozenCandidateBatch,
    *,
    target_mode: TargetMode,
    lambda_novelty: float,
    quality_threshold: float,
    quality_gate_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate = torch.sigmoid(quality_gate_beta * (batch.quality - quality_threshold))
    gated_novelty = gate * batch.novelty
    if target_mode == "quality":
        target = batch.quality
    elif target_mode == "novelty_only":
        target = batch.novelty
    elif target_mode == "ungated":
        target = batch.quality + lambda_novelty * batch.novelty
    elif target_mode == "gated":
        target = batch.quality + lambda_novelty * gated_novelty
    else:
        raise ValueError(f"unknown target mode {target_mode!r}")
    return target.detach(), gated_novelty.detach()


def _ridge_solve(
    features: torch.Tensor, targets: torch.Tensor, ridge: float
) -> torch.Tensor:
    x = features.reshape(-1, features.shape[-1])
    y = targets.reshape(-1)
    identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    identity[0, 0] = 0.0
    return torch.linalg.solve(x.T @ x + ridge * identity, x.T @ y)


def fit_regression_router(
    batches: Sequence[FrozenCandidateBatch],
    *,
    feature_mode: FeatureMode,
    target_mode: TargetMode,
    lambda_novelty: float,
    quality_threshold: float,
    quality_gate_beta: float,
    config: RouterExperimentConfig,
) -> LinearRouter:
    feature_batches = [
        router_features(batch, feature_mode, config) for batch in batches
    ]
    features = torch.cat(
        [value.reshape(-1, value.shape[-1]) for value in feature_batches], dim=0
    )
    targets = torch.cat(
        [
            routing_target(
                batch,
                target_mode=target_mode,
                lambda_novelty=lambda_novelty,
                quality_threshold=quality_threshold,
                quality_gate_beta=quality_gate_beta,
            )[0].reshape(-1)
            for batch in batches
        ],
        dim=0,
    )
    device = features.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = perf_counter()
    coefficients = _ridge_solve(features, targets, config.ridge)
    synchronize(device)
    elapsed = perf_counter() - started
    return LinearRouter(
        coefficients=coefficients.detach(),
        feature_mode=feature_mode,
        target_mode=f"{target_mode}_regression",
        ridge=config.ridge,
        training_seconds=elapsed,
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    )


def _pairwise_examples(
    batch: FrozenCandidateBatch,
    feature_mode: FeatureMode,
    target: torch.Tensor,
    config: RouterExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = router_features(batch, feature_mode, config)
    count = batch.candidate_count
    upper = torch.triu_indices(count, count, offset=1, device=features.device)
    differences = features[:, upper[0], :] - features[:, upper[1], :]
    target_difference = target[:, upper[0]] - target[:, upper[1]]
    keep = torch.abs(target_difference) > config.target_epsilon
    signs = torch.sign(target_difference[keep])
    return differences[keep], signs


def fit_pairwise_router(
    batches: Sequence[FrozenCandidateBatch],
    *,
    feature_mode: FeatureMode,
    lambda_novelty: float,
    quality_threshold: float,
    quality_gate_beta: float,
    config: RouterExperimentConfig,
) -> LinearRouter:
    regression = fit_regression_router(
        batches,
        feature_mode=feature_mode,
        target_mode="gated",
        lambda_novelty=lambda_novelty,
        quality_threshold=quality_threshold,
        quality_gate_beta=quality_gate_beta,
        config=config,
    )
    examples, labels = [], []
    for batch in batches:
        target, _ = routing_target(
            batch,
            target_mode="gated",
            lambda_novelty=lambda_novelty,
            quality_threshold=quality_threshold,
            quality_gate_beta=quality_gate_beta,
        )
        x, y = _pairwise_examples(batch, feature_mode, target, config)
        examples.append(x)
        labels.append(y)
    x = torch.cat(examples, dim=0)
    y = torch.cat(labels, dim=0)
    coefficients = torch.nn.Parameter(regression.coefficients.clone())
    optimizer = torch.optim.AdamW(
        [coefficients],
        lr=config.pairwise_learning_rate,
        weight_decay=config.pairwise_weight_decay,
    )
    device = coefficients.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = perf_counter()
    for _ in range(config.pairwise_steps):
        loss = F.softplus(-y * torch.matmul(x, coefficients)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    synchronize(device)
    elapsed = perf_counter() - started
    return LinearRouter(
        coefficients=coefficients.detach(),
        feature_mode=feature_mode,
        target_mode="gated_pairwise_ranking",
        ridge=config.ridge,
        training_seconds=regression.training_seconds + elapsed,
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    )


def target_distribution(
    batches: Sequence[FrozenCandidateBatch],
    *,
    lambda_novelty: float,
    quality_threshold: float,
    quality_gate_beta: float,
) -> dict[str, float]:
    quality = torch.cat([batch.quality.reshape(-1) for batch in batches])
    novelty = torch.cat([batch.novelty.reshape(-1) for batch in batches])
    gates = torch.sigmoid(quality_gate_beta * (quality - quality_threshold))
    gated = gates * novelty
    target = quality + lambda_novelty * gated
    output: dict[str, float] = {}
    for name, value in [
        ("quality", quality),
        ("novelty", novelty),
        ("gated_novelty", gated),
        ("useful_novelty_target", target),
    ]:
        output.update(
            {
                f"{name}_mean": float(torch.mean(value).cpu()),
                f"{name}_sd": float(torch.std(value, correction=0).cpu()),
                f"{name}_min": float(torch.amin(value).cpu()),
                f"{name}_max": float(torch.amax(value).cpu()),
            }
        )
    return output


def mean_spearman(scores: torch.Tensor, target: torch.Tensor) -> float:
    values = [
        spearman_rank_correlation(
            scores[index].detach().cpu().numpy(),
            target[index].detach().cpu().numpy(),
        )
        for index in range(scores.shape[0])
    ]
    return float(np.mean(values))
