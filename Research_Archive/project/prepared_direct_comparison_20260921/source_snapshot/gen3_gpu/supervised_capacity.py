"""Privileged supervised representability controls for the frozen CSD architecture."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from time import perf_counter
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config_capacity_diagnostic import SupervisedCapacityConfig
from .operator import BatchedCreativeOperator
from .runtime import synchronize
from .task_batch import TorchTaskBatch
from .training_failure_diagnostics import differentiable_policy_candidates

SupervisionMode = Literal["single_target", "multimode"]
_ASSIGNMENT_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


class LinearDirectionPolicy(nn.Module):
    """Linear direction control with the unchanged contextual-linear alpha head."""

    def __init__(
        self,
        *,
        dimension: int,
        candidates: int,
        latent_dimension: int,
        hidden_dimension: int,
        seed: int,
        device: torch.device,
        alpha_max: float = 6.0,
    ) -> None:
        super().__init__()
        del hidden_dimension
        self.dimension = dimension
        self.candidates = candidates
        self.latent_dimension = latent_dimension
        self.input_dimension = 2 * dimension + latent_dimension
        self.alpha_input_dimension = 3 * dimension + latent_dimension
        self.alpha_max = float(alpha_max)
        rng = np.random.default_rng(seed)
        self.direction_weight = nn.Parameter(
            torch.as_tensor(
                rng.normal(scale=0.04, size=(dimension, self.input_dimension)),
                dtype=torch.float32,
                device=device,
            )
        )
        self.direction_bias = nn.Parameter(
            torch.zeros(dimension, dtype=torch.float32, device=device)
        )
        self.alpha_weight = nn.Parameter(
            torch.zeros(
                (1, self.alpha_input_dimension), dtype=torch.float32, device=device
            )
        )
        self.alpha_bias = nn.Parameter(
            torch.zeros(1, dtype=torch.float32, device=device)
        )
        reference = BatchedCreativeOperator(
            dimension=dimension,
            candidates=candidates,
            latent_dimension=latent_dimension,
            hidden_dimension=16,
            seed=seed,
            variant="gen3_contextual_linear_alpha",
            device=device,
            dtype=torch.float32,
            alpha_max=alpha_max,
        )
        self.register_buffer("codes", reference.codes.detach().clone())

    @property
    def architecture_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def _base_inputs(self, h: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        batch = len(h)
        return torch.cat(
            [
                h[:, None, :].expand(batch, self.candidates, self.dimension),
                goal[:, None, :].expand(batch, self.candidates, self.dimension),
                self.codes[None, :, :].expand(
                    batch, self.candidates, self.latent_dimension
                ),
            ],
            dim=-1,
        )

    def direction_means(self, h: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return F.linear(
            self._base_inputs(h, goal),
            self.direction_weight,
            self.direction_bias,
        )

    def alpha_means(
        self, h: torch.Tensor, goal: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        batch = len(h)
        inputs = torch.cat(
            [
                h[:, None, :].expand(batch, self.candidates, self.dimension),
                goal[:, None, :].expand(batch, self.candidates, self.dimension),
                self.codes[None, :, :].expand(
                    batch, self.candidates, self.latent_dimension
                ),
                direction,
            ],
            dim=-1,
        )
        return F.linear(inputs, self.alpha_weight, self.alpha_bias).squeeze(-1)


@dataclass
class SupervisedTrainingResult:
    model: nn.Module
    history: list[dict[str, float]]
    elapsed_seconds: float
    steps: int
    examples_seen: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


@dataclass
class SupervisedEvaluation:
    summary: dict[str, float]
    per_set: list[dict[str, float]]


def useful_targets(
    h: torch.Tensor, tasks: TorchTaskBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = tasks.useful_centers - h[:, None, :]
    alpha = torch.linalg.vector_norm(delta, dim=-1)
    direction = F.normalize(delta, p=2, dim=-1, eps=1e-8)
    target_direction = torch.gather(
        direction,
        1,
        tasks.goal_index[:, None, None].expand(-1, 1, h.shape[-1]),
    ).squeeze(1)
    target_alpha = torch.gather(alpha, 1, tasks.goal_index[:, None]).squeeze(1)
    return direction, alpha, target_direction, target_alpha


def exact_assignment_indices(
    pair_cost: torch.Tensor, target_count: int
) -> torch.Tensor:
    """Exact rectangular assignment for small K and M, entirely on device."""
    candidate_count = pair_cost.shape[1]
    key = (candidate_count, target_count, str(pair_cost.device))
    assignments = _ASSIGNMENT_CACHE.get(key)
    if assignments is None:
        assignments = torch.as_tensor(
            list(permutations(range(candidate_count), target_count)),
            dtype=torch.long,
            device=pair_cost.device,
        )
        _ASSIGNMENT_CACHE[key] = assignments
    target = torch.arange(target_count, device=pair_cost.device)
    all_costs = pair_cost[:, assignments, target].sum(dim=-1)
    best = torch.argmin(all_costs, dim=1)
    return assignments.index_select(0, best)


def supervised_objective(
    model: nn.Module,
    h: torch.Tensor,
    tasks: TorchTaskBatch,
    *,
    mode: SupervisionMode,
    lambda_alpha: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    candidates, directions, alphas = differentiable_policy_candidates(
        model, h, tasks.goals
    )
    useful_direction, useful_alpha, target_direction, target_alpha = useful_targets(
        h, tasks
    )
    if torch.any(tasks.n_useful != tasks.n_useful[0]):
        raise ValueError("exact supervised matching expects equal useful counts")
    target_count = int(tasks.n_useful[0].detach().cpu())
    cosine_all = torch.einsum("bkd,bmd->bkm", directions, useful_direction)
    if mode == "single_target":
        cosine = torch.sum(directions[:, 0] * target_direction, dim=-1)
        alpha_error = alphas[:, 0] - target_alpha
        direction_loss = torch.mean(1.0 - cosine)
        alpha_loss = torch.mean(alpha_error.square())
        matched_indices = torch.zeros((len(h), 1), dtype=torch.long, device=h.device)
        matched_target = tasks.goal_index[:, None]
        matched_cosine = cosine[:, None]
        matched_alpha_error = alpha_error[:, None]
    elif mode == "multimode":
        pair_alpha = alphas[:, :, None] - useful_alpha[:, None, :]
        pair_cost = 1.0 - cosine_all + lambda_alpha * pair_alpha.square()
        matched_indices = exact_assignment_indices(pair_cost, target_count)
        matched_target = torch.arange(target_count, device=h.device)[None, :].expand(
            len(h), -1
        )
        batch = torch.arange(len(h), device=h.device)[:, None]
        matched_cosine = cosine_all[batch, matched_indices, matched_target]
        matched_alpha_error = pair_alpha[batch, matched_indices, matched_target]
        direction_loss = torch.mean(1.0 - matched_cosine)
        alpha_loss = torch.mean(matched_alpha_error.square())
    else:
        raise ValueError(mode)
    loss = direction_loss + lambda_alpha * alpha_loss
    batch = torch.arange(len(h), device=h.device)[:, None]
    matched_candidate = candidates[batch, matched_indices]
    matched_center = tasks.useful_centers[batch, matched_target]
    proximity = torch.linalg.vector_norm(matched_candidate - matched_center, dim=-1)
    best_target = torch.amax(
        torch.einsum("bkd,bd->bk", directions, target_direction), dim=1
    )
    return loss, {
        "direction_loss": direction_loss,
        "alpha_loss": alpha_loss,
        "best_target_cosine": torch.mean(best_target),
        "best_useful_cosine": torch.mean(torch.amax(cosine_all, dim=(1, 2))),
        "mean_matched_cosine": torch.mean(matched_cosine),
        "mean_angular_error_degrees": torch.mean(
            torch.rad2deg(torch.acos(torch.clamp(matched_cosine, -1.0, 1.0)))
        ),
        "mean_alpha_absolute_error": torch.mean(torch.abs(matched_alpha_error)),
        "mean_matched_proximity": torch.mean(proximity),
        "candidate_zero_target_cosine": torch.mean(
            torch.sum(directions[:, 0] * target_direction, dim=-1)
        ),
    }


def _sample_indices(
    count: int,
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if batch_size >= count:
        return torch.randperm(count, generator=generator, device=device)
    return torch.randperm(count, generator=generator, device=device)[:batch_size]


@torch.no_grad()
def _record_metrics(
    model: nn.Module,
    h: torch.Tensor,
    tasks: TorchTaskBatch,
    *,
    mode: SupervisionMode,
    lambda_alpha: float,
    step: int,
    refinement_steps: int,
    refinement_rate: float,
    useful_threshold: float,
) -> dict[str, float]:
    loss, metrics = supervised_objective(
        model, h, tasks, mode=mode, lambda_alpha=lambda_alpha
    )
    candidates, _, _ = differentiable_policy_candidates(model, h, tasks.goals)
    refined = tasks.refine(candidates, steps=refinement_steps, rate=refinement_rate)
    quality = tasks.quality(refined)
    regions = tasks.regions(refined)
    useful = (
        (regions >= 1)
        & (regions <= tasks.n_useful[:, None])
        & (quality >= useful_threshold)
    )
    return {
        "step": float(step),
        "loss": float(loss.cpu()),
        "proposal_recall": float(torch.mean(torch.any(useful, dim=1).float()).cpu()),
        **{name: float(value.cpu()) for name, value in metrics.items()},
    }


def train_supervised(
    model: nn.Module,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    config: SupervisedCapacityConfig,
    *,
    mode: SupervisionMode,
    steps: int,
    batch_size: int,
    seed: int,
    refinement_steps: int = 9,
    refinement_rate: float = 0.28,
    useful_threshold: float = 0.62,
) -> SupervisedTrainingResult:
    device = starts.device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    log_steps = set(value for value in config.log_steps if value <= steps)
    log_steps.add(steps)
    history = [
        _record_metrics(
            model,
            starts,
            tasks,
            mode=mode,
            lambda_alpha=config.lambda_alpha,
            step=0,
            refinement_steps=refinement_steps,
            refinement_rate=refinement_rate,
            useful_threshold=useful_threshold,
        )
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = perf_counter()
    for step in range(1, steps + 1):
        indices = _sample_indices(
            len(starts), batch_size, generator=generator, device=device
        )
        batch_tasks = tasks.index_select(indices)
        h = starts.index_select(0, indices)
        loss, _ = supervised_objective(
            model,
            h,
            batch_tasks,
            mode=mode,
            lambda_alpha=config.lambda_alpha,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.gradient_clip
        )
        optimizer.step()
        if step in log_steps:
            history.append(
                _record_metrics(
                    model,
                    starts,
                    tasks,
                    mode=mode,
                    lambda_alpha=config.lambda_alpha,
                    step=step,
                    refinement_steps=refinement_steps,
                    refinement_rate=refinement_rate,
                    useful_threshold=useful_threshold,
                )
            )
    synchronize(device)
    elapsed = perf_counter() - started
    return SupervisedTrainingResult(
        model=model,
        history=history,
        elapsed_seconds=elapsed,
        steps=steps,
        examples_seen=steps * min(batch_size, len(starts)),
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        peak_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    )


def benchmark_supervised_batches(
    make_model,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    config: SupervisedCapacityConfig,
    *,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    device = starts.device
    for batch_size in config.batch_sizes:
        model = make_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        generator = torch.Generator(device=device).manual_seed(seed)

        def step() -> None:
            indices = torch.randint(
                len(starts),
                (batch_size,),
                generator=generator,
                device=device,
            )
            loss, _ = supervised_objective(
                model,
                starts.index_select(0, indices),
                tasks.index_select(indices),
                mode="single_target",
                lambda_alpha=config.lambda_alpha,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        for _ in range(config.throughput_warmup_steps):
            step()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        for _ in range(config.throughput_timing_steps):
            step()
        synchronize(device)
        elapsed = perf_counter() - started
        examples = batch_size * config.throughput_timing_steps
        rows.append(
            {
                "batch_size": batch_size,
                "elapsed_seconds": elapsed,
                "steps_per_second": config.throughput_timing_steps / elapsed,
                "examples_per_second": examples / elapsed,
                "peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
                "peak_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(device))
                    if device.type == "cuda"
                    else 0
                ),
            }
        )
    return rows


@torch.no_grad()
def evaluate_supervised(
    model: nn.Module,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    *,
    mode: SupervisionMode,
    lambda_alpha: float,
    refinement_steps: int,
    refinement_rate: float,
    useful_threshold: float,
    capture_distance: float = 3.0,
) -> SupervisedEvaluation:
    candidates, directions, alphas = differentiable_policy_candidates(
        model, starts, tasks.goals
    )
    useful_direction, useful_alpha, target_direction, target_alpha = useful_targets(
        starts, tasks
    )
    target_count = int(tasks.n_useful[0].detach().cpu())
    cosine_all = torch.einsum("bkd,bmd->bkm", directions, useful_direction)
    target_cosine = torch.einsum("bkd,bd->bk", directions, target_direction)
    if mode == "single_target":
        matched_indices = torch.zeros(
            (len(starts), 1), dtype=torch.long, device=starts.device
        )
        matched_target = tasks.goal_index[:, None]
    else:
        pair_alpha = alphas[:, :, None] - useful_alpha[:, None, :]
        pair_cost = 1.0 - cosine_all + lambda_alpha * pair_alpha.square()
        matched_indices = exact_assignment_indices(pair_cost, target_count)
        matched_target = torch.arange(target_count, device=starts.device)[
            None, :
        ].expand(len(starts), -1)
    batch = torch.arange(len(starts), device=starts.device)[:, None]
    matched_cosine = cosine_all[batch, matched_indices, matched_target]
    matched_alpha = alphas[batch, matched_indices]
    matched_alpha_target = useful_alpha[batch, matched_target]
    matched_candidates = candidates[batch, matched_indices]
    matched_centers = tasks.useful_centers[batch, matched_target]
    proximity = torch.linalg.vector_norm(matched_candidates - matched_centers, dim=-1)

    refined = tasks.refine(candidates, steps=refinement_steps, rate=refinement_rate)
    quality = tasks.quality(refined)
    regions = tasks.regions(refined)
    useful = (
        (regions >= 1)
        & (regions <= tasks.n_useful[:, None])
        & (quality >= useful_threshold)
    )
    target = (regions == tasks.target_region[:, None]) & (quality >= useful_threshold)
    decoy = regions > tasks.n_useful[:, None]
    anchor = regions == 0
    useful_counts = []
    for index in range(len(starts)):
        useful_counts.append(float(torch.unique(regions[index][useful[index]]).numel()))
    useful_count = torch.as_tensor(
        useful_counts, dtype=torch.float32, device=starts.device
    )
    singular = torch.linalg.svdvals(directions)
    energy = singular.square()
    probability = energy / torch.clamp(
        torch.sum(energy, dim=1, keepdim=True), min=1e-12
    )
    effective_rank = torch.exp(
        -torch.sum(
            torch.where(
                probability > 0,
                probability * torch.log(probability),
                torch.zeros_like(probability),
            ),
            dim=1,
        )
    )
    candidate_zero_target = target[:, 0]
    matched_angular = torch.rad2deg(torch.acos(torch.clamp(matched_cosine, -1.0, 1.0)))
    proposal = torch.any(useful, dim=1)
    summary = {
        "best_target_cosine": float(torch.mean(torch.amax(target_cosine, dim=1)).cpu()),
        "candidate_zero_target_cosine": float(torch.mean(target_cosine[:, 0]).cpu()),
        "best_useful_cosine": float(
            torch.mean(torch.amax(cosine_all, dim=(1, 2))).cpu()
        ),
        "mean_matched_cosine": float(torch.mean(matched_cosine).cpu()),
        "mean_angular_error_degrees": float(torch.mean(matched_angular).cpu()),
        "mean_alpha_absolute_error": float(
            torch.mean(torch.abs(matched_alpha - matched_alpha_target)).cpu()
        ),
        "mean_matched_proximity": float(torch.mean(proximity).cpu()),
        "matched_capture_entry_rate": float(
            torch.mean((proximity <= capture_distance).float()).cpu()
        ),
        "proposal_recall": float(torch.mean(proposal.float()).cpu()),
        "candidate_zero_target_recall": float(
            torch.mean(candidate_zero_target.float()).cpu()
        ),
        "target_proposal_recall": float(
            torch.mean(torch.any(target, dim=1).float()).cpu()
        ),
        "useful_candidate_rate": float(torch.mean(useful.float()).cpu()),
        "useful_region_count": float(torch.mean(useful_count).cpu()),
        "candidate_mean_quality": float(torch.mean(quality).cpu()),
        "candidate_decoy_rate": float(torch.mean(decoy.float()).cpu()),
        "candidate_anchor_rate": float(torch.mean(anchor.float()).cpu()),
        "effective_directional_rank": float(torch.mean(effective_rank).cpu()),
        "alpha_mean": float(torch.mean(alphas).cpu()),
        "alpha_sd": float(torch.std(alphas, correction=0).cpu()),
    }
    per_set = [
        {
            "set_index": float(index),
            "best_target_cosine": float(torch.amax(target_cosine[index]).cpu()),
            "candidate_zero_target_cosine": float(target_cosine[index, 0].cpu()),
            "best_useful_cosine": float(torch.amax(cosine_all[index]).cpu()),
            "mean_matched_cosine": float(torch.mean(matched_cosine[index]).cpu()),
            "mean_angular_error_degrees": float(
                torch.mean(matched_angular[index]).cpu()
            ),
            "mean_alpha_absolute_error": float(
                torch.mean(
                    torch.abs(matched_alpha[index] - matched_alpha_target[index])
                ).cpu()
            ),
            "mean_matched_proximity": float(torch.mean(proximity[index]).cpu()),
            "proposal_recall": float(proposal[index].cpu()),
            "target_proposal_recall": float(torch.any(target[index]).float().cpu()),
            "useful_region_count": float(useful_count[index].cpu()),
        }
        for index in range(len(starts))
    ]
    return SupervisedEvaluation(summary=summary, per_set=per_set)
