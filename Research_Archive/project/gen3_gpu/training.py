"""Population/task/candidate-batched ES training with unchanged mathematics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from creativity_toy.config_v2 import TrainingV2Config

from .frozen_inputs import FrozenESSchedule
from .operator import BatchedCreativeOperator, OperatorOutput
from .runtime import synchronize
from .task_batch import TorchTaskBatch


@dataclass
class ObjectiveResult:
    reward: torch.Tensor
    components: dict[str, torch.Tensor]
    operator_output: OperatorOutput | None = None
    refined: torch.Tensor | None = None
    quality: torch.Tensor | None = None
    raw_quality: torch.Tensor | None = None


@dataclass
class TrainingResult:
    parameters: torch.Tensor
    history: list[dict[str, float]]
    checkpoints: dict[int, torch.Tensor]
    elapsed_seconds: float
    population_objective_calls: int
    full_objective_calls: int


def operator_objective_batched(
    operator: BatchedCreativeOperator,
    theta: torch.Tensor,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    config: TrainingV2Config,
    *,
    capture: bool = False,
) -> ObjectiveResult:
    """Exact V2 reward, vectorized over ES population, tasks, and candidates."""
    output = operator.forward(theta, starts, tasks.goals)
    refined = tasks.refine(
        output.candidates,
        steps=config.refinement_steps,
        rate=config.refinement_rate,
    )
    quality = tasks.quality(refined)
    raw_quality = tasks.quality(output.candidates)
    gate_argument = torch.clamp(
        (quality - config.quality_gate) / config.gate_temperature,
        min=-50.0,
        max=50.0,
    )
    gate = torch.sigmoid(gate_argument)
    quality_weights = torch.softmax(config.diversity_beta * quality, dim=-1) * gate
    quality_weights = quality_weights / (
        torch.sum(quality_weights, dim=-1, keepdim=True) + 1e-12
    )

    distances = torch.cdist(refined, refined, p=2)
    pair_mass_matrix = quality_weights.unsqueeze(-1) * quality_weights.unsqueeze(-2)
    upper = torch.triu(
        torch.ones(
            (operator.candidates, operator.candidates),
            dtype=torch.bool,
            device=theta.device,
        ),
        diagonal=1,
    )
    pair_mass = torch.sum(
        torch.where(upper, pair_mass_matrix, torch.zeros_like(pair_mass_matrix)),
        dim=(-2, -1),
    )
    pair_reward = torch.sum(
        torch.where(
            upper,
            pair_mass_matrix * torch.clamp(distances / config.diversity_scale, max=1.5),
            torch.zeros_like(pair_mass_matrix),
        ),
        dim=(-2, -1),
    )
    diversity = pair_reward / torch.clamp(pair_mass, min=1e-12)

    final_distance = torch.linalg.vector_norm(
        refined - starts[None, :, None, :], dim=-1
    )
    raw_delta = output.candidates - starts[None, :, None, :]
    raw_distance = torch.linalg.vector_norm(raw_delta, dim=-1)
    population = output.candidates.shape[0]
    zero = torch.zeros(population, dtype=theta.dtype, device=theta.device)
    components = {
        "quality": torch.mean(quality, dim=(-2, -1)),
        "best_quality": torch.mean(torch.amax(quality, dim=-1), dim=-1),
        "raw_quality": torch.mean(raw_quality, dim=(-2, -1)),
        "gated_novelty": torch.mean(
            gate * torch.clamp(final_distance / config.diversity_scale, max=1.5),
            dim=(-2, -1),
        ),
        "weighted_diversity": torch.mean(diversity, dim=-1),
        "step_penalty": torch.mean(raw_distance * raw_distance, dim=(-2, -1))
        / config.semantic_scale**2,
        "low_quality_distance": torch.mean(
            (1.0 - gate) * torch.clamp(raw_distance / config.semantic_scale, max=2.0),
            dim=(-2, -1),
        ),
        "regularization": operator.direction_regularization(theta),
        "alpha_regularization": (
            torch.mean((output.alphas - config.alpha_reference) ** 2, dim=(-2, -1))
            if operator.additional_alpha_parameters
            else zero
        ),
    }
    reward = (
        config.quality_weight * components["quality"]
        + config.best_quality_weight * components["best_quality"]
        + config.raw_quality_weight * components["raw_quality"]
        + config.novelty_weight * components["gated_novelty"]
        + config.diversity_weight * components["weighted_diversity"]
        - config.step_weight * components["step_penalty"]
        - config.low_quality_distance_weight * components["low_quality_distance"]
        - config.regularization_weight * components["regularization"]
        - config.alpha_regularization_weight * components["alpha_regularization"]
    )
    components["reward"] = reward
    return ObjectiveResult(
        reward=reward,
        components=components,
        operator_output=output if capture else None,
        refined=refined if capture else None,
        quality=quality if capture else None,
        raw_quality=raw_quality if capture else None,
    )


def rank_utilities(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(len(values), dtype=values.dtype, device=values.device)
    ranks = ranks / max(len(values) - 1, 1)
    utilities = ranks - 0.5
    return utilities / (torch.std(utilities, correction=0) + 1e-12)


def _history_row(
    *,
    iteration: int,
    sigma: float,
    rewards: torch.Tensor,
    gradient_norm: torch.Tensor,
    accepted_elite: bool,
    components: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    row = {
        "iteration": float(iteration),
        "sigma": float(sigma),
        "population_reward_mean": float(torch.mean(rewards).detach().cpu()),
        "population_reward_std": float(torch.std(rewards, correction=0).detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "accepted_elite": float(accepted_elite),
    }
    row.update(
        {
            key: float(value.reshape(-1)[0].detach().cpu())
            for key, value in components.items()
        }
    )
    return row


def train_operator_batched(
    operator: BatchedCreativeOperator,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    config: TrainingV2Config,
    *,
    schedule: FrozenESSchedule,
    checkpoint_steps: tuple[int, ...] = (0, 1, 10),
) -> TrainingResult:
    """Preserve the ES/Adam update and evaluation budget; batch only independents."""
    if tasks.batch_size != len(starts):
        raise ValueError("tasks and starts must have equal length")
    iterations = schedule.iterations
    population = config.population + config.population % 2
    if schedule.noise.shape != (iterations, population, operator.parameter_count):
        raise ValueError(
            "frozen noise shape does not match this operator/configuration"
        )
    index_tensor, noise_tensor = schedule.tensors(
        device=operator.device, dtype=operator.dtype
    )
    theta = operator.initial_parameters.clone()
    adam_m = torch.zeros_like(theta)
    adam_v = torch.zeros_like(theta)
    initial = operator_objective_batched(operator, theta, tasks, starts, config)
    best_reward = initial.reward[0].clone()
    current_reward = best_reward.clone()
    best_theta = theta.clone()
    checkpoints: dict[int, torch.Tensor] = {0: theta.detach().clone()}
    history: list[dict[str, float]] = []
    full_objective_calls = 1
    population_objective_calls = 0

    synchronize(operator.device)
    start_time = perf_counter()
    for iteration in range(iterations):
        progress = iteration / max(iterations - 1, 1)
        sigma = config.sigma_start * np.power(
            config.sigma_end / config.sigma_start, progress
        )
        indices = index_tensor[iteration]
        batch_tasks = tasks.index_select(indices)
        batch_starts = starts.index_select(0, indices)
        noise = noise_tensor[iteration]
        population_theta = theta.unsqueeze(0) + sigma * noise
        population_result = operator_objective_batched(
            operator, population_theta, batch_tasks, batch_starts, config
        )
        rewards = population_result.reward
        population_objective_calls += population

        utilities = rank_utilities(rewards)
        gradient = torch.mean(utilities.unsqueeze(-1) * noise, dim=0) / sigma
        gradient_norm = torch.linalg.vector_norm(gradient)
        gradient = gradient * torch.minimum(
            torch.ones_like(gradient_norm),
            torch.as_tensor(
                config.gradient_clip,
                dtype=gradient.dtype,
                device=gradient.device,
            )
            / torch.clamp(gradient_norm, min=1e-12),
        )

        elite_index = torch.argmax(rewards)
        elite_theta = theta + sigma * noise[elite_index]
        elite_result = operator_objective_batched(
            operator, elite_theta, tasks, starts, config
        )
        full_objective_calls += 1
        accepted_elite = bool((elite_result.reward[0] > current_reward).item())
        if accepted_elite:
            theta = elite_theta
            current_reward = elite_result.reward[0]
            components = elite_result.components
            adam_m = adam_m * 0.5
            adam_v = adam_v * 0.5
        else:
            adam_m = 0.9 * adam_m + 0.1 * gradient
            adam_v = 0.999 * adam_v + 0.001 * gradient * gradient
            step = iteration + 1
            update = (
                config.learning_rate
                * (adam_m / (1.0 - 0.9**step))
                / (torch.sqrt(adam_v / (1.0 - 0.999**step)) + 1e-8)
            )
            active, _ = operator.exploration_mask(iteration, iterations)
            active_tensor = torch.as_tensor(
                active, dtype=theta.dtype, device=theta.device
            )
            theta = theta + update * active_tensor
            current_result = operator_objective_batched(
                operator, theta, tasks, starts, config
            )
            full_objective_calls += 1
            current_reward = current_result.reward[0]
            components = current_result.components

        improved = current_reward > best_reward
        best_theta = torch.where(improved, theta, best_theta)
        best_reward = torch.maximum(best_reward, current_reward)
        completed_step = iteration + 1
        if completed_step in checkpoint_steps or completed_step == iterations:
            checkpoints[completed_step] = theta.detach().clone()
        history.append(
            _history_row(
                iteration=iteration,
                sigma=float(sigma),
                rewards=rewards,
                gradient_norm=gradient_norm,
                accepted_elite=accepted_elite,
                components=components,
            )
        )
    synchronize(operator.device)
    elapsed = perf_counter() - start_time
    return TrainingResult(
        parameters=best_theta.detach().clone(),
        history=history,
        checkpoints=checkpoints,
        elapsed_seconds=elapsed,
        population_objective_calls=population_objective_calls,
        full_objective_calls=full_objective_calls,
    )
