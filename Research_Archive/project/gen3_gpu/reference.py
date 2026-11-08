"""Read-only adapters that expose checkpoints from the original CPU equations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from creativity_toy.config_v2 import TrainingV2Config
from creativity_toy.creative_operator_v2 import BaseCreativeOperatorV2
from creativity_toy.landscapes_v2 import V2Task
from creativity_toy.training_v2 import operator_objective_v2

from .frozen_inputs import FrozenESSchedule


@dataclass
class ReferenceTrainingResult:
    parameters: np.ndarray
    checkpoints: dict[int, np.ndarray]
    accepted_elites: list[bool]
    elapsed_seconds: float


def _rank_utilities(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    ranks /= max(len(values) - 1, 1)
    utilities = ranks - 0.5
    return utilities / (np.std(utilities) + 1e-12)


def train_reference_with_schedule(
    operator: BaseCreativeOperatorV2,
    tasks: Sequence[V2Task],
    starts: Sequence[np.ndarray],
    config: TrainingV2Config,
    *,
    schedule: FrozenESSchedule,
    checkpoint_steps: tuple[int, ...] = (0, 1, 10),
) -> ReferenceTrainingResult:
    """Original scalar NumPy training with externally frozen stochastic inputs."""
    iterations = schedule.iterations
    theta = operator.get_parameters()
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    best_theta = theta.copy()
    best_reward, _ = operator_objective_v2(operator, tasks, starts, config)
    current_reward = best_reward
    checkpoints = {0: theta.copy()}
    accepted_elites: list[bool] = []
    start_time = perf_counter()
    for iteration in range(iterations):
        progress = iteration / max(iterations - 1, 1)
        sigma = config.sigma_start * np.power(
            config.sigma_end / config.sigma_start, progress
        )
        indices = schedule.batch_indices[iteration]
        batch_tasks = [tasks[int(index)] for index in indices]
        batch_starts = [starts[int(index)] for index in indices]
        active_mask, _ = operator.exploration_mask(iteration, iterations)
        noise = schedule.noise[iteration]
        rewards = np.empty(len(noise), dtype=np.float64)
        for member in range(len(noise)):
            operator.set_parameters(theta + sigma * noise[member])
            rewards[member], _ = operator_objective_v2(
                operator, batch_tasks, batch_starts, config
            )
        utilities = _rank_utilities(rewards)
        gradient = np.mean(utilities[:, None] * noise, axis=0) / sigma
        norm = np.linalg.norm(gradient)
        if norm > config.gradient_clip:
            gradient *= config.gradient_clip / norm
        elite_index = int(np.argmax(rewards))
        elite_theta = theta + sigma * noise[elite_index]
        operator.set_parameters(elite_theta)
        elite_reward, _ = operator_objective_v2(operator, tasks, starts, config)
        accepted = elite_reward > current_reward
        accepted_elites.append(bool(accepted))
        if accepted:
            theta = elite_theta
            current_reward = elite_reward
            adam_m *= 0.5
            adam_v *= 0.5
        else:
            adam_m = 0.9 * adam_m + 0.1 * gradient
            adam_v = 0.999 * adam_v + 0.001 * gradient * gradient
            step = iteration + 1
            update = (
                config.learning_rate
                * (adam_m / (1.0 - 0.9**step))
                / (np.sqrt(adam_v / (1.0 - 0.999**step)) + 1e-8)
            )
            theta += update * active_mask
            operator.set_parameters(theta)
            current_reward, _ = operator_objective_v2(operator, tasks, starts, config)
        if current_reward > best_reward:
            best_reward = current_reward
            best_theta = theta.copy()
        completed = iteration + 1
        if completed in checkpoint_steps or completed == iterations:
            checkpoints[completed] = theta.copy()
    operator.set_parameters(best_theta)
    checkpoints[-1] = best_theta.copy()
    return ReferenceTrainingResult(
        parameters=best_theta,
        checkpoints=checkpoints,
        accepted_elites=accepted_elites,
        elapsed_seconds=perf_counter() - start_time,
    )
