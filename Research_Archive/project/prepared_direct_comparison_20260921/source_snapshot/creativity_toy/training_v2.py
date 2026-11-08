"""Label-free ES training and critic fitting for Generation 2."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config_v2 import TrainingV2Config
from .creative_operator_v2 import BaseCreativeOperatorV2
from .critics import GoalConditionalRidgeCritic
from .landscapes_v2 import V2Task

Array = np.ndarray


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _softmax(x: Array) -> Array:
    shifted = x - np.max(x)
    values = np.exp(shifted)
    return values / np.sum(values)


def operator_objective_v2(
    operator: BaseCreativeOperatorV2,
    tasks: Sequence[V2Task],
    starts: Sequence[Array],
    config: TrainingV2Config,
    *,
    alpha: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Quality-weighted diversity objective; no region labels are queried."""
    active_alpha = config.alpha if alpha is None else alpha
    rows: dict[str, list[float]] = {
        "quality": [],
        "best_quality": [],
        "raw_quality": [],
        "gated_novelty": [],
        "weighted_diversity": [],
        "step_penalty": [],
        "low_quality_distance": [],
        "alpha_regularization": [],
    }
    for task, h in zip(tasks, starts):
        candidates = operator.apply(h, task.goal, active_alpha)
        refined = task.refine(
            candidates,
            steps=config.refinement_steps,
            rate=config.refinement_rate,
        )
        quality = task.quality(refined)
        raw_quality = task.quality(candidates)
        gate = _sigmoid((quality - config.quality_gate) / config.gate_temperature)
        quality_weights = _softmax(config.diversity_beta * quality) * gate
        quality_weights /= np.sum(quality_weights) + 1e-12

        pair_reward = 0.0
        pair_mass = 0.0
        for i in range(operator.candidates):
            for j in range(i + 1, operator.candidates):
                mass = float(quality_weights[i] * quality_weights[j])
                distance = float(np.linalg.norm(refined[i] - refined[j]))
                pair_reward += mass * min(distance / config.diversity_scale, 1.5)
                pair_mass += mass
        diversity = pair_reward / max(pair_mass, 1e-12)

        final_distance = np.linalg.norm(refined - h[None, :], axis=-1)
        raw_delta = candidates - h[None, :]
        raw_distance = np.linalg.norm(raw_delta, axis=-1)
        rows["quality"].append(float(np.mean(quality)))
        rows["best_quality"].append(float(np.max(quality)))
        rows["raw_quality"].append(float(np.mean(raw_quality)))
        rows["gated_novelty"].append(
            float(
                np.mean(gate * np.minimum(final_distance / config.diversity_scale, 1.5))
            )
        )
        rows["weighted_diversity"].append(float(diversity))
        rows["step_penalty"].append(
            float(np.mean(raw_distance * raw_distance) / config.semantic_scale**2)
        )
        rows["low_quality_distance"].append(
            float(
                np.mean(
                    (1.0 - gate) * np.minimum(raw_distance / config.semantic_scale, 2.0)
                )
            )
        )
        alpha_regularizer = getattr(operator, "alpha_regularization", None)
        rows["alpha_regularization"].append(
            float(alpha_regularizer(h, task.goal, config.alpha_reference))
            if callable(alpha_regularizer)
            else 0.0
        )

    components = {key: float(np.mean(values)) for key, values in rows.items()}
    components["regularization"] = operator.regularization()
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
    components["reward"] = float(reward)
    return float(reward), components


def _rank_utilities(values: Array) -> Array:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    ranks /= max(len(values) - 1, 1)
    utilities = ranks - 0.5
    return utilities / (np.std(utilities) + 1e-12)


def train_operator_v2(
    operator: BaseCreativeOperatorV2,
    tasks: Sequence[V2Task],
    starts: Sequence[Array],
    config: TrainingV2Config,
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, float]]:
    if len(tasks) != len(starts):
        raise ValueError("tasks and starts must have equal length")
    rng = np.random.default_rng(seed)
    population = config.population + config.population % 2
    half = population // 2
    theta = operator.get_parameters()
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    best_theta = theta.copy()
    best_reward, _ = operator_objective_v2(operator, tasks, starts, config)
    current_reward = best_reward
    history: list[dict[str, float]] = []

    for iteration in range(iterations):
        progress = iteration / max(iterations - 1, 1)
        sigma = config.sigma_start * np.power(
            config.sigma_end / config.sigma_start, progress
        )
        indices = rng.choice(
            len(tasks), size=min(config.batch_size, len(tasks)), replace=False
        )
        batch_tasks = [tasks[int(index)] for index in indices]
        batch_starts = [starts[int(index)] for index in indices]

        active_mask, noise_scale = operator.exploration_mask(iteration, iterations)
        noise_half = rng.normal(size=(half, theta.size)) * noise_scale[None, :]
        noise_half *= active_mask[None, :]
        noise = np.concatenate([noise_half, -noise_half], axis=0)
        rewards = np.empty(population, dtype=np.float64)
        for member in range(population):
            operator.set_parameters(theta + sigma * noise[member])
            rewards[member], _ = operator_objective_v2(
                operator, batch_tasks, batch_starts, config
            )

        utilities = _rank_utilities(rewards)
        gradient = np.mean(utilities[:, None] * noise, axis=0) / sigma
        norm = np.linalg.norm(gradient)
        if norm > config.gradient_clip:
            gradient *= config.gradient_clip / norm
        # Direct elite acceptance preserves rare basin-crossing improvements that
        # a population-averaged update can lose. Full-pool evaluation ensures that
        # acceptance reflects the label-free training objective, not one minibatch.
        elite_index = int(np.argmax(rewards))
        elite_theta = theta + sigma * noise[elite_index]
        operator.set_parameters(elite_theta)
        elite_reward, elite_components = operator_objective_v2(
            operator, tasks, starts, config
        )
        accepted_elite = elite_reward > current_reward
        if accepted_elite:
            theta = elite_theta
            current_reward = elite_reward
            components = elite_components
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
            current_reward, components = operator_objective_v2(
                operator, tasks, starts, config
            )
        if current_reward > best_reward:
            best_reward = current_reward
            best_theta = theta.copy()
        history.append(
            {
                "iteration": float(iteration),
                "sigma": float(sigma),
                "population_reward_mean": float(np.mean(rewards)),
                "population_reward_std": float(np.std(rewards)),
                "gradient_norm": float(norm),
                "accepted_elite": float(accepted_elite),
                **components,
            }
        )
    operator.set_parameters(best_theta)
    return history


def fit_critic_v2(
    operator: BaseCreativeOperatorV2,
    tasks: Sequence[V2Task],
    starts: Sequence[Array],
    config: TrainingV2Config,
    *,
    seed: int,
) -> GoalConditionalRidgeCritic:
    """Fit on objective labels from operator and broad exploratory candidates."""
    rng = np.random.default_rng(seed)
    feature_rows: list[Array] = []
    target_rows: list[Array] = []
    for task, h in zip(tasks, starts):
        learned = operator.apply(h, task.goal, config.alpha)
        broad_count = max(2 * operator.candidates, 8)
        directions = rng.normal(size=(broad_count, task.dimension))
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
        )
        radii = rng.uniform(0.0, config.pure_divergence_radius, size=broad_count)
        broad = h[None, :] + radii[:, None] * directions
        candidates = np.concatenate([learned, broad], axis=0)
        refined = task.refine(
            candidates,
            steps=config.refinement_steps,
            rate=config.refinement_rate,
        )
        feature_rows.append(
            GoalConditionalRidgeCritic.features(h, candidates, task.goal)
        )
        target_rows.append(task.quality(refined))
    critic = GoalConditionalRidgeCritic(
        ridge=config.critic_ridge,
        temperature=config.critic_temperature,
    )
    critic.fit(np.concatenate(feature_rows), np.concatenate(target_rows))
    return critic
