"""Black-box training for the creative operator.

An antithetic evolution strategy is used because local differentiation at the
obvious peak makes W=0 a trivial attractor.  Crucially, the trainer queries only
landscape quality and never receives island IDs or peak centers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np

from .config import TrainingConfig
from .landscapes import GaussianIslandLandscape
from .operators import CreativeOperator, LinearCritic

Array = np.ndarray


def _sigmoid(x: Array) -> Array:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def operator_objective(
    operator: CreativeOperator,
    landscapes: Sequence[GaussianIslandLandscape],
    starts: Sequence[Array],
    config: TrainingConfig,
) -> tuple[float, dict[str, float]]:
    """Quality-gated, bounded quality-diversity objective (higher is better)."""
    quality_values: list[float] = []
    raw_quality_values: list[float] = []
    novelty_values: list[float] = []
    diversity_values: list[float] = []
    semantic_values: list[float] = []

    for landscape, h in zip(landscapes, starts):
        candidates = operator.apply(h, config.alpha_train)
        refined = landscape.refine(
            candidates,
            steps=config.refinement_steps,
            rate=config.refinement_rate,
        )
        quality = landscape.quality(refined)
        raw_quality = landscape.quality(candidates)
        gate = _sigmoid((quality - config.quality_gate) / config.gate_temperature)

        novelty_distance = np.linalg.norm(refined - h[None, :], axis=-1)
        # Bounded novelty prevents an infinite-distance optimum.
        novelty = gate * np.minimum(novelty_distance / config.diversity_scale, 1.5)

        pair_rewards: list[float] = []
        for i in range(operator.heads):
            for j in range(i + 1, operator.heads):
                pair_gate = float(gate[i] * gate[j])
                distance = np.linalg.norm(refined[i] - refined[j])
                pair_rewards.append(
                    pair_gate * min(distance / config.diversity_scale, 1.5)
                )

        delta = candidates - h[None, :]
        semantic = np.mean(np.sum(delta * delta, axis=-1)) / (config.semantic_scale**2)
        quality_values.append(float(np.mean(quality)))
        raw_quality_values.append(float(np.mean(raw_quality)))
        novelty_values.append(float(np.mean(novelty)))
        diversity_values.append(float(np.mean(pair_rewards)) if pair_rewards else 0.0)
        semantic_values.append(float(semantic))

    matrices = operator.matrices()
    regularization = float(
        np.mean(np.sum(matrices * matrices, axis=(1, 2))) / operator.dimension
    )
    components = {
        "quality": float(np.mean(quality_values)),
        "raw_quality": float(np.mean(raw_quality_values)),
        "gated_novelty": float(np.mean(novelty_values)),
        "gated_diversity": float(np.mean(diversity_values)),
        "semantic_penalty": float(np.mean(semantic_values)),
        "regularization": regularization,
    }
    reward = (
        config.quality_weight * components["quality"]
        + config.raw_quality_weight * components["raw_quality"]
        + config.novelty_weight * components["gated_novelty"]
        + config.diversity_weight * components["gated_diversity"]
        - config.semantic_weight * components["semantic_penalty"]
        - config.regularization_weight * components["regularization"]
    )
    components["reward"] = float(reward)
    return float(reward), components


def _centered_rank_utilities(values: Array) -> Array:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    if values.size > 1:
        ranks /= values.size - 1
    utilities = ranks - 0.5
    utilities /= np.std(utilities) + 1e-12
    return utilities


def train_operator(
    operator: CreativeOperator,
    landscapes: Sequence[GaussianIslandLandscape],
    starts: Sequence[Array],
    config: TrainingConfig,
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, float]]:
    """Train with antithetic parameter perturbations and Adam ES updates."""
    if len(landscapes) != len(starts):
        raise ValueError("landscapes and starts must have equal length")
    rng = np.random.default_rng(seed)
    population = config.population + (config.population % 2)
    half = population // 2
    theta = operator.get_parameters()
    adam_m = np.zeros_like(theta)
    adam_v = np.zeros_like(theta)
    best_theta = theta.copy()
    best_reward, _ = operator_objective(operator, landscapes, starts, config)
    history: list[dict[str, float]] = []

    for iteration in range(iterations):
        progress = iteration / max(iterations - 1, 1)
        sigma = config.sigma_start * np.power(
            config.sigma_end / config.sigma_start, progress
        )
        batch_size = min(config.batch_size, len(landscapes))
        indices = rng.choice(len(landscapes), size=batch_size, replace=False)
        batch_landscapes = [landscapes[int(i)] for i in indices]
        batch_starts = [starts[int(i)] for i in indices]

        base_noise = rng.normal(size=(half, theta.size))
        active_mask = np.ones(theta.size, dtype=np.float64)
        if operator.rank is not None:
            assert operator.a is not None
            split = operator.a.size
            # Initial A-only updates and later alternating blocks avoid the symmetric
            # quadratic term that dominates joint +/- mutations of A and B.
            if iteration < max(2, iterations // 4) or iteration % 2 == 0:
                base_noise[:, :split] *= np.sqrt(operator.dimension / operator.rank)
                base_noise[:, split:] = 0.0
                active_mask[split:] = 0.0
            else:
                base_noise[:, :split] = 0.0
                active_mask[:split] = 0.0
        noise = np.concatenate([base_noise, -base_noise], axis=0)
        rewards = np.empty(population, dtype=np.float64)
        for member in range(population):
            operator.set_parameters(theta + sigma * noise[member])
            rewards[member], _ = operator_objective(
                operator, batch_landscapes, batch_starts, config
            )

        utilities = _centered_rank_utilities(rewards)
        gradient = (utilities[:, None] * noise).mean(axis=0) / sigma
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > config.gradient_clip:
            gradient *= config.gradient_clip / gradient_norm

        adam_m = config.adam_beta1 * adam_m + (1.0 - config.adam_beta1) * gradient
        adam_v = config.adam_beta2 * adam_v + (1.0 - config.adam_beta2) * (
            gradient * gradient
        )
        step_number = iteration + 1
        m_hat = adam_m / (1.0 - config.adam_beta1**step_number)
        v_hat = adam_v / (1.0 - config.adam_beta2**step_number)
        theta = theta + (
            config.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8) * active_mask
        )
        operator.set_parameters(theta)

        # A fixed comparison pool prevents minibatch variation from choosing the checkpoint.
        current_reward, components = operator_objective(
            operator, landscapes, starts, config
        )
        if current_reward > best_reward:
            best_reward = current_reward
            best_theta = theta.copy()
        row = {
            "iteration": float(iteration),
            "sigma": float(sigma),
            "population_reward_mean": float(np.mean(rewards)),
            "population_reward_std": float(np.std(rewards)),
            "gradient_norm": float(gradient_norm),
            **components,
        }
        history.append(row)

    operator.set_parameters(best_theta)
    return history


def fit_critic(
    operator: CreativeOperator,
    landscapes: Sequence[GaussianIslandLandscape],
    starts: Sequence[Array],
    config: TrainingConfig,
) -> LinearCritic:
    """Fit a tiny critic to post-refinement quality; no island IDs are used."""
    all_features: list[Array] = []
    all_targets: list[Array] = []
    for landscape, h in zip(landscapes, starts):
        candidates = operator.apply(h, config.alpha_train)
        refined = landscape.refine(
            candidates,
            steps=config.refinement_steps,
            rate=config.refinement_rate,
        )
        goal_center = np.mean(landscape.centers, axis=0)
        features = LinearCritic.features(h, candidates, goal_center)
        all_features.append(features)
        all_targets.append(landscape.quality(refined))
    critic = LinearCritic()
    critic.fit(np.concatenate(all_features), np.concatenate(all_targets))
    return critic


def training_metadata(config: TrainingConfig) -> dict[str, object]:
    return asdict(config)
