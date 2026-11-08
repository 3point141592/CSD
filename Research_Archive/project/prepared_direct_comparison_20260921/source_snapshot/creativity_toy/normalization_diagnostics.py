"""Candidate, selector, norm, and equivariance diagnostics for Gen2."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from .config_v2 import TrainingV2Config
from .creative_operator_v2 import ConditionalMLPOperatorV2
from .critics import GoalConditionalRidgeCritic
from .landscapes_v2 import V2Task
from .metrics_v2 import summarize_v2_records

Array = np.ndarray


def _average_ranks(values: Array) -> Array:
    """Return zero-based average ranks, including exact-tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_rank_correlation(x: Array, y: Array) -> float:
    """Dependency-free Spearman correlation for one candidate set."""
    x_ranks = _average_ranks(np.asarray(x))
    y_ranks = _average_ranks(np.asarray(y))
    if np.std(x_ranks) < 1e-12 or np.std(y_ranks) < 1e-12:
        return 0.0
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def _relative_vector_error(actual: Array, expected: Array) -> float:
    numerator = np.linalg.norm(actual - expected, axis=-1)
    scale = 0.5 * (np.linalg.norm(actual, axis=-1) + np.linalg.norm(expected, axis=-1))
    return float(np.mean(numerator / np.maximum(scale, 1e-8)))


def _orthogonal_matrix(dimension: int, rng: np.random.Generator) -> Array:
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[None, :]


def equivariance_diagnostics(
    operator: ConditionalMLPOperatorV2,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[Array]],
    *,
    seed: int,
    trials: int,
    translation_magnitude: float,
) -> dict[str, float]:
    """Measure, but do not train, rotation equivariance and translation invariance.

    The V2 goal is a spatial direction. Under a coordinate rotation R the expected
    mapping is f(Rh, Rg, z) = R f(h, g, z). Under a translation t the goal does
    not change, and the expected mapping is f(h+t, g, z) = f(h, g, z).
    Errors are head-wise L2 errors divided by the mean norm of the compared pair.
    """
    rng = np.random.default_rng(seed)
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for task, task_starts in zip(tasks, starts):
        for h in task_starts:
            base = operator.directions(h, task.goal)
            for _ in range(trials):
                rotation = _orthogonal_matrix(task.dimension, rng)
                rotated = operator.directions(rotation @ h, rotation @ task.goal)
                expected_rotated = base @ rotation.T
                rotation_errors.append(
                    _relative_vector_error(rotated, expected_rotated)
                )

                translation = rng.normal(size=task.dimension)
                translation /= max(np.linalg.norm(translation), 1e-12)
                translation *= translation_magnitude
                translated = operator.directions(h + translation, task.goal)
                translation_errors.append(_relative_vector_error(translated, base))
    return {
        "rotation_error": float(np.mean(rotation_errors)),
        "translation_error": float(np.mean(translation_errors)),
    }


def evaluate_candidate_sets(
    *,
    variant: str,
    operator: ConditionalMLPOperatorV2,
    critic: GoalConditionalRidgeCritic,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[Array]],
    config: TrainingV2Config,
    seed: int,
    alpha: float,
    useful_threshold: float,
    high_quality_threshold: float,
) -> tuple[
    dict[str, float],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Evaluate one generated batch once, then reuse it for all selectors."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    selector_observations: dict[str, list[dict[str, float]]] = {
        "learned_critic": [],
        "oracle": [],
        "random": [],
    }
    start_time = time.perf_counter()

    for task_id, (task, task_starts) in enumerate(zip(tasks, starts)):
        for trial, h in enumerate(task_starts):
            direction_stats = operator.direction_diagnostics(h, task.goal, alpha)
            candidates = operator.apply(h, task.goal, alpha)
            candidate_alphas = np.asarray(
                direction_stats.get(
                    "alpha_values", direction_stats["perturbation_norm"]
                ),
                dtype=np.float64,
            )
            refined = task.refine(
                candidates,
                steps=config.refinement_steps,
                rate=config.refinement_rate,
            )
            raw_qualities = task.quality(candidates)
            qualities = task.quality(refined)
            regions = task.region(refined)
            features = GoalConditionalRidgeCritic.features(h, candidates, task.goal)
            critic_scores = critic.predict(features)
            critic_index = int(np.argmax(critic_scores))
            oracle_index = int(np.argmax(qualities))
            random_index = int(rng.integers(0, operator.candidates))
            rank_correlation = spearman_rank_correlation(critic_scores, qualities)

            set_rows.append(
                {
                    "variant": variant,
                    "task_id": int(task_id),
                    "geometry_id": int(task.geometry_id),
                    "goal_index": int(task.goal_index),
                    "trial": int(trial),
                    "alpha": float(alpha),
                    "critic_index": critic_index,
                    "oracle_index": oracle_index,
                    "random_index": random_index,
                    "oracle_quality": float(qualities[oracle_index]),
                    "selected_quality": float(qualities[critic_index]),
                    "critic_regret": float(
                        qualities[oracle_index] - qualities[critic_index]
                    ),
                    "critic_top1": float(critic_index == oracle_index),
                    "spearman_rank_correlation": rank_correlation,
                    "alpha_mean": float(np.mean(candidate_alphas)),
                    "alpha_sd": float(np.std(candidate_alphas, ddof=0)),
                }
            )

            selector_indices = {
                "learned_critic": critic_index,
                "oracle": oracle_index,
                "random": random_index,
            }
            for selector, index in selector_indices.items():
                region = int(regions[index])
                selector_observations[selector].append(
                    {
                        "selected_quality": float(qualities[index]),
                        "target_hit_rate": float(
                            region == task.target_region
                            and qualities[index] >= useful_threshold
                        ),
                        "useful_exploration_rate": float(
                            1 <= region <= task.n_useful
                            and qualities[index] >= useful_threshold
                        ),
                        "decoy_rate": float(region > task.n_useful),
                    }
                )

            for candidate_index in range(operator.candidates):
                region = int(regions[candidate_index])
                records.append(
                    {
                        "method": variant,
                        "variant": variant,
                        "task_id": int(task_id),
                        "geometry_id": int(task.geometry_id),
                        "goal_index": int(task.goal_index),
                        "target_region": int(task.target_region),
                        "trial": int(trial),
                        "candidate_id": int(candidate_index),
                        "n_useful": int(task.n_useful),
                        "alpha": float(alpha),
                        "quality": float(qualities[candidate_index]),
                        "true_utility": float(qualities[candidate_index]),
                        "raw_true_utility": float(raw_qualities[candidate_index]),
                        "critic_score": float(critic_scores[candidate_index]),
                        "region": region,
                        "is_useful": bool(1 <= region <= task.n_useful),
                        "is_decoy": bool(region > task.n_useful),
                        "is_target": bool(region == task.target_region),
                        "selected": bool(candidate_index == critic_index),
                        "critic_selected": bool(candidate_index == critic_index),
                        "oracle_best": bool(candidate_index == oracle_index),
                        "oracle_selected": bool(candidate_index == oracle_index),
                        "random_selected": bool(candidate_index == random_index),
                        "raw_norm": float(direction_stats["raw_norm"][candidate_index]),
                        "normalized_norm": float(
                            direction_stats["normalized_norm"][candidate_index]
                        ),
                        "applied_direction_norm": float(
                            direction_stats["applied_direction_norm"][candidate_index]
                        ),
                        "perturbation_norm": float(
                            direction_stats["perturbation_norm"][candidate_index]
                        ),
                        "candidate_alpha": float(candidate_alphas[candidate_index]),
                        "perturbation": float(
                            np.linalg.norm(candidates[candidate_index] - h)
                        ),
                        "displacement": float(
                            np.linalg.norm(refined[candidate_index] - h)
                        ),
                        "objective_evaluations": int(config.refinement_steps + 1),
                        "optimization_steps": int(config.refinement_steps),
                        "start": np.asarray(h).copy(),
                        "goal": task.goal.copy(),
                        "candidate": candidates[candidate_index].copy(),
                        "final": np.asarray(refined[candidate_index]).copy(),
                    }
                )

    elapsed = time.perf_counter() - start_time
    for record in records:
        record["method_wall_clock_seconds"] = float(elapsed)

    portfolio = summarize_v2_records(
        records,
        useful_threshold=useful_threshold,
        high_quality_threshold=high_quality_threshold,
        diversity_scale=config.diversity_scale,
    )
    portfolio.update(
        {
            "critic_regret": float(
                np.mean([float(row["critic_regret"]) for row in set_rows])
            ),
            "critic_top1_accuracy": float(
                np.mean([float(row["critic_top1"]) for row in set_rows])
            ),
            "spearman_rank_correlation": float(
                np.mean([float(row["spearman_rank_correlation"]) for row in set_rows])
            ),
            "raw_norm": float(np.mean([float(row["raw_norm"]) for row in records])),
            "normalized_norm": float(
                np.mean([float(row["normalized_norm"]) for row in records])
            ),
            "actual_perturbation_magnitude": float(
                np.mean([float(row["perturbation_norm"]) for row in records])
            ),
            "mean_alpha": float(
                np.mean([float(row["candidate_alpha"]) for row in records])
            ),
            "alpha_sd": float(
                np.std([float(row["candidate_alpha"]) for row in records], ddof=0)
            ),
            "alpha_p05": float(
                np.quantile([float(row["candidate_alpha"]) for row in records], 0.05)
            ),
            "alpha_p25": float(
                np.quantile([float(row["candidate_alpha"]) for row in records], 0.25)
            ),
            "alpha_p50": float(
                np.quantile([float(row["candidate_alpha"]) for row in records], 0.50)
            ),
            "alpha_p75": float(
                np.quantile([float(row["candidate_alpha"]) for row in records], 0.75)
            ),
            "alpha_p95": float(
                np.quantile([float(row["candidate_alpha"]) for row in records], 0.95)
            ),
            "within_set_alpha_sd": float(
                np.mean([float(row["alpha_sd"]) for row in set_rows])
            ),
        }
    )

    selector_rows: list[dict[str, object]] = []
    for selector, observations in selector_observations.items():
        selector_rows.append(
            {
                "variant": variant,
                "selector": selector,
                "alpha": float(alpha),
                **{
                    metric: float(np.mean([row[metric] for row in observations]))
                    for metric in [
                        "selected_quality",
                        "target_hit_rate",
                        "useful_exploration_rate",
                        "decoy_rate",
                    ]
                },
            }
        )
    return portfolio, records, set_rows, selector_rows
