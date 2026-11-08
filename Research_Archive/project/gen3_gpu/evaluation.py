"""Batched inference/classification with exact reference reporting semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from creativity_toy.config_v2 import TrainingV2Config
from creativity_toy.landscapes_v2 import V2Task
from creativity_toy.metrics_v2 import summarize_v2_records
from creativity_toy.normalization_diagnostics import spearman_rank_correlation

from .critic import BatchedRidgeCritic, critic_features
from .operator import BatchedCreativeOperator
from .runtime import synchronize
from .task_batch import TorchTaskBatch


@dataclass
class EvaluationResult:
    summary: dict[str, float]
    records: list[dict[str, object]]
    candidate_sets: list[dict[str, object]]
    selectors: list[dict[str, object]]
    elapsed_seconds: float
    tensors: dict[str, torch.Tensor]


def _flatten_starts(
    starts: Sequence[Sequence[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat: list[np.ndarray] = []
    task_ids: list[int] = []
    trials: list[int] = []
    for task_id, task_starts in enumerate(starts):
        for trial, h in enumerate(task_starts):
            flat.append(np.asarray(h, dtype=np.float64))
            task_ids.append(task_id)
            trials.append(trial)
    return np.stack(flat), np.asarray(task_ids), np.asarray(trials)


def _random_indices(seed: int, sets: int, candidates: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray([int(rng.integers(0, candidates)) for _ in range(sets)])


def evaluate_batched(
    *,
    variant: str,
    operator: BatchedCreativeOperator,
    theta: torch.Tensor,
    critic: BatchedRidgeCritic,
    tasks: Sequence[V2Task],
    packed_tasks: TorchTaskBatch,
    starts: Sequence[Sequence[np.ndarray]],
    config: TrainingV2Config,
    seed: int,
    useful_threshold: float,
    high_quality_threshold: float,
) -> EvaluationResult:
    start_array, task_ids, trials = _flatten_starts(starts)
    device, dtype = operator.device, operator.dtype
    h = torch.as_tensor(np.ascontiguousarray(start_array), dtype=dtype, device=device)
    task_index = torch.as_tensor(task_ids, dtype=torch.long, device=device)
    batch_tasks = packed_tasks.index_select(task_index)

    synchronize(device)
    start_time = perf_counter()
    output = operator.forward(theta, h, batch_tasks.goals)
    candidates = output.candidates.squeeze(0)
    raw_directions = output.raw_directions.squeeze(0)
    directions = output.directions.squeeze(0)
    alphas = output.alphas.squeeze(0)
    refined = batch_tasks.refine(
        candidates,
        steps=config.refinement_steps,
        rate=config.refinement_rate,
    )
    raw_quality = batch_tasks.quality(candidates)
    quality = batch_tasks.quality(refined)
    regions = batch_tasks.regions(refined)
    features = critic_features(h, candidates, batch_tasks.goals)
    critic_scores = critic.predict(features)
    critic_indices = torch.argmax(critic_scores, dim=-1)
    oracle_indices = torch.argmax(quality, dim=-1)
    random_indices = torch.as_tensor(
        _random_indices(seed, len(h), operator.candidates),
        dtype=torch.long,
        device=device,
    )
    synchronize(device)
    elapsed = perf_counter() - start_time

    # Delay host reporting until scoring finishes to keep transfers outside timing.
    # Reusing reference metric definitions keeps CPU/GPU reports comparable.
    payload = {
        "h": h,
        "candidates": candidates,
        "raw_directions": raw_directions,
        "directions": directions,
        "alphas": alphas,
        "refined": refined,
        "raw_quality": raw_quality,
        "quality": quality,
        "regions": regions,
        "critic_scores": critic_scores,
        "critic_indices": critic_indices,
        "oracle_indices": oracle_indices,
        "random_indices": random_indices,
    }
    numpy_payload = {
        key: value.detach().cpu().numpy() for key, value in payload.items()
    }
    records: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    selector_observations: dict[str, list[dict[str, float]]] = {
        "learned_critic": [],
        "oracle": [],
        "random": [],
    }
    raw_norm = np.linalg.norm(numpy_payload["raw_directions"], axis=-1)
    normalized = (
        torch.nn.functional.normalize(raw_directions, p=2, dim=-1, eps=1e-8)
        .detach()
        .cpu()
        .numpy()
    )
    normalized_norm = np.linalg.norm(normalized, axis=-1)
    applied_norm = np.linalg.norm(numpy_payload["directions"], axis=-1)
    perturbation_norm = np.linalg.norm(
        numpy_payload["candidates"] - numpy_payload["h"][:, None, :], axis=-1
    )

    diagnostic_alphas = (
        perturbation_norm if variant == "gen2_unnormalized" else numpy_payload["alphas"]
    )
    reported_alpha = 1.0 if variant == "gen2_unnormalized" else 3.0
    for set_index in range(len(start_array)):
        task_id = int(task_ids[set_index])
        task = tasks[task_id]
        critic_index = int(numpy_payload["critic_indices"][set_index])
        oracle_index = int(numpy_payload["oracle_indices"][set_index])
        random_index = int(numpy_payload["random_indices"][set_index])
        qualities = numpy_payload["quality"][set_index]
        scores = numpy_payload["critic_scores"][set_index]
        candidate_alphas = diagnostic_alphas[set_index]
        set_rows.append(
            {
                "variant": variant,
                "task_id": task_id,
                "geometry_id": int(task.geometry_id),
                "goal_index": int(task.goal_index),
                "trial": int(trials[set_index]),
                "alpha": reported_alpha,
                "critic_index": critic_index,
                "oracle_index": oracle_index,
                "random_index": random_index,
                "oracle_quality": float(qualities[oracle_index]),
                "selected_quality": float(qualities[critic_index]),
                "critic_regret": float(
                    qualities[oracle_index] - qualities[critic_index]
                ),
                "critic_top1": float(critic_index == oracle_index),
                "spearman_rank_correlation": spearman_rank_correlation(
                    scores, qualities
                ),
                "alpha_mean": float(np.mean(candidate_alphas)),
                "alpha_sd": float(np.std(candidate_alphas, ddof=0)),
            }
        )
        selector_indices = {
            "learned_critic": critic_index,
            "oracle": oracle_index,
            "random": random_index,
        }
        for selector, candidate_index in selector_indices.items():
            region = int(numpy_payload["regions"][set_index, candidate_index])
            q = float(qualities[candidate_index])
            selector_observations[selector].append(
                {
                    "selected_quality": q,
                    "target_hit_rate": float(
                        region == task.target_region and q >= useful_threshold
                    ),
                    "useful_exploration_rate": float(
                        1 <= region <= task.n_useful and q >= useful_threshold
                    ),
                    "decoy_rate": float(region > task.n_useful),
                }
            )
        for candidate_id in range(operator.candidates):
            region = int(numpy_payload["regions"][set_index, candidate_id])
            q = float(qualities[candidate_id])
            records.append(
                {
                    "method": variant,
                    "variant": variant,
                    "task_id": task_id,
                    "geometry_id": int(task.geometry_id),
                    "goal_index": int(task.goal_index),
                    "target_region": int(task.target_region),
                    "trial": int(trials[set_index]),
                    "candidate_id": candidate_id,
                    "n_useful": int(task.n_useful),
                    "alpha": reported_alpha,
                    "quality": q,
                    "true_utility": q,
                    "raw_true_utility": float(
                        numpy_payload["raw_quality"][set_index, candidate_id]
                    ),
                    "critic_score": float(scores[candidate_id]),
                    "region": region,
                    "is_useful": bool(1 <= region <= task.n_useful),
                    "is_decoy": bool(region > task.n_useful),
                    "is_target": bool(region == task.target_region),
                    "selected": bool(candidate_id == critic_index),
                    "critic_selected": bool(candidate_id == critic_index),
                    "oracle_best": bool(candidate_id == oracle_index),
                    "oracle_selected": bool(candidate_id == oracle_index),
                    "random_selected": bool(candidate_id == random_index),
                    "raw_norm": float(raw_norm[set_index, candidate_id]),
                    "normalized_norm": float(normalized_norm[set_index, candidate_id]),
                    "applied_direction_norm": float(
                        applied_norm[set_index, candidate_id]
                    ),
                    "perturbation_norm": float(
                        perturbation_norm[set_index, candidate_id]
                    ),
                    "candidate_alpha": float(candidate_alphas[candidate_id]),
                    "perturbation": float(perturbation_norm[set_index, candidate_id]),
                    "displacement": float(
                        np.linalg.norm(
                            numpy_payload["refined"][set_index, candidate_id]
                            - start_array[set_index]
                        )
                    ),
                    "objective_evaluations": int(config.refinement_steps + 1),
                    "optimization_steps": int(config.refinement_steps),
                    "start": start_array[set_index].copy(),
                    "goal": task.goal.copy(),
                    "candidate": numpy_payload["candidates"][
                        set_index, candidate_id
                    ].copy(),
                    "final": numpy_payload["refined"][set_index, candidate_id].copy(),
                    "method_wall_clock_seconds": float(elapsed),
                }
            )

    summary = summarize_v2_records(
        records,
        useful_threshold=useful_threshold,
        high_quality_threshold=high_quality_threshold,
        diversity_scale=config.diversity_scale,
    )
    alpha_values = diagnostic_alphas.reshape(-1)
    summary.update(
        {
            "critic_regret": float(np.mean([row["critic_regret"] for row in set_rows])),
            "critic_top1_accuracy": float(
                np.mean([row["critic_top1"] for row in set_rows])
            ),
            "spearman_rank_correlation": float(
                np.mean([row["spearman_rank_correlation"] for row in set_rows])
            ),
            "raw_norm": float(np.mean(raw_norm)),
            "normalized_norm": float(np.mean(normalized_norm)),
            "actual_perturbation_magnitude": float(np.mean(perturbation_norm)),
            "mean_alpha": float(np.mean(alpha_values)),
            "alpha_sd": float(np.std(alpha_values, ddof=0)),
            "alpha_p05": float(np.quantile(alpha_values, 0.05)),
            "alpha_p25": float(np.quantile(alpha_values, 0.25)),
            "alpha_p50": float(np.quantile(alpha_values, 0.50)),
            "alpha_p75": float(np.quantile(alpha_values, 0.75)),
            "alpha_p95": float(np.quantile(alpha_values, 0.95)),
            "within_set_alpha_sd": float(
                np.mean([row["alpha_sd"] for row in set_rows])
            ),
        }
    )
    selector_rows = [
        {
            "variant": variant,
            "selector": selector,
            "alpha": reported_alpha,
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
        for selector, observations in selector_observations.items()
    ]
    return EvaluationResult(
        summary=summary,
        records=records,
        candidate_sets=set_rows,
        selectors=selector_rows,
        elapsed_seconds=elapsed,
        tensors=payload,
    )
