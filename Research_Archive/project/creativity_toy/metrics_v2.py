"""Region-aware metrics for useful novelty versus arbitrary distance."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

Array = np.ndarray


def _mean_pairwise(points: Array) -> float:
    if len(points) < 2:
        return 0.0
    distances: list[float] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(float(np.linalg.norm(points[i] - points[j])))
    return float(np.mean(distances))


def summarize_v2_records(
    records: Sequence[Mapping[str, object]],
    *,
    useful_threshold: float,
    high_quality_threshold: float,
    diversity_scale: float,
) -> dict[str, float]:
    if not records:
        raise ValueError("Cannot summarize empty V2 records")
    qualities = np.array([float(row["quality"]) for row in records])
    displacements = np.array([float(row["displacement"]) for row in records])
    perturbations = np.array([float(row["perturbation"]) for row in records])
    selected = [row for row in records if bool(row["selected"])]
    oracle = [row for row in records if bool(row["oracle_best"])]
    task_ids = sorted({int(row["task_id"]) for row in records})

    coverages: list[float] = []
    useful_diversities: list[float] = []
    strict_diversities: list[float] = []
    collapses: list[float] = []
    target_hits: list[float] = []
    for task_id in task_ids:
        subset = [row for row in records if int(row["task_id"]) == task_id]
        good_useful = [
            row
            for row in subset
            if bool(row["is_useful"]) and float(row["quality"]) >= useful_threshold
        ]
        discovered = {int(row["region"]) for row in good_useful}
        coverages.append(len(discovered) / int(subset[0]["n_useful"]))
        useful_points = (
            np.stack([np.asarray(row["final"]) for row in good_useful])
            if good_useful
            else np.empty((0, 1))
        )
        useful_diversities.append(_mean_pairwise(useful_points))
        high_quality = [
            row
            for row in good_useful
            if float(row["quality"]) >= high_quality_threshold
        ]
        points = (
            np.stack([np.asarray(row["final"]) for row in high_quality])
            if high_quality
            else np.empty((0, 1))
        )
        strict_diversities.append(_mean_pairwise(points))
        if len(good_useful) < 2:
            collapses.append(1.0)
        else:
            counts = Counter(int(row["region"]) for row in good_useful)
            same = sum(count * (count - 1) for count in counts.values())
            all_pairs = len(good_useful) * (len(good_useful) - 1)
            collapses.append(same / all_pairs)

        trial_ids = sorted({int(row["trial"]) for row in subset})
        for trial in trial_ids:
            trial_rows = [row for row in subset if int(row["trial"]) == trial]
            target_hits.append(
                float(
                    any(
                        bool(row["is_target"])
                        and float(row["quality"]) >= useful_threshold
                        for row in trial_rows
                    )
                )
            )

    diversity = float(np.mean(useful_diversities))
    strict_diversity = float(np.mean(strict_diversities))
    mean_quality = float(np.mean(qualities))
    normalized_diversity = min(diversity / max(diversity_scale, 1e-12), 1.0)
    qd_score = (
        2.0
        * mean_quality
        * normalized_diversity
        / max(mean_quality + normalized_diversity, 1e-12)
    )
    if np.std(displacements) < 1e-12 or np.std(qualities) < 1e-12:
        distance_quality_correlation = 0.0
    else:
        distance_quality_correlation = float(
            np.corrcoef(displacements, qualities)[0, 1]
        )

    first_task = task_ids[0]
    first_trial = min(
        int(row["trial"]) for row in records if int(row["task_id"]) == first_task
    )
    candidates_per_trial = len(
        [
            row
            for row in records
            if int(row["task_id"]) == first_task and int(row["trial"]) == first_trial
        ]
    )
    evaluations_per_candidate = int(records[0]["objective_evaluations"])
    steps_per_candidate = int(records[0]["optimization_steps"])
    total_method_seconds = float(records[0]["method_wall_clock_seconds"])
    trial_count = len(selected)
    return {
        "mean_quality": mean_quality,
        "selected_quality": float(np.mean([float(row["quality"]) for row in selected])),
        "oracle_best_quality": float(
            np.mean([float(row["quality"]) for row in oracle])
        ),
        "euclidean_diversity": diversity,
        "strict_high_quality_diversity": strict_diversity,
        "coverage": float(np.mean(coverages)),
        "collapse_rate": float(np.mean(collapses)),
        "decoy_rate": float(np.mean([bool(row["is_decoy"]) for row in records])),
        "selected_decoy_rate": float(
            np.mean([bool(row["is_decoy"]) for row in selected])
        ),
        "useful_exploration_rate": float(
            np.mean(
                [
                    bool(row["is_useful"]) and float(row["quality"]) >= useful_threshold
                    for row in records
                ]
            )
        ),
        "anchor_escape_rate": float(
            np.mean([int(row["region"]) != 0 for row in records])
        ),
        "target_hit_rate": float(np.mean(target_hits)),
        "selected_target_rate": float(
            np.mean(
                [
                    bool(row["is_target"]) and float(row["quality"]) >= useful_threshold
                    for row in selected
                ]
            )
        ),
        "quality_diversity_score": float(qd_score),
        "distance_quality_correlation": distance_quality_correlation,
        "perturbation_magnitude": float(np.mean(perturbations)),
        "final_displacement": float(np.mean(displacements)),
        "candidate_budget_per_trial": float(candidates_per_trial),
        "objective_evaluations_per_trial": float(
            candidates_per_trial * evaluations_per_candidate
        ),
        "optimization_steps_per_trial": float(
            candidates_per_trial * steps_per_candidate
        ),
        "wall_clock_seconds_per_trial": float(
            total_method_seconds / max(trial_count, 1)
        ),
    }
