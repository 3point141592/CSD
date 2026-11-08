"""Quality, diversity, coverage, collapse, and uncertainty metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

Array = np.ndarray


def _pairwise_distances(points: Array) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    euclidean: list[float] = []
    cosine: list[float] = []
    norms = np.linalg.norm(points, axis=1)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            euclidean.append(float(np.linalg.norm(points[i] - points[j])))
            similarity = float(points[i] @ points[j]) / max(
                float(norms[i] * norms[j]), 1e-12
            )
            cosine.append(1.0 - np.clip(similarity, -1.0, 1.0))
    return float(np.mean(euclidean)), float(np.mean(cosine))


def summarize_records(
    records: Sequence[Mapping[str, object]], quality_threshold: float
) -> dict[str, float]:
    if not records:
        raise ValueError("Cannot summarize zero records")
    qualities = np.array([float(row["quality"]) for row in records])
    perturbations = np.array([float(row["perturbation"]) for row in records])
    landscape_ids = sorted({int(row["landscape_id"]) for row in records})

    coverage: list[float] = []
    euclidean: list[float] = []
    cosine: list[float] = []
    collapse: list[float] = []
    high_quality_fraction: list[float] = []
    anchor_escape: list[float] = []

    for landscape_id in landscape_ids:
        subset = [row for row in records if int(row["landscape_id"]) == landscape_id]
        good = [row for row in subset if float(row["quality"]) >= quality_threshold]
        n_islands = int(subset[0]["n_islands"])
        discovered = {int(row["island"]) for row in good}
        coverage.append(len(discovered) / n_islands)
        high_quality_fraction.append(len(good) / len(subset))
        anchor_escape.append(
            sum(int(row["island"]) != 0 for row in good) / len(good) if good else 0.0
        )
        points = (
            np.stack([np.asarray(row["final"]) for row in good])
            if good
            else np.empty((0, 1))
        )
        euc, cos = _pairwise_distances(points)
        euclidean.append(euc)
        cosine.append(cos)

        if len(good) < 2:
            collapse.append(1.0)
        else:
            counts = Counter(int(row["island"]) for row in good)
            same_pairs = sum(count * (count - 1) for count in counts.values())
            all_pairs = len(good) * (len(good) - 1)
            collapse.append(same_pairs / all_pairs)

    per_candidate_budget = int(records[0]["evaluation_budget"])
    candidates_per_trial = max(
        1,
        len(
            [
                row
                for row in records
                if int(row["landscape_id"]) == landscape_ids[0]
                and int(row["trial"]) == 0
            ]
        ),
    )
    return {
        "anchor_escape_rate": float(np.mean(anchor_escape)),
        "mean_quality": float(np.mean(qualities)),
        "quality_std_across_candidates": float(np.std(qualities)),
        "high_quality_fraction": float(np.mean(high_quality_fraction)),
        "euclidean_diversity": float(np.mean(euclidean)),
        "cosine_diversity": float(np.mean(cosine)),
        "coverage": float(np.mean(coverage)),
        "collapse_rate": float(np.mean(collapse)),
        "perturbation_magnitude": float(np.mean(perturbations)),
        "candidate_budget_per_trial": float(candidates_per_trial),
        "objective_calls_per_trial": float(candidates_per_trial * per_candidate_budget),
    }


def aggregate_seeds(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Mean, sample SD, and normal 95% CI over independent experiment seeds."""
    if not rows:
        return {}
    excluded = {"seed", "method", "alpha", "variant", "rank", "parameters"}
    keys = [
        key
        for key, value in rows[0].items()
        if key not in excluded and isinstance(value, (int, float, np.number))
    ]
    result: dict[str, float] = {"n_seeds": float(len(rows))}
    for key in keys:
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(len(values)) if len(values) > 1 else 0.0
        result[f"{key}_mean"] = mean
        result[f"{key}_sd"] = sd
        result[f"{key}_ci95"] = float(ci)
    return result


def paired_seed_differences(
    rows: Sequence[Mapping[str, object]],
    method_a: str,
    method_b: str,
    metric: str,
) -> dict[str, float]:
    """Paired effect and confidence interval using seeds as independent replicates."""
    a = {
        int(row["seed"]): float(row[metric])
        for row in rows
        if row["method"] == method_a
    }
    b = {
        int(row["seed"]): float(row[metric])
        for row in rows
        if row["method"] == method_b
    }
    common = sorted(set(a) & set(b))
    if not common:
        return {"mean_difference": float("nan"), "ci95": float("nan"), "n": 0.0}
    differences = np.array([a[seed] - b[seed] for seed in common])
    sd = float(np.std(differences, ddof=1)) if len(common) > 1 else 0.0
    return {
        "mean_difference": float(np.mean(differences)),
        "ci95": float(1.96 * sd / np.sqrt(len(common))) if len(common) > 1 else 0.0,
        "n": float(len(common)),
    }
