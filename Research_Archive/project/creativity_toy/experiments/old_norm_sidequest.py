"""Read-only diagnostic of implicit magnitude in saved unnormalized Gen2 outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..analysis_utils import (
    classify_candidate,
    pearson_correlation,
    spearman_correlation,
    statistics,
)
from ..visualization_gen3 import (
    plot_old_norm_categories,
    plot_old_norm_identity_goal,
    plot_old_norm_quality,
)
from .study_gen3 import _load_csv
from .study_v2 import _write_csv

RELATIONSHIPS = {
    "norm_vs_quality": lambda row: float(row["true_utility"]),
    "norm_vs_critic_score": lambda row: float(row["critic_score"]),
    "norm_vs_useful_indicator": lambda row: float(
        row["category"] in {"target_hit", "useful_non_target"}
    ),
    "norm_vs_decoy_indicator": lambda row: float(row["category"] == "decoy"),
}


def _per_seed_correlations(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for seed in sorted({int(row["seed"]) for row in records}):
        rows = [row for row in records if int(row["seed"]) == seed]
        norm = [float(row["raw_norm"]) for row in rows]
        for name, target in RELATIONSHIPS.items():
            outcome = [target(row) for row in rows]
            output.append(
                {
                    "seed": seed,
                    "relationship": name,
                    "pearson": pearson_correlation(norm, outcome),
                    "spearman": spearman_correlation(norm, outcome),
                    "count": len(rows),
                }
            )
    return output


def _correlation_summary(
    per_seed: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for relationship in RELATIONSHIPS:
        rows = [row for row in per_seed if row["relationship"] == relationship]
        pearson = statistics([float(row["pearson"]) for row in rows])
        spearman = statistics([float(row["spearman"]) for row in rows])
        output.append(
            {
                "relationship": relationship,
                "pearson_mean": pearson["mean"],
                "pearson_sd": pearson["sd"],
                "pearson_ci95": pearson["ci95"],
                "spearman_mean": spearman["mean"],
                "spearman_sd": spearman["sd"],
                "spearman_ci95": spearman["ci95"],
                "n_seeds": pearson["n"],
            }
        )
    return output


def _conditional_correlations(
    records: Sequence[Mapping[str, object]], key: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for seed in sorted({int(row["seed"]) for row in records}):
        seed_rows = [row for row in records if int(row["seed"]) == seed]
        for group in sorted({int(row[key]) for row in seed_rows}):
            rows = [row for row in seed_rows if int(row[key]) == group]
            norm = [float(row["raw_norm"]) for row in rows]
            quality = [float(row["true_utility"]) for row in rows]
            output.append(
                {
                    "seed": seed,
                    key: group,
                    "pearson_norm_quality": pearson_correlation(norm, quality),
                    "spearman_norm_quality": spearman_correlation(norm, quality),
                    "count": len(rows),
                }
            )
    return output


def _category_table(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    categories = ["anchor", "useful_non_target", "target_hit", "decoy", "other"]
    output: list[dict[str, object]] = []
    for category in categories:
        rows = [row for row in records if row["category"] == category]
        norm = np.asarray([float(row["raw_norm"]) for row in rows])
        seed_means = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            seed_values = [
                float(row["raw_norm"]) for row in rows if int(row["seed"]) == seed
            ]
            if seed_values:
                seed_means.append(float(np.mean(seed_values)))
        seed_stats = statistics(seed_means)
        output.append(
            {
                "category": category,
                "mean_old_norm": float(np.mean(norm)) if len(norm) else 0.0,
                "median_old_norm": float(np.median(norm)) if len(norm) else 0.0,
                "old_norm_sd": float(np.std(norm, ddof=1)) if len(norm) > 1 else 0.0,
                "seed_mean_ci95": seed_stats["ci95"],
                "mean_quality": float(
                    np.mean([float(row["true_utility"]) for row in rows])
                )
                if rows
                else 0.0,
                "count": len(rows),
            }
        )
    return output


def _identity_table(
    records: Sequence[Mapping[str, object]], key: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for group in sorted({int(row[key]) for row in records}):
        rows = [row for row in records if int(row[key]) == group]
        norm = np.asarray([float(row["raw_norm"]) for row in rows])
        output.append(
            {
                key: group,
                "mean_norm": float(np.mean(norm)),
                "median_norm": float(np.median(norm)),
                "norm_sd": float(np.std(norm, ddof=1)),
                "mean_quality": float(
                    np.mean([float(row["true_utility"]) for row in rows])
                ),
                "useful_rate": float(
                    np.mean(
                        [
                            row["category"] in {"target_hit", "useful_non_target"}
                            for row in rows
                        ]
                    )
                ),
                "target_rate": float(
                    np.mean([row["category"] == "target_hit" for row in rows])
                ),
                "decoy_rate": float(
                    np.mean([row["category"] == "decoy" for row in rows])
                ),
                "critic_selected_rate": float(
                    np.mean([bool(row["critic_selected"]) for row in rows])
                ),
                "oracle_selected_rate": float(
                    np.mean([bool(row["oracle_selected"]) for row in rows])
                ),
                "count": len(rows),
            }
        )
    return output


def _identity_goal_table(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for candidate_id in sorted({int(row["candidate_id"]) for row in records}):
        for goal_index in sorted({int(row["goal_index"]) for row in records}):
            rows = [
                row
                for row in records
                if int(row["candidate_id"]) == candidate_id
                and int(row["goal_index"]) == goal_index
            ]
            output.append(
                {
                    "candidate_id": candidate_id,
                    "goal_index": goal_index,
                    "mean_norm": float(
                        np.mean([float(row["raw_norm"]) for row in rows])
                    ),
                    "mean_quality": float(
                        np.mean([float(row["true_utility"]) for row in rows])
                    ),
                    "target_rate": float(
                        np.mean([row["category"] == "target_hit" for row in rows])
                    ),
                    "decoy_rate": float(
                        np.mean([row["category"] == "decoy" for row in rows])
                    ),
                    "count": len(rows),
                }
            )
    return output


def _selection_table(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for selector_key, label in [
        ("critic_selected", "critic_selected"),
        ("critic_selected", "critic_not_selected"),
        ("oracle_selected", "oracle_selected"),
        ("oracle_selected", "oracle_not_selected"),
    ]:
        selected_value = not label.endswith("not_selected")
        rows = [row for row in records if bool(row[selector_key]) == selected_value]
        norm = np.asarray([float(row["raw_norm"]) for row in rows])
        output.append(
            {
                "group": label,
                "mean_norm": float(np.mean(norm)),
                "median_norm": float(np.median(norm)),
                "norm_sd": float(np.std(norm, ddof=1)),
                "mean_quality": float(
                    np.mean([float(row["true_utility"]) for row in rows])
                ),
                "count": len(rows),
            }
        )
    return output


def _binned_curve(
    records: Sequence[Mapping[str, object]], bins: int = 10
) -> list[dict[str, object]]:
    norm = np.asarray([float(row["raw_norm"]) for row in records])
    edges = np.quantile(norm, np.linspace(0.0, 1.0, bins + 1))
    output: list[dict[str, object]] = []
    for index in range(bins):
        selected = [
            row
            for row in records
            if float(row["raw_norm"]) >= edges[index]
            and (
                float(row["raw_norm"]) <= edges[index + 1]
                if index == bins - 1
                else float(row["raw_norm"]) < edges[index + 1]
            )
        ]
        output.append(
            {
                "quantile_bin": index + 1,
                "lower_edge": float(edges[index]),
                "upper_edge": float(edges[index + 1]),
                "mean_norm": float(
                    np.mean([float(row["raw_norm"]) for row in selected])
                ),
                "mean_quality": float(
                    np.mean([float(row["true_utility"]) for row in selected])
                ),
                "useful_rate": float(
                    np.mean(
                        [
                            row["category"] in {"target_hit", "useful_non_target"}
                            for row in selected
                        ]
                    )
                ),
                "decoy_rate": float(
                    np.mean([row["category"] == "decoy" for row in selected])
                ),
                "count": len(selected),
            }
        )
    return output


def _regression_features(
    norm: np.ndarray, mean: float, scale: float, quadratic: bool
) -> np.ndarray:
    z = (norm - mean) / max(scale, 1e-12)
    columns = [np.ones_like(z), z]
    if quadratic:
        columns.append(z * z)
    return np.stack(columns, axis=1)


def _quality_regression_cv(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seeds = sorted({int(row["seed"]) for row in records})
    for quadratic, name in [(False, "linear"), (True, "quadratic")]:
        fold_rows: list[dict[str, float]] = []
        for held_out in seeds:
            train = [row for row in records if int(row["seed"]) != held_out]
            test = [row for row in records if int(row["seed"]) == held_out]
            train_norm = np.asarray([float(row["raw_norm"]) for row in train])
            train_y = np.asarray([float(row["true_utility"]) for row in train])
            test_norm = np.asarray([float(row["raw_norm"]) for row in test])
            test_y = np.asarray([float(row["true_utility"]) for row in test])
            mean = float(np.mean(train_norm))
            scale = float(np.std(train_norm))
            beta = np.linalg.lstsq(
                _regression_features(train_norm, mean, scale, quadratic),
                train_y,
                rcond=None,
            )[0]
            prediction = _regression_features(test_norm, mean, scale, quadratic) @ beta
            mse = float(np.mean((test_y - prediction) ** 2))
            denominator = float(np.sum((test_y - np.mean(test_y)) ** 2))
            r2 = 1.0 - float(np.sum((test_y - prediction) ** 2)) / max(
                denominator, 1e-12
            )
            fold_rows.append({"r2": r2, "rmse": float(np.sqrt(mse))})
        for metric in ["r2", "rmse"]:
            stats = statistics([row[metric] for row in fold_rows])
            output.append(
                {
                    "task": "quality_regression",
                    "model": name,
                    "metric": metric,
                    **stats,
                }
            )
    return output


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=np.float64)
    identity = np.eye(x.shape[1]) * 1e-4
    identity[0, 0] = 0.0
    for _ in range(50):
        probability = _sigmoid(x @ beta)
        weight = np.maximum(probability * (1.0 - probability), 1e-6)
        hessian = x.T @ (weight[:, None] * x) + identity
        gradient = x.T @ (y - probability) - identity @ beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.linalg.norm(step) < 1e-8:
            break
    return beta


def _auc(y: np.ndarray, probability: np.ndarray) -> float:
    positive = probability[y == 1.0]
    negative = probability[y == 0.0]
    if len(positive) == 0 or len(negative) == 0:
        return 0.5
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean((comparison > 0.0) + 0.5 * (comparison == 0.0)))


def _classification_cv(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    usable = [
        row
        for row in records
        if row["category"] in {"target_hit", "useful_non_target", "decoy"}
    ]
    seeds = sorted({int(row["seed"]) for row in usable})
    output: list[dict[str, object]] = []
    for quadratic, name in [(False, "linear"), (True, "quadratic")]:
        folds: list[dict[str, float]] = []
        for held_out in seeds:
            train = [row for row in usable if int(row["seed"]) != held_out]
            test = [row for row in usable if int(row["seed"]) == held_out]
            train_norm = np.asarray([float(row["raw_norm"]) for row in train])
            train_y = np.asarray([float(row["category"] != "decoy") for row in train])
            test_norm = np.asarray([float(row["raw_norm"]) for row in test])
            test_y = np.asarray([float(row["category"] != "decoy") for row in test])
            mean = float(np.mean(train_norm))
            scale = float(np.std(train_norm))
            beta = _fit_logistic(
                _regression_features(train_norm, mean, scale, quadratic), train_y
            )
            probability = _sigmoid(
                _regression_features(test_norm, mean, scale, quadratic) @ beta
            )
            prediction = probability >= 0.5
            positive_accuracy = (
                float(np.mean(prediction[test_y == 1.0]))
                if np.any(test_y == 1.0)
                else 0.5
            )
            negative_accuracy = (
                float(np.mean(~prediction[test_y == 0.0]))
                if np.any(test_y == 0.0)
                else 0.5
            )
            folds.append(
                {
                    "balanced_accuracy": 0.5 * (positive_accuracy + negative_accuracy),
                    "roc_auc": _auc(test_y, probability),
                    "log_loss": float(
                        -np.mean(
                            test_y * np.log(np.maximum(probability, 1e-12))
                            + (1.0 - test_y)
                            * np.log(np.maximum(1.0 - probability, 1e-12))
                        )
                    ),
                }
            )
        for metric in ["balanced_accuracy", "roc_auc", "log_loss"]:
            stats = statistics([row[metric] for row in folds])
            output.append(
                {
                    "task": "useful_vs_decoy_classification",
                    "model": name,
                    "metric": metric,
                    **stats,
                }
            )
    return output


def _write_tables_markdown(
    path: Path,
    categories: Sequence[Mapping[str, object]],
    correlations: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# Old unnormalized magnitude diagnostic tables",
        "",
        "Classification precedence is target hit, useful non-target, decoy, anchor, then other. Target/useful require quality at least 0.62.",
        "",
        "| Category | Mean Old Norm | Median | Std | 95% CI of seed means | Mean Quality | Count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in categories:
        lines.append(
            f"| {str(row['category']).replace('_', ' ')} | "
            f"{float(row['mean_old_norm']):.3f} | {float(row['median_old_norm']):.3f} | "
            f"{float(row['old_norm_sd']):.3f} | ±{float(row['seed_mean_ci95']):.3f} | "
            f"{float(row['mean_quality']):.3f} | {int(row['count'])} |"
        )
    lines.extend(
        [
            "",
            "| Relationship | Pearson mean | Pearson 95% CI | Spearman mean | Spearman 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in correlations:
        lines.append(
            f"| {str(row['relationship']).replace('_', ' ')} | "
            f"{float(row['pearson_mean']):.3f} | ±{float(row['pearson_ci95']):.3f} | "
            f"{float(row['spearman_mean']):.3f} | ±{float(row['spearman_ci95']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_by(
    rows: Sequence[Mapping[str, object]], key: str, value: str
) -> Mapping[str, object]:
    return next(row for row in rows if row[key] == value)


def _write_interpretation(
    path: Path,
    correlations: Sequence[Mapping[str, object]],
    categories: Sequence[Mapping[str, object]],
    identities: Sequence[Mapping[str, object]],
    goals: Sequence[Mapping[str, object]],
    selections: Sequence[Mapping[str, object]],
    predictors: Sequence[Mapping[str, object]],
) -> None:
    quality_corr = _row_by(correlations, "relationship", "norm_vs_quality")
    useful = _row_by(categories, "category", "useful_non_target")
    target = _row_by(categories, "category", "target_hit")
    decoy = _row_by(categories, "category", "decoy")
    critic_selected = _row_by(selections, "group", "critic_selected")
    critic_not = _row_by(selections, "group", "critic_not_selected")
    linear_r2 = next(
        row
        for row in predictors
        if row["task"] == "quality_regression"
        and row["model"] == "linear"
        and row["metric"] == "r2"
    )
    quadratic_r2 = next(
        row
        for row in predictors
        if row["task"] == "quality_regression"
        and row["model"] == "quadratic"
        and row["metric"] == "r2"
    )
    identity_means = [float(row["mean_norm"]) for row in identities]
    goal_means = [float(row["mean_norm"]) for row in goals]
    nonlinear_gain = float(quadratic_r2["mean"]) - float(linear_r2["mean"])
    category_gap = abs(float(useful["mean_old_norm"]) - float(decoy["mean_old_norm"]))
    plausible = (
        category_gap > 0.15 or nonlinear_gain > 0.01 or np.ptp(identity_means) > 0.15
    )
    lines = [
        "# Old norm diagnostic",
        "",
        "This is a read-only descriptive analysis of the saved unnormalized Generation-2 candidates. It does not establish causal use of magnitude.",
        "",
        f"1. **Correlation with quality:** per-seed Pearson was {float(quality_corr['pearson_mean']):.3f} ±{float(quality_corr['pearson_ci95']):.3f} (95% CI half-width); Spearman was {float(quality_corr['spearman_mean']):.3f} ±{float(quality_corr['spearman_ci95']):.3f}.",
        f"2. **Monotonic versus nonlinear:** leave-one-seed-out quality R² was {float(linear_r2['mean']):.3f} for linear norm and {float(quadratic_r2['mean']):.3f} for norm plus norm² (difference {nonlinear_gain:+.3f}).",
        f"3. **Useful versus decoy:** useful non-target mean norm was {float(useful['mean_old_norm']):.3f}; decoy mean norm was {float(decoy['mean_old_norm']):.3f}.",
        f"4. **Target candidates:** target-hit mean norm was {float(target['mean_old_norm']):.3f}, compared with {float(useful['mean_old_norm']):.3f} for other useful candidates.",
        f"5. **Candidate identities:** mean norms ranged from {min(identity_means):.3f} to {max(identity_means):.3f} across z/candidate indices.",
        f"6. **Goals:** mean norms ranged from {min(goal_means):.3f} to {max(goal_means):.3f} across goal identities.",
        f"7. **Critic preference:** critic-selected candidates had mean norm {float(critic_selected['mean_norm']):.3f}; unselected candidates averaged {float(critic_not['mean_norm']):.3f}.",
        f"8. **Implicit magnitude channel:** {'the descriptive evidence is compatible with the old norm carrying an implicit magnitude-control signal' if plausible else 'the evidence for a meaningful implicit magnitude channel is weak in this environment'}. This remains correlational.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_old_norm_sidequest(
    source: Path, output: Path, *, useful_threshold: float = 0.62
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    source_file = source / "candidate_diagnostics.csv"
    if not source_file.exists():
        raise FileNotFoundError(f"Missing saved candidate diagnostics: {source_file}")
    records = [
        row for row in _load_csv(source_file) if row["variant"] == "gen2_unnormalized"
    ]
    for row in records:
        row["category"] = classify_candidate(row, useful_threshold)
    if len({int(row["seed"]) for row in records}) < 30:
        raise ValueError("Old-norm side quest requires the completed 30-seed baseline")

    per_seed = _per_seed_correlations(records)
    correlations = _correlation_summary(per_seed)
    per_goal_corr = _conditional_correlations(records, "goal_index")
    per_candidate_corr = _conditional_correlations(records, "candidate_id")
    categories = _category_table(records)
    identities = _identity_table(records, "candidate_id")
    goals = _identity_table(records, "goal_index")
    identity_goal = _identity_goal_table(records)
    selections = _selection_table(records)
    binned = _binned_curve(records)
    predictors = [*_quality_regression_cv(records), *_classification_cv(records)]

    _write_csv(output / "old_norm_candidates.csv", records)
    _write_csv(output / "correlation_per_seed.csv", per_seed)
    _write_csv(output / "correlation_summary.csv", correlations)
    _write_csv(output / "correlation_per_goal_seed.csv", per_goal_corr)
    _write_csv(output / "correlation_per_candidate_seed.csv", per_candidate_corr)
    _write_csv(output / "category_summary.csv", categories)
    _write_csv(output / "candidate_identity_summary.csv", identities)
    _write_csv(output / "goal_summary.csv", goals)
    _write_csv(output / "candidate_goal_summary.csv", identity_goal)
    _write_csv(output / "selection_norm_summary.csv", selections)
    _write_csv(output / "magnitude_quality_bins.csv", binned)
    _write_csv(output / "norm_only_predictors.csv", predictors)
    _write_tables_markdown(output / "OLD_NORM_TABLES.md", categories, correlations)
    _write_interpretation(
        output / "OLD_NORM_DIAGNOSTIC.md",
        correlations,
        categories,
        identities,
        goals,
        selections,
        predictors,
    )
    plot_old_norm_quality(records, output / "old_norm_vs_quality.png")
    plot_old_norm_categories(records, output / "old_norm_by_category.png")
    plot_old_norm_identity_goal(
        identities, goals, output / "old_norm_by_candidate_goal.png"
    )

    aggregate: dict[str, object] = {
        "source": str(source),
        "seed_count": len({int(row["seed"]) for row in records}),
        "candidate_count": len(records),
        "correlations": correlations,
        "categories": categories,
        "predictors": predictors,
    }
    (output / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
