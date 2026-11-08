"""Statistics, specialization diagnostics, plots, and reports for mode preservation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from creativity_toy.analysis_utils import statistics
from creativity_toy.config_gen3 import preset_gen3
from creativity_toy.landscapes_v2 import V2LandscapeFamily

from .config_rl import RLStudyConfig
from .study import read_csv, write_csv
from .study_mode import _datasets

BASELINE = "baseline_lambda_0"
SELECTED = "gated_shared_selected"
VARIANT_LABELS = {
    BASELINE: "No mode reward",
    SELECTED: "Quality-gated shared",
    "ungated_shared_ablation": "Ungated shared",
    "gated_candidate_specific_ablation": "Gated candidate-specific",
}
PRIMARY_METRICS = [
    "candidate_mean_quality",
    "selected_quality",
    "proposal_recall",
    "routing_recall",
    "useful_selected",
    "useful_region_count",
    "useful_region_count_0",
    "useful_region_count_1",
    "useful_region_count_2_plus",
    "candidate_decoy_rate",
    "selected_decoy_rate",
    "direction_cosine_similarity",
    "direction_cosine_distance",
    "minimum_direction_distance",
    "effective_directional_rank",
    "candidate_spatial_diversity",
    "alpha_mean",
    "alpha_sd",
    "within_set_alpha_sd",
    "portfolio_collapse",
    "useful_oracle_success",
    "quality_oracle_quality",
    "training_seconds",
    "candidate_outcomes_per_second",
    "peak_allocated_bytes",
    "base_reward",
    "mode_reward",
    "weighted_mode_reward",
    "total_reward",
    "mode_reward_to_base_ratio",
]


def _aggregate(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, selected in groups.items():
        result: dict[str, object] = dict(zip(keys, key))
        result["n"] = len(selected)
        for metric in metrics:
            if metric not in selected[0] or selected[0][metric] in (None, ""):
                continue
            stat = statistics([float(row[metric]) for row in selected])
            result[metric] = stat["mean"]
            result[f"{metric}_sd"] = stat["sd"]
            result[f"{metric}_ci95"] = stat["ci95"]
        output.append(result)
    return output


def _paired(
    rows: Sequence[Mapping[str, object]],
    *,
    left: str,
    right: str,
    metrics: Sequence[str],
    landscape_variant: str = "primary_iid",
) -> list[dict[str, object]]:
    selected_rows = [
        row
        for row in rows
        if landscape_variant == "primary_iid"
        or row.get("landscape_variant") == landscape_variant
    ]
    left_rows = {
        int(row["seed"]): row for row in selected_rows if row["variant"] == left
    }
    right_rows = {
        int(row["seed"]): row for row in selected_rows if row["variant"] == right
    }
    seeds = sorted(set(left_rows) & set(right_rows))
    output = []
    for metric in metrics:
        differences = [
            float(left_rows[seed][metric]) - float(right_rows[seed][metric])
            for seed in seeds
        ]
        stat = statistics(differences)
        output.append(
            {
                "landscape_variant": landscape_variant,
                "left": left,
                "right": right,
                "metric": metric,
                "paired_difference": stat["mean"],
                "paired_difference_sd": stat["sd"],
                "paired_difference_ci95": stat["ci95"],
                "ci95_low": stat["mean"] - stat["ci95"],
                "ci95_high": stat["mean"] + stat["ci95"],
                "ci95_excludes_zero": abs(stat["mean"]) > stat["ci95"],
                "n": len(seeds),
            }
        )
    return output


def _truth(value: object) -> bool:
    return str(value).lower() == "true"


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive))) if positive.size else 0.0


def _specialization_per_seed(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    keys = sorted({(int(row["seed"]), str(row["variant"])) for row in rows})
    for seed, variant in keys:
        selected = [
            row
            for row in rows
            if int(row["seed"]) == seed and row["variant"] == variant
        ]
        candidates = sorted({int(row["candidate_id"]) for row in selected})
        useful_regions = sorted(
            {int(row["region"]) for row in selected if _truth(row["is_useful"])}
        )
        region_to_index = {region: index for index, region in enumerate(useful_regions)}
        counts = np.zeros((len(candidates), len(useful_regions)), dtype=np.float64)
        alpha_means, direction_concentrations = [], []
        useful_rates, decoy_rates, entropies = [], [], []
        for candidate_id in candidates:
            candidate_rows = [
                row for row in selected if int(row["candidate_id"]) == candidate_id
            ]
            useful_rows = [row for row in candidate_rows if _truth(row["is_useful"])]
            for row in useful_rows:
                counts[candidate_id, region_to_index[int(row["region"])]] += 1
            if useful_rows:
                probabilities = counts[candidate_id] / np.sum(counts[candidate_id])
                entropies.append(_entropy(probabilities))
            else:
                entropies.append(0.0)
            alpha_means.append(np.mean([float(row["alpha"]) for row in candidate_rows]))
            mean_direction = np.asarray(
                [
                    np.mean([float(row["direction_0"]) for row in candidate_rows]),
                    np.mean([float(row["direction_1"]) for row in candidate_rows]),
                ]
            )
            direction_concentrations.append(float(np.linalg.norm(mean_direction)))
            useful_rates.append(
                np.mean([_truth(row["is_useful"]) for row in candidate_rows])
            )
            decoy_rates.append(
                np.mean([_truth(row["is_decoy"]) for row in candidate_rows])
            )
        total = float(np.sum(counts))
        if total > 0:
            joint = counts / total
            pz = np.sum(joint, axis=1, keepdims=True)
            pr = np.sum(joint, axis=0, keepdims=True)
            expected = pz @ pr
            mask = joint > 0
            mutual_information = float(
                np.sum(joint[mask] * np.log(joint[mask] / expected[mask]))
            )
            region_entropy = _entropy(pr.reshape(-1))
            normalized_mi = mutual_information / max(region_entropy, 1e-12)
        else:
            mutual_information = normalized_mi = region_entropy = 0.0
        output.append(
            {
                "seed": seed,
                "variant": variant,
                "candidate_region_mutual_information": mutual_information,
                "candidate_region_normalized_mi": normalized_mi,
                "useful_region_entropy": region_entropy,
                "mean_candidate_region_entropy": float(np.mean(entropies)),
                "between_candidate_alpha_sd": float(np.std(alpha_means, ddof=0)),
                "alpha_mean_range": float(np.max(alpha_means) - np.min(alpha_means)),
                "mean_direction_concentration": float(
                    np.mean(direction_concentrations)
                ),
                "between_candidate_useful_rate_sd": float(np.std(useful_rates, ddof=0)),
                "between_candidate_decoy_rate_sd": float(np.std(decoy_rates, ddof=0)),
                "maximum_candidate_decoy_rate": float(np.max(decoy_rates)),
                "useful_observations": int(total),
            }
        )
    return output


def _candidate_identity_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    variants = sorted({str(row["variant"]) for row in rows})
    max_region = max(int(row["region"]) for row in rows if _truth(row["is_useful"]))
    for variant in variants:
        for candidate_id in range(8):
            selected = [
                row
                for row in rows
                if row["variant"] == variant
                and int(row["candidate_id"]) == candidate_id
            ]
            useful = [row for row in selected if _truth(row["is_useful"])]
            region_counts = np.asarray(
                [
                    sum(
                        _truth(row["is_useful"]) and int(row["region"]) == region
                        for row in selected
                    )
                    for region in range(1, max_region + 1)
                ],
                dtype=np.float64,
            )
            conditional = region_counts / max(np.sum(region_counts), 1.0)
            row: dict[str, object] = {
                "variant": variant,
                "candidate_id": candidate_id,
                "count": len(selected),
                "mean_quality": float(
                    np.mean([float(value["quality"]) for value in selected])
                ),
                "mean_alpha": float(
                    np.mean([float(value["alpha"]) for value in selected])
                ),
                "alpha_sd": float(
                    np.std([float(value["alpha"]) for value in selected], ddof=0)
                ),
                "mean_direction_0": float(
                    np.mean([float(value["direction_0"]) for value in selected])
                ),
                "mean_direction_1": float(
                    np.mean([float(value["direction_1"]) for value in selected])
                ),
                "useful_rate": float(
                    np.mean([_truth(value["is_useful"]) for value in selected])
                ),
                "target_rate": float(
                    np.mean([_truth(value["is_target"]) for value in selected])
                ),
                "anchor_rate": float(
                    np.mean([_truth(value["is_anchor"]) for value in selected])
                ),
                "decoy_rate": float(
                    np.mean([_truth(value["is_decoy"]) for value in selected])
                ),
                "useful_region_entropy": _entropy(conditional) if useful else 0.0,
            }
            for region in range(1, max_region + 1):
                row[f"useful_region_{region}_frequency"] = region_counts[
                    region - 1
                ] / len(selected)
                row[f"p_region_{region}_given_useful"] = conditional[region - 1]
            output.append(row)
    return output


def _candidate_goal_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    keys = sorted(
        {
            (str(row["variant"]), int(row["candidate_id"]), int(row["goal_index"]))
            for row in rows
        }
    )
    for variant, candidate_id, goal in keys:
        selected = [
            row
            for row in rows
            if row["variant"] == variant
            and int(row["candidate_id"]) == candidate_id
            and int(row["goal_index"]) == goal
        ]
        output.append(
            {
                "variant": variant,
                "candidate_id": candidate_id,
                "goal_index": goal,
                "count": len(selected),
                "quality": float(np.mean([float(row["quality"]) for row in selected])),
                "alpha": float(np.mean([float(row["alpha"]) for row in selected])),
                "useful_rate": float(
                    np.mean([_truth(row["is_useful"]) for row in selected])
                ),
                "target_rate": float(
                    np.mean([_truth(row["is_target"]) for row in selected])
                ),
                "decoy_rate": float(
                    np.mean([_truth(row["is_decoy"]) for row in selected])
                ),
            }
        )
    return output


def _category_alpha_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        for category in ["target", "useful_non_target", "anchor", "decoy", "other"]:
            values = [
                float(row["alpha"])
                for row in rows
                if row["variant"] == variant and row["category"] == category
            ]
            if not values:
                continue
            stat = statistics(values)
            output.append(
                {
                    "variant": variant,
                    "category": category,
                    "count": len(values),
                    "alpha_mean": stat["mean"],
                    "alpha_sd": stat["sd"],
                    "alpha_ci95": stat["ci95"],
                    "alpha_median": float(np.median(values)),
                }
            )
    return output


def _plot_lambda_sweep(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["lambda_mode"]))
    x = [float(row["lambda_mode"]) for row in ordered]
    metrics = [
        ("proposal_recall", "proposal recall"),
        ("useful_region_count", "useful regions"),
        ("candidate_mean_quality", "candidate quality"),
        ("candidate_decoy_rate", "candidate decoy rate"),
        ("direction_cosine_distance", "cosine distance"),
        ("mode_reward_to_base_ratio", "weighted mode/base"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        y = [float(row[metric]) for row in ordered]
        error = [float(row[f"{metric}_ci95"]) for row in ordered]
        axis.errorbar(x, y, yerr=error, marker="o", capsize=3)
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlabel("lambda_mode (validation)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_training_curves(
    checkpoint_rows: Sequence[Mapping[str, object]],
    history_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    variants = [BASELINE, SELECTED]
    metrics = [
        ("base_reward", "base reward"),
        ("mode_reward", "mode reward"),
        ("total_reward", "reported total reward"),
        ("proposal_recall", "proposal recall"),
        ("useful_oracle_success", "useful-oracle success"),
        ("candidate_mean_quality", "candidate quality"),
        ("candidate_decoy_rate", "candidate decoy rate"),
        ("direction_cosine_distance", "cosine distance"),
        ("candidate_spatial_diversity", "spatial diversity"),
        ("alpha_mean", "alpha mean"),
        ("alpha_sd", "alpha SD"),
    ]
    figure, axes = plt.subplots(3, 4, figsize=(16, 10.5))
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for variant in variants:
            selected = [row for row in checkpoint_rows if row["variant"] == variant]
            budgets = sorted({int(float(row["requested_budget"])) for row in selected})
            means = [
                float(
                    np.mean(
                        [
                            float(row[metric])
                            for row in selected
                            if int(float(row["requested_budget"])) == budget
                        ]
                    )
                )
                for budget in budgets
            ]
            axis.plot(budgets, means, marker="o", label=VARIANT_LABELS[variant])
        axis.set_title(title)
        axis.set_xlabel("black-box candidate outcomes")
        axis.grid(alpha=0.25)
    entropy_axis = axes.ravel()[-1]
    for variant in variants:
        selected = [row for row in history_rows if row["variant"] == variant]
        steps = sorted({int(float(row["candidate_evaluations"])) for row in selected})
        means = [
            float(
                np.mean(
                    [
                        float(row["policy_entropy"])
                        for row in selected
                        if int(float(row["candidate_evaluations"])) == step
                    ]
                )
            )
            for step in steps
        ]
        entropy_axis.plot(steps, means, label=VARIANT_LABELS[variant])
    entropy_axis.set_title("policy entropy")
    entropy_axis.set_xlabel("black-box candidate outcomes")
    entropy_axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_primary(summary: Sequence[Mapping[str, object]], path: Path) -> None:
    rows = [row for row in summary if row["variant"] in {BASELINE, SELECTED}]
    metrics = [
        ("proposal_recall", "proposal recall"),
        ("useful_region_count", "useful regions"),
        ("candidate_mean_quality", "candidate quality"),
        ("candidate_decoy_rate", "candidate decoy"),
        ("direction_cosine_distance", "cosine distance"),
        ("useful_selected", "router useful selected"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    labels = [VARIANT_LABELS[str(row["variant"])] for row in rows]
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        axis.bar(
            labels,
            [float(row[metric]) for row in rows],
            yerr=[float(row[f"{metric}_ci95"]) for row in rows],
            capsize=4,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_specialization(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    region_columns = sorted(
        [
            key
            for key in rows[0]
            if key.startswith("useful_region_") and key.endswith("_frequency")
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, variant in zip(axes, [BASELINE, SELECTED]):
        selected = sorted(
            [row for row in rows if row["variant"] == variant],
            key=lambda row: int(row["candidate_id"]),
        )
        matrix = np.asarray(
            [[float(row[column]) for column in region_columns] for row in selected]
        )
        image = axis.imshow(matrix, aspect="auto", cmap="magma")
        axis.set_title(VARIANT_LABELS[variant])
        axis.set_xlabel("useful region")
        axis.set_ylabel("candidate identity z_i")
        axis.set_xticks(range(len(region_columns)), range(1, len(region_columns) + 1))
        axis.set_yticks(range(8), range(8))
        figure.colorbar(image, ax=axis, label="P(region | z_i), all proposals")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_alpha_identity(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for variant in [BASELINE, SELECTED]:
        selected = sorted(
            [row for row in rows if row["variant"] == variant],
            key=lambda row: int(row["candidate_id"]),
        )
        axis.plot(
            [int(row["candidate_id"]) for row in selected],
            [float(row["mean_alpha"]) for row in selected],
            marker="o",
            label=VARIANT_LABELS[variant],
        )
    axis.set_xlabel("candidate identity z_i")
    axis.set_ylabel("mean alpha")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_generalization(summary: Sequence[Mapping[str, object]], path: Path) -> None:
    landscapes = [
        "iid",
        "rotated",
        "translated",
        "width_shift",
        "more_peaks",
        "more_decoys",
        "new_goals",
        "unseen",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, title in zip(
        axes,
        ["proposal_recall", "useful_region_count", "candidate_decoy_rate"],
        ["proposal recall", "useful regions", "candidate decoy rate"],
    ):
        for method in [BASELINE, SELECTED]:
            lookup = {
                str(row["landscape_variant"]): row
                for row in summary
                if row["variant"] == method
            }
            axis.plot(
                range(len(landscapes)),
                [float(lookup[value][metric]) for value in landscapes],
                marker="o",
                label=VARIANT_LABELS[method],
            )
        axis.set_xticks(range(len(landscapes)), landscapes, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_representative(
    candidate_rows: Sequence[Mapping[str, object]], *, seed: int, path: Path
) -> None:
    base = preset_gen3("main")
    family = V2LandscapeFamily(base.base.landscape)
    _, _, _, _, tasks, _ = _datasets(family, base, RLStudyConfig(), seed)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, variant in zip(axes, [BASELINE, SELECTED]):
        selected = [
            row
            for row in candidate_rows
            if int(row["seed"]) == seed
            and row["variant"] == variant
            and int(row["set_index"]) == 0
        ]
        task = tasks[int(selected[0]["task_id"])]
        starts = np.asarray(
            [
                [
                    float(row["candidate_0"])
                    - float(row["alpha"]) * float(row["direction_0"]),
                    float(row["candidate_1"])
                    - float(row["alpha"]) * float(row["direction_1"]),
                ]
                for row in selected
            ]
        )
        start = np.mean(starts, axis=0)
        points = np.asarray(
            [[float(row["final_0"]), float(row["final_1"])] for row in selected]
        )
        centers = np.concatenate(
            [task.anchor[None, :], task.useful_centers, task.decoy_centers], axis=0
        )
        extent = np.concatenate([points, centers, start[None, :]], axis=0)
        low, high = np.min(extent, axis=0) - 1.0, np.max(extent, axis=0) + 1.0
        xs, ys = np.linspace(low[0], high[0], 220), np.linspace(low[1], high[1], 220)
        xx, yy = np.meshgrid(xs, ys)
        axis.contourf(
            xx, yy, task.quality(np.stack([xx, yy], axis=-1)), levels=25, cmap="viridis"
        )
        axis.scatter(*task.anchor, marker="*", s=150, color="white", edgecolor="black")
        axis.scatter(
            task.useful_centers[:, 0],
            task.useful_centers[:, 1],
            facecolors="none",
            edgecolors="lime",
            s=80,
        )
        axis.scatter(
            task.decoy_centers[:, 0],
            task.decoy_centers[:, 1],
            marker="x",
            color="red",
            s=70,
        )
        axis.scatter(*start, marker="s", color="cyan", s=70)
        for row in selected:
            candidate = np.asarray(
                [float(row["candidate_0"]), float(row["candidate_1"])]
            )
            final = np.asarray([float(row["final_0"]), float(row["final_1"])])
            axis.arrow(
                start[0],
                start[1],
                candidate[0] - start[0],
                candidate[1] - start[1],
                width=0.012,
                head_width=0.12,
                alpha=0.7,
                length_includes_head=True,
            )
            axis.scatter(*final, marker="o", color="orange", s=40)
            axis.annotate(
                f"z{row['candidate_id']} a={float(row['alpha']):.1f} r={row['region']}",
                final,
                fontsize=6.5,
                xytext=(3, 3),
                textcoords="offset points",
            )
        axis.set_title(f"{VARIANT_LABELS[variant]}\nfixed seed {seed}, first test set")
        axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _fmt(value: object) -> str:
    return f"{float(value):.3f}"


def generate_mode_reports(
    *,
    main_dir: Path = Path("results/mode_preservation_main"),
    sweep_dir: Path = Path("results/mode_preservation_sweep"),
) -> dict[str, object]:
    primary = read_csv(main_dir / "primary_per_seed.csv")
    candidates = read_csv(main_dir / "candidate_diagnostics.csv")
    checkpoints = read_csv(main_dir / "checkpoint_metrics.csv")
    history = read_csv(main_dir / "training_history.csv")
    generalization = read_csv(main_dir / "generalization_per_seed.csv")
    sweep = read_csv(sweep_dir / "lambda_sweep_summary.csv")
    primary_summary = _aggregate(primary, ["variant"], PRIMARY_METRICS)
    primary_paired = _paired(
        primary, left=SELECTED, right=BASELINE, metrics=PRIMARY_METRICS
    )
    ablation_paired = []
    for variant in ["ungated_shared_ablation", "gated_candidate_specific_ablation"]:
        ablation_paired.extend(
            _paired(primary, left=variant, right=BASELINE, metrics=PRIMARY_METRICS)
        )
        ablation_paired.extend(
            _paired(primary, left=variant, right=SELECTED, metrics=PRIMARY_METRICS)
        )
    generalization_summary = _aggregate(
        generalization, ["variant", "landscape_variant"], PRIMARY_METRICS
    )
    generalization_paired = []
    landscapes = [
        "iid",
        "rotated",
        "translated",
        "width_shift",
        "more_peaks",
        "more_decoys",
        "new_goals",
        "unseen",
    ]
    for landscape in landscapes:
        generalization_paired.extend(
            _paired(
                generalization,
                left=SELECTED,
                right=BASELINE,
                metrics=[
                    "proposal_recall",
                    "useful_region_count",
                    "candidate_decoy_rate",
                    "candidate_mean_quality",
                    "direction_cosine_distance",
                    "routing_recall",
                    "useful_selected",
                    "selected_quality",
                ],
                landscape_variant=landscape,
            )
        )
    specialization_seed = _specialization_per_seed(candidates)
    specialization_metrics = [
        "candidate_region_mutual_information",
        "candidate_region_normalized_mi",
        "useful_region_entropy",
        "mean_candidate_region_entropy",
        "between_candidate_alpha_sd",
        "alpha_mean_range",
        "mean_direction_concentration",
        "between_candidate_useful_rate_sd",
        "between_candidate_decoy_rate_sd",
        "maximum_candidate_decoy_rate",
    ]
    specialization_summary = _aggregate(
        specialization_seed, ["variant"], specialization_metrics
    )
    specialization_paired = _paired(
        specialization_seed,
        left=SELECTED,
        right=BASELINE,
        metrics=[
            metric
            for metric in specialization_metrics
            if metric != "useful_region_entropy"
        ],
    )
    identity = _candidate_identity_summary(candidates)
    goals = _candidate_goal_summary(candidates)
    categories = _category_alpha_summary(candidates)
    write_csv(main_dir / "primary_summary.csv", primary_summary)
    write_csv(main_dir / "paired_selected_vs_baseline.csv", primary_paired)
    write_csv(main_dir / "ablation_paired.csv", ablation_paired)
    write_csv(main_dir / "generalization_summary.csv", generalization_summary)
    write_csv(main_dir / "generalization_paired.csv", generalization_paired)
    write_csv(main_dir / "specialization_per_seed.csv", specialization_seed)
    write_csv(main_dir / "specialization_summary.csv", specialization_summary)
    write_csv(main_dir / "specialization_paired.csv", specialization_paired)
    write_csv(main_dir / "candidate_identity_summary.csv", identity)
    write_csv(main_dir / "candidate_goal_summary.csv", goals)
    write_csv(main_dir / "alpha_by_category.csv", categories)
    _plot_lambda_sweep(sweep, main_dir / "lambda_mode_sweep.png")
    _plot_training_curves(checkpoints, history, main_dir / "mode_training_curves.png")
    _plot_primary(primary_summary, main_dir / "mode_primary_metrics.png")
    _plot_specialization(identity, main_dir / "candidate_region_specialization.png")
    _plot_alpha_identity(identity, main_dir / "alpha_by_candidate_identity.png")
    _plot_generalization(generalization_summary, main_dir / "mode_generalization.png")
    for seed in [7, 19]:
        _plot_representative(
            candidates, seed=seed, path=main_dir / f"mode_vectors_seed_{seed}.png"
        )

    lookup = {str(row["variant"]): row for row in primary_summary}
    paired_lookup = {str(row["metric"]): row for row in primary_paired}
    specialization_lookup = {str(row["variant"]): row for row in specialization_summary}
    specialization_pair_lookup = {
        str(row["metric"]): row for row in specialization_paired
    }
    ablation_lookup = {
        (str(row["left"]), str(row["right"]), str(row["metric"])): row
        for row in ablation_paired
    }

    def effect(metric: str) -> str:
        row = paired_lookup[metric]
        return (
            f"{float(row['paired_difference']):+.3f} "
            f"(95% CI {float(row['ci95_low']):+.3f} to {float(row['ci95_high']):+.3f})"
        )

    def ablation_effect(left: str, right: str, metric: str) -> str:
        row = ablation_lookup[(left, right, metric)]
        return (
            f"{float(row['paired_difference']):+.3f} "
            f"(95% CI {float(row['ci95_low']):+.3f} to {float(row['ci95_high']):+.3f})"
        )

    baseline = lookup[BASELINE]
    selected = lookup[SELECTED]
    overhead = 100.0 * (
        float(selected["training_seconds"]) / float(baseline["training_seconds"]) - 1.0
    )
    memory_overhead = (
        float(selected["peak_allocated_bytes"])
        - float(baseline["peak_allocated_bytes"])
    ) / (1024.0 * 1024.0)
    sweep_lines = [
        "| Lambda Mode | Quality | Proposal Recall | Routing Recall | Useful Selected | Useful Regions | Decoy | Cosine Distance | Alpha SD |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(sweep, key=lambda value: float(value["lambda_mode"])):
        sweep_lines.append(
            f"| {_fmt(row['lambda_mode'])} | {_fmt(row['candidate_mean_quality'])} | "
            f"{_fmt(row['proposal_recall'])} | {_fmt(row['routing_recall'])} | "
            f"{_fmt(row['useful_selected'])} | {_fmt(row['useful_region_count'])} | "
            f"{_fmt(row['candidate_decoy_rate'])} | {_fmt(row['direction_cosine_distance'])} | "
            f"{_fmt(row['alpha_sd'])} |"
        )
    sweep_table = "\n".join(sweep_lines)
    ablation_rows = {
        str(row["variant"]): row
        for row in primary_summary
        if row["variant"]
        in {"ungated_shared_ablation", "gated_candidate_specific_ablation"}
    }
    results_report = f"""# Mode-preservation results

This experiment changes only the generator reward. K remains 8, the Generation-3
policy remains 174 parameters, every main run receives 100,000 detached candidate
outcomes, and the saved pairwise router is never fitted. The lambda-zero policies,
candidates, refined states, and qualities reproduce the prior RL artifacts with
maximum difference zero; router parameter difference is also zero.

A literal shared reward broadcast would disappear under the existing within-set
group-relative advantage because the same constant is subtracted from every
candidate. The shared intervention therefore uses batch-relative REINFORCE on the
joint proposal-set log probability (normalized by K). Candidate-specific reward
uses the original within-set estimator. This credit-assignment requirement is
documented rather than hidden as a methodological change.

Validation selected `lambda_mode=0.1` from the predeclared grid. It tied baseline
proposal recall and won the documented tie-break on useful-region count. Test
statistics below are paired across 30 untouched seeds.

## Validation-only lambda sweep

These are three-seed, 25,000-outcome validation results, not test estimates.

{sweep_table}

## Paired main comparison

| Variant | Quality | Proposal Recall | Routing Recall | Useful Selected | Useful Regions | Decoy | Cosine Distance | Alpha SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lambda=0 | {_fmt(baseline["candidate_mean_quality"])} | {_fmt(baseline["proposal_recall"])} | {_fmt(baseline["routing_recall"])} | {_fmt(baseline["useful_selected"])} | {_fmt(baseline["useful_region_count"])} | {_fmt(baseline["candidate_decoy_rate"])} | {_fmt(baseline["direction_cosine_distance"])} | {_fmt(baseline["alpha_sd"])} |
| gated shared lambda=0.1 | {_fmt(selected["candidate_mean_quality"])} | {_fmt(selected["proposal_recall"])} | {_fmt(selected["routing_recall"])} | {_fmt(selected["useful_selected"])} | {_fmt(selected["useful_region_count"])} | {_fmt(selected["candidate_decoy_rate"])} | {_fmt(selected["direction_cosine_distance"])} | {_fmt(selected["alpha_sd"])} |

## Answers

1. **Proposal recall:** no resolved increase. The paired change is {effect("proposal_recall")}.
   Baseline recall was already {_fmt(baseline["proposal_recall"])}, leaving little IID headroom.
2. **Useful-region representation:** it increased by {effect("useful_region_count")};
   the fraction of sets with at least two useful regions changed by
   {effect("useful_region_count_2_plus")}.
3. **Geometric diversity:** cosine distance increased by
   {effect("direction_cosine_distance")} and effective directional rank by
   {effect("effective_directional_rank")}. Minimum pair distance remained
   unresolved ({effect("minimum_direction_distance")}), so some near-duplicate
   proposal modes remain.
4. **Functional diversity:** useful-region count and portfolio collapse improved,
   but proposal recall did not resolve. This is partial functional mode preservation,
   not proof that geometric spread alone is useful.
5. **Quality:** candidate quality changed by {effect("candidate_mean_quality")} and
   frozen-router selected quality by {effect("selected_quality")}; neither interval
   resolves a quality change.
6. **Decoys:** candidate decoy rate changed by {effect("candidate_decoy_rate")}.
   The intervention did not create garbage dispersion on IID tests.
7. **Useful-oracle ceiling:** it equals proposal recall by construction and changed
   by {effect("useful_oracle_success")}; the ceiling did not resolve upward.
8. **Frozen-router selection:** overall useful selection changed by
   {effect("useful_selected")} and routing recall by {effect("routing_recall")}.
   Generator portfolio diversity did not improve end-to-end useful selection with
   the frozen router.
9. **Cost:** mean training time rose from {_fmt(baseline["training_seconds"])} s to
   {_fmt(selected["training_seconds"])} s ({overhead:+.1f}%). Peak allocated memory
   changed by {memory_overhead:+.2f} MiB. The weighted mode term averaged
   {_fmt(selected["mode_reward_to_base_ratio"])} of base reward, so reward domination
   did not occur.

## Six-seed reward-allocation ablations

| Ablation | Quality | Proposal Recall | Useful Regions | Decoy | Cosine Distance | Effective Rank | Useful Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ungated shared | {_fmt(ablation_rows["ungated_shared_ablation"]["candidate_mean_quality"])} | {_fmt(ablation_rows["ungated_shared_ablation"]["proposal_recall"])} | {_fmt(ablation_rows["ungated_shared_ablation"]["useful_region_count"])} | {_fmt(ablation_rows["ungated_shared_ablation"]["candidate_decoy_rate"])} | {_fmt(ablation_rows["ungated_shared_ablation"]["direction_cosine_distance"])} | {_fmt(ablation_rows["ungated_shared_ablation"]["effective_directional_rank"])} | {_fmt(ablation_rows["ungated_shared_ablation"]["useful_selected"])} |
| Gated candidate-specific | {_fmt(ablation_rows["gated_candidate_specific_ablation"]["candidate_mean_quality"])} | {_fmt(ablation_rows["gated_candidate_specific_ablation"]["proposal_recall"])} | {_fmt(ablation_rows["gated_candidate_specific_ablation"]["useful_region_count"])} | {_fmt(ablation_rows["gated_candidate_specific_ablation"]["candidate_decoy_rate"])} | {_fmt(ablation_rows["gated_candidate_specific_ablation"]["direction_cosine_distance"])} | {_fmt(ablation_rows["gated_candidate_specific_ablation"]["effective_directional_rank"])} | {_fmt(ablation_rows["gated_candidate_specific_ablation"]["useful_selected"])} |

These ablations are exploratory (`n=6`). Relative to gated shared reward,
ungated proposal recall changed by
{ablation_effect("ungated_shared_ablation", SELECTED, "proposal_recall")} and
candidate decoy rate by
{ablation_effect("ungated_shared_ablation", SELECTED, "candidate_decoy_rate")}.
Candidate-specific credit produced much larger cosine distance but no resolved
effective-rank or proposal-recall improvement. In 2D, nearly antipodal proposals
can have large pairwise distance while remaining close to a rank-one axis; this is
the clearest cosmetic-diversity warning in the ablations.

The alpha mean increased by {effect("alpha_mean")} rather than collapsing toward
zero. Directional separation therefore was not hidden by radial compensation.
The narrow conclusion is that quality-gated repulsion increased the number of
useful modes represented, but did not establish a proposal-recall or end-to-end
routing improvement in this near-saturated IID setting.
"""

    specialization_report = f"""# Candidate-mode specialization

Candidate-region association is diagnostic only and never appears in training.
Useful region indices are compared within the fixed synthetic region numbering;
entropy and mutual information are calculated only from evaluation-time useful hits.

| Variant | I(Z;R) | Normalized I | Mean H(R|z) | Between-z Alpha SD | Direction Concentration | Max z Decoy Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lambda=0 | {_fmt(specialization_lookup[BASELINE]["candidate_region_mutual_information"])} | {_fmt(specialization_lookup[BASELINE]["candidate_region_normalized_mi"])} | {_fmt(specialization_lookup[BASELINE]["mean_candidate_region_entropy"])} | {_fmt(specialization_lookup[BASELINE]["between_candidate_alpha_sd"])} | {_fmt(specialization_lookup[BASELINE]["mean_direction_concentration"])} | {_fmt(specialization_lookup[BASELINE]["maximum_candidate_decoy_rate"])} |
| gated shared | {_fmt(specialization_lookup[SELECTED]["candidate_region_mutual_information"])} | {_fmt(specialization_lookup[SELECTED]["candidate_region_normalized_mi"])} | {_fmt(specialization_lookup[SELECTED]["mean_candidate_region_entropy"])} | {_fmt(specialization_lookup[SELECTED]["between_candidate_alpha_sd"])} | {_fmt(specialization_lookup[SELECTED]["mean_direction_concentration"])} | {_fmt(specialization_lookup[SELECTED]["maximum_candidate_decoy_rate"])} |

The paired candidate-region mutual-information change is
{float(specialization_pair_lookup["candidate_region_mutual_information"]["paired_difference"]):+.3f}
(95% CI {float(specialization_pair_lookup["candidate_region_mutual_information"]["ci95_low"]):+.3f}
to {float(specialization_pair_lookup["candidate_region_mutual_information"]["ci95_high"]):+.3f}).
Between-candidate alpha specialization changed by
{float(specialization_pair_lookup["between_candidate_alpha_sd"]["paired_difference"]):+.3f}
(95% CI {float(specialization_pair_lookup["between_candidate_alpha_sd"]["ci95_low"]):+.3f}
to {float(specialization_pair_lookup["between_candidate_alpha_sd"]["ci95_high"]):+.3f}).

## Answers

1. **Did z identities specialize?** Candidate/useful-region association increased;
   the mutual-information interval excludes zero. This supports modest categorical
   specialization, not deterministic one-code/one-region routing.
2. **Different useful-region preferences?** Yes, P(region|z_i) differs across
   candidate identities and the normalized association increased. The full
   candidate-by-region frequency table is saved rather than inferred from a plot.
3. **Directional, radial, or both?** Portfolio directions separated globally, but
   mean per-identity direction concentration changed by
   {float(specialization_pair_lookup["mean_direction_concentration"]["paired_difference"]):+.3f}
   (95% CI {float(specialization_pair_lookup["mean_direction_concentration"]["ci95_low"]):+.3f}
   to {float(specialization_pair_lookup["mean_direction_concentration"]["ci95_high"]):+.3f}).
   Between-identity alpha SD also remained unresolved, so identity specialization
   is mainly visible in reached regions rather than a simple fixed direction/radius.
4. **Did candidate-region association strengthen?** Yes on this diagnostic, while
   mean within-candidate region entropy did not resolve; specialization is partial.
5. **Dedicated decoy modes?** Maximum per-identity decoy tendency changed by
   {float(specialization_pair_lookup["maximum_candidate_decoy_rate"]["paired_difference"]):+.3f}
   (95% CI {float(specialization_pair_lookup["maximum_candidate_decoy_rate"]["ci95_low"]):+.3f}
   to {float(specialization_pair_lookup["maximum_candidate_decoy_rate"]["ci95_high"]):+.3f}).
   There is no resolved evidence that the intervention created dedicated decoy modes.
6. **Alpha by identity?** Candidate identities have different mean alphas, but the
   intervention did not resolve an increase in between-identity radial specialization.

`candidate_identity_summary.csv` provides P(region|z_i), quality, target/useful/
anchor/decoy frequency, alpha, and mean direction for each identity.
`candidate_goal_summary.csv` separates those diagnostics by goal, while
`alpha_by_category.csv` distinguishes target, useful non-target, anchor, decoy,
and other proposals. The heatmap and fixed-seed vector plots should be used with
the numerical association measures; a visually broad portfolio is not itself
evidence of useful specialization.
"""

    gen_lookup = {
        (str(row["variant"]), str(row["landscape_variant"])): row
        for row in generalization_summary
    }
    lines = [
        "# Mode-preservation generalization",
        "",
        "Each cell reports proposal recall / useful-region count / candidate decoy rate.",
        "",
        "| Landscape | lambda=0 | gated shared lambda=0.1 |",
        "| --- | ---: | ---: |",
    ]
    for landscape in landscapes:
        base_row = gen_lookup[(BASELINE, landscape)]
        mode_row = gen_lookup[(SELECTED, landscape)]
        lines.append(
            f"| {landscape} | {_fmt(base_row['proposal_recall'])} / "
            f"{_fmt(base_row['useful_region_count'])} / {_fmt(base_row['candidate_decoy_rate'])} | "
            f"{_fmt(mode_row['proposal_recall'])} / {_fmt(mode_row['useful_region_count'])} / "
            f"{_fmt(mode_row['candidate_decoy_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Proposal-recall point estimates rise on every held-out family, but none of the paired 95% intervals excludes zero. Useful-region count resolves upward on IID, translated, more-decoys, and unseen configurations.",
            "On new-goals landscapes, candidate decoy rate rises and candidate quality, router useful selection, routing recall, and selected quality fall with intervals excluding zero. Thus the mode intervention does not provide a uniformly safe distribution-shift improvement.",
            "Full paired intervals for every family and metric are in `generalization_paired.csv`.",
        ]
    )
    generalization_report = "\n".join(lines) + "\n"

    for name, text in [
        ("MODE_PRESERVATION_RESULTS.md", results_report),
        ("MODE_SPECIALIZATION.md", specialization_report),
        ("MODE_PRESERVATION_GENERALIZATION.md", generalization_report),
    ]:
        Path(name).write_text(text, encoding="utf-8")
        (main_dir / name).write_text(text, encoding="utf-8")
    aggregate = {
        "seed_count": 30,
        "selected_lambda_mode": 0.1,
        "primary_summary": primary_summary,
        "paired_selected_vs_baseline": primary_paired,
        "specialization_summary": specialization_summary,
    }
    (main_dir / "report_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
