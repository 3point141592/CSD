"""Statistical comparison and reports for black-box RL versus preserved ES."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from creativity_toy.analysis_utils import statistics

from .study import read_csv, write_csv

METRICS = (
    "mean_quality",
    "selected_quality",
    "oracle_best_quality",
    "euclidean_diversity",
    "coverage",
    "collapse_rate",
    "useful_exploration_rate",
    "decoy_rate",
    "target_hit_rate",
    "selected_target_rate",
    "critic_regret",
    "mean_alpha",
    "alpha_sd",
)


def _rows(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv(path)


def _stat_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    label: str,
    fields: Iterable[str] = METRICS,
) -> dict[str, object]:
    output: dict[str, object] = {"training_regime": label}
    for field in fields:
        if not rows or not all(field in row for row in rows):
            continue
        values = [float(row[field]) for row in rows]
        result = statistics(values)
        output[field] = result["mean"]
        output[f"{field}_sd"] = result["sd"]
        output[f"{field}_ci95"] = result["ci95"]
    return output


def _es_candidate_budget(row: Mapping[str, object]) -> int:
    # Convert objective counts to candidate counts for a common ES/RL budget.
    # The factors retain 12/30 tasks per population/full call and eight candidates.
    return (
        int(row["population_objective_calls"]) * 12 * 8
        + int(row["full_objective_calls"]) * 30 * 8
    )


def _paired(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    *,
    left_name: str,
    right_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    left_by_seed = {int(row["seed"]): row for row in left}
    right_by_seed = {int(row["seed"]): row for row in right}
    seeds = sorted(set(left_by_seed) & set(right_by_seed))
    detail: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for metric in METRICS:
        differences = []
        for seed in seeds:
            left_value = float(left_by_seed[seed][metric])
            right_value = float(right_by_seed[seed][metric])
            difference = left_value - right_value
            differences.append(difference)
            detail.append(
                {
                    "seed": seed,
                    "metric": metric,
                    left_name: left_value,
                    right_name: right_value,
                    "difference_left_minus_right": difference,
                }
            )
        result = statistics(differences)
        summary.append(
            {
                "metric": metric,
                "left": left_name,
                "right": right_name,
                "paired_difference_mean": result["mean"],
                "paired_difference_sd": result["sd"],
                "paired_difference_ci95": result["ci95"],
                "ci95_low": result["mean"] - result["ci95"],
                "ci95_high": result["mean"] + result["ci95"],
                "paired_ci_excludes_zero": abs(result["mean"]) > result["ci95"],
                "n": int(result["n"]),
            }
        )
    return detail, summary


def _selector_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for mode in ["rl_adamw_stochastic", "rl_adamw_deterministic"]:
        for selector in ["learned_critic", "oracle", "random"]:
            selected = [
                row
                for row in rows
                if row["evaluation_mode"] == mode and row["selector"] == selector
            ]
            result: dict[str, object] = {
                "evaluation_mode": mode,
                "selector": selector,
            }
            for field in [
                "selected_quality",
                "target_hit_rate",
                "useful_exploration_rate",
                "decoy_rate",
            ]:
                stat = statistics([float(row[field]) for row in selected])
                result[field] = stat["mean"]
                result[f"{field}_sd"] = stat["sd"]
                result[f"{field}_ci95"] = stat["ci95"]
            output.append(result)
    return output


def _generalization_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    variants = sorted({str(row["landscape_variant"]) for row in rows})
    methods = ["gpu_es", "gpu_rl_stochastic", "gpu_rl_deterministic"]
    for variant in variants:
        for method in methods:
            selected = [
                row
                for row in rows
                if row["landscape_variant"] == variant and row["method"] == method
            ]
            result: dict[str, object] = {
                "landscape_variant": variant,
                "method": method,
            }
            for metric in [
                "mean_quality",
                "coverage",
                "useful_exploration_rate",
                "decoy_rate",
                "target_hit_rate",
                "mean_alpha",
                "alpha_sd",
            ]:
                stat = statistics([float(row[metric]) for row in selected])
                result[metric] = stat["mean"]
                result[f"{metric}_sd"] = stat["sd"]
                result[f"{metric}_ci95"] = stat["ci95"]
            output.append(result)
    return output


def _generalization_paired(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    variants = sorted({str(row["landscape_variant"]) for row in rows})
    for variant in variants:
        es = {
            int(row["seed"]): row
            for row in rows
            if row["landscape_variant"] == variant and row["method"] == "gpu_es"
        }
        rl = {
            int(row["seed"]): row
            for row in rows
            if row["landscape_variant"] == variant
            and row["method"] == "gpu_rl_stochastic"
        }
        for metric in [
            "mean_quality",
            "coverage",
            "useful_exploration_rate",
            "decoy_rate",
            "target_hit_rate",
        ]:
            differences = [
                float(rl[seed][metric]) - float(es[seed][metric])
                for seed in sorted(set(es) & set(rl))
            ]
            stat = statistics(differences)
            output.append(
                {
                    "landscape_variant": variant,
                    "metric": metric,
                    "paired_rl_minus_es_mean": stat["mean"],
                    "paired_difference_sd": stat["sd"],
                    "paired_difference_ci95": stat["ci95"],
                    "ci95_low": stat["mean"] - stat["ci95"],
                    "ci95_high": stat["mean"] + stat["ci95"],
                    "paired_ci_excludes_zero": abs(stat["mean"]) > stat["ci95"],
                    "n": int(stat["n"]),
                }
            )
    return output


def _alpha_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    dimensions = {
        "category": "category",
        "candidate_identity": "candidate_id",
        "goal": "goal_index",
    }
    for mode in ["rl_adamw_stochastic", "rl_adamw_deterministic"]:
        mode_rows = [row for row in rows if row["evaluation_mode"] == mode]
        for dimension, key in dimensions.items():
            values = sorted({str(row[key]) for row in mode_rows})
            for value in values:
                selected = [row for row in mode_rows if str(row[key]) == value]
                seed_means: list[float] = []
                for seed in sorted({int(row["seed"]) for row in selected}):
                    seed_values = [
                        float(row["candidate_alpha"])
                        for row in selected
                        if int(row["seed"]) == seed
                    ]
                    seed_means.append(float(np.mean(seed_values)))
                stat = statistics(seed_means)
                pooled = np.asarray(
                    [float(row["candidate_alpha"]) for row in selected],
                    dtype=np.float64,
                )
                output.append(
                    {
                        "evaluation_mode": mode,
                        "grouping": dimension,
                        "group": value,
                        "alpha_mean": stat["mean"],
                        "alpha_sd_across_seed_means": stat["sd"],
                        "alpha_ci95_across_seed_means": stat["ci95"],
                        "alpha_pooled_median": float(np.median(pooled)),
                        "candidate_count": len(pooled),
                        "seed_count": int(stat["n"]),
                    }
                )
    return output


def _endpoint_diagnostics(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    endpoints: list[Mapping[str, object]] = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        endpoints.append(
            max(seed_rows, key=lambda row: float(row["candidate_evaluations"]))
        )
    output: list[dict[str, object]] = []
    for field in [
        "mean_reward",
        "reward_std",
        "mean_advantage_magnitude",
        "advantage_group_mean_abs",
        "policy_entropy",
        "direction_std",
        "alpha_action_std",
        "gradient_norm_before_clip",
        "gradient_norm_after_clip",
        "approximate_kl",
        "candidate_direction_cosine",
        "candidate_spatial_diversity",
        "within_set_alpha_sd",
        "fraction_nearly_deterministic",
    ]:
        stat = statistics([float(row[field]) for row in endpoints])
        output.append(
            {
                "diagnostic": field,
                "mean": stat["mean"],
                "sd": stat["sd"],
                "ci95": stat["ci95"],
                "n": int(stat["n"]),
            }
        )
    return output


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _plot_regimes(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    metrics = ["mean_quality", "coverage", "useful_exploration_rate", "decoy_rate"]
    labels = [str(row["training_regime"]) for row in rows]
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for axis, metric in zip(axes, metrics):
        values = [float(row[metric]) for row in rows]
        errors = [float(row[f"{metric}_ci95"]) for row in rows]
        axis.bar(np.arange(len(rows)), values, yerr=errors, capsize=3)
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_paired(
    es_rows: Sequence[Mapping[str, object]],
    rl_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    es = {int(row["seed"]): row for row in es_rows}
    rl = {int(row["seed"]): row for row in rl_rows}
    metrics = ["mean_quality", "coverage", "useful_exploration_rate", "decoy_rate"]
    figure, axes = plt.subplots(1, 4, figsize=(14.5, 3.5))
    for axis, metric in zip(axes, metrics):
        x = np.asarray([float(es[seed][metric]) for seed in sorted(es)])
        y = np.asarray([float(rl[seed][metric]) for seed in sorted(es)])
        axis.scatter(x, y, alpha=0.75)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        axis.plot([low, high], [low, high], "--", color="black", linewidth=1)
        correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0
        axis.set_title(f"{metric.replace('_', ' ')}\nr={correlation:.2f}")
        axis.set_xlabel("GPU ES")
        axis.set_ylabel("GPU RL")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_budget_curve(
    checkpoints: Sequence[Mapping[str, object]],
    es_summary: Mapping[str, object],
    es_budget: float,
    path: Path,
) -> None:
    metrics = [
        ("stochastic_mean_quality", "mean_quality", "quality"),
        ("stochastic_coverage", "coverage", "coverage"),
        ("stochastic_useful_exploration_rate", "useful_exploration_rate", "useful"),
        ("stochastic_decoy_rate", "decoy_rate", "decoy"),
    ]
    budgets = sorted({int(float(row["requested_budget"])) for row in checkpoints})
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for axis, (rl_metric, es_metric, title) in zip(axes, metrics):
        means, errors = [], []
        for budget in budgets:
            values = [
                float(row[rl_metric])
                for row in checkpoints
                if int(float(row["requested_budget"])) == budget
            ]
            stat = statistics(values)
            means.append(stat["mean"])
            errors.append(stat["ci95"])
        axis.errorbar(budgets, means, yerr=errors, marker="o", label="RL")
        axis.errorbar(
            [es_budget],
            [float(es_summary[es_metric])],
            yerr=[float(es_summary[f"{es_metric}_ci95"])],
            marker="s",
            linestyle="none",
            label="ES final",
        )
        axis.set_title(title)
        axis.set_xlabel("candidate scalar outcomes")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_alpha(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    stochastic = [
        row for row in rows if row["evaluation_mode"] == "rl_adamw_stochastic"
    ]
    alpha = np.asarray([float(row["candidate_alpha"]) for row in stochastic])
    quality = np.asarray([float(row["quality"]) for row in stochastic])
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(alpha, bins=35, alpha=0.8)
    axes[0].set_title("learned alpha")
    axes[0].set_xlabel("alpha")
    axes[1].hexbin(alpha, quality, gridsize=35, mincnt=1)
    axes[1].set_title("alpha vs candidate quality")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("Q")
    categories = ["anchor", "useful_non_target", "target_hit", "decoy", "other"]
    box_values = [
        [
            float(row["candidate_alpha"])
            for row in stochastic
            if row["category"] == category
        ]
        for category in categories
    ]
    axes[2].boxplot(box_values, tick_labels=categories, showfliers=False)
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_title("alpha by outcome category")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def generate_training_regime_comparison(
    *,
    output: Path = Path("results/training_regime_comparison"),
    rl_dir: Path = Path("results/rl_adamw_main"),
    gpu_es_dir: Path = Path("results/gen3_gpu"),
    cpu_es_dir: Path = Path("results/gen3_learned_alpha"),
    baseline_dir: Path = Path("results/v2_main"),
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    rl_all = _rows(rl_dir / "primary_per_seed.csv")
    rl = [row for row in rl_all if row["evaluation_mode"] == "rl_adamw_stochastic"]
    rl_deterministic = [
        row for row in rl_all if row["evaluation_mode"] == "rl_adamw_deterministic"
    ]
    gpu_all = _rows(gpu_es_dir / "primary_per_seed.csv")
    gpu_es = [
        row for row in gpu_all if row["variant"] == "gen3_contextual_linear_alpha"
    ]
    cpu_all = _rows(cpu_es_dir / "primary_per_seed.csv")
    cpu_es = [
        row for row in cpu_all if row["variant"] == "gen3_contextual_linear_alpha"
    ]

    primary = [
        _stat_rows(cpu_es, label="CPU ES"),
        _stat_rows(gpu_es, label="GPU ES"),
        _stat_rows(rl, label="GPU RL stochastic"),
        _stat_rows(rl_deterministic, label="GPU RL mean action"),
    ]
    gpu_budget_by_seed = {int(row["seed"]): _es_candidate_budget(row) for row in gpu_es}
    router_candidate_evaluations = 30 * (8 + 16)
    for row in primary:
        regime = str(row["training_regime"])
        if "RL" in regime:
            source = rl
            policy_budgets = [int(item["candidate_evaluations"]) for item in source]
            q_budgets = [int(item["q_equivalent_evaluations"]) for item in source]
            runtimes = [
                float(item["training_seconds"]) + float(item["critic_fit_seconds"])
                for item in source
            ]
            training_only = [float(item["training_seconds"]) for item in source]
            optimizer_steps = [int(item["optimizer_steps"]) for item in source]
            peak_allocated = max(int(item["peak_allocated_bytes"]) for item in source)
            parameter_count = 174
        else:
            source = gpu_es if regime == "GPU ES" else cpu_es
            policy_budgets = [gpu_budget_by_seed[int(item["seed"])] for item in source]
            q_budgets = [budget * 11 for budget in policy_budgets]
            runtimes = [float(item["training_wall_clock_seconds"]) for item in source]
            training_only = (
                [float(item["training_only_seconds"]) for item in source]
                if regime == "GPU ES"
                else runtimes
            )
            optimizer_steps = [65] * len(source)
            peak_allocated = 36_580_352 if regime == "GPU ES" else 0
            parameter_count = 172
        total_budgets = [
            budget + router_candidate_evaluations for budget in policy_budgets
        ]
        total_q = [budget + router_candidate_evaluations * 10 for budget in q_budgets]
        row.update(
            {
                "feedback": "detached scalar reward",
                "policy_candidate_outcomes": float(np.mean(policy_budgets)),
                "total_candidate_outcomes_including_router": float(
                    np.mean(total_budgets)
                ),
                "q_equivalent_calls_including_router": float(np.mean(total_q)),
                "runtime_seconds": float(np.mean(runtimes)),
                "runtime_seconds_sd": float(np.std(runtimes, ddof=1)),
                "candidate_outcomes_per_second": float(
                    np.mean(np.asarray(policy_budgets) / np.asarray(training_only))
                ),
                "q_equivalent_calls_per_second": float(
                    np.mean(np.asarray(q_budgets) / np.asarray(training_only))
                ),
                "evaluation_candidate_outcomes_per_second": float(
                    np.mean(
                        [
                            8.0 / float(item["wall_clock_seconds_per_trial"])
                            for item in source
                        ]
                    )
                ),
                "optimizer_steps": float(np.mean(optimizer_steps)),
                "optimizer_steps_per_second": float(
                    np.mean(np.asarray(optimizer_steps) / np.asarray(training_only))
                ),
                "parameter_count": parameter_count,
                "peak_gpu_allocated_bytes": peak_allocated,
            }
        )

    paired_detail, paired_summary = _paired(
        rl, gpu_es, left_name="gpu_rl", right_name="gpu_es"
    )
    selectors = _selector_summary(_rows(rl_dir / "selector_per_seed.csv"))
    generalization_rows = _rows(rl_dir / "generalization_metrics.csv")
    generalization = _generalization_summary(generalization_rows)
    generalization_paired = _generalization_paired(generalization_rows)
    alpha = _alpha_summary(_rows(rl_dir / "candidate_diagnostics.csv"))
    endpoints = _endpoint_diagnostics(_rows(rl_dir / "training_history.csv"))

    baselines = _rows(baseline_dir / "primary_metrics.csv")
    baseline_summary = []
    for method in [
        "baseline",
        "random",
        "pure_divergence",
        "generation1",
        "generation2",
    ]:
        selected = [row for row in baselines if row["method"] == method]
        baseline_summary.append(_stat_rows(selected, label=method))
    baseline_summary.append(_stat_rows(rl, label="gpu_rl_stochastic"))

    speed: list[dict[str, object]] = []
    for regime, source in [
        ("CPU ES", cpu_es),
        ("GPU ES", gpu_es),
        ("GPU RL stochastic", rl),
    ]:
        if regime == "GPU RL stochastic":
            elapsed = np.asarray(
                [
                    float(row["training_seconds"]) + float(row["critic_fit_seconds"])
                    for row in source
                ]
            )
        else:
            elapsed = np.asarray(
                [float(row["training_wall_clock_seconds"]) for row in source]
            )
        regime_summary = next(
            row for row in primary if row["training_regime"] == regime
        )
        speed.append(
            {
                "training_regime": regime,
                "timing_repetitions": len(elapsed),
                "timing_scope": "per-seed policy training plus router fit",
                "mean_seconds": float(np.mean(elapsed)),
                "median_seconds": float(np.median(elapsed)),
                "sd_seconds": float(np.std(elapsed, ddof=1)),
                "minimum_seconds": float(np.min(elapsed)),
                "sum_seconds": float(np.sum(elapsed)),
                "policy_candidate_outcomes": regime_summary[
                    "policy_candidate_outcomes"
                ],
                "candidate_outcomes_per_second": regime_summary[
                    "candidate_outcomes_per_second"
                ],
                "q_equivalent_calls_per_second": regime_summary[
                    "q_equivalent_calls_per_second"
                ],
                "evaluation_candidate_outcomes_per_second": regime_summary[
                    "evaluation_candidate_outcomes_per_second"
                ],
                "optimizer_steps_per_second": regime_summary[
                    "optimizer_steps_per_second"
                ],
                "peak_gpu_allocated_bytes": regime_summary["peak_gpu_allocated_bytes"],
            }
        )

    write_csv(output / "primary_comparison.csv", primary)
    write_csv(output / "paired_rl_minus_es_per_seed.csv", paired_detail)
    write_csv(output / "paired_rl_minus_es_summary.csv", paired_summary)
    write_csv(output / "selector_comparison.csv", selectors)
    write_csv(output / "generalization_summary.csv", generalization)
    write_csv(output / "generalization_paired_rl_minus_es.csv", generalization_paired)
    write_csv(output / "alpha_analysis.csv", alpha)
    write_csv(output / "rl_endpoint_diagnostics.csv", endpoints)
    write_csv(output / "legacy_baseline_comparison.csv", baseline_summary)
    write_csv(output / "speed_benchmark.csv", speed)

    _plot_regimes(primary, output / "training_regime_metrics.png")
    _plot_paired(gpu_es, rl, output / "gpu_es_vs_rl_per_seed.png")
    es_summary = next(row for row in primary if row["training_regime"] == "GPU ES")
    _plot_budget_curve(
        _rows(rl_dir / "checkpoint_metrics.csv"),
        es_summary,
        float(np.mean(list(gpu_budget_by_seed.values()))),
        output / "performance_vs_candidate_outcomes.png",
    )
    _plot_alpha(
        _rows(rl_dir / "candidate_diagnostics.csv"),
        output / "rl_alpha_diagnostics.png",
    )

    paired_lookup = {str(row["metric"]): row for row in paired_summary}
    selector_lookup = {
        (str(row["evaluation_mode"]), str(row["selector"])): row for row in selectors
    }
    learned = selector_lookup[("rl_adamw_stochastic", "learned_critic")]
    oracle = selector_lookup[("rl_adamw_stochastic", "oracle")]
    random = selector_lookup[("rl_adamw_stochastic", "random")]
    rl_primary = next(
        row for row in primary if row["training_regime"] == "GPU RL stochastic"
    )
    baseline_lookup = {str(row["training_regime"]): row for row in baseline_summary}
    alpha_lookup = {
        (str(row["evaluation_mode"]), str(row["grouping"]), str(row["group"])): row
        for row in alpha
    }
    ood_variants = {
        str(row["landscape_variant"])
        for row in generalization
        if row["landscape_variant"] != "iid"
    }
    ood: dict[str, dict[str, float]] = {}
    for method in ["gpu_es", "gpu_rl_stochastic"]:
        selected = [
            row
            for row in generalization
            if row["method"] == method and row["landscape_variant"] in ood_variants
        ]
        ood[method] = {
            metric: float(np.mean([float(row[metric]) for row in selected]))
            for metric in [
                "mean_quality",
                "coverage",
                "useful_exploration_rate",
                "decoy_rate",
            ]
        }

    main_table = _markdown_table(
        [
            "Training",
            "Feedback",
            "Quality",
            "Coverage",
            "Useful",
            "Decoy",
            "Target",
            "Alpha SD",
            "Candidate outcomes",
            "Runtime (s)",
        ],
        [
            [
                row["training_regime"],
                "scalar",
                _fmt(float(row["mean_quality"])),
                _fmt(float(row["coverage"])),
                _fmt(float(row["useful_exploration_rate"])),
                _fmt(float(row["decoy_rate"])),
                _fmt(float(row["target_hit_rate"])),
                _fmt(float(row["alpha_sd"])),
                f"{float(row['total_candidate_outcomes_including_router']):.0f}",
                _fmt(float(row["runtime_seconds"])),
            ]
            for row in primary[:3]
        ]
        + [
            ["Oracle AdamW", "gradient Q", "not run", "—", "—", "—", "—", "—", "—", "—"]
        ],
    )
    (output / "PRIMARY_COMPARISON.md").write_text(
        "# Primary training-regime comparison\n\n" + main_table + "\n",
        encoding="utf-8",
    )

    def paired_sentence(metric: str) -> str:
        row = paired_lookup[metric]
        return (
            f"{float(row['paired_difference_mean']):+.3f} "
            f"(95% paired CI {float(row['ci95_low']):+.3f} to "
            f"{float(row['ci95_high']):+.3f})"
        )

    results_report = f"""# RL results

This is a 30-seed result. Hyperparameters (`lr=3e-3`, `weight_decay=0`) were
chosen by mean stochastic reward on held-out training-family validation
landscapes; the test landscapes were not used for selection.

## Main finding

The stochastic policy learned high-quality, low-decoy proposals from detached
scalar feedback: quality {_fmt(float(rl_primary["mean_quality"]))} ±
{_fmt(float(rl_primary["mean_quality_ci95"]))} (95% CI), useful exploration
{_fmt(float(rl_primary["useful_exploration_rate"]))}, coverage
{_fmt(float(rl_primary["coverage"]))}, and decoy rate
{_fmt(float(rl_primary["decoy_rate"]))}. It did not preserve a broad portfolio:
collapse was {_fmt(float(rl_primary["collapse_rate"]))}. Thus it learned useful
divergence, but with substantial mode concentration.

Learned magnitude was not constant: mean alpha was
{_fmt(float(rl_primary["mean_alpha"]))} and pooled within-evaluation alpha SD was
{_fmt(float(rl_primary["alpha_sd"]))}. The alpha tables and plot show candidate,
goal, and outcome-category stratifications. Stochastic and mean-action test
metrics were close, indicating that the learned means—not only sampling
noise—accounted for the final behavior.

## Generator versus router

Using exactly the same stochastic candidate sets, oracle selection reached
quality {_fmt(float(oracle["selected_quality"]))} and useful/target hit rate
{_fmt(float(oracle["useful_exploration_rate"]))}; random selection reached
quality {_fmt(float(random["selected_quality"]))} and useful rate
{_fmt(float(random["useful_exploration_rate"]))}. The learned router reached
quality {_fmt(float(learned["selected_quality"]))} but useful/target rate only
{_fmt(float(learned["useful_exploration_rate"]))}. The mean oracle-minus-router
quality gap was {_fmt(float(oracle["selected_quality"]) - float(learned["selected_quality"]))}.
The generator is producing alternatives, while the existing router strongly
prefers the high-quality anchor. Routing is therefore the larger bottleneck for
selected useful exploration, even though quality regret is numerically modest.

## Collapse and goal conditioning

Target coverage changes with the goal because target identity is goal-defined,
and the policy attains a candidate-set target-hit rate of
{_fmt(float(rl_primary["target_hit_rate"]))}. This supports goal-responsive
proposal generation in the synthetic setting, but the high collapse rate and
low learned-router target rate prevent a stronger claim. Region labels were
never used in the reward.

The experiment establishes only that a stochastic goal-conditioned proposal
policy can learn constrained exploration from black-box outcomes. It does not
equate reinforcement learning with creativity.
"""
    results_report = f"""# RL results

This is a 30-seed result. Hyperparameters (`lr=3e-3`, `weight_decay=0`) were
selected by mean stochastic reward on held-out training-family validation
landscapes. Test landscapes were not used for selection.

## Main finding

The stochastic policy learned high-quality, low-decoy proposals from detached
scalar feedback: quality {_fmt(float(rl_primary["mean_quality"]))} +/-
{_fmt(float(rl_primary["mean_quality_ci95"]))} (95% CI), useful exploration
{_fmt(float(rl_primary["useful_exploration_rate"]))}, coverage
{_fmt(float(rl_primary["coverage"]))}, and decoy rate
{_fmt(float(rl_primary["decoy_rate"]))}. It did not preserve a broad portfolio:
collapse was {_fmt(float(rl_primary["collapse_rate"]))}. It therefore learned
useful divergence, but with substantial mode concentration.

Learned magnitude was not constant: mean alpha was
{_fmt(float(rl_primary["mean_alpha"]))} and pooled alpha SD was
{_fmt(float(rl_primary["alpha_sd"]))}. Evaluation-only categories show mean
alpha {_fmt(float(alpha_lookup[("rl_adamw_stochastic", "category", "anchor")]["alpha_mean"]))}
for anchors, {_fmt(float(alpha_lookup[("rl_adamw_stochastic", "category", "useful_non_target")]["alpha_mean"]))}
for useful non-targets, {_fmt(float(alpha_lookup[("rl_adamw_stochastic", "category", "target_hit")]["alpha_mean"]))}
for target hits, and {_fmt(float(alpha_lookup[("rl_adamw_stochastic", "category", "decoy")]["alpha_mean"]))}
for decoys. Goal-level means range from
{min(float(alpha_lookup[("rl_adamw_stochastic", "goal", str(goal))]["alpha_mean"]) for goal in range(3)):.3f}
to {max(float(alpha_lookup[("rl_adamw_stochastic", "goal", str(goal))]["alpha_mean"]) for goal in range(3)):.3f}.
These associations are diagnostic, not causal. Stochastic and mean-action test
metrics were close, indicating that learned means, not only sampling noise,
accounted for final behavior.

## Generator versus router

Using exactly the same stochastic candidate sets, oracle selection reached
quality {_fmt(float(oracle["selected_quality"]))} and useful/target hit rate
{_fmt(float(oracle["useful_exploration_rate"]))}; random selection reached
quality {_fmt(float(random["selected_quality"]))} and useful rate
{_fmt(float(random["useful_exploration_rate"]))}. The learned router reached
quality {_fmt(float(learned["selected_quality"]))} but useful/target rate only
{_fmt(float(learned["useful_exploration_rate"]))}. The mean oracle-minus-router
quality gap was {_fmt(float(oracle["selected_quality"]) - float(learned["selected_quality"]))}.
The generator is producing alternatives, while the existing router strongly
prefers the high-quality anchor. Routing is the larger bottleneck for selected
useful exploration, even though quality regret is numerically modest.

## Legacy baselines

RL did not beat the exploitation baseline on raw quality
({_fmt(float(rl_primary["mean_quality"]))} versus
{_fmt(float(baseline_lookup["baseline"]["mean_quality"]))}), but the baseline has
zero coverage and useful exploration. RL had much higher quality and much lower
decoy rate than random ({_fmt(float(baseline_lookup["random"]["mean_quality"]))},
{_fmt(float(baseline_lookup["random"]["decoy_rate"]))}) and pure divergence
({_fmt(float(baseline_lookup["pure_divergence"]["mean_quality"]))},
{_fmt(float(baseline_lookup["pure_divergence"]["decoy_rate"]))}). Random achieved
higher coverage but only {_fmt(float(baseline_lookup["random"]["useful_exploration_rate"]))}
useful exploration. The prior deterministic Generation-2 operator retained
higher coverage and useful exploration than RL, so RL is not the best portfolio
generator in this environment.

## Collapse and goal conditioning

The policy attained candidate-set target-hit rate
{_fmt(float(rl_primary["target_hit_rate"]))}. This is consistent with
goal-responsive proposal generation but is not a causal goal-conditioning
ablation. High collapse and low learned-router target selection prevent a
stronger claim. Region labels were never used in the reward.

The experiment establishes only that a stochastic goal-conditioned proposal
policy can learn constrained exploration from black-box outcomes. It does not
equate reinforcement learning with creativity.
"""
    Path("RL_RESULTS.md").write_text(results_report, encoding="utf-8")

    comparison_report = f"""# Training-regime comparison

{main_table}

Candidate outcomes include the 720 candidates used to fit the unchanged router.
The separate `q_equivalent_calls_including_router` column in the CSV expands the
unchanged refinement recurrence. RL used about 100,720 total candidate outcomes;
GPU ES used about {float(next(row for row in primary if row["training_regime"] == "GPU ES")["total_candidate_outcomes_including_router"]):.0f},
so the black-box budgets are close but not exactly equal.

## Paired RL minus GPU ES

- Quality: {paired_sentence("mean_quality")}.
- Coverage: {paired_sentence("coverage")}.
- Useful exploration: {paired_sentence("useful_exploration_rate")}.
- Decoy rate: {paired_sentence("decoy_rate")}.
- Alpha SD: {paired_sentence("alpha_sd")}.

A paired interval that excludes zero is evidence of a reproducible difference
under these seeds, not a universal claim. RL traded lower decoy incidence for
lower useful exploration and greater collapse; it did not dominate ES on the
quality-diversity tradeoff.

## Efficiency

Recorded per-seed training plus router time was
{_fmt(float(next(row for row in primary if row["training_regime"] == "CPU ES")["runtime_seconds"]))} s for CPU ES,
{_fmt(float(next(row for row in primary if row["training_regime"] == "GPU ES")["runtime_seconds"]))} s for GPU ES, and
{_fmt(float(rl_primary["runtime_seconds"]))} s for GPU RL. RL executes many more
sequential AdamW steps (1,044 versus 65 ES iterations), so this implementation
is slower despite its smaller policy-level outcome budget. Peak allocated VRAM
was {float(rl_primary["peak_gpu_allocated_bytes"]) / 2**20:.1f} MiB for RL and
34.9 MiB in the recorded GPU ES benchmark. GPU utilization was not sampled, so
no utilization percentage is claimed.

## Generalization and optimizer dependence

`generalization_summary.csv` applies the unchanged IID/rotated/translated/width,
peak-count, decoy-count, new-goal, and unseen protocols with frozen routers.
The table shows that conclusions depend on optimizer choice quantitatively:
RL produces fewer decoys but more concentrated portfolios. It does not overturn
the architectural conclusion that useful alternatives can be generated, nor
does it establish that RL is the superior optimizer.

The optional differentiable-oracle AdamW control was deliberately not run. It
is not part of the primary black-box question and would introduce privileged
`dQ/dh` feedback.
"""
    comparison_report = f"""# Training-regime comparison

{main_table.replace(chr(0xFFFD), "-")}

Candidate outcomes include 720 candidates used to fit the unchanged router.
RL used 100,720 total candidate outcomes; GPU ES used about
{float(next(row for row in primary if row["training_regime"] == "GPU ES")["total_candidate_outcomes_including_router"]):.0f}.
The black-box budgets are close, not identical. The CSV also reports internal
Q-equivalent calls induced by the unchanged refinement recurrence.

## Paired RL minus GPU ES

- Quality: {paired_sentence("mean_quality")}.
- Coverage: {paired_sentence("coverage")}.
- Useful exploration: {paired_sentence("useful_exploration_rate")}.
- Decoy rate: {paired_sentence("decoy_rate")}.
- Alpha SD: {paired_sentence("alpha_sd")}.

A paired interval excluding zero is evidence of a reproducible difference for
these seeds, not a universal claim. RL traded lower decoy incidence for lower
useful exploration; it did not dominate ES on the quality-diversity tradeoff.
The alpha-SD interval includes zero, so this experiment does not establish that
either optimizer learns more meaningful magnitude variation.

## Efficiency

Recorded mean per-seed training plus router time was
{_fmt(float(next(row for row in primary if row["training_regime"] == "CPU ES")["runtime_seconds"]))} s for CPU ES,
{_fmt(float(next(row for row in primary if row["training_regime"] == "GPU ES")["runtime_seconds"]))} s for GPU ES, and
{_fmt(float(rl_primary["runtime_seconds"]))} s for GPU RL. RL executes 1,044
sequential AdamW steps versus 65 ES iterations and is slower despite a smaller
policy-level outcome budget. Peak allocated VRAM was
{float(rl_primary["peak_gpu_allocated_bytes"]) / 2**20:.1f} MiB for RL and 34.9 MiB
in the recorded GPU ES benchmark. Policy-training candidate throughput was
{float(rl_primary["candidate_outcomes_per_second"]):.0f}/s for RL versus
{float(next(row for row in primary if row["training_regime"] == "GPU ES")["candidate_outcomes_per_second"]):.0f}/s
for GPU ES; Q-equivalent and held-out evaluation throughputs are in
`speed_benchmark.csv`. GPU utilization was not sampled.

## Generalization and optimizer dependence

Across seven OOD families, unweighted per-family means were quality
{ood["gpu_rl_stochastic"]["mean_quality"]:.3f} versus {ood["gpu_es"]["mean_quality"]:.3f},
coverage {ood["gpu_rl_stochastic"]["coverage"]:.3f} versus {ood["gpu_es"]["coverage"]:.3f},
useful exploration {ood["gpu_rl_stochastic"]["useful_exploration_rate"]:.3f}
versus {ood["gpu_es"]["useful_exploration_rate"]:.3f}, and decoy rate
{ood["gpu_rl_stochastic"]["decoy_rate"]:.3f} versus {ood["gpu_es"]["decoy_rate"]:.3f}
for RL versus ES. Per-variant paired intervals are in
`generalization_paired_rl_minus_es.csv`. RL generalizes better on quality and
decoy avoidance in this aggregate, while coverage and useful exploration are
mixed across transformations. RL does not establish optimizer superiority.

The optional differentiable-oracle AdamW control was not run. It is outside the
primary black-box question and would introduce privileged `dQ/dh` feedback.

The performance-versus-budget plot contains all preregistered RL checkpoints
and the final preserved ES point. Intermediate held-out portfolio checkpoints
were absent from the original ES artifacts and were not reconstructed with a
different ES implementation merely to draw a smoother curve.
"""
    Path("TRAINING_REGIME_COMPARISON.md").write_text(
        comparison_report, encoding="utf-8"
    )
    aggregate = {
        "seeds": 30,
        "primary": primary,
        "paired_rl_minus_gpu_es": paired_summary,
        "black_box_contract": True,
        "oracle_gradient_baseline_run": False,
        "test_used_for_hyperparameter_selection": False,
    }
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
