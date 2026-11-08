"""Controlled paired study of Gen2 direction normalization only."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..config_normalization import NormalizationStudyConfig
from ..creative_operator_v2 import ConditionalMLPOperatorV2
from ..landscapes_v2 import V2LandscapeFamily
from ..normalization_diagnostics import (
    equivariance_diagnostics,
    evaluate_candidate_sets,
)
from ..training_v2 import fit_critic_v2, train_operator_v2
from ..visualization_normalization import (
    plot_alpha_normalization_comparison,
    plot_perturbation_distributions,
    plot_selector_comparison,
)
from .study_v2 import _make_starts, _training_starts, _write_csv

Array = np.ndarray

VARIANTS = ["gen2_unnormalized", "gen2_normalized"]
TABLE_METRICS = [
    "oracle_best_quality",
    "selected_quality",
    "critic_regret",
    "critic_top1_accuracy",
    "spearman_rank_correlation",
    "coverage",
    "decoy_rate",
    "useful_exploration_rate",
    "rotation_error",
    "translation_error",
]
PAIRED_METRICS = TABLE_METRICS + [
    "mean_quality",
    "actual_perturbation_magnitude",
    "raw_norm",
    "normalized_norm",
]


def _statistics(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    sd = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    ci = float(1.96 * sd / np.sqrt(len(data))) if len(data) > 1 else 0.0
    return {
        "mean": float(np.mean(data)),
        "sd": sd,
        "ci95": ci,
        "n": float(len(data)),
    }


def _experiment_table(
    primary_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for variant in VARIANTS:
        subset = [row for row in primary_rows if row["variant"] == variant]
        row: dict[str, object] = {"variant": variant}
        for metric in TABLE_METRICS:
            stats = _statistics([float(item[metric]) for item in subset])
            for name, value in stats.items():
                row[f"{metric}_{name}"] = value
        output.append(row)
    return output


def _paired_table(
    primary_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_key = {(int(row["seed"]), str(row["variant"])): row for row in primary_rows}
    seeds = sorted({int(row["seed"]) for row in primary_rows})
    output: list[dict[str, object]] = []
    for metric in PAIRED_METRICS:
        differences = [
            float(by_key[(seed, "gen2_normalized")][metric])
            - float(by_key[(seed, "gen2_unnormalized")][metric])
            for seed in seeds
        ]
        stats = _statistics(differences)
        mean = stats["mean"]
        ci = stats["ci95"]
        output.append(
            {
                "contrast": "normalized_minus_unnormalized",
                "metric": metric,
                "mean_difference": mean,
                "sd_difference": stats["sd"],
                "ci95": ci,
                "ci95_low": mean - ci,
                "ci95_high": mean + ci,
                "n": stats["n"],
                "ci_excludes_zero": bool(abs(mean) > ci),
            }
        )
    return output


def _selector_table(
    selector_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    metrics = [
        "selected_quality",
        "target_hit_rate",
        "useful_exploration_rate",
        "decoy_rate",
    ]
    output: list[dict[str, object]] = []
    for variant in VARIANTS:
        for selector in ["learned_critic", "oracle", "random"]:
            subset = [
                row
                for row in selector_rows
                if row["variant"] == variant and row["selector"] == selector
            ]
            result: dict[str, object] = {
                "variant": variant,
                "selector": selector,
            }
            for metric in metrics:
                stats = _statistics([float(row[metric]) for row in subset])
                for name, value in stats.items():
                    result[f"{metric}_{name}"] = value
            output.append(result)
    return output


def _format_triplet(row: Mapping[str, object], metric: str) -> str:
    return (
        f"{float(row[f'{metric}_mean']):.3f} ± "
        f"{float(row[f'{metric}_sd']):.3f} "
        f"(CI ±{float(row[f'{metric}_ci95']):.3f})"
    )


def _write_experiment_markdown(
    path: Path,
    table: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
    selector_table: Sequence[Mapping[str, object]],
) -> None:
    labels = {
        "gen2_unnormalized": "Gen2 unnormalized",
        "gen2_normalized": "Gen2 normalized",
    }
    lines = [
        "# Direction-normalization diagnostic tables",
        "",
        "Values are mean ± sample SD with the approximate 95% CI half-width in parentheses across independent seeds.",
        "Portfolio coverage, decoy rate, and useful exploration use all K candidates; selector outcomes are separated below.",
        "",
        "| Variant | Oracle Quality | Selected Quality | Critic Regret | Critic Top-1 | Rank Corr. | Coverage | Decoy Rate | Useful Exploration | Rotation Error | Translation Error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {labels[str(row['variant'])]} | "
            + " | ".join(_format_triplet(row, metric) for metric in TABLE_METRICS)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paired normalized minus unnormalized effects",
            "",
            "A paired interval excludes zero only when the last column is `yes`.",
            "",
            "| Metric | Mean difference | SD difference | 95% CI | Excludes zero? |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {str(row['metric']).replace('_', ' ')} | "
            f"{float(row['mean_difference']):.4f} | "
            f"{float(row['sd_difference']):.4f} | "
            f"[{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}] | "
            f"{'yes' if bool(row['ci_excludes_zero']) else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Selector baselines on identical candidates",
            "",
            "| Variant | Selector | Selected Quality | Target Hit | Useful Exploration | Decoy Rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in selector_table:
        lines.append(
            f"| {labels[str(row['variant'])]} | {str(row['selector']).replace('_', ' ')} | "
            f"{_format_triplet(row, 'selected_quality')} | "
            f"{_format_triplet(row, 'target_hit_rate')} | "
            f"{_format_triplet(row, 'useful_exploration_rate')} | "
            f"{_format_triplet(row, 'decoy_rate')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paired_lookup(
    paired: Sequence[Mapping[str, object]], metric: str
) -> Mapping[str, object]:
    return next(row for row in paired if row["metric"] == metric)


def _direction_statement(
    paired: Sequence[Mapping[str, object]],
    metric: str,
    *,
    higher_is_better: bool,
) -> str:
    row = _paired_lookup(paired, metric)
    difference = float(row["mean_difference"])
    ci = float(row["ci95"])
    if not bool(row["ci_excludes_zero"]):
        return f"not resolved (paired difference {difference:+.3f}, 95% CI ±{ci:.3f})"
    favorable = difference > 0.0 if higher_is_better else difference < 0.0
    return (
        "improved" if favorable else "worsened"
    ) + f" (paired difference {difference:+.3f}, 95% CI ±{ci:.3f})"


def _write_interpretation(
    path: Path,
    primary_rows: Sequence[Mapping[str, object]],
    alpha_rows: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
) -> None:
    by_variant = {
        variant: [row for row in primary_rows if row["variant"] == variant]
        for variant in VARIANTS
    }
    norm_regret = float(
        np.mean([float(row["critic_regret"]) for row in by_variant["gen2_normalized"]])
    )
    unnorm_regret = float(
        np.mean(
            [float(row["critic_regret"]) for row in by_variant["gen2_unnormalized"]]
        )
    )
    norm_oracle = float(
        np.mean(
            [float(row["oracle_best_quality"]) for row in by_variant["gen2_normalized"]]
        )
    )
    generator_shortfall = max(0.0, 1.0 - norm_oracle)
    unnorm_oracle = float(
        np.mean(
            [
                float(row["oracle_best_quality"])
                for row in by_variant["gen2_unnormalized"]
            ]
        )
    )
    unnorm_generator_shortfall = max(0.0, 1.0 - unnorm_oracle)
    norm_bottleneck = (
        "critic/selector" if norm_regret > generator_shortfall else "generator"
    )
    unnorm_bottleneck = (
        "critic/selector" if unnorm_regret > unnorm_generator_shortfall else "generator"
    )

    alpha_errors: dict[str, float] = {}
    for variant in VARIANTS:
        subset = [row for row in alpha_rows if row["variant"] == variant]
        alpha_errors[variant] = float(
            np.mean(
                [
                    abs(
                        float(row["actual_perturbation_magnitude"])
                        - float(row["alpha"])
                    )
                    for row in subset
                ]
            )
        )
    alpha_better = alpha_errors["gen2_normalized"] < alpha_errors["gen2_unnormalized"]

    useful_effect = _paired_lookup(paired, "useful_exploration_rate")
    coverage_effect = _paired_lookup(paired, "coverage")
    anchor_collapse = (
        bool(useful_effect["ci_excludes_zero"])
        and float(useful_effect["mean_difference"]) < 0.0
        and bool(coverage_effect["ci_excludes_zero"])
        and float(coverage_effect["mean_difference"]) < 0.0
        and np.mean(
            [
                float(row["useful_exploration_rate"])
                for row in by_variant["gen2_normalized"]
            ]
        )
        < 0.01
    )
    if anchor_collapse:
        verdict = (
            "harmful: it collapsed to exploitation rather than controlled divergence"
        )
        quality_context = (
            "The higher mean/selected quality reflects anchor retention rather than "
            "useful exploration."
        )
        normalized_gap_context = (
            "The near-zero normalized gap is degenerate because all candidates "
            "return to the anchor, not evidence of a better ranker."
        )
    else:
        directional_checks = [
            ("mean_quality", True),
            ("useful_exploration_rate", True),
            ("decoy_rate", False),
            ("coverage", True),
        ]
        favorable = 0
        harmful = 0
        for metric, higher_is_better in directional_checks:
            row = _paired_lookup(paired, metric)
            if not bool(row["ci_excludes_zero"]):
                continue
            direction_good = (
                float(row["mean_difference"]) > 0.0
                if higher_is_better
                else float(row["mean_difference"]) < 0.0
            )
            favorable += int(direction_good)
            harmful += int(not direction_good)
        if harmful > favorable:
            verdict = "harmful on balance"
        elif favorable > harmful:
            verdict = "beneficial on balance"
        else:
            verdict = "neutral or unresolved overall"
        quality_context = (
            "Quality is mixed and should be interpreted alongside useful "
            "exploration and decoy entry."
        )
        normalized_gap_context = (
            "The normalized candidate set contains useful alternatives, so this "
            "remaining gap reflects genuine selector headroom."
        )

    lines = [
        "# Automatic interpretation: direction normalization",
        "",
        "Statistical wording below is based on paired seed differences. A difference is called resolved only when its approximate 95% confidence interval excludes zero; this is not a multiple-comparison-adjusted significance test.",
        "",
        f"1. **Quality:** mean candidate quality {_direction_statement(paired, 'mean_quality', higher_is_better=True)}, selected quality {_direction_statement(paired, 'selected_quality', higher_is_better=True)}, and oracle-best quality {_direction_statement(paired, 'oracle_best_quality', higher_is_better=True)}. {quality_context}",
        f"2. **Useful exploration:** normalization {_direction_statement(paired, 'useful_exploration_rate', higher_is_better=True)}.",
        f"3. **Decoy rate:** normalization {_direction_statement(paired, 'decoy_rate', higher_is_better=False)}.",
        f"4. **Coverage:** normalization {_direction_statement(paired, 'coverage', higher_is_better=True)}.",
        f"5. **Alpha interpretation:** {'improved' if alpha_better else 'did not improve'}; mean |actual perturbation − alpha| was {alpha_errors['gen2_normalized']:.4f} normalized versus {alpha_errors['gen2_unnormalized']:.4f} unnormalized.",
        f"6. **Current bottleneck:** the normalized model is {norm_bottleneck}-limited under the regret-versus-oracle-shortfall diagnostic (regret {norm_regret:.4f}; oracle shortfall {generator_shortfall:.4f}). The unnormalized model is {unnorm_bottleneck}-limited (regret {unnorm_regret:.4f}; oracle shortfall {unnorm_generator_shortfall:.4f}).",
        f"7. **Oracle-versus-critic gap:** unnormalized mean regret was {unnorm_regret:.4f}; normalized regret was {norm_regret:.4f}. {normalized_gap_context}",
        f"8. **Coordinate consistency:** rotation error {_direction_statement(paired, 'rotation_error', higher_is_better=False)} and translation error {_direction_statement(paired, 'translation_error', higher_is_better=False)}.",
        f"9. **Overall:** normalization was **{verdict}** at the tested fixed-alpha values.",
        "",
        "The coordinate diagnostics measure function consistency only; no equivariance loss or transformed-data training was added.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_normalization_study(
    config: NormalizationStudyConfig, output: Path
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    base = config.base
    family = V2LandscapeFamily(base.landscape)
    primary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    candidate_set_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []

    for seed_index, seed in enumerate(base.seeds):
        print(
            f"[Normalization {seed_index + 1}/{len(base.seeds)}] seed {seed}",
            flush=True,
        )
        train_tasks = family.sample_tasks(
            geometries=base.train_geometries,
            seed=seed + 10_000,
            variant="iid",
            all_goals=True,
        )
        train_nested_starts = _make_starts(train_tasks, trials=1, seed=seed + 10_001)
        train_starts = _training_starts(train_nested_starts)
        test_tasks = family.sample_tasks(
            geometries=base.test_geometries,
            seed=seed + 20_000,
            variant="iid",
            all_goals=True,
        )
        test_starts = _make_starts(
            test_tasks, trials=base.trials_per_task, seed=seed + 20_001
        )

        operators = {
            variant: ConditionalMLPOperatorV2(
                base.landscape.dimension,
                base.training.candidates,
                base.training.latent_dimension,
                base.training.hidden_dimension,
                seed + 10_020,
                use_goal=True,
                normalize_direction=(variant == "gen2_normalized"),
            )
            for variant in VARIANTS
        }
        np.testing.assert_array_equal(
            operators["gen2_unnormalized"].get_parameters(),
            operators["gen2_normalized"].get_parameters(),
        )
        np.testing.assert_array_equal(
            operators["gen2_unnormalized"].codes,
            operators["gen2_normalized"].codes,
        )

        for variant in VARIANTS:
            operator = operators[variant]
            training_start = time.perf_counter()
            history = train_operator_v2(
                operator,
                train_tasks,
                train_starts,
                base.training,
                iterations=base.training.gen2_iterations,
                seed=seed + 10_021,
            )
            critic = fit_critic_v2(
                operator,
                train_tasks,
                train_starts,
                base.training,
                seed=seed + 10_022,
            )
            training_seconds = time.perf_counter() - training_start
            operator.save(
                model_dir / f"seed_{seed}_{variant}.npz",
                seed=seed,
                alpha=base.training.alpha,
            )
            critic.save(model_dir / f"seed_{seed}_{variant}_critic.npz")
            if seed_index == 0:
                history_rows.extend({"variant": variant, **row} for row in history)

            main_summary, records, set_records, selectors = evaluate_candidate_sets(
                variant=variant,
                operator=operator,
                critic=critic,
                tasks=test_tasks,
                starts=test_starts,
                config=base.training,
                seed=seed + 21_004,
                alpha=base.training.alpha,
                useful_threshold=base.landscape.useful_threshold,
                high_quality_threshold=base.landscape.high_quality_threshold,
            )
            equivariance = equivariance_diagnostics(
                operator,
                test_tasks,
                test_starts,
                seed=seed + 20_600,
                trials=config.equivariance_trials,
                translation_magnitude=config.translation_magnitude,
            )
            primary_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "training_wall_clock_seconds": training_seconds,
                    **main_summary,
                    **equivariance,
                }
            )
            candidate_rows.extend({"seed": seed, **row} for row in records)
            candidate_set_rows.extend({"seed": seed, **row} for row in set_records)
            selector_rows.extend({"seed": seed, **row} for row in selectors)

            for alpha_index, alpha in enumerate(config.alpha_values):
                if abs(alpha - base.training.alpha) < 1e-12:
                    alpha_summary = main_summary
                else:
                    alpha_summary, _, _, _ = evaluate_candidate_sets(
                        variant=variant,
                        operator=operator,
                        critic=critic,
                        tasks=test_tasks,
                        starts=test_starts,
                        config=base.training,
                        seed=seed + 22_000 + alpha_index,
                        alpha=alpha,
                        useful_threshold=base.landscape.useful_threshold,
                        high_quality_threshold=base.landscape.high_quality_threshold,
                    )
                alpha_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "alpha": alpha,
                        **alpha_summary,
                    }
                )

    experiment_table = _experiment_table(primary_rows)
    paired_table = _paired_table(primary_rows)
    selector_table = _selector_table(selector_rows)

    _write_csv(output / "primary_per_seed.csv", primary_rows)
    _write_csv(output / "candidate_diagnostics.csv", candidate_rows)
    _write_csv(output / "candidate_set_diagnostics.csv", candidate_set_rows)
    _write_csv(output / "selector_per_seed.csv", selector_rows)
    _write_csv(output / "alpha_sweep.csv", alpha_rows)
    _write_csv(output / "training_history_first_seed.csv", history_rows)
    _write_csv(output / "experiment_table.csv", experiment_table)
    _write_csv(output / "paired_differences.csv", paired_table)
    _write_csv(output / "selector_summary.csv", selector_table)
    _write_experiment_markdown(
        output / "EXPERIMENT_TABLE.md",
        experiment_table,
        paired_table,
        selector_table,
    )
    _write_interpretation(
        output / "INTERPRETATION.md", primary_rows, alpha_rows, paired_table
    )

    plot_alpha_normalization_comparison(
        alpha_rows, output / "alpha_sweep_normalized_vs_unnormalized.png"
    )
    plot_perturbation_distributions(
        candidate_rows, output / "perturbation_norm_distributions.png"
    )
    plot_selector_comparison(selector_rows, output / "selector_comparison.png")

    aggregates: dict[str, object] = {
        "seed_count": len(base.seeds),
        "experiment_table": experiment_table,
        "paired_differences": paired_table,
        "selector_summary": selector_table,
    }
    (output / "aggregate_metrics.json").write_text(
        json.dumps(aggregates, indent=2), encoding="utf-8"
    )
    return aggregates
