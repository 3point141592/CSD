"""End-to-end multi-seed experimental study."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..baselines import evaluate_method
from ..config import ExperimentConfig
from ..landscapes import GaussianIslandLandscape, make_landscapes, starting_point
from ..metrics import aggregate_seeds, paired_seed_differences, summarize_records
from ..operators import CreativeOperator
from ..training import fit_critic, train_operator
from ..visualization import (
    plot_2d_discoveries,
    plot_alpha_metric,
    plot_generalization,
    plot_low_rank,
    plot_quality_diversity_frontier,
    plot_training_curves,
)

Array = np.ndarray


def _family_args(config: ExperimentConfig) -> dict[str, object]:
    landscape = config.landscape
    return {
        "dimension": landscape.dimension,
        "intrinsic_dimension": landscape.intrinsic_dimension,
        "family_seed": landscape.family_seed,
        "max_family_peaks": landscape.max_family_peaks,
        "n_peaks": landscape.train_peaks,
        "radius": landscape.peak_radius,
        "width": landscape.peak_width,
    }


def _make_starts(
    landscapes: Sequence[GaussianIslandLandscape],
    *,
    trials: int,
    seed: int,
    noise: float,
) -> list[list[Array]]:
    rng = np.random.default_rng(seed)
    return [
        [starting_point(landscape, rng, noise) for _ in range(trials)]
        for landscape in landscapes
    ]


def _flatten_training_starts(starts: Sequence[Sequence[Array]]) -> list[Array]:
    return [landscape_starts[0] for landscape_starts in starts]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and not isinstance(row[key], np.ndarray):
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _averaged_rows(
    rows: Sequence[Mapping[str, object]], group_key: str
) -> list[dict[str, object]]:
    values = sorted(
        {row[group_key] for row in rows},
        key=lambda value: (
            float(value) if isinstance(value, (int, float)) else str(value)
        ),
    )
    result: list[dict[str, object]] = []
    excluded = {"seed", group_key}
    for value in values:
        subset = [row for row in rows if row[group_key] == value]
        output: dict[str, object] = {group_key: value}
        for key, candidate in subset[0].items():
            if key in excluded:
                continue
            if isinstance(candidate, (int, float, np.number)):
                output[key] = float(np.mean([float(row[key]) for row in subset]))
            else:
                output[key] = candidate
        result.append(output)
    return result


def _evaluate_and_summarize(
    *,
    method: str,
    landscapes: Sequence[GaussianIslandLandscape],
    starts: Sequence[Sequence[Array]],
    operator: CreativeOperator,
    alpha: float,
    config: ExperimentConfig,
    seed: int,
    critic: object = None,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    records = evaluate_method(
        method=method,
        landscapes=landscapes,
        starts=starts,
        operator=operator,
        alpha=alpha,
        config=config.training,
        seed=seed,
        critic=critic,  # type: ignore[arg-type]
    )
    return summarize_records(records, config.quality_threshold), records


def _run_2d_demo(
    config: ExperimentConfig,
    output: Path,
    seed: int,
) -> None:
    landscape_cfg = replace(
        config.landscape,
        dimension=2,
        intrinsic_dimension=2,
        family_seed=config.landscape.family_seed + 5,
    )
    training_cfg = replace(
        config.training,
        single_iterations=max(25, config.training.single_iterations // 2),
        multi_iterations=max(35, config.training.multi_iterations // 2),
        population=max(12, config.training.population),
        batch_size=min(8, config.train_landscapes),
    )
    landscapes = make_landscapes(
        count=max(8, config.train_landscapes // 2),
        seed=seed + 800,
        dimension=2,
        intrinsic_dimension=2,
        family_seed=landscape_cfg.family_seed,
        max_family_peaks=landscape_cfg.max_family_peaks,
        n_peaks=landscape_cfg.train_peaks,
        radius=landscape_cfg.peak_radius,
        width=landscape_cfg.peak_width,
    )
    starts_nested = _make_starts(
        landscapes,
        trials=1,
        seed=seed + 801,
        noise=landscape_cfg.initial_noise,
    )
    starts = _flatten_training_starts(starts_nested)
    single = CreativeOperator(2, heads=1, seed=seed + 802)
    train_operator(
        single,
        landscapes,
        starts,
        training_cfg,
        iterations=training_cfg.single_iterations,
        seed=seed + 803,
    )
    operator = CreativeOperator(2, heads=config.heads, seed=seed + 804)
    operator.initialize_first_head(single)
    train_operator(
        operator,
        landscapes,
        starts,
        training_cfg,
        iterations=training_cfg.multi_iterations,
        seed=seed + 805,
    )

    demo_landscape = landscapes[0]
    demo_starts = _make_starts(
        [demo_landscape],
        trials=8,
        seed=seed + 806,
        noise=landscape_cfg.initial_noise,
    )
    method_records: dict[str, Sequence[Mapping[str, object]]] = {}
    for index, method in enumerate(
        ["baseline", "random", "pure_divergence", "creative"]
    ):
        method_records[method] = evaluate_method(
            method=method,
            landscapes=[demo_landscape],
            starts=demo_starts,
            operator=operator,
            alpha=1.0,
            config=training_cfg,
            seed=seed + 900 + index,
        )
    plot_2d_discoveries(
        demo_landscape,
        method_records,
        output / "landscape_2d_discoveries.png",
    )


def _format_mean_sd(rows: Sequence[Mapping[str, object]], metric: str) -> str:
    values = np.array([float(row[metric]) for row in rows])
    sd = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{np.mean(values):.3f} ± {sd:.3f}"


def _write_report(
    path: Path,
    comparison_rows: Sequence[Mapping[str, object]],
    generalization_rows: Sequence[Mapping[str, object]],
    low_rank_rows: Sequence[Mapping[str, object]],
    effects: Mapping[str, object],
) -> None:
    methods = ["baseline", "random", "pure_divergence", "creative"]
    lines = [
        "# Experimental report",
        "",
        "This report is generated from all configured seeds; no seed was selected or discarded.",
        "The prespecified comparison point is alpha = 1.0 (the training coefficient).",
        "",
        "## Main budget-matched comparison (mean ± sample SD across seeds)",
        "",
        "| method | quality | Euclidean diversity | coverage | collapse | perturbation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        subset = [row for row in comparison_rows if row["method"] == method]
        lines.append(
            f"| {method} | {_format_mean_sd(subset, 'mean_quality')} | "
            f"{_format_mean_sd(subset, 'euclidean_diversity')} | "
            f"{_format_mean_sd(subset, 'coverage')} | "
            f"{_format_mean_sd(subset, 'collapse_rate')} | "
            f"{_format_mean_sd(subset, 'perturbation_magnitude')} |"
        )
    lines.extend(
        [
            "",
            "All four main methods receive the same number of candidate slots and the same fixed refinement steps. Random and pure-divergence perturbations are matched to the learned operator's RMS perturbation budget per head.",
            "Diversity is calculated only over endpoints passing the quality threshold.",
            "",
            "## Paired seed effects",
            "",
            "```json",
            json.dumps(effects, indent=2),
            "```",
            "",
            "## Generalization",
            "",
            "| variant | quality | diversity | coverage |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant in sorted({str(row["variant"]) for row in generalization_rows}):
        subset = [row for row in generalization_rows if row["variant"] == variant]
        lines.append(
            f"| {variant} | {_format_mean_sd(subset, 'mean_quality')} | "
            f"{_format_mean_sd(subset, 'euclidean_diversity')} | "
            f"{_format_mean_sd(subset, 'coverage')} |"
        )
    lines.extend(
        [
            "",
            "`shifted` tests the lack of translation equivariance; `rotated` changes the latent embedding; `non_gaussian` changes peak shape; `more_peaks` doubles the number of valid islands while keeping the head budget fixed.",
            "",
            "## Parameter efficiency",
            "",
            "| parameterization | parameters | quality | diversity | coverage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for rank in sorted({int(row["rank"]) for row in low_rank_rows}):
        subset = [row for row in low_rank_rows if int(row["rank"]) == rank]
        label = "full" if rank < 0 else f"rank {rank}"
        lines.append(
            f"| {label} | {int(np.mean([float(row['parameters']) for row in subset]))} | "
            f"{_format_mean_sd(subset, 'mean_quality')} | "
            f"{_format_mean_sd(subset, 'euclidean_diversity')} | "
            f"{_format_mean_sd(subset, 'coverage')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "Passing the toy benchmark is evidence only that a linear operator can learn reusable exploration directions in a structured family. Failure on shifted or rotated families is direct evidence against treating this context-free matrix as a general creativity mechanism. The learned critic is saved and evaluated as a secondary ablation, but head averaging is not used for the main coverage claim because averaging distinct modes can land in a low-quality valley.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(config: ExperimentConfig, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    sweep_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    single_rows: list[dict[str, object]] = []
    generalization_rows: list[dict[str, object]] = []
    low_rank_rows: list[dict[str, object]] = []
    critic_rows: list[dict[str, object]] = []
    saved_histories: dict[str, Sequence[Mapping[str, float]]] = {}
    family_args = _family_args(config)

    for seed_index, seed in enumerate(config.seeds):
        print(
            f"[{seed_index + 1}/{len(config.seeds)}] training/evaluating seed {seed}",
            flush=True,
        )
        train_landscapes = make_landscapes(
            count=config.train_landscapes,
            seed=seed + 1_000,
            **family_args,
        )
        training_starts_nested = _make_starts(
            train_landscapes,
            trials=1,
            seed=seed + 1_001,
            noise=config.landscape.initial_noise,
        )
        training_starts = _flatten_training_starts(training_starts_nested)

        single = CreativeOperator(
            config.landscape.dimension, heads=1, seed=seed + 1_010
        )
        single_history = train_operator(
            single,
            train_landscapes,
            training_starts,
            config.training,
            iterations=config.training.single_iterations,
            seed=seed + 1_011,
        )
        single.save(model_dir / f"seed_{seed}_single_full.npz", seed=seed)

        operator = CreativeOperator(
            config.landscape.dimension,
            heads=config.heads,
            seed=seed + 1_020,
        )
        operator.initialize_first_head(single)
        full_history = train_operator(
            operator,
            train_landscapes,
            training_starts,
            config.training,
            iterations=config.training.multi_iterations,
            seed=seed + 1_021,
        )
        operator.save(model_dir / f"seed_{seed}_multi_full.npz", seed=seed)
        critic = fit_critic(
            operator, train_landscapes, training_starts, config.training
        )
        np.savez_compressed(
            model_dir / f"seed_{seed}_critic.npz",
            coefficients=critic.coefficients,
            temperature=critic.temperature,
        )
        if seed_index == 0:
            saved_histories["single full"] = single_history
            saved_histories["multi full"] = full_history

        test_landscapes = make_landscapes(
            count=config.test_landscapes,
            seed=seed + 2_000,
            **family_args,
        )
        test_starts = _make_starts(
            test_landscapes,
            trials=config.trials_per_landscape,
            seed=seed + 2_001,
            noise=config.landscape.initial_noise,
        )

        single_summary, _ = _evaluate_and_summarize(
            method="creative",
            landscapes=test_landscapes,
            starts=test_starts,
            operator=single,
            alpha=config.training.alpha_train,
            config=config,
            seed=seed + 2_050,
        )
        single_rows.append(
            {"seed": seed, "method": "single_creative", **single_summary}
        )

        for alpha_index, alpha in enumerate(config.alpha_values):
            for method_index, method in enumerate(
                ["baseline", "random", "pure_divergence", "creative"]
            ):
                summary, _ = _evaluate_and_summarize(
                    method=method,
                    landscapes=test_landscapes,
                    starts=test_starts,
                    operator=operator,
                    alpha=alpha,
                    config=config,
                    seed=seed + 3_000 + 100 * alpha_index + method_index,
                )
                row: dict[str, object] = {
                    "seed": seed,
                    "method": method,
                    "alpha": alpha,
                    **summary,
                }
                sweep_rows.append(row)
                if alpha == config.training.alpha_train:
                    comparison_rows.append(row)

        for method_index, method in enumerate(["critic_selected", "critic_mixture"]):
            summary, _ = _evaluate_and_summarize(
                method=method,
                landscapes=test_landscapes,
                starts=test_starts,
                operator=operator,
                alpha=config.training.alpha_train,
                config=config,
                seed=seed + 4_000 + method_index,
                critic=critic,
            )
            critic_rows.append({"seed": seed, "method": method, **summary})

        for variant_index, variant in enumerate(
            ["iid", "more_peaks", "width_shift", "shifted", "rotated", "non_gaussian"]
        ):
            variant_landscapes = make_landscapes(
                count=config.test_landscapes,
                seed=seed + 5_000 + variant_index,
                variant=variant,
                **family_args,
            )
            variant_starts = _make_starts(
                variant_landscapes,
                trials=config.trials_per_landscape,
                seed=seed + 5_100 + variant_index,
                noise=config.landscape.initial_noise,
            )
            summary, _ = _evaluate_and_summarize(
                method="creative",
                landscapes=variant_landscapes,
                starts=variant_starts,
                operator=operator,
                alpha=config.training.alpha_train,
                config=config,
                seed=seed + 5_200 + variant_index,
            )
            generalization_rows.append({"seed": seed, "variant": variant, **summary})

        full_summary = [
            row
            for row in comparison_rows
            if row["seed"] == seed and row["method"] == "creative"
        ][0]
        low_rank_rows.append(
            {
                "seed": seed,
                "rank": -1,
                "parameters": operator.parameter_count,
                **{
                    key: value
                    for key, value in full_summary.items()
                    if key not in {"seed", "method", "alpha"}
                },
            }
        )

        if config.run_low_rank:
            for rank_index, rank in enumerate(config.low_ranks):
                low_rank = CreativeOperator(
                    config.landscape.dimension,
                    heads=config.heads,
                    rank=rank,
                    seed=seed + 6_000 + rank_index,
                )
                rank_history = train_operator(
                    low_rank,
                    train_landscapes,
                    training_starts,
                    config.training,
                    iterations=config.training.multi_iterations,
                    seed=seed + 6_100 + rank_index,
                )
                low_rank.save(
                    model_dir / f"seed_{seed}_multi_rank_{rank}.npz", seed=seed
                )
                if seed_index == 0:
                    saved_histories[f"rank {rank}"] = rank_history
                summary, _ = _evaluate_and_summarize(
                    method="creative",
                    landscapes=test_landscapes,
                    starts=test_starts,
                    operator=low_rank,
                    alpha=config.training.alpha_train,
                    config=config,
                    seed=seed + 6_200 + rank_index,
                )
                low_rank_rows.append(
                    {
                        "seed": seed,
                        "rank": rank,
                        "parameters": low_rank.parameter_count,
                        **summary,
                    }
                )

    _write_csv(output / "alpha_sweep_metrics.csv", sweep_rows)
    _write_csv(output / "main_comparison_metrics.csv", comparison_rows)
    _write_csv(output / "generalization_metrics.csv", generalization_rows)
    _write_csv(output / "low_rank_metrics.csv", low_rank_rows)
    _write_csv(output / "critic_metrics.csv", critic_rows)

    _write_csv(output / "single_head_metrics.csv", single_rows)
    history_rows: list[dict[str, object]] = []
    for model_name, history in saved_histories.items():
        history_rows.extend({"model": model_name, **row} for row in history)
    _write_csv(output / "training_history_first_seed.csv", history_rows)

    comparison_for_effect = [
        row
        for row in comparison_rows
        if float(row.get("alpha", 1.0)) == config.training.alpha_train
    ]
    effect_metrics = [
        "mean_quality",
        "euclidean_diversity",
        "coverage",
        "collapse_rate",
        "anchor_escape_rate",
    ]
    effects_vs_random = {
        metric: paired_seed_differences(
            comparison_for_effect, "creative", "random", metric
        )
        for metric in effect_metrics
    }
    effects_vs_pure = {
        metric: paired_seed_differences(
            comparison_for_effect, "creative", "pure_divergence", metric
        )
        for metric in effect_metrics
    }
    effects = {
        "creative_minus_random": effects_vs_random,
        "creative_minus_pure_divergence": effects_vs_pure,
    }
    aggregates: dict[str, object] = {
        "main": {
            method: aggregate_seeds(
                [row for row in comparison_rows if row["method"] == method]
            )
            for method in ["baseline", "random", "pure_divergence", "creative"]
        },
        "paired_creative_minus_random": effects_vs_random,
        "paired_creative_minus_pure_divergence": effects_vs_pure,
        "critic": {
            method: aggregate_seeds(
                [row for row in critic_rows if row["method"] == method]
            )
            for method in ["critic_selected", "critic_mixture"]
        },
    }
    (output / "aggregate_metrics.json").write_text(
        json.dumps(aggregates, indent=2), encoding="utf-8"
    )

    plot_training_curves(saved_histories, output / "training_curves.png")
    plot_alpha_metric(
        sweep_rows, "mean_quality", "mean quality", output / "alpha_vs_quality.png"
    )
    plot_alpha_metric(
        sweep_rows,
        "euclidean_diversity",
        "high-quality Euclidean diversity",
        output / "alpha_vs_diversity.png",
    )
    plot_alpha_metric(
        sweep_rows, "coverage", "solution coverage", output / "alpha_vs_coverage.png"
    )
    plot_alpha_metric(
        sweep_rows,
        "perturbation_magnitude",
        "raw perturbation magnitude",
        output / "alpha_vs_perturbation.png",
    )
    plot_quality_diversity_frontier(
        sweep_rows, output / "quality_diversity_frontier.png"
    )
    plot_low_rank(
        _averaged_rows(low_rank_rows, "rank"), output / "low_rank_comparison.png"
    )
    plot_generalization(
        _averaged_rows(generalization_rows, "variant"),
        output / "generalization.png",
    )
    if config.run_2d_demo:
        _run_2d_demo(config, output, config.seeds[0])

    _write_report(
        output / "REPORT.md",
        comparison_rows,
        generalization_rows,
        low_rank_rows,
        effects,
    )
    return aggregates
