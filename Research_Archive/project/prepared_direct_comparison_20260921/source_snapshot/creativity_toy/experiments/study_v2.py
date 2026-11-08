"""End-to-end Generation-2 study orchestration."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..baselines_v2 import evaluate_method_v2
from ..config_v2 import ExperimentV2Config
from ..creative_operator_v2 import (
    BaseCreativeOperatorV2,
    ConditionalLinearOperatorV2,
    ConditionalMLPOperatorV2,
    Generation1OperatorV2,
)
from ..landscapes_v2 import V2LandscapeFamily, V2Task, starting_state_v2
from ..metrics import aggregate_seeds, paired_seed_differences
from ..metrics_v2 import summarize_v2_records
from ..training_v2 import fit_critic_v2, train_operator_v2
from ..visualization_v2 import (
    plot_goal_conditioning_v2,
    plot_grouped_study_v2,
    plot_method_geometry_v2,
    plot_metric_bars_v2,
    plot_quality_displacement_v2,
    plot_quality_diversity_v2,
    plot_training_v2,
)

Array = np.ndarray


def _make_starts(
    tasks: Sequence[V2Task],
    *,
    trials: int,
    seed: int,
) -> list[list[Array]]:
    rng = np.random.default_rng(seed)
    return [[starting_state_v2(task, rng) for _ in range(trials)] for task in tasks]


def _training_starts(starts: Sequence[Sequence[Array]]) -> list[Array]:
    return [task_starts[0] for task_starts in starts]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in keys and not isinstance(value, np.ndarray):
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(key, "")
                    for key in keys
                    if not isinstance(row.get(key), np.ndarray)
                }
            )


def _scalar_endpoint(row: Mapping[str, object], seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        **{
            key: value
            for key, value in row.items()
            if not isinstance(value, np.ndarray)
        },
    }


def _summarize(
    records: Sequence[Mapping[str, object]], config: ExperimentV2Config
) -> dict[str, float]:
    return summarize_v2_records(
        records,
        useful_threshold=config.landscape.useful_threshold,
        high_quality_threshold=config.landscape.high_quality_threshold,
        diversity_scale=config.training.diversity_scale,
    )


def _evaluate(
    *,
    method: str,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[Array]],
    config: ExperimentV2Config,
    seed: int,
    candidates: int,
    operator: BaseCreativeOperatorV2 | None = None,
    critic: object = None,
    alpha: float | None = None,
    trajectories: bool = False,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    records = evaluate_method_v2(
        method=method,
        tasks=tasks,
        starts=starts,
        config=config.training,
        seed=seed,
        candidates=candidates,
        operator=operator,
        critic=critic,  # type: ignore[arg-type]
        alpha=alpha,
        record_trajectories=trajectories,
    )
    return _summarize(records, config), records


def _format_mean_sd(rows: Sequence[Mapping[str, object]], metric: str) -> str:
    values = np.array([float(row[metric]) for row in rows])
    sd = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{np.mean(values):.3f} ± {sd:.3f}"


def _paired_effects(
    rows: Sequence[Mapping[str, object]],
    method_a: str,
    method_b: str,
) -> dict[str, dict[str, float]]:
    metrics = [
        "mean_quality",
        "selected_quality",
        "euclidean_diversity",
        "strict_high_quality_diversity",
        "coverage",
        "decoy_rate",
        "useful_exploration_rate",
        "collapse_rate",
        "target_hit_rate",
        "selected_target_rate",
        "quality_diversity_score",
    ]
    return {
        metric: paired_seed_differences(rows, method_a, method_b, metric)
        for metric in metrics
    }


def _write_report(
    path: Path,
    primary: Sequence[Mapping[str, object]],
    generalization: Sequence[Mapping[str, object]],
    ablations: Sequence[Mapping[str, object]],
    effects: Mapping[str, object],
) -> None:
    methods = [
        "baseline",
        "random",
        "pure_divergence",
        "generation1",
        "generation2",
    ]
    lines = [
        "# Generation-2 experimental report",
        "",
        "All configured seeds are included. Region labels were used only for evaluation, never operator training.",
        "",
        "## Primary equal-budget comparison (mean ± sample SD across seeds)",
        "",
        "| method | quality | selected quality | useful diversity | strict diversity | coverage | decoy rate | useful exploration | target hit | target selection | collapse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        subset = [row for row in primary if row["method"] == method]
        lines.append(
            f"| {method} | {_format_mean_sd(subset, 'mean_quality')} | "
            f"{_format_mean_sd(subset, 'selected_quality')} | "
            f"{_format_mean_sd(subset, 'euclidean_diversity')} | "
            f"{_format_mean_sd(subset, 'strict_high_quality_diversity')} | "
            f"{_format_mean_sd(subset, 'coverage')} | "
            f"{_format_mean_sd(subset, 'decoy_rate')} | "
            f"{_format_mean_sd(subset, 'useful_exploration_rate')} | "
            f"{_format_mean_sd(subset, 'target_hit_rate')} | "
            f"{_format_mean_sd(subset, 'selected_target_rate')} | "
            f"{_format_mean_sd(subset, 'collapse_rate')} |"
        )
    lines.extend(
        [
            "",
            "Every primary method receives the same candidate count and refinement steps. The critic is shared across methods for selection; portfolio metrics include every candidate.",
            "",
            "## Paired seed effects",
            "",
            "```json",
            json.dumps(effects, indent=2),
            "```",
            "",
            "## Generalization (Generation 2)",
            "",
            "| variant | quality | coverage | decoy rate | target selection |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant in sorted({str(row["variant"]) for row in generalization}):
        subset = [
            row
            for row in generalization
            if row["variant"] == variant and row["method"] == "generation2"
        ]
        lines.append(
            f"| {variant} | {_format_mean_sd(subset, 'mean_quality')} | "
            f"{_format_mean_sd(subset, 'coverage')} | "
            f"{_format_mean_sd(subset, 'decoy_rate')} | "
            f"{_format_mean_sd(subset, 'selected_target_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| ablation | selected quality | coverage | decoy rate | target selection | candidate budget |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in sorted({str(row["ablation"]) for row in ablations}):
        subset = [row for row in ablations if row["ablation"] == name]
        lines.append(
            f"| {name} | {_format_mean_sd(subset, 'selected_quality')} | "
            f"{_format_mean_sd(subset, 'coverage')} | "
            f"{_format_mean_sd(subset, 'decoy_rate')} | "
            f"{_format_mean_sd(subset, 'selected_target_rate')} | "
            f"{_format_mean_sd(subset, 'candidate_budget_per_trial')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "A conditional operator is supported only if it improves useful exploration and/or goal selection while reducing decoy entry at comparable quality and test-time budget. Diversity or distance alone is not treated as evidence of creativity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study_v2(config: ExperimentV2Config, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    family = V2LandscapeFamily(config.landscape)
    primary_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    generalization_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    histories: dict[str, Sequence[Mapping[str, float]]] = {}
    representative_task: V2Task | None = None
    representative_records: dict[str, Sequence[Mapping[str, object]]] = {}
    goal_records: list[Mapping[str, object]] = []

    for seed_index, seed in enumerate(config.seeds):
        print(
            f"[V2 {seed_index + 1}/{len(config.seeds)}] seed {seed}",
            flush=True,
        )
        train_tasks = family.sample_tasks(
            geometries=config.train_geometries,
            seed=seed + 10_000,
            variant="iid",
            all_goals=True,
        )
        train_starts_nested = _make_starts(train_tasks, trials=1, seed=seed + 10_001)
        train_starts = _training_starts(train_starts_nested)

        generation1 = Generation1OperatorV2(
            config.landscape.dimension,
            config.training.candidates,
            seed + 10_010,
        )
        gen1_start_time = time.perf_counter()
        gen1_history = train_operator_v2(
            generation1,
            train_tasks,
            train_starts,
            config.training,
            iterations=config.training.gen1_iterations,
            seed=seed + 10_011,
        )
        gen1_training_seconds = time.perf_counter() - gen1_start_time
        generation1.save(model_dir / f"seed_{seed}_generation1.npz", seed=seed)

        generation2 = ConditionalMLPOperatorV2(
            config.landscape.dimension,
            config.training.candidates,
            config.training.latent_dimension,
            config.training.hidden_dimension,
            seed + 10_020,
        )
        gen2_start_time = time.perf_counter()
        gen2_history = train_operator_v2(
            generation2,
            train_tasks,
            train_starts,
            config.training,
            iterations=config.training.gen2_iterations,
            seed=seed + 10_021,
        )
        critic = fit_critic_v2(
            generation2,
            train_tasks,
            train_starts,
            config.training,
            seed=seed + 10_022,
        )
        gen2_training_seconds = time.perf_counter() - gen2_start_time
        generation2.save(model_dir / f"seed_{seed}_generation2.npz", seed=seed)
        critic.save(model_dir / f"seed_{seed}_critic_v2.npz")
        if seed_index == 0:
            histories["generation 1"] = gen1_history
            histories["generation 2"] = gen2_history

        test_tasks = family.sample_tasks(
            geometries=config.test_geometries,
            seed=seed + 20_000,
            variant="iid",
            all_goals=True,
        )
        test_starts = _make_starts(
            test_tasks,
            trials=config.trials_per_task,
            seed=seed + 20_001,
        )
        method_specs = [
            ("baseline", None, 0.0),
            ("random", None, 0.0),
            ("pure_divergence", None, 0.0),
            ("generation1", generation1, gen1_training_seconds),
            ("generation2", generation2, gen2_training_seconds),
        ]
        for method_index, (method, operator, training_seconds) in enumerate(
            method_specs
        ):
            summary, records = _evaluate(
                method=method,
                tasks=test_tasks,
                starts=test_starts,
                config=config,
                seed=seed + 21_000 + method_index,
                candidates=config.training.candidates,
                operator=operator,
                critic=critic,
                trajectories=seed_index == 0,
            )
            row: dict[str, object] = {
                "seed": seed,
                "method": method,
                "training_wall_clock_seconds": float(training_seconds),
                **summary,
            }
            primary_rows.append(row)
            endpoint_rows.extend(_scalar_endpoint(record, seed) for record in records)
            if method == "generation2":
                goal_records.extend(records)
            if seed_index == 0:
                representative_task = test_tasks[0]
                representative_records[method] = [
                    record for record in records if int(record["task_id"]) == 0
                ]

        for alpha_index, alpha in enumerate(config.alpha_values):
            summary, _ = _evaluate(
                method="generation2",
                tasks=test_tasks,
                starts=test_starts,
                config=config,
                seed=seed + 22_000 + alpha_index,
                candidates=config.training.candidates,
                operator=generation2,
                critic=critic,
                alpha=alpha,
            )
            alpha_rows.append(
                {"seed": seed, "method": "generation2", "alpha": alpha, **summary}
            )

        if config.run_generalization:
            variants = [
                "iid",
                "rotated",
                "translated",
                "width_shift",
                "more_peaks",
                "more_decoys",
                "new_goals",
                "unseen",
            ]
            for variant_index, variant in enumerate(variants):
                variant_tasks = family.sample_tasks(
                    geometries=config.test_geometries,
                    seed=seed + 30_000 + variant_index,
                    variant=variant,
                    all_goals=True,
                )
                variant_starts = _make_starts(
                    variant_tasks,
                    trials=config.trials_per_task,
                    seed=seed + 30_100 + variant_index,
                )
                for method_index, (method, operator) in enumerate(
                    [
                        ("pure_divergence", None),
                        ("generation1", generation1),
                        ("generation2", generation2),
                    ]
                ):
                    summary, _ = _evaluate(
                        method=method,
                        tasks=variant_tasks,
                        starts=variant_starts,
                        config=config,
                        seed=seed + 30_500 + 20 * variant_index + method_index,
                        candidates=config.training.candidates,
                        operator=operator,
                        critic=critic,
                    )
                    generalization_rows.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "method": method,
                            **summary,
                        }
                    )

        if config.run_ablations and seed_index < config.ablation_seeds:
            main_summary = next(
                row
                for row in primary_rows
                if row["seed"] == seed and row["method"] == "generation2"
            )
            ablation_rows.append(
                {
                    "seed": seed,
                    "ablation": f"K={config.training.candidates}",
                    **{
                        key: value
                        for key, value in main_summary.items()
                        if key not in {"seed", "method", "training_wall_clock_seconds"}
                    },
                }
            )
            no_critic_summary, _ = _evaluate(
                method="generation2_no_critic",
                tasks=test_tasks,
                starts=test_starts,
                config=config,
                seed=seed + 40_001,
                candidates=config.training.candidates,
                operator=generation2,
                critic=None,
            )
            ablation_rows.append(
                {"seed": seed, "ablation": "no_critic", **no_critic_summary}
            )

            special_specs = [
                ("no_diversity", True, False, config.training.candidates),
                ("no_goal", False, False, config.training.candidates),
                ("linear_conditional", True, True, config.training.candidates),
            ]
            for special_index, (
                name,
                use_goal,
                is_linear,
                candidate_count,
            ) in enumerate(special_specs):
                special_training = (
                    replace(config.training, diversity_weight=0.0)
                    if name == "no_diversity"
                    else config.training
                )
                if is_linear:
                    special_operator: BaseCreativeOperatorV2 = (
                        ConditionalLinearOperatorV2(
                            config.landscape.dimension,
                            candidate_count,
                            config.training.latent_dimension,
                            seed + 41_000 + special_index,
                            use_goal=use_goal,
                        )
                    )
                else:
                    special_operator = ConditionalMLPOperatorV2(
                        config.landscape.dimension,
                        candidate_count,
                        config.training.latent_dimension,
                        config.training.hidden_dimension,
                        seed + 41_000 + special_index,
                        use_goal=use_goal,
                    )
                train_operator_v2(
                    special_operator,
                    train_tasks,
                    train_starts,
                    special_training,
                    iterations=config.training.gen2_iterations,
                    seed=seed + 41_100 + special_index,
                )
                special_operator.save(
                    model_dir / f"seed_{seed}_ablation_{name}.npz", seed=seed
                )
                summary, _ = _evaluate(
                    method=name,
                    tasks=test_tasks,
                    starts=test_starts,
                    config=config,
                    seed=seed + 41_200 + special_index,
                    candidates=candidate_count,
                    operator=special_operator,
                    critic=critic,
                )
                ablation_rows.append({"seed": seed, "ablation": name, **summary})

            for k_index, candidate_count in enumerate(config.k_values):
                if candidate_count == config.training.candidates:
                    continue
                k_training = replace(config.training, candidates=candidate_count)
                k_operator = ConditionalMLPOperatorV2(
                    config.landscape.dimension,
                    candidate_count,
                    config.training.latent_dimension,
                    config.training.hidden_dimension,
                    seed + 42_000 + k_index,
                )
                train_operator_v2(
                    k_operator,
                    train_tasks,
                    train_starts,
                    k_training,
                    iterations=config.training.gen2_iterations,
                    seed=seed + 42_100 + k_index,
                )
                k_operator.save(
                    model_dir / f"seed_{seed}_ablation_k_{candidate_count}.npz",
                    seed=seed,
                )
                summary, _ = _evaluate(
                    method=f"K={candidate_count}",
                    tasks=test_tasks,
                    starts=test_starts,
                    config=config,
                    seed=seed + 42_200 + k_index,
                    candidates=candidate_count,
                    operator=k_operator,
                    critic=critic,
                )
                ablation_rows.append(
                    {"seed": seed, "ablation": f"K={candidate_count}", **summary}
                )

    _write_csv(output / "primary_metrics.csv", primary_rows)
    _write_csv(output / "endpoint_metrics.csv", endpoint_rows)
    _write_csv(output / "alpha_ablation.csv", alpha_rows)
    _write_csv(output / "generalization_metrics.csv", generalization_rows)
    _write_csv(output / "ablation_metrics.csv", ablation_rows)
    history_rows: list[dict[str, object]] = []
    for model, history in histories.items():
        history_rows.extend({"model": model, **row} for row in history)
    _write_csv(output / "training_history_first_seed.csv", history_rows)

    effects = {
        "generation2_minus_random": _paired_effects(
            primary_rows, "generation2", "random"
        ),
        "generation2_minus_pure_divergence": _paired_effects(
            primary_rows, "generation2", "pure_divergence"
        ),
        "generation2_minus_generation1": _paired_effects(
            primary_rows, "generation2", "generation1"
        ),
    }
    aggregates: dict[str, object] = {
        "primary": {
            method: aggregate_seeds(
                [row for row in primary_rows if row["method"] == method]
            )
            for method in [
                "baseline",
                "random",
                "pure_divergence",
                "generation1",
                "generation2",
            ]
        },
        "paired_effects": effects,
        "seed_count": len(config.seeds),
        "ablation_seed_count": min(config.ablation_seeds, len(config.seeds)),
    }
    (output / "aggregate_metrics.json").write_text(
        json.dumps(aggregates, indent=2), encoding="utf-8"
    )
    _write_report(
        output / "REPORT.md",
        primary_rows,
        generalization_rows,
        ablation_rows,
        effects,
    )

    plot_quality_diversity_v2(primary_rows, output / "quality_vs_diversity.png")
    plot_metric_bars_v2(primary_rows, output / "coverage_decoy_collapse.png")
    plot_quality_displacement_v2(endpoint_rows, output / "quality_vs_displacement.png")
    plot_training_v2(histories, output / "training_curves_v2.png")
    plot_goal_conditioning_v2(
        goal_records,
        config.landscape.useful_peaks,
        output / "goal_conditioned_selection.png",
    )
    if representative_task is not None:
        plot_method_geometry_v2(
            representative_task,
            representative_records,
            output / "landscape_methods_trajectories.png",
        )
    if generalization_rows:
        plot_grouped_study_v2(
            [row for row in generalization_rows if row["method"] == "generation2"],
            "variant",
            output / "generalization_v2.png",
        )
    if ablation_rows:
        plot_grouped_study_v2(ablation_rows, "ablation", output / "ablations_v2.png")
    plot_grouped_study_v2(alpha_rows, "alpha", output / "alpha_ablation_v2.png")
    return aggregates
