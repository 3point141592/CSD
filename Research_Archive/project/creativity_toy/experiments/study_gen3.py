"""Generation-3 learned-alpha comparison using unchanged V2 tasks and budgets."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..analysis_utils import (
    classify_candidate,
    pearson_correlation,
    spearman_correlation,
    statistics,
)
from ..config_gen3 import Gen3StudyConfig
from ..creative_operator_v2 import ConditionalMLPOperatorV2
from ..creative_operator_v3 import ConditionalLearnedAlphaOperatorV3
from ..landscapes_v2 import V2LandscapeFamily
from ..normalization_diagnostics import (
    equivariance_diagnostics,
    evaluate_candidate_sets,
)
from ..training_v2 import fit_critic_v2, train_operator_v2
from ..visualization_gen3 import (
    plot_alpha_by_candidate_and_goal,
    plot_alpha_by_category,
    plot_alpha_relationships,
    plot_gen3_main_comparison,
    plot_learned_alpha_histograms,
)
from .study_v2 import _make_starts, _training_starts, _write_csv

VARIANTS = [
    "gen2_unnormalized",
    "gen2_normalized_fixed_alpha3",
    "gen3_global_alpha",
    "gen3_contextual_linear_alpha",
    "gen3_contextual_mlp_alpha",
]
MODE_TO_VARIANT = {
    "global": "gen3_global_alpha",
    "contextual_linear": "gen3_contextual_linear_alpha",
    "contextual_mlp": "gen3_contextual_mlp_alpha",
}
MAIN_METRICS = [
    "mean_quality",
    "oracle_best_quality",
    "coverage",
    "useful_exploration_rate",
    "decoy_rate",
    "target_hit_rate",
    "critic_regret",
    "mean_alpha",
    "alpha_sd",
]


def _coerce(value: str) -> object:
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "":
        return ""
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


def _load_csv(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _group_mean_sd(records: Sequence[Mapping[str, object]], key: str) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in records:
        grouped[int(row[key])].append(float(row["candidate_alpha"]))
    means = [float(np.mean(values)) for values in grouped.values()]
    return float(np.std(means, ddof=0)) if means else 0.0


def _enrich_alpha_summary(
    summary: dict[str, object], records: Sequence[dict[str, object]]
) -> None:
    alpha = np.asarray([float(row["candidate_alpha"]) for row in records])
    quality = np.asarray([float(row["true_utility"]) for row in records])
    critic = np.asarray([float(row["critic_score"]) for row in records])
    sets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in records:
        sets[(int(row["task_id"]), int(row["trial"]))].append(
            float(row["candidate_alpha"])
        )
    summary.update(
        {
            "mean_alpha": float(np.mean(alpha)),
            "alpha_sd": float(np.std(alpha, ddof=0)),
            "alpha_p05": float(np.quantile(alpha, 0.05)),
            "alpha_p25": float(np.quantile(alpha, 0.25)),
            "alpha_p50": float(np.quantile(alpha, 0.50)),
            "alpha_p75": float(np.quantile(alpha, 0.75)),
            "alpha_p95": float(np.quantile(alpha, 0.95)),
            "within_set_alpha_sd": float(
                np.mean([np.std(values, ddof=0) for values in sets.values()])
            ),
            "between_goal_alpha_sd": _group_mean_sd(records, "goal_index"),
            "between_geometry_alpha_sd": _group_mean_sd(records, "geometry_id"),
            "between_candidate_alpha_sd": _group_mean_sd(records, "candidate_id"),
            "alpha_quality_pearson": pearson_correlation(alpha, quality),
            "alpha_quality_spearman": spearman_correlation(alpha, quality),
            "alpha_critic_pearson": pearson_correlation(alpha, critic),
            "alpha_critic_spearman": spearman_correlation(alpha, critic),
        }
    )


def _prepare_candidate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    variant: str,
    useful_threshold: float,
    implicit: bool,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for original in rows:
        row = dict(original)
        row["variant"] = variant
        row["method"] = variant
        if "candidate_alpha" not in row:
            row["candidate_alpha"] = (
                float(row["raw_norm"]) if implicit else float(row["perturbation_norm"])
            )
        row["category"] = classify_candidate(row, useful_threshold)
        output.append(row)
    return output


def _load_baseline(
    root: Path,
    source_variant: str,
    target_variant: str,
    seeds: Sequence[int],
    *,
    useful_threshold: float,
    implicit: bool,
    direction_parameters: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if not root.exists():
        raise FileNotFoundError(
            f"Required completed baseline directory is missing: {root}"
        )
    seed_set = set(seeds)
    primary_source = [
        row
        for row in _load_csv(root / "primary_per_seed.csv")
        if row["variant"] == source_variant and int(row["seed"]) in seed_set
    ]
    candidate_source = [
        row
        for row in _load_csv(root / "candidate_diagnostics.csv")
        if row["variant"] == source_variant and int(row["seed"]) in seed_set
    ]
    selector_source = [
        row
        for row in _load_csv(root / "selector_per_seed.csv")
        if row["variant"] == source_variant and int(row["seed"]) in seed_set
    ]
    candidates = _prepare_candidate_rows(
        candidate_source,
        variant=target_variant,
        useful_threshold=useful_threshold,
        implicit=implicit,
    )
    candidates_by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        candidates_by_seed[int(row["seed"])].append(row)
    primary: list[dict[str, object]] = []
    for original in primary_source:
        row = dict(original)
        row["variant"] = target_variant
        row["method"] = target_variant
        row["parameter_count"] = direction_parameters
        row["additional_alpha_parameters"] = 0
        _enrich_alpha_summary(row, candidates_by_seed[int(row["seed"])])
        primary.append(row)
    selectors = []
    for original in selector_source:
        row = dict(original)
        row["variant"] = target_variant
        selectors.append(row)
    if len(primary) != len(seeds):
        raise ValueError(
            f"Baseline {target_variant} has {len(primary)} rows; expected {len(seeds)}"
        )
    return primary, candidates, selectors


def _aggregate_table(
    primary: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for variant in VARIANTS:
        subset = [row for row in primary if row["variant"] == variant]
        result: dict[str, object] = {
            "variant": variant,
            "parameter_count": int(subset[0]["parameter_count"]),
            "additional_alpha_parameters": int(
                subset[0]["additional_alpha_parameters"]
            ),
        }
        for metric in MAIN_METRICS + [
            "selected_quality",
            "selected_target_rate",
            "alpha_p05",
            "alpha_p25",
            "alpha_p50",
            "alpha_p75",
            "alpha_p95",
            "within_set_alpha_sd",
            "between_goal_alpha_sd",
            "between_geometry_alpha_sd",
            "between_candidate_alpha_sd",
            "alpha_quality_pearson",
            "alpha_quality_spearman",
            "alpha_critic_pearson",
            "alpha_critic_spearman",
        ]:
            stats = statistics([float(row[metric]) for row in subset])
            for key, value in stats.items():
                result[f"{metric}_{key}"] = value
        output.append(result)
    return output


def _paired_effects(
    primary: Sequence[Mapping[str, object]],
    comparisons: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    lookup = {(int(row["seed"]), str(row["variant"])): row for row in primary}
    seeds = sorted({int(row["seed"]) for row in primary})
    metrics = MAIN_METRICS + [
        "selected_quality",
        "selected_target_rate",
        "within_set_alpha_sd",
    ]
    output: list[dict[str, object]] = []
    for left, right in comparisons:
        for metric in metrics:
            differences = [
                float(lookup[(seed, left)][metric])
                - float(lookup[(seed, right)][metric])
                for seed in seeds
            ]
            stats = statistics(differences)
            output.append(
                {
                    "contrast": f"{left}_minus_{right}",
                    "metric": metric,
                    "mean_difference": stats["mean"],
                    "sd_difference": stats["sd"],
                    "ci95": stats["ci95"],
                    "ci95_low": stats["mean"] - stats["ci95"],
                    "ci95_high": stats["mean"] + stats["ci95"],
                    "n": stats["n"],
                    "ci_excludes_zero": bool(abs(stats["mean"]) > stats["ci95"]),
                }
            )
    return output


def _group_alpha_rows(
    candidates: Sequence[Mapping[str, object]], key: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for variant in VARIANTS:
        subset = [row for row in candidates if row["variant"] == variant]
        groups = sorted({int(row[key]) for row in subset})
        for group in groups:
            rows = [row for row in subset if int(row[key]) == group]
            alpha = [float(row["candidate_alpha"]) for row in rows]
            output.append(
                {
                    "variant": variant,
                    key: group,
                    "mean_alpha": float(np.mean(alpha)),
                    "median_alpha": float(np.median(alpha)),
                    "alpha_sd": float(np.std(alpha, ddof=1)) if len(alpha) > 1 else 0.0,
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
                    "count": len(rows),
                }
            )
    return output


def _category_alpha_rows(
    candidates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    categories = ["anchor", "useful_non_target", "target_hit", "decoy", "other"]
    for variant in VARIANTS:
        for category in categories:
            rows = [
                row
                for row in candidates
                if row["variant"] == variant and row["category"] == category
            ]
            if not rows:
                continue
            alpha = np.asarray([float(row["candidate_alpha"]) for row in rows])
            output.append(
                {
                    "variant": variant,
                    "category": category,
                    "mean_alpha": float(np.mean(alpha)),
                    "median_alpha": float(np.median(alpha)),
                    "alpha_sd": float(np.std(alpha, ddof=1)) if len(alpha) > 1 else 0.0,
                    "mean_quality": float(
                        np.mean([float(row["true_utility"]) for row in rows])
                    ),
                    "count": len(rows),
                }
            )
    return output


def _format(table_row: Mapping[str, object], metric: str) -> str:
    return (
        f"{float(table_row[f'{metric}_mean']):.3f} ± "
        f"{float(table_row[f'{metric}_sd']):.3f}"
    )


def _write_main_table(path: Path, table: Sequence[Mapping[str, object]]) -> None:
    labels = {
        "gen2_unnormalized": "Gen2 unnormalized",
        "gen2_normalized_fixed_alpha3": "Normalized alpha=3",
        "gen3_global_alpha": "Learned global alpha",
        "gen3_contextual_linear_alpha": "Learned contextual alpha",
        "gen3_contextual_mlp_alpha": "Learned contextual alpha MLP",
    }
    lines = [
        "# Generation-3 main comparison",
        "",
        "Values are mean ± sample SD across seeds. Every row uses K=8 and the same downstream evaluation budget.",
        "",
        "| Variant | Quality | Oracle Q | Coverage | Useful | Decoy | Target Hit | Critic Regret | Mean Alpha | Alpha SD | Parameters | Added Alpha Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {labels[str(row['variant'])]} | {_format(row, 'mean_quality')} | "
            f"{_format(row, 'oracle_best_quality')} | {_format(row, 'coverage')} | "
            f"{_format(row, 'useful_exploration_rate')} | {_format(row, 'decoy_rate')} | "
            f"{_format(row, 'target_hit_rate')} | {_format(row, 'critic_regret')} | "
            f"{_format(row, 'mean_alpha')} | {_format(row, 'alpha_sd')} | "
            f"{int(row['parameter_count'])} | {int(row['additional_alpha_parameters'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _effect(
    paired: Sequence[Mapping[str, object]], contrast: str, metric: str
) -> Mapping[str, object]:
    return next(
        row for row in paired if row["contrast"] == contrast and row["metric"] == metric
    )


def _effect_text(row: Mapping[str, object], higher_is_better: bool) -> str:
    mean = float(row["mean_difference"])
    ci = float(row["ci95"])
    if not bool(row["ci_excludes_zero"]):
        return f"unresolved ({mean:+.3f}, 95% CI ±{ci:.3f})"
    good = mean > 0.0 if higher_is_better else mean < 0.0
    return f"{'improved' if good else 'worsened'} ({mean:+.3f}, 95% CI ±{ci:.3f})"


def _write_interpretation(
    path: Path,
    table: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
) -> None:
    contrast = "gen3_contextual_linear_alpha_minus_gen2_normalized_fixed_alpha3"
    linear = next(
        row for row in table if row["variant"] == "gen3_contextual_linear_alpha"
    )
    global_row = next(row for row in table if row["variant"] == "gen3_global_alpha")
    mlp = next(row for row in table if row["variant"] == "gen3_contextual_mlp_alpha")
    mean_alpha = float(linear["mean_alpha_mean"])
    alpha_sd = float(linear["alpha_sd_mean"])
    within_sd = float(linear["within_set_alpha_sd_mean"])
    nontrivial = alpha_sd > 0.10 and within_sd > 0.05
    useful_effect = _effect(paired, contrast, "useful_exploration_rate")
    decoy_effect = _effect(paired, contrast, "decoy_rate")
    quality_effect = _effect(paired, contrast, "selected_quality")
    coverage_effect = _effect(paired, contrast, "coverage")
    beneficial = (
        bool(useful_effect["ci_excludes_zero"])
        and float(useful_effect["mean_difference"]) > 0.0
        and not (
            bool(quality_effect["ci_excludes_zero"])
            and float(quality_effect["mean_difference"]) < 0.0
        )
    ) or (
        bool(decoy_effect["ci_excludes_zero"])
        and float(decoy_effect["mean_difference"]) < 0.0
        and not (
            bool(coverage_effect["ci_excludes_zero"])
            and float(coverage_effect["mean_difference"]) < 0.0
        )
    )
    if beneficial and nontrivial:
        verdict = "context-dependent magnitude is supported in this environment"
    elif not nontrivial:
        verdict = (
            "adaptive magnitude appears unnecessary; alpha stayed near a global radius"
        )
    else:
        verdict = "alpha varied, but a performance benefit over fixed alpha=3 was not established"

    lines = [
        "# Generation-3 interpretation",
        "",
        "This report concerns a synthetic exploration mechanism only. Paired effects use approximate 95% confidence intervals across seeds and are not multiple-comparison adjusted.",
        "",
        f"1. The primary contextual linear alpha averaged {mean_alpha:.3f}, with pooled alpha SD {alpha_sd:.3f} and mean within-candidate-set SD {within_sd:.3f}. Alpha variation was {'nontrivial' if nontrivial else 'small'} relative to the null alpha=3.",
        f"2. Versus normalized fixed alpha=3, selected quality was {_effect_text(quality_effect, True)}; useful exploration was {_effect_text(useful_effect, True)}; decoy rate was {_effect_text(decoy_effect, False)}; coverage was {_effect_text(coverage_effect, True)}.",
        f"3. The global learned scalar converged to mean alpha {float(global_row['mean_alpha_mean']):.3f}; its useful exploration was {float(global_row['useful_exploration_rate_mean']):.3f} and decoy rate {float(global_row['decoy_rate_mean']):.3f}.",
        f"4. The contextual MLP used {int(mlp['additional_alpha_parameters'])} added parameters versus {int(linear['additional_alpha_parameters'])} for the linear head. Its useful exploration was {float(mlp['useful_exploration_rate_mean']):.3f}, decoy rate {float(mlp['decoy_rate_mean']):.3f}, and selected quality {float(mlp['selected_quality_mean']):.3f}.",
        f"5. **Conclusion:** {verdict}.",
        "",
        "The alpha head received no extra objective evaluations, the alpha regularizer remained zero, and no environment or critic changes were made.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gen3_study(config: Gen3StudyConfig, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    base = config.base
    family = V2LandscapeFamily(base.landscape)
    direction_parameters = ConditionalMLPOperatorV2(
        base.landscape.dimension,
        base.training.candidates,
        base.training.latent_dimension,
        base.training.hidden_dimension,
        seed=0,
    ).parameter_count

    unnorm_primary, unnorm_candidates, unnorm_selectors = _load_baseline(
        Path(config.unnormalized_results),
        "gen2_unnormalized",
        "gen2_unnormalized",
        base.seeds,
        useful_threshold=base.landscape.useful_threshold,
        implicit=True,
        direction_parameters=direction_parameters,
    )
    fixed_primary, fixed_candidates, fixed_selectors = _load_baseline(
        Path(config.fixed_alpha_results),
        "gen2_normalized",
        "gen2_normalized_fixed_alpha3",
        base.seeds,
        useful_threshold=base.landscape.useful_threshold,
        implicit=False,
        direction_parameters=direction_parameters,
    )
    primary_rows = [*unnorm_primary, *fixed_primary]
    candidate_rows = [*unnorm_candidates, *fixed_candidates]
    selector_rows = [*unnorm_selectors, *fixed_selectors]
    candidate_set_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []

    for seed_index, seed in enumerate(base.seeds):
        print(f"[Gen3 {seed_index + 1}/{len(base.seeds)}] seed {seed}", flush=True)
        train_tasks = family.sample_tasks(
            geometries=base.train_geometries,
            seed=seed + 10_000,
            variant="iid",
            all_goals=True,
        )
        train_starts = _training_starts(
            _make_starts(train_tasks, trials=1, seed=seed + 10_001)
        )
        test_tasks = family.sample_tasks(
            geometries=base.test_geometries,
            seed=seed + 20_000,
            variant="iid",
            all_goals=True,
        )
        test_starts = _make_starts(
            test_tasks, trials=base.trials_per_task, seed=seed + 20_001
        )

        for mode in config.alpha_modes:
            variant = MODE_TO_VARIANT[mode]
            operator = ConditionalLearnedAlphaOperatorV3(
                base.landscape.dimension,
                base.training.candidates,
                base.training.latent_dimension,
                base.training.hidden_dimension,
                seed + 10_020,
                magnitude_mode=mode,
                alpha_max=config.alpha_max,
                alpha_hidden_dimension=config.alpha_hidden_dimension,
            )
            train_start = time.perf_counter()
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
            training_seconds = time.perf_counter() - train_start
            operator.save(
                model_dir / f"seed_{seed}_{variant}.npz",
                seed=seed,
                alpha_regularization_weight=base.training.alpha_regularization_weight,
            )
            critic.save(model_dir / f"seed_{seed}_{variant}_critic.npz")
            if seed_index == 0:
                history_rows.extend({"variant": variant, **row} for row in history)

            summary, records, set_records, selectors = evaluate_candidate_sets(
                variant=variant,
                operator=operator,
                critic=critic,
                tasks=test_tasks,
                starts=test_starts,
                config=base.training,
                seed=seed + 21_004,
                alpha=config.alpha_initial,
                useful_threshold=base.landscape.useful_threshold,
                high_quality_threshold=base.landscape.high_quality_threshold,
            )
            for row in records:
                row["seed"] = seed
                row["category"] = classify_candidate(
                    row, base.landscape.useful_threshold
                )
            for row in set_records:
                row["seed"] = seed
            for row in selectors:
                row["seed"] = seed
            _enrich_alpha_summary(summary, records)
            equivariance = equivariance_diagnostics(
                operator,
                test_tasks,
                test_starts,
                seed=seed + 20_600,
                trials=3,
                translation_magnitude=2.0,
            )
            primary_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "method": variant,
                    "training_wall_clock_seconds": training_seconds,
                    "parameter_count": operator.parameter_count,
                    "additional_alpha_parameters": operator.additional_alpha_parameters,
                    **summary,
                    **equivariance,
                }
            )
            candidate_rows.extend(records)
            candidate_set_rows.extend(set_records)
            selector_rows.extend(selectors)

    table = _aggregate_table(primary_rows)
    comparisons = [
        ("gen3_contextual_linear_alpha", "gen2_normalized_fixed_alpha3"),
        ("gen3_contextual_linear_alpha", "gen2_unnormalized"),
        ("gen3_global_alpha", "gen2_normalized_fixed_alpha3"),
        ("gen3_contextual_mlp_alpha", "gen2_normalized_fixed_alpha3"),
        ("gen3_contextual_linear_alpha", "gen3_global_alpha"),
        ("gen3_contextual_mlp_alpha", "gen3_contextual_linear_alpha"),
    ]
    paired = _paired_effects(primary_rows, comparisons)
    by_candidate = _group_alpha_rows(candidate_rows, "candidate_id")
    by_goal = _group_alpha_rows(candidate_rows, "goal_index")
    by_geometry = _group_alpha_rows(candidate_rows, "geometry_id")
    by_category = _category_alpha_rows(candidate_rows)

    _write_csv(output / "primary_per_seed.csv", primary_rows)
    _write_csv(output / "candidate_diagnostics.csv", candidate_rows)
    _write_csv(output / "candidate_set_diagnostics.csv", candidate_set_rows)
    _write_csv(output / "selector_per_seed.csv", selector_rows)
    _write_csv(output / "training_history_first_seed.csv", history_rows)
    _write_csv(output / "GEN3_TABLE.csv", table)
    _write_csv(output / "paired_differences.csv", paired)
    _write_csv(output / "alpha_by_candidate.csv", by_candidate)
    _write_csv(output / "alpha_by_goal.csv", by_goal)
    _write_csv(output / "alpha_by_geometry.csv", by_geometry)
    _write_csv(output / "alpha_by_category.csv", by_category)
    _write_main_table(output / "GEN3_TABLE.md", table)
    _write_interpretation(output / "GEN3_INTERPRETATION.md", table, paired)

    plot_gen3_main_comparison(primary_rows, output / "gen3_main_comparison.png")
    plot_learned_alpha_histograms(
        candidate_rows, output / "learned_alpha_histograms.png"
    )
    plot_alpha_by_candidate_and_goal(
        candidate_rows,
        "gen3_contextual_linear_alpha",
        output / "contextual_alpha_by_candidate_goal.png",
    )
    plot_alpha_relationships(
        candidate_rows,
        "gen3_contextual_linear_alpha",
        output / "contextual_alpha_relationships.png",
    )
    plot_alpha_by_category(
        candidate_rows,
        "gen3_contextual_linear_alpha",
        output / "contextual_alpha_by_category.png",
    )

    aggregate: dict[str, object] = {
        "seed_count": len(base.seeds),
        "main_table": table,
        "paired_differences": paired,
        "source_baselines": {
            "gen2_unnormalized": config.unnormalized_results,
            "gen2_normalized_fixed_alpha3": config.fixed_alpha_results,
        },
    }
    (output / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
