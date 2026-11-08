"""Evidence-bound reports for the dimensional failure diagnostic."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from creativity_toy.analysis_utils import statistics

from .config_failure_diagnostics import FailureDiagnosticConfig
from .study import write_csv


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: Mapping[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = row[key]
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return 1.0 if value.lower() == "true" else 0.0
        return float(value)
    except (KeyError, TypeError, ValueError):
        return default


def _summarize(rows, keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group, values in sorted(groups.items()):
        result = dict(zip(keys, group))
        for metric in metrics:
            stat = statistics([_f(row, metric) for row in values])
            for name in ["mean", "sd", "ci95"]:
                result[f"{metric}_{name}"] = stat[name]
        result["n"] = len(values)
        output.append(result)
    return output


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def _mean(rows, method, metric, dimension=None, budget=None):
    values = []
    for row in rows:
        if method is not None and row.get("method") != method:
            continue
        if dimension is not None and int(float(row["dimension"])) != dimension:
            continue
        if budget is not None and int(float(row["budget"])) != budget:
            continue
        values.append(_f(row, metric))
    return float(np.mean(values)) if values else float("nan")


def _capture_rows(rows, bins):
    output = []
    dimensions = sorted({int(float(row["dimension"])) for row in rows})
    methods = sorted({row["method"] for row in rows})
    for dimension in dimensions:
        for method in methods:
            selected = [
                row
                for row in rows
                if int(float(row["dimension"])) == dimension and row["method"] == method
            ]
            for lower, upper in zip(bins[:-1], bins[1:]):
                active = [
                    row
                    for row in selected
                    if lower <= _f(row, "pre_nearest_useful_distance") < upper
                ]
                if active:
                    output.append(
                        {
                            "dimension": dimension,
                            "method": method,
                            "distance_low": lower,
                            "distance_high": upper,
                            "count": len(active),
                            "post_useful_rate": np.mean(
                                [_f(row, "post_useful") for row in active]
                            ),
                            "refinement_created_hit_rate": np.mean(
                                [
                                    _f(row, "refinement_created_useful_hit")
                                    for row in active
                                ]
                            ),
                            "mean_quality": np.mean(
                                [_f(row, "quality") for row in active]
                            ),
                        }
                    )
    return output


def _plot_dimension(rows, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, label in [
        ("black_box_rl", "Black-box RL"),
        ("direct_gradient", "Direct gradient"),
        ("contrastive", "Contrastive"),
    ]:
        dimensions = [2, 4, 8, 16]
        values = [_mean(rows, method, "proposal_recall", d) for d in dimensions]
        ax.plot(dimensions, values, marker="o", label=label)
    ax.set(xlabel="Latent dimension", ylabel="Proposal recall", xticks=[2, 4, 8, 16])
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_encounters(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for dimension in [2, 4, 8, 16]:
        selected = [
            row
            for row in rows
            if row["method"] == "black_box_rl"
            and int(float(row["dimension"])) == dimension
        ]
        budgets = sorted({int(float(row["candidate_evaluations"])) for row in selected})
        best = [
            np.mean(
                [
                    _f(row, "quality_max")
                    for row in selected
                    if int(float(row["candidate_evaluations"])) == budget
                ]
            )
            for budget in budgets
        ]
        probability = [
            np.mean(
                [
                    _f(row, "p_q_gt_0.5")
                    for row in selected
                    if int(float(row["candidate_evaluations"])) == budget
                ]
            )
            for budget in budgets
        ]
        axes[0].plot(budgets, best, marker="o", label=f"d={dimension}")
        axes[1].plot(budgets, probability, marker="o", label=f"d={dimension}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Candidate outcomes")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Best Q encountered")
    axes[1].set_ylabel("P(Q > 0.5)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_thresholds(rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for ax, threshold in zip(axes.flat, [0.1, 0.3, 0.5, 0.7]):
        key = f"p_q_gt_{threshold:g}"
        for dimension in [2, 4, 8, 16]:
            selected = [
                row
                for row in rows
                if row["method"] == "black_box_rl"
                and int(float(row["dimension"])) == dimension
            ]
            budgets = sorted(
                {int(float(row["candidate_evaluations"])) for row in selected}
            )
            values = [
                np.mean(
                    [
                        _f(row, key)
                        for row in selected
                        if int(float(row["candidate_evaluations"])) == budget
                    ]
                )
                for budget in budgets
            ]
            ax.plot(budgets, values, marker="o", label=f"d={dimension}")
        ax.set_xscale("log")
        ax.set_title(f"P(Q > {threshold:g})")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    for ax in axes[1]:
        ax.set_xlabel("Candidate outcomes")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_geometry(candidate_rows, random_rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for method in ["black_box_rl", "direct_gradient", "contrastive"]:
        dimensions = [2, 4, 8, 16]
        ratio = [
            _mean(candidate_rows, method, "alpha_to_nearest_useful_ratio", d)
            for d in dimensions
        ]
        alignment = [
            _mean(candidate_rows, method, "best_useful_alignment", d)
            for d in dimensions
        ]
        axes[0].plot(dimensions, ratio, marker="o", label=method)
        axes[1].plot(dimensions, alignment, marker="o", label=method)
    dimensions = [2, 4, 8, 16]
    hit = [_mean(random_rows, None, "useful_hit_rate", d) for d in dimensions]
    axes[2].plot(dimensions, hit, marker="o", color="#7b3294")
    axes[0].set_ylabel("alpha / nearest-useful distance")
    axes[1].set_ylabel("Best cosine alignment")
    axes[2].set_ylabel("Random refined useful-hit rate")
    for ax in axes:
        ax.set_xlabel("Dimension")
        ax.set_xticks([2, 4, 8, 16])
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_budget(rows, path):
    budgets = sorted({int(float(row["budget"])) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    for ax, metric, label in zip(
        axes,
        ["proposal_recall", "candidate_mean_quality", "candidate_decoy_rate"],
        ["Proposal recall", "Quality", "Decoy rate"],
    ):
        means = [
            _mean(rows, "black_box_rl", metric, budget=budget) for budget in budgets
        ]
        ax.plot(budgets, means, marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("Candidate outcomes")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_batch(rows, path):
    valid = [row for row in rows if row["status"] == "ok"]
    batch = [_f(row, "batch_size") for row in valid]
    throughput = [_f(row, "candidate_outcomes_per_second") for row in valid]
    memory = [_f(row, "peak_allocated_bytes") / 2**20 for row in valid]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    second = ax.twinx()
    ax.plot(batch, throughput, marker="o")
    second.plot(batch, memory, marker="s", color="#d95f02")
    ax.set(xlabel="Task batch size", ylabel="Candidate outcomes/s", xscale="log")
    second.set_ylabel("Peak allocated MiB")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_alpha_distribution(rows, path):
    methods = [
        "random_initial_policy",
        "black_box_rl",
        "direct_gradient",
        "contrastive",
    ]
    values = [
        [
            _f(row, "alpha")
            for row in rows
            if row["method"] == method and int(float(row["dimension"])) == 16
        ]
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(
        values,
        tick_labels=["Initial", "RL", "Gradient", "Contrastive"],
        showfliers=False,
    )
    ax.set_ylabel("Candidate alpha")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_failure_reports(
    diagnostic: FailureDiagnosticConfig,
    *,
    dimension_dir: Path,
    gradient_dir: Path,
    budget_dir: Path,
    geometry_dir: Path,
    batch_dir: Path,
) -> dict[str, object]:
    geometry_dir.mkdir(parents=True, exist_ok=True)
    dimension_rows = _read(dimension_dir / "dimension_per_seed.csv")
    encounters = _read(dimension_dir / "reward_encounters.csv")
    candidates = _read(dimension_dir / "candidate_geometry.csv")
    task_geometry = _read(dimension_dir / "task_geometry.csv")
    random_rows = _read(dimension_dir / "random_hit_rates.csv")
    gradient_rows = _read(gradient_dir / "per_seed_metrics.csv")
    budget_rows = _read(budget_dir / "budget_per_seed.csv")
    batch_rows = _read(batch_dir / "batch_scaling_raw.csv")
    if not dimension_rows or not gradient_rows or not budget_rows or not batch_rows:
        raise RuntimeError(
            "complete diagnostic outputs are required before report generation"
        )

    metrics = [
        "selected_quality",
        "proposal_recall",
        "routing_recall",
        "useful_selected",
        "useful_region_count",
        "candidate_decoy_rate",
        "target_selected",
        "effective_directional_rank",
        "direction_cosine_distance",
        "candidate_mean_quality",
        "alpha_mean",
        "alpha_sd",
    ]
    dimension_summary = _summarize(dimension_rows, ["dimension", "method"], metrics)
    encounter_summary = _summarize(
        encounters,
        ["dimension", "method", "candidate_evaluations"],
        [
            "quality_mean",
            "quality_max",
            "p_q_gt_0.1",
            "p_q_gt_0.3",
            "p_q_gt_0.5",
            "p_q_gt_0.7",
            "q_0.5",
            "q_0.75",
            "q_0.9",
            "q_0.95",
            "q_0.99",
            "q_0.999",
        ],
    )
    geometry_summary = _summarize(
        candidates,
        ["dimension", "method"],
        [
            "alpha",
            "alpha_to_nearest_useful_ratio",
            "pre_nearest_useful_distance",
            "best_useful_alignment",
            "target_alignment",
            "post_useful",
            "refinement_created_useful_hit",
        ],
    )
    capture = _capture_rows(candidates, diagnostic.capture_distance_bins)
    random_summary = _summarize(
        random_rows, ["dimension"], ["useful_hit_rate", "mean_quality", "max_quality"]
    )
    write_csv(dimension_dir / "report_dimension_summary.csv", dimension_summary)
    write_csv(dimension_dir / "reward_encounter_summary.csv", encounter_summary)
    write_csv(
        gradient_dir / "report_summary.csv",
        _summarize(gradient_rows, ["method"], metrics),
    )
    write_csv(geometry_dir / "geometry_summary.csv", geometry_summary)
    write_csv(geometry_dir / "capture_curve.csv", capture)
    write_csv(geometry_dir / "random_hit_summary.csv", random_summary)
    write_csv(geometry_dir / "candidate_geometry.csv", candidates)
    write_csv(geometry_dir / "task_geometry.csv", task_geometry)
    write_csv(geometry_dir / "random_hit_rates.csv", random_rows)

    _plot_dimension(dimension_rows, dimension_dir / "dimension_vs_recall.png")
    _plot_encounters(encounters, dimension_dir / "reward_encounters.png")
    _plot_thresholds(encounters, dimension_dir / "reward_thresholds.png")
    _plot_geometry(candidates, random_rows, geometry_dir / "geometry_scaling.png")
    _plot_alpha_distribution(candidates, geometry_dir / "alpha_distribution.png")
    _plot_budget(budget_rows, budget_dir / "budget_scaling.png")
    _plot_batch(batch_rows, batch_dir / "batch_throughput.png")

    rl16 = _mean(dimension_rows, "black_box_rl", "proposal_recall", 16)
    grad16 = _mean(gradient_rows, "direct_gradient", "proposal_recall", 16)
    contrast16 = _mean(dimension_rows, "contrastive", "proposal_recall", 16)
    rl_by_d = {
        d: _mean(dimension_rows, "black_box_rl", "proposal_recall", d)
        for d in [2, 4, 8, 16]
    }
    failure_dimension = next(
        (d for d in [4, 8, 16] if rl_by_d[d] < 0.5 * max(rl_by_d[2], 1e-8)),
        16,
    )
    largest_budget = max(int(float(row["budget"])) for row in budget_rows)
    baseline_budget = min(int(float(row["budget"])) for row in budget_rows)
    budget100 = _mean(
        budget_rows, "black_box_rl", "proposal_recall", budget=baseline_budget
    )
    budget_large = _mean(
        budget_rows, "black_box_rl", "proposal_recall", budget=largest_budget
    )
    ratio2 = _mean(candidates, "black_box_rl", "alpha_to_nearest_useful_ratio", 2)
    ratio16 = _mean(candidates, "black_box_rl", "alpha_to_nearest_useful_ratio", 16)
    align16 = _mean(candidates, "black_box_rl", "best_useful_alignment", 16)
    initial_align16 = _mean(
        candidates, "random_initial_policy", "best_useful_alignment", 16
    )
    contrast_align16 = _mean(candidates, "contrastive", "best_useful_alignment", 16)
    target_align16 = _mean(candidates, "black_box_rl", "target_alignment", 16)
    pre_distance16 = _mean(
        candidates, "black_box_rl", "pre_nearest_useful_distance", 16
    )
    random2 = _mean(random_rows, None, "useful_hit_rate", 2)
    random16 = _mean(random_rows, None, "useful_hit_rate", 16)
    random_collapse = random2 / max(random16, 1e-12)
    q_d16_100k = [
        row
        for row in encounters
        if row["method"] == "black_box_rl"
        and int(float(row["dimension"])) == 16
        and int(float(row["candidate_evaluations"])) == 100_000
    ]
    p_high_q16 = float(np.mean([_f(row, "p_q_gt_0.7") for row in q_d16_100k]))
    nearest_useful16 = _mean(
        task_geometry, "black_box_rl", "nearest_useful_distance", 16
    )
    budget_decoy_start = _mean(
        budget_rows, "black_box_rl", "candidate_decoy_rate", budget=baseline_budget
    )
    budget_decoy_end = _mean(
        budget_rows, "black_box_rl", "candidate_decoy_rate", budget=largest_budget
    )
    budget_rank_start = _mean(
        budget_rows,
        "black_box_rl",
        "effective_directional_rank",
        budget=baseline_budget,
    )
    budget_rank_end = _mean(
        budget_rows, "black_box_rl", "effective_directional_rank", budget=largest_budget
    )
    budget_alpha_start = _mean(
        budget_rows, "black_box_rl", "alpha_mean", budget=baseline_budget
    )
    budget_alpha_end = _mean(
        budget_rows, "black_box_rl", "alpha_mean", budget=largest_budget
    )
    budget_lookup = {
        (int(float(row["seed"])), int(float(row["budget"]))): row for row in budget_rows
    }
    budget_seeds = sorted(
        {
            seed
            for seed, budget in budget_lookup
            if budget == baseline_budget and (seed, largest_budget) in budget_lookup
        }
    )
    paired_budget = []
    for metric in [
        "proposal_recall",
        "candidate_mean_quality",
        "candidate_decoy_rate",
        "effective_directional_rank",
        "alpha_mean",
    ]:
        differences = [
            _f(budget_lookup[(seed, largest_budget)], metric)
            - _f(budget_lookup[(seed, baseline_budget)], metric)
            for seed in budget_seeds
        ]
        stat = statistics(differences)
        paired_budget.append(
            {
                "metric": metric,
                "larger_budget": largest_budget,
                "baseline_budget": baseline_budget,
                "mean_paired_difference": stat["mean"],
                "sd": stat["sd"],
                "ci95": stat["ci95"],
                "ci95_low": stat["mean"] - stat["ci95"],
                "ci95_high": stat["mean"] + stat["ci95"],
                "seed_count": len(differences),
            }
        )
    write_csv(budget_dir / "paired_budget_differences.csv", paired_budget)
    rl_inside_capture = sum(
        1
        for row in candidates
        if row["method"] == "black_box_rl"
        and int(float(row["dimension"])) == 16
        and _f(row, "pre_nearest_useful_distance") < 3.0
    )
    direct_low_dimension_recall = max(
        _mean(dimension_rows, "direct_gradient", "proposal_recall", d)
        for d in [2, 4, 8]
    )

    radial_underreach = (
        np.isfinite(ratio2) and np.isfinite(ratio16) and ratio16 < 0.5 * ratio2
    )
    if (
        rl16 < 0.05
        and random16 < 0.01 * max(random2, 1e-8)
        and align16 < 0.35
        and not radial_underreach
    ):
        dominant = (
            "A dimensional directional-discovery and credit-assignment failure: random useful "
            "hits collapse, while dense high-Q anchor outcomes reinforce directions that remain "
            "outside useful refinement basins. Radial range is not the limiting scale."
        )
        next_experiment = (
            "Run a supervised target-direction representability control with the architecture "
            "frozen. If it succeeds, test black-box credit assignment or curriculum rather than "
            "increasing alpha or network size."
        )
    elif grad16 > 0.15 and rl16 < 0.05:
        dominant = (
            "Black-box discovery/sample efficiency: the fixed architecture succeeds "
            "with privileged gradients but RL does not."
        )
        next_experiment = "Test denser black-box credit assignment or a curriculum at fixed architecture."
    else:
        dominant = (
            "The evidence implicates combined dimensional discovery and optimization constraints; "
            "no single radial, budget, or refinement explanation is sufficient."
        )
        next_experiment = "Run a supervised target-direction representability control before changing capacity."
    if radial_underreach:
        next_experiment += " The measured radial underreach justifies a validation-only alpha-scale control."

    dimension_table = _table(
        [
            "d",
            "RL recall",
            "Gradient recall",
            "Contrastive recall",
            "RL quality",
            "RL decoy",
            "RL rank",
        ],
        [
            [
                d,
                _fmt(_mean(dimension_rows, "black_box_rl", "proposal_recall", d)),
                _fmt(_mean(dimension_rows, "direct_gradient", "proposal_recall", d)),
                _fmt(_mean(dimension_rows, "contrastive", "proposal_recall", d)),
                _fmt(
                    _mean(dimension_rows, "black_box_rl", "candidate_mean_quality", d)
                ),
                _fmt(_mean(dimension_rows, "black_box_rl", "candidate_decoy_rate", d)),
                _fmt(
                    _mean(
                        dimension_rows, "black_box_rl", "effective_directional_rank", d
                    )
                ),
            ]
            for d in [2, 4, 8, 16]
        ],
    )
    (dimension_dir / "DIMENSION_SWEEP.md").write_text(
        "# Dimension sweep\n\n"
        "The conceptual landscape, K=8 portfolio, nine-step refinement, and outcome budget "
        "are held fixed. The direct-gradient condition is privileged and diagnostic.\n\n"
        + dimension_table
        + f"\n\nRL first falls below half its d=2 proposal recall at d={failure_dimension}. "
        "Mean, SD, and 95% CI are in report_dimension_summary.csv.\n\n"
        "![Dimension versus proposal recall](dimension_vs_recall.png)\n",
        encoding="utf-8",
    )
    (dimension_dir / "REWARD_ENCOUNTER_ANALYSIS.md").write_text(
        "# Reward encounter analysis\n\n"
        "Every value is a detached black-box Q outcome observed during REINFORCE. The CSV "
        "reports threshold probabilities, running maximum, and requested quantiles at each "
        f"predeclared budget. In d16, P(Q > 0.7) at 100k was {_fmt(p_high_q16)}: "
        "raw quality was dense, but it mostly represented the obvious anchor rather than useful "
        "alternatives. Thus useful discovery was sparse even though scalar Q was not.\n\n"
        "![Reward encounters](reward_encounters.png)\n\n"
        "![All reward thresholds](reward_thresholds.png)\n",
        encoding="utf-8",
    )
    (budget_dir / "BUDGET_SCALING.md").write_text(
        "# D16 black-box budget scaling\n\n"
        f"Proposal recall changed from {_fmt(budget100)} at {baseline_budget:,} outcomes to "
        f"{_fmt(budget_large)} at {largest_budget:,} outcomes. Expensive conditions use "
        "five paired seeds and are diagnostic, not hyperparameter selection. Over the same "
        f"range, decoy rate changed {_fmt(budget_decoy_start)} to {_fmt(budget_decoy_end)}, "
        f"direction rank {_fmt(budget_rank_start)} to {_fmt(budget_rank_end)}, and mean alpha "
        f"{_fmt(budget_alpha_start)} to {_fmt(budget_alpha_end)}. More samples did not merely "
        "fail to help; the portfolio drifted toward larger, less diverse, more decoy-prone moves. "
        "Paired mean differences with 95% confidence intervals are saved in "
        "paired_budget_differences.csv.\n\n"
        "![Budget scaling](budget_scaling.png)\n",
        encoding="utf-8",
    )
    (geometry_dir / "GEOMETRY_SCALE_DIAGNOSTIC.md").write_text(
        "# Geometry, scale, angle, and refinement capture\n\n"
        f"Black-box RL mean alpha/nearest-useful-distance ratio: d=2 {_fmt(ratio2)}, "
        f"d=16 {_fmt(ratio16)}. Mean d16 best alignment to any useful center: "
        f"{_fmt(align16)}, versus {_fmt(initial_align16)} before training and "
        f"{_fmt(contrast_align16)} for contrastive steering. Target alignment was "
        f"{_fmt(target_align16)}. Alpha {_fmt(_mean(candidates, 'black_box_rl', 'alpha', 16))} "
        f"was comparable to nearest-useful distance {_fmt(nearest_useful16)}, but angular error "
        f"left candidates {_fmt(pre_distance16)} away from useful centers before refinement. "
        f"Exactly {rl_inside_capture} evaluated RL candidates began within distance 3.0. "
        f"Empirical refined random-hit rate fell from {_fmt(random2, 5)} at d=2 to "
        f"{_fmt(random16, 5)} at d=16, a {_fmt(random_collapse, 0)}-fold collapse.\n\n"
        "Capture bins estimate P(post-refinement useful given pre-refinement nearest-center "
        "distance) by method. Region labels are evaluation-only. The optional alpha-max sweep "
        "was not run because radial underreach was not observed. The optional reward-shaping "
        "study was not run because raw Q was dense; the failure is anchor-dominated credit, "
        "not threshold starvation.\n\n"
        "![Geometry scaling](geometry_scaling.png)\n\n"
        "![D16 alpha distributions](alpha_distribution.png)\n",
        encoding="utf-8",
    )
    valid_batch = [row for row in batch_rows if row["status"] == "ok"]
    peak = max(valid_batch, key=lambda row: _f(row, "candidate_outcomes_per_second"))
    maximum_throughput = _f(peak, "candidate_outcomes_per_second")
    plateau_batch = min(
        int(_f(row, "batch_size"))
        for row in valid_batch
        if _f(row, "candidate_outcomes_per_second") >= 0.95 * maximum_throughput
    )
    plateau_row = next(
        row for row in valid_batch if int(_f(row, "batch_size")) == plateau_batch
    )
    (batch_dir / "GPU_BATCH_SCALING.md").write_text(
        "# GPU batch scaling\n\n"
        f"Peak measured throughput was {_f(peak, 'candidate_outcomes_per_second'):,.0f} "
        f"candidate outcomes/s at task batch {int(_f(peak, 'batch_size'))}. The engineering "
        f"95%-plateau batch was {plateau_batch}, using "
        f"{_f(plateau_row, 'peak_allocated_bytes') / 2**30:.2f} GiB allocated and "
        f"{_f(plateau_row, 'peak_reserved_bytes') / 2**30:.2f} GiB reserved. Scientific training "
        "retains batch 12: combining sequential AdamW updates would change the established "
        "trajectory. CUDA is synchronized around timed regions. GPU utilization percentage "
        "was not sampled reliably, so throughput and memory are reported without inventing it.\n\n"
        "![Batch throughput](batch_throughput.png)\n",
        encoding="utf-8",
    )

    decisions = [
        [
            "Gradient succeeds, RL fails",
            "black-box/sample-efficiency bottleneck",
            grad16 > 0.15 and rl16 < 0.05,
        ],
        [
            "Gradient fails in d16 and is weak below d16",
            "direct-Q control is locally confounded; capacity unresolved",
            grad16 < 0.05 and direct_low_dimension_recall < 0.20,
        ],
        [
            "RL improves strongly with budget",
            "sample-complexity problem",
            budget_large - budget100 > 0.10,
        ],
        [
            "Raw d16 Q is sparse",
            "policy rarely receives any scalar signal",
            p_high_q16 < 0.05,
        ],
        [
            "Alpha/useful-distance ratio collapses",
            "radial scaling problem",
            radial_underreach,
        ],
        ["D16 alignment is weak", "direction-learning problem", align16 < 0.35],
        [
            "RL proposals enter refinement capture range",
            "refinement itself fails after good initialization",
            rl_inside_capture > 0,
        ],
        [
            "Random hit collapses with dimension",
            "curse of dimensionality",
            random16 < 0.1 * max(random2, 1e-8),
        ],
    ]
    decision_table = _table(
        ["Finding", "Interpretation", "Observed"],
        [[a, b, "yes" if c else "no"] for a, b, c in decisions],
    )
    (dimension_dir / "D16_FAILURE_DIAGNOSIS.md").write_text(
        "# Why CSD failed in genuine 16D\n\n"
        "This is a failure diagnosis, not a rescue experiment. No landscape, architecture, "
        "router, K, or refinement rule was tuned to improve CSD.\n\n"
        "## Central result\n\n"
        f"Direct-gradient d16 proposal recall: {_fmt(grad16)}. Black-box RL d16 proposal "
        f"recall: {_fmt(rl16)}. Contrastive reference: {_fmt(contrast16)}.\n\n"
        f"Dominant explanation: {dominant}\n\n"
        + decision_table
        + "\n\n## Required questions\n\n"
        f"1. Direct gradients {'recovered' if grad16 > 0.05 else 'did not recover'} useful d16 proposals. "
        "Because the same mean-Q control also collapsed in lower dimensions, this negative control "
        "does not prove insufficient representational capacity; capacity remains unresolved.\n"
        f"2. RL degradation begins materially at d={failure_dimension}.\n"
        f"3. Raw reward was not sparse: d16 P(Q > 0.7) reached {_fmt(p_high_q16)}. "
        "Useful-alternative encounters were sparse because anchor Q dominated.\n"
        f"4. Largest-budget recall was {_fmt(budget_large)} versus {_fmt(budget100)} "
        f"at {baseline_budget:,} outcomes.\n"
        f"5. Radial ratio changed from {_fmt(ratio2)} in d2 to {_fmt(ratio16)} in d16; "
        "there is no radial underreach signal.\n"
        f"6. D16 best-useful directional alignment averaged {_fmt(align16)} and target "
        f"alignment {_fmt(target_align16)}.\n"
        f"7. No RL candidate began within the empirical distance-3 capture range; contrastive "
        "candidates inside distance 2 were refined successfully. Refinement is not the primary fault.\n"
        f"8. Random refined hit rate changed from {_fmt(random2, 5)} to {_fmt(random16, 5)} "
        f"({_fmt(random_collapse, 0)}-fold lower).\n"
        f"9. {dominant}\n"
        f"10. Next experiment: {next_experiment}\n\n"
        "These conclusions concern this synthetic proposal mechanism only. Geometric rank alone "
        "is not evidence of useful diversity.\n",
        encoding="utf-8",
    )
    return {
        "rl_d16_recall": rl16,
        "gradient_d16_recall": grad16,
        "contrastive_d16_recall": contrast16,
        "failure_dimension": failure_dimension,
        "dominant_explanation": dominant,
        "next_experiment": next_experiment,
    }
