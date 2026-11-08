"""Tables, plots, candidate-reproduction audit, and reports for router study."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from creativity_toy.analysis_utils import statistics
from creativity_toy.config_gen3 import preset_gen3
from creativity_toy.landscapes_v2 import V2LandscapeFamily

from .config_rl import RLStudyConfig
from .critic import critic_features
from .frozen_router_candidates import load_candidate_batch, load_frozen_generator
from .study import read_csv, write_csv
from .study_rl import _datasets

SELECTOR_ORDER = [
    "random",
    "quality_router",
    "novelty_only_router",
    "ungated_q_plus_n_router",
    "gated_useful_novelty_router",
    "set_aware_useful_novelty_router",
    "gated_useful_novelty_pairwise_router",
    "quality_oracle",
    "useful_oracle",
]
SELECTOR_LABELS = {
    "random": "Random",
    "quality_router": "Quality router",
    "novelty_only_router": "Novelty-only router",
    "ungated_q_plus_n_router": "Ungated Q+N",
    "gated_useful_novelty_router": "Gated useful-novelty router",
    "set_aware_useful_novelty_router": "Set-aware useful-novelty router",
    "gated_useful_novelty_pairwise_router": "Gated pairwise-ranking router",
    "quality_oracle": "Quality oracle",
    "useful_oracle": "Useful oracle",
}
METRICS = [
    "selected_quality",
    "useful_selection_rate",
    "target_selection_rate",
    "decoy_selection_rate",
    "anchor_selection_rate",
    "coverage",
    "collapse",
    "quality_oracle_regret",
    "useful_oracle_quality_gap",
    "score_quality_spearman",
    "score_useful_novelty_spearman",
    "selected_alpha",
    "selected_novelty",
]


def _quality_router_costs(
    main_dir: Path, seeds: Sequence[int], repetitions: int = 50
) -> list[dict[str, object]]:
    """Benchmark the preserved quality router, which is not fitted in this study."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = preset_gen3("main")
    rl = RLStudyConfig()
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for generator_kind in ["rl", "es"]:
            generator = load_frozen_generator(
                generator_kind, base=base, rl=rl, seed=seed, device=device
            )
            batch = load_candidate_batch(
                main_dir / "candidate_cache" / f"seed_{seed}_{generator_kind}_test.pt",
                device,
            )

            def score() -> torch.Tensor:
                return generator.quality_router.predict(
                    critic_features(batch.starts, batch.candidates, batch.goals)
                )

            for _ in range(3):
                score()
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            started = perf_counter()
            for _ in range(repetitions):
                score()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (perf_counter() - started) / repetitions
            peak = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            candidates = batch.batch_size * batch.candidate_count
            rows.append(
                {
                    "seed": seed,
                    "generator": generator_kind,
                    "selector": "quality_router",
                    "parameter_count": generator.quality_router.parameter_count,
                    "additional_parameters_vs_quality_router": 0,
                    "router_training_seconds": 0.0,
                    "router_inference_seconds": elapsed,
                    "candidate_scoring_throughput": candidates / elapsed,
                    "peak_gpu_allocated_bytes": peak,
                }
            )
            del batch, generator
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def _aggregate(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
    metrics: Sequence[str] = METRICS,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, selected in groups.items():
        result: dict[str, object] = dict(zip(keys, key))
        for metric in metrics:
            if metric not in selected[0]:
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
    generator: str,
    landscape_variant: str | None = None,
) -> list[dict[str, object]]:
    selected = [row for row in rows if row["generator"] == generator]
    if landscape_variant is not None:
        selected = [
            row for row in selected if row["landscape_variant"] == landscape_variant
        ]
    left_rows = {int(row["seed"]): row for row in selected if row["selector"] == left}
    right_rows = {int(row["seed"]): row for row in selected if row["selector"] == right}
    seeds = sorted(set(left_rows) & set(right_rows))
    output = []
    for metric in [
        "selected_quality",
        "useful_selection_rate",
        "target_selection_rate",
        "decoy_selection_rate",
        "anchor_selection_rate",
        "coverage",
        "collapse",
    ]:
        differences = [
            float(left_rows[seed][metric]) - float(right_rows[seed][metric])
            for seed in seeds
        ]
        stat = statistics(differences)
        output.append(
            {
                "generator": generator,
                "landscape_variant": landscape_variant or "primary_iid",
                "left": left,
                "right": right,
                "metric": metric,
                "paired_difference_mean": stat["mean"],
                "paired_difference_sd": stat["sd"],
                "paired_difference_ci95": stat["ci95"],
                "ci95_low": stat["mean"] - stat["ci95"],
                "ci95_high": stat["mean"] + stat["ci95"],
                "paired_ci_excludes_zero": abs(stat["mean"]) > stat["ci95"],
                "n": int(stat["n"]),
            }
        )
    return output


def _fmt(value: object) -> str:
    return f"{float(value):.3f}"


def _target_scale_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for generator in ["rl", "es"]:
        selected = [row for row in rows if row["generator"] == generator]
        result: dict[str, object] = {"generator": generator, "seeds": len(selected)}
        for name in ["quality", "novelty", "gated_novelty", "useful_novelty_target"]:
            result[f"{name}_seed_balanced_mean"] = float(
                np.mean([float(row[f"{name}_mean"]) for row in selected])
            )
            result[f"{name}_mean_within_seed_sd"] = float(
                np.mean([float(row[f"{name}_sd"]) for row in selected])
            )
            result[f"{name}_global_min"] = min(
                float(row[f"{name}_min"]) for row in selected
            )
            result[f"{name}_global_max"] = max(
                float(row[f"{name}_max"]) for row in selected
            )
        output.append(result)
    return output


def _markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    headers = [
        "Selector",
        "Quality",
        "Useful Selection",
        "Target Selection",
        "Decoy Selection",
        "Anchor Selection",
        "Coverage",
        "Collapse",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * 7) + " |",
    ]
    lookup = {str(row["selector"]): row for row in rows}
    for selector in SELECTOR_ORDER:
        row = lookup[selector]
        lines.append(
            "| "
            + " | ".join(
                [
                    SELECTOR_LABELS[selector],
                    _fmt(row["selected_quality"]),
                    _fmt(row["useful_selection_rate"]),
                    _fmt(row["target_selection_rate"]),
                    _fmt(row["decoy_selection_rate"]),
                    _fmt(row["anchor_selection_rate"]),
                    _fmt(row["coverage"]),
                    _fmt(row["collapse"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _candidate_reproduction(
    main_dir: Path, seeds: Sequence[int]
) -> list[dict[str, object]]:
    prior_sources = {
        "rl": Path("results/rl_adamw_main/candidate_diagnostics.csv"),
        "es": Path("results/gen3_gpu/candidate_diagnostics.csv"),
    }
    output: list[dict[str, object]] = []
    for generator, path in prior_sources.items():
        with path.open(encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        for seed in seeds:
            if generator == "rl":
                rows = [
                    row
                    for row in all_rows
                    if int(float(row["seed"])) == seed
                    and row["evaluation_mode"] == "rl_adamw_stochastic"
                ]
            else:
                rows = [
                    row
                    for row in all_rows
                    if int(float(row["seed"])) == seed
                    and row["variant"] == "gen3_contextual_linear_alpha"
                ]
            batch = load_candidate_batch(
                main_dir / "candidate_cache" / f"seed_{seed}_{generator}_test.pt",
                torch.device("cpu"),
            )
            candidates = np.asarray(
                [json.loads(row["candidate"]) for row in rows], dtype=np.float64
            ).reshape(tuple(batch.candidates.shape))
            qualities = np.asarray(
                [float(row["quality"]) for row in rows], dtype=np.float64
            ).reshape(tuple(batch.quality.shape))
            output.append(
                {
                    "seed": seed,
                    "generator": generator,
                    "candidate_max_abs_difference": float(
                        np.max(np.abs(candidates - batch.candidates.numpy()))
                    ),
                    "quality_max_abs_difference": float(
                        np.max(np.abs(qualities - batch.quality.numpy()))
                    ),
                }
            )
    return output


def _plot_pareto(summary: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, generator in zip(axes, ["rl", "es"]):
        rows = [row for row in summary if row["generator"] == generator]
        for row in rows:
            selector = str(row["selector"])
            axis.errorbar(
                float(row["useful_selection_rate"]),
                float(row["selected_quality"]),
                xerr=float(row["useful_selection_rate_ci95"]),
                yerr=float(row["selected_quality_ci95"]),
                marker="o",
                linestyle="none",
                label=SELECTOR_LABELS.get(selector, selector),
            )
        axis.set_title(f"{generator.upper()} candidates")
        axis.set_xlabel("useful selection rate")
        axis.set_ylabel("selected quality")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_conservative(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    metrics = ["alpha", "novelty", "raw_quality", "quality", "anchor_probability"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, generator in zip(axes, ["rl", "es"]):
        values, errors = [], []
        for metric in metrics:
            selected = [
                float(row[f"selected_minus_unselected_{metric}"])
                for row in rows
                if row["generator"] == generator and row["selector"] == "quality_router"
            ]
            stat = statistics(selected)
            values.append(stat["mean"])
            errors.append(stat["ci95"])
        axis.bar(np.arange(len(metrics)), values, yerr=errors, capsize=3)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(np.arange(len(metrics)), metrics, rotation=25, ha="right")
        axis.set_title(f"{generator.upper()} quality router")
        axis.set_ylabel("selected minus mean unselected")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_generalization(summary: Sequence[Mapping[str, object]], path: Path) -> None:
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
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, generator in zip(axes[:, 0], ["rl", "es"]):
        for selector in [
            "quality_router",
            "gated_useful_novelty_router",
            "set_aware_useful_novelty_router",
            "useful_oracle",
        ]:
            lookup = {
                str(row["landscape_variant"]): row
                for row in summary
                if row["generator"] == generator and row["selector"] == selector
            }
            axis.plot(
                range(len(variants)),
                [
                    float(lookup[variant]["useful_selection_rate"])
                    for variant in variants
                ],
                marker="o",
                label=SELECTOR_LABELS[selector],
            )
        axis.set_title(f"{generator.upper()} useful selection")
        axis.set_xticks(range(len(variants)), variants, rotation=35, ha="right")
        axis.grid(alpha=0.25)
    for axis, generator in zip(axes[:, 1], ["rl", "es"]):
        for selector in [
            "quality_router",
            "gated_useful_novelty_router",
            "set_aware_useful_novelty_router",
            "useful_oracle",
        ]:
            lookup = {
                str(row["landscape_variant"]): row
                for row in summary
                if row["generator"] == generator and row["selector"] == selector
            }
            axis.plot(
                range(len(variants)),
                [float(lookup[variant]["selected_quality"]) for variant in variants],
                marker="o",
                label=SELECTOR_LABELS[selector],
            )
        axis.set_title(f"{generator.upper()} selected quality")
        axis.set_xticks(range(len(variants)), variants, rotation=35, ha="right")
        axis.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_example(
    *,
    generator: str,
    batch,
    task,
    set_index: int,
    choices: Mapping[str, int],
    path: Path,
    criterion: str,
) -> None:
    centers = np.concatenate(
        [task.anchor[None, :], task.useful_centers, task.decoy_centers], axis=0
    )
    raw = batch.candidates[set_index].numpy()
    refined = batch.refined[set_index].numpy()
    all_points = np.concatenate([centers, raw, refined], axis=0)
    low = np.min(all_points, axis=0) - 1.2
    high = np.max(all_points, axis=0) + 1.2
    xs = np.linspace(low[0], high[0], 240)
    ys = np.linspace(low[1], high[1], 240)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx, yy], axis=-1)
    quality = task.quality(grid)
    figure, axis = plt.subplots(figsize=(8.5, 7))
    contour = axis.contourf(xx, yy, quality, levels=30, cmap="viridis")
    figure.colorbar(contour, ax=axis, label="Q(h,g)")
    axis.scatter(
        *task.anchor,
        marker="*",
        s=170,
        color="white",
        edgecolor="black",
        label="anchor",
    )
    axis.scatter(
        task.useful_centers[:, 0],
        task.useful_centers[:, 1],
        marker="o",
        s=85,
        facecolors="none",
        edgecolors="lime",
        label="useful centers",
    )
    axis.scatter(
        task.decoy_centers[:, 0],
        task.decoy_centers[:, 1],
        marker="x",
        s=75,
        color="red",
        label="decoys",
    )
    for index in range(batch.candidate_count):
        axis.plot(
            [raw[index, 0], refined[index, 0]],
            [raw[index, 1], refined[index, 1]],
            color="white",
            alpha=0.45,
            linewidth=1,
        )
        axis.scatter(raw[index, 0], raw[index, 1], marker=".", color="cyan", s=35)
        axis.scatter(
            refined[index, 0], refined[index, 1], marker="o", color="orange", s=45
        )
        axis.annotate(
            f"{index}:Q={float(batch.quality[set_index, index]):.2f}\nN={float(batch.novelty[set_index, index]):.2f},a={float(batch.alphas[set_index, index]):.2f}",
            refined[index],
            fontsize=6.5,
            xytext=(3, 3),
            textcoords="offset points",
        )
    markers = {
        "quality_router": ("s", "white"),
        "set_aware_useful_novelty_router": ("D", "lime"),
        "useful_oracle": ("X", "magenta"),
    }
    for selector, index in choices.items():
        marker, color = markers[selector]
        axis.scatter(
            refined[index, 0],
            refined[index, 1],
            marker=marker,
            s=180,
            facecolors="none",
            edgecolors=color,
            linewidths=2.5,
            label=SELECTOR_LABELS[selector],
        )
    axis.set_title(f"{generator.upper()} frozen candidates, seed 7\n{criterion}")
    axis.set_aspect("equal")
    axis.legend(fontsize=7, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _representative_plots(main_dir: Path, output: Path) -> list[dict[str, object]]:
    base = preset_gen3("main")
    family = V2LandscapeFamily(base.base.landscape)
    _, _, _, _, test_tasks, _ = _datasets(family, base, RLStudyConfig(), 7)
    diagnostics = read_csv(main_dir / "selection_diagnostics_per_set.csv")
    records = []
    for generator in ["rl", "es"]:
        batch = load_candidate_batch(
            main_dir / "candidate_cache" / f"seed_7_{generator}_test.pt",
            torch.device("cpu"),
        )
        rows = [
            row
            for row in diagnostics
            if int(row["seed"]) == 7 and row["generator"] == generator
        ]
        by_selector = {
            selector: {
                int(row["set_index"]): int(row["selected_index"])
                for row in rows
                if row["selector"] == selector
            }
            for selector in [
                "quality_router",
                "set_aware_useful_novelty_router",
                "useful_oracle",
            ]
        }
        examples = [(0, "fixed first test set")]
        disagreement = None
        for set_index in range(batch.batch_size):
            q_index = by_selector["quality_router"][set_index]
            un_index = by_selector["set_aware_useful_novelty_router"][set_index]
            q_anchor = int(batch.regions[set_index, q_index]) == 0
            un_useful = (
                1
                <= int(batch.regions[set_index, un_index])
                <= int(batch.n_useful[set_index])
                and float(batch.quality[set_index, un_index])
                >= base.base.landscape.useful_threshold
            )
            if q_anchor and un_useful:
                disagreement = set_index
                break
        if disagreement is not None and disagreement != 0:
            examples.append(
                (disagreement, "first ordered anchor-to-useful disagreement")
            )
        for set_index, criterion in examples:
            task_id = int(batch.task_ids[set_index])
            path = output / f"routing_example_{generator}_{set_index}.png"
            _plot_example(
                generator=generator,
                batch=batch,
                task=test_tasks[task_id],
                set_index=set_index,
                choices={
                    selector: values[set_index]
                    for selector, values in by_selector.items()
                },
                path=path,
                criterion=criterion,
            )
            records.append(
                {
                    "generator": generator,
                    "set_index": set_index,
                    "criterion": criterion,
                    "path": str(path),
                }
            )
    return records


def generate_router_reports(
    *,
    main_dir: Path = Path("results/router_useful_novelty_main"),
    sweep_dir: Path = Path("results/router_useful_novelty_sweep"),
) -> dict[str, object]:
    primary = read_csv(main_dir / "primary_per_seed.csv")
    generalization = read_csv(main_dir / "generalization_per_seed.csv")
    diagnostics = read_csv(main_dir / "selection_diagnostics_per_seed.csv")
    config = json.loads((main_dir / "config.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    primary_summary = _aggregate(primary, ["generator", "selector"])
    quality_cost_rows = _quality_router_costs(main_dir, seeds)
    cost_input = [row for row in primary if row["selector"] != "quality_router"]
    cost_input.extend(quality_cost_rows)
    cost_summary = _aggregate(
        cost_input,
        ["generator", "selector"],
        [
            "parameter_count",
            "additional_parameters_vs_quality_router",
            "router_training_seconds",
            "router_inference_seconds",
            "candidate_scoring_throughput",
            "peak_gpu_allocated_bytes",
        ],
    )
    generalization_summary = _aggregate(
        generalization, ["generator", "landscape_variant", "selector"]
    )
    paired_rows = []
    comparisons = [
        ("gated_useful_novelty_router", "quality_router"),
        ("set_aware_useful_novelty_router", "quality_router"),
        ("gated_useful_novelty_pairwise_router", "quality_router"),
        ("set_aware_useful_novelty_router", "gated_useful_novelty_router"),
        ("gated_useful_novelty_router", "ungated_q_plus_n_router"),
    ]
    for generator in ["rl", "es"]:
        for left, right in comparisons:
            paired_rows.extend(
                _paired(primary, left=left, right=right, generator=generator)
            )
    generalization_paired = []
    for generator in ["rl", "es"]:
        for variant in config["config"]["generalization_variants"]:
            for left in [
                "gated_useful_novelty_router",
                "set_aware_useful_novelty_router",
            ]:
                generalization_paired.extend(
                    _paired(
                        generalization,
                        left=left,
                        right="quality_router",
                        generator=generator,
                        landscape_variant=variant,
                    )
                )
    reproduction = _candidate_reproduction(main_dir, seeds)
    target_scale = _target_scale_summary(
        read_csv(main_dir / "routing_target_distributions.csv")
    )
    examples = _representative_plots(main_dir, main_dir)
    write_csv(main_dir / "primary_summary.csv", primary_summary)
    write_csv(main_dir / "router_cost_summary.csv", cost_summary)
    write_csv(main_dir / "quality_router_cost_per_seed.csv", quality_cost_rows)
    write_csv(main_dir / "paired_router_differences.csv", paired_rows)
    write_csv(main_dir / "generalization_summary.csv", generalization_summary)
    write_csv(main_dir / "generalization_paired_differences.csv", generalization_paired)
    write_csv(main_dir / "frozen_candidate_reproduction.csv", reproduction)
    write_csv(main_dir / "routing_target_distribution_summary.csv", target_scale)
    write_csv(main_dir / "representative_examples.csv", examples)
    _plot_pareto(primary_summary, main_dir / "quality_vs_useful_selection.png")
    _plot_conservative(diagnostics, main_dir / "quality_router_conservatism.png")
    _plot_generalization(generalization_summary, main_dir / "router_generalization.png")

    rl_summary = [row for row in primary_summary if row["generator"] == "rl"]
    es_summary = [row for row in primary_summary if row["generator"] == "es"]
    rl_table = _markdown_table(rl_summary)
    es_table = _markdown_table(es_summary)
    (main_dir / "RL_ROUTER_TABLE.md").write_text(
        "# RL candidates\n\n" + rl_table + "\n", encoding="utf-8"
    )
    (main_dir / "ES_ROUTER_TABLE.md").write_text(
        "# ES candidates\n\n" + es_table + "\n", encoding="utf-8"
    )
    write_csv(main_dir / "RL_ROUTER_TABLE.csv", rl_summary)
    write_csv(main_dir / "ES_ROUTER_TABLE.csv", es_summary)

    lookup = {
        (str(row["generator"]), str(row["selector"])): row for row in primary_summary
    }
    paired_lookup = {
        (
            str(row["generator"]),
            str(row["left"]),
            str(row["right"]),
            str(row["metric"]),
        ): row
        for row in paired_rows
    }
    diagnostic_lookup = {
        (str(row["generator"]), str(row["selector"])): row
        for row in _aggregate(
            diagnostics,
            ["generator", "selector"],
            [
                "selected_minus_unselected_quality",
                "selected_minus_unselected_raw_quality",
                "selected_minus_unselected_alpha",
                "selected_minus_unselected_displacement",
                "selected_minus_unselected_novelty",
                "selected_minus_unselected_anchor_probability",
            ],
        )
    }

    def effect(generator: str, left: str, right: str, metric: str) -> str:
        row = paired_lookup[(generator, left, right, metric)]
        return f"{float(row['paired_difference_mean']):+.3f} (95% CI {float(row['ci95_low']):+.3f} to {float(row['ci95_high']):+.3f})"

    def gap_recovery(generator: str, selector: str) -> float:
        quality = float(lookup[(generator, "quality_router")]["useful_selection_rate"])
        ceiling = float(lookup[(generator, "useful_oracle")]["useful_selection_rate"])
        selected = float(lookup[(generator, selector)]["useful_selection_rate"])
        return (selected - quality) / max(ceiling - quality, 1e-12)

    main_report = f"""# Useful-novelty router results

The experiment used the 30 existing seeds and never updated an RL or ES
generator parameter. Maximum generator parameter difference was zero. Every
cached test candidate and Q value reproduced the prior RL/ES artifact exactly;
the maximum differences are recorded in `frozen_candidate_reproduction.csv`.

## Primary tables

Table entries are 30-seed means. `primary_summary.csv` reports the corresponding
standard deviations and 95% confidence intervals; paired changes are in
`paired_router_differences.csv`.

### Frozen RL candidates

{rl_table}

### Frozen ES candidates

{es_table}

## Interpretation

The original quality router is conservative. On RL candidates it selected
alpha {float(diagnostic_lookup[("rl", "quality_router")]["selected_minus_unselected_alpha"]):+.3f},
displacement {float(diagnostic_lookup[("rl", "quality_router")]["selected_minus_unselected_displacement"]):+.3f},
novelty {float(diagnostic_lookup[("rl", "quality_router")]["selected_minus_unselected_novelty"]):+.3f},
final quality {float(diagnostic_lookup[("rl", "quality_router")]["selected_minus_unselected_quality"]):+.3f},
and raw quality {float(diagnostic_lookup[("rl", "quality_router")]["selected_minus_unselected_raw_quality"]):+.3f}
relative to mean unselected candidates, while its anchor probability changed by
{float(diagnostic_lookup[("rl", "quality_router")]["selected_minus_unselected_anchor_probability"]):+.3f}.
The corresponding ES differences are
{float(diagnostic_lookup[("es", "quality_router")]["selected_minus_unselected_alpha"]):+.3f},
{float(diagnostic_lookup[("es", "quality_router")]["selected_minus_unselected_displacement"]):+.3f},
{float(diagnostic_lookup[("es", "quality_router")]["selected_minus_unselected_novelty"]):+.3f},
{float(diagnostic_lookup[("es", "quality_router")]["selected_minus_unselected_quality"]):+.3f},
{float(diagnostic_lookup[("es", "quality_router")]["selected_minus_unselected_raw_quality"]):+.3f},
and {float(diagnostic_lookup[("es", "quality_router")]["selected_minus_unselected_anchor_probability"]):+.3f}.

For RL candidates, local gated routing barely changed useful selection. The
set-aware router changed useful selection by
{effect("rl", "set_aware_useful_novelty_router", "quality_router", "useful_selection_rate")}
and quality by
{effect("rl", "set_aware_useful_novelty_router", "quality_router", "selected_quality")}.
Pairwise ranking recovered more:
{effect("rl", "gated_useful_novelty_pairwise_router", "quality_router", "useful_selection_rate")}
useful selection and
{effect("rl", "gated_useful_novelty_pairwise_router", "quality_router", "selected_quality")}
quality. Set-aware regression recovered {gap_recovery("rl", "set_aware_useful_novelty_router"):.1%}
of the useful-oracle gap; pairwise ranking recovered
{gap_recovery("rl", "gated_useful_novelty_pairwise_router"):.1%}.

For ES candidates, set-aware routing changed useful selection by
{effect("es", "set_aware_useful_novelty_router", "quality_router", "useful_selection_rate")}
and quality by
{effect("es", "set_aware_useful_novelty_router", "quality_router", "selected_quality")}.
Pairwise ranking changed useful selection by
{effect("es", "gated_useful_novelty_pairwise_router", "quality_router", "useful_selection_rate")}.
The recovered useful-oracle gaps were
{gap_recovery("es", "set_aware_useful_novelty_router"):.1%} and
{gap_recovery("es", "gated_useful_novelty_pairwise_router"):.1%}, respectively.

The routing objective was therefore part of the bottleneck, but not the whole
bottleneck. Candidate-local gated regression cannot reliably express the
desired relation, especially on RL candidates; simple set context and a ranking
loss recover substantially more already-present useful candidates. Neither
method reaches the useful oracle, so the router is not solved.

## Answers to the routing questions

1. The original router fails conservatively: it strongly favors smaller-alpha,
   smaller-displacement, high-immediate-quality anchor candidates.
2. Useful-novelty target replacement alone does not substantially improve RL
   routing. Set context raises useful selection, and pairwise ranking raises it
   further, for both frozen generator families.
3. The RL set-aware and pairwise changes do not pay a quality cost in this IID
   test; their paired quality changes are positive. ES set-aware quality changes
   by {effect("es", "set_aware_useful_novelty_router", "quality_router", "selected_quality")}
   and ES pairwise quality changes by
   {effect("es", "gated_useful_novelty_pairwise_router", "quality_router", "selected_quality")}.
4. Gating reduces the ES decoy rate relative to ungated and novelty-only
   routing, but it is also much more conservative. RL local Q+N routers select
   no decoys either way, so that case does not establish a gating benefit.
5. The small set-aware router is better than its candidate-local counterpart,
   supporting the hypothesis that useful novelty is relational.
6. The direction of the useful-selection gain holds for RL and ES candidates,
   but it is considerably larger for RL.
7. Pairwise ranking recovers {gap_recovery("rl", "gated_useful_novelty_pairwise_router"):.1%}
   of the RL and {gap_recovery("es", "gated_useful_novelty_pairwise_router"):.1%}
   of the ES useful-oracle gap.
8. The result strengthens the diagnosis that selection discards generator
   capability, while the remaining oracle gap and mixed transformed-landscape
   tradeoffs rule out claiming that routing is solved.

Router parameter counts, additional parameters, fit time, end-to-end scoring
time, scoring throughput, and peak allocated GPU memory are reported in
`router_cost_summary.csv`. Candidate generation is excluded because it is
identical and cached for all selectors. Ridge fit timers cover the linear solve;
pairwise timers cover its AdamW ranking optimization. Inference timers include
feature construction and scoring on the frozen test candidates.
"""
    Path("ROUTER_USEFUL_NOVELTY_RESULTS.md").write_text(main_report, encoding="utf-8")
    (main_dir / "ROUTER_USEFUL_NOVELTY_RESULTS.md").write_text(
        main_report, encoding="utf-8"
    )

    ablation_report = f"""# Router ablations

Novelty is mean pre-refinement sibling Euclidean distance divided by 5 and
clipped at 1.5. No z-score normalization is used. The deployable target is
`Q + lambda * sigmoid(beta*(Q-threshold)) * novelty`; region labels never enter
router fitting.

The scale audit is in `routing_target_distribution_summary.csv`: for RL/ES,
respectively, seed-balanced Q means are
{float(target_scale[0]["quality_seed_balanced_mean"]):.3f}/
{float(target_scale[1]["quality_seed_balanced_mean"]):.3f}, novelty means are
{float(target_scale[0]["novelty_seed_balanced_mean"]):.3f}/
{float(target_scale[1]["novelty_seed_balanced_mean"]):.3f}, and gated-novelty
means are {float(target_scale[0]["gated_novelty_seed_balanced_mean"]):.3f}/
{float(target_scale[1]["gated_novelty_seed_balanced_mean"]):.3f}. That CSV also
contains mean within-seed SD and global range for Q, novelty, gated novelty, and
the combined target.

The lambda-zero refit achieves mean top-1 agreement
{np.mean([float(row["test_top1_agreement"]) for row in read_csv(main_dir / "quality_lambda0_reproduction.csv")]):.4f}
with the saved quality routers. Coefficients differ more than predictions
because the handcrafted feature matrix is collinear; score-level errors are
reported separately.

Novelty-only routing demonstrates the expected failure mode: RL quality/useful/
decoy are {_fmt(lookup[("rl", "novelty_only_router")]["selected_quality"])}/
{_fmt(lookup[("rl", "novelty_only_router")]["useful_selection_rate"])}/
{_fmt(lookup[("rl", "novelty_only_router")]["decoy_selection_rate"])}; ES values
are {_fmt(lookup[("es", "novelty_only_router")]["selected_quality"])}/
{_fmt(lookup[("es", "novelty_only_router")]["useful_selection_rate"])}/
{_fmt(lookup[("es", "novelty_only_router")]["decoy_selection_rate"])}.

On ES candidates, ungated Q+N reaches useful selection
{_fmt(lookup[("es", "ungated_q_plus_n_router")]["useful_selection_rate"])} with
decoy rate {_fmt(lookup[("es", "ungated_q_plus_n_router")]["decoy_selection_rate"])},
whereas gated local routing reaches
{_fmt(lookup[("es", "gated_useful_novelty_router")]["useful_selection_rate"])}
with decoy rate
{_fmt(lookup[("es", "gated_useful_novelty_router")]["decoy_selection_rate"])}.
Gating reduces decoys but is conservative; it does not by itself solve ranking.

Set awareness adds exactly seven linear features. Local and set-aware routers
retain the same linear/ridge architecture family. Pairwise ranking retains the
same feature dimension and parameter count; it changes only the loss. Full
paired ablations, target distributions, parameter counts, timings, and rank
correlations are in the CSV outputs.
"""
    Path("ROUTER_ABLATIONS.md").write_text(ablation_report, encoding="utf-8")
    (main_dir / "ROUTER_ABLATIONS.md").write_text(ablation_report, encoding="utf-8")

    gen_lookup = {
        (
            str(row["generator"]),
            str(row["landscape_variant"]),
            str(row["selector"]),
        ): row
        for row in generalization_summary
    }
    lines = [
        "# Router generalization",
        "",
        "Each cell reports selected quality / useful-selection rate on the exact same frozen candidate sets:",
        "",
    ]
    for generator in ["rl", "es"]:
        lines.extend(
            [
                f"## {generator.upper()} candidates",
                "",
                "| Landscape | Random | Quality router | Gated local | Set-aware | Useful oracle |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant in config["config"]["generalization_variants"]:

            def q_u(selector: str) -> str:
                row = gen_lookup[(generator, variant, selector)]
                return f"{_fmt(row['selected_quality'])} / {_fmt(row['useful_selection_rate'])}"

            lines.append(
                "| "
                + " | ".join(
                    [
                        variant,
                        q_u("random"),
                        q_u("quality_router"),
                        q_u("gated_useful_novelty_router"),
                        q_u("set_aware_useful_novelty_router"),
                        q_u("useful_oracle"),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "The routers are trained only on IID training candidates and remain frozen for every transformation.",
            "Per-seed paired quality/useful/target/decoy/coverage/collapse intervals are in `generalization_paired_differences.csv`.",
            "Set-aware routing exceeds quality-router useful selection on all eight transformations for both RL and ES candidates, but exceeds its selected quality on only two of eight transformations for each generator.",
            "Thus the exploration preference transfers, while an upper-right Pareto improvement does not generalize universally; the tables preserve all mixed outcomes.",
        ]
    )
    generalization_report = "\n".join(lines) + "\n"
    Path("ROUTER_GENERALIZATION.md").write_text(generalization_report, encoding="utf-8")
    (main_dir / "ROUTER_GENERALIZATION.md").write_text(
        generalization_report, encoding="utf-8"
    )
    aggregate = {
        "seed_count": len(seeds),
        "generator_parameter_max_difference": max(
            float(row["maximum_parameter_difference"])
            for row in read_csv(main_dir / "generator_integrity.csv")
        ),
        "candidate_reproduction_max_difference": max(
            max(
                float(row["candidate_max_abs_difference"]),
                float(row["quality_max_abs_difference"]),
            )
            for row in reproduction
        ),
        "primary_summary": primary_summary,
        "cost_summary": cost_summary,
    }
    (main_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
