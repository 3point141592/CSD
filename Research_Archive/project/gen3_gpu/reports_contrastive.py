"""Tables, confidence intervals, plots, and reports for contrastive steering."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from creativity_toy.config_gen3 import preset_gen3
from creativity_toy.landscapes_v2 import V2LandscapeFamily

from .config_contrastive import ContrastiveExperimentConfig
from .config_rl import RLStudyConfig
from .config_router import RouterExperimentConfig
from .contrastive_evaluation import flatten_starts, make_evaluated_batch
from .contrastive_steering import build_contrastive_vectors, contrastive_candidates
from .frozen_router_candidates import generate_frozen_candidates, load_frozen_generator
from .study import read_csv, write_csv
from .study_contrastive import _datasets

METRICS = [
    "candidate_mean_quality",
    "selected_quality",
    "proposal_recall",
    "useful_region_count",
    "useful_region_count_2_plus",
    "useful_selected",
    "candidate_decoy_rate",
    "candidate_target_rate",
    "selected_decoy_rate",
    "routing_recall",
    "direction_cosine_distance",
    "candidate_spatial_diversity",
    "evaluation_seconds",
    "candidate_evaluations",
    "q_equivalent_evaluations",
]

DISPLAY = {
    "random": "Random",
    "pure_divergence": "Pure divergence",
    "contrast_global_1": "Contrastive global",
    "contrast_goal_1": "Contrastive goal",
    "contrast_target_anchor_1": "Contrastive target-anchor",
    "contrast_useful_decoy_1": "Contrastive useful-decoy",
    "contrast_global_k": "Contrastive global K-radius",
    "contrast_goal_k": "Contrastive K-radius",
    "contrast_target_anchor_k": "Contrastive target-anchor K",
    "contrast_useful_decoy_k": "Contrastive useful-decoy K",
    "csd_rl": "CSD RL",
    "csd_mode": "CSD + mode preservation",
    "csd_es": "CSD ES",
    "contrast_goal_k_recomputed_rotated": "Contrastive K, rotated retraining",
}


def _aggregate(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
    metrics: Sequence[str] = METRICS,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group, values in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        result = {key: value for key, value in zip(keys, group)}
        result["n"] = len(values)
        for metric in metrics:
            if metric not in values[0]:
                continue
            array = np.asarray([float(row[metric]) for row in values], dtype=np.float64)
            result[metric] = float(np.mean(array))
            result[f"{metric}_sd"] = float(np.std(array, ddof=0))
            result[f"{metric}_ci95"] = (
                float(1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))
                if len(array) > 1
                else 0.0
            )
        output.append(result)
    return output


def _paired(
    rows: Sequence[Mapping[str, object]],
    *,
    left: str,
    right: str,
    metrics: Sequence[str],
    landscape_variant: str | None = None,
) -> list[dict[str, object]]:
    active = [
        row
        for row in rows
        if landscape_variant is None
        or row.get("landscape_variant") == landscape_variant
    ]
    lookup = {(int(row["seed"]), str(row["method"])): row for row in active}
    seeds = sorted(
        {seed for seed, method in lookup if method == left}
        & {seed for seed, method in lookup if method == right}
    )
    output = []
    for metric in metrics:
        differences = np.asarray(
            [
                float(lookup[(seed, left)][metric])
                - float(lookup[(seed, right)][metric])
                for seed in seeds
            ]
        )
        mean = float(np.mean(differences))
        half = float(1.96 * np.std(differences, ddof=1) / np.sqrt(len(differences)))
        output.append(
            {
                "left": left,
                "right": right,
                "landscape_variant": landscape_variant or "primary",
                "metric": metric,
                "n": len(seeds),
                "paired_difference": mean,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "resolved": bool(mean - half > 0 or mean + half < 0),
            }
        )
    return output


def _lookup(
    rows: Sequence[Mapping[str, object]], key: str = "method"
) -> dict[str, Mapping[str, object]]:
    return {str(row[key]): row for row in rows}


def _fmt(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _effect(
    rows: Sequence[Mapping[str, object]], left: str, right: str, metric: str
) -> str:
    row = next(
        row
        for row in rows
        if row["left"] == left
        and row["right"] == right
        and row["metric"] == metric
        and row["landscape_variant"] == "primary"
    )
    return (
        f"{float(row['paired_difference']):+.3f} "
        f"(95% CI {float(row['ci95_low']):+.3f} to {float(row['ci95_high']):+.3f})"
    )


def _plot_frontiers(summary: Sequence[Mapping[str, object]], output: Path) -> None:
    methods = [row for row in summary if row["method"] in DISPLAY]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, y, label in [
        (axes[0], "useful_region_count", "Distinct useful regions"),
        (axes[1], "useful_selected", "Useful selected (frozen router)"),
    ]:
        for row in methods:
            axis.scatter(float(row["selected_quality"]), float(row[y]), s=55)
            axis.annotate(
                DISPLAY[str(row["method"])],
                (float(row["selected_quality"]), float(row[y])),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        axis.set_xlabel("Selected quality")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.suptitle("Quality–portfolio tradeoffs (30 paired seeds)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_alpha_sweep(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for kind in ["global", "goal", "target_anchor", "useful_decoy"]:
        for form, style in [("single", "-"), ("k_radius", "--")]:
            subset = sorted(
                [row for row in rows if row["kind"] == kind and row["form"] == form],
                key=lambda row: float(row["alpha"]),
            )
            label = f"{kind} {form}"
            x = [float(row["alpha"]) for row in subset]
            axes[0, 0].plot(
                x,
                [float(row["candidate_mean_quality"]) for row in subset],
                style,
                label=label,
            )
            axes[0, 1].plot(
                x, [float(row["proposal_recall"]) for row in subset], style, label=label
            )
            axes[1, 0].plot(
                x,
                [float(row["useful_region_count"]) for row in subset],
                style,
                label=label,
            )
            axes[1, 1].plot(
                x,
                [float(row["candidate_decoy_rate"]) for row in subset],
                style,
                label=label,
            )
    for axis, title in zip(
        axes.flat,
        ["Candidate quality", "Proposal recall", "Useful regions", "Decoy rate"],
    ):
        axis.set_title(title)
        axis.set_xlabel("alpha / maximum K radius")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6, ncol=2)
    figure.suptitle("Validation-only contrastive radius sweep")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_generalization(summary: Sequence[Mapping[str, object]], output: Path) -> None:
    methods = ["contrast_goal_1", "contrast_goal_k", "csd_rl", "csd_mode", "csd_es"]
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
    lookup = {
        (str(row["landscape_variant"]), str(row["method"])): row for row in summary
    }
    x = np.arange(len(variants))
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for method in methods:
        axes[0].plot(
            x,
            [
                float(lookup[(variant, method)]["proposal_recall"])
                for variant in variants
            ],
            marker="o",
            label=DISPLAY[method],
        )
        axes[1].plot(
            x,
            [
                float(lookup[(variant, method)]["useful_region_count"])
                for variant in variants
            ],
            marker="o",
            label=DISPLAY[method],
        )
    axes[0].set_ylabel("Proposal recall")
    axes[1].set_ylabel("Distinct useful regions")
    axes[1].set_xticks(x, variants, rotation=25, ha="right")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    figure.suptitle("Held-out generalization")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_representative(
    *,
    seed: int,
    selection: Mapping[str, object],
    output: Path,
    device: torch.device,
) -> None:
    base = preset_gen3("main")
    rl = RLStudyConfig()
    router_config = RouterExperimentConfig()
    family = V2LandscapeFamily(base.base.landscape)
    train, _, _, _, test, test_starts = _datasets(family, base, rl, seed)
    vectors = build_contrastive_vectors(
        train,
        split="train_iid",
        seed=seed + 60_001,
        config=ContrastiveExperimentConfig().data,
    )
    start_array, task_ids, _ = flatten_starts(test_starts)
    start_tensor = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    goals = torch.as_tensor(
        np.stack([test[index].goal for index in task_ids]),
        dtype=torch.float32,
        device=device,
    )
    alphas = selection["selected_alphas"]
    assert isinstance(alphas, dict)
    contrast = contrastive_candidates(
        start_tensor,
        goals,
        vectors,
        kind="goal",
        alpha=float(alphas["goal:k_radius"]),
        form="k_radius",
        candidate_count=base.base.training.candidates,
    )
    contrast_batch, _, _, _ = make_evaluated_batch(
        source="contrast_goal_k",
        tasks=test,
        starts=test_starts,
        proposal=contrast,
        base=base,
        router_config=router_config,
        device=device,
        refinement=True,
    )
    generator = load_frozen_generator("rl", base=base, rl=rl, seed=seed, device=device)
    csd_batch = generate_frozen_candidates(
        generator,
        tasks=test,
        starts=test_starts,
        action_seed=seed + 41_004,
        base=base,
        router_config=router_config,
        device=device,
        source="csd_rl",
    )
    task = test[0]
    index = 0
    start = start_array[index]
    low = np.minimum(np.min(task.decoy_centers, axis=0), task.anchor) - 1.0
    high = np.maximum(np.max(task.decoy_centers, axis=0), task.anchor) + 1.0
    x = np.linspace(low[0], high[0], 220)
    y = np.linspace(low[1], high[1], 220)
    xx, yy = np.meshgrid(x, y)
    quality = task.quality(np.stack([xx, yy], axis=-1))
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    datasets = [
        (
            "Contrastive goal K-radius",
            contrast_batch.candidates[index].cpu().numpy(),
            contrast_batch.refined[index].cpu().numpy(),
        ),
        (
            "Frozen CSD RL",
            csd_batch.candidates[index].cpu().numpy(),
            csd_batch.refined[index].cpu().numpy(),
        ),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, base.base.training.candidates))
    for axis, (title, candidates, refined) in zip(axes, datasets):
        axis.contourf(xx, yy, quality, levels=24, cmap="viridis", alpha=0.75)
        axis.scatter(
            task.anchor[0],
            task.anchor[1],
            marker="*",
            s=180,
            c="white",
            edgecolor="black",
            label="anchor",
        )
        axis.scatter(
            task.useful_centers[:, 0],
            task.useful_centers[:, 1],
            marker="o",
            s=70,
            c="lime",
            edgecolor="black",
            label="useful",
        )
        axis.scatter(
            task.decoy_centers[:, 0],
            task.decoy_centers[:, 1],
            marker="X",
            s=60,
            c="red",
            edgecolor="black",
            label="decoy",
        )
        axis.scatter(
            start[0],
            start[1],
            marker="s",
            s=65,
            c="cyan",
            edgecolor="black",
            label="start",
        )
        for candidate_id, color in enumerate(colors):
            delta = candidates[candidate_id] - start
            axis.arrow(
                start[0],
                start[1],
                delta[0],
                delta[1],
                color=color,
                width=0.018,
                head_width=0.16,
                length_includes_head=True,
                alpha=0.8,
            )
            axis.scatter(
                refined[candidate_id, 0],
                refined[candidate_id, 1],
                c=[color],
                marker="D",
                s=32,
            )
            axis.annotate(f"z{candidate_id}", refined[candidate_id], fontsize=7)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    axes[0].legend(fontsize=7, loc="lower left")
    figure.suptitle(f"Predeclared seed {seed}, first test task/trial")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def generate_contrastive_reports(
    *,
    main_dir: Path = Path("results/contrastive_steering_main"),
    sweep_dir: Path = Path("results/contrastive_steering_sweep"),
    benchmark_dir: Path = Path("results/contrastive_steering_benchmark"),
    device: torch.device = torch.device("cpu"),
) -> dict[str, object]:
    primary = read_csv(main_dir / "primary_per_seed.csv")
    selectors = read_csv(main_dir / "selector_per_seed.csv")
    no_refinement = read_csv(main_dir / "no_refinement_per_seed.csv")
    generalization = read_csv(main_dir / "generalization_per_seed.csv")
    training_data = read_csv(main_dir / "contrastive_training_data.csv")
    sweep = read_csv(sweep_dir / "alpha_sweep_summary.csv")
    selection = json.loads(
        (sweep_dir / "CONTRASTIVE_ALPHA_SELECTION.json").read_text(encoding="utf-8")
    )
    if training_data and "sampling_seconds" not in training_data[0]:
        base_for_timing = preset_gen3("main")
        rl_for_timing = RLStudyConfig()
        family_for_timing = V2LandscapeFamily(base_for_timing.base.landscape)
        timing_rows = []
        for seed in sorted(int(row["seed"]) for row in training_data):
            train, _, _, _, _, _ = _datasets(
                family_for_timing, base_for_timing, rl_for_timing, seed
            )
            vectors = build_contrastive_vectors(
                train,
                split="train_iid",
                seed=seed + 60_001,
                config=ContrastiveExperimentConfig().data,
            )
            timing_rows.append(
                {
                    "seed": seed,
                    "sampling_seconds": vectors.sampling_seconds,
                    "vector_seconds": vectors.vector_seconds,
                    "construction_seconds": vectors.construction_seconds,
                }
            )
        write_csv(main_dir / "contrastive_timing_per_seed.csv", timing_rows)
        timing_lookup = {int(row["seed"]): row for row in timing_rows}
        training_data = [
            {**row, **timing_lookup[int(row["seed"])]} for row in training_data
        ]
    primary_summary = _aggregate(primary, ["method"])
    selector_summary = _aggregate(
        selectors,
        ["method", "selector"],
        [
            "selected_quality",
            "proposal_recall",
            "routing_recall",
            "useful_selected",
            "target_selected",
            "decoy_selected",
            "anchor_selected",
            "coverage",
            "collapse",
        ],
    )
    no_refinement_summary = _aggregate(no_refinement, ["method"])
    generalization_summary = _aggregate(generalization, ["landscape_variant", "method"])
    paired_rows: list[dict[str, object]] = []
    paired_metrics = [
        "selected_quality",
        "proposal_recall",
        "useful_region_count",
        "useful_region_count_2_plus",
        "useful_selected",
        "candidate_decoy_rate",
        "candidate_target_rate",
    ]
    for left in [
        "contrast_goal_1",
        "contrast_goal_k",
        "contrast_target_anchor_1",
        "contrast_target_anchor_k",
    ]:
        for right in ["csd_rl", "csd_mode", "csd_es"]:
            paired_rows.extend(
                _paired(primary, left=left, right=right, metrics=paired_metrics)
            )
    generalization_paired: list[dict[str, object]] = []
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
    for variant in variants:
        for left in ["contrast_goal_1", "contrast_goal_k"]:
            for right in ["csd_rl", "csd_mode"]:
                generalization_paired.extend(
                    _paired(
                        generalization,
                        left=left,
                        right=right,
                        metrics=paired_metrics,
                        landscape_variant=variant,
                    )
                )
    generalization_paired.extend(
        _paired(
            generalization,
            left="contrast_goal_k_recomputed_rotated",
            right="contrast_goal_k",
            metrics=paired_metrics,
            landscape_variant="rotated",
        )
    )
    write_csv(main_dir / "primary_summary.csv", primary_summary)
    write_csv(main_dir / "selector_summary.csv", selector_summary)
    write_csv(main_dir / "no_refinement_summary.csv", no_refinement_summary)
    write_csv(main_dir / "primary_paired_differences.csv", paired_rows)
    write_csv(main_dir / "generalization_summary.csv", generalization_summary)
    write_csv(main_dir / "generalization_paired_differences.csv", generalization_paired)

    benchmark = _lookup(read_csv(benchmark_dir / "primary_per_seed.csv"))
    summary_lookup = _lookup(primary_summary)
    construction = np.asarray(
        [float(row["construction_seconds"]) for row in training_data]
    )
    sampling_time = float(
        np.mean([float(row["sampling_seconds"]) for row in training_data])
    )
    vector_time = float(
        np.mean([float(row["vector_seconds"]) for row in training_data])
    )
    rl_rows = [
        row
        for row in read_csv(Path("results/rl_adamw_main/primary_per_seed.csv"))
        if row["evaluation_mode"] == "rl_adamw_stochastic"
    ]
    mode_rows = [
        row
        for row in read_csv(Path("results/mode_preservation_main/primary_per_seed.csv"))
        if row["variant"] == "gated_shared_selected"
    ]
    es_rows = [
        row
        for row in read_csv(Path("results/gen3_gpu/primary_per_seed.csv"))
        if row["variant"] == "gen3_contextual_linear_alpha"
    ]
    regime_rows = read_csv(
        Path("results/training_regime_comparison/primary_comparison.csv")
    )
    gpu_es_cost = next(row for row in regime_rows if row["training_regime"] == "GPU ES")
    external = {
        "csd_rl": {
            "training_seconds": float(
                np.mean([float(row["training_seconds"]) for row in rl_rows])
            ),
            "parameters": int(float(rl_rows[0]["policy_parameter_count"])),
            "training_peak": float(
                np.mean([float(row["peak_allocated_bytes"]) for row in rl_rows])
            ),
            "training_outcomes": int(float(rl_rows[0]["candidate_evaluations"])),
        },
        "csd_mode": {
            "training_seconds": float(
                np.mean([float(row["training_seconds"]) for row in mode_rows])
            ),
            "parameters": int(float(mode_rows[0]["policy_parameter_count"])),
            "training_peak": float(
                np.mean([float(row["peak_allocated_bytes"]) for row in mode_rows])
            ),
            "training_outcomes": int(float(mode_rows[0]["candidate_evaluations"])),
        },
        "csd_es": {
            "training_seconds": float(gpu_es_cost["runtime_seconds"]),
            "parameters": int(float(es_rows[0]["parameter_count"])),
            "training_peak": float(gpu_es_cost["peak_gpu_allocated_bytes"]),
            "training_outcomes": int(float(gpu_es_cost["policy_candidate_outcomes"])),
        },
    }
    cost_rows: list[dict[str, object]] = []
    stored_scalars = int(
        float(read_csv(main_dir / "cost_per_seed.csv")[0]["stored_direction_scalars"])
    )
    for method in [
        "contrast_goal_1",
        "contrast_goal_k",
        "csd_rl",
        "csd_mode",
        "csd_es",
    ]:
        row = summary_lookup[method]
        if method.startswith("contrast_"):
            training_seconds, parameters, training_peak = (
                float(np.mean(construction)),
                0,
                0.0,
            )
            training_outcomes, stored = 0, stored_scalars
        else:
            training_seconds = external[method]["training_seconds"]
            parameters = external[method]["parameters"]
            training_peak = external[method]["training_peak"]
            training_outcomes = external[method]["training_outcomes"]
            stored = 0
        elapsed = float(row["evaluation_seconds"])
        calls = float(row["candidate_evaluations"])
        cost_rows.append(
            {
                "method": method,
                "training_or_construction_seconds": training_seconds,
                "data_collection_seconds": sampling_time
                if method.startswith("contrast_")
                else 0.0,
                "vector_construction_seconds": vector_time
                if method.startswith("contrast_")
                else 0.0,
                "training_candidate_outcomes": training_outcomes,
                "learned_parameter_count": parameters,
                "stored_contrast_scalars": stored,
                "inference_seconds": elapsed,
                "candidate_throughput_per_second": calls / elapsed,
                "candidate_evaluations": calls,
                "q_equivalent_evaluations": float(row["q_equivalent_evaluations"]),
                "incremental_inference_vram_bytes": float(
                    benchmark[method]["peak_incremental_bytes"]
                ),
                "training_peak_allocated_bytes": training_peak,
            }
        )
    write_csv(main_dir / "cost_summary.csv", cost_rows)
    _plot_frontiers(primary_summary, main_dir / "quality_portfolio_frontiers.png")
    _plot_alpha_sweep(sweep, main_dir / "contrastive_alpha_sweep.png")
    _plot_generalization(
        generalization_summary, main_dir / "contrastive_generalization.png"
    )
    for seed in [7, 19]:
        _plot_representative(
            seed=seed,
            selection=selection,
            output=main_dir / f"contrastive_vs_csd_seed_{seed}.png",
            device=device,
        )
    lookup = _lookup(primary_summary)
    no_ref_lookup = _lookup(no_refinement_summary)
    cost_lookup = _lookup(cost_rows)

    table_methods = [
        "random",
        "pure_divergence",
        "contrast_global_1",
        "contrast_goal_1",
        "contrast_target_anchor_1",
        "contrast_goal_k",
        "csd_rl",
        "csd_mode",
        "csd_es",
    ]
    table_lines = [
        "| Method | Quality | Proposal Recall | Useful Regions | 2+ Useful Regions | Useful Selected | Decoy | Target | Runtime (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in table_methods:
        row = lookup[method]
        table_lines.append(
            f"| {DISPLAY[method]} | {_fmt(row['selected_quality'])} | {_fmt(row['proposal_recall'])} | "
            f"{_fmt(row['useful_region_count'])} | {_fmt(row['useful_region_count_2_plus'])} | "
            f"{_fmt(row['useful_selected'])} | {_fmt(row['candidate_decoy_rate'])} | "
            f"{_fmt(row['candidate_target_rate'])} | {_fmt(row['evaluation_seconds'], 4)} |"
        )
    main_table = "\n".join(table_lines)
    selector_lookup = {
        (str(row["method"]), str(row["selector"])): row for row in selector_summary
    }
    selector_lines = [
        "| Method | Selector | Quality | Useful | Target | Decoy | Routing Recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ["contrast_goal_k", "csd_rl", "csd_mode"]:
        for selector in [
            "random",
            "quality_router",
            "frozen_pairwise_router",
            "quality_oracle",
            "useful_oracle",
        ]:
            row = selector_lookup[(method, selector)]
            selector_lines.append(
                f"| {DISPLAY[method]} | {selector} | {_fmt(row['selected_quality'])} | "
                f"{_fmt(row['useful_selected'])} | {_fmt(row['target_selected'])} | "
                f"{_fmt(row['decoy_selected'])} | {_fmt(row['routing_recall'])} |"
            )
    selector_table = "\n".join(selector_lines)
    data_example = training_data[0]
    results_report = f"""# Contrastive steering results

This is a 2D toy analogue of contrastive activation steering, not an LLM CAA
experiment. Contrast vectors are differences of sampled training-state class
means. No validation or test state contributes to a vector, and all radii were
frozen using the six-seed IID validation sweep before the 30-seed test was read.

Each seed-level training pool contains {int(float(data_example["anchor"]))} anchor,
{int(float(data_example["useful"]))} useful, {int(float(data_example["decoy"]))}
decoy, and {int(float(data_example["background"]))} irrelevant/background sampled
states. Region centroids are balanced within categories before class means are
contrasted. The goal baseline uses target-positive versus balanced anchor,
non-target-useful, decoy, and background negatives. Target-anchor is privileged.

Validation selected single/K maximum radii of
`{selection["selected_alphas"]}`. K-radius uses eight positive radii on the same
direction; it is compute-matched but is not given eight semantic vectors.

## Primary matched-refinement comparison

Quality is frozen-pairwise-router selected quality. Proposal/portfolio metrics are
computed before routing. Runtime covers 30 task/trial sets per seed.

{main_table}

Thirty-seed mean ± standard deviation values are in `primary_summary.csv`; paired
95% intervals are in `primary_paired_differences.csv`.

## Selector decomposition

Every selector sees exactly the same generated candidate tensor.

{selector_table}

## Refinement control

Contrastive goal-single without refinement has quality
{_fmt(no_ref_lookup["contrast_goal_1"]["candidate_mean_quality"])} and proposal
recall {_fmt(no_ref_lookup["contrast_goal_1"]["proposal_recall"])}; after the same
nine refinement steps these become {_fmt(lookup["contrast_goal_1"]["candidate_mean_quality"])}
and {_fmt(lookup["contrast_goal_1"]["proposal_recall"])}. Goal K-radius proposal
recall rises from {_fmt(no_ref_lookup["contrast_goal_k"]["proposal_recall"])} to
{_fmt(lookup["contrast_goal_k"]["proposal_recall"])}. Refinement is a major part of
the successful contrastive pipeline and is matched exactly in the primary
comparison (10 Q-equivalent calls per candidate).

## Direct findings

- One goal contrast reaches a useful target in
  {_fmt(lookup["contrast_goal_1"]["proposal_recall"])} of IID sets, with selected
  quality {_fmt(lookup["contrast_goal_1"]["selected_quality"])}.
- Global steering is weak: single proposal recall is
  {_fmt(lookup["contrast_global_1"]["proposal_recall"])}; its class means largely
  cancel across the three goals.
- K radii do not create semantic modes. Goal K-radius represents
  {_fmt(lookup["contrast_goal_k"]["useful_region_count"])} useful regions and never
  represents two, despite proposal recall {_fmt(lookup["contrast_goal_k"]["proposal_recall"])}.
- CSD mode preservation represents {_fmt(lookup["csd_mode"]["useful_region_count"])}
  useful regions and has {_fmt(lookup["csd_mode"]["useful_region_count_2_plus"])}
  of sets with at least two. This is the clearest CSD advantage.
- Random search represents more regions but pays a decoy rate of
  {_fmt(lookup["random"]["candidate_decoy_rate"])}; pure divergence is mostly decoys
  ({_fmt(lookup["pure_divergence"]["candidate_decoy_rate"])}).

The simple baseline is not weak: for a known goal in IID geometry it is better
than CSD on scalar quality and direct target selection, while CSD's distinct
advantage is multi-mode representation.
"""

    cost_ratio = float(cost_lookup["csd_rl"]["training_or_construction_seconds"]) / max(
        float(cost_lookup["contrast_goal_k"]["training_or_construction_seconds"]), 1e-12
    )
    vs_report = f"""# Contrastive steering versus CSD

All claims below refer only to the existing 2D synthetic environment.

1. **Quality:** contrastive goal-single exceeds CSD RL by
   {_effect(paired_rows, "contrast_goal_1", "csd_rl", "selected_quality")} and
   CSD mode preservation by
   {_effect(paired_rows, "contrast_goal_1", "csd_mode", "selected_quality")}.
   At matched K, contrastive goal K differs from CSD RL by
   {_effect(paired_rows, "contrast_goal_k", "csd_rl", "selected_quality")}.
2. **Distinct useful alternatives:** matched-K contrast is lower than CSD RL by
   {_effect(paired_rows, "contrast_goal_k", "csd_rl", "useful_region_count")} and
   lower than mode-preserved CSD by
   {_effect(paired_rows, "contrast_goal_k", "csd_mode", "useful_region_count")}.
   Its 2+ useful-region rate is exactly zero.
3. **Proposal recall:** contrastive K versus CSD RL is
   {_effect(paired_rows, "contrast_goal_k", "csd_rl", "proposal_recall")}; both are
   near ceiling on IID tests.
4. **Useful selection:** goal-single is {_fmt(lookup["contrast_goal_1"]["useful_selected"])}
   because it needs no routing choice. At matched K, contrastive versus CSD RL is
   {_effect(paired_rows, "contrast_goal_k", "csd_rl", "useful_selected")}.
5. **Decoys:** goal-single versus CSD RL is
   {_effect(paired_rows, "contrast_goal_1", "csd_rl", "candidate_decoy_rate")}.
   K-radius and CSD are similar in IID decoy exposure.
6. **Cost:** contrast construction averages
   {_fmt(cost_lookup["contrast_goal_k"]["training_or_construction_seconds"], 4)} s,
   uses zero learned parameters and zero Q outcomes, while CSD RL training averages
   {_fmt(cost_lookup["csd_rl"]["training_or_construction_seconds"])} s and 100,000
   candidate outcomes. The wall-clock ratio is approximately {cost_ratio:.0f}x.
   Exact inference throughput and incremental VRAM are in `cost_summary.csv`.
7. **Advantage beyond a fixed direction:** yes, but narrowly. CSD produces multiple
   distinct useful modes and is more robust when the fixed goal-direction lookup is
   geometrically misaligned. It does not justify its complexity for merely reaching
   one known IID target.
8. **Generalization:** fixed contrast steering is excellent under translations and
   width shifts, but fails under rotations, unseen geometries, more peaks, and new
   goal orientations. CSD is less brittle on those shifts. Recomputing contrast in
   rotated training geometry restores performance, showing coordinate dependence.

The experiment rejects a broad claim that CSD is generally superior to a
contrast-derived direction. It supports the narrower claim that CSD buys portfolio
breadth and some fixed-policy geometric robustness at substantial training cost.
"""

    gen_lookup = {
        (str(row["landscape_variant"]), str(row["method"])): row
        for row in generalization_summary
    }
    gen_lines = [
        "# Contrastive steering generalization",
        "",
        "Each cell is selected quality / proposal recall / useful regions / candidate decoy rate.",
        "",
        "| Landscape | Contrast goal-1 | Contrast goal-K | CSD RL | CSD mode | CSD ES |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in [
        "iid",
        "rotated",
        "translated",
        "width_shift",
        "more_peaks",
        "more_decoys",
        "new_goals",
        "unseen",
    ]:
        cells = []
        for method in [
            "contrast_goal_1",
            "contrast_goal_k",
            "csd_rl",
            "csd_mode",
            "csd_es",
        ]:
            row = gen_lookup[(variant, method)]
            cells.append(
                " / ".join(
                    _fmt(row[key])
                    for key in [
                        "selected_quality",
                        "proposal_recall",
                        "useful_region_count",
                        "candidate_decoy_rate",
                    ]
                )
            )
        gen_lines.append(f"| {variant} | " + " | ".join(cells) + " |")

    def gen_effect(left: str, right: str, metric: str, variant: str) -> str:
        row = next(
            row
            for row in generalization_paired
            if row["left"] == left
            and row["right"] == right
            and row["metric"] == metric
            and row["landscape_variant"] == variant
        )
        return (
            f"{float(row['paired_difference']):+.3f} "
            f"(95% CI {float(row['ci95_low']):+.3f} to {float(row['ci95_high']):+.3f})"
        )

    fixed_rotated = gen_lookup[("rotated", "contrast_goal_k")]
    recomputed_rotated = gen_lookup[("rotated", "contrast_goal_k_recomputed_rotated")]
    generalization_report = (
        "\n".join(gen_lines)
        + f"""

## Rotation diagnostic

The fixed IID contrast vector reaches proposal recall
{_fmt(fixed_rotated["proposal_recall"])} after coordinate rotation. Recomputing
the same estimator from separate rotated **training** geometries reaches
{_fmt(recomputed_rotated["proposal_recall"])}; the paired gain is
{gen_effect("contrast_goal_k_recomputed_rotated", "contrast_goal_k", "proposal_recall", "rotated")}.
No rotated test state contributes to that recomputed vector. This separates method
capability under retraining from robustness of one frozen vector.

## Interpretation

- Translation and width shift preserve the learned direction relationship;
  contrastive goal-single remains near ceiling and exceeds the frozen CSD methods.
- Fixed contrast is brittle under rotation and new goals. On new goals, goal-K
  proposal recall is {_fmt(gen_lookup[("new_goals", "contrast_goal_k")]["proposal_recall"])}
  versus {_fmt(gen_lookup[("new_goals", "csd_rl")]["proposal_recall"])} for CSD RL
  and {_fmt(gen_lookup[("new_goals", "csd_mode")]["proposal_recall"])} for CSD mode.
- More peaks and unseen configurations expose the discrete nearest-training-goal
  lookup: contrast has high quality only when its fixed direction remains aligned.
- CSD retains more distinct useful-region coverage across difficult shifts, but
  recomputed contrast remains a strong cheap baseline when representative training
  geometry for the shift is available.

Full 30-seed paired intervals are in `generalization_paired_differences.csv`.
"""
    )

    for name, text in [
        ("CONTRASTIVE_STEERING_RESULTS.md", results_report),
        ("CONTRASTIVE_STEERING_GENERALIZATION.md", generalization_report),
        ("CONTRASTIVE_VS_CSD.md", vs_report),
    ]:
        Path(name).write_text(text, encoding="utf-8")
        (main_dir / name).write_text(text, encoding="utf-8")
    aggregate = {
        "seed_count": 30,
        "selected_alphas": selection["selected_alphas"],
        "primary_summary": primary_summary,
        "paired_differences": paired_rows,
        "cost_summary": cost_rows,
    }
    (main_dir / "report_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
