"""Full 30-seed vectorized GPU execution and paired CPU comparison."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from creativity_toy.analysis_utils import classify_candidate, statistics
from creativity_toy.config_gen3 import Gen3StudyConfig
from creativity_toy.creative_operator_v3 import ConditionalLearnedAlphaOperatorV3
from creativity_toy.experiments.study_gen3 import (
    VARIANTS,
    _aggregate_table,
    _enrich_alpha_summary,
)
from creativity_toy.experiments.study_v2 import _make_starts, _training_starts
from creativity_toy.landscapes_v2 import V2LandscapeFamily
from creativity_toy.normalization_diagnostics import equivariance_diagnostics

from .critic import BatchedRidgeCritic, fit_critic_batched
from .evaluation import evaluate_batched
from .frozen_inputs import make_numpy_schedule
from .operator import BatchedCreativeOperator, Variant
from .runtime import RuntimeInfo, assert_device, synchronize
from .task_batch import pack_tasks
from .training import train_operator_batched

LEARNED_VARIANTS: tuple[Variant, ...] = (
    "gen3_global_alpha",
    "gen3_contextual_linear_alpha",
    "gen3_contextual_mlp_alpha",
)
REFERENCE_VARIANTS: tuple[Variant, ...] = (
    "gen2_unnormalized",
    "gen2_normalized_fixed_alpha3",
)
COMPARISON_METRICS = [
    "selected_quality",
    "oracle_best_quality",
    "coverage",
    "useful_exploration_rate",
    "decoy_rate",
    "target_hit_rate",
    "selected_target_rate",
    "critic_regret",
    "mean_alpha",
    "alpha_sd",
]


def _serializable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), separators=(",", ":"))
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serializable(row.get(key, "")) for key in keys})


def read_csv(path: Path) -> list[dict[str, object]]:
    def coerce(value: str) -> object:
        if value in {"True", "False"}:
            return value == "True"
        if value == "":
            return ""
        try:
            number = float(value)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value

    with path.open(encoding="utf-8") as handle:
        return [
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _cpu_operator_from_gpu(
    variant: Variant,
    config: Gen3StudyConfig,
    seed: int,
    parameters: np.ndarray,
) -> ConditionalLearnedAlphaOperatorV3:
    mode = {
        "gen3_global_alpha": "global",
        "gen3_contextual_linear_alpha": "contextual_linear",
        "gen3_contextual_mlp_alpha": "contextual_mlp",
    }[variant]
    base = config.base
    operator = ConditionalLearnedAlphaOperatorV3(
        base.landscape.dimension,
        base.training.candidates,
        base.training.latent_dimension,
        base.training.hidden_dimension,
        seed,
        magnitude_mode=mode,
        alpha_max=config.alpha_max,
        alpha_hidden_dimension=config.alpha_hidden_dimension,
    )
    operator.set_parameters(parameters)
    return operator


def _model_paths(
    config: Gen3StudyConfig, seed: int, variant: Variant
) -> tuple[Path, Path]:
    if variant == "gen2_unnormalized":
        root = Path(config.unnormalized_results) / "models"
        stem = f"seed_{seed}_gen2_unnormalized"
    elif variant == "gen2_normalized_fixed_alpha3":
        root = Path(config.fixed_alpha_results) / "models"
        stem = f"seed_{seed}_gen2_normalized"
    else:
        raise ValueError(variant)
    return root / f"{stem}.npz", root / f"{stem}_critic.npz"


def _load_reference_model(
    config: Gen3StudyConfig,
    seed: int,
    variant: Variant,
    operator: BatchedCreativeOperator,
) -> tuple[torch.Tensor, BatchedRidgeCritic]:
    model_path, critic_path = _model_paths(config, seed, variant)
    model = np.load(model_path, allow_pickle=False)
    critic_payload = np.load(critic_path, allow_pickle=False)
    theta = operator.parameters_from_numpy(model["parameters"])
    coefficients = torch.as_tensor(
        np.ascontiguousarray(critic_payload["coefficients"]),
        dtype=operator.dtype,
        device=operator.device,
    )
    return theta, BatchedRidgeCritic(
        coefficients=coefficients,
        ridge=float(critic_payload["ridge"]),
        temperature=float(critic_payload["temperature"]),
    )


def _save_model(
    path: Path,
    *,
    operator: BatchedCreativeOperator,
    theta: torch.Tensor,
    critic: BatchedRidgeCritic,
    seed: int,
) -> None:
    np.savez_compressed(
        path,
        parameters=operator.parameters_to_numpy(theta),
        codes=operator.codes.detach().cpu().numpy(),
        variant=operator.variant,
        seed=seed,
        dtype=str(operator.dtype),
        device=str(operator.device),
        alpha_max=operator.alpha_max,
        parameter_count=operator.parameter_count,
        additional_alpha_parameters=operator.additional_alpha_parameters,
    )
    np.savez_compressed(
        path.with_name(path.stem + "_critic.npz"),
        coefficients=critic.coefficients.detach().cpu().to(torch.float64).numpy(),
        ridge=critic.ridge,
        temperature=critic.temperature,
    )


def _comparison_rows(
    cpu_rows: list[dict[str, object]], gpu_rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cpu = {(int(row["seed"]), str(row["variant"])): row for row in cpu_rows}
    gpu = {(int(row["seed"]), str(row["variant"])): row for row in gpu_rows}
    per_seed: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for variant in VARIANTS:
        keys = sorted(key for key in cpu if key[1] == variant and key in gpu)
        for metric in COMPARISON_METRICS:
            cpu_values = np.asarray([float(cpu[key][metric]) for key in keys])
            gpu_values = np.asarray([float(gpu[key][metric]) for key in keys])
            difference = gpu_values - cpu_values
            for key, cpu_value, gpu_value, delta in zip(
                keys, cpu_values, gpu_values, difference
            ):
                per_seed.append(
                    {
                        "seed": key[0],
                        "variant": variant,
                        "metric": metric,
                        "cpu": cpu_value,
                        "gpu": gpu_value,
                        "absolute_difference": abs(delta),
                        "signed_difference": delta,
                        "relative_difference": abs(delta) / max(abs(cpu_value), 1e-12),
                    }
                )
            cpu_stats = statistics(cpu_values)
            delta_stats = statistics(difference)
            correlation = (
                float(np.corrcoef(cpu_values, gpu_values)[0, 1])
                if len(keys) > 1
                and np.std(cpu_values) > 1e-12
                and np.std(gpu_values) > 1e-12
                else 0.0
            )
            cpu_mean = float(np.mean(cpu_values))
            gpu_mean = float(np.mean(gpu_values))
            summary.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "cpu_mean": cpu_mean,
                    "gpu_mean": gpu_mean,
                    "absolute_difference": abs(gpu_mean - cpu_mean),
                    "relative_difference": abs(gpu_mean - cpu_mean)
                    / max(abs(cpu_mean), 1e-12),
                    "paired_mean_difference_gpu_minus_cpu": delta_stats["mean"],
                    "paired_difference_ci95": delta_stats["ci95"],
                    "cpu_mean_ci95": cpu_stats["ci95"],
                    "within_cpu_statistical_uncertainty": abs(gpu_mean - cpu_mean)
                    <= cpu_stats["ci95"],
                    "per_seed_correlation": correlation,
                    "n": len(keys),
                }
            )
    return per_seed, summary


def _plot_cpu_gpu(per_seed: list[dict[str, object]], path: Path) -> None:
    metrics = COMPARISON_METRICS[:8]
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for axis, metric in zip(axes.ravel(), metrics):
        rows = [row for row in per_seed if row["metric"] == metric]
        for variant in VARIANTS:
            subset = [row for row in rows if row["variant"] == variant]
            if subset:
                axis.scatter(
                    [float(row["cpu"]) for row in subset],
                    [float(row["gpu"]) for row in subset],
                    s=13,
                    alpha=0.65,
                    label=variant.replace("gen3_", "G3 ").replace("gen2_", "G2 "),
                )
        values = [float(row[key]) for row in rows for key in ("cpu", "gpu")]
        low, high = min(values), max(values)
        pad = max((high - low) * 0.05, 1e-6)
        axis.plot([low - pad, high + pad], [low - pad, high + pad], "k--", lw=0.8)
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("CPU")
        axis.set_ylabel("GPU")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _ranking_signature(rows: list[dict[str, object]], metric: str) -> list[str]:
    means = {
        variant: float(
            np.mean([float(row[metric]) for row in rows if row["variant"] == variant])
        )
        for variant in VARIANTS
    }
    reverse = metric not in {"decoy_rate", "critic_regret"}
    return sorted(means, key=means.get, reverse=reverse)  # type: ignore[arg-type]


def run_gpu_study(
    config: Gen3StudyConfig,
    *,
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    schedule_dir = output / "frozen_schedules"
    model_dir.mkdir(exist_ok=True)
    schedule_dir.mkdir(exist_ok=True)
    config_payload = config.to_dict()
    config_payload["gpu_runtime"] = runtime.to_dict()
    config_payload["implementation_only_changes"] = [
        "population/task/candidate tensor batching",
        "torch device placement",
        "float32 without TF32/AMP",
    ]
    (output / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    base = config.base
    family = V2LandscapeFamily(base.landscape)
    primary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    candidate_set_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    run_start = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for seed_index, seed in enumerate(base.seeds):
        print(f"[GPU Gen3 {seed_index + 1}/{len(base.seeds)}] seed {seed}", flush=True)
        train_tasks = family.sample_tasks(
            geometries=base.train_geometries,
            seed=seed + 10_000,
            variant="iid",
            all_goals=True,
        )
        train_starts_list = _training_starts(
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
        packed_train = pack_tasks(train_tasks, device=device, dtype=torch.float32)
        packed_test = pack_tasks(test_tasks, device=device, dtype=torch.float32)
        train_starts = torch.as_tensor(
            np.stack(train_starts_list), dtype=torch.float32, device=device
        )

        for variant in (*REFERENCE_VARIANTS, *LEARNED_VARIANTS):
            operator = BatchedCreativeOperator(
                dimension=base.landscape.dimension,
                candidates=base.training.candidates,
                latent_dimension=base.training.latent_dimension,
                hidden_dimension=base.training.hidden_dimension,
                seed=seed + 10_020,
                variant=variant,
                device=device,
                dtype=torch.float32,
                alpha_max=config.alpha_max,
                alpha_hidden_dimension=config.alpha_hidden_dimension,
            )
            assert_device(
                [
                    operator.initial_parameters,
                    operator.codes,
                    train_starts,
                    *packed_train.tensors(),
                ],
                device,
            )
            if variant in REFERENCE_VARIANTS:
                theta, critic = _load_reference_model(config, seed, variant, operator)
                training_seconds = 0.0
                critic_seconds = 0.0
                population_calls = 0
                full_calls = 0
            else:
                schedule = make_numpy_schedule(
                    operator=operator,
                    task_count=len(train_tasks),
                    config=base.training,
                    iterations=base.training.gen2_iterations,
                    seed=seed + 10_021,
                )
                schedule.save(schedule_dir / f"seed_{seed}_{variant}.npz")
                training = train_operator_batched(
                    operator,
                    packed_train,
                    train_starts,
                    base.training,
                    schedule=schedule,
                    checkpoint_steps=(0, 1, 10, base.training.gen2_iterations),
                )
                theta = training.parameters
                training_seconds = training.elapsed_seconds
                population_calls = training.population_objective_calls
                full_calls = training.full_objective_calls
                synchronize(device)
                critic_start = perf_counter()
                critic, _ = fit_critic_batched(
                    operator,
                    theta,
                    packed_train,
                    train_starts,
                    base.training,
                    seed=seed + 10_022,
                )
                synchronize(device)
                critic_seconds = perf_counter() - critic_start
                _save_model(
                    model_dir / f"seed_{seed}_{variant}.npz",
                    operator=operator,
                    theta=theta,
                    critic=critic,
                    seed=seed,
                )
                if seed_index == 0:
                    history_rows.extend(
                        {"variant": variant, **row} for row in training.history
                    )

            evaluation = evaluate_batched(
                variant=variant,
                operator=operator,
                theta=theta,
                critic=critic,
                tasks=test_tasks,
                packed_tasks=packed_test,
                starts=test_starts,
                config=base.training,
                seed=seed + 21_004,
                useful_threshold=base.landscape.useful_threshold,
                high_quality_threshold=base.landscape.high_quality_threshold,
            )
            for row in evaluation.records:
                row["seed"] = seed
                row["category"] = classify_candidate(
                    row, base.landscape.useful_threshold
                )
            for row in evaluation.candidate_sets:
                row["seed"] = seed
            for row in evaluation.selectors:
                row["seed"] = seed
            _enrich_alpha_summary(evaluation.summary, evaluation.records)
            if variant in LEARNED_VARIANTS:
                cpu_operator = _cpu_operator_from_gpu(
                    variant,
                    config,
                    seed + 10_020,
                    operator.parameters_to_numpy(theta),
                )
                equivariance = equivariance_diagnostics(
                    cpu_operator,
                    test_tasks,
                    test_starts,
                    seed=seed + 20_600,
                    trials=3,
                    translation_magnitude=2.0,
                )
            else:
                # Saved coordinate diagnostics keep reused models on their original evaluation;
                # the placeholders below are filled from those records after this loop.
                equivariance = {"rotation_error": np.nan, "translation_error": np.nan}
            primary_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "method": variant,
                    "training_wall_clock_seconds": training_seconds + critic_seconds,
                    "training_only_seconds": training_seconds,
                    "critic_fit_seconds": critic_seconds,
                    "evaluation_seconds": evaluation.elapsed_seconds,
                    "parameter_count": operator.parameter_count,
                    "additional_alpha_parameters": operator.additional_alpha_parameters,
                    "population_objective_calls": population_calls,
                    "full_objective_calls": full_calls,
                    **evaluation.summary,
                    **equivariance,
                }
            )
            candidate_rows.extend(evaluation.records)
            candidate_set_rows.extend(evaluation.candidate_sets)
            selector_rows.extend(evaluation.selectors)

    synchronize(device)
    end_to_end = perf_counter() - run_start
    cpu_primary = read_csv(Path("results/gen3_learned_alpha/primary_per_seed.csv"))
    # Seed-and-variant lookup keeps each reused model paired with its original diagnostics.
    cpu_lookup = {(int(row["seed"]), str(row["variant"])): row for row in cpu_primary}
    for row in primary_rows:
        if row["variant"] in REFERENCE_VARIANTS:
            reference = cpu_lookup[(int(row["seed"]), str(row["variant"]))]
            row["rotation_error"] = float(reference["rotation_error"])
            row["translation_error"] = float(reference["translation_error"])
    per_seed_comparison, comparison_summary = _comparison_rows(
        cpu_primary, primary_rows
    )
    gpu_table = _aggregate_table(primary_rows)
    write_csv(output / "primary_per_seed.csv", primary_rows)
    write_csv(output / "candidate_diagnostics.csv", candidate_rows)
    write_csv(output / "candidate_set_diagnostics.csv", candidate_set_rows)
    write_csv(output / "selector_per_seed.csv", selector_rows)
    write_csv(output / "training_history_first_seed.csv", history_rows)
    write_csv(output / "GPU_GEN3_TABLE.csv", gpu_table)
    write_csv(output / "cpu_gpu_per_seed_comparison.csv", per_seed_comparison)
    write_csv(output / "cpu_gpu_metric_summary.csv", comparison_summary)
    _plot_cpu_gpu(per_seed_comparison, output / "cpu_vs_gpu_metrics.png")
    rankings = {
        metric: {
            "cpu": _ranking_signature(cpu_primary, metric),
            "gpu": _ranking_signature(primary_rows, metric),
        }
        for metric in COMPARISON_METRICS[:8]
    }
    ranking_unchanged = all(value["cpu"] == value["gpu"] for value in rankings.values())
    memory = {
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    }
    aggregate = {
        "seed_count": len(base.seeds),
        "runtime": runtime.to_dict(),
        "end_to_end_seconds": end_to_end,
        "batch_semantics": {
            "es_population": base.training.population + base.training.population % 2,
            "training_tasks": base.train_geometries * base.landscape.useful_peaks,
            "candidates": base.training.candidates,
            "dimension": base.landscape.dimension,
        },
        "memory": memory,
        "method_rankings": rankings,
        "all_reported_rankings_unchanged": ranking_unchanged,
        "gpu_table": gpu_table,
    }
    (output / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
