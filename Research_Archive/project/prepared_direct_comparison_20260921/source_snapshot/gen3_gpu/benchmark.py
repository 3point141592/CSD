"""Fair repeated timing of original CPU and vectorized CUDA workloads."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from creativity_toy.config_gen3 import Gen3StudyConfig
from creativity_toy.creative_operator_v3 import ConditionalLearnedAlphaOperatorV3
from creativity_toy.critics import GoalConditionalRidgeCritic
from creativity_toy.experiments.study_v2 import _make_starts, _training_starts
from creativity_toy.landscapes_v2 import V2LandscapeFamily
from creativity_toy.normalization_diagnostics import evaluate_candidate_sets
from creativity_toy.training_v2 import operator_objective_v2

from .critic import BatchedRidgeCritic
from .evaluation import evaluate_batched
from .frozen_inputs import make_numpy_schedule
from .operator import BatchedCreativeOperator
from .reference import train_reference_with_schedule
from .runtime import RuntimeInfo, synchronize
from .study import write_csv
from .task_batch import pack_tasks
from .training import operator_objective_batched, train_operator_batched


def _timing_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_seconds": float(np.mean(array)),
        "median_seconds": float(np.median(array)),
        "sd_seconds": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum_seconds": float(np.min(array)),
    }


def _time_repetitions(
    callback: Callable[[], object],
    *,
    repetitions: int,
    device: torch.device | None = None,
    warmups: int = 1,
) -> list[float]:
    for _ in range(warmups):
        callback()
    values: list[float] = []
    for _ in range(repetitions):
        if device is not None:
            synchronize(device)
        start = perf_counter()
        callback()
        if device is not None:
            synchronize(device)
        values.append(perf_counter() - start)
    return values


def _mtime_runtime(root: Path) -> float | None:
    config = root / "config.json"
    aggregate = root / "aggregate_metrics.json"
    if not config.exists() or not aggregate.exists():
        return None
    return max(aggregate.stat().st_mtime - config.stat().st_mtime, 0.0)


def _load_cpu_model_and_critic(
    config: Gen3StudyConfig, seed: int
) -> tuple[ConditionalLearnedAlphaOperatorV3, GoalConditionalRidgeCritic]:
    model_path = Path("results/gen3_learned_alpha/models") / (
        f"seed_{seed}_gen3_contextual_linear_alpha.npz"
    )
    critic_path = model_path.with_name(model_path.stem + "_critic.npz")
    model = np.load(model_path, allow_pickle=False)
    critic_payload = np.load(critic_path, allow_pickle=False)
    base = config.base
    operator = ConditionalLearnedAlphaOperatorV3(
        base.landscape.dimension,
        base.training.candidates,
        base.training.latent_dimension,
        base.training.hidden_dimension,
        seed + 10_020,
        magnitude_mode="contextual_linear",
        alpha_max=config.alpha_max,
        alpha_hidden_dimension=config.alpha_hidden_dimension,
    )
    operator.set_parameters(model["parameters"])
    critic = GoalConditionalRidgeCritic(
        ridge=float(critic_payload["ridge"]),
        temperature=float(critic_payload["temperature"]),
        coefficients=critic_payload["coefficients"].copy(),
    )
    return operator, critic


def run_benchmarks(
    config: Gen3StudyConfig,
    *,
    output: Path,
    gpu_results: Path,
    device: torch.device,
    runtime: RuntimeInfo,
    repetitions: int = 5,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    base = config.base
    seed = base.seeds[0]
    family = V2LandscapeFamily(base.landscape)
    train_tasks = family.sample_tasks(
        geometries=base.train_geometries,
        seed=seed + 10_000,
        variant="iid",
        all_goals=True,
    )
    train_starts_list = _training_starts(
        _make_starts(train_tasks, trials=1, seed=seed + 10_001)
    )
    train_starts_np = np.stack(train_starts_list)
    packed = pack_tasks(train_tasks, device=device, dtype=torch.float32)
    starts_gpu = torch.as_tensor(train_starts_np, dtype=torch.float32, device=device)
    gpu_operator = BatchedCreativeOperator(
        dimension=base.landscape.dimension,
        candidates=base.training.candidates,
        latent_dimension=base.training.latent_dimension,
        hidden_dimension=base.training.hidden_dimension,
        seed=seed + 10_020,
        variant="gen3_contextual_linear_alpha",
        device=device,
        dtype=torch.float32,
        alpha_max=config.alpha_max,
        alpha_hidden_dimension=config.alpha_hidden_dimension,
    )
    cpu_operator = ConditionalLearnedAlphaOperatorV3(
        base.landscape.dimension,
        base.training.candidates,
        base.training.latent_dimension,
        base.training.hidden_dimension,
        seed + 10_020,
        magnitude_mode="contextual_linear",
        alpha_max=config.alpha_max,
        alpha_hidden_dimension=config.alpha_hidden_dimension,
    )
    initial_cpu = cpu_operator.get_parameters().copy()
    population = base.training.population + base.training.population % 2

    # Reuse population noise and tasks so timings compare the same objective.
    objective_schedule = make_numpy_schedule(
        operator=gpu_operator,
        task_count=len(train_tasks),
        config=base.training,
        iterations=1,
        seed=seed + 10_021,
    )
    indices = objective_schedule.batch_indices[0]
    batch_tasks_cpu = [train_tasks[int(index)] for index in indices]
    batch_starts_cpu = [train_starts_list[int(index)] for index in indices]
    batch_indices_gpu = torch.as_tensor(indices, dtype=torch.long, device=device)
    batch_tasks_gpu = packed.index_select(batch_indices_gpu)
    batch_starts_gpu = starts_gpu.index_select(0, batch_indices_gpu)
    noise = objective_schedule.noise[0]
    progress_sigma = base.training.sigma_start
    population_theta_gpu = gpu_operator.initial_parameters[
        None, :
    ] + progress_sigma * torch.as_tensor(noise, dtype=torch.float32, device=device)

    def cpu_population_objective() -> None:
        for member in range(population):
            cpu_operator.set_parameters(initial_cpu + progress_sigma * noise[member])
            operator_objective_v2(
                cpu_operator, batch_tasks_cpu, batch_starts_cpu, base.training
            )

    def gpu_population_objective() -> None:
        operator_objective_batched(
            gpu_operator,
            population_theta_gpu,
            batch_tasks_gpu,
            batch_starts_gpu,
            base.training,
        )

    cpu_objective_times = _time_repetitions(
        cpu_population_objective, repetitions=repetitions, warmups=1
    )
    gpu_objective_times = _time_repetitions(
        gpu_population_objective,
        repetitions=repetitions,
        device=device,
        warmups=3,
    )

    # Average a fixed five-step block so CPU/GPU timings cover equal work.
    step_iterations = 5
    step_schedule = make_numpy_schedule(
        operator=gpu_operator,
        task_count=len(train_tasks),
        config=base.training,
        iterations=step_iterations,
        seed=seed + 10_021,
    )

    def cpu_training_block() -> None:
        cpu_operator.set_parameters(initial_cpu)
        train_reference_with_schedule(
            cpu_operator,
            train_tasks,
            train_starts_list,
            base.training,
            schedule=step_schedule,
            checkpoint_steps=(step_iterations,),
        )

    def gpu_training_block() -> None:
        train_operator_batched(
            gpu_operator,
            packed,
            starts_gpu,
            base.training,
            schedule=step_schedule,
            checkpoint_steps=(step_iterations,),
        )

    cpu_training_times = [
        value / step_iterations
        for value in _time_repetitions(
            cpu_training_block, repetitions=repetitions, warmups=1
        )
    ]
    gpu_training_times = [
        value / step_iterations
        for value in _time_repetitions(
            gpu_training_block,
            repetitions=repetitions,
            device=device,
            warmups=2,
        )
    ]

    # Reuse learned weights so model differences cannot confound timing.
    cpu_eval_operator, cpu_critic = _load_cpu_model_and_critic(config, seed)
    test_tasks = family.sample_tasks(
        geometries=base.test_geometries,
        seed=seed + 20_000,
        variant="iid",
        all_goals=True,
    )
    test_starts = _make_starts(
        test_tasks, trials=base.trials_per_task, seed=seed + 20_001
    )
    gpu_eval_operator = BatchedCreativeOperator(
        dimension=base.landscape.dimension,
        candidates=base.training.candidates,
        latent_dimension=base.training.latent_dimension,
        hidden_dimension=base.training.hidden_dimension,
        seed=seed + 10_020,
        variant="gen3_contextual_linear_alpha",
        device=device,
        dtype=torch.float32,
        alpha_max=config.alpha_max,
        alpha_hidden_dimension=config.alpha_hidden_dimension,
    )
    model = np.load(
        Path("results/gen3_learned_alpha/models")
        / f"seed_{seed}_gen3_contextual_linear_alpha.npz",
        allow_pickle=False,
    )
    gpu_eval_theta = gpu_eval_operator.parameters_from_numpy(model["parameters"])
    gpu_eval_critic = BatchedRidgeCritic(
        coefficients=torch.as_tensor(
            cpu_critic.coefficients,
            dtype=torch.float32,
            device=device,
        ),
        ridge=cpu_critic.ridge,
        temperature=cpu_critic.temperature,
    )
    packed_test = pack_tasks(test_tasks, device=device, dtype=torch.float32)

    def cpu_evaluation() -> None:
        evaluate_candidate_sets(
            variant="gen3_contextual_linear_alpha",
            operator=cpu_eval_operator,
            critic=cpu_critic,
            tasks=test_tasks,
            starts=test_starts,
            config=base.training,
            seed=seed + 21_004,
            alpha=3.0,
            useful_threshold=base.landscape.useful_threshold,
            high_quality_threshold=base.landscape.high_quality_threshold,
        )

    def gpu_evaluation() -> None:
        evaluate_batched(
            variant="gen3_contextual_linear_alpha",
            operator=gpu_eval_operator,
            theta=gpu_eval_theta,
            critic=gpu_eval_critic,
            tasks=test_tasks,
            packed_tasks=packed_test,
            starts=test_starts,
            config=base.training,
            seed=seed + 21_004,
            useful_threshold=base.landscape.useful_threshold,
            high_quality_threshold=base.landscape.high_quality_threshold,
        )

    cpu_evaluation_times = _time_repetitions(
        cpu_evaluation, repetitions=repetitions, warmups=1
    )
    gpu_evaluation_times = _time_repetitions(
        gpu_evaluation, repetitions=repetitions, device=device, warmups=2
    )

    def gpu_forward_only() -> None:
        gpu_operator.forward(
            population_theta_gpu, batch_starts_gpu, batch_tasks_gpu.goals
        )

    gpu_forward_times = _time_repetitions(
        gpu_forward_only,
        repetitions=repetitions,
        device=device,
        warmups=3,
    )
    synchronize(device)
    profiled_eval_start = perf_counter()
    profiled_evaluation = evaluate_batched(
        variant="gen3_contextual_linear_alpha",
        operator=gpu_eval_operator,
        theta=gpu_eval_theta,
        critic=gpu_eval_critic,
        tasks=test_tasks,
        packed_tasks=packed_test,
        starts=test_starts,
        config=base.training,
        seed=seed + 21_004,
        useful_threshold=base.landscape.useful_threshold,
        high_quality_threshold=base.landscape.high_quality_threshold,
    )
    synchronize(device)
    profiled_eval_total = perf_counter() - profiled_eval_start

    raw_rows: list[dict[str, object]] = []
    timing_groups = {
        ("population_objective", "cpu"): cpu_objective_times,
        ("population_objective", "gpu"): gpu_objective_times,
        ("per_training_step", "cpu"): cpu_training_times,
        ("per_training_step", "gpu"): gpu_training_times,
        ("evaluation", "cpu"): cpu_evaluation_times,
        ("evaluation", "gpu"): gpu_evaluation_times,
    }
    for (workload, implementation), values in timing_groups.items():
        for repetition, seconds in enumerate(values, start=1):
            raw_rows.append(
                {
                    "workload": workload,
                    "implementation": implementation,
                    "repetition": repetition,
                    "seconds": seconds,
                }
            )
    summary_rows: list[dict[str, object]] = []
    for workload in ["population_objective", "per_training_step", "evaluation"]:
        cpu_values = timing_groups[(workload, "cpu")]
        gpu_values = timing_groups[(workload, "gpu")]
        cpu_stats = _timing_stats(cpu_values)
        gpu_stats = _timing_stats(gpu_values)
        if workload == "population_objective":
            units = population * len(indices) * base.training.candidates
            unit_name = "candidate_trajectories"
        elif workload == "per_training_step":
            units = population * min(base.training.batch_size, len(train_tasks))
            unit_name = "population_landscapes"
        else:
            units = len(test_tasks) * base.trials_per_task * base.training.candidates
            unit_name = "candidate_evaluations"
        summary_rows.append(
            {
                "workload": workload,
                "cpu_mean_seconds": cpu_stats["mean_seconds"],
                "cpu_median_seconds": cpu_stats["median_seconds"],
                "cpu_sd_seconds": cpu_stats["sd_seconds"],
                "cpu_minimum_seconds": cpu_stats["minimum_seconds"],
                "gpu_mean_seconds": gpu_stats["mean_seconds"],
                "gpu_median_seconds": gpu_stats["median_seconds"],
                "gpu_sd_seconds": gpu_stats["sd_seconds"],
                "gpu_minimum_seconds": gpu_stats["minimum_seconds"],
                "speedup_cpu_over_gpu": cpu_stats["mean_seconds"]
                / gpu_stats["mean_seconds"],
                "throughput_unit": unit_name,
                "cpu_throughput_per_second": units / cpu_stats["mean_seconds"],
                "gpu_throughput_per_second": units / gpu_stats["mean_seconds"],
                "repetitions": repetitions,
            }
        )

    # Repeated tasks isolate utilization scaling without adding new examples.
    # The scientific sample count therefore stays fixed.
    scaling_rows: list[dict[str, object]] = []
    for batch_size in [32, 64, 128, 256, 512, 1024]:
        index = torch.arange(batch_size, device=device) % len(train_tasks)
        scale_tasks = packed.index_select(index)
        scale_starts = starts_gpu.index_select(0, index)

        def scaling_objective() -> None:
            operator_objective_batched(
                gpu_operator,
                population_theta_gpu,
                scale_tasks,
                scale_starts,
                base.training,
            )

        times = _time_repetitions(
            scaling_objective,
            repetitions=repetitions,
            device=device,
            warmups=3,
        )
        stats = _timing_stats(times)
        scaling_rows.append(
            {
                "batch_size": batch_size,
                **stats,
                "candidate_trajectories_per_second": (
                    population
                    * batch_size
                    * base.training.candidates
                    / stats["mean_seconds"]
                ),
            }
        )
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(
        [row["batch_size"] for row in scaling_rows],
        [row["candidate_trajectories_per_second"] for row in scaling_rows],
        marker="o",
    )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Synthetic benchmark batch size (scientific run unchanged)")
    axis.set_ylabel("Candidate trajectories / second")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "gpu_batch_scaling.png", dpi=170)
    plt.close(figure)

    write_csv(output / "benchmark_raw_timings.csv", raw_rows)
    write_csv(output / "benchmark_summary.csv", summary_rows)
    write_csv(output / "gpu_batch_scaling.csv", scaling_rows)
    profile_rows = [
        {
            "component": "gpu_operator_forward_population_batch",
            **_timing_stats(gpu_forward_times),
            "interpretation": "direction, normalization, alpha, candidate states",
        },
        {
            "component": "gpu_full_population_objective",
            **_timing_stats(gpu_objective_times),
            "interpretation": "forward plus 9-step refinement and exact loss",
        },
        {
            "component": "gpu_evaluation_tensor_compute",
            "mean_seconds": profiled_evaluation.elapsed_seconds,
            "median_seconds": profiled_evaluation.elapsed_seconds,
            "sd_seconds": 0.0,
            "minimum_seconds": profiled_evaluation.elapsed_seconds,
            "interpretation": "CUDA forward/refinement/quality/regions/critic",
        },
        {
            "component": "gpu_evaluation_reporting_and_transfer",
            "mean_seconds": max(
                profiled_eval_total - profiled_evaluation.elapsed_seconds, 0.0
            ),
            "median_seconds": max(
                profiled_eval_total - profiled_evaluation.elapsed_seconds, 0.0
            ),
            "sd_seconds": 0.0,
            "minimum_seconds": max(
                profiled_eval_total - profiled_evaluation.elapsed_seconds, 0.0
            ),
            "interpretation": "single transfer plus ragged exact metric reporting",
        },
    ]
    write_csv(output / "profile_summary.csv", profile_rows)
    cpu_end_to_end = _mtime_runtime(Path("results/gen3_learned_alpha"))
    gpu_end_to_end = _mtime_runtime(gpu_results)
    end_to_end_speedup = (
        cpu_end_to_end / gpu_end_to_end
        if cpu_end_to_end is not None and gpu_end_to_end not in {None, 0.0}
        else None
    )
    aggregate = {
        "runtime": runtime.to_dict(),
        "repetitions": repetitions,
        "warmup_excluded": True,
        "cuda_synchronized_timers": True,
        "cpu_end_to_end_seconds_from_output_timestamps": cpu_end_to_end,
        "gpu_end_to_end_seconds_from_output_timestamps": gpu_end_to_end,
        "end_to_end_speedup": end_to_end_speedup,
        "timings": summary_rows,
        "profile": profile_rows,
        "batch_size": {
            "scientific_training_batch": base.training.batch_size,
            "es_population": population,
            "candidates": base.training.candidates,
        },
        "peak_gpu_memory": (
            json.loads(
                (gpu_results / "aggregate_metrics.json").read_text(encoding="utf-8")
            )["memory"]
            if (gpu_results / "aggregate_metrics.json").exists()
            else {}
        ),
        "retained_loops": [
            "9 refinement steps: sequential recurrence",
            "65 ES iterations: optimizer state recurrence",
            "per-task broad-candidate RNG generation: preserves reference RNG call order",
            "ragged coverage/diversity report construction: final CPU reporting only",
        ],
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
