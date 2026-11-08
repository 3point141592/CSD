"""Mode-A identical-input and Mode-B normal-seeded numerical validation."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch
from creativity_toy.config import STUDY_SEEDS
from creativity_toy.config_gen3 import Gen3StudyConfig
from creativity_toy.creative_operator_v3 import ConditionalLearnedAlphaOperatorV3
from creativity_toy.critics import GoalConditionalRidgeCritic
from creativity_toy.experiments.study_gen3 import _make_starts, _training_starts
from creativity_toy.landscapes_v2 import V2LandscapeFamily, V2Task
from creativity_toy.training_v2 import operator_objective_v2

from .critic import critic_features, make_numpy_broad_candidates
from .frozen_inputs import (
    make_numpy_schedule,
    make_torch_native_schedule,
    save_validation_inputs,
)
from .operator import BatchedCreativeOperator, Variant
from .reference import train_reference_with_schedule
from .runtime import RuntimeInfo, assert_device
from .task_batch import TorchTaskBatch, pack_tasks
from .training import operator_objective_batched, train_operator_batched

FORWARD_ATOL = 1e-5
FORWARD_RTOL = 1e-4
EARLY_TRAINING_MAX_ABS = 5e-4
VALIDATION_MODES: tuple[Variant, ...] = (
    "gen3_global_alpha",
    "gen3_contextual_linear_alpha",
    "gen3_contextual_mlp_alpha",
)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def error_statistics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    difference = np.abs(reference - actual)
    relative = difference / np.maximum(np.abs(reference), 1e-8)
    return {
        "max_abs_error": float(np.max(difference)) if difference.size else 0.0,
        "mean_abs_error": float(np.mean(difference)) if difference.size else 0.0,
        "max_relative_error": float(np.max(relative)) if relative.size else 0.0,
    }


def _cpu_operator(
    mode: Variant, config: Gen3StudyConfig, seed: int
) -> ConditionalLearnedAlphaOperatorV3:
    magnitude_mode = {
        "gen3_global_alpha": "global",
        "gen3_contextual_linear_alpha": "contextual_linear",
        "gen3_contextual_mlp_alpha": "contextual_mlp",
    }[mode]
    base = config.base
    return ConditionalLearnedAlphaOperatorV3(
        base.landscape.dimension,
        base.training.candidates,
        base.training.latent_dimension,
        base.training.hidden_dimension,
        seed,
        magnitude_mode=magnitude_mode,
        alpha_max=config.alpha_max,
        alpha_hidden_dimension=config.alpha_hidden_dimension,
    )


def _task_arrays(tasks: list[V2Task]) -> dict[str, np.ndarray]:
    return {
        "anchor": np.stack([task.anchor for task in tasks]),
        "goal": np.stack([task.goal for task in tasks]),
        "component_centers": np.stack([task.component_centers for task in tasks]),
        "component_precisions": np.stack([task.component_precisions for task in tasks]),
        "component_amplitudes": np.stack([task.component_amplitudes for task in tasks]),
        "useful_centers": np.stack([task.useful_centers for task in tasks]),
        "decoy_centers": np.stack([task.decoy_centers for task in tasks]),
        "goal_index": np.asarray([task.goal_index for task in tasks]),
        "geometry_id": np.asarray([task.geometry_id for task in tasks]),
    }


def _forward_reference(
    operator: ConditionalLearnedAlphaOperatorV3,
    tasks: list[V2Task],
    starts: list[np.ndarray],
) -> dict[str, np.ndarray]:
    raw = np.stack(
        [operator.raw_directions(h, task.goal) for task, h in zip(tasks, starts)]
    )
    normalized = np.stack([operator.unit_directions(row) for row in raw])
    alpha = np.stack(
        [
            operator.learned_alphas(h, task.goal, direction)
            for task, h, direction in zip(tasks, starts, normalized)
        ]
    )
    candidates = np.stack(
        [operator.apply(h, task.goal, 3.0) for task, h in zip(tasks, starts)]
    )
    refined = np.stack(
        [task.refine(row, steps=9, rate=0.28) for task, row in zip(tasks, candidates)]
    )
    objective = np.stack([task.quality(row) for task, row in zip(tasks, refined)])
    return {
        "raw_directions": raw,
        "normalized_directions": normalized,
        "learned_alpha": alpha,
        "candidate_states": candidates,
        "objective_values": objective,
    }


def _critic_scores_with_shared_coefficients(
    cpu_operator: ConditionalLearnedAlphaOperatorV3,
    gpu_operator: BatchedCreativeOperator,
    gpu_theta: torch.Tensor,
    tasks: list[V2Task],
    starts: list[np.ndarray],
    packed: TorchTaskBatch,
    config: Gen3StudyConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    start_array = np.stack(starts)
    learned = np.stack(
        [cpu_operator.apply(h, task.goal, 3.0) for task, h in zip(tasks, starts)]
    )
    broad = make_numpy_broad_candidates(
        start_array,
        candidates=config.base.training.candidates,
        pure_divergence_radius=config.base.training.pure_divergence_radius,
        seed=seed,
    )
    all_candidates = np.concatenate([learned, broad], axis=1)
    feature_rows = np.stack(
        [
            GoalConditionalRidgeCritic.features(h, row, task.goal)
            for h, row, task in zip(starts, all_candidates, tasks)
        ]
    )
    targets = np.concatenate(
        [
            task.quality(task.refine(row, steps=9, rate=0.28))
            for task, row in zip(tasks, all_candidates)
        ]
    )
    critic = GoalConditionalRidgeCritic(
        ridge=config.base.training.critic_ridge,
        temperature=config.base.training.critic_temperature,
    )
    critic.fit(feature_rows.reshape(-1, feature_rows.shape[-1]), targets)
    cpu_scores = np.stack([critic.predict(row) for row in feature_rows])
    h_gpu = torch.as_tensor(
        start_array, dtype=gpu_operator.dtype, device=gpu_operator.device
    )
    learned_gpu = gpu_operator.forward(
        gpu_theta, h_gpu, packed.goals
    ).candidates.squeeze(0)
    broad_gpu = torch.as_tensor(
        broad, dtype=gpu_operator.dtype, device=gpu_operator.device
    )
    features_gpu = critic_features(
        h_gpu, torch.cat([learned_gpu, broad_gpu], dim=1), packed.goals
    )
    coefficients_gpu = torch.as_tensor(
        critic.coefficients,
        dtype=gpu_operator.dtype,
        device=gpu_operator.device,
    )
    gpu_scores = torch.matmul(features_gpu, coefficients_gpu).detach().cpu().numpy()
    return cpu_scores, gpu_scores


def run_numerical_validation(
    config: Gen3StudyConfig,
    *,
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
    seeds: int = 3,
    iterations: int = 12,
) -> dict[str, object]:
    """Run both validation modes; Mode A is the scientific correctness gate."""
    output.mkdir(parents=True, exist_ok=True)
    input_dir = output / "frozen_inputs"
    input_dir.mkdir(exist_ok=True)
    base = config.base
    family = V2LandscapeFamily(base.landscape)
    tensor_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    normal_rows: list[dict[str, object]] = []

    for seed in STUDY_SEEDS[:seeds]:
        tasks = family.sample_tasks(
            geometries=min(base.train_geometries, 4),
            seed=seed + 10_000,
            variant="iid",
            all_goals=True,
        )
        starts = _training_starts(_make_starts(tasks, trials=1, seed=seed + 10_001))
        start_array = np.stack(starts)
        for variant in VALIDATION_MODES:
            init_seed = seed + 10_020
            cpu_operator = _cpu_operator(variant, config, init_seed)
            gpu_operator = BatchedCreativeOperator(
                dimension=base.landscape.dimension,
                candidates=base.training.candidates,
                latent_dimension=base.training.latent_dimension,
                hidden_dimension=base.training.hidden_dimension,
                seed=init_seed,
                variant=variant,
                device=device,
                dtype=torch.float32,
                alpha_max=config.alpha_max,
                alpha_hidden_dimension=config.alpha_hidden_dimension,
            )
            packed = pack_tasks(tasks, device=device, dtype=torch.float32)
            h_gpu = torch.as_tensor(start_array, dtype=torch.float32, device=device)
            assert_device(
                [
                    gpu_operator.initial_parameters,
                    gpu_operator.codes,
                    h_gpu,
                    *packed.tensors(),
                ],
                device,
            )
            initial_cpu = cpu_operator.get_parameters()
            initial_gpu = gpu_operator.initial_parameters.detach().cpu().numpy()
            cast_cpu = initial_cpu.astype(np.float32)
            param_error = error_statistics(cast_cpu, initial_gpu)
            tensor_rows.append(
                {
                    "validation_mode": "identical_input",
                    "seed": seed,
                    "variant": variant,
                    "stage": "initialization",
                    "tensor": "parameters_after_float32_transfer",
                    **param_error,
                    "atol": 0.0,
                    "rtol": 0.0,
                    "passed": param_error["max_abs_error"] == 0.0,
                }
            )

            frozen_reference = _forward_reference(cpu_operator, tasks, starts)
            gpu_forward = gpu_operator.forward(
                gpu_operator.initial_parameters, h_gpu, packed.goals
            )
            gpu_refined = packed.refine(
                gpu_forward.candidates.squeeze(0),
                steps=base.training.refinement_steps,
                rate=base.training.refinement_rate,
            )
            gpu_values = {
                "raw_directions": gpu_forward.raw_directions.squeeze(0),
                "normalized_directions": gpu_forward.directions.squeeze(0),
                "learned_alpha": gpu_forward.alphas.squeeze(0),
                "candidate_states": gpu_forward.candidates.squeeze(0),
                "objective_values": packed.quality(gpu_refined),
            }
            for name, gpu_value in gpu_values.items():
                stats = error_statistics(
                    frozen_reference[name], gpu_value.detach().cpu().numpy()
                )
                passed = bool(
                    np.allclose(
                        frozen_reference[name],
                        gpu_value.detach().cpu().numpy(),
                        atol=FORWARD_ATOL,
                        rtol=FORWARD_RTOL,
                    )
                )
                tensor_rows.append(
                    {
                        "validation_mode": "identical_input",
                        "seed": seed,
                        "variant": variant,
                        "stage": "forward",
                        "tensor": name,
                        **stats,
                        "atol": FORWARD_ATOL,
                        "rtol": FORWARD_RTOL,
                        "passed": passed,
                    }
                )

            cpu_reward, cpu_components = operator_objective_v2(
                cpu_operator, tasks, starts, base.training
            )
            gpu_objective = operator_objective_batched(
                gpu_operator,
                gpu_operator.initial_parameters,
                packed,
                h_gpu,
                base.training,
            )
            for name, cpu_value, gpu_value in [
                ("reward", cpu_reward, float(gpu_objective.reward[0].detach().cpu())),
                ("loss", -cpu_reward, -float(gpu_objective.reward[0].detach().cpu())),
                *[
                    (
                        f"loss_component_{key}",
                        value,
                        float(gpu_objective.components[key][0].detach().cpu()),
                    )
                    for key, value in cpu_components.items()
                    if key != "reward"
                ],
            ]:
                stats = error_statistics(np.asarray(cpu_value), np.asarray(gpu_value))
                tensor_rows.append(
                    {
                        "validation_mode": "identical_input",
                        "seed": seed,
                        "variant": variant,
                        "stage": "loss",
                        "tensor": name,
                        **stats,
                        "atol": FORWARD_ATOL,
                        "rtol": FORWARD_RTOL,
                        "passed": bool(
                            np.isclose(
                                cpu_value,
                                gpu_value,
                                atol=FORWARD_ATOL,
                                rtol=FORWARD_RTOL,
                            )
                        ),
                    }
                )
            cpu_scores, gpu_scores = _critic_scores_with_shared_coefficients(
                cpu_operator,
                gpu_operator,
                gpu_operator.initial_parameters,
                tasks,
                starts,
                packed,
                config,
                seed + 10_022,
            )
            score_stats = error_statistics(cpu_scores, gpu_scores)
            tensor_rows.append(
                {
                    "validation_mode": "identical_input",
                    "seed": seed,
                    "variant": variant,
                    "stage": "critic",
                    "tensor": "critic_scores_shared_coefficients",
                    **score_stats,
                    "atol": FORWARD_ATOL,
                    "rtol": FORWARD_RTOL,
                    "passed": bool(
                        np.allclose(
                            cpu_scores, gpu_scores, atol=FORWARD_ATOL, rtol=FORWARD_RTOL
                        )
                    ),
                }
            )

            schedule = make_numpy_schedule(
                operator=gpu_operator,
                task_count=len(tasks),
                config=base.training,
                iterations=iterations,
                seed=seed + 10_021,
            )
            prefix = f"seed_{seed}_{variant}"
            schedule.save(input_dir / f"{prefix}_es_schedule.npz")
            save_validation_inputs(
                input_dir / f"{prefix}_tensors.npz",
                starts=start_array,
                goals=np.stack([task.goal for task in tasks]),
                initial_parameters=initial_cpu,
                latent_codes=cpu_operator.codes,
                task_arrays=_task_arrays(tasks),
            )
            cpu_training = train_reference_with_schedule(
                cpu_operator,
                tasks,
                starts,
                base.training,
                schedule=schedule,
                checkpoint_steps=(0, 1, 10, iterations),
            )
            gpu_training = train_operator_batched(
                gpu_operator,
                packed,
                h_gpu,
                base.training,
                schedule=schedule,
                checkpoint_steps=(0, 1, 10, iterations),
            )
            for step in [0, 1, 10, iterations, -1]:
                cpu_parameters = (
                    cpu_training.parameters
                    if step == -1
                    else cpu_training.checkpoints[step]
                )
                gpu_parameters = (
                    (
                        gpu_training.parameters
                        if step == -1
                        else gpu_training.checkpoints[step]
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                stats = error_statistics(cpu_parameters, gpu_parameters)
                training_rows.append(
                    {
                        "validation_mode": "identical_input",
                        "seed": seed,
                        "variant": variant,
                        "step": "best_final" if step == -1 else step,
                        **stats,
                        "accepted_elites_cpu": sum(cpu_training.accepted_elites),
                        "accepted_elites_gpu": sum(
                            bool(row["accepted_elite"]) for row in gpu_training.history
                        ),
                        "passed_early_threshold": (
                            stats["max_abs_error"] <= EARLY_TRAINING_MAX_ABS
                            if step in {0, 1}
                            else "not_applicable"
                        ),
                    }
                )

            # Native RNGs exercise each device's normal sampling path. Equal integer
            # seeds can yield different draws, so this mode has no sample-equality gate.
            cpu_vectorized = BatchedCreativeOperator(
                dimension=base.landscape.dimension,
                candidates=base.training.candidates,
                latent_dimension=base.training.latent_dimension,
                hidden_dimension=base.training.hidden_dimension,
                seed=init_seed,
                variant=variant,
                device=torch.device("cpu"),
                dtype=torch.float32,
                alpha_max=config.alpha_max,
                alpha_hidden_dimension=config.alpha_hidden_dimension,
            )
            cpu_packed = pack_tasks(
                tasks, device=torch.device("cpu"), dtype=torch.float32
            )
            cpu_h = torch.as_tensor(start_array, dtype=torch.float32)
            cpu_schedule = make_torch_native_schedule(
                operator=cpu_vectorized,
                task_count=len(tasks),
                config=base.training,
                iterations=iterations,
                seed=seed + 10_021,
                generator_device=torch.device("cpu"),
            )
            gpu_native_schedule = make_torch_native_schedule(
                operator=gpu_operator,
                task_count=len(tasks),
                config=base.training,
                iterations=iterations,
                seed=seed + 10_021,
                generator_device=device,
            )
            cpu_native = train_operator_batched(
                cpu_vectorized,
                cpu_packed,
                cpu_h,
                base.training,
                schedule=cpu_schedule,
                checkpoint_steps=(iterations,),
            )
            gpu_native = train_operator_batched(
                gpu_operator,
                packed,
                h_gpu,
                base.training,
                schedule=gpu_native_schedule,
                checkpoint_steps=(iterations,),
            )
            cpu_final = operator_objective_batched(
                cpu_vectorized,
                cpu_native.parameters,
                cpu_packed,
                cpu_h,
                base.training,
            )
            gpu_final = operator_objective_batched(
                gpu_operator,
                gpu_native.parameters,
                packed,
                h_gpu,
                base.training,
            )
            normal_rows.append(
                {
                    "validation_mode": "normal_seeded",
                    "seed": seed,
                    "variant": variant,
                    "cpu_schedule_source": cpu_schedule.source,
                    "gpu_schedule_source": gpu_native_schedule.source,
                    "cpu_reward": float(cpu_final.reward[0]),
                    "gpu_reward": float(gpu_final.reward[0].detach().cpu()),
                    "absolute_reward_difference": abs(
                        float(cpu_final.reward[0])
                        - float(gpu_final.reward[0].detach().cpu())
                    ),
                    "cpu_quality": float(cpu_final.components["quality"][0]),
                    "gpu_quality": float(
                        gpu_final.components["quality"][0].detach().cpu()
                    ),
                    "cpu_mean_alpha": float(
                        cpu_vectorized.forward(
                            cpu_native.parameters, cpu_h, cpu_packed.goals
                        ).alphas.mean()
                    ),
                    "gpu_mean_alpha": float(
                        gpu_operator.forward(gpu_native.parameters, h_gpu, packed.goals)
                        .alphas.mean()
                        .detach()
                        .cpu()
                    ),
                }
            )

    _write_csv(output / "forward_validation.csv", tensor_rows)
    _write_csv(output / "training_validation.csv", training_rows)
    _write_csv(output / "normal_seeded_validation.csv", normal_rows)
    all_forward_pass = all(bool(row["passed"]) for row in tensor_rows)
    early_training_pass = all(
        row["passed_early_threshold"] in {True, "not_applicable"}
        for row in training_rows
    )
    aggregate = {
        "criteria_defined_before_results": {
            "forward_atol": FORWARD_ATOL,
            "forward_rtol": FORWARD_RTOL,
            "early_training_parameter_max_abs": EARLY_TRAINING_MAX_ABS,
            "full_training": (
                "means should remain within CPU statistical uncertainty; method "
                "rankings and scientific conclusions should remain unchanged"
            ),
        },
        "runtime": runtime.to_dict(),
        "seeds": list(STUDY_SEEDS[:seeds]),
        "iterations": iterations,
        "identical_input_forward_passed": all_forward_pass,
        "identical_input_early_training_passed": early_training_pass,
        "max_forward_abs_error": max(
            float(row["max_abs_error"]) for row in tensor_rows
        ),
        "max_step1_parameter_abs_error": max(
            float(row["max_abs_error"])
            for row in training_rows
            if str(row["step"]) == "1"
        ),
        "max_final_parameter_abs_error": max(
            float(row["max_abs_error"])
            for row in training_rows
            if str(row["step"]) == "best_final"
        ),
    }
    (output / "validation_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
