"""Orchestration for the isolated 32D supervised-generalization diagnosis."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from creativity_toy.analysis_utils import statistics
from creativity_toy.experiments.study_v2 import _make_starts
from creativity_toy.landscapes_v2 import V2Task

from .config_capacity_diagnostic import SupervisedCapacityConfig
from .config_d32_generalization import D32GeneralizationConfig
from .d32_generalization_models import (
    DirectionProbe,
    NoZSetPolicy,
    evaluate_direction_probe,
    probe_inputs,
    train_direction_probe,
    train_supervised_variant,
)
from .runtime import RuntimeInfo
from .study import write_csv
from .study_capacity_diagnostic import _aligned
from .study_d16 import _datasets, _environment, _policy
from .study_failure_diagnostics import _dimension_config, _family
from .supervised_capacity import (
    benchmark_supervised_batches,
    evaluate_supervised,
    exact_assignment_indices,
    useful_targets,
)
from .task_batch import TorchTaskBatch, pack_tasks
from .training_anchor_escape import evaluate_anchor_policy
from .training_endpoint_anchor import train_policy_endpoint_anchor

SCALING_METRICS = (
    "best_target_cosine",
    "mean_matched_cosine",
    "mean_angular_error_degrees",
    "mean_alpha_absolute_error",
    "proposal_recall",
    "target_proposal_recall",
    "useful_region_count",
    "candidate_mean_quality",
    "candidate_decoy_rate",
)


def _benchmark_config(config: D32GeneralizationConfig, geometries: int):
    benchmark = _dimension_config(32, smoke=False)
    benchmark.base.base.train_geometries = geometries
    return benchmark


def _data(
    config: D32GeneralizationConfig,
    geometries: int,
    seed: int,
) -> tuple[object, list[V2Task], list[V2Task], list[V2Task], list[list[np.ndarray]]]:
    benchmark = _benchmark_config(config, geometries)
    family = _family(benchmark)
    train, _, validation, _, test, test_starts = _datasets(family, benchmark, seed)
    return benchmark, train, validation, test, test_starts


def _training_batch(
    tasks: Sequence[V2Task],
    *,
    trials: int,
    seed: int,
    device: torch.device,
) -> tuple[TorchTaskBatch, torch.Tensor]:
    nested = _make_starts(tasks, trials=trials, seed=seed)
    return _aligned(tasks, nested, device)


def _single_start_batch(
    tasks: Sequence[V2Task], seed: int, device: torch.device
) -> tuple[TorchTaskBatch, torch.Tensor]:
    nested = _make_starts(tasks, trials=1, seed=seed)
    return _aligned(tasks, nested, device)


def _save_model(
    path: Path,
    model: torch.nn.Module,
    *,
    history: list[dict[str, float]],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "history": history,
            "metadata": metadata,
        },
        path,
    )


def _load_model(
    path: Path,
    make_model: Callable[[], torch.nn.Module],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = make_model()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _summary(
    rows: list[dict[str, object]],
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        result = dict(zip(keys, group))
        for metric in metrics:
            numbers = [float(value[metric]) for value in values]
            stat = statistics(numbers)
            result.update(
                {
                    f"{metric}_mean": stat["mean"],
                    f"{metric}_sd": stat["sd"],
                    f"{metric}_ci95": stat["ci95"],
                }
            )
        result["n"] = len(values)
        output.append(result)
    return output


def _eval(
    model: torch.nn.Module,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    benchmark,
    mode: str,
    lambda_alpha: float,
) -> dict[str, float]:
    return evaluate_supervised(
        model,
        tasks,
        starts,
        mode=mode,
        lambda_alpha=lambda_alpha,
        refinement_steps=benchmark.base.base.training.refinement_steps,
        refinement_rate=benchmark.base.base.training.refinement_rate,
        useful_threshold=benchmark.base.base.landscape.useful_threshold,
        capture_distance=3.0,
    ).summary


def _model_path(
    output: Path, geometries: int, seed: int, mode: str, variant: str
) -> Path:
    return output / "models" / f"n{geometries}_{variant}_{mode}_seed_{seed}.pt"


def _make_csd(benchmark, seed: int, device: torch.device):
    return _policy(benchmark, seed, device)


def _fit_scaling_model(
    config: D32GeneralizationConfig,
    output: Path,
    *,
    geometries: int,
    seed: int,
    mode: str,
    device: torch.device,
    variant: str = "fixed_z",
) -> tuple[torch.nn.Module, dict[str, object], object, list[V2Task]]:
    benchmark, train, validation, test, test_starts = _data(config, geometries, seed)
    packed_train, h_train = _training_batch(
        train,
        trials=config.scaling.starts_per_task,
        seed=seed + 41_001,
        device=device,
    )
    packed_validation, h_validation = _single_start_batch(
        validation, seed + 51_001, device
    )
    packed_test, h_test = _aligned(test, test_starts, device)
    if variant in {"fixed_z", "permuted_z"}:
        make_model = lambda: _make_csd(benchmark, seed, device)
    elif variant == "no_z_set":
        make_model = lambda: NoZSetPolicy(
            dimension=32,
            candidates=benchmark.base.base.training.candidates,
            hidden_dimension=config.slots.hidden_dimension,
            alpha_max=benchmark.rl.alpha_max,
            seed=seed + 10_020,
            device=device,
        )
    else:
        raise ValueError(variant)
    path = _model_path(output, geometries, seed, mode, variant)
    if path.exists():
        model, payload = _load_model(path, make_model, device)
        metadata = dict(payload.get("metadata", {}))
        history = list(payload.get("history", []))
    else:
        model = make_model()
        steps = (
            config.scaling.single_steps
            if mode == "single_target"
            else (
                config.slots.steps
                if variant != "fixed_z"
                else config.scaling.multimode_steps
            )
        )
        batch_size = min(
            config.slots.batch_size
            if variant != "fixed_z"
            else config.scaling.batch_size,
            len(h_train),
        )

        def callback(current: torch.nn.Module, _: int) -> dict[str, float]:
            if seed != config.scaling.endpoint_seeds[0]:
                return {}
            with torch.no_grad():
                train_eval = _eval(
                    current,
                    packed_train,
                    h_train,
                    benchmark,
                    mode,
                    config.scaling.lambda_alpha,
                )
                val_eval = _eval(
                    current,
                    packed_validation,
                    h_validation,
                    benchmark,
                    mode,
                    config.scaling.lambda_alpha,
                )
                test_eval = _eval(
                    current,
                    packed_test,
                    h_test,
                    benchmark,
                    mode,
                    config.scaling.lambda_alpha,
                )
            return {
                "training_cosine": train_eval["mean_matched_cosine"],
                "validation_cosine": val_eval["mean_matched_cosine"],
                "heldout_cosine": test_eval["mean_matched_cosine"],
                "training_recall": train_eval["proposal_recall"],
                "validation_recall": val_eval["proposal_recall"],
                "heldout_recall": test_eval["proposal_recall"],
            }

        result = train_supervised_variant(
            model,
            packed_train,
            h_train,
            mode=mode,
            steps=steps,
            batch_size=batch_size,
            learning_rate=config.scaling.learning_rate,
            weight_decay=config.scaling.weight_decay,
            gradient_clip=config.scaling.gradient_clip,
            lambda_alpha=config.scaling.lambda_alpha,
            seed=seed + 41_101,
            curve_steps=config.scaling.curve_steps,
            permute_codes=(variant == "permuted_z"),
            evaluation_callback=callback,
        )
        history = result.history
        metadata = {
            "dimension": 32,
            "unique_training_geometries": geometries,
            "training_tasks": len(train),
            "supervised_examples": len(h_train),
            "total_examples_seen": result.examples_seen,
            "training_seconds": result.elapsed_seconds,
            "batch_size": batch_size,
            "steps": steps,
            "variant": variant,
            "mode": mode,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "peak_allocated_bytes": result.peak_allocated_bytes,
            "peak_reserved_bytes": result.peak_reserved_bytes,
        }
        _save_model(path, model, history=history, metadata=metadata)
    return model, {"history": history, **metadata}, benchmark, train


def run_data_scaling(
    config: D32GeneralizationConfig,
    root: Path,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output = root / "d32_data_scaling"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    endpoints = {config.scaling.geometry_counts[0], config.scaling.geometry_counts[-1]}
    for geometries in config.scaling.geometry_counts:
        seeds = list(config.scaling.grid_seeds)
        if geometries in endpoints:
            seeds = list(dict.fromkeys(seeds + config.scaling.endpoint_seeds))
        for seed in seeds:
            for mode in ("single_target", "multimode"):
                model, metadata, benchmark, _ = _fit_scaling_model(
                    config,
                    output,
                    geometries=geometries,
                    seed=seed,
                    mode=mode,
                    device=device,
                )
                _, _, _, test, test_starts = _data(config, geometries, seed)
                packed_test, h_test = _aligned(test, test_starts, device)
                evaluation = _eval(
                    model,
                    packed_test,
                    h_test,
                    benchmark,
                    mode,
                    config.scaling.lambda_alpha,
                )
                rows.append(
                    {
                        "dimension": 32,
                        "unique_training_geometries": geometries,
                        "seed": seed,
                        "mode": mode,
                        **evaluation,
                        **{
                            key: value
                            for key, value in metadata.items()
                            if key != "history"
                        },
                    }
                )
                curves.extend(
                    {
                        "unique_training_geometries": geometries,
                        "seed": seed,
                        "mode": mode,
                        **point,
                    }
                    for point in metadata.get("history", [])
                )
    write_csv(output / "data_scaling_per_seed.csv", rows)
    write_csv(
        output / "data_scaling_summary.csv",
        _summary(
            rows,
            ("unique_training_geometries", "mode"),
            SCALING_METRICS,
        ),
    )
    write_csv(output / "training_generalization_curves.csv", curves)
    smallest = config.scaling.geometry_counts[0]
    largest = config.scaling.geometry_counts[-1]
    paired_rows = []
    indexed = {
        (
            int(row["seed"]),
            str(row["mode"]),
            int(row["unique_training_geometries"]),
        ): row
        for row in rows
    }
    for seed in config.scaling.endpoint_seeds:
        for mode in ("single_target", "multimode"):
            base = indexed[(seed, mode, smallest)]
            changed = indexed[(seed, mode, largest)]
            paired_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "smallest_geometries": smallest,
                    "largest_geometries": largest,
                    **{
                        f"delta_{metric}": float(changed[metric]) - float(base[metric])
                        for metric in SCALING_METRICS
                    },
                }
            )
    write_csv(output / "data_scaling_paired.csv", paired_rows)
    write_csv(
        output / "data_scaling_paired_summary.csv",
        _summary(
            paired_rows,
            ("mode",),
            tuple(f"delta_{metric}" for metric in SCALING_METRICS),
        ),
    )
    return rows, curves


def _probe_spec(
    kind: str, hidden: int, device: torch.device, seed: int
) -> DirectionProbe:
    input_dimension = 64 if kind in {"state_goal_linear", "state_goal_mlp"} else 32
    hidden_dimension = hidden if kind == "state_goal_mlp" else None
    return DirectionProbe(
        input_dimension,
        32,
        hidden_dimension=hidden_dimension,
        seed=seed + 71_001,
        device=device,
    )


def run_probes(
    config: D32GeneralizationConfig,
    root: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    output = root / "d32_identifiability"
    (output / "models").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    kinds = (
        "goal",
        "state",
        "state_goal_linear",
        "state_goal_mlp",
        "privileged_geometry",
    )
    for seed in config.probes.seeds:
        benchmark, train, _, test, test_starts = _data(
            config, config.probes.training_geometries, seed
        )
        packed_train, h_train = _training_batch(
            train,
            trials=config.scaling.starts_per_task,
            seed=seed + 41_001,
            device=device,
        )
        packed_test, h_test = _aligned(test, test_starts, device)
        _, _, target_train, _ = useful_targets(h_train, packed_train)
        for kind in kinds:
            make_probe = lambda: _probe_spec(
                kind, config.probes.hidden_dimension, device, seed
            )
            path = output / "models" / f"{kind}_seed_{seed}.pt"
            if path.exists():
                probe, payload = _load_model(path, make_probe, device)
                training_seconds = float(
                    payload.get("metadata", {}).get("training_seconds", 0.0)
                )
            else:
                probe = make_probe()
                history, training_seconds = train_direction_probe(
                    probe,
                    probe_inputs(kind, h_train, packed_train),
                    target_train,
                    steps=config.probes.steps,
                    batch_size=config.probes.batch_size,
                    learning_rate=config.probes.learning_rate,
                    weight_decay=config.probes.weight_decay,
                    seed=seed + 71_101,
                )
                _save_model(
                    path,
                    probe,
                    history=history,
                    metadata={
                        "kind": kind,
                        "training_seconds": training_seconds,
                        "training_geometries": config.probes.training_geometries,
                        "training_examples": len(h_train),
                        "test_labels_used": False,
                    },
                )
            metrics = evaluate_direction_probe(
                probe,
                probe_inputs(kind, h_test, packed_test),
                h_test,
                packed_test,
                refinement_steps=benchmark.base.base.training.refinement_steps,
                refinement_rate=benchmark.base.base.training.refinement_rate,
                useful_threshold=benchmark.base.base.landscape.useful_threshold,
            )
            rows.append(
                {
                    "seed": seed,
                    "inputs": kind,
                    "model": "MLP" if kind == "state_goal_mlp" else "Linear",
                    "privileged": kind == "privileged_geometry",
                    "training_seconds": training_seconds,
                    **metrics,
                }
            )
    write_csv(output / "probe_per_seed.csv", rows)
    write_csv(
        output / "probe_summary.csv",
        _summary(
            rows,
            ("inputs", "model", "privileged"),
            ("heldout_cosine", "angular_error_degrees", "recall"),
        ),
    )
    return rows


@torch.no_grad()
def _evaluate_direction_set(
    directions: torch.Tensor,
    alphas: torch.Tensor,
    starts: torch.Tensor,
    tasks: TorchTaskBatch,
    benchmark,
    *,
    mode: str,
    lambda_alpha: float,
) -> dict[str, float]:
    candidates = starts[:, None, :] + alphas[:, :, None] * directions
    useful_direction, useful_alpha, target_direction, _ = useful_targets(starts, tasks)
    cosine_all = torch.einsum("bkd,bmd->bkm", directions, useful_direction)
    target_cosine = torch.einsum("bkd,bd->bk", directions, target_direction)
    target_count = int(tasks.n_useful[0].cpu())
    if mode == "single_target":
        matched_cosine = target_cosine[:, :1]
        goal_alpha = torch.gather(useful_alpha, 1, tasks.goal_index[:, None])
        matched_alpha_error = alphas[:, :1] - goal_alpha
    else:
        pair_alpha = alphas[:, :, None] - useful_alpha[:, None, :]
        cost = 1.0 - cosine_all + lambda_alpha * pair_alpha.square()
        assignment = exact_assignment_indices(cost, target_count)
        target = torch.arange(target_count, device=starts.device)[None, :].expand(
            len(starts), -1
        )
        batch = torch.arange(len(starts), device=starts.device)[:, None]
        matched_cosine = cosine_all[batch, assignment, target]
        matched_alpha_error = pair_alpha[batch, assignment, target]
    refined = tasks.refine(
        candidates,
        steps=benchmark.base.base.training.refinement_steps,
        rate=benchmark.base.base.training.refinement_rate,
    )
    quality = tasks.quality(refined)
    regions = tasks.regions(refined)
    threshold = benchmark.base.base.landscape.useful_threshold
    useful = (
        (regions >= 1) & (regions <= tasks.n_useful[:, None]) & (quality >= threshold)
    )
    target_hit = (regions == tasks.target_region[:, None]) & (quality >= threshold)
    proposal = torch.any(useful, dim=1)
    useful_count = torch.as_tensor(
        [
            float(torch.unique(regions[index][useful[index]]).numel())
            for index in range(len(starts))
        ],
        device=starts.device,
    )
    return {
        "best_target_cosine": float(torch.mean(torch.amax(target_cosine, dim=1)).cpu()),
        "best_useful_cosine": float(
            torch.mean(torch.amax(cosine_all, dim=(1, 2))).cpu()
        ),
        "mean_matched_cosine": float(torch.mean(matched_cosine).cpu()),
        "mean_angular_error_degrees": float(
            torch.mean(
                torch.rad2deg(torch.acos(torch.clamp(matched_cosine, -1.0, 1.0)))
            ).cpu()
        ),
        "mean_alpha_absolute_error": float(
            torch.mean(torch.abs(matched_alpha_error)).cpu()
        ),
        "proposal_recall": float(torch.mean(proposal.float()).cpu()),
        "target_proposal_recall": float(
            torch.mean(torch.any(target_hit, dim=1).float()).cpu()
        ),
        "useful_region_count": float(torch.mean(useful_count).cpu()),
        "candidate_mean_quality": float(torch.mean(quality).cpu()),
        "candidate_decoy_rate": float(
            torch.mean((regions > tasks.n_useful[:, None]).float()).cpu()
        ),
    }


def _standardized_features(
    train_h: torch.Tensor,
    train_tasks: TorchTaskBatch,
    test_h: torch.Tensor,
    test_tasks: TorchTaskBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    train = torch.cat([train_h, train_tasks.goals], dim=-1)
    test = torch.cat([test_h, test_tasks.goals], dim=-1)
    mean = torch.mean(train, dim=0, keepdim=True)
    scale = torch.std(train, dim=0, correction=0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / scale, (test - mean) / scale


def _nearest_indices(
    train: torch.Tensor,
    test: torch.Tensor,
    metric: str,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if metric == "euclidean":
        distance = torch.cdist(test, train)
    elif metric == "cosine":
        similarity = torch.matmul(
            F.normalize(test, dim=-1, eps=1e-8),
            F.normalize(train, dim=-1, eps=1e-8).T,
        )
        distance = 1.0 - similarity
    else:
        raise ValueError(metric)
    return torch.topk(distance, k=min(k, len(train)), largest=False, dim=1)


def run_nearest_neighbor(
    config: D32GeneralizationConfig,
    root: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    output = root / "d32_identifiability"
    rows: list[dict[str, object]] = []
    geometries = config.probes.training_geometries
    candidates = 8
    for seed in config.probes.seeds:
        benchmark, train, _, test, test_starts = _data(config, geometries, seed)
        packed_train, h_train = _training_batch(
            train,
            trials=config.scaling.starts_per_task,
            seed=seed + 41_001,
            device=device,
        )
        packed_test, h_test = _aligned(test, test_starts, device)
        train_x, test_x = _standardized_features(
            h_train, packed_train, h_test, packed_test
        )
        useful_dir, useful_alpha, target_dir, target_alpha = useful_targets(
            h_train, packed_train
        )
        for metric in ("euclidean", "cosine"):
            distance, nearest = _nearest_indices(train_x, test_x, metric, 1)
            index = nearest[:, 0]
            single_dir = target_dir.index_select(0, index)[:, None, :].expand(
                -1, candidates, -1
            )
            single_alpha = target_alpha.index_select(0, index)[:, None].expand(
                -1, candidates
            )
            multi_dir = useful_dir.index_select(0, index)
            multi_alpha = useful_alpha.index_select(0, index)
            repeats = (candidates + multi_dir.shape[1] - 1) // multi_dir.shape[1]
            multi_dir = multi_dir.repeat(1, repeats, 1)[:, :candidates]
            multi_alpha = multi_alpha.repeat(1, repeats)[:, :candidates]
            for mode, direction, alpha in (
                ("single_target", single_dir, single_alpha),
                ("multimode", multi_dir, multi_alpha),
            ):
                result = _evaluate_direction_set(
                    direction,
                    alpha,
                    h_test,
                    packed_test,
                    benchmark,
                    mode=mode,
                    lambda_alpha=config.scaling.lambda_alpha,
                )
                rows.append(
                    {
                        "seed": seed,
                        "metric": metric,
                        "mode": mode,
                        "mean_nearest_input_distance": float(
                            torch.mean(distance[:, 0]).cpu()
                        ),
                        **result,
                    }
                )
    write_csv(output / "nearest_neighbor_per_seed.csv", rows)
    write_csv(
        output / "nearest_neighbor_summary.csv",
        _summary(
            rows,
            ("metric", "mode"),
            SCALING_METRICS,
        ),
    )
    return rows


def _sample_condition(
    benchmark,
    train: Sequence[V2Task],
    seed: int,
    condition: str,
) -> tuple[list[V2Task], list[list[np.ndarray]]]:
    if condition == "same_geometry_new_starts":
        tasks = list(train)
        starts = _make_starts(
            tasks,
            trials=benchmark.base.base.trials_per_task,
            seed=seed + 61_001,
        )
        return tasks, starts
    variant = "iid" if condition == "new_iid_geometry" else condition
    family = _family(benchmark)
    variant_offset = {
        "iid": 0,
        "rotated": 1,
        "translated": 2,
        "width_shift": 3,
        "more_peaks": 4,
        "more_decoys": 5,
        "new_goals": 6,
        "unseen": 7,
    }[variant]
    tasks = family.sample_tasks(
        geometries=benchmark.base.base.test_geometries,
        seed=seed + 70_000 + 1_000 * variant_offset,
        variant=variant,
        all_goals=True,
    )
    starts = _make_starts(
        tasks,
        trials=benchmark.base.base.trials_per_task,
        seed=seed + 70_001 + 1_000 * variant_offset,
    )
    return tasks, starts


def run_geometry_generalization(
    config: D32GeneralizationConfig,
    root: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    output = root / "d32_geometry_generalization"
    output.mkdir(parents=True, exist_ok=True)
    scaling_output = root / "d32_data_scaling"
    geometries = config.scaling.geometry_counts[-1]
    rows: list[dict[str, object]] = []
    conditions = ["same_geometry_new_starts", "new_iid_geometry"] + [
        value for value in config.generalization_variants if value != "iid"
    ]
    for seed in config.scaling.endpoint_seeds:
        for mode in ("single_target", "multimode"):
            model, _, benchmark, train = _fit_scaling_model(
                config,
                scaling_output,
                geometries=geometries,
                seed=seed,
                mode=mode,
                device=device,
            )
            for condition in conditions:
                tasks, starts = _sample_condition(benchmark, train, seed, condition)
                packed, h = _aligned(tasks, starts, device)
                metrics = _eval(
                    model, packed, h, benchmark, mode, config.scaling.lambda_alpha
                )
                rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "condition": condition,
                        "condition_group": (
                            "train_geometry_new_starts"
                            if condition == "same_geometry_new_starts"
                            else "new_iid_geometry"
                            if condition == "new_iid_geometry"
                            else "heldout_family"
                        ),
                        **metrics,
                    }
                )
    write_csv(output / "geometry_generalization_per_seed.csv", rows)
    write_csv(
        output / "geometry_generalization_summary.csv",
        _summary(
            rows,
            ("mode", "condition"),
            SCALING_METRICS,
        ),
    )
    grouped = []
    by_seed_group: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_seed_group.setdefault(
            (int(row["seed"]), str(row["mode"]), str(row["condition_group"])), []
        ).append(row)
    for (seed, mode, group), values in by_seed_group.items():
        grouped.append(
            {
                "seed": seed,
                "mode": mode,
                "condition_group": group,
                **{
                    metric: float(np.mean([float(value[metric]) for value in values]))
                    for metric in SCALING_METRICS
                },
            }
        )
    write_csv(
        output / "geometry_group_summary.csv",
        _summary(
            grouped,
            ("mode", "condition_group"),
            SCALING_METRICS,
        ),
    )
    grouped_index = {
        (int(row["seed"]), str(row["mode"]), str(row["condition_group"])): row
        for row in grouped
    }
    paired = []
    for seed in config.scaling.endpoint_seeds:
        for mode in ("single_target", "multimode"):
            familiar = grouped_index[(seed, mode, "train_geometry_new_starts")]
            iid = grouped_index[(seed, mode, "new_iid_geometry")]
            paired.append(
                {
                    "seed": seed,
                    "mode": mode,
                    **{
                        f"delta_familiar_minus_iid_{metric}": float(familiar[metric])
                        - float(iid[metric])
                        for metric in SCALING_METRICS
                    },
                }
            )
    write_csv(output / "geometry_paired.csv", paired)
    write_csv(
        output / "geometry_paired_summary.csv",
        _summary(
            paired,
            ("mode",),
            tuple(f"delta_familiar_minus_iid_{metric}" for metric in SCALING_METRICS),
        ),
    )
    return rows


def _set_disagreement(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Permutation-invariant 1-cos set distance for aligned batches [B,M,d]."""
    cost = 1.0 - torch.einsum("bmd,bnd->bmn", first, second)
    assignment = exact_assignment_indices(cost, second.shape[1])
    target = torch.arange(second.shape[1], device=first.device)[None, :].expand(
        len(first), -1
    )
    batch = torch.arange(len(first), device=first.device)[:, None]
    return torch.mean(cost[batch, assignment, target], dim=1)


def run_multimode_identifiability(
    config: D32GeneralizationConfig,
    root: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    output = root / "d32_multimode_analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    geometries = config.probes.training_geometries
    scaling_output = root / "d32_data_scaling"
    for seed in config.probes.seeds:
        benchmark, train, _, test, _ = _data(config, geometries, seed)
        packed_train, h_train = _single_start_batch(train, seed + 81_001, device)
        packed_test, h_test = _single_start_batch(test, seed + 82_001, device)
        train_x, test_x = _standardized_features(
            h_train, packed_train, h_test, packed_test
        )
        train_direction, _, _, _ = useful_targets(h_train, packed_train)
        test_direction, _, _, _ = useful_targets(h_test, packed_test)
        distance, neighbor = _nearest_indices(
            train_x, test_x, "euclidean", config.identifiability.nearest_neighbors
        )
        disagreements = []
        for rank in range(neighbor.shape[1]):
            comparison = train_direction.index_select(0, neighbor[:, rank])
            disagreements.append(_set_disagreement(comparison, test_direction))
        disagreement = torch.stack(disagreements, dim=1)
        local_target_variance = torch.mean(disagreement, dim=1)
        model, _, _, _ = _fit_scaling_model(
            config,
            scaling_output,
            geometries=geometries,
            seed=seed,
            mode="multimode",
            device=device,
        )
        evaluation = evaluate_supervised(
            model,
            packed_test,
            h_test,
            mode="multimode",
            lambda_alpha=config.scaling.lambda_alpha,
            refinement_steps=benchmark.base.base.training.refinement_steps,
            refinement_rate=benchmark.base.base.training.refinement_rate,
            useful_threshold=benchmark.base.base.landscape.useful_threshold,
        )
        csd_error = torch.as_tensor(
            [1.0 - float(value["mean_matched_cosine"]) for value in evaluation.per_set],
            device=device,
        )
        for index in range(len(h_test)):
            rows.append(
                {
                    "seed": seed,
                    "task_index": index,
                    "nearest_input_distance": float(distance[index, 0].cpu()),
                    "nearest_target_set_disagreement": float(
                        disagreement[index, 0].cpu()
                    ),
                    "mean_k_neighbor_disagreement": float(
                        local_target_variance[index].cpu()
                    ),
                    "csd_generalization_error": float(csd_error[index].cpu()),
                }
            )
        input_distance = distance[:, 0].detach().cpu().numpy()
        set_error = disagreement[:, 0].detach().cpu().numpy()
        variance = local_target_variance.detach().cpu().numpy()
        csd = csd_error.detach().cpu().numpy()
        summary_rows.append(
            {
                "seed": seed,
                "mean_nearest_input_distance": float(np.mean(input_distance)),
                "mean_nearest_target_set_disagreement": float(np.mean(set_error)),
                "mean_local_target_variance": float(np.mean(variance)),
                "corr_input_distance_target_disagreement": float(
                    np.corrcoef(input_distance, set_error)[0, 1]
                ),
                "corr_local_variance_csd_error": float(
                    np.corrcoef(variance, csd)[0, 1]
                ),
                "high_ambiguity_rate": float(
                    np.mean(
                        (input_distance <= np.median(input_distance))
                        & (set_error >= np.median(set_error))
                    )
                ),
            }
        )
    write_csv(output / "local_identifiability_per_task.csv", rows)
    write_csv(output / "local_identifiability_per_seed.csv", summary_rows)
    write_csv(
        output / "local_identifiability_summary.csv",
        _summary(
            summary_rows,
            (),
            (
                "mean_nearest_input_distance",
                "mean_nearest_target_set_disagreement",
                "mean_local_target_variance",
                "corr_input_distance_target_disagreement",
                "corr_local_variance_csd_error",
                "high_ambiguity_rate",
            ),
        ),
    )
    return summary_rows


def run_slot_ablations(
    config: D32GeneralizationConfig,
    root: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    output = root / "d32_multimode_analysis"
    scaling_output = root / "d32_data_scaling"
    rows: list[dict[str, object]] = []
    geometries = config.slots.training_geometries
    for seed in config.slots.seeds:
        for variant in ("fixed_z", "permuted_z", "no_z_set"):
            base_output = scaling_output if variant == "fixed_z" else output
            model, metadata, benchmark, _ = _fit_scaling_model(
                config,
                base_output,
                geometries=geometries,
                seed=seed,
                mode="multimode",
                device=device,
                variant=variant,
            )
            _, _, _, test, test_starts = _data(config, geometries, seed)
            packed_test, h_test = _aligned(test, test_starts, device)
            result = _eval(
                model,
                packed_test,
                h_test,
                benchmark,
                "multimode",
                config.scaling.lambda_alpha,
            )
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "parameter_count": metadata.get(
                        "parameter_count", sum(p.numel() for p in model.parameters())
                    ),
                    **result,
                }
            )
    write_csv(output / "slot_ablation_per_seed.csv", rows)
    write_csv(
        output / "slot_ablation_summary.csv",
        _summary(
            rows,
            ("variant",),
            SCALING_METRICS,
        ),
    )
    indexed = {(int(row["seed"]), str(row["variant"])): row for row in rows}
    paired = []
    for seed in config.slots.seeds:
        fixed = indexed[(seed, "fixed_z")]
        for variant in ("permuted_z", "no_z_set"):
            changed = indexed[(seed, variant)]
            paired.append(
                {
                    "seed": seed,
                    "variant": variant,
                    **{
                        f"delta_{metric}": float(changed[metric]) - float(fixed[metric])
                        for metric in SCALING_METRICS
                    },
                }
            )
    write_csv(output / "slot_ablation_paired.csv", paired)
    write_csv(
        output / "slot_ablation_paired_summary.csv",
        _summary(
            paired,
            ("variant",),
            tuple(f"delta_{metric}" for metric in SCALING_METRICS),
        ),
    )
    return rows


def run_batch_scaling(
    config: D32GeneralizationConfig,
    root: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    output = root / "d32_data_scaling"
    seed = config.scaling.endpoint_seeds[0]
    benchmark, train, _, _, _ = _data(config, config.scaling.geometry_counts[-1], seed)
    packed, h = _training_batch(
        train,
        trials=config.scaling.starts_per_task,
        seed=seed + 41_001,
        device=device,
    )
    capacity = SupervisedCapacityConfig(
        primary_dimension=32,
        dimensions=[32],
        batch_sizes=list(config.batch_sizes),
        throughput_warmup_steps=config.batch_warmup_steps,
        throughput_timing_steps=config.batch_timing_steps,
        lambda_alpha=config.scaling.lambda_alpha,
        learning_rate=config.scaling.learning_rate,
        weight_decay=config.scaling.weight_decay,
    )
    rows = benchmark_supervised_batches(
        lambda: _make_csd(benchmark, seed, device),
        packed,
        h,
        capacity,
        seed=seed + 91_001,
    )
    write_csv(output / "batch_scaling.csv", rows)
    return rows


def _endpoint_checkpoint(root: Path, seed: int, lambda_anchor: float) -> Path:
    label = str(lambda_anchor).replace(".", "p")
    return root / "models" / f"endpoint_lambda_{label}_seed_{seed}.pt"


def _fit_endpoint_policy(
    config: D32GeneralizationConfig,
    output: Path,
    *,
    seed: int,
    lambda_anchor: float,
    device: torch.device,
) -> tuple[
    torch.nn.Module, dict[str, object], object, list[V2Task], list[list[np.ndarray]]
]:
    benchmark, train, validation, test, test_starts = _data(config, 10, seed)
    family = _family(benchmark)
    train, train_starts, validation, validation_starts, test, test_starts = _datasets(
        family, benchmark, seed
    )
    packed_train = pack_tasks(train, device=device, dtype=torch.float32)
    h_train = torch.as_tensor(
        np.stack(train_starts), dtype=torch.float32, device=device
    )
    packed_validation = pack_tasks(validation, device=device, dtype=torch.float32)
    h_validation = torch.as_tensor(
        np.stack(validation_starts), dtype=torch.float32, device=device
    )
    environment = _environment(benchmark)
    make_policy = lambda: _policy(benchmark, seed, device)
    path = _endpoint_checkpoint(output, seed, lambda_anchor)
    if path.exists():
        policy, payload = _load_model(path, make_policy, device)
        metadata = dict(payload.get("metadata", {}))
        history = list(payload.get("history", []))
        validation_history = list(payload.get("validation_history", []))
    else:
        policy = make_policy()
        training = benchmark.rl.training
        training.max_candidate_evaluations = config.endpoint_anchor.candidate_budget
        training.checkpoint_budgets = sorted(
            set(
                [
                    value
                    for value in training.checkpoint_budgets
                    if value <= training.max_candidate_evaluations
                ]
                + [0, training.max_candidate_evaluations]
            )
        )

        def validation_callback(current, budget: int) -> dict[str, float]:
            return evaluate_anchor_policy(
                current,
                environment,
                packed_validation,
                h_validation,
                useful_threshold=benchmark.base.base.landscape.useful_threshold,
                capture_distance=config.endpoint_anchor.capture_distance,
                action_seed=seed + 92_000 + budget,
                deterministic=False,
            )

        result = train_policy_endpoint_anchor(
            policy,
            environment,
            packed_train,
            h_train,
            training,
            seed=seed + 92_101,
            lambda_anchor=lambda_anchor,
            anchor_radius=config.endpoint_anchor.anchor_radius,
            validation_callback=validation_callback,
        )
        history = result.history
        validation_history = result.validation_history
        metadata = {
            "seed": seed,
            "lambda_anchor": lambda_anchor,
            "anchor_radius": config.endpoint_anchor.anchor_radius,
            "penalty_endpoint": "final_refined_state",
            "q_detached": True,
            "refinement_detached": True,
            "mode_preservation": False,
            "candidate_evaluations": result.candidate_evaluations,
            "q_equivalent_evaluations": result.q_equivalent_evaluations,
            "optimizer_steps": result.optimizer_steps,
            "training_seconds": result.elapsed_seconds,
            "peak_allocated_bytes": result.peak_allocated_bytes,
            "peak_reserved_bytes": result.peak_reserved_bytes,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in policy.state_dict().items()
                },
                "history": history,
                "validation_history": validation_history,
                "metadata": metadata,
            },
            path,
        )
    return (
        policy,
        {
            **metadata,
            "history": history,
            "validation_history": validation_history,
        },
        benchmark,
        test,
        test_starts,
    )


def _select_endpoint_lambda(
    summary: list[dict[str, object]],
    tolerance: float,
) -> tuple[float, list[dict[str, object]]]:
    baseline = next(row for row in summary if float(row["lambda_anchor"]) == 0.0)
    baseline_decoy = float(baseline["candidate_decoy_rate_mean"])
    scored = []
    for row in summary:
        decoy_excess = max(
            0.0,
            float(row["candidate_decoy_rate_mean"]) - baseline_decoy - tolerance,
        )
        score = (
            float(row["proposal_recall_mean"])
            + 0.10 * float(row["candidate_mean_quality_mean"])
            - decoy_excess
        )
        scored.append({**row, "validation_selection_score": score})
    selected = max(
        scored,
        key=lambda row: (
            float(row["validation_selection_score"]),
            -float(row["lambda_anchor"]),
        ),
    )
    return float(selected["lambda_anchor"]), scored


def run_endpoint_anchor(
    config: D32GeneralizationConfig,
    root: Path,
    device: torch.device,
) -> dict[str, object]:
    output = root / "d32_endpoint_anchor"
    output.mkdir(parents=True, exist_ok=True)
    validation_rows: list[dict[str, object]] = []
    for seed in config.endpoint_anchor.validation_seeds:
        for lambda_anchor in config.endpoint_anchor.lambda_values:
            policy, metadata, benchmark, _, _ = _fit_endpoint_policy(
                config,
                output,
                seed=seed,
                lambda_anchor=lambda_anchor,
                device=device,
            )
            last = metadata.get("validation_history", [])[-1]
            validation_rows.append(
                {
                    "seed": seed,
                    "lambda_anchor": lambda_anchor,
                    **{
                        key: value
                        for key, value in last.items()
                        if key not in {"requested_budget"}
                    },
                }
            )
    validation_metrics = (
        "proposal_recall",
        "useful_candidate_rate",
        "useful_region_count",
        "candidate_decoy_rate",
        "candidate_anchor_rate",
        "candidate_mean_quality",
        "best_useful_alignment",
        "capture_entry_rate",
        "alpha_mean",
        "alpha_sd",
        "effective_directional_rank",
    )
    validation_summary = _summary(
        validation_rows, ("lambda_anchor",), validation_metrics
    )
    selected, scored = _select_endpoint_lambda(
        validation_summary, config.endpoint_anchor.decoy_tolerance
    )
    write_csv(output / "validation_lambda_per_seed.csv", validation_rows)
    write_csv(output / "validation_lambda_summary.csv", scored)
    (output / "selected_lambda.json").write_text(
        json.dumps(
            {
                "selected_lambda_anchor": selected,
                "selection_rule_predeclared": "proposal_recall + 0.1*quality - decoy_excess_over_baseline_plus_tolerance; tie -> smaller lambda",
                "decoy_tolerance": config.endpoint_anchor.decoy_tolerance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    main_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    training_curve_rows: list[dict[str, object]] = []
    lambdas = list(dict.fromkeys([0.0, selected]))
    for seed in config.endpoint_anchor.main_seeds:
        for lambda_anchor in lambdas:
            policy, metadata, benchmark, test, test_starts = _fit_endpoint_policy(
                config,
                output,
                seed=seed,
                lambda_anchor=lambda_anchor,
                device=device,
            )
            packed_test, h_test = _aligned(test, test_starts, device)
            result = evaluate_anchor_policy(
                policy,
                _environment(benchmark),
                packed_test,
                h_test,
                useful_threshold=benchmark.base.base.landscape.useful_threshold,
                capture_distance=config.endpoint_anchor.capture_distance,
                action_seed=seed + 93_001,
                deterministic=False,
            )
            label = "baseline" if lambda_anchor == 0.0 else "endpoint_anchor"
            main_rows.append(
                {
                    "seed": seed,
                    "variant": label,
                    "lambda_anchor": lambda_anchor,
                    **result,
                    **{
                        key: value
                        for key, value in metadata.items()
                        if key not in {"history", "validation_history"}
                    },
                }
            )
            curve_rows.extend(
                {
                    "seed": seed,
                    "variant": label,
                    "lambda_anchor": lambda_anchor,
                    **point,
                }
                for point in metadata.get("validation_history", [])
            )
            training_curve_rows.extend(
                {
                    "seed": seed,
                    "variant": label,
                    "lambda_anchor": lambda_anchor,
                    **point,
                }
                for point in metadata.get("history", [])
            )
    write_csv(output / "endpoint_anchor_per_seed.csv", main_rows)
    write_csv(
        output / "endpoint_anchor_summary.csv",
        _summary(main_rows, ("variant", "lambda_anchor"), validation_metrics),
    )
    write_csv(output / "endpoint_anchor_curves.csv", curve_rows)
    write_csv(output / "endpoint_anchor_training_curves.csv", training_curve_rows)
    paired = []
    if selected != 0.0:
        by_key = {(int(row["seed"]), str(row["variant"])): row for row in main_rows}
        for seed in config.endpoint_anchor.main_seeds:
            base = by_key[(seed, "baseline")]
            changed = by_key[(seed, "endpoint_anchor")]
            paired.append(
                {
                    "seed": seed,
                    **{
                        f"delta_{metric}": float(changed[metric]) - float(base[metric])
                        for metric in validation_metrics
                    },
                }
            )
        write_csv(output / "endpoint_anchor_paired.csv", paired)
        write_csv(
            output / "endpoint_anchor_paired_summary.csv",
            _summary(
                paired, (), tuple(f"delta_{metric}" for metric in validation_metrics)
            ),
        )
    return {
        "selected_lambda_anchor": selected,
        "main_rows": len(main_rows),
        "penalty_uses_final_refined_endpoint": True,
        "mode_preservation_off": True,
    }


def run_d32_generalization_study(
    config: D32GeneralizationConfig,
    *,
    results_root: Path,
    device: torch.device,
    runtime: RuntimeInfo,
    sections: Iterable[str],
) -> dict[str, object]:
    selected = set(sections)
    results_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config.to_dict(),
        "runtime": runtime.to_dict(),
        "scientific_constraints": {
            "dimension": 32,
            "does_not_run_d64": True,
            "existing_results_modified": False,
            "test_set_used_for_tuning": False,
            "endpoint_q_and_refinement_detached": True,
        },
    }
    (results_root / "d32_generalization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    completed: dict[str, object] = {}
    if "scaling" in selected:
        scaling, curves = run_data_scaling(config, results_root, device)
        completed["scaling_rows"] = len(scaling)
        completed["curve_rows"] = len(curves)
    if "probes" in selected:
        completed["probe_rows"] = len(run_probes(config, results_root, device))
        completed["nearest_neighbor_rows"] = len(
            run_nearest_neighbor(config, results_root, device)
        )
    if "geometry" in selected:
        completed["geometry_rows"] = len(
            run_geometry_generalization(config, results_root, device)
        )
    if "multimode" in selected:
        completed["identifiability_rows"] = len(
            run_multimode_identifiability(config, results_root, device)
        )
        completed["slot_rows"] = len(run_slot_ablations(config, results_root, device))
    if "endpoint" in selected:
        completed["endpoint"] = run_endpoint_anchor(config, results_root, device)
    if "batch" in selected:
        completed["batch_rows"] = len(run_batch_scaling(config, results_root, device))
    completion_path = results_root / "d32_generalization_completion.json"
    accumulated: dict[str, object] = {}
    if completion_path.exists():
        accumulated = json.loads(completion_path.read_text(encoding="utf-8"))
    accumulated.update(completed)
    completion_path.write_text(json.dumps(accumulated, indent=2), encoding="utf-8")
    return completed
