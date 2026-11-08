"""Sequential supervised-representability -> anchor-escape CSD diagnosis."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from creativity_toy.analysis_utils import statistics
from creativity_toy.experiments.study_v2 import _make_starts
from creativity_toy.landscapes_d16 import geometry_diagnostics
from creativity_toy.landscapes_v2 import V2Task

from .config_capacity_diagnostic import CapacityDiagnosticConfig
from .contrastive_evaluation import flatten_starts
from .runtime import RuntimeInfo
from .study import write_csv
from .study_d16 import _datasets, _environment, _policy
from .study_failure_diagnostics import _dimension_config, _family
from .supervised_capacity import (
    LinearDirectionPolicy,
    benchmark_supervised_batches,
    evaluate_supervised,
    train_supervised,
)
from .task_batch import TorchTaskBatch, pack_tasks
from .training_anchor_escape import (
    evaluate_anchor_policy,
    train_policy_anchor_escape,
)

SUPERVISED_METRICS = (
    "best_target_cosine",
    "candidate_zero_target_cosine",
    "best_useful_cosine",
    "mean_matched_cosine",
    "mean_angular_error_degrees",
    "mean_alpha_absolute_error",
    "mean_matched_proximity",
    "matched_capture_entry_rate",
    "proposal_recall",
    "candidate_zero_target_recall",
    "target_proposal_recall",
    "useful_candidate_rate",
    "useful_region_count",
    "candidate_mean_quality",
    "candidate_decoy_rate",
    "candidate_anchor_rate",
    "effective_directional_rank",
    "alpha_mean",
    "alpha_sd",
)


def _aligned(
    tasks: Sequence[V2Task],
    nested_starts: Sequence[Sequence[np.ndarray]],
    device: torch.device,
) -> tuple[TorchTaskBatch, torch.Tensor]:
    points, task_ids, _ = flatten_starts(nested_starts)
    packed = pack_tasks(tasks, device=device, dtype=torch.float32)
    indices = torch.as_tensor(task_ids, dtype=torch.long, device=device)
    return (
        packed.index_select(indices),
        torch.as_tensor(points, dtype=torch.float32, device=device),
    )


def _model(
    architecture: str,
    benchmark,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    if architecture == "current_mlp":
        return _policy(benchmark, seed, device)
    if architecture == "linear":
        base = benchmark.base.base
        return LinearDirectionPolicy(
            dimension=benchmark.dimension,
            candidates=base.training.candidates,
            latent_dimension=base.training.latent_dimension,
            hidden_dimension=base.training.hidden_dimension,
            seed=seed + 10_020,
            device=device,
            alpha_max=benchmark.rl.alpha_max,
        )
    raise ValueError(architecture)


def _save_checkpoint(
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


def _load_checkpoint(
    path: Path,
    make_model: Callable[[], torch.nn.Module],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = make_model()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _summary_rows(
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
            stat = statistics([float(value[metric]) for value in values])
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


def _train_or_load_supervised(
    *,
    architecture: str,
    mode: str,
    dimension: int,
    seed: int,
    benchmark,
    capacity: CapacityDiagnosticConfig,
    train_tasks: Sequence[V2Task],
    test_tasks: Sequence[V2Task],
    test_starts: Sequence[Sequence[np.ndarray]],
    selected_batch_size: int,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    training_nested = _make_starts(
        train_tasks,
        trials=capacity.supervised.starts_per_training_task,
        seed=seed + 41_001,
    )
    packed_train, h_train = _aligned(train_tasks, training_nested, device)
    packed_test, h_test = _aligned(test_tasks, test_starts, device)
    model_path = (
        output / "models" / f"d{dimension}_{architecture}_{mode}_seed_{seed}.pt"
    )
    make_model = lambda: _model(architecture, benchmark, seed, device)
    steps = (
        capacity.supervised.single_steps
        if mode == "single_target"
        else capacity.supervised.multimode_steps
    )
    if model_path.exists():
        model, payload = _load_checkpoint(model_path, make_model, device)
        history = list(payload.get("history", []))
        metadata = dict(payload.get("metadata", {}))
    else:
        model = make_model()
        result = train_supervised(
            model,
            packed_train,
            h_train,
            capacity.supervised,
            mode=mode,
            steps=steps,
            batch_size=min(selected_batch_size, len(h_train)),
            seed=seed + 41_101,
            refinement_steps=benchmark.base.base.training.refinement_steps,
            refinement_rate=benchmark.base.base.training.refinement_rate,
            useful_threshold=benchmark.base.base.landscape.useful_threshold,
        )
        history = result.history
        metadata = {
            "training_seconds": result.elapsed_seconds,
            "steps": result.steps,
            "examples_seen": result.examples_seen,
            "peak_allocated_bytes": result.peak_allocated_bytes,
            "peak_reserved_bytes": result.peak_reserved_bytes,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "loss": "1-cosine + lambda_alpha*(alpha-target_distance)^2",
            "uses_q_for_training": False,
            "matching": "exact permutation-invariant assignment"
            if mode == "multimode"
            else "candidate_0_to_goal_target",
        }
        _save_checkpoint(model_path, model, history=history, metadata=metadata)
    evaluation = evaluate_supervised(
        model,
        packed_test,
        h_test,
        mode=mode,
        lambda_alpha=capacity.supervised.lambda_alpha,
        refinement_steps=benchmark.base.base.training.refinement_steps,
        refinement_rate=benchmark.base.base.training.refinement_rate,
        useful_threshold=benchmark.base.base.landscape.useful_threshold,
        capture_distance=capacity.anchor_escape.capture_distance,
    )
    training_evaluation = evaluate_supervised(
        model,
        packed_train,
        h_train,
        mode=mode,
        lambda_alpha=capacity.supervised.lambda_alpha,
        refinement_steps=benchmark.base.base.training.refinement_steps,
        refinement_rate=benchmark.base.base.training.refinement_rate,
        useful_threshold=benchmark.base.base.landscape.useful_threshold,
        capture_distance=capacity.anchor_escape.capture_distance,
    )
    row: dict[str, object] = {
        "dimension": dimension,
        "seed": seed,
        "architecture": architecture,
        "mode": mode,
        **evaluation.summary,
        **{
            f"training_{key}": value
            for key, value in training_evaluation.summary.items()
        },
        **metadata,
    }
    curves = [
        {
            "dimension": dimension,
            "seed": seed,
            "architecture": architecture,
            "mode": mode,
            **point,
        }
        for point in history
    ]
    return row, curves


def _select_batch(rows: list[dict[str, object]], plateau: float) -> int:
    maximum = max(float(row["examples_per_second"]) for row in rows)
    eligible = [
        int(row["batch_size"])
        for row in rows
        if float(row["examples_per_second"]) >= plateau * maximum
    ]
    return min(eligible)


def run_supervised_representability(
    capacity: CapacityDiagnosticConfig,
    *,
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
    smoke: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(capacity.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "runtime.json").write_text(
        json.dumps(runtime.to_dict(), indent=2), encoding="utf-8"
    )
    primary_dimension = capacity.supervised.primary_dimension
    primary = _dimension_config(primary_dimension, smoke=smoke)
    family = _family(primary)
    seed0 = capacity.supervised.d16_seeds[0]
    train, _, _, _, _, _ = _datasets(family, primary, seed0)
    (output / "geometry_audit.json").write_text(
        json.dumps(geometry_diagnostics(train), indent=2), encoding="utf-8"
    )
    benchmark_nested = _make_starts(
        train,
        trials=capacity.supervised.starts_per_training_task,
        seed=seed0 + 41_001,
    )
    benchmark_tasks, benchmark_starts = _aligned(train, benchmark_nested, device)
    batch_rows = benchmark_supervised_batches(
        lambda: _model("current_mlp", primary, seed0, device),
        benchmark_tasks,
        benchmark_starts,
        capacity.supervised,
        seed=seed0 + 41_201,
    )
    selected_batch = _select_batch(batch_rows, capacity.supervised.plateau_fraction)
    write_csv(output / "batch_scaling.csv", batch_rows)
    (output / "selected_batch.json").write_text(
        json.dumps(
            {
                "selection_rule": f"smallest batch within {capacity.supervised.plateau_fraction:.0%} of maximum throughput",
                "selected_batch_size": selected_batch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for seed in capacity.supervised.d16_seeds:
        train, _, _, _, test, test_starts = _datasets(family, primary, seed)
        for architecture in ("current_mlp", "linear"):
            row, history = _train_or_load_supervised(
                architecture=architecture,
                mode="single_target",
                dimension=primary_dimension,
                seed=seed,
                benchmark=primary,
                capacity=capacity,
                train_tasks=train,
                test_tasks=test,
                test_starts=test_starts,
                selected_batch_size=selected_batch,
                output=output,
                device=device,
            )
            rows.append(row)
            curves.extend(history)
            write_csv(output / "per_seed.csv", rows)
            write_csv(output / "training_curves.csv", curves)

    current_single = [
        row
        for row in rows
        if row["architecture"] == "current_mlp" and row["mode"] == "single_target"
    ]
    heldout_cosine = float(
        np.mean([float(row["candidate_zero_target_cosine"]) for row in current_single])
    )
    heldout_recall = float(
        np.mean([float(row["candidate_zero_target_recall"]) for row in current_single])
    )
    training_cosine = float(
        np.mean(
            [
                float(row["training_candidate_zero_target_cosine"])
                for row in current_single
            ]
        )
    )
    training_recall = float(
        np.mean(
            [
                float(row["training_candidate_zero_target_recall"])
                for row in current_single
            ]
        )
    )
    single_pass = (
        training_cosine > capacity.supervised.best_cosine_success
        and training_recall >= capacity.supervised.proposal_recall_success
    )
    gate: dict[str, object] = {
        "primary_dimension": primary_dimension,
        "single_target_pass": single_pass,
        "criterion": "on explicitly supervised training labels: mean candidate-0 target cosine > 0.90 and mean candidate-0 target recall >= 0.75",
        "mean_candidate_zero_target_cosine": heldout_cosine,
        "mean_candidate_zero_target_recall": heldout_recall,
        "mean_training_candidate_zero_target_cosine": training_cosine,
        "mean_training_candidate_zero_target_recall": training_recall,
        "single_target_heldout_generalization_pass": (
            heldout_cosine > capacity.supervised.best_cosine_success
            and heldout_recall >= capacity.supervised.proposal_recall_success
        ),
        "multimode_run": False,
        "anchor_escape_allowed": False,
        "capacity_basis": "fit to explicitly supplied training labels; held-out metrics are reported separately as generalization",
    }
    if single_pass:
        for dimension in capacity.supervised.dimensions:
            benchmark = _dimension_config(dimension, smoke=smoke)
            dimensional_family = _family(benchmark)
            seeds = (
                capacity.supervised.d16_seeds
                if dimension == primary_dimension
                else capacity.supervised.dimension_seeds
            )
            for seed in seeds:
                train, _, _, _, test, test_starts = _datasets(
                    dimensional_family, benchmark, seed
                )
                for architecture in ("current_mlp", "linear"):
                    row, history = _train_or_load_supervised(
                        architecture=architecture,
                        mode="multimode",
                        dimension=dimension,
                        seed=seed,
                        benchmark=benchmark,
                        capacity=capacity,
                        train_tasks=train,
                        test_tasks=test,
                        test_starts=test_starts,
                        selected_batch_size=selected_batch,
                        output=output,
                        device=device,
                    )
                    rows.append(row)
                    curves.extend(history)
                    write_csv(output / "per_seed.csv", rows)
                    write_csv(output / "training_curves.csv", curves)
        gate["multimode_run"] = True
        primary_multi = [
            row
            for row in rows
            if int(row["dimension"]) == primary_dimension
            and row["architecture"] == "current_mlp"
            and row["mode"] == "multimode"
        ]
        multi_cosine = float(
            np.mean([float(row["mean_matched_cosine"]) for row in primary_multi])
        )
        multi_recall = float(
            np.mean([float(row["proposal_recall"]) for row in primary_multi])
        )
        training_cosine = float(
            np.mean(
                [float(row["training_mean_matched_cosine"]) for row in primary_multi]
            )
        )
        training_recall = float(
            np.mean([float(row["training_proposal_recall"]) for row in primary_multi])
        )
        multimode_fit_pass = (
            training_cosine > capacity.supervised.best_cosine_success
            and training_recall >= capacity.supervised.proposal_recall_success
        )
        heldout_multimode_pass = (
            multi_cosine > capacity.supervised.best_cosine_success
            and multi_recall >= capacity.supervised.proposal_recall_success
        )
        gate.update(
            {
                "primary_dimension": primary_dimension,
                "mean_primary_multimode_matched_cosine": multi_cosine,
                "mean_primary_multimode_proposal_recall": multi_recall,
                "mean_primary_multimode_training_matched_cosine": training_cosine,
                "mean_primary_multimode_training_proposal_recall": training_recall,
                "multimode_training_fit_pass": multimode_fit_pass,
                "multimode_heldout_generalization_pass": heldout_multimode_pass,
                "anchor_escape_allowed": single_pass and multimode_fit_pass,
            }
        )
    summaries = _summary_rows(
        rows, ("dimension", "architecture", "mode"), SUPERVISED_METRICS
    )
    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "training_curves.csv", curves)
    (output / "decision_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    return gate


ANCHOR_METRICS = (
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
    "direction_cosine_distance",
)


def _anchor_radius(benchmark) -> float:
    landscape = benchmark.base.base.landscape
    return float(landscape.anchor_width * landscape.region_mahalanobis_threshold)


def _train_anchor_seed(
    *,
    seed: int,
    lambda_anchor: float,
    benchmark,
    capacity: CapacityDiagnosticConfig,
    output: Path,
    device: torch.device,
    evaluate_on_validation: bool,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    family = _family(benchmark)
    train, train_starts, validation, validation_starts, test, test_starts = _datasets(
        family, benchmark, seed
    )
    packed_train = pack_tasks(train, device=device, dtype=torch.float32)
    h_train = torch.as_tensor(
        np.stack(train_starts), dtype=torch.float32, device=device
    )
    eval_tasks = validation if evaluate_on_validation else test
    eval_nested = (
        [[point] for point in validation_starts]
        if evaluate_on_validation
        else test_starts
    )
    packed_eval, h_eval = _aligned(eval_tasks, eval_nested, device)
    environment = _environment(benchmark)
    model_path = output / "models" / f"lambda_{lambda_anchor:g}_seed_{seed}.pt"
    policy = _policy(benchmark, seed, device)
    radius = _anchor_radius(benchmark)

    def validation_callback(current, checkpoint):
        return evaluate_anchor_policy(
            current,
            environment,
            packed_eval,
            h_eval,
            useful_threshold=benchmark.base.base.landscape.useful_threshold,
            capture_distance=capacity.anchor_escape.capture_distance,
            action_seed=seed + 52_000 + checkpoint,
            deterministic=False,
        )

    if model_path.exists():
        payload = torch.load(model_path, map_location=device, weights_only=True)
        policy.load_state_dict(payload["state_dict"], strict=True)
        history = list(payload.get("history", []))
        validation_history = list(payload.get("validation_history", []))
        metadata = dict(payload.get("metadata", {}))
    else:
        result = train_policy_anchor_escape(
            policy,
            environment,
            packed_train,
            h_train,
            benchmark.rl.training,
            seed=seed + 51_001,
            lambda_anchor=lambda_anchor,
            anchor_radius=radius,
            validation_callback=validation_callback,
        )
        history = result.history
        validation_history = result.validation_history
        metadata = {
            "training_seconds": result.elapsed_seconds,
            "optimizer_steps": result.optimizer_steps,
            "candidate_evaluations": result.candidate_evaluations,
            "q_equivalent_evaluations": result.q_equivalent_evaluations,
            "peak_allocated_bytes": result.peak_allocated_bytes,
            "peak_reserved_bytes": result.peak_reserved_bytes,
        }
        model_path.parent.mkdir(parents=True, exist_ok=True)
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
            model_path,
        )
    final = evaluate_anchor_policy(
        policy,
        environment,
        packed_eval,
        h_eval,
        useful_threshold=benchmark.base.base.landscape.useful_threshold,
        capture_distance=capacity.anchor_escape.capture_distance,
        action_seed=seed + 53_001,
        deterministic=False,
    )
    row: dict[str, object] = {
        "seed": seed,
        "lambda_anchor": lambda_anchor,
        "split": "validation" if evaluate_on_validation else "test",
        "anchor_radius": radius,
        **final,
        **metadata,
    }
    curves = [
        {
            "seed": seed,
            "lambda_anchor": lambda_anchor,
            "curve": "training",
            **point,
        }
        for point in history
    ]
    validation_curves = [
        {
            "seed": seed,
            "lambda_anchor": lambda_anchor,
            "curve": "validation",
            **point,
        }
        for point in validation_history
    ]
    return row, curves, validation_curves


def _select_lambda(
    rows: list[dict[str, object]], tolerance: float
) -> tuple[float, dict[str, object]]:
    summaries = _summary_rows(rows, ("lambda_anchor",), ANCHOR_METRICS)
    baseline = next(row for row in summaries if float(row["lambda_anchor"]) == 0.0)
    decoy_limit = float(baseline["candidate_decoy_rate_mean"]) + tolerance
    eligible = [
        row
        for row in summaries
        if float(row["candidate_decoy_rate_mean"]) <= decoy_limit
    ]
    chosen = max(
        eligible,
        key=lambda row: (
            float(row["proposal_recall_mean"]),
            float(row["candidate_mean_quality_mean"]),
            -float(row["lambda_anchor"]),
        ),
    )
    return float(chosen["lambda_anchor"]), {
        "rule": "maximize validation proposal recall subject to decoy rate <= lambda0 + 0.10; tie-break by quality then smaller lambda",
        "baseline_decoy_rate": float(baseline["candidate_decoy_rate_mean"]),
        "decoy_limit": decoy_limit,
        "selected_lambda": float(chosen["lambda_anchor"]),
    }


def run_anchor_escape(
    capacity: CapacityDiagnosticConfig,
    *,
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
    smoke: bool,
    allowed: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(capacity.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "runtime.json").write_text(
        json.dumps(runtime.to_dict(), indent=2), encoding="utf-8"
    )
    if not allowed:
        decision = {
            "status": "skipped",
            "reason": "Part A did not pass the predefined representability gate",
            "lambda_mode": 0.0,
        }
        (output / "decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        return decision
    benchmark = _dimension_config(capacity.supervised.primary_dimension, smoke=smoke)
    benchmark.rl.training.max_candidate_evaluations = (
        capacity.anchor_escape.candidate_budget
    )
    benchmark.rl.training.checkpoint_budgets = sorted(
        set(
            [
                0,
                min(10_000, capacity.anchor_escape.candidate_budget),
                min(25_000, capacity.anchor_escape.candidate_budget),
                min(50_000, capacity.anchor_escape.candidate_budget),
                capacity.anchor_escape.candidate_budget,
            ]
        )
    )
    validation_rows: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    validation_curves: list[dict[str, object]] = []
    validation_dir = output / "validation"
    for value in capacity.anchor_escape.lambda_values:
        for seed in capacity.anchor_escape.validation_seeds:
            row, training, validation = _train_anchor_seed(
                seed=seed,
                lambda_anchor=value,
                benchmark=benchmark,
                capacity=capacity,
                output=validation_dir,
                device=device,
                evaluate_on_validation=True,
            )
            validation_rows.append(row)
            curves.extend(training)
            validation_curves.extend(validation)
            write_csv(output / "validation_per_seed.csv", validation_rows)
    selected, selection = _select_lambda(
        validation_rows, capacity.anchor_escape.validation_decoy_tolerance
    )
    write_csv(
        output / "validation_summary.csv",
        _summary_rows(validation_rows, ("lambda_anchor",), ANCHOR_METRICS),
    )
    (output / "lambda_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    test_rows: list[dict[str, object]] = []
    main_dir = output / "main"
    for value in sorted(set([0.0, selected])):
        for seed in capacity.anchor_escape.main_seeds:
            row, training, validation = _train_anchor_seed(
                seed=seed,
                lambda_anchor=value,
                benchmark=benchmark,
                capacity=capacity,
                output=main_dir,
                device=device,
                evaluate_on_validation=False,
            )
            test_rows.append(row)
            curves.extend(training)
            validation_curves.extend(validation)
            write_csv(output / "test_per_seed.csv", test_rows)
            write_csv(output / "training_curves.csv", curves)
            write_csv(output / "evaluation_curves.csv", validation_curves)
    summaries = _summary_rows(test_rows, ("lambda_anchor",), ANCHOR_METRICS)
    write_csv(output / "test_summary.csv", summaries)
    write_csv(output / "training_curves.csv", curves)
    write_csv(output / "evaluation_curves.csv", validation_curves)
    decision = {
        "status": "completed",
        "selected_lambda": selected,
        "anchor_radius": _anchor_radius(benchmark),
        "anchor_radius_basis": "anchor_width * existing Mahalanobis region threshold",
        "lambda_mode": 0.0,
        "portfolio_escape_run": False,
    }
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return decision
