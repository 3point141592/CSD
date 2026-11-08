"""Orchestration for the controlled dimensional-failure diagnostic study.

This module does not alter the established CSD trainers.  It runs isolated
instrumented replicas plus a privileged differentiable-Q control.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from creativity_toy.landscapes_d16 import (
    D16LandscapeFamily,
    DimensionalLandscapeFamily,
    geometry_diagnostics,
)
from creativity_toy.landscapes_v2 import V2LandscapeFamily, V2Task

from .config_d16 import D16BenchmarkConfig
from .config_failure_diagnostics import FailureDiagnosticConfig
from .contrastive_evaluation import CandidateEvaluation, flatten_starts
from .contrastive_steering import (
    ContrastiveCandidates,
    build_contrastive_vectors,
    contrastive_candidates,
)
from .creative_policy_rl import StochasticCreativePolicy
from .failure_geometry import (
    benchmark_batch_sizes,
    candidate_geometry_rows,
    empirical_random_hit_rate,
)
from .runtime import RuntimeInfo
from .study import write_csv
from .study_d16 import (
    _datasets,
    _environment,
    _evaluate_proposal,
    _fit_shared_routers,
    _policy,
    _policy_proposal,
    _summaries,
)
from .task_batch import pack_tasks
from .training_failure_diagnostics import (
    DiagnosticTrainingResult,
    differentiable_policy_candidates,
    train_direct_gradient,
    train_instrumented_rl,
)

DIAGNOSTIC_METRICS = (
    "selected_quality",
    "proposal_recall",
    "routing_recall",
    "useful_selected",
    "useful_region_count",
    "useful_region_count_2_plus",
    "candidate_decoy_rate",
    "target_selected",
    "effective_directional_rank",
    "direction_cosine_distance",
    "candidate_mean_quality",
    "alpha_mean",
    "alpha_sd",
)


def _dimension_config(dimension: int, *, smoke: bool = False) -> D16BenchmarkConfig:
    config = D16BenchmarkConfig(name=f"failure_d{dimension}{'_smoke' if smoke else ''}")
    config.dimension = dimension
    config.base.name = config.name
    config.base.base.name = config.name
    config.base.base.landscape.dimension = dimension
    config.contrastive.name = config.name
    config.contrastive.dimension = dimension
    if smoke:
        config.base.base.train_geometries = 5
        config.base.base.test_geometries = 2
        config.base.base.trials_per_task = 1
        config.rl.validation_geometries = 2
        config.contrastive.data.samples_per_region = 12
        config.router.pairwise_steps = 30
    return config


def _family(config: D16BenchmarkConfig):
    if config.dimension == 2:
        return V2LandscapeFamily(config.base.base.landscape)
    if config.dimension == 16:
        return D16LandscapeFamily(config.base.base.landscape)
    return DimensionalLandscapeFamily(config.base.base.landscape)


def _checkpoint_payload(
    policy: StochasticCreativePolicy,
    result: DiagnosticTrainingResult,
    *,
    method: str,
    dimension: int,
    seed: int,
    budget: int,
    batch_size: int,
) -> dict[str, object]:
    return {
        "state_dict": {
            key: value.detach().cpu() for key, value in policy.state_dict().items()
        },
        "encounter_rows": result.encounter_rows,
        "history": result.history,
        "metadata": {
            "method": method,
            "dimension": dimension,
            "seed": seed,
            "budget": budget,
            "batch_size": batch_size,
            "optimizer_steps": result.optimizer_steps,
            "candidate_evaluations": result.candidate_evaluations,
            "q_equivalent_evaluations": result.q_equivalent_evaluations,
            "training_seconds": result.elapsed_seconds,
            "peak_allocated_bytes": result.peak_allocated_bytes,
            "peak_reserved_bytes": result.peak_reserved_bytes,
        },
    }


def _load_checkpoint(
    path: Path,
    config: D16BenchmarkConfig,
    seed: int,
    device: torch.device,
) -> tuple[StochasticCreativePolicy, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    policy = _policy(config, seed, device)
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    return policy, payload


def _train_or_load(
    *,
    method: str,
    config: D16BenchmarkConfig,
    diagnostic: FailureDiagnosticConfig,
    train_tasks: Sequence[V2Task],
    train_starts: Sequence[np.ndarray],
    seed: int,
    budget: int,
    batch_size: int,
    path: Path,
    device: torch.device,
) -> tuple[StochasticCreativePolicy, dict[str, object]]:
    if path.exists():
        return _load_checkpoint(path, config, seed, device)
    policy = _policy(config, seed, device)
    packed = pack_tasks(train_tasks, device=device, dtype=torch.float32)
    starts = torch.as_tensor(np.stack(train_starts), dtype=torch.float32, device=device)
    checkpoints = [value for value in diagnostic.encounter_budgets if value <= budget]
    if method == "direct_gradient":
        result = train_direct_gradient(
            policy,
            packed,
            starts,
            refinement_steps=config.base.base.training.refinement_steps,
            refinement_rate=config.base.base.training.refinement_rate,
            seed=seed + 70_101,
            budget=budget,
            batch_size=batch_size,
            learning_rate=diagnostic.direct_gradient_learning_rate,
            weight_decay=diagnostic.direct_gradient_weight_decay,
            gradient_clip=diagnostic.direct_gradient_clip,
            encounter_checkpoints=checkpoints,
            thresholds=diagnostic.quality_thresholds,
            quantiles=diagnostic.quality_quantiles,
        )
    elif method == "black_box_rl":
        result = train_instrumented_rl(
            policy,
            _environment(config),
            packed,
            starts,
            config.rl.training,
            seed=seed + 70_201,
            budget=budget,
            batch_size=batch_size,
            encounter_checkpoints=checkpoints,
            thresholds=diagnostic.quality_thresholds,
            quantiles=diagnostic.quality_quantiles,
        )
    else:
        raise ValueError(method)
    payload = _checkpoint_payload(
        policy,
        result,
        method=method,
        dimension=config.dimension,
        seed=seed,
        budget=budget,
        batch_size=batch_size,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    policy.eval()
    return policy, payload


@torch.no_grad()
def _direct_proposal(
    policy: StochasticCreativePolicy,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    *,
    device: torch.device,
) -> ContrastiveCandidates:
    start_array, task_ids, _ = flatten_starts(starts)
    packed = pack_tasks(tasks, device=device, dtype=torch.float32)
    indices = torch.as_tensor(task_ids, dtype=torch.long, device=device)
    aligned = packed.index_select(indices)
    h = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    candidates, directions, alphas = differentiable_policy_candidates(
        policy, h, aligned.goals
    )
    return ContrastiveCandidates(candidates, directions, alphas)


def _evaluate_direct(
    *,
    method: str,
    policy: StochasticCreativePolicy,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    config: D16BenchmarkConfig,
    quality_router,
    pairwise_router,
    seed: int,
    device: torch.device,
) -> tuple[CandidateEvaluation, ContrastiveCandidates]:
    proposal = _direct_proposal(policy, tasks, starts, device=device)
    result = _evaluate_proposal(
        method=method,
        tasks=tasks,
        starts=starts,
        proposal=proposal,
        config=config,
        quality_router=quality_router,
        pairwise_router=pairwise_router,
        random_seed=seed + 79_001,
        device=device,
        proposal_seconds=0.0,
    )
    return result, proposal


def _portfolio_row(
    result: CandidateEvaluation,
    *,
    dimension: int,
    seed: int,
    method: str,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dimension": dimension,
        "seed": seed,
        "method": method,
        **result.portfolio,
    }
    if metadata:
        row.update({f"training_{key}": value for key, value in metadata.items()})
    return row


def _encounter_rows(
    payload: Mapping[str, object], *, dimension: int, seed: int, method: str
) -> list[dict[str, object]]:
    values = payload.get("encounter_rows", [])
    return [
        {"dimension": dimension, "seed": seed, "method": method, **row}
        for row in values
    ]


def run_batch_scaling(
    diagnostic: FailureDiagnosticConfig,
    *,
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
    smoke: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    config = _dimension_config(16, smoke=smoke)
    family = _family(config)
    seed = diagnostic.smoke_seeds[0]
    train, train_starts, _, _, _, _ = _datasets(family, config, seed)
    packed = pack_tasks(train, device=device, dtype=torch.float32)
    starts = torch.as_tensor(np.stack(train_starts), dtype=torch.float32, device=device)
    rows = benchmark_batch_sizes(
        make_policy=lambda: _policy(config, seed, device),
        environment=_environment(config),
        tasks=packed,
        starts=starts,
        training=config.rl.training,
        batch_sizes=diagnostic.batch_sizes,
        warmup_steps=diagnostic.batch_warmup_steps,
        timing_steps=diagnostic.batch_timing_steps,
        seed=seed + 80_001,
        device=device,
    )
    valid = [row for row in rows if row["status"] == "ok"]
    if not valid:
        raise RuntimeError("every diagnostic batch size failed")
    maximum = max(float(row["candidate_outcomes_per_second"]) for row in valid)
    threshold = diagnostic.batch_plateau_fraction * maximum
    selected = min(
        int(row["batch_size"])
        for row in valid
        if float(row["candidate_outcomes_per_second"]) >= threshold
    )
    write_csv(output / "batch_scaling_raw.csv", rows)
    summary = {
        "runtime": runtime.to_dict(),
        "selected_batch_size": selected,
        "scientific_training_batch_size": config.rl.training.batch_size,
        "maximum_candidate_outcomes_per_second": maximum,
        "plateau_fraction": diagnostic.batch_plateau_fraction,
        "batch_sizes": diagnostic.batch_sizes,
        "semantic_note": (
            "The throughput-optimal batch is an engineering measurement. Scientific "
            "RL studies retain batch_size=12 because batching sequential AdamW updates "
            "would change the established training trajectory."
        ),
    }
    (output / "batch_scaling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _load_or_fit_router(
    *,
    config: D16BenchmarkConfig,
    train_tasks: Sequence[V2Task],
    train_starts: Sequence[np.ndarray],
    first_policy: StochasticCreativePolicy,
    second_policy: StochasticCreativePolicy,
    seed: int,
    path: Path,
    device: torch.device,
):
    return _fit_shared_routers(
        config=config,
        train_tasks=train_tasks,
        train_starts=train_starts,
        rl_policy=first_policy,
        mode_policy=second_policy,
        seed=seed,
        device=device,
        path=path,
    )


def run_gradient_oracle_study(
    diagnostic: FailureDiagnosticConfig,
    *,
    output: Path,
    device: torch.device,
    smoke: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    config = _dimension_config(16, smoke=smoke)
    family = _family(config)
    seeds = diagnostic.gradient_seeds
    result_rows: list[dict[str, object]] = []
    encounter_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    for index, seed in enumerate(seeds):
        train, train_starts, _, _, test, test_starts = _datasets(family, config, seed)
        model_path = output / "models" / f"seed_{seed}_direct_gradient.pt"
        policy, payload = _train_or_load(
            method="direct_gradient",
            config=config,
            diagnostic=diagnostic,
            train_tasks=train,
            train_starts=train_starts,
            seed=seed,
            budget=diagnostic.direct_gradient_budget,
            batch_size=diagnostic.direct_gradient_batch_size,
            path=model_path,
            device=device,
        )
        existing_router = Path("results/highdim_d16/routers") / (
            f"seed_{seed}_shared_pairwise.pt"
        )
        router_path = (
            existing_router
            if existing_router.exists() and not smoke
            else output / "routers" / f"seed_{seed}.pt"
        )
        quality_router, pairwise_router, _ = _load_or_fit_router(
            config=config,
            train_tasks=train,
            train_starts=train_starts,
            first_policy=policy,
            second_policy=policy,
            seed=seed,
            path=router_path,
            device=device,
        )
        evaluation, proposal = _evaluate_direct(
            method="direct_gradient",
            policy=policy,
            tasks=test,
            starts=test_starts,
            config=config,
            quality_router=quality_router,
            pairwise_router=pairwise_router,
            seed=seed,
            device=device,
        )
        metadata = payload["metadata"]
        result_rows.append(
            _portfolio_row(
                evaluation,
                dimension=16,
                seed=seed,
                method="direct_gradient",
                metadata=metadata,
            )
        )
        encounter_rows.extend(
            _encounter_rows(payload, dimension=16, seed=seed, method="direct_gradient")
        )
        candidates, tasks_scale = candidate_geometry_rows(
            dimension=16,
            seed=seed,
            method="direct_gradient",
            tasks=test,
            starts=test_starts,
            proposal=proposal,
            evaluation=evaluation,
            device=device,
            useful_threshold=config.base.base.landscape.useful_threshold,
        )
        geometry_rows.extend(candidates)
        scale_rows.extend(tasks_scale)
        print(
            f"[gradient oracle {index + 1}/{len(seeds)}] seed={seed} "
            f"recall={evaluation.portfolio['proposal_recall']:.3f}",
            flush=True,
        )
    summaries = _summaries(
        result_rows,
        group_keys=["dimension", "method"],
        metrics=DIAGNOSTIC_METRICS,
    )
    write_csv(output / "per_seed_metrics.csv", result_rows)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "reward_encounters.csv", encounter_rows)
    write_csv(output / "candidate_geometry.csv", geometry_rows)
    write_csv(output / "task_geometry.csv", scale_rows)
    (output / "config.json").write_text(
        json.dumps(diagnostic.to_dict(), indent=2), encoding="utf-8"
    )
    return {"rows": result_rows, "summary": summaries}


@torch.no_grad()
def _contrast_proposal(
    *,
    vectors,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    alpha: float,
    candidates: int,
    device: torch.device,
) -> ContrastiveCandidates:
    start_array, task_ids, _ = flatten_starts(starts)
    goals = np.stack([tasks[int(index)].goal for index in task_ids])
    h = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    g = torch.as_tensor(goals, dtype=torch.float32, device=device)
    return contrastive_candidates(
        h,
        g,
        vectors,
        kind="goal",
        alpha=alpha,
        form="k_radius",
        candidate_count=candidates,
    )


def _evaluate_fixed_proposal(
    *,
    method: str,
    proposal: ContrastiveCandidates,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    config: D16BenchmarkConfig,
    quality_router,
    pairwise_router,
    seed: int,
    device: torch.device,
) -> CandidateEvaluation:
    return _evaluate_proposal(
        method=method,
        tasks=tasks,
        starts=starts,
        proposal=proposal,
        config=config,
        quality_router=quality_router,
        pairwise_router=pairwise_router,
        random_seed=seed + 89_001,
        device=device,
        proposal_seconds=0.0,
    )


def run_dimension_sweep(
    diagnostic: FailureDiagnosticConfig,
    *,
    output: Path,
    device: torch.device,
    smoke: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    seeds = diagnostic.dimension_seeds
    budget = diagnostic.direct_gradient_budget if smoke else 100_000
    result_rows: list[dict[str, object]] = []
    encounter_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    landscape_rows: list[dict[str, object]] = []
    for dimension in diagnostic.dimensions:
        config = _dimension_config(dimension, smoke=smoke)
        family = _family(config)
        scientific_batch_size = config.rl.training.batch_size
        for seed_index, seed in enumerate(seeds):
            train, train_starts, _, _, test, test_starts = _datasets(
                family, config, seed
            )
            rl_policy, rl_payload = _train_or_load(
                method="black_box_rl",
                config=config,
                diagnostic=diagnostic,
                train_tasks=train,
                train_starts=train_starts,
                seed=seed,
                budget=budget,
                batch_size=scientific_batch_size,
                path=output
                / "models"
                / (
                    f"d{dimension}_seed_{seed}_black_box_rl_{budget}_"
                    f"batch_{scientific_batch_size}.pt"
                ),
                device=device,
            )
            shared_d16_gradient = (
                Path("results/d16_gradient_oracle/models")
                / f"seed_{seed}_direct_gradient.pt"
            )
            direct_path = (
                shared_d16_gradient
                if dimension == 16 and shared_d16_gradient.exists() and not smoke
                else output
                / "models"
                / f"d{dimension}_seed_{seed}_direct_gradient_{budget}.pt"
            )
            direct_policy, direct_payload = _train_or_load(
                method="direct_gradient",
                config=config,
                diagnostic=diagnostic,
                train_tasks=train,
                train_starts=train_starts,
                seed=seed,
                budget=budget,
                batch_size=diagnostic.direct_gradient_batch_size,
                path=direct_path,
                device=device,
            )
            quality_router, pairwise_router, _ = _load_or_fit_router(
                config=config,
                train_tasks=train,
                train_starts=train_starts,
                first_policy=rl_policy,
                second_policy=direct_policy,
                seed=seed,
                path=output / "routers" / f"d{dimension}_seed_{seed}.pt",
                device=device,
            )
            vectors = build_contrastive_vectors(
                train,
                split="train_iid",
                seed=seed + 88_001,
                config=config.contrastive.data,
            )
            proposals: dict[str, ContrastiveCandidates] = {
                "black_box_rl": _policy_proposal(
                    rl_policy,
                    test,
                    test_starts,
                    seed=seed + 88_101,
                    device=device,
                ),
                "direct_gradient": _direct_proposal(
                    direct_policy, test, test_starts, device=device
                ),
                "contrastive": _contrast_proposal(
                    vectors=vectors,
                    tasks=test,
                    starts=test_starts,
                    alpha=diagnostic.contrastive_alpha,
                    candidates=config.base.base.training.candidates,
                    device=device,
                ),
            }
            initial_policy = _policy(config, seed, device)
            proposals["random_initial_policy"] = _policy_proposal(
                initial_policy,
                test,
                test_starts,
                seed=seed + 88_101,
                device=device,
            )
            evaluations: dict[str, CandidateEvaluation] = {}
            for method, proposal in proposals.items():
                evaluation = _evaluate_fixed_proposal(
                    method=method,
                    proposal=proposal,
                    tasks=test,
                    starts=test_starts,
                    config=config,
                    quality_router=quality_router,
                    pairwise_router=pairwise_router,
                    seed=seed,
                    device=device,
                )
                evaluations[method] = evaluation
                metadata = (
                    rl_payload["metadata"]
                    if method == "black_box_rl"
                    else direct_payload["metadata"]
                    if method == "direct_gradient"
                    else None
                )
                result_rows.append(
                    _portfolio_row(
                        evaluation,
                        dimension=dimension,
                        seed=seed,
                        method=method,
                        metadata=metadata,
                    )
                )
                candidates, tasks_scale = candidate_geometry_rows(
                    dimension=dimension,
                    seed=seed,
                    method=method,
                    tasks=test,
                    starts=test_starts,
                    proposal=proposal,
                    evaluation=evaluation,
                    device=device,
                    useful_threshold=config.base.base.landscape.useful_threshold,
                )
                geometry_rows.extend(candidates)
                scale_rows.extend(tasks_scale)
            encounter_rows.extend(
                _encounter_rows(
                    rl_payload,
                    dimension=dimension,
                    seed=seed,
                    method="black_box_rl",
                )
            )
            encounter_rows.extend(
                _encounter_rows(
                    direct_payload,
                    dimension=dimension,
                    seed=seed,
                    method="direct_gradient",
                )
            )
            alpha_samples = proposals["black_box_rl"].alphas.reshape(-1)
            random_hit = empirical_random_hit_rate(
                tasks=test,
                starts=test_starts,
                alpha_samples=alpha_samples,
                outcomes=diagnostic.random_hit_outcomes,
                refinement_steps=config.base.base.training.refinement_steps,
                refinement_rate=config.base.base.training.refinement_rate,
                useful_threshold=config.base.base.landscape.useful_threshold,
                seed=seed + 88_301,
                device=device,
            )
            random_rows.append({"dimension": dimension, "seed": seed, **random_hit})
            if seed_index == 0:
                landscape_rows.append(
                    {
                        "dimension": dimension,
                        **{
                            key: value
                            for key, value in geometry_diagnostics(test).items()
                            if not isinstance(value, list)
                        },
                    }
                )
            print(
                f"[dimension d={dimension} {seed_index + 1}/{len(seeds)}] "
                f"RL={evaluations['black_box_rl'].portfolio['proposal_recall']:.3f} "
                f"grad={evaluations['direct_gradient'].portfolio['proposal_recall']:.3f}",
                flush=True,
            )
    summary = _summaries(
        result_rows,
        group_keys=["dimension", "method"],
        metrics=DIAGNOSTIC_METRICS,
    )
    write_csv(output / "dimension_per_seed.csv", result_rows)
    write_csv(output / "dimension_summary.csv", summary)
    write_csv(output / "reward_encounters.csv", encounter_rows)
    write_csv(output / "candidate_geometry.csv", geometry_rows)
    write_csv(output / "task_geometry.csv", scale_rows)
    write_csv(output / "random_hit_rates.csv", random_rows)
    write_csv(output / "landscape_geometry.csv", landscape_rows)
    (output / "config.json").write_text(
        json.dumps(diagnostic.to_dict(), indent=2), encoding="utf-8"
    )
    return {"rows": result_rows, "summary": summary}


def run_budget_scaling(
    diagnostic: FailureDiagnosticConfig,
    *,
    output: Path,
    device: torch.device,
    smoke: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    config = _dimension_config(16, smoke=smoke)
    family = _family(config)
    scientific_batch_size = config.rl.training.batch_size
    rows: list[dict[str, object]] = []
    encounters: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    geometry: list[dict[str, object]] = []
    for budget in diagnostic.candidate_budgets:
        for seed_index, seed in enumerate(diagnostic.budget_seeds):
            train, train_starts, _, _, test, test_starts = _datasets(
                family, config, seed
            )
            shared_dimension_model = Path("results/dimension_diagnostics/models") / (
                f"d16_seed_{seed}_black_box_rl_{budget}_"
                f"batch_{scientific_batch_size}.pt"
            )
            model_path = (
                shared_dimension_model
                if budget == 100_000 and shared_dimension_model.exists() and not smoke
                else output
                / "models"
                / f"seed_{seed}_budget_{budget}_batch_{scientific_batch_size}.pt"
            )
            policy, payload = _train_or_load(
                method="black_box_rl",
                config=config,
                diagnostic=diagnostic,
                train_tasks=train,
                train_starts=train_starts,
                seed=seed,
                budget=budget,
                batch_size=scientific_batch_size,
                path=model_path,
                device=device,
            )
            existing_router = Path("results/highdim_d16/routers") / (
                f"seed_{seed}_shared_pairwise.pt"
            )
            router_path = (
                existing_router
                if existing_router.exists() and not smoke
                else output / "routers" / f"seed_{seed}.pt"
            )
            quality_router, pairwise_router, _ = _load_or_fit_router(
                config=config,
                train_tasks=train,
                train_starts=train_starts,
                first_policy=policy,
                second_policy=policy,
                seed=seed,
                path=router_path,
                device=device,
            )
            proposal = _policy_proposal(
                policy,
                test,
                test_starts,
                seed=seed + 90_101,
                device=device,
            )
            evaluation = _evaluate_fixed_proposal(
                method="black_box_rl",
                proposal=proposal,
                tasks=test,
                starts=test_starts,
                config=config,
                quality_router=quality_router,
                pairwise_router=pairwise_router,
                seed=seed,
                device=device,
            )
            row = _portfolio_row(
                evaluation,
                dimension=16,
                seed=seed,
                method="black_box_rl",
                metadata=payload["metadata"],
            )
            row["budget"] = budget
            rows.append(row)
            encounters.extend(
                [
                    {**entry, "budget": budget}
                    for entry in _encounter_rows(
                        payload,
                        dimension=16,
                        seed=seed,
                        method="black_box_rl",
                    )
                ]
            )
            histories.extend(
                [
                    {
                        "dimension": 16,
                        "seed": seed,
                        "method": "black_box_rl",
                        "budget": budget,
                        **entry,
                    }
                    for entry in payload.get("history", [])
                ]
            )
            candidate_rows, _ = candidate_geometry_rows(
                dimension=16,
                seed=seed,
                method=f"black_box_rl_{budget}",
                tasks=test,
                starts=test_starts,
                proposal=proposal,
                evaluation=evaluation,
                device=device,
                useful_threshold=config.base.base.landscape.useful_threshold,
            )
            geometry.extend(candidate_rows)
            print(
                f"[budget {budget} {seed_index + 1}/{len(diagnostic.budget_seeds)}] "
                f"seed={seed} recall={evaluation.portfolio['proposal_recall']:.3f}",
                flush=True,
            )
    summary = _summaries(
        rows,
        group_keys=["budget", "method"],
        metrics=DIAGNOSTIC_METRICS,
    )
    write_csv(output / "budget_per_seed.csv", rows)
    write_csv(output / "budget_summary.csv", summary)
    write_csv(output / "reward_encounters.csv", encounters)
    write_csv(output / "training_history.csv", histories)
    write_csv(output / "candidate_geometry.csv", geometry)
    (output / "config.json").write_text(
        json.dumps(diagnostic.to_dict(), indent=2), encoding="utf-8"
    )
    return {"rows": rows, "summary": summary}
