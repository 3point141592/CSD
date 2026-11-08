"""End-to-end orchestration for the isolated 16D CSD benchmark."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from creativity_toy.analysis_utils import statistics
from creativity_toy.experiments.study_v2 import _make_starts, _training_starts
from creativity_toy.landscapes_d16 import (
    D16LandscapeFamily,
    geometry_diagnostics,
    random_orthogonal,
)
from creativity_toy.landscapes_v2 import V2Task

from .black_box_env import BlackBoxLandscapeEnvironment
from .config_d16 import D16BenchmarkConfig
from .contrastive_evaluation import (
    CandidateEvaluation,
    evaluate_candidate_batch,
    flatten_starts,
    make_evaluated_batch,
)
from .contrastive_steering import (
    ContrastiveCandidates,
    ContrastiveVectors,
    build_contrastive_vectors,
    contrastive_candidates,
)
from .creative_policy_rl import StochasticCreativePolicy
from .critic import BatchedRidgeCritic
from .evaluation_mode import FrozenPairwiseRouter
from .frozen_router_candidates import generate_broad_training_candidates
from .router_useful_novelty import (
    fit_pairwise_router,
    fit_regression_router,
)
from .runtime import RuntimeInfo, assert_device, synchronize
from .study import write_csv
from .task_batch import pack_tasks
from .training_mode_gpu import train_policy_mode_preservation
from .training_rl_gpu import RLTrainingResult, train_policy_reinforce

METHODS = (
    "contrast_goal_1",
    "contrast_goal_k",
    "csd_rl_pairwise",
    "csd_mode_pairwise",
)
PRIMARY_METRICS = (
    "selected_quality",
    "proposal_recall",
    "routing_recall",
    "useful_selected",
    "useful_region_count",
    "useful_region_count_2_plus",
    "selected_decoy_rate",
    "target_selected",
    "effective_directional_rank",
    "direction_cosine_distance",
    "candidate_mean_quality",
    "candidate_decoy_rate",
)


def _nested(starts: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
    return [[np.asarray(value, dtype=np.float64)] for value in starts]


def _datasets(
    family: D16LandscapeFamily,
    config: D16BenchmarkConfig,
    seed: int,
) -> tuple[
    list[V2Task],
    list[np.ndarray],
    list[V2Task],
    list[np.ndarray],
    list[V2Task],
    list[list[np.ndarray]],
]:
    base = config.base.base
    train = family.sample_tasks(
        geometries=base.train_geometries,
        seed=seed + 10_000,
        variant="iid",
        all_goals=True,
    )
    train_starts = _training_starts(_make_starts(train, trials=1, seed=seed + 10_001))
    validation = family.sample_tasks(
        geometries=config.rl.validation_geometries,
        seed=seed + 15_000,
        variant="iid",
        all_goals=True,
    )
    validation_starts = _training_starts(
        _make_starts(validation, trials=1, seed=seed + 15_001)
    )
    test = family.sample_tasks(
        geometries=base.test_geometries,
        seed=seed + 20_000,
        variant="iid",
        all_goals=True,
    )
    test_starts = _make_starts(
        test,
        trials=base.trials_per_task,
        seed=seed + 20_001,
    )
    return train, train_starts, validation, validation_starts, test, test_starts


def _policy(
    config: D16BenchmarkConfig,
    seed: int,
    device: torch.device,
) -> StochasticCreativePolicy:
    base = config.base.base
    training = config.rl.training
    return StochasticCreativePolicy(
        dimension=config.dimension,
        candidates=base.training.candidates,
        latent_dimension=base.training.latent_dimension,
        hidden_dimension=base.training.hidden_dimension,
        seed=seed + 10_020,
        device=device,
        dtype=torch.float32,
        alpha_max=config.rl.alpha_max,
        direction_std_init=training.direction_std_init,
        alpha_std_init=training.alpha_std_init,
    )


def _environment(config: D16BenchmarkConfig) -> BlackBoxLandscapeEnvironment:
    return BlackBoxLandscapeEnvironment.from_configs(
        config.base.base.training, config.rl.training.reward
    )


def _train_one_policy(
    *,
    method: str,
    config: D16BenchmarkConfig,
    train_tasks: Sequence[V2Task],
    train_starts: Sequence[np.ndarray],
    seed: int,
    device: torch.device,
    path: Path,
) -> tuple[StochasticCreativePolicy, RLTrainingResult | None, dict[str, object]]:
    policy = _policy(config, seed, device)
    if path.exists():
        payload = torch.load(path, map_location=device, weights_only=True)
        policy.load_state_dict(payload["state_dict"], strict=True)
        policy.eval()
        return policy, None, dict(payload["metadata"])

    packed = pack_tasks(train_tasks, device=device, dtype=torch.float32)
    starts = torch.as_tensor(np.stack(train_starts), dtype=torch.float32, device=device)
    assert_device([starts, *packed.tensors(), *policy.parameters()], device)
    environment = _environment(config)
    if method == "csd_rl_pairwise":
        result = train_policy_reinforce(
            policy,
            environment,
            packed,
            starts,
            config.rl.training,
            seed=seed + 10_021,
        )
    elif method == "csd_mode_pairwise":
        result = train_policy_mode_preservation(
            policy,
            environment,
            packed,
            starts,
            config.rl.training,
            config.mode,
            seed=seed + 10_021,
        )
    else:
        raise ValueError(method)
    metadata: dict[str, object] = {
        "method": method,
        "seed": seed,
        "dimension": config.dimension,
        "candidate_count": policy.candidates,
        "parameter_count": policy.policy_parameter_count,
        "architecture_parameter_count": policy.architecture_parameter_count,
        "optimizer_steps": result.optimizer_steps,
        "candidate_evaluations": result.candidate_evaluations,
        "q_equivalent_evaluations": result.q_equivalent_evaluations,
        "training_seconds": result.elapsed_seconds,
        "peak_allocated_bytes": result.peak_allocated_bytes,
        "peak_reserved_bytes": result.peak_reserved_bytes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in policy.state_dict().items()
            },
            "metadata": metadata,
        },
        path,
    )
    policy.eval()
    return policy, result, metadata


@torch.no_grad()
def _policy_proposal(
    policy: StochasticCreativePolicy,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    *,
    seed: int,
    device: torch.device,
) -> ContrastiveCandidates:
    start_array, task_ids, _ = flatten_starts(starts)
    packed = pack_tasks(tasks, device=device, dtype=torch.float32)
    indices = torch.as_tensor(task_ids, dtype=torch.long, device=device)
    aligned = packed.index_select(indices)
    h = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    actions = policy.act(h, aligned.goals, generator=generator, deterministic=False)
    return ContrastiveCandidates(
        candidates=actions.candidate.detach(),
        directions=actions.direction.detach(),
        alphas=actions.alpha.detach(),
    )


def _tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _fit_shared_routers(
    *,
    config: D16BenchmarkConfig,
    train_tasks: Sequence[V2Task],
    train_starts: Sequence[np.ndarray],
    rl_policy: StochasticCreativePolicy,
    mode_policy: StochasticCreativePolicy,
    seed: int,
    device: torch.device,
    path: Path,
) -> tuple[BatchedRidgeCritic, FrozenPairwiseRouter, dict[str, object]]:
    if path.exists():
        payload = torch.load(path, map_location=device, weights_only=True)
        quality_coefficients = payload["quality_coefficients"].to(device)
        pairwise_coefficients = payload["pairwise_coefficients"].to(device)
        quality = BatchedRidgeCritic(
            coefficients=quality_coefficients,
            ridge=config.router.ridge,
            temperature=config.base.base.training.critic_temperature,
        )
        pairwise = FrozenPairwiseRouter(
            coefficients=pairwise_coefficients,
            feature_mode="local",
            target_mode="gated_pairwise_ranking",
            parameter_hash=_tensor_hash(pairwise_coefficients),
        )
        return quality, pairwise, dict(payload["metadata"])

    nested = _nested(train_starts)
    batches = []
    for method, policy, offset in [
        ("router_train_rl", rl_policy, 0),
        ("router_train_mode", mode_policy, 1),
    ]:
        proposal = _policy_proposal(
            policy,
            train_tasks,
            nested,
            seed=seed + 31_000 + offset,
            device=device,
        )
        batch, _, _, _ = make_evaluated_batch(
            source=method,
            tasks=train_tasks,
            starts=nested,
            proposal=proposal,
            base=config.base,
            router_config=config.router,
            device=device,
            refinement=True,
        )
        batches.append(batch)
    broad = generate_broad_training_candidates(
        tasks=train_tasks,
        starts=nested,
        broad_seed=seed + 31_010,
        base=config.base,
        router_config=config.router,
        device=device,
        source="router_train_broad",
    )
    batches.append(broad)
    started = perf_counter()
    regression = fit_regression_router(
        batches,
        feature_mode="local",
        target_mode="quality",
        lambda_novelty=0.0,
        quality_threshold=config.router_quality_threshold,
        quality_gate_beta=config.router_quality_gate_beta,
        config=config.router,
    )
    pairwise_fit = fit_pairwise_router(
        batches,
        feature_mode="local",
        lambda_novelty=config.router_lambda_novelty,
        quality_threshold=config.router_quality_threshold,
        quality_gate_beta=config.router_quality_gate_beta,
        config=config.router,
    )
    synchronize(device)
    elapsed = perf_counter() - started
    quality = BatchedRidgeCritic(
        coefficients=regression.coefficients,
        ridge=config.router.ridge,
        temperature=config.base.base.training.critic_temperature,
    )
    pairwise = FrozenPairwiseRouter(
        coefficients=pairwise_fit.coefficients,
        feature_mode="local",
        target_mode="gated_pairwise_ranking",
        parameter_hash=_tensor_hash(pairwise_fit.coefficients),
    )
    metadata: dict[str, object] = {
        "training_seconds": elapsed,
        "quality_parameters": quality.parameter_count,
        "pairwise_parameters": pairwise.parameter_count,
        "training_sources": [batch.source for batch in batches],
        "generator_method_identity_feature": False,
        "feature_mode": "local",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "quality_coefficients": quality.coefficients.detach().cpu(),
            "pairwise_coefficients": pairwise.coefficients.detach().cpu(),
            "metadata": metadata,
        },
        path,
    )
    return quality, pairwise, metadata


def _evaluate_proposal(
    *,
    method: str,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    proposal: ContrastiveCandidates,
    config: D16BenchmarkConfig,
    quality_router: BatchedRidgeCritic,
    pairwise_router: FrozenPairwiseRouter,
    random_seed: int,
    device: torch.device,
    proposal_seconds: float,
) -> CandidateEvaluation:
    batch, evaluations, q_equivalent, environment_seconds = make_evaluated_batch(
        source=method,
        tasks=tasks,
        starts=starts,
        proposal=proposal,
        base=config.base,
        router_config=config.router,
        device=device,
        refinement=True,
    )
    result = evaluate_candidate_batch(
        method=method,
        batch=batch,
        directions=proposal.directions,
        quality_router=quality_router,
        pairwise_router=pairwise_router,
        useful_threshold=config.base.base.landscape.useful_threshold,
        random_seed=random_seed,
        elapsed_seconds=proposal_seconds + environment_seconds,
        candidate_evaluations=evaluations,
        q_equivalent_evaluations=q_equivalent,
    )
    return result


def _evaluate_policy(
    *,
    method: str,
    policy: StochasticCreativePolicy,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    config: D16BenchmarkConfig,
    quality_router: BatchedRidgeCritic,
    pairwise_router: FrozenPairwiseRouter,
    action_seed: int,
    random_seed: int,
    device: torch.device,
) -> CandidateEvaluation:
    synchronize(device)
    started = perf_counter()
    proposal = _policy_proposal(policy, tasks, starts, seed=action_seed, device=device)
    synchronize(device)
    proposal_seconds = perf_counter() - started
    return _evaluate_proposal(
        method=method,
        tasks=tasks,
        starts=starts,
        proposal=proposal,
        config=config,
        quality_router=quality_router,
        pairwise_router=pairwise_router,
        random_seed=random_seed,
        device=device,
        proposal_seconds=proposal_seconds,
    )


def _evaluate_contrastive(
    *,
    method: str,
    form: str,
    alpha: float,
    vectors: ContrastiveVectors,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    config: D16BenchmarkConfig,
    quality_router: BatchedRidgeCritic,
    pairwise_router: FrozenPairwiseRouter,
    random_seed: int,
    device: torch.device,
) -> CandidateEvaluation:
    start_array, task_ids, _ = flatten_starts(starts)
    goals = np.stack([tasks[int(index)].goal for index in task_ids])
    h = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    g = torch.as_tensor(goals, dtype=torch.float32, device=device)
    synchronize(device)
    started = perf_counter()
    proposal = contrastive_candidates(
        h,
        g,
        vectors,
        kind="goal",
        alpha=alpha,
        form=form,
        candidate_count=config.base.base.training.candidates,
    )
    synchronize(device)
    proposal_seconds = perf_counter() - started
    return _evaluate_proposal(
        method=method,
        tasks=tasks,
        starts=starts,
        proposal=proposal,
        config=config,
        quality_router=quality_router,
        pairwise_router=pairwise_router,
        random_seed=random_seed,
        device=device,
        proposal_seconds=proposal_seconds,
    )


def _save_vectors(path: Path, vectors: ContrastiveVectors) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_direction": vectors.global_direction,
            "goal_directions": vectors.goal_directions,
            "target_anchor_directions": vectors.target_anchor_directions,
            "useful_decoy_directions": vectors.useful_decoy_directions,
            "goal_prototypes": vectors.goal_prototypes,
            "raw_norms": vectors.raw_norms,
            "sample_counts": vectors.sample_counts,
            "split": vectors.split,
            "task_fingerprint": vectors.task_fingerprint,
            "sampling_seconds": vectors.sampling_seconds,
            "vector_seconds": vectors.vector_seconds,
            "construction_seconds": vectors.construction_seconds,
        },
        path,
    )


def _load_vectors(path: Path) -> ContrastiveVectors:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return ContrastiveVectors(**payload)


def _summaries(
    rows: Sequence[Mapping[str, object]],
    *,
    group_keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = {name: value for name, value in zip(group_keys, key)}
        for metric in metrics:
            stats = statistics([float(row[metric]) for row in values])
            summary.update(
                {
                    f"{metric}_mean": stats["mean"],
                    f"{metric}_sd": stats["sd"],
                    f"{metric}_ci95": stats["ci95"],
                }
            )
        summary["seed_count"] = len(values)
        output.append(summary)
    return output


def _paired_rows(
    rows: Sequence[Mapping[str, object]],
    left: str,
    right: str,
) -> list[dict[str, object]]:
    lookup = {(int(row["seed"]), str(row["method"])): row for row in rows}
    seeds = sorted(
        seed for seed, method in lookup if method == left and (seed, right) in lookup
    )
    output = []
    for metric in PRIMARY_METRICS:
        differences = [
            float(lookup[(seed, left)][metric]) - float(lookup[(seed, right)][metric])
            for seed in seeds
        ]
        stats = statistics(differences)
        output.append(
            {
                "left": left,
                "right": right,
                "metric": metric,
                "mean_paired_difference": stats["mean"],
                "sd": stats["sd"],
                "ci95": stats["ci95"],
                "ci95_low": stats["mean"] - stats["ci95"],
                "ci95_high": stats["mean"] + stats["ci95"],
                "resolved_from_zero": abs(stats["mean"]) > stats["ci95"],
                "seed_count": len(seeds),
            }
        )
    return output


def _select_alphas(
    rows: Sequence[Mapping[str, object]], config: D16BenchmarkConfig
) -> tuple[dict[str, float], list[dict[str, object]]]:
    selected: dict[str, float] = {}
    summaries: list[dict[str, object]] = []
    for form in config.contrastive.proposal_forms:
        grouped: list[dict[str, object]] = []
        for alpha in config.contrastive.sweep.alphas:
            subset = [
                row
                for row in rows
                if row["form"] == form and float(row["alpha"]) == alpha
            ]
            summary: dict[str, object] = {"kind": "goal", "form": form, "alpha": alpha}
            for metric in [
                "candidate_mean_quality",
                "proposal_recall",
                "useful_region_count",
                "useful_region_count_2_plus",
                "candidate_decoy_rate",
                "candidate_target_rate",
                "selected_quality",
                "useful_selected",
            ]:
                stats = statistics([float(row[metric]) for row in subset])
                summary[f"{metric}_mean"] = stats["mean"]
                summary[f"{metric}_sd"] = stats["sd"]
                summary[f"{metric}_ci95"] = stats["ci95"]
                # Retain these keys so the shared tie-break rule ranks the same mean metrics.
                summary[metric] = stats["mean"]
            grouped.append(summary)
            summaries.append(summary)
        reference = next(row for row in grouped if float(row["alpha"]) == 0.0)
        eligible = [
            row
            for row in grouped
            if float(row["candidate_mean_quality"])
            >= float(reference["candidate_mean_quality"])
            - config.contrastive.sweep.quality_tolerance
            and float(row["candidate_decoy_rate"])
            <= float(reference["candidate_decoy_rate"])
            + config.contrastive.sweep.decoy_tolerance
        ]
        winner = max(
            eligible or grouped,
            key=lambda row: (
                float(row["proposal_recall"]),
                float(row["useful_region_count"]),
                float(row["candidate_target_rate"]),
                float(row["candidate_mean_quality"]),
                -float(row["alpha"]),
            ),
        )
        selected[f"goal:{form}"] = float(winner["alpha"])
    return selected, summaries


def _geometry_audit(
    family: D16LandscapeFamily,
    config: D16BenchmarkConfig,
    output: Path,
) -> dict[str, object]:
    variants: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for offset, variant in enumerate(config.generalization_variants):
        tasks = family.sample_tasks(
            geometries=config.base.base.test_geometries,
            seed=91_000 + offset,
            variant=variant,
            all_goals=True,
        )
        diagnostic = geometry_diagnostics(tasks)
        variants[variant] = diagnostic
        rows.append(
            {
                "variant": variant,
                **{
                    key: value
                    for key, value in diagnostic.items()
                    if not isinstance(value, list)
                },
            }
        )
    rotation = random_orthogonal(np.random.default_rng(91_999), config.dimension)
    payload: dict[str, object] = {
        "dimension": config.dimension,
        "variant_diagnostics": variants,
        "rotation_orthogonality_max_error": float(
            np.max(np.abs(rotation.T @ rotation - np.eye(config.dimension)))
        ),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "all_variants_not_effectively_2d": all(
            bool(value["not_effectively_2d"])
            for value in variants.values()  # type: ignore[union-attr]
        ),
    }
    (output / "geometry_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(output / "geometry_audit.csv", rows)
    return payload


def run_d16_study(
    config: D16BenchmarkConfig,
    *,
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
    run_generalization: bool = True,
) -> dict[str, object]:
    if config.dimension != 16 or config.base.base.landscape.dimension != 16:
        raise ValueError("the high-dimensional benchmark is fixed at d=16")
    if config.base.base.training.candidates != 8:
        raise ValueError("the matched benchmark requires K=8")
    if config.base.base.training.refinement_steps != 9:
        raise ValueError("the matched benchmark requires nine refinement steps")
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    vector_dir = output / "vectors"
    router_dir = output / "routers"
    for directory in [model_dir, vector_dir, router_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    config_payload = config.to_dict()
    config_payload["runtime"] = runtime.to_dict()
    config_payload["scientific_changes"] = [
        "latent dimension 2 -> 16",
        "full-rank 16D orientation frame replacing legacy 2D plane embedding",
    ]
    config_payload["frozen_semantics"] = [
        "K=8",
        "nine refinement steps",
        "100000 returned candidate outcomes per trained seed in main",
        "float32 without TF32 or AMP",
        "same local pairwise useful-novelty router methodology for every method",
    ]
    (output / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    family = D16LandscapeFamily(config.base.base.landscape)
    geometry = _geometry_audit(family, config, output)
    if not bool(geometry["all_variants_not_effectively_2d"]):
        raise RuntimeError("d=16 geometry audit detected an effectively 2D variant")

    training_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    contrast_data_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    run_started = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Finish fitting and validation selection first to exclude test outcomes.
    for seed_index, seed in enumerate(config.seeds):
        print(
            f"[d16 train {seed_index + 1}/{len(config.seeds)}] seed {seed}",
            flush=True,
        )
        train, train_starts, validation, validation_starts, _, _ = _datasets(
            family, config, seed
        )
        rl_policy, rl_result, rl_metadata = _train_one_policy(
            method="csd_rl_pairwise",
            config=config,
            train_tasks=train,
            train_starts=train_starts,
            seed=seed,
            device=device,
            path=model_dir / f"seed_{seed}_csd_rl.pt",
        )
        mode_policy, mode_result, mode_metadata = _train_one_policy(
            method="csd_mode_pairwise",
            config=config,
            train_tasks=train,
            train_starts=train_starts,
            seed=seed,
            device=device,
            path=model_dir / f"seed_{seed}_csd_mode.pt",
        )
        for metadata in [rl_metadata, mode_metadata]:
            training_rows.append(dict(metadata))
        for method, result in [
            ("csd_rl_pairwise", rl_result),
            ("csd_mode_pairwise", mode_result),
        ]:
            if result is not None:
                history_rows.extend(
                    {"seed": seed, "method": method, **row} for row in result.history
                )

        vector_path = vector_dir / f"seed_{seed}_contrastive.pt"
        if vector_path.exists():
            vectors = _load_vectors(vector_path)
        else:
            vectors = build_contrastive_vectors(
                train,
                split="train_iid",
                seed=seed + 60_001,
                config=config.contrastive.data,
            )
            _save_vectors(vector_path, vectors)
        contrast_data_rows.append(
            {
                "seed": seed,
                "dimension": config.dimension,
                "split": vectors.split,
                "task_fingerprint": vectors.task_fingerprint,
                "construction_seconds": vectors.construction_seconds,
                **vectors.sample_counts,
                **{
                    f"raw_norm_{kind}_mean": float(np.mean(values))
                    for kind, values in vectors.raw_norms.items()
                },
            }
        )
        quality_router, pairwise_router, router_metadata = _fit_shared_routers(
            config=config,
            train_tasks=train,
            train_starts=train_starts,
            rl_policy=rl_policy,
            mode_policy=mode_policy,
            seed=seed,
            device=device,
            path=router_dir / f"seed_{seed}_shared_pairwise.pt",
        )
        training_rows.append(
            {"seed": seed, "method": "shared_pairwise_router", **router_metadata}
        )

        if seed_index < config.contrastive.sweep.seeds:
            validation_nested = _nested(validation_starts)
            for form in config.contrastive.proposal_forms:
                for alpha in config.contrastive.sweep.alphas:
                    result = _evaluate_contrastive(
                        method=f"contrast_goal_{'1' if form == 'single' else 'k'}",
                        form=form,
                        alpha=alpha,
                        vectors=vectors,
                        tasks=validation,
                        starts=validation_nested,
                        config=config,
                        quality_router=quality_router,
                        pairwise_router=pairwise_router,
                        random_seed=seed + 61_001,
                        device=device,
                    )
                    alpha_rows.append(
                        {"seed": seed, "form": form, "alpha": alpha, **result.portfolio}
                    )

    selected_alphas, alpha_summary = _select_alphas(alpha_rows, config)
    selection_payload = {
        "selection_data": "held-out IID validation landscapes",
        "test_set_used": False,
        "direction_data_split": "training only",
        "seeds": config.seeds[: config.contrastive.sweep.seeds],
        "alpha_grid": config.contrastive.sweep.alphas,
        "selection_rule": (
            "eligible when candidate quality >= alpha0-0.05 and decoy <= "
            "alpha0+0.05; maximize proposal recall, then distinct useful "
            "regions, target rate, quality, and prefer smaller alpha"
        ),
        "selected_alphas": selected_alphas,
    }
    (output / "CONTRASTIVE_ALPHA_SELECTION.json").write_text(
        json.dumps(selection_payload, indent=2), encoding="utf-8"
    )
    write_csv(output / "alpha_sweep_per_seed.csv", alpha_rows)
    write_csv(output / "alpha_sweep_summary.csv", alpha_summary)
    write_csv(output / "contrastive_training_data.csv", contrast_data_rows)
    write_csv(output / "training_cost_per_seed.csv", training_rows)
    write_csv(output / "training_history.csv", history_rows)

    primary_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    generalization_rows: list[dict[str, object]] = []

    # Share tasks, starts, routers and budgets so method comparisons stay paired.
    for seed_index, seed in enumerate(config.seeds):
        print(
            f"[d16 evaluate {seed_index + 1}/{len(config.seeds)}] seed {seed}",
            flush=True,
        )
        train, train_starts, _, _, test, test_starts = _datasets(family, config, seed)
        rl_policy, _, _ = _train_one_policy(
            method="csd_rl_pairwise",
            config=config,
            train_tasks=train,
            train_starts=train_starts,
            seed=seed,
            device=device,
            path=model_dir / f"seed_{seed}_csd_rl.pt",
        )
        mode_policy, _, _ = _train_one_policy(
            method="csd_mode_pairwise",
            config=config,
            train_tasks=train,
            train_starts=train_starts,
            seed=seed,
            device=device,
            path=model_dir / f"seed_{seed}_csd_mode.pt",
        )
        vectors = _load_vectors(vector_dir / f"seed_{seed}_contrastive.pt")
        quality_router, pairwise_router, _ = _fit_shared_routers(
            config=config,
            train_tasks=train,
            train_starts=train_starts,
            rl_policy=rl_policy,
            mode_policy=mode_policy,
            seed=seed,
            device=device,
            path=router_dir / f"seed_{seed}_shared_pairwise.pt",
        )
        results = {
            "contrast_goal_1": _evaluate_contrastive(
                method="contrast_goal_1",
                form="single",
                alpha=selected_alphas["goal:single"],
                vectors=vectors,
                tasks=test,
                starts=test_starts,
                config=config,
                quality_router=quality_router,
                pairwise_router=pairwise_router,
                random_seed=seed + 71_001,
                device=device,
            ),
            "contrast_goal_k": _evaluate_contrastive(
                method="contrast_goal_k",
                form="k_radius",
                alpha=selected_alphas["goal:k_radius"],
                vectors=vectors,
                tasks=test,
                starts=test_starts,
                config=config,
                quality_router=quality_router,
                pairwise_router=pairwise_router,
                random_seed=seed + 71_001,
                device=device,
            ),
            "csd_rl_pairwise": _evaluate_policy(
                method="csd_rl_pairwise",
                policy=rl_policy,
                tasks=test,
                starts=test_starts,
                config=config,
                quality_router=quality_router,
                pairwise_router=pairwise_router,
                action_seed=seed + 71_010,
                random_seed=seed + 71_001,
                device=device,
            ),
            "csd_mode_pairwise": _evaluate_policy(
                method="csd_mode_pairwise",
                policy=mode_policy,
                tasks=test,
                starts=test_starts,
                config=config,
                quality_router=quality_router,
                pairwise_router=pairwise_router,
                action_seed=seed + 71_010,
                random_seed=seed + 71_001,
                device=device,
            ),
        }
        for method, result in results.items():
            primary_rows.append({"seed": seed, "method": method, **result.portfolio})
            selector_rows.extend({"seed": seed, **row} for row in result.selector_rows)
            set_rows.extend({"seed": seed, **row} for row in result.set_rows)

        if run_generalization:
            for variant_index, variant in enumerate(config.generalization_variants):
                tasks = family.sample_tasks(
                    geometries=config.base.base.test_geometries,
                    seed=seed + 80_000 + 1_000 * variant_index,
                    variant=variant,
                    all_goals=True,
                )
                starts = _make_starts(
                    tasks,
                    trials=config.base.base.trials_per_task,
                    seed=seed + 80_001 + 1_000 * variant_index,
                )
                variant_results = {
                    "contrast_goal_1": _evaluate_contrastive(
                        method="contrast_goal_1",
                        form="single",
                        alpha=selected_alphas["goal:single"],
                        vectors=vectors,
                        tasks=tasks,
                        starts=starts,
                        config=config,
                        quality_router=quality_router,
                        pairwise_router=pairwise_router,
                        random_seed=seed + 81_001 + variant_index,
                        device=device,
                    ),
                    "contrast_goal_k": _evaluate_contrastive(
                        method="contrast_goal_k",
                        form="k_radius",
                        alpha=selected_alphas["goal:k_radius"],
                        vectors=vectors,
                        tasks=tasks,
                        starts=starts,
                        config=config,
                        quality_router=quality_router,
                        pairwise_router=pairwise_router,
                        random_seed=seed + 81_001 + variant_index,
                        device=device,
                    ),
                    "csd_rl_pairwise": _evaluate_policy(
                        method="csd_rl_pairwise",
                        policy=rl_policy,
                        tasks=tasks,
                        starts=starts,
                        config=config,
                        quality_router=quality_router,
                        pairwise_router=pairwise_router,
                        action_seed=seed + 81_010 + variant_index,
                        random_seed=seed + 81_001 + variant_index,
                        device=device,
                    ),
                    "csd_mode_pairwise": _evaluate_policy(
                        method="csd_mode_pairwise",
                        policy=mode_policy,
                        tasks=tasks,
                        starts=starts,
                        config=config,
                        quality_router=quality_router,
                        pairwise_router=pairwise_router,
                        action_seed=seed + 81_010 + variant_index,
                        random_seed=seed + 81_001 + variant_index,
                        device=device,
                    ),
                }
                for method, result in variant_results.items():
                    generalization_rows.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "method": method,
                            **result.portfolio,
                        }
                    )

    primary_summary = _summaries(
        primary_rows, group_keys=["method"], metrics=PRIMARY_METRICS
    )
    generalization_summary = (
        _summaries(
            generalization_rows,
            group_keys=["variant", "method"],
            metrics=PRIMARY_METRICS,
        )
        if generalization_rows
        else []
    )
    paired = []
    for left, right in [
        ("csd_rl_pairwise", "contrast_goal_k"),
        ("csd_mode_pairwise", "contrast_goal_k"),
        ("csd_mode_pairwise", "csd_rl_pairwise"),
    ]:
        paired.extend(_paired_rows(primary_rows, left, right))

    write_csv(output / "primary_per_seed.csv", primary_rows)
    write_csv(output / "primary_summary.csv", primary_summary)
    write_csv(output / "selector_per_seed.csv", selector_rows)
    write_csv(output / "candidate_set_diagnostics.csv", set_rows)
    write_csv(output / "paired_differences.csv", paired)
    write_csv(output / "generalization_per_seed.csv", generalization_rows)
    write_csv(output / "generalization_summary.csv", generalization_summary)
    synchronize(device)
    aggregate = {
        "seed_count": len(config.seeds),
        "dimension": config.dimension,
        "candidate_count": config.base.base.training.candidates,
        "refinement_steps": config.base.base.training.refinement_steps,
        "selected_alphas": selected_alphas,
        "geometry_audit": geometry,
        "runtime": runtime.to_dict(),
        "end_to_end_seconds": perf_counter() - run_started,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "primary_summary": primary_summary,
    }
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
