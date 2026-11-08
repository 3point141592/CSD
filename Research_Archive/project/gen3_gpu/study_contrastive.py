"""Validation-only steering selection and frozen 30-seed CSD comparison."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from creativity_toy.config import STUDY_SEEDS
from creativity_toy.config_gen3 import Gen3StudyConfig, preset_gen3
from creativity_toy.experiments.study_v2 import _make_starts
from creativity_toy.landscapes_v2 import V2LandscapeFamily, V2Task

from .config_contrastive import ContrastiveExperimentConfig
from .config_rl import RLStudyConfig
from .config_router import RouterExperimentConfig
from .contrastive_evaluation import (
    CandidateEvaluation,
    established_baseline_candidates,
    evaluate_candidate_batch,
    flatten_starts,
    make_evaluated_batch,
)
from .contrastive_steering import (
    ContrastiveCandidates,
    ContrastiveVectors,
    build_contrastive_vectors,
    contrastive_candidates,
    vectors_to_numpy,
)
from .creative_policy_rl import StochasticCreativePolicy
from .evaluation_mode import load_frozen_pairwise_router, router_integrity
from .frozen_router_candidates import (
    FrozenGenerator,
    generate_frozen_candidates,
    generator_integrity,
    load_frozen_generator,
)
from .runtime import RuntimeInfo, synchronize
from .study import write_csv
from .study_rl import _datasets

PRIMARY_CONTRAST_METHODS = (
    "contrast_global_1",
    "contrast_goal_1",
    "contrast_target_anchor_1",
    "contrast_useful_decoy_1",
    "contrast_global_k",
    "contrast_goal_k",
    "contrast_target_anchor_k",
    "contrast_useful_decoy_k",
)
GENERALIZATION_METHODS = (
    "random",
    "pure_divergence",
    "contrast_global_1",
    "contrast_goal_1",
    "contrast_target_anchor_1",
    "contrast_global_k",
    "contrast_goal_k",
    "contrast_target_anchor_k",
    "csd_rl",
    "csd_mode",
    "csd_es",
)


def default_configs() -> tuple[Gen3StudyConfig, RLStudyConfig, RouterExperimentConfig]:
    return preset_gen3("main"), RLStudyConfig(), RouterExperimentConfig()


def _nested(starts: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
    return [[np.asarray(value, dtype=np.float64)] for value in starts]


def _method_spec(method: str) -> tuple[str, str]:
    if not method.startswith("contrast_"):
        raise ValueError(method)
    suffix = "k_radius" if method.endswith("_k") else "single"
    root = method[len("contrast_") : -2]
    aliases = {
        "global": "global",
        "goal": "goal",
        "target_anchor": "target_anchor",
        "useful_decoy": "useful_decoy",
    }
    return aliases[root], suffix


def _mode_generator(
    *,
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    seed: int,
    device: torch.device,
    quality_source: FrozenGenerator,
) -> FrozenGenerator:
    policy = StochasticCreativePolicy(
        dimension=base.base.landscape.dimension,
        candidates=base.base.training.candidates,
        latent_dimension=base.base.training.latent_dimension,
        hidden_dimension=base.base.training.hidden_dimension,
        seed=seed + 10_020,
        device=device,
        dtype=torch.float32,
        alpha_max=rl.alpha_max,
        direction_std_init=rl.training.direction_std_init,
        alpha_std_init=rl.training.alpha_std_init,
    )
    payload = torch.load(
        Path("results/mode_preservation_main/models")
        / f"seed_{seed}_gated_shared_selected.pt",
        map_location=device,
        weights_only=True,
    )
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    snapshot = [value.detach().clone() for value in policy.state_dict().values()]
    return FrozenGenerator(
        kind="rl",
        operator=policy,
        theta=torch.empty(0, dtype=torch.float32, device=device),
        quality_router=quality_source.quality_router,
        parameter_snapshot=snapshot,
        parameter_hash=_tensor_hash(snapshot),
    )


def _tensor_hash(tensors: Sequence[torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _evaluate_proposal(
    *,
    method: str,
    proposal: ContrastiveCandidates,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    base: Gen3StudyConfig,
    router_config: RouterExperimentConfig,
    quality_generator: FrozenGenerator,
    pairwise_router,
    device: torch.device,
    refinement: bool,
    random_seed: int,
) -> CandidateEvaluation:
    baseline_memory = 0
    if device.type == "cuda":
        baseline_memory = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    batch, calls, q_calls, elapsed = make_evaluated_batch(
        source=method,
        tasks=tasks,
        starts=starts,
        proposal=proposal,
        base=base,
        router_config=router_config,
        device=device,
        refinement=refinement,
    )
    result = evaluate_candidate_batch(
        method=method,
        batch=batch,
        directions=proposal.directions,
        quality_router=quality_generator.quality_router,
        pairwise_router=pairwise_router,
        useful_threshold=base.base.landscape.useful_threshold,
        random_seed=random_seed,
        elapsed_seconds=elapsed,
        candidate_evaluations=calls,
        q_equivalent_evaluations=q_calls,
    )
    if device.type == "cuda":
        result.portfolio["peak_incremental_bytes"] = max(
            0, int(torch.cuda.max_memory_allocated(device)) - baseline_memory
        )
    else:
        result.portfolio["peak_incremental_bytes"] = 0
    return result


def _evaluate_frozen(
    *,
    method: str,
    generator: FrozenGenerator,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    base: Gen3StudyConfig,
    router_config: RouterExperimentConfig,
    quality_generator: FrozenGenerator,
    pairwise_router,
    device: torch.device,
    action_seed: int,
    random_seed: int,
) -> CandidateEvaluation:
    baseline_memory = 0
    if device.type == "cuda":
        baseline_memory = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = perf_counter()
    batch = generate_frozen_candidates(
        generator,
        tasks=tasks,
        starts=starts,
        action_seed=action_seed,
        base=base,
        router_config=router_config,
        device=device,
        source=method,
    )
    synchronize(device)
    elapsed = perf_counter() - started
    delta = batch.candidates - batch.starts[:, None, :]
    directions = torch.nn.functional.normalize(delta, p=2, dim=-1, eps=1e-8)
    calls = batch.batch_size * batch.candidate_count
    result = evaluate_candidate_batch(
        method=method,
        batch=batch,
        directions=directions,
        quality_router=quality_generator.quality_router,
        pairwise_router=pairwise_router,
        useful_threshold=base.base.landscape.useful_threshold,
        random_seed=random_seed,
        elapsed_seconds=elapsed,
        candidate_evaluations=calls,
        q_equivalent_evaluations=calls * (base.base.training.refinement_steps + 1),
    )
    if device.type == "cuda":
        result.portfolio["peak_incremental_bytes"] = max(
            0, int(torch.cuda.max_memory_allocated(device)) - baseline_memory
        )
    else:
        result.portfolio["peak_incremental_bytes"] = 0
    return result


def _attach(
    rows: Sequence[Mapping[str, object]], **values: object
) -> list[dict[str, object]]:
    return [{**values, **dict(row)} for row in rows]


def _selection_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_set_used") is not False:
        raise ValueError("contrastive alpha selection was not validation-only")
    return payload


def run_contrastive_sweep(
    *,
    config: ContrastiveExperimentConfig,
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
) -> dict[str, object]:
    """Select all steering radii using IID validation landscapes only."""
    output.mkdir(parents=True, exist_ok=True)
    base, rl, router_config = default_configs()
    if base.base.landscape.dimension != config.dimension or config.dimension != 2:
        raise ValueError("contrastive steering is intentionally fixed at d=2")
    family = V2LandscapeFamily(base.base.landscape)
    seeds = list(STUDY_SEEDS[: config.sweep.seeds])
    rows: list[dict[str, object]] = []
    data_rows: list[dict[str, object]] = []
    for seed in seeds:
        train, _, validation, validation_starts, _, _ = _datasets(
            family, base, rl, seed
        )
        vectors = build_contrastive_vectors(
            train, split="train_iid", seed=seed + 60_001, config=config.data
        )
        data_rows.append(
            {
                "seed": seed,
                "split": vectors.split,
                "task_fingerprint": vectors.task_fingerprint,
                "sampling_seconds": vectors.sampling_seconds,
                "vector_seconds": vectors.vector_seconds,
                "construction_seconds": vectors.construction_seconds,
                **vectors.sample_counts,
                **{
                    f"raw_norm_{kind}_mean": float(np.mean(values))
                    for kind, values in vectors.raw_norms.items()
                },
            }
        )
        starts = _nested(validation_starts)
        start_array, _, _ = flatten_starts(starts)
        goals = torch.as_tensor(
            np.stack([task.goal for task in validation]),
            dtype=torch.float32,
            device=device,
        )
        start_tensor = torch.as_tensor(start_array, dtype=torch.float32, device=device)
        quality_generator = load_frozen_generator(
            "rl", base=base, rl=rl, seed=seed, device=device
        )
        pairwise = load_frozen_pairwise_router(seed, device)
        for kind in config.direction_kinds:
            for form in config.proposal_forms:
                for alpha in config.sweep.alphas:
                    proposal = contrastive_candidates(
                        start_tensor,
                        goals,
                        vectors,
                        kind=kind,
                        alpha=alpha,
                        form=form,
                        candidate_count=config.candidate_count,
                    )
                    result = _evaluate_proposal(
                        method=f"contrast_{kind}_{'1' if form == 'single' else 'k'}",
                        proposal=proposal,
                        tasks=validation,
                        starts=starts,
                        base=base,
                        router_config=router_config,
                        quality_generator=quality_generator,
                        pairwise_router=pairwise,
                        device=device,
                        refinement=True,
                        random_seed=seed + 61_001,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "kind": kind,
                            "form": form,
                            "alpha": alpha,
                            **result.portfolio,
                        }
                    )
    write_csv(output / "alpha_sweep_per_seed.csv", rows)
    write_csv(output / "contrastive_training_data.csv", data_rows)
    selected: dict[str, float] = {}
    aggregate_rows: list[dict[str, object]] = []
    for kind in config.direction_kinds:
        for form in config.proposal_forms:
            grouped = []
            subset = [
                row for row in rows if row["kind"] == kind and row["form"] == form
            ]
            for alpha in config.sweep.alphas:
                alpha_rows = [row for row in subset if float(row["alpha"]) == alpha]
                summary: dict[str, object] = {
                    "kind": kind,
                    "form": form,
                    "alpha": alpha,
                }
                for metric in [
                    "candidate_mean_quality",
                    "proposal_recall",
                    "useful_region_count",
                    "useful_region_count_2_plus",
                    "candidate_decoy_rate",
                    "candidate_target_rate",
                    "useful_selected",
                    "selected_quality",
                ]:
                    values = np.asarray([float(row[metric]) for row in alpha_rows])
                    summary[metric] = float(np.mean(values))
                    summary[f"{metric}_sd"] = float(np.std(values, ddof=0))
                    summary[f"{metric}_ci95"] = (
                        float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else 0.0
                    )
                grouped.append(summary)
                aggregate_rows.append(summary)
            reference = next(row for row in grouped if float(row["alpha"]) == 0.0)
            eligible = [
                row
                for row in grouped
                if float(row["candidate_mean_quality"])
                >= float(reference["candidate_mean_quality"])
                - config.sweep.quality_tolerance
                and float(row["candidate_decoy_rate"])
                <= float(reference["candidate_decoy_rate"])
                + config.sweep.decoy_tolerance
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
            selected[f"{kind}:{form}"] = float(winner["alpha"])
    write_csv(output / "alpha_sweep_summary.csv", aggregate_rows)
    payload: dict[str, object] = {
        "selection_data": "held-out IID validation landscapes from the training family",
        "test_set_used": False,
        "direction_data_split": "training only",
        "seeds": seeds,
        "alpha_grid": config.sweep.alphas,
        "selection_rule": "eligible when candidate quality >= alpha0-0.05 and decoy <= alpha0+0.05; maximize proposal recall, then distinct useful regions, target rate, quality, and prefer smaller alpha",
        "k_radius_schedule": "K positive radii evenly spaced from alpha/K through alpha",
        "selected_alphas": selected,
        "config": config.to_dict(),
        "runtime": runtime.to_dict(),
    }
    (output / "CONTRASTIVE_ALPHA_SELECTION.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _contrast_proposal_for_method(
    method: str,
    *,
    starts: torch.Tensor,
    goals: torch.Tensor,
    vectors: ContrastiveVectors,
    selected_alphas: Mapping[str, object],
    candidate_count: int,
) -> ContrastiveCandidates:
    kind, form = _method_spec(method)
    return contrastive_candidates(
        starts,
        goals,
        vectors,
        kind=kind,
        alpha=float(selected_alphas[f"{kind}:{form}"]),
        form=form,
        candidate_count=candidate_count,
    )


def _record_evaluation(
    result: CandidateEvaluation,
    *,
    seed: int,
    method: str,
    refinement: str,
    primary_rows: list[dict[str, object]],
    selector_rows: list[dict[str, object]],
    set_rows: list[dict[str, object]],
) -> None:
    primary_rows.append(
        {"seed": seed, "method": method, "refinement": refinement, **result.portfolio}
    )
    selector_rows.extend(
        _attach(result.selector_rows, seed=seed, method=method, refinement=refinement)
    )
    set_rows.extend(
        _attach(result.set_rows, seed=seed, method=method, refinement=refinement)
    )


def run_contrastive_main(
    *,
    config: ContrastiveExperimentConfig,
    selection_path: Path,
    seeds: Sequence[int],
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
) -> dict[str, object]:
    """Run paired frozen-policy, matched-K tests and held-out generalization."""
    output.mkdir(parents=True, exist_ok=True)
    vector_dir = output / "vectors"
    vector_dir.mkdir(exist_ok=True)
    base, rl, router_config = default_configs()
    if base.base.landscape.dimension != 2 or config.dimension != 2:
        raise ValueError("this benchmark must remain in d=2")
    selection = _selection_payload(selection_path)
    selected_alphas = selection["selected_alphas"]
    assert isinstance(selected_alphas, dict)
    family = V2LandscapeFamily(base.base.landscape)
    payload = {
        "generation3": base.to_dict(),
        "rl": rl.to_dict(),
        "router": router_config.to_dict(),
        "contrastive": config.to_dict(),
        "alpha_selection": selection,
        "seeds": list(seeds),
        "runtime": runtime.to_dict(),
        "scientific_contract": {
            "dimension": 2,
            "csd_retrained": False,
            "router_retrained": False,
            "directions_use_training_states_only": True,
            "alpha_uses_validation_only": True,
            "test_used_for_fitting": False,
            "matched_candidate_count": config.candidate_count,
            "matched_refinement_steps": base.base.training.refinement_steps,
        },
    }
    (output / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    primary_rows, selector_rows, set_rows = [], [], []
    no_refinement_rows, generalization_rows = [], []
    data_rows, cost_rows, integrity_rows = [], [], []

    for seed_index, seed in enumerate(seeds):
        print(
            f"[contrastive main {seed_index + 1}/{len(seeds)}] seed {seed}", flush=True
        )
        train, _, _, _, test, test_starts = _datasets(family, base, rl, seed)
        vectors = build_contrastive_vectors(
            train, split="train_iid", seed=seed + 60_001, config=config.data
        )
        np.savez(
            vector_dir / f"seed_{seed}_contrastive_vectors.npz",
            **vectors_to_numpy(vectors),
            split=np.asarray(vectors.split),
            task_fingerprint=np.asarray(vectors.task_fingerprint),
        )
        data_rows.append(
            {
                "seed": seed,
                "split": vectors.split,
                "task_fingerprint": vectors.task_fingerprint,
                "sampling_seconds": vectors.sampling_seconds,
                "vector_seconds": vectors.vector_seconds,
                "construction_seconds": vectors.construction_seconds,
                **vectors.sample_counts,
                **{
                    f"raw_norm_{kind}_mean": float(np.mean(values))
                    for kind, values in vectors.raw_norms.items()
                },
            }
        )
        rl_generator = load_frozen_generator(
            "rl", base=base, rl=rl, seed=seed, device=device
        )
        es_generator = load_frozen_generator(
            "es", base=base, rl=rl, seed=seed, device=device
        )
        mode_generator = _mode_generator(
            base=base, rl=rl, seed=seed, device=device, quality_source=rl_generator
        )
        pairwise = load_frozen_pairwise_router(seed, device)
        router_before = pairwise.parameter_hash
        start_array, task_ids, _ = flatten_starts(test_starts)
        starts_tensor = torch.as_tensor(start_array, dtype=torch.float32, device=device)
        goals_tensor = torch.as_tensor(
            np.stack([test[index].goal for index in task_ids]),
            dtype=torch.float32,
            device=device,
        )

        for method in PRIMARY_CONTRAST_METHODS:
            proposal = _contrast_proposal_for_method(
                method,
                starts=starts_tensor,
                goals=goals_tensor,
                vectors=vectors,
                selected_alphas=selected_alphas,
                candidate_count=config.candidate_count,
            )
            matched = _evaluate_proposal(
                method=method,
                proposal=proposal,
                tasks=test,
                starts=test_starts,
                base=base,
                router_config=router_config,
                quality_generator=rl_generator,
                pairwise_router=pairwise,
                device=device,
                refinement=True,
                random_seed=seed + 62_101,
            )
            _record_evaluation(
                matched,
                seed=seed,
                method=method,
                refinement="matched_9_step",
                primary_rows=primary_rows,
                selector_rows=selector_rows,
                set_rows=set_rows,
            )
            unrefined = _evaluate_proposal(
                method=method,
                proposal=proposal,
                tasks=test,
                starts=test_starts,
                base=base,
                router_config=router_config,
                quality_generator=rl_generator,
                pairwise_router=pairwise,
                device=device,
                refinement=False,
                random_seed=seed + 62_101,
            )
            no_refinement_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "refinement": "none",
                    **unrefined.portfolio,
                }
            )
        for method in ["random", "pure_divergence"]:
            proposal = established_baseline_candidates(
                start_array,
                method=method,
                candidates=config.candidate_count,
                random_radius=base.base.training.random_radius,
                pure_divergence_radius=base.base.training.pure_divergence_radius,
                seed=seed + 62_301 + (1 if method == "random" else 2),
                device=device,
            )
            result = _evaluate_proposal(
                method=method,
                proposal=proposal,
                tasks=test,
                starts=test_starts,
                base=base,
                router_config=router_config,
                quality_generator=rl_generator,
                pairwise_router=pairwise,
                device=device,
                refinement=True,
                random_seed=seed + 62_401,
            )
            _record_evaluation(
                result,
                seed=seed,
                method=method,
                refinement="matched_9_step",
                primary_rows=primary_rows,
                selector_rows=selector_rows,
                set_rows=set_rows,
            )
        for method, generator, action_seed in [
            ("csd_rl", rl_generator, seed + 41_004),
            ("csd_mode", mode_generator, seed + 41_004),
            ("csd_es", es_generator, seed + 21_004),
        ]:
            result = _evaluate_frozen(
                method=method,
                generator=generator,
                tasks=test,
                starts=test_starts,
                base=base,
                router_config=router_config,
                quality_generator=rl_generator,
                pairwise_router=pairwise,
                device=device,
                action_seed=action_seed,
                random_seed=seed + 62_501,
            )
            _record_evaluation(
                result,
                seed=seed,
                method=method,
                refinement="matched_9_step",
                primary_rows=primary_rows,
                selector_rows=selector_rows,
                set_rows=set_rows,
            )
        for method, generator in [
            ("csd_rl", rl_generator),
            ("csd_mode", mode_generator),
            ("csd_es", es_generator),
        ]:
            integrity_rows.append(
                {"seed": seed, "method": method, **generator_integrity(generator)}
            )
        integrity_rows.append(
            {
                "seed": seed,
                "method": "frozen_pairwise_router",
                **router_integrity(pairwise, router_before),
            }
        )
        cost_rows.append(
            {
                "seed": seed,
                "method": "contrastive_vectors",
                "data_collection_and_construction_seconds": vectors.construction_seconds,
                "data_collection_seconds": vectors.sampling_seconds,
                "vector_construction_seconds": vectors.vector_seconds,
                "learned_parameter_count": 0,
                "stored_direction_scalars": int(
                    vectors.global_direction.numel()
                    + vectors.goal_directions.numel()
                    + vectors.target_anchor_directions.numel()
                    + vectors.useful_decoy_directions.numel()
                    + vectors.goal_prototypes.numel()
                ),
            }
        )

        for variant_index, landscape_variant in enumerate(
            config.generalization_variants
        ):
            variant_tasks = family.sample_tasks(
                geometries=base.base.test_geometries,
                seed=seed + 30_000 + variant_index,
                variant=landscape_variant,
                all_goals=True,
            )
            variant_starts = _make_starts(
                variant_tasks,
                trials=base.base.trials_per_task,
                seed=seed + 30_100 + variant_index,
            )
            variant_start_array, variant_task_ids, _ = flatten_starts(variant_starts)
            variant_start_tensor = torch.as_tensor(
                variant_start_array, dtype=torch.float32, device=device
            )
            variant_goals = torch.as_tensor(
                np.stack([variant_tasks[index].goal for index in variant_task_ids]),
                dtype=torch.float32,
                device=device,
            )
            for method in GENERALIZATION_METHODS:
                if method in {"random", "pure_divergence"}:
                    proposal = established_baseline_candidates(
                        variant_start_array,
                        method=method,
                        candidates=config.candidate_count,
                        random_radius=base.base.training.random_radius,
                        pure_divergence_radius=base.base.training.pure_divergence_radius,
                        seed=seed
                        + 63_001
                        + 20 * variant_index
                        + (1 if method == "random" else 2),
                        device=device,
                    )
                    result = _evaluate_proposal(
                        method=method,
                        proposal=proposal,
                        tasks=variant_tasks,
                        starts=variant_starts,
                        base=base,
                        router_config=router_config,
                        quality_generator=rl_generator,
                        pairwise_router=pairwise,
                        device=device,
                        refinement=True,
                        random_seed=seed + 63_101 + 20 * variant_index,
                    )
                elif method.startswith("contrast_"):
                    proposal = _contrast_proposal_for_method(
                        method,
                        starts=variant_start_tensor,
                        goals=variant_goals,
                        vectors=vectors,
                        selected_alphas=selected_alphas,
                        candidate_count=config.candidate_count,
                    )
                    result = _evaluate_proposal(
                        method=method,
                        proposal=proposal,
                        tasks=variant_tasks,
                        starts=variant_starts,
                        base=base,
                        router_config=router_config,
                        quality_generator=rl_generator,
                        pairwise_router=pairwise,
                        device=device,
                        refinement=True,
                        random_seed=seed + 63_101 + 20 * variant_index,
                    )
                else:
                    generator = {
                        "csd_rl": rl_generator,
                        "csd_mode": mode_generator,
                        "csd_es": es_generator,
                    }[method]
                    action_seed = (
                        seed + 30_500 + 20 * variant_index + 1
                        if method != "csd_es"
                        else seed + 30_500 + 20 * variant_index + 2
                    )
                    result = _evaluate_frozen(
                        method=method,
                        generator=generator,
                        tasks=variant_tasks,
                        starts=variant_starts,
                        base=base,
                        router_config=router_config,
                        quality_generator=rl_generator,
                        pairwise_router=pairwise,
                        device=device,
                        action_seed=action_seed,
                        random_seed=seed + 63_101 + 20 * variant_index,
                    )
                generalization_rows.append(
                    {
                        "seed": seed,
                        "landscape_variant": landscape_variant,
                        "method": method,
                        **result.portfolio,
                    }
                )
            if landscape_variant == "rotated":
                rotated_train = family.sample_tasks(
                    geometries=base.base.train_geometries,
                    seed=seed + 10_000,
                    variant="rotated",
                    all_goals=True,
                )
                rotated_vectors = build_contrastive_vectors(
                    rotated_train,
                    split="train_rotated",
                    seed=seed + 60_101,
                    config=config.data,
                )
                proposal = _contrast_proposal_for_method(
                    "contrast_goal_k",
                    starts=variant_start_tensor,
                    goals=variant_goals,
                    vectors=rotated_vectors,
                    selected_alphas=selected_alphas,
                    candidate_count=config.candidate_count,
                )
                result = _evaluate_proposal(
                    method="contrast_goal_k_recomputed_rotated",
                    proposal=proposal,
                    tasks=variant_tasks,
                    starts=variant_starts,
                    base=base,
                    router_config=router_config,
                    quality_generator=rl_generator,
                    pairwise_router=pairwise,
                    device=device,
                    refinement=True,
                    random_seed=seed + 63_501,
                )
                generalization_rows.append(
                    {
                        "seed": seed,
                        "landscape_variant": landscape_variant,
                        "method": "contrast_goal_k_recomputed_rotated",
                        **result.portfolio,
                    }
                )

    write_csv(output / "primary_per_seed.csv", primary_rows)
    write_csv(output / "selector_per_seed.csv", selector_rows)
    write_csv(output / "candidate_set_diagnostics.csv", set_rows)
    write_csv(output / "no_refinement_per_seed.csv", no_refinement_rows)
    write_csv(output / "generalization_per_seed.csv", generalization_rows)
    write_csv(output / "contrastive_training_data.csv", data_rows)
    write_csv(output / "cost_per_seed.csv", cost_rows)
    write_csv(output / "integrity.csv", integrity_rows)
    maximum_generator_difference = max(
        float(row.get("maximum_parameter_difference", 0.0))
        for row in integrity_rows
        if row["method"] != "frozen_pairwise_router"
    )
    maximum_router_difference = max(
        float(row.get("maximum_router_parameter_difference", 0.0))
        for row in integrity_rows
        if row["method"] == "frozen_pairwise_router"
    )
    aggregate = {
        "seed_count": len(seeds),
        "dimension": 2,
        "candidate_count": config.candidate_count,
        "refinement_steps": base.base.training.refinement_steps,
        "selected_alphas": selected_alphas,
        "maximum_frozen_generator_parameter_difference": maximum_generator_difference,
        "maximum_frozen_router_parameter_difference": maximum_router_difference,
        "test_data_used_for_direction": False,
        "test_data_used_for_alpha": False,
        "csd_retrained": False,
    }
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
