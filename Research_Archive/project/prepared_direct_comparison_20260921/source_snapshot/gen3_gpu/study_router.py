"""Validation-only selection and 30-seed frozen-generator router study."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from creativity_toy.config_gen3 import Gen3StudyConfig
from creativity_toy.experiments.study_v2 import _make_starts
from creativity_toy.landscapes_v2 import V2LandscapeFamily

from .config_rl import RLStudyConfig
from .config_router import RouterExperimentConfig
from .critic import critic_features
from .frozen_router_candidates import (
    FrozenGenerator,
    generate_broad_training_candidates,
    generate_frozen_candidates,
    generator_integrity,
    load_frozen_generator,
    save_candidate_batch,
)
from .router_evaluation import (
    conservative_diagnostic_summary,
    evaluate_selector,
    random_selector_scores,
    useful_oracle_scores,
)
from .router_useful_novelty import (
    FrozenCandidateBatch,
    LinearRouter,
    fit_pairwise_router,
    fit_regression_router,
    routing_target,
    target_distribution,
)
from .runtime import RuntimeInfo, synchronize
from .study import write_csv
from .study_rl import _datasets

GENERATOR_KINDS = ("rl", "es")
PRIMARY_SELECTORS = (
    "random",
    "quality_router",
    "novelty_only_router",
    "ungated_q_plus_n_router",
    "gated_useful_novelty_router",
    "set_aware_useful_novelty_router",
    "gated_useful_novelty_pairwise_router",
    "quality_oracle",
    "useful_oracle",
)
GENERALIZATION_SELECTORS = (
    "random",
    "quality_router",
    "gated_useful_novelty_router",
    "set_aware_useful_novelty_router",
    "quality_oracle",
    "useful_oracle",
)


def _nested(starts: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
    return [[np.asarray(value, dtype=np.float64)] for value in starts]


def _candidate_seeds(seed: int, generator: str) -> dict[str, int]:
    if generator == "rl":
        return {
            "train_action": seed + 40_022,
            "validation_action": seed + 50_000,
            "test_action": seed + 41_004,
            "broad": seed + 40_023,
            "random_test": seed + 41_104,
        }
    return {
        "train_action": seed + 10_020,
        "validation_action": seed + 15_020,
        "test_action": seed + 21_004,
        "broad": seed + 10_022,
        "random_test": seed + 21_004,
    }


def _make_training_batches(
    generator: FrozenGenerator,
    *,
    train_tasks,
    train_starts,
    seeds: Mapping[str, int],
    base: Gen3StudyConfig,
    config: RouterExperimentConfig,
    device: torch.device,
) -> tuple[FrozenCandidateBatch, FrozenCandidateBatch]:
    nested = _nested(train_starts)
    learned = generate_frozen_candidates(
        generator,
        tasks=train_tasks,
        starts=nested,
        action_seed=int(seeds["train_action"]),
        base=base,
        router_config=config,
        device=device,
        source=f"{generator.kind}_train_generated",
    )
    broad = generate_broad_training_candidates(
        tasks=train_tasks,
        starts=nested,
        broad_seed=int(seeds["broad"]),
        base=base,
        router_config=config,
        device=device,
        source=f"{generator.kind}_train_broad",
    )
    return learned, broad


def _validation_objective(
    batch: FrozenCandidateBatch,
    scores: torch.Tensor,
    *,
    useful_threshold: float,
    useful_weight: float,
    decoy_penalty: float,
) -> dict[str, float]:
    indices = torch.argmax(scores, dim=1)
    quality = torch.gather(batch.quality, 1, indices[:, None]).squeeze(1)
    regions = torch.gather(batch.regions, 1, indices[:, None]).squeeze(1)
    useful = (
        (regions >= 1) & (regions <= batch.n_useful) & (quality >= useful_threshold)
    )
    decoy = regions > batch.n_useful
    selected_quality = float(torch.mean(quality).cpu())
    useful_rate = float(torch.mean(useful.float()).cpu())
    decoy_rate = float(torch.mean(decoy.float()).cpu())
    return {
        "validation_objective": selected_quality
        + useful_weight * useful_rate
        - decoy_penalty * decoy_rate,
        "selected_quality": selected_quality,
        "useful_selection_rate": useful_rate,
        "decoy_selection_rate": decoy_rate,
    }


def run_router_sweep(
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    config: RouterExperimentConfig,
    *,
    seeds: Sequence[int],
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
) -> dict[tuple[str, str], dict[str, float]]:
    """Pre-test sweep; labels enter only the declared validation objective."""
    output.mkdir(parents=True, exist_ok=True)
    family = V2LandscapeFamily(base.base.landscape)
    sweep_seeds = list(seeds[: config.sweep.selection_seeds])
    cached: dict[
        tuple[int, str], tuple[list[FrozenCandidateBatch], FrozenCandidateBatch]
    ] = {}
    integrity_rows: list[dict[str, object]] = []
    for seed in sweep_seeds:
        train, train_starts, validation, validation_starts, _, _ = _datasets(
            family, base, rl, seed
        )
        for generator_kind in GENERATOR_KINDS:
            generator = load_frozen_generator(
                generator_kind, base=base, rl=rl, seed=seed, device=device
            )
            stream = _candidate_seeds(seed, generator_kind)
            learned, broad = _make_training_batches(
                generator,
                train_tasks=train,
                train_starts=train_starts,
                seeds=stream,
                base=base,
                config=config,
                device=device,
            )
            validation_batch = generate_frozen_candidates(
                generator,
                tasks=validation,
                starts=_nested(validation_starts),
                action_seed=stream["validation_action"],
                base=base,
                router_config=config,
                device=device,
                source=f"{generator_kind}_validation",
            )
            cached[(seed, generator_kind)] = ([learned, broad], validation_batch)
            integrity_rows.append(
                {"seed": seed, "stage": "sweep", **generator_integrity(generator)}
            )

    trial_rows: list[dict[str, object]] = []
    selected: dict[tuple[str, str], dict[str, float]] = {}
    for generator_kind in GENERATOR_KINDS:
        for feature_mode in ["local", "set_aware"]:
            for lambda_novelty in config.sweep.lambda_novelty:
                for quality_threshold in config.sweep.quality_thresholds:
                    for beta in config.sweep.quality_gate_betas:
                        for seed in sweep_seeds:
                            train_batches, validation = cached[(seed, generator_kind)]
                            router = fit_regression_router(
                                train_batches,
                                feature_mode=feature_mode,  # type: ignore[arg-type]
                                target_mode="gated",
                                lambda_novelty=lambda_novelty,
                                quality_threshold=quality_threshold,
                                quality_gate_beta=beta,
                                config=config,
                            )
                            scores = router.scores(validation, config)
                            result = _validation_objective(
                                validation,
                                scores,
                                useful_threshold=base.base.landscape.useful_threshold,
                                useful_weight=config.sweep.useful_weight,
                                decoy_penalty=config.sweep.decoy_penalty,
                            )
                            trial_rows.append(
                                {
                                    "generator": generator_kind,
                                    "feature_mode": feature_mode,
                                    "seed": seed,
                                    "lambda_novelty": lambda_novelty,
                                    "quality_threshold": quality_threshold,
                                    "quality_gate_beta": beta,
                                    **result,
                                }
                            )
            combinations = []
            for lambda_novelty in config.sweep.lambda_novelty:
                for quality_threshold in config.sweep.quality_thresholds:
                    for beta in config.sweep.quality_gate_betas:
                        rows = [
                            row
                            for row in trial_rows
                            if row["generator"] == generator_kind
                            and row["feature_mode"] == feature_mode
                            and row["lambda_novelty"] == lambda_novelty
                            and row["quality_threshold"] == quality_threshold
                            and row["quality_gate_beta"] == beta
                        ]
                        combinations.append(
                            {
                                "lambda_novelty": lambda_novelty,
                                "quality_threshold": quality_threshold,
                                "quality_gate_beta": beta,
                                "validation_objective": float(
                                    np.mean(
                                        [
                                            float(row["validation_objective"])
                                            for row in rows
                                        ]
                                    )
                                ),
                                "selected_quality": float(
                                    np.mean(
                                        [float(row["selected_quality"]) for row in rows]
                                    )
                                ),
                                "useful_selection_rate": float(
                                    np.mean(
                                        [
                                            float(row["useful_selection_rate"])
                                            for row in rows
                                        ]
                                    )
                                ),
                                "decoy_selection_rate": float(
                                    np.mean(
                                        [
                                            float(row["decoy_selection_rate"])
                                            for row in rows
                                        ]
                                    )
                                ),
                            }
                        )
            best = max(
                combinations,
                key=lambda row: (
                    row["validation_objective"],
                    row["selected_quality"],
                    -row["decoy_selection_rate"],
                    -row["lambda_novelty"],
                ),
            )
            selected[(generator_kind, feature_mode)] = {
                key: float(best[key])
                for key in ["lambda_novelty", "quality_threshold", "quality_gate_beta"]
            }
    write_csv(output / "router_hyperparameter_trials.csv", trial_rows)
    write_csv(output / "generator_integrity_sweep.csv", integrity_rows)
    payload = {
        "selection_data": "held-out training-family validation landscapes",
        "test_set_used": False,
        "selection_rule": "selected_quality + 0.25*useful_selection - 0.25*decoy_selection; ties: quality, lower decoy, lower lambda",
        "normalization": config.novelty_normalization,
        "selected": {
            f"{generator}_{mode}": values
            for (generator, mode), values in selected.items()
        },
        "config": config.to_dict(),
        "runtime": runtime.to_dict(),
    }
    (output / "ROUTER_HYPERPARAMETER_SELECTION.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return selected


def load_router_selection(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[tuple[str, str], dict[str, float]] = {}
    for key, values in payload["selected"].items():
        generator, mode = key.split("_", maxsplit=1)
        output[(generator, mode)] = {
            name: float(value) for name, value in values.items()
        }
    return output


def _time_router(
    router: LinearRouter,
    batch: FrozenCandidateBatch,
    config: RouterExperimentConfig,
) -> tuple[float, float]:
    for _ in range(3):
        router.scores(batch, config)
    synchronize(batch.candidates.device)
    started = perf_counter()
    for _ in range(config.inference_repetitions):
        router.scores(batch, config)
    synchronize(batch.candidates.device)
    elapsed = (perf_counter() - started) / config.inference_repetitions
    return elapsed, batch.batch_size * batch.candidate_count / elapsed


def _save_router(
    path: Path, router: LinearRouter, metadata: Mapping[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coefficients=router.coefficients.detach().cpu().to(torch.float64).numpy(),
        feature_mode=router.feature_mode,
        target_mode=router.target_mode,
        ridge=router.ridge,
        **metadata,
    )


def _router_bundle(
    *,
    generator: FrozenGenerator,
    train_batches: Sequence[FrozenCandidateBatch],
    selected: dict[tuple[str, str], dict[str, float]],
    config: RouterExperimentConfig,
) -> dict[str, LinearRouter]:
    local = selected[(generator.kind, "local")]
    set_aware = selected[(generator.kind, "set_aware")]
    return {
        "quality_refit": fit_regression_router(
            train_batches,
            feature_mode="local",
            target_mode="quality",
            lambda_novelty=0.0,
            quality_threshold=local["quality_threshold"],
            quality_gate_beta=local["quality_gate_beta"],
            config=config,
        ),
        "novelty_only_router": fit_regression_router(
            train_batches,
            feature_mode="local",
            target_mode="novelty_only",
            lambda_novelty=1.0,
            quality_threshold=local["quality_threshold"],
            quality_gate_beta=local["quality_gate_beta"],
            config=config,
        ),
        "ungated_q_plus_n_router": fit_regression_router(
            train_batches,
            feature_mode="local",
            target_mode="ungated",
            lambda_novelty=local["lambda_novelty"],
            quality_threshold=local["quality_threshold"],
            quality_gate_beta=local["quality_gate_beta"],
            config=config,
        ),
        "gated_useful_novelty_router": fit_regression_router(
            train_batches,
            feature_mode="local",
            target_mode="gated",
            lambda_novelty=local["lambda_novelty"],
            quality_threshold=local["quality_threshold"],
            quality_gate_beta=local["quality_gate_beta"],
            config=config,
        ),
        "set_aware_useful_novelty_router": fit_regression_router(
            train_batches,
            feature_mode="set_aware",
            target_mode="gated",
            lambda_novelty=set_aware["lambda_novelty"],
            quality_threshold=set_aware["quality_threshold"],
            quality_gate_beta=set_aware["quality_gate_beta"],
            config=config,
        ),
        "gated_useful_novelty_pairwise_router": fit_pairwise_router(
            train_batches,
            feature_mode="local",
            lambda_novelty=local["lambda_novelty"],
            quality_threshold=local["quality_threshold"],
            quality_gate_beta=local["quality_gate_beta"],
            config=config,
        ),
    }


def _selector_scores(
    selector: str,
    *,
    generator: FrozenGenerator,
    routers: Mapping[str, LinearRouter],
    batch: FrozenCandidateBatch,
    random_seed: int,
    config: RouterExperimentConfig,
    useful_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor | None, LinearRouter | None]:
    if selector == "random":
        scores, indices = random_selector_scores(batch, random_seed)
        return scores, indices, None
    if selector == "quality_router":
        return (
            generator.quality_router.predict(
                critic_features(batch.starts, batch.candidates, batch.goals)
            ),
            None,
            None,
        )
    if selector == "quality_oracle":
        return batch.quality, None, None
    if selector == "useful_oracle":
        return useful_oracle_scores(batch, useful_threshold), None, None
    router = routers[selector]
    return router.scores(batch, config), None, router


def _evaluate_all(
    *,
    seed: int,
    generator: FrozenGenerator,
    routers: Mapping[str, LinearRouter],
    batch: FrozenCandidateBatch,
    selectors: Sequence[str],
    random_seed: int,
    selected_params: Mapping[str, float],
    config: RouterExperimentConfig,
    useful_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    reference_target, _ = routing_target(
        batch,
        target_mode="gated",
        lambda_novelty=selected_params["lambda_novelty"],
        quality_threshold=selected_params["quality_threshold"],
        quality_gate_beta=selected_params["quality_gate_beta"],
    )
    summary_rows, diagnostic_rows = [], []
    for selector in selectors:
        scores, indices, router = _selector_scores(
            selector,
            generator=generator,
            routers=routers,
            batch=batch,
            random_seed=random_seed,
            config=config,
            useful_threshold=useful_threshold,
        )
        evaluation = evaluate_selector(
            generator=generator.kind,
            selector=selector,
            batch=batch,
            scores=scores,
            useful_novelty_target=reference_target,
            useful_threshold=useful_threshold,
            selected_indices=indices,
        )
        if router is not None:
            inference_seconds, throughput = _time_router(router, batch, config)
            training_seconds = router.training_seconds
            parameter_count = router.parameter_count
            peak = router.peak_allocated_bytes
        else:
            inference_seconds, throughput = 0.0, 0.0
            training_seconds = 0.0
            parameter_count = (
                generator.quality_router.parameter_count
                if selector == "quality_router"
                else 0
            )
            peak = 0
        summary_rows.append(
            {
                "seed": seed,
                "generator": generator.kind,
                "selector": selector,
                "lambda_novelty": selected_params["lambda_novelty"],
                "quality_threshold": selected_params["quality_threshold"],
                "quality_gate_beta": selected_params["quality_gate_beta"],
                "parameter_count": parameter_count,
                "additional_parameters_vs_quality_router": parameter_count
                - generator.quality_router.parameter_count,
                "router_training_seconds": training_seconds,
                "router_inference_seconds": inference_seconds,
                "candidate_scoring_throughput": throughput,
                "peak_gpu_allocated_bytes": peak,
                **evaluation.summary,
            }
        )
        diagnostic_rows.extend({"seed": seed, **row} for row in evaluation.set_rows)
    return summary_rows, diagnostic_rows


def run_router_main(
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    config: RouterExperimentConfig,
    *,
    seeds: Sequence[int],
    selected: dict[tuple[str, str], dict[str, float]],
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "candidate_cache"
    model_dir = output / "routers"
    family = V2LandscapeFamily(base.base.landscape)
    primary_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    diagnostic_summary_rows: list[dict[str, object]] = []
    integrity_rows: list[dict[str, object]] = []
    coefficient_checks: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    generalization_rows: list[dict[str, object]] = []

    for seed_index, seed in enumerate(seeds):
        print(f"[router main {seed_index + 1}/{len(seeds)}] seed {seed}", flush=True)
        train, train_starts, _, _, test, test_starts = _datasets(family, base, rl, seed)
        for generator_kind in GENERATOR_KINDS:
            generator = load_frozen_generator(
                generator_kind, base=base, rl=rl, seed=seed, device=device
            )
            stream = _candidate_seeds(seed, generator_kind)
            learned, broad = _make_training_batches(
                generator,
                train_tasks=train,
                train_starts=train_starts,
                seeds=stream,
                base=base,
                config=config,
                device=device,
            )
            test_batch = generate_frozen_candidates(
                generator,
                tasks=test,
                starts=test_starts,
                action_seed=stream["test_action"],
                base=base,
                router_config=config,
                device=device,
                source=f"{generator_kind}_test",
            )
            for split, candidate_batch in [
                ("train_generated", learned),
                ("train_broad", broad),
                ("test", test_batch),
            ]:
                cache_path = cache_dir / f"seed_{seed}_{generator_kind}_{split}.pt"
                candidate_hash = save_candidate_batch(cache_path, candidate_batch)
                cache_rows.append(
                    {
                        "seed": seed,
                        "generator": generator_kind,
                        "split": split,
                        "path": str(cache_path),
                        "candidate_hash": candidate_hash,
                        "candidate_sets": candidate_batch.batch_size,
                        "candidates_per_set": candidate_batch.candidate_count,
                    }
                )
            routers = _router_bundle(
                generator=generator,
                train_batches=[learned, broad],
                selected=selected,
                config=config,
            )
            quality_refit = routers["quality_refit"]
            coefficient_difference = float(
                torch.max(
                    torch.abs(
                        quality_refit.coefficients
                        - generator.quality_router.coefficients
                    )
                ).cpu()
            )
            existing_scores = generator.quality_router.predict(
                critic_features(
                    test_batch.starts, test_batch.candidates, test_batch.goals
                )
            )
            refit_scores = quality_refit.scores(test_batch, config)
            coefficient_checks.append(
                {
                    "seed": seed,
                    "generator": generator_kind,
                    "maximum_coefficient_difference": coefficient_difference,
                    "maximum_test_score_difference": float(
                        torch.max(torch.abs(existing_scores - refit_scores)).cpu()
                    ),
                    "mean_test_score_difference": float(
                        torch.mean(torch.abs(existing_scores - refit_scores)).cpu()
                    ),
                    "test_top1_agreement": float(
                        torch.mean(
                            (
                                torch.argmax(existing_scores, dim=1)
                                == torch.argmax(refit_scores, dim=1)
                            ).float()
                        ).cpu()
                    ),
                }
            )
            for name, router in routers.items():
                _save_router(
                    model_dir / f"seed_{seed}_{generator_kind}_{name}.npz",
                    router,
                    {"seed": seed, "generator": generator_kind},
                )
            local = selected[(generator_kind, "local")]
            target_rows.append(
                {
                    "seed": seed,
                    "generator": generator_kind,
                    **target_distribution(
                        [learned, broad],
                        lambda_novelty=local["lambda_novelty"],
                        quality_threshold=local["quality_threshold"],
                        quality_gate_beta=local["quality_gate_beta"],
                    ),
                }
            )
            seed_primary, seed_diagnostics = _evaluate_all(
                seed=seed,
                generator=generator,
                routers=routers,
                batch=test_batch,
                selectors=PRIMARY_SELECTORS,
                random_seed=stream["random_test"],
                selected_params=local,
                config=config,
                useful_threshold=base.base.landscape.useful_threshold,
            )
            primary_rows.extend(seed_primary)
            diagnostic_rows.extend(seed_diagnostics)
            for selector in PRIMARY_SELECTORS:
                selected_rows = [
                    row for row in seed_diagnostics if row["selector"] == selector
                ]
                diagnostic_summary_rows.append(
                    {
                        "seed": seed,
                        "generator": generator_kind,
                        "selector": selector,
                        **conservative_diagnostic_summary(selected_rows),
                    }
                )

            for variant_index, variant in enumerate(config.generalization_variants):
                variant_tasks = family.sample_tasks(
                    geometries=base.base.test_geometries,
                    seed=seed + 30_000 + variant_index,
                    variant=variant,
                    all_goals=True,
                )
                variant_starts = _make_starts(
                    variant_tasks,
                    trials=base.base.trials_per_task,
                    seed=seed + 30_100 + variant_index,
                )
                action_seed = (
                    seed + 30_500 + 20 * variant_index + 1
                    if generator_kind == "rl"
                    else seed + 30_500 + 20 * variant_index
                )
                variant_batch = generate_frozen_candidates(
                    generator,
                    tasks=variant_tasks,
                    starts=variant_starts,
                    action_seed=action_seed,
                    base=base,
                    router_config=config,
                    device=device,
                    source=f"{generator_kind}_{variant}",
                )
                cache_path = (
                    cache_dir
                    / f"seed_{seed}_{generator_kind}_generalization_{variant}.pt"
                )
                candidate_hash = save_candidate_batch(cache_path, variant_batch)
                cache_rows.append(
                    {
                        "seed": seed,
                        "generator": generator_kind,
                        "split": f"generalization_{variant}",
                        "path": str(cache_path),
                        "candidate_hash": candidate_hash,
                        "candidate_sets": variant_batch.batch_size,
                        "candidates_per_set": variant_batch.candidate_count,
                    }
                )
                variant_rows, _ = _evaluate_all(
                    seed=seed,
                    generator=generator,
                    routers=routers,
                    batch=variant_batch,
                    selectors=GENERALIZATION_SELECTORS,
                    random_seed=seed + 30_700 + 20 * variant_index,
                    selected_params=local,
                    config=config,
                    useful_threshold=base.base.landscape.useful_threshold,
                )
                generalization_rows.extend(
                    {"landscape_variant": variant, **row} for row in variant_rows
                )
            integrity_rows.append(
                {
                    "seed": seed,
                    "stage": "main_after_all_router_work",
                    **generator_integrity(generator),
                }
            )

    write_csv(output / "primary_per_seed.csv", primary_rows)
    write_csv(output / "selection_diagnostics_per_set.csv", diagnostic_rows)
    write_csv(output / "selection_diagnostics_per_seed.csv", diagnostic_summary_rows)
    write_csv(output / "generator_integrity.csv", integrity_rows)
    write_csv(output / "quality_lambda0_reproduction.csv", coefficient_checks)
    write_csv(output / "routing_target_distributions.csv", target_rows)
    write_csv(output / "candidate_cache_manifest.csv", cache_rows)
    write_csv(output / "generalization_per_seed.csv", generalization_rows)
    payload = {
        "seeds": list(seeds),
        "runtime": runtime.to_dict(),
        "config": config.to_dict(),
        "selected_hyperparameters": {
            f"{generator}_{mode}": values
            for (generator, mode), values in selected.items()
        },
        "generator_retrained": False,
        "region_labels_used_for_deployable_router_training": False,
        "test_used_for_hyperparameter_selection": False,
    }
    (output / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
