"""Orchestration for the isolated d32 coordinate-symmetry diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from creativity_toy.analysis_utils import statistics
from creativity_toy.experiments.study_v2 import _make_starts
from creativity_toy.landscapes_d16 import geometry_diagnostics

from .config_landscape_representation import LandscapeRepresentationStudyConfig
from .config_symmetry import SymmetryStudyConfig
from .landscape_representation import (
    LandscapeAttentionPolicy,
    build_component_descriptors,
    descriptor_normalization,
)
from .runtime import RuntimeInfo
from .study import write_csv
from .study_capacity_diagnostic import _aligned
from .study_d16 import _datasets
from .study_failure_diagnostics import _dimension_config, _family
from .study_landscape_representation import PRIMARY_METRICS, _summary
from .symmetry import (
    EquivariantLandscapeAttention,
    SETransform,
    build_equivariant_components,
    invariant_descriptor_dimension,
    invariant_normalization,
    proper_orthogonal_matrices,
    random_translations,
    transform_current_descriptors,
    transform_points,
    transform_tasks,
    transform_vectors,
)
from .training_landscape_supervised import evaluate_landscape_policy
from .training_symmetry import set_equivariance_metrics, train_symmetry_policy

CONDITIONS = (
    "CurrentAttention",
    "CurrentAttentionAugmented",
    "EquivariantAttention",
    "EquivariantAttentionAugmented",
)
PAIRINGS = (
    ("CurrentAttentionAugmented", "CurrentAttention"),
    ("EquivariantAttention", "CurrentAttention"),
    ("EquivariantAttention", "CurrentAttentionAugmented"),
    ("EquivariantAttentionAugmented", "EquivariantAttention"),
)


def _benchmark(config: SymmetryStudyConfig):
    benchmark = _dimension_config(config.dimension, smoke=False)
    benchmark.base.base.train_geometries = config.training_geometries
    return benchmark


def _current_model(
    config: SymmetryStudyConfig,
    *,
    seed: int,
    device: torch.device,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> LandscapeAttentionPolicy:
    base = LandscapeRepresentationStudyConfig().representation
    return LandscapeAttentionPolicy(
        dimension=config.dimension,
        candidates=config.candidates,
        latent_dimension=config.latent_dimension,
        task_token_dimension=config.task_token_dimension,
        alpha_max=config.alpha_max,
        seed=seed + 10_020,
        device=device,
        descriptor_dimension=mean.numel(),
        component_hidden_dimension=base.component_hidden_dimension,
        component_token_dimension=base.component_token_dimension,
        descriptor_mean=mean,
        descriptor_scale=scale,
        proposal_hidden_dimension=base.proposal_hidden_dimension,
    )


def _equivariant_model(
    config: SymmetryStudyConfig,
    *,
    seed: int,
    device: torch.device,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> EquivariantLandscapeAttention:
    return EquivariantLandscapeAttention(
        dimension=config.dimension,
        candidates=config.candidates,
        latent_dimension=config.latent_dimension,
        task_token_dimension=config.task_token_dimension,
        alpha_max=config.alpha_max,
        seed=seed + 10_020,
        device=device,
        invariant_dimension=invariant_descriptor_dimension(config.dimension),
        component_hidden_dimension=config.invariant_component_hidden_dimension,
        component_token_dimension=config.invariant_component_token_dimension,
        invariant_mean=mean,
        invariant_scale=scale,
    )


def _save(path: Path, model, *, history, metadata) -> None:
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


def _assert_nonoracle_model(model) -> None:
    if bool(getattr(model, "uses_original_goal", True)):
        raise AssertionError("symmetry model must not consume the original goal g")
    forbidden = set(getattr(model, "forbidden_input_fields", ()))
    required_forbidden = {
        "goals",
        "useful_centers",
        "decoy_centers",
        "target_region",
    }
    if not required_forbidden.issubset(forbidden):
        raise AssertionError("non-oracle forbidden-input contract is incomplete")


def _load(path: Path, factory, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    model = factory()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, list(payload.get("history", [])), dict(payload.get("metadata", {}))


def _load_current_baseline(config, *, seed, device, mean, scale, baseline_root: Path):
    path = baseline_root / "models" / f"LandscapeAttention_seed_{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen CurrentAttention baseline: {path}")
    return _load(
        path,
        lambda: _current_model(
            config, seed=seed, device=device, mean=mean, scale=scale
        ),
        device,
    )


@torch.no_grad()
def _augmented_current_normalization(
    components,
    rotation_pool: torch.Tensor,
    *,
    seed: int,
    dimension: int,
    sample_count: int = 4_096,
):
    count = min(sample_count, components.batch_size)
    generator = torch.Generator(device=components.values.device).manual_seed(seed)
    example = torch.randperm(
        components.batch_size, generator=generator, device=components.values.device
    )[:count]
    rotation_index = torch.randint(
        len(rotation_pool),
        (count,),
        generator=generator,
        device=components.values.device,
    )
    transformed = transform_current_descriptors(
        components.index_select(example),
        rotation_pool.index_select(0, rotation_index),
        dimension=dimension,
    )
    return descriptor_normalization(transformed)


def _train_or_load(
    condition: str,
    *,
    config: SymmetryStudyConfig,
    seed: int,
    device: torch.device,
    path: Path,
    tasks,
    starts,
    current_components,
    equivariant_components,
    current_stats,
    current_aug_stats,
    equivariant_stats,
    rotation_pool,
):
    is_current = condition.startswith("Current")
    augmented = condition.endswith("Augmented")
    if is_current:
        mean, scale = current_aug_stats if augmented else current_stats
    else:
        mean, scale = equivariant_stats
    factory = (
        (
            lambda: _current_model(
                config, seed=seed, device=device, mean=mean, scale=scale
            )
        )
        if is_current
        else (
            lambda: _equivariant_model(
                config, seed=seed, device=device, mean=mean, scale=scale
            )
        )
    )
    if path.exists():
        loaded = _load(path, factory, device)
        _assert_nonoracle_model(loaded[0])
        return loaded
    model = factory()
    _assert_nonoracle_model(model)
    components = current_components if is_current else equivariant_components
    result = train_symmetry_policy(
        model,
        tasks,
        starts,
        components,
        component_kind="current" if is_current else "equivariant",
        augmented=augmented,
        rotation_pool=rotation_pool,
        translation_radius_max=config.augmentation.translation_radius_max,
        steps=config.supervised_steps,
        batch_size=min(config.batch_size, len(starts)),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
        lambda_alpha=config.lambda_alpha,
        seed=seed + 61_101,
        curve_steps=config.curve_steps,
    )
    metadata = {
        "condition": condition,
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": result.elapsed_seconds,
        "examples_seen": result.examples_seen,
        "effective_transform_draws": result.examples_seen if augmented else 0,
        "batch_size": min(config.batch_size, len(starts)),
        "steps": config.supervised_steps,
        "peak_allocated_bytes": result.peak_allocated_bytes,
        "peak_reserved_bytes": result.peak_reserved_bytes,
    }
    _save(path, model, history=result.history, metadata=metadata)
    model.eval()
    return model, result.history, metadata


def _components(condition: str, tasks, starts):
    return (
        build_component_descriptors(tasks, starts)
        if condition.startswith("Current")
        else build_equivariant_components(tasks, starts)
    )


@torch.no_grad()
def _evaluate(condition, model, tasks, starts, benchmark, config):
    components = _components(condition, tasks, starts)
    result = evaluate_landscape_policy(
        model,
        tasks,
        starts,
        components,
        lambda_alpha=config.lambda_alpha,
        refinement_steps=benchmark.base.base.training.refinement_steps,
        refinement_rate=benchmark.base.base.training.refinement_rate,
        useful_threshold=benchmark.base.base.landscape.useful_threshold,
        collect_per_set=False,
        collect_attention_rows=False,
    )
    return result.summary


@torch.no_grad()
def _propose(condition, model, tasks, starts):
    return model(starts, _components(condition, tasks, starts))


def _exact_transforms(config, starts, *, seed: int):
    generator = torch.Generator(device=starts.device).manual_seed(seed)
    count, dimension = len(starts), starts.shape[-1]
    rotation = proper_orthogonal_matrices(
        count, dimension, generator=generator, device=starts.device, dtype=starts.dtype
    )
    translation = random_translations(
        count,
        dimension,
        config.exact_translation_radius,
        generator=generator,
        device=starts.device,
        dtype=starts.dtype,
    )
    translation = (
        torch.nn.functional.normalize(translation, dim=-1, eps=1e-8)
        * config.exact_translation_radius
    )
    identity = torch.eye(dimension, dtype=starts.dtype, device=starts.device)[
        None
    ].expand(count, -1, -1)
    zero = torch.zeros_like(translation)
    return {
        "exact_translated": SETransform(identity, translation),
        "exact_rotated": SETransform(rotation, zero),
        "exact_rotation_translation": SETransform(rotation, translation),
    }


def _paired(rows, comparisons):
    lookup = {(int(row["seed"]), row["condition"], row["split"]): row for row in rows}
    output = []
    for left, right in comparisons:
        for split in sorted({row["split"] for row in rows}):
            available = sorted({int(row["seed"]) for row in rows})
            for metric in PRIMARY_METRICS:
                values = []
                for seed in available:
                    a, b = (
                        lookup.get((seed, left, split)),
                        lookup.get((seed, right, split)),
                    )
                    if a is not None and b is not None:
                        values.append(float(a[metric]) - float(b[metric]))
                if not values:
                    continue
                value = statistics(values)
                output.append(
                    {
                        "condition_a": left,
                        "condition_b": right,
                        "split": split,
                        "metric": metric,
                        "mean_difference": value["mean"],
                        "sd_difference": value["sd"],
                        "ci95_difference": value["ci95"],
                        "n": len(values),
                    }
                )
    return output


@torch.no_grad()
def _audit_current_seed(
    model,
    tasks,
    starts,
    transforms,
    benchmark,
    *,
    seed: int,
):
    dimension = starts.shape[-1]
    base_components = build_component_descriptors(tasks, starts)
    base = model(starts, base_components)
    translation = transforms["exact_translated"]
    translated_tasks = transform_tasks(tasks, translation)
    translated_starts = transform_points(
        starts, translation.rotation, translation.translation
    )
    translated_components = build_component_descriptors(
        translated_tasks, translated_starts
    )
    translated = model(translated_starts, translated_components)
    neutralized = model(starts, translated_components)
    translation_cosine = torch.sum(base.direction * translated.direction, dim=-1)
    neutralized_cosine = torch.sum(base.direction * neutralized.direction, dim=-1)
    relative_error = torch.max(
        torch.abs(
            base_components.values[..., :dimension]
            - translated_components.values[..., :dimension]
        )
    )
    covariance_error = torch.max(
        torch.abs(
            base_components.values[..., dimension + 1 :]
            - translated_components.values[..., dimension + 1 :]
        )
    )
    normalized_base = (
        base_components.values - model.descriptor_mean
    ) / model.descriptor_scale
    normalized_translated = (
        translated_components.values - model.descriptor_mean
    ) / model.descriptor_scale

    rotated_transform = transforms["exact_rotation_translation"]
    rotated_tasks = transform_tasks(tasks, rotated_transform)
    rotated_starts = transform_points(
        starts, rotated_transform.rotation, rotated_transform.translation
    )
    rotated_components = build_component_descriptors(rotated_tasks, rotated_starts)
    relative_expected = transform_vectors(
        base_components.values[..., :dimension], rotated_transform.rotation
    )
    relative_rotation_error = torch.max(
        torch.abs(relative_expected - rotated_components.values[..., :dimension])
    )

    base_refined = tasks.refine(
        base.candidate,
        steps=benchmark.base.base.training.refinement_steps,
        rate=benchmark.base.base.training.refinement_rate,
    )
    transformed_candidate = transform_points(
        base.candidate, rotated_transform.rotation, rotated_transform.translation
    )
    transformed_refined = rotated_tasks.refine(
        transformed_candidate,
        steps=benchmark.base.base.training.refinement_steps,
        rate=benchmark.base.base.training.refinement_rate,
    )
    expected_refined = transform_points(
        base_refined, rotated_transform.rotation, rotated_transform.translation
    )
    refinement_error = torch.linalg.vector_norm(
        transformed_refined - expected_refined, dim=-1
    )
    quality_error = torch.abs(
        tasks.quality(base_refined) - rotated_tasks.quality(transformed_refined)
    )
    region_agreement = tasks.regions(base_refined) == rotated_tasks.regions(
        transformed_refined
    )
    return {
        "seed": seed,
        "relative_center_translation_max_error": float(relative_error.cpu()),
        "covariance_translation_max_error": float(covariance_error.cpu()),
        "normalized_descriptor_translation_max_error": float(
            torch.max(torch.abs(normalized_base - normalized_translated)).cpu()
        ),
        "absolute_h_translation_mean": float(
            torch.mean(
                torch.linalg.vector_norm(translated_starts - starts, dim=-1)
            ).cpu()
        ),
        "current_translation_direction_cosine": float(
            torch.mean(translation_cosine).cpu()
        ),
        "current_translation_alpha_error": float(
            torch.mean(torch.abs(translated.alpha - base.alpha)).cpu()
        ),
        "h_neutralized_translation_direction_cosine": float(
            torch.mean(neutralized_cosine).cpu()
        ),
        "h_neutralized_translation_alpha_error": float(
            torch.mean(torch.abs(neutralized.alpha - base.alpha)).cpu()
        ),
        "relative_center_rotation_law_max_error": float(relative_rotation_error.cpu()),
        "raw_normalized_descriptor_rotation_mean_change": float(
            torch.mean(
                torch.abs(
                    (rotated_components.values - model.descriptor_mean)
                    / model.descriptor_scale
                    - normalized_base
                )
            ).cpu()
        ),
        "refinement_equivariance_mean_error": float(torch.mean(refinement_error).cpu()),
        "refinement_equivariance_max_error": float(torch.max(refinement_error).cpu()),
        "quality_invariance_max_error": float(torch.max(quality_error).cpu()),
        "region_classification_agreement": float(
            torch.mean(region_agreement.float()).cpu()
        ),
    }


def _condition_tasks(family, benchmark, *, seed: int, variant: str):
    base = benchmark.base.base
    tasks = family.sample_tasks(
        geometries=base.test_geometries,
        seed=seed,
        variant=variant,
        all_goals=True,
    )
    starts = _make_starts(tasks, trials=base.trials_per_task, seed=seed + 1)
    return _aligned(
        tasks,
        starts,
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )


def run_symmetry_study(
    config: SymmetryStudyConfig,
    *,
    results_root: Path,
    device: torch.device,
    runtime: RuntimeInfo,
):
    prefix = f"d{config.dimension}"
    audit_dir = results_root / f"{prefix}_symmetry_audit"
    augmentation_dir = results_root / f"{prefix}_symmetry_augmentation"
    equivariant_dir = results_root / f"{prefix}_equivariant_attention"
    for path in (audit_dir, augmentation_dir, equivariant_dir):
        path.mkdir(parents=True, exist_ok=True)
    baseline_root = Path("results/d32_landscape_representation")
    reuse_frozen_baseline = config.dimension == 32
    manifest = {
        "config": config.to_dict(),
        "runtime": runtime.to_dict(),
        "constraints": {
            "dimension": config.dimension,
            "candidates": config.candidates,
            "training_geometries": config.training_geometries,
            "d64_run": config.dimension == 64,
            "prior_outputs_overwritten": False,
            "baseline_retrained": not reuse_frozen_baseline,
            "test_tuning": False,
            "uses_original_goal_g": False,
            "supervision": "privileged useful-center directions/radii in Hungarian loss only",
        },
    }
    (audit_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    input_contract = {
        "dimension": config.dimension,
        "uses_original_goal_g": False,
        "task_representation": "learned constant task token",
        "model_inputs": [
            "current state h",
            "constant task token",
            "fixed candidate identity z_i",
            "unlabeled component center relative to h",
            "component amplitude",
            "component covariance/precision-derived geometry",
        ],
        "forbidden_model_inputs": [
            "tasks.goals (the old target-direction g)",
            "useful-region labels or useful centers",
            "decoy labels or decoy centers",
            "target labels/center/direction/radius",
            "region identity",
            "refinement outcome",
        ],
        "privileged_supervision_only": [
            "normalized directions from h to useful centers",
            "radial distances from h to useful centers",
            "Hungarian permutation-invariant matching",
        ],
        "loss": "mean(1-cosine) + lambda_alpha * mean(alpha_error^2)",
        "supervision_is_oracle_like": True,
        "labels_are_inputs": False,
        "validated_by": (
            "test_d64_model_input_excludes_goal_and_labels_but_loss_remains_oracle_like"
        ),
    }
    (audit_dir / "nonoracle_input_audit.json").write_text(
        json.dumps(input_contract, indent=2), encoding="utf-8"
    )
    implementation_notes = {
        "dimension": config.dimension,
        "covariance_eigenvalue_batch_size": 256 if config.dimension >= 64 else None,
        "reason": (
            "avoid excessive torch.linalg.eigvalsh CUDA workspace at d64; "
            "components are independent across this axis"
        ),
        "scientific_semantics_changed": False,
        "examples_or_updates_changed": False,
        "numerical_equivalence_test": (
            "test_d64_equivariant_rule_is_exact_and_parameter_matched"
        ),
    }
    (audit_dir / "implementation_notes.json").write_text(
        json.dumps(implementation_notes, indent=2), encoding="utf-8"
    )
    augmentation_config = {
        **config.augmentation.__dict__,
        "rotation_generation": "Gaussian QR, positive-diagonal Haar correction, determinant +1 correction",
        "translation_generation": (
            f"uniform radius in [0,{config.augmentation.translation_radius_max:g}], "
            "isotropic unit direction"
        ),
        "effective_transform_draws_per_augmented_seed": config.supervised_steps
        * min(
            config.batch_size,
            config.training_geometries * 3 * config.starts_per_task,
        ),
        "fixed_rotation_per_geometry": False,
        "test_set_used_to_select_distribution": False,
    }
    (augmentation_dir / "augmentation_config.json").write_text(
        json.dumps(augmentation_config, indent=2), encoding="utf-8"
    )

    rows, equivariance_rows, curve_rows = [], [], []
    metadata_rows, audit_rows = [], []
    for seed in config.seeds:
        benchmark = _benchmark(config)
        family = _family(benchmark)
        train, _, _, _, test, test_starts = _datasets(family, benchmark, seed)
        geometry_audit_path = audit_dir / "geometry_audit.json"
        if not geometry_audit_path.exists():
            geometry_audit = geometry_diagnostics(train)
            if config.dimension >= 16 and not bool(
                geometry_audit["not_effectively_2d"]
            ):
                raise AssertionError(
                    "high-dimensional symmetry geometry is effectively low-rank"
                )
            geometry_audit_path.write_text(
                json.dumps(geometry_audit, indent=2), encoding="utf-8"
            )
        train_nested = _make_starts(
            train, trials=config.starts_per_task, seed=seed + 41_001
        )
        packed_train, h_train = _aligned(train, train_nested, device)
        current_components = build_component_descriptors(packed_train, h_train)
        equivariant_components = build_equivariant_components(packed_train, h_train)
        current_stats = descriptor_normalization(current_components)
        equivariant_stats = invariant_normalization(equivariant_components)
        rotation_generator = torch.Generator(device=device).manual_seed(seed + 51_001)
        rotation_pool = proper_orthogonal_matrices(
            config.augmentation.rotation_pool_size,
            config.dimension,
            generator=rotation_generator,
            device=device,
            dtype=torch.float32,
        )
        current_aug_stats = _augmented_current_normalization(
            current_components,
            rotation_pool,
            seed=seed + 51_002,
            dimension=config.dimension,
        )

        if reuse_frozen_baseline:
            current, baseline_history, baseline_metadata = _load_current_baseline(
                config,
                seed=seed,
                device=device,
                mean=current_stats[0],
                scale=current_stats[1],
                baseline_root=baseline_root,
            )
            baseline_source = str(
                baseline_root / "models" / f"LandscapeAttention_seed_{seed}.pt"
            )
        else:
            current, baseline_history, baseline_metadata = _train_or_load(
                "CurrentAttention",
                config=config,
                seed=seed,
                device=device,
                path=augmentation_dir / "models" / f"CurrentAttention_seed_{seed}.pt",
                tasks=packed_train,
                starts=h_train,
                current_components=current_components,
                equivariant_components=equivariant_components,
                current_stats=current_stats,
                current_aug_stats=current_aug_stats,
                equivariant_stats=equivariant_stats,
                rotation_pool=rotation_pool,
            )
            baseline_source = str(
                augmentation_dir / "models" / f"CurrentAttention_seed_{seed}.pt"
            )
        models = {"CurrentAttention": current}
        _assert_nonoracle_model(current)
        baseline_metadata = {
            **baseline_metadata,
            "condition": "CurrentAttention",
            "checkpoint_reused": reuse_frozen_baseline,
            "source": baseline_source,
        }
        metadata_rows.append(baseline_metadata)
        curve_rows.extend(
            {"seed": seed, "condition": "CurrentAttention", **point}
            for point in baseline_history
        )

        # Auditing the frozen representation before training a symmetry intervention
        # keeps the diagnostic measurements independent of that intervention.
        packed_test, h_test = _aligned(test, test_starts, device)
        transforms = _exact_transforms(config, h_test, seed=seed + 81_001)
        audit_rows.append(
            _audit_current_seed(
                current, packed_test, h_test, transforms, benchmark, seed=seed
            )
        )

        specifications = [
            (
                "CurrentAttentionAugmented",
                augmentation_dir
                / "models"
                / f"CurrentAttentionAugmented_seed_{seed}.pt",
            ),
            (
                "EquivariantAttention",
                equivariant_dir / "models" / f"EquivariantAttention_seed_{seed}.pt",
            ),
            (
                "EquivariantAttentionAugmented",
                equivariant_dir
                / "models"
                / f"EquivariantAttentionAugmented_seed_{seed}.pt",
            ),
        ]
        for condition, path in specifications:
            model, history, metadata = _train_or_load(
                condition,
                config=config,
                seed=seed,
                device=device,
                path=path,
                tasks=packed_train,
                starts=h_train,
                current_components=current_components,
                equivariant_components=equivariant_components,
                current_stats=current_stats,
                current_aug_stats=current_aug_stats,
                equivariant_stats=equivariant_stats,
                rotation_pool=rotation_pool,
            )
            models[condition] = model
            metadata_rows.append(metadata)
            curve_rows.extend(
                {"seed": seed, "condition": condition, **point} for point in history
            )

        eval_sets = {"iid": (packed_test, h_test)}
        for split, transform in transforms.items():
            eval_sets[split] = (
                transform_tasks(packed_test, transform),
                transform_points(h_test, transform.rotation, transform.translation),
            )
        secondary = {
            "width_shift": "width_shift",
            "more_peaks": "more_peaks",
            "more_decoys": "more_decoys",
            "new_goals": "new_goals",
            "unseen": "unseen",
            "family_translated": "translated",
            "family_rotated": "rotated",
        }
        for index, (split, variant) in enumerate(secondary.items()):
            tasks_numpy = family.sample_tasks(
                geometries=benchmark.base.base.test_geometries,
                seed=seed + 90_000 + 100 * index,
                variant=variant,
                all_goals=True,
            )
            nested = _make_starts(
                tasks_numpy,
                trials=benchmark.base.base.trials_per_task,
                seed=seed + 90_001 + 100 * index,
            )
            eval_sets[split] = _aligned(tasks_numpy, nested, device)

        for split, (tasks_eval, starts_eval) in eval_sets.items():
            for condition, model in models.items():
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "split": split,
                        **_evaluate(
                            condition, model, tasks_eval, starts_eval, benchmark, config
                        ),
                    }
                )

        base_proposals = {
            condition: _propose(condition, model, packed_test, h_test)
            for condition, model in models.items()
        }
        for transform_name, transform in transforms.items():
            transformed_tasks, transformed_starts = eval_sets[transform_name]
            for condition, model in models.items():
                transformed_proposal = _propose(
                    condition, model, transformed_tasks, transformed_starts
                )
                equivariance_rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "transform": transform_name,
                        **set_equivariance_metrics(
                            base_proposals[condition],
                            transformed_proposal,
                            transform.rotation,
                        ),
                    }
                )
    summary = _summary(rows, ("condition", "split"), PRIMARY_METRICS)
    paired = _paired(rows, PAIRINGS)
    equivariance_summary = _summary(
        equivariance_rows,
        ("condition", "transform"),
        (
            "equivariance_cosine",
            "equivariance_angular_error_degrees",
            "alpha_invariance_error",
            "hungarian_set_alpha_error",
            "direct_slot_equivariance_cosine",
        ),
    )
    audit_summary = _summary(
        audit_rows, tuple(), tuple(key for key in audit_rows[0] if key != "seed")
    )
    parameter_counts = {
        condition: {
            "parameters": int(
                round(
                    np.mean(
                        [
                            float(row["parameter_count"])
                            for row in metadata_rows
                            if row["condition"] == condition
                        ]
                    )
                )
            ),
            "mean_training_seconds": float(
                np.mean(
                    [
                        float(row.get("training_seconds", 0.0))
                        for row in metadata_rows
                        if row["condition"] == condition
                    ]
                )
            ),
            "checkpoint_reused": all(
                bool(row.get("checkpoint_reused", False))
                for row in metadata_rows
                if row["condition"] == condition
            ),
        }
        for condition in CONDITIONS
    }
    write_csv(augmentation_dir / "per_seed.csv", rows)
    write_csv(augmentation_dir / "summary.csv", summary)
    write_csv(augmentation_dir / "paired_summary.csv", paired)
    write_csv(augmentation_dir / "training_curves.csv", curve_rows)
    write_csv(audit_dir / "equivariance_metrics.csv", equivariance_rows)
    write_csv(audit_dir / "equivariance_summary.csv", equivariance_summary)
    write_csv(audit_dir / "symmetry_audit_per_seed.csv", audit_rows)
    write_csv(audit_dir / "symmetry_audit_summary.csv", audit_summary)
    (equivariant_dir / "parameter_counts.json").write_text(
        json.dumps(parameter_counts, indent=2), encoding="utf-8"
    )
    symmetry_audit = {
        "feature_laws": {
            "absolute_h": "translation and rotation equivariant vector; current scalar MLP treats coordinates arbitrarily",
            "relative_center_mu_minus_h": "translation invariant, rotation equivariant vector",
            "amplitude": "translation/rotation invariant scalar",
            "raw_covariance_lower_triangle": "translation invariant but coordinate-basis dependent under rotation",
            "task_token": "non-geometric invariant",
            "candidate_code": "non-geometric invariant",
            "component_order": "unordered; both architectures are permutation invariant",
            "normalization": "translation safe for relative descriptors; coordinate-specific under rotation",
            "refinement_quality_regions": "exactly transformed geometric subsystem; numerically equivariant",
        },
        "translation_failure_cause": "absolute h is concatenated into CurrentAttention query, direction MLP, and alpha head; descriptors themselves are translation invariant",
        "per_seed_summary": audit_summary,
        "dimension": config.dimension,
        "d64_run": config.dimension == 64,
    }
    (audit_dir / "symmetry_audit.json").write_text(
        json.dumps(symmetry_audit, indent=2), encoding="utf-8"
    )
    completion = {
        "seeds": len(config.seeds),
        "result_rows": len(rows),
        "equivariance_rows": len(equivariance_rows),
        "paired_rows": len(paired),
        "dimension": config.dimension,
        "d64_run": config.dimension == 64,
    }
    (audit_dir / "completion.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    return completion
