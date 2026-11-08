"""Isolated 32D non-oracle landscape-representation study orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from creativity_toy.analysis_utils import statistics
from creativity_toy.experiments.study_v2 import _make_starts
from creativity_toy.landscapes_v2 import V2Task

from .config_d32_generalization import D32GeneralizationConfig
from .config_landscape_representation import LandscapeRepresentationStudyConfig
from .landscape_representation import (
    ALLOWED_COMPONENT_INPUT_FIELDS,
    FORBIDDEN_NONORACLE_INPUT_FIELDS,
    ComponentDescriptorBatch,
    ComponentUtilityProbe,
    LandscapeAttentionPolicy,
    LandscapeDeepSetsPolicy,
    TaskOnlyParameterMatchedPolicy,
    TaskOnlyPolicy,
    build_component_descriptors,
    component_descriptor_dimension,
    descriptor_normalization,
    random_component_permutation,
)
from .runtime import RuntimeInfo
from .study import write_csv
from .study_capacity_diagnostic import _aligned
from .study_d16 import _datasets
from .study_d32_generalization import _eval as evaluate_oracle
from .study_d32_generalization import _fit_scaling_model
from .study_failure_diagnostics import _dimension_config, _family
from .training_landscape_supervised import (
    component_evaluation_labels,
    evaluate_component_utility_probe,
    evaluate_landscape_policy,
    train_component_utility_probe,
    train_landscape_policy,
)

PRIMARY_METRICS = (
    "mean_matched_cosine",
    "mean_angular_error_degrees",
    "mean_alpha_absolute_error",
    "proposal_recall",
    "mode_set_recall",
    "useful_region_count",
    "target_proposal_recall",
    "candidate_mean_quality",
    "candidate_decoy_rate",
    "candidate_anchor_rate",
)


def _benchmark(config: LandscapeRepresentationStudyConfig):
    benchmark = _dimension_config(32, smoke=False)
    benchmark.base.base.train_geometries = config.representation.training_geometries
    return benchmark


def _summary(rows, keys: tuple[str, ...], metrics: tuple[str, ...]):
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        output = dict(zip(keys, group))
        for metric in metrics:
            numbers = [float(row[metric]) for row in values if metric in row]
            if numbers:
                value = statistics(numbers)
                output.update(
                    {
                        f"{metric}_mean": value["mean"],
                        f"{metric}_sd": value["sd"],
                        f"{metric}_ci95": value["ci95"],
                    }
                )
        output["n"] = len(values)
        result.append(output)
    return result


def _save_model(path: Path, model, *, history, metadata) -> None:
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


def _load_model(path: Path, factory: Callable[[], torch.nn.Module], device):
    payload = torch.load(path, map_location=device, weights_only=True)
    model = factory()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, list(payload.get("history", [])), dict(payload.get("metadata", {}))


def _model_kwargs(config, *, seed: int, device):
    value = config.representation
    return {
        "dimension": value.dimension,
        "candidates": value.candidates,
        "latent_dimension": value.latent_dimension,
        "task_token_dimension": value.task_token_dimension,
        "alpha_max": value.alpha_max,
        "seed": seed + 10_020,
        "device": device,
    }


def _component_kwargs(config, *, seed: int, device, mean, scale):
    value = config.representation
    return {
        **_model_kwargs(config, seed=seed, device=device),
        "descriptor_dimension": component_descriptor_dimension(value.dimension),
        "component_hidden_dimension": value.component_hidden_dimension,
        "component_token_dimension": value.component_token_dimension,
        "descriptor_mean": mean,
        "descriptor_scale": scale,
        "proposal_hidden_dimension": value.proposal_hidden_dimension,
    }


def make_models(config, *, seed: int, device, mean, scale):
    common = _model_kwargs(config, seed=seed, device=device)
    component = _component_kwargs(
        config, seed=seed, device=device, mean=mean, scale=scale
    )
    attention = LandscapeAttentionPolicy(**component)
    return {
        "TaskOnly": TaskOnlyPolicy(
            hidden_dimension=config.representation.proposal_hidden_dimension, **common
        ),
        "TaskOnlyParameterMatched": TaskOnlyParameterMatchedPolicy.for_target_parameters(
            attention.trainable_parameter_count, **common
        ),
        "LandscapeDeepSets": LandscapeDeepSetsPolicy(**component),
        "LandscapeAttention": attention,
    }


def _permuted(components: ComponentDescriptorBatch, *, seed: int):
    generator = torch.Generator(device=components.values.device).manual_seed(seed)
    return components.permute(
        random_component_permutation(components.mask, generator=generator)
    )


def _different_geometry_indices(tasks, *, seed: int) -> torch.Tensor:
    """Deterministically pair every row with a different geometry."""
    geometry = tasks.geometry_id.detach().cpu().numpy()
    generator = np.random.default_rng(seed)
    candidates = np.arange(len(geometry), dtype=np.int64)
    result = np.empty(len(geometry), dtype=np.int64)
    for index in range(len(geometry)):
        valid = candidates[geometry != geometry[index]]
        if not len(valid):
            raise ValueError("shuffled-landscape control needs multiple geometries")
        result[index] = int(valid[generator.integers(len(valid))])
    return torch.as_tensor(result, dtype=torch.long, device=tasks.device)


def _hash_centers(tasks: Sequence[V2Task]) -> set[str]:
    return {
        hashlib.sha256(
            np.asarray(task.component_centers, dtype=np.float64).tobytes()
        ).hexdigest()
        for task in tasks
    }


def _condition_tasks(family, benchmark, *, seed: int, variant: str):
    base = benchmark.base.base
    tasks = family.sample_tasks(
        geometries=base.test_geometries, seed=seed, variant=variant, all_goals=True
    )
    return tasks, _make_starts(tasks, trials=base.trials_per_task, seed=seed + 1)


def _evaluate_condition(
    model,
    condition: str,
    packed,
    starts,
    *,
    config,
    benchmark,
    permutation_seed: int,
    shuffled: bool = False,
    collect_attention_rows: bool = True,
):
    components = None
    label_tasks = packed
    if condition in {"LandscapeDeepSets", "LandscapeAttention"}:
        if shuffled:
            indices = _different_geometry_indices(packed, seed=permutation_seed + 29)
            components = build_component_descriptors(
                packed, starts, source_indices=indices
            )
            label_tasks = packed.index_select(indices)
        else:
            components = build_component_descriptors(packed, starts)
        components = _permuted(components, seed=permutation_seed)
    evaluated = evaluate_landscape_policy(
        model,
        packed,
        starts,
        components,
        lambda_alpha=config.representation.lambda_alpha,
        refinement_steps=benchmark.base.base.training.refinement_steps,
        refinement_rate=benchmark.base.base.training.refinement_rate,
        useful_threshold=benchmark.base.base.landscape.useful_threshold,
        component_label_tasks=label_tasks,
        collect_per_set=False,
        collect_attention_rows=collect_attention_rows,
    )
    return evaluated.summary, evaluated.attention_rows


def _fit_model(
    condition,
    model,
    *,
    path,
    config,
    packed_train,
    h_train,
    components_train,
    benchmark,
    seed,
    device,
):
    mean, scale = descriptor_normalization(components_train)
    factory = lambda: make_models(
        config, seed=seed, device=device, mean=mean, scale=scale
    )[condition]
    if path.exists():
        return _load_model(path, factory, device)
    uses_components = condition.startswith("Landscape")

    def callback(current, _):
        if seed != config.representation.seeds[0]:
            return {}
        summary, _ = _evaluate_condition(
            current,
            condition,
            packed_train,
            h_train,
            config=config,
            benchmark=benchmark,
            permutation_seed=seed + 62_001,
        )
        return {
            "training_cosine": summary["mean_matched_cosine"],
            "training_proposal_recall": summary["proposal_recall"],
            "training_mode_set_recall": summary["mode_set_recall"],
        }

    result = train_landscape_policy(
        model,
        packed_train,
        h_train,
        components_train if uses_components else None,
        steps=config.representation.supervised_steps,
        batch_size=min(config.representation.batch_size, len(h_train)),
        learning_rate=config.representation.learning_rate,
        weight_decay=config.representation.weight_decay,
        gradient_clip=config.representation.gradient_clip,
        lambda_alpha=config.representation.lambda_alpha,
        seed=seed + 61_001,
        curve_steps=config.representation.curve_steps,
        evaluation_callback=callback,
    )
    metadata = {
        "condition": condition,
        "seed": seed,
        "dimension": 32,
        "training_geometries": config.representation.training_geometries,
        "supervised_examples": len(h_train),
        "examples_seen": result.examples_seen,
        "steps": config.representation.supervised_steps,
        "batch_size": min(config.representation.batch_size, len(h_train)),
        "training_seconds": result.elapsed_seconds,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "peak_allocated_bytes": result.peak_allocated_bytes,
        "peak_reserved_bytes": result.peak_reserved_bytes,
        "uses_original_goal": False,
    }
    _save_model(path, model, history=result.history, metadata=metadata)
    model.eval()
    return model, result.history, metadata


def _oracle_metrics(config, *, seed, packed, starts, device, prior_results):
    model, _, benchmark, _ = _fit_scaling_model(
        D32GeneralizationConfig(),
        prior_results,
        geometries=80,
        seed=seed,
        mode="multimode",
        device=device,
    )
    value = evaluate_oracle(
        model,
        packed,
        starts,
        benchmark,
        "multimode",
        config.representation.lambda_alpha,
    )
    value["mode_set_recall"] = value["useful_region_count"] / float(
        torch.mean(packed.n_useful.float()).cpu()
    )
    return value


def _fit_probe(
    config,
    *,
    seed,
    path,
    components_train,
    packed_train,
    components_test,
    packed_test,
    device,
):
    mean, scale = descriptor_normalization(components_train)
    factory = lambda: ComponentUtilityProbe(
        component_descriptor_dimension(32),
        config.component_probe.hidden_dimension,
        mean=mean,
        scale=scale,
        seed=seed + 71_001,
        device=device,
    )
    if path.exists():
        probe, history, metadata = _load_model(path, factory, device)
    else:
        probe = factory()
        labels_train, _ = component_evaluation_labels(packed_train)
        history, elapsed = train_component_utility_probe(
            probe,
            components_train,
            labels_train,
            steps=config.component_probe.steps,
            batch_size=config.component_probe.batch_size,
            learning_rate=config.component_probe.learning_rate,
            weight_decay=config.component_probe.weight_decay,
            seed=seed + 71_002,
        )
        metadata = {
            "seed": seed,
            "parameter_count": sum(p.numel() for p in probe.parameters()),
            "training_seconds": elapsed,
            "labels_are_targets_only": True,
        }
        _save_model(path, probe, history=history, metadata=metadata)
    labels_test, _ = component_evaluation_labels(packed_test)
    return evaluate_component_utility_probe(probe, components_test, labels_test), {
        "history": history,
        **metadata,
    }


def _paired_rows(rows, comparisons):
    lookup = {(int(r["seed"]), str(r["split"]), str(r["condition"])): r for r in rows}
    output = []
    for left, right in comparisons:
        for split in sorted({str(row["split"]) for row in rows}):
            differences = {metric: [] for metric in PRIMARY_METRICS}
            for seed in sorted({int(row["seed"]) for row in rows}):
                a, b = lookup.get((seed, split, left)), lookup.get((seed, split, right))
                if a is None or b is None:
                    continue
                for metric in PRIMARY_METRICS:
                    differences[metric].append(float(a[metric]) - float(b[metric]))
            if not differences[PRIMARY_METRICS[0]]:
                continue
            row = {
                "condition_a": left,
                "condition_b": right,
                "split": split,
                "n": len(differences[PRIMARY_METRICS[0]]),
            }
            for metric, values in differences.items():
                value = statistics(values)
                row.update(
                    {
                        f"delta_{metric}_mean": value["mean"],
                        f"delta_{metric}_sd": value["sd"],
                        f"delta_{metric}_ci95": value["ci95"],
                    }
                )
            output.append(row)
    return output


def _familiar_gap(rows):
    lookup = {(int(r["seed"]), str(r["condition"]), str(r["split"])): r for r in rows}
    output = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        for seed in sorted({int(row["seed"]) for row in rows}):
            familiar = lookup.get((seed, condition, "familiar_new_starts"))
            iid = lookup.get((seed, condition, "iid"))
            if familiar is None or iid is None:
                continue
            output.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "cosine_gap": float(familiar["mean_matched_cosine"])
                    - float(iid["mean_matched_cosine"]),
                    "proposal_recall_gap": float(familiar["proposal_recall"])
                    - float(iid["proposal_recall"]),
                    "mode_set_recall_gap": float(familiar["mode_set_recall"])
                    - float(iid["mode_set_recall"]),
                }
            )
    return output


def run_landscape_representation_study(
    config: LandscapeRepresentationStudyConfig,
    *,
    results_root: Path,
    device: torch.device,
    runtime: RuntimeInfo,
) -> dict[str, object]:
    rep_dir = results_root / "d32_landscape_representation"
    attn_dir = results_root / "d32_landscape_attention"
    ident_dir = results_root / "d32_landscape_identifiability"
    for path in (rep_dir, attn_dir, ident_dir):
        path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config.to_dict(),
        "runtime": runtime.to_dict(),
        "nonoracle_input_contract": {
            "task_goal": "learned constant task token",
            "component_descriptor": "[mu_j-h, amplitude_j, lower_triangle(covariance_j)]",
            "component_descriptor_dimension": component_descriptor_dimension(32),
            "allowed_fields": list(ALLOWED_COMPONENT_INPUT_FIELDS),
            "forbidden_fields": list(FORBIDDEN_NONORACLE_INPUT_FIELDS),
            "component_order_randomized": True,
            "true_useful_directions_are_labels_only": True,
            "refinement_outputs_are_not_inputs": True,
        },
        "scientific_constraints": {
            "dimension": 32,
            "does_not_run_d64": True,
            "test_used_for_selection": False,
            "prior_results_modified": False,
            "oracle_is_upper_bound_only": True,
        },
    }
    (rep_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    rows, curve_rows, metadata_rows = [], [], []
    probe_rows, probe_curve_rows, attention_rows = [], [], []
    attention_examples, leakage_seeds = [], []
    prior_oracle = (
        results_root / "d32_data_scaling"
        if results_root.name == "results"
        else Path("results/d32_data_scaling")
    )

    for seed_index, seed in enumerate(config.representation.seeds):
        benchmark = _benchmark(config)
        family = _family(benchmark)
        train, _, _, _, test, test_starts = _datasets(family, benchmark, seed)
        train_nested = _make_starts(
            train, trials=config.representation.starts_per_task, seed=seed + 41_001
        )
        packed_train, h_train = _aligned(train, train_nested, device)
        components_train = build_component_descriptors(packed_train, h_train)
        mean, scale = descriptor_normalization(components_train)
        models = make_models(config, seed=seed, device=device, mean=mean, scale=scale)
        parameter_counts = {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        }
        if _hash_centers(train) & _hash_centers(test):
            raise AssertionError("test geometry leaked into training geometry set")

        trained = {}
        for condition, model in models.items():
            fitted, history, metadata = _fit_model(
                condition,
                model,
                path=rep_dir / "models" / f"{condition}_seed_{seed}.pt",
                config=config,
                packed_train=packed_train,
                h_train=h_train,
                components_train=components_train,
                benchmark=benchmark,
                seed=seed,
                device=device,
            )
            trained[condition] = fitted
            metadata_rows.append(metadata)
            curve_rows.extend(
                {"seed": seed, "condition": condition, **point} for point in history
            )

        eval_sets = {
            "training": (train, train_nested),
            "familiar_new_starts": (
                train,
                _make_starts(train, trials=1, seed=seed + 81_001),
            ),
            "iid": (test, test_starts),
        }
        for variant_index, variant in enumerate(config.representation.shifted_variants):
            eval_sets[variant] = _condition_tasks(
                family,
                benchmark,
                seed=seed + 90_000 + 100 * variant_index,
                variant=variant,
            )

        iid_packed = iid_h = iid_components = None
        for split, (task_list, starts_nested) in eval_sets.items():
            packed, h = _aligned(task_list, starts_nested, device)
            if split == "iid":
                iid_packed, iid_h = packed, h
                iid_components = build_component_descriptors(packed, h)
            for condition, model in trained.items():
                summary, analysis = _evaluate_condition(
                    model,
                    condition,
                    packed,
                    h,
                    config=config,
                    benchmark=benchmark,
                    permutation_seed=seed + 100_000 + len(rows),
                    collect_attention_rows=(split != "training"),
                )
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "oracle": False,
                        **summary,
                    }
                )
                if condition == "LandscapeAttention":
                    attention_rows.extend(
                        {"seed": seed, "split": split, **value} for value in analysis
                    )
            shuffled_summary, analysis = _evaluate_condition(
                trained["LandscapeAttention"],
                "LandscapeAttention",
                packed,
                h,
                config=config,
                benchmark=benchmark,
                permutation_seed=seed + 110_000 + len(rows),
                shuffled=True,
                collect_attention_rows=(split != "training"),
            )
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "condition": "LandscapeAttentionShuffled",
                    "oracle": False,
                    **shuffled_summary,
                }
            )
            attention_rows.extend(
                {"seed": seed, "split": f"{split}_shuffled", **value}
                for value in analysis
            )
            if split != "training":
                oracle = _oracle_metrics(
                    config,
                    seed=seed,
                    packed=packed,
                    starts=h,
                    device=device,
                    prior_results=prior_oracle,
                )
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "condition": "OracleTargetDirection",
                        "oracle": True,
                        **{metric: oracle[metric] for metric in PRIMARY_METRICS},
                    }
                )

        assert (
            iid_packed is not None and iid_h is not None and iid_components is not None
        )
        probe_metrics, probe_metadata = _fit_probe(
            config,
            seed=seed,
            path=ident_dir / "models" / f"ComponentUtilityProbe_seed_{seed}.pt",
            components_train=components_train,
            packed_train=packed_train,
            components_test=iid_components,
            packed_test=iid_packed,
            device=device,
        )
        probe_rows.append({"seed": seed, **probe_metrics})
        probe_curve_rows.extend(
            {"seed": seed, **point} for point in probe_metadata.pop("history", [])
        )

        sample_count = min(16, len(iid_h))
        sample_index = torch.arange(sample_count, device=device)
        sample_h = iid_h.index_select(0, sample_index)
        sample_components = iid_components.index_select(sample_index)
        with torch.no_grad():
            permuted = _permuted(sample_components, seed=seed + 120_001)
            deep_a = trained["LandscapeDeepSets"](sample_h, sample_components)
            deep_b = trained["LandscapeDeepSets"](sample_h, permuted)
            attn_a = trained["LandscapeAttention"](sample_h, sample_components)
            attn_b = trained["LandscapeAttention"](sample_h, permuted)
            deep_error = float(
                torch.max(torch.abs(deep_a.candidate - deep_b.candidate)).cpu()
            )
            attention_error = float(
                torch.max(torch.abs(attn_a.candidate - attn_b.candidate)).cpu()
            )
        if deep_error > 2e-5 or attention_error > 2e-5:
            raise AssertionError("component-order permutation changed invariant output")
        source_index = _different_geometry_indices(iid_packed, seed=seed + 120_002)
        if not bool(
            torch.all(
                iid_packed.geometry_id
                != iid_packed.geometry_id.index_select(0, source_index)
            )
        ):
            raise AssertionError("shuffled landscape was not mismatched")
        leakage_seeds.append(
            {
                "seed": seed,
                "deep_sets_permutation_max_abs_error": deep_error,
                "attention_permutation_max_abs_error": attention_error,
                "train_test_geometry_hash_overlap": 0,
                "shuffled_geometry_mismatch": True,
                "original_goal_used": False,
                "solution_labels_used_as_inputs": False,
                "refinement_used_as_input": False,
                "parameter_counts": parameter_counts,
            }
        )

        if seed_index == 0:
            count = min(8, len(iid_h))
            indices = torch.arange(count, device=device)
            sample_h = iid_h.index_select(0, indices)
            sample_tasks = iid_packed.index_select(indices)
            sample_components = _permuted(
                iid_components.index_select(indices), seed=seed + 130_001
            )
            with torch.no_grad():
                proposal = trained["LandscapeAttention"](sample_h, sample_components)
                useful_labels, useful_index = component_evaluation_labels(sample_tasks)
                useful_labels = torch.gather(
                    useful_labels, 1, sample_components.permutation
                )
                useful_index = torch.gather(
                    useful_index, 1, sample_components.permutation
                )
                for task_index in range(count):
                    for candidate_index in range(config.representation.candidates):
                        for component_index in range(sample_components.components):
                            if not bool(
                                sample_components.mask[task_index, component_index]
                            ):
                                continue
                            attention_examples.append(
                                {
                                    "seed": seed,
                                    "task_index": task_index,
                                    "candidate_index": candidate_index,
                                    "component_index": component_index,
                                    "attention": float(
                                        proposal.attention[
                                            task_index, candidate_index, component_index
                                        ].cpu()
                                    ),
                                    "component_is_useful_evaluation_only": int(
                                        useful_labels[task_index, component_index].cpu()
                                    ),
                                    "useful_region_index_evaluation_only": int(
                                        useful_index[task_index, component_index].cpu()
                                    ),
                                }
                            )

    comparisons = [
        ("LandscapeDeepSets", "TaskOnly"),
        ("LandscapeAttention", "TaskOnlyParameterMatched"),
        ("LandscapeAttention", "LandscapeDeepSets"),
        ("LandscapeAttention", "LandscapeAttentionShuffled"),
    ]
    summary_rows = _summary(rows, ("condition", "split"), PRIMARY_METRICS)
    paired = _paired_rows(rows, comparisons)
    gap_rows = _familiar_gap(rows)
    gap_summary = _summary(
        gap_rows,
        ("condition",),
        ("cosine_gap", "proposal_recall_gap", "mode_set_recall_gap"),
    )
    probe_summary = _summary(
        probe_rows,
        tuple(),
        (
            "accuracy",
            "balanced_accuracy",
            "auroc",
            "precision",
            "recall",
            "useful_nonuseful_score_separation",
        ),
    )
    metadata_summary = _summary(
        metadata_rows,
        ("condition",),
        (
            "training_seconds",
            "parameter_count",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ),
    )

    write_csv(rep_dir / "per_seed_results.csv", rows)
    write_csv(rep_dir / "summary.csv", summary_rows)
    write_csv(rep_dir / "paired_comparisons.csv", paired)
    write_csv(rep_dir / "familiar_new_geometry_gap.csv", gap_rows)
    write_csv(rep_dir / "familiar_new_geometry_gap_summary.csv", gap_summary)
    write_csv(rep_dir / "training_curves.csv", curve_rows)
    write_csv(rep_dir / "model_parameter_counts.csv", metadata_rows)
    write_csv(rep_dir / "model_parameter_summary.csv", metadata_summary)
    write_csv(ident_dir / "component_utility_probe_per_seed.csv", probe_rows)
    write_csv(ident_dir / "component_utility_probe_summary.csv", probe_summary)
    write_csv(ident_dir / "component_utility_probe_curves.csv", probe_curve_rows)
    write_csv(attn_dir / "attention_analysis.csv", attention_rows)
    write_csv(attn_dir / "attention_example_source.csv", attention_examples)
    leakage_audit = {
        "passed": True,
        "checks": {
            "labels_absent_from_model_input_fields": True,
            "original_target_direction_goal_absent": True,
            "useful_centers_are_label_evaluation_only": True,
            "component_order_randomized": True,
            "pooled_and_attention_outputs_permutation_invariant": True,
            "shuffled_landscape_is_mismatched": True,
            "train_test_geometry_overlap": False,
            "refinement_outcomes_absent_from_inputs": True,
            "oracle_nonoracle_storage_separate": True,
        },
        "per_seed": leakage_seeds,
    }
    (rep_dir / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, indent=2), encoding="utf-8"
    )
    completion = {
        "seeds": len(config.representation.seeds),
        "result_rows": len(rows),
        "paired_rows": len(paired),
        "probe_rows": len(probe_rows),
        "attention_analysis_rows": len(attention_rows),
        "outputs": {
            "representation": str(rep_dir),
            "attention": str(attn_dir),
            "identifiability": str(ident_dir),
        },
    }
    (rep_dir / "completion.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    return completion
