"""Paired Direct-Vector CSD versus Relational CSD scaling study.

This paper-specific rerun uses the same 30 seeds, generated tasks, starts,
optimizer, step count, effective batches, candidate count, local refinement,
and evaluation metrics for both architectures at d in {2, 4, 8, 16, 32, 64}.
Outputs are isolated beneath the revision-4 paper directory; canonical result
artifacts are read through repository code but are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
from creativity_toy.analysis_utils import statistics
from creativity_toy.config import STUDY_SEEDS
from creativity_toy.experiments.study_v2 import _make_starts
from creativity_toy.landscapes_d16 import geometry_diagnostics
from gen3_gpu.config_symmetry import SymmetryStudyConfig
from gen3_gpu.landscape_representation import (
    build_component_descriptors,
    descriptor_normalization,
)
from gen3_gpu.runtime import configure_runtime, print_runtime, runtime_info
from gen3_gpu.study import write_csv
from gen3_gpu.study_landscape_representation import PRIMARY_METRICS, _summary
from gen3_gpu.study_symmetry import (
    _aligned,
    _assert_nonoracle_model,
    _benchmark,
    _current_model,
    _datasets,
    _equivariant_model,
    _evaluate,
    _exact_transforms,
    _family,
    _load,
    _save,
)
from gen3_gpu.symmetry import (
    build_equivariant_components,
    invariant_normalization,
    transform_points,
    transform_tasks,
)
from gen3_gpu.training_symmetry import train_symmetry_policy

DEFAULT_OUTPUT = PAPER / "assets" / "source_data" / "matched_scaling_study"
DIMENSIONS = (2, 4, 8, 16, 32, 64)
CONDITIONS = ("CurrentAttention", "EquivariantAttention")
PAPER_NAMES = {
    "CurrentAttention": "Direct-Vector CSD",
    "EquivariantAttention": "Relational CSD",
}

# A fixed 32-dimensional component token is used for both architectures.  The
# Relational CSD hidden width is chosen before training to minimize parameter-
# count mismatch with the unchanged Direct-Vector CSD at each dimension.
RELATIONAL_HIDDEN = {2: 81, 4: 92, 8: 127, 16: 234, 32: 543, 64: 1340}
RELATIONAL_TOKEN = 32
MAX_PARAMETER_MISMATCH = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def matched_config(dimension: int) -> SymmetryStudyConfig:
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}, got {dimension}")
    config = SymmetryStudyConfig(
        name=f"paper_matched_scaling_d{dimension}",
        dimension=dimension,
        invariant_component_hidden_dimension=RELATIONAL_HIDDEN[dimension],
        invariant_component_token_dimension=RELATIONAL_TOKEN,
    )
    config.conditions = list(CONDITIONS)
    config.seeds = list(STUDY_SEEDS)
    return config


def _manifest(config: SymmetryStudyConfig, runtime) -> dict[str, object]:
    return {
        "status": "fresh paper-specific matched rerun",
        "conditions": list(CONDITIONS),
        "paper_names": PAPER_NAMES,
        "config": config.to_dict(),
        "runtime": runtime.to_dict(),
        "protocol_lock": {
            "dimensions": list(DIMENSIONS),
            "seeds": list(STUDY_SEEDS),
            "paired_task_and_start_generation": True,
            "paired_effective_batch_seed": True,
            "training_geometries": 80,
            "starts_per_task": 32,
            "optimizer_steps": 1500,
            "batch_size": 960,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "lambda_alpha": 0.1,
            "candidate_count": 8,
            "component_token_dimension": RELATIONAL_TOKEN,
            "relational_hidden_by_dimension": RELATIONAL_HIDDEN,
            "parameter_match_tolerance": MAX_PARAMETER_MISMATCH,
            "primary_split": "iid",
            "secondary_split": "exact_rotated",
            "model_selection": "none",
            "original_results_modified": False,
        },
        "source_hashes": {
            "gen3_gpu/landscape_representation.py": _sha256(
                REPO / "gen3_gpu" / "landscape_representation.py"
            ),
            "gen3_gpu/symmetry.py": _sha256(REPO / "gen3_gpu" / "symmetry.py"),
            "gen3_gpu/training_symmetry.py": _sha256(
                REPO / "gen3_gpu" / "training_symmetry.py"
            ),
            "gen3_gpu/training_landscape_supervised.py": _sha256(
                REPO / "gen3_gpu" / "training_landscape_supervised.py"
            ),
            "gen3_gpu/study_symmetry.py": _sha256(
                REPO / "gen3_gpu" / "study_symmetry.py"
            ),
        },
    }


def _paired_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (
            int(row["dimension"]),
            int(row["seed"]),
            str(row["condition"]),
            str(row["split"]),
        ): row
        for row in rows
    }
    output: list[dict[str, object]] = []
    for dimension in DIMENSIONS:
        for split in ("iid", "exact_rotated"):
            for metric in PRIMARY_METRICS:
                deltas = []
                for seed in STUDY_SEEDS:
                    direct = lookup.get((dimension, seed, "CurrentAttention", split))
                    relational = lookup.get(
                        (dimension, seed, "EquivariantAttention", split)
                    )
                    if direct is not None and relational is not None:
                        deltas.append(float(relational[metric]) - float(direct[metric]))
                if not deltas:
                    continue
                result = statistics(deltas)
                output.append(
                    {
                        "dimension": dimension,
                        "split": split,
                        "metric": metric,
                        "contrast": "Relational CSD - Direct-Vector CSD",
                        "mean_difference": result["mean"],
                        "sd_difference": result["sd"],
                        "ci95_difference": result["ci95"],
                        "n": len(deltas),
                    }
                )
    return output


def run_dimension(
    dimension: int, *, output_root: Path, device: torch.device, runtime
) -> dict[str, object]:
    config = matched_config(dimension)
    root = output_root / f"d{dimension}"
    model_dir = root / "models"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "manifest.json", _manifest(config, runtime))

    rows: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for seed_index, seed in enumerate(config.seeds, 1):
        benchmark = _benchmark(config)
        family = _family(benchmark)
        train, _, _, _, test, test_starts = _datasets(family, benchmark, seed)
        geometry_path = root / "geometry_audit.json"
        if not geometry_path.exists():
            audit = geometry_diagnostics(train)
            if dimension >= 16 and not bool(audit["not_effectively_2d"]):
                raise AssertionError(
                    "high-dimensional task coordinates are effectively 2D"
                )
            _atomic_json(geometry_path, audit)

        train_nested = _make_starts(
            train, trials=config.starts_per_task, seed=seed + 41_001
        )
        packed_train, h_train = _aligned(train, train_nested, device)
        current_components = build_component_descriptors(packed_train, h_train)
        relational_components = build_equivariant_components(packed_train, h_train)
        current_mean, current_scale = descriptor_normalization(current_components)
        relational_mean, relational_scale = invariant_normalization(
            relational_components
        )

        factories = {
            "CurrentAttention": lambda: _current_model(
                config,
                seed=seed,
                device=device,
                mean=current_mean,
                scale=current_scale,
            ),
            "EquivariantAttention": lambda: _equivariant_model(
                config,
                seed=seed,
                device=device,
                mean=relational_mean,
                scale=relational_scale,
            ),
        }
        components = {
            "CurrentAttention": current_components,
            "EquivariantAttention": relational_components,
        }
        component_kinds = {
            "CurrentAttention": "current",
            "EquivariantAttention": "equivariant",
        }

        seed_counts: dict[str, int] = {}
        for condition in CONDITIONS:
            factory = factories[condition]
            checkpoint = model_dir / f"{condition}_seed_{seed}.pt"
            if checkpoint.exists():
                model, history, metadata = _load(checkpoint, factory, device)
                metadata = {**metadata, "checkpoint_reused_on_resume": True}
            else:
                model = factory()
                _assert_nonoracle_model(model)
                result = train_symmetry_policy(
                    model,
                    packed_train,
                    h_train,
                    components[condition],
                    component_kind=component_kinds[condition],
                    augmented=False,
                    rotation_pool=torch.empty(0, dimension, dimension, device=device),
                    translation_radius_max=config.augmentation.translation_radius_max,
                    steps=config.supervised_steps,
                    batch_size=min(config.batch_size, len(h_train)),
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                    gradient_clip=config.gradient_clip,
                    lambda_alpha=config.lambda_alpha,
                    seed=seed + 61_101,
                    curve_steps=config.curve_steps,
                )
                history = result.history
                metadata = {
                    "seed": seed,
                    "dimension": dimension,
                    "condition": condition,
                    "paper_name": PAPER_NAMES[condition],
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "training_seconds": result.elapsed_seconds,
                    "examples_seen": result.examples_seen,
                    "batch_size": min(config.batch_size, len(h_train)),
                    "steps": config.supervised_steps,
                    "effective_batch_seed": seed + 61_101,
                    "checkpoint_reused_on_resume": False,
                }
                _save(checkpoint, model, history=history, metadata=metadata)
            _assert_nonoracle_model(model)
            seed_counts[condition] = int(metadata["parameter_count"])
            metadata_rows.append(metadata)
            curves.extend(
                {
                    "dimension": dimension,
                    "seed": seed,
                    "condition": condition,
                    "paper_name": PAPER_NAMES[condition],
                    **point,
                }
                for point in history
            )

            packed_test, h_test = _aligned(test, test_starts, device)
            rotation = _exact_transforms(config, h_test, seed=seed + 81_001)[
                "exact_rotated"
            ]
            eval_sets = {
                "iid": (packed_test, h_test),
                "exact_rotated": (
                    transform_tasks(packed_test, rotation),
                    transform_points(h_test, rotation.rotation, rotation.translation),
                ),
            }
            for split, (tasks_eval, starts_eval) in eval_sets.items():
                rows.append(
                    {
                        "dimension": dimension,
                        "seed": seed,
                        "condition": condition,
                        "paper_name": PAPER_NAMES[condition],
                        "split": split,
                        **_evaluate(
                            condition, model, tasks_eval, starts_eval, benchmark, config
                        ),
                    }
                )

        mismatch = abs(
            seed_counts["CurrentAttention"] - seed_counts["EquivariantAttention"]
        )
        if mismatch > MAX_PARAMETER_MISMATCH:
            raise AssertionError(
                f"parameter mismatch {mismatch} exceeds {MAX_PARAMETER_MISMATCH}"
            )

        write_csv(root / "per_seed.csv", rows)
        write_csv(root / "training_curves.csv", curves)
        write_csv(root / "metadata.csv", metadata_rows)
        _atomic_json(
            root / "progress.json",
            {
                "completed_seeds": sorted(
                    {int(value["seed"]) for value in metadata_rows}
                ),
                "completed_seed_count": seed_index,
                "required_seed_count": len(config.seeds),
            },
        )
        print(
            f"d={dimension}: completed paired seed {seed} "
            f"({seed_index}/{len(config.seeds)})",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = _summary(
        rows, ("dimension", "condition", "paper_name", "split"), PRIMARY_METRICS
    )
    write_csv(root / "summary.csv", summary)
    write_csv(root / "paired_summary.csv", _paired_summary(rows))
    counts_by_condition = {
        condition: sorted(
            {
                int(row["parameter_count"])
                for row in metadata_rows
                if row["condition"] == condition
            }
        )
        for condition in CONDITIONS
    }
    completion = {
        "dimension": dimension,
        "conditions": list(CONDITIONS),
        "seeds": len(config.seeds),
        "result_rows": len(rows),
        "parameter_counts": counts_by_condition,
        "parameter_mismatch": abs(
            counts_by_condition["CurrentAttention"][0]
            - counts_by_condition["EquivariantAttention"][0]
        ),
        "mean_training_seconds": {
            condition: float(
                np.mean(
                    [
                        float(row.get("training_seconds", 0.0))
                        for row in metadata_rows
                        if row["condition"] == condition
                    ]
                )
            )
            for condition in CONDITIONS
        },
        "all_finite": all(
            np.isfinite(float(row[metric]))
            for row in rows
            for metric in PRIMARY_METRICS
            if metric in row
        ),
    }
    _atomic_json(root / "completion.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DIMENSIONS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dimensions = tuple(args.dimensions)
    if any(value not in DIMENSIONS for value in dimensions):
        raise SystemExit(f"allowed dimensions are {DIMENSIONS}")
    device = configure_runtime(args.device)
    runtime = runtime_info(device, torch.float32)
    print_runtime(runtime)
    completions = [
        run_dimension(
            value,
            output_root=args.output_root.resolve(),
            device=device,
            runtime=runtime,
        )
        for value in dimensions
    ]
    completion_path = args.output_root.resolve() / "completion.json"
    prior_runs: dict[int, dict[str, object]] = {}
    if completion_path.exists():
        prior = json.loads(completion_path.read_text(encoding="utf-8"))
        prior_runs = {int(item["dimension"]): item for item in prior.get("runs", [])}
    prior_runs.update({int(item["dimension"]): item for item in completions})
    merged = [prior_runs[value] for value in sorted(prior_runs)]
    _atomic_json(
        completion_path,
        {
            "dimensions": [int(item["dimension"]) for item in merged],
            "conditions": list(CONDITIONS),
            "complete": (
                set(prior_runs) == set(DIMENSIONS)
                and all(int(item["seeds"]) == len(STUDY_SEEDS) for item in merged)
            ),
            "runs": merged,
        },
    )
    print(json.dumps(completions, indent=2))


if __name__ == "__main__":
    main()
