"""Fresh matched d=2/16/64 runs of the final equivariant CSD architecture.

This paper-specific study writes only beneath
``paper_csd_toy/assets/source_data/equivariant_scaling_rerun``.  It imports the
canonical task construction, loss, optimizer loop, model, and evaluators without
changing any original experiment artifact.
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
from creativity_toy.config import STUDY_SEEDS
from creativity_toy.experiments.study_v2 import _make_starts
from creativity_toy.landscapes_d16 import geometry_diagnostics
from gen3_gpu.config_symmetry import SymmetryStudyConfig
from gen3_gpu.runtime import configure_runtime, print_runtime, runtime_info
from gen3_gpu.study import write_csv
from gen3_gpu.study_landscape_representation import PRIMARY_METRICS, _summary
from gen3_gpu.study_symmetry import (
    _aligned,
    _assert_nonoracle_model,
    _benchmark,
    _datasets,
    _equivariant_model,
    _evaluate,
    _exact_transforms,
    _family,
    _load,
    _propose,
    _save,
)
from gen3_gpu.symmetry import (
    build_equivariant_components,
    invariant_normalization,
    transform_points,
    transform_tasks,
)
from gen3_gpu.training_symmetry import set_equivariance_metrics, train_symmetry_policy

DEFAULT_OUTPUT = PAPER / "assets" / "source_data" / "equivariant_scaling_rerun"
DIMENSIONS = (2, 16, 64)
CONDITION = "EquivariantAttention"


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
        name=f"paper_equivariant_scaling_d{dimension}",
        dimension=dimension,
        # Fixed learned widths avoid an extra width change in this dimension sweep;
        # descriptor length can still reflect the ambient dimension.
        invariant_component_hidden_dimension=981,
        invariant_component_token_dimension=64,
    )
    config.conditions = [CONDITION]
    config.seeds = list(STUDY_SEEDS)
    return config


def _manifest(config: SymmetryStudyConfig, runtime) -> dict[str, object]:
    return {
        "status": "fresh paper-specific rerun",
        "condition": CONDITION,
        "architecture": "EquivariantLandscapeAttention",
        "augmentation": False,
        "config": config.to_dict(),
        "runtime": runtime.to_dict(),
        "protocol_lock": {
            "dimensions": list(DIMENSIONS),
            "seeds": list(STUDY_SEEDS),
            "training_geometries": 80,
            "starts_per_task": 32,
            "optimizer_steps": 1500,
            "batch_size": 960,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "lambda_alpha": 0.1,
            "candidate_count": 8,
            "component_hidden_dimension": 981,
            "component_token_dimension": 64,
            "model_selection": "none",
            "original_results_modified": False,
        },
        "source_hashes": {
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
    equivariance_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for index, seed in enumerate(config.seeds, 1):
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
        components = build_equivariant_components(packed_train, h_train)
        mean, scale = invariant_normalization(components)
        factory = lambda: _equivariant_model(
            config, seed=seed, device=device, mean=mean, scale=scale
        )
        checkpoint = model_dir / f"EquivariantAttention_seed_{seed}.pt"
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
                components,
                component_kind="equivariant",
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
                "condition": CONDITION,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "training_seconds": result.elapsed_seconds,
                "examples_seen": result.examples_seen,
                "batch_size": min(config.batch_size, len(h_train)),
                "steps": config.supervised_steps,
                "checkpoint_reused_on_resume": False,
            }
            _save(checkpoint, model, history=history, metadata=metadata)
        _assert_nonoracle_model(model)
        metadata_rows.append(metadata)
        curves.extend(
            {"dimension": dimension, "seed": seed, "condition": CONDITION, **point}
            for point in history
        )

        packed_test, h_test = _aligned(test, test_starts, device)
        transforms = _exact_transforms(config, h_test, seed=seed + 81_001)
        eval_sets = {"iid": (packed_test, h_test)}
        for split, transform in transforms.items():
            eval_sets[split] = (
                transform_tasks(packed_test, transform),
                transform_points(h_test, transform.rotation, transform.translation),
            )
        for split, (tasks_eval, starts_eval) in eval_sets.items():
            rows.append(
                {
                    "dimension": dimension,
                    "seed": seed,
                    "condition": CONDITION,
                    "split": split,
                    **_evaluate(
                        CONDITION, model, tasks_eval, starts_eval, benchmark, config
                    ),
                }
            )

        base = _propose(CONDITION, model, packed_test, h_test)
        for transform_name, transform in transforms.items():
            tasks_eval, starts_eval = eval_sets[transform_name]
            transformed = _propose(CONDITION, model, tasks_eval, starts_eval)
            equivariance_rows.append(
                {
                    "dimension": dimension,
                    "seed": seed,
                    "condition": CONDITION,
                    "transform": transform_name,
                    **set_equivariance_metrics(base, transformed, transform.rotation),
                }
            )

        write_csv(root / "per_seed.csv", rows)
        write_csv(root / "training_curves.csv", curves)
        write_csv(root / "equivariance_per_seed.csv", equivariance_rows)
        _atomic_json(
            root / "progress.json",
            {
                "completed_seeds": [int(value["seed"]) for value in metadata_rows],
                "completed_count": index,
                "required_count": len(config.seeds),
            },
        )
        print(
            f"d={dimension}: completed seed {seed} ({index}/{len(config.seeds)})",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = _summary(rows, ("dimension", "condition", "split"), PRIMARY_METRICS)
    equivariance_summary = _summary(
        equivariance_rows,
        ("dimension", "condition", "transform"),
        (
            "equivariance_cosine",
            "equivariance_angular_error_degrees",
            "alpha_invariance_error",
            "hungarian_set_alpha_error",
            "direct_slot_equivariance_cosine",
        ),
    )
    write_csv(root / "summary.csv", summary)
    write_csv(root / "equivariance_summary.csv", equivariance_summary)
    completion = {
        "dimension": dimension,
        "condition": CONDITION,
        "seeds": len(config.seeds),
        "result_rows": len(rows),
        "equivariance_rows": len(equivariance_rows),
        "parameter_count_min": min(
            int(row["parameter_count"]) for row in metadata_rows
        ),
        "parameter_count_max": max(
            int(row["parameter_count"]) for row in metadata_rows
        ),
        "mean_training_seconds": float(
            np.mean([float(row.get("training_seconds", 0.0)) for row in metadata_rows])
        ),
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
