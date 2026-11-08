"""Evaluation-only d=32 reference corruption for frozen Relational CSD models.

The landscape, supervised targets, starts, local refiner, and model weights stay
fixed.  Only the descriptor/reference set supplied to the model is changed.
This is deliberately a small robustness diagnostic, not a retraining study.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
from creativity_toy.config import STUDY_SEEDS
from gen3_gpu.config_symmetry import SymmetryStudyConfig
from gen3_gpu.study_capacity_diagnostic import _aligned
from gen3_gpu.study_d16 import _datasets
from gen3_gpu.study_failure_diagnostics import _family
from gen3_gpu.study_symmetry import _benchmark, _equivariant_model, _load
from gen3_gpu.symmetry import (
    build_equivariant_components,
    invariant_descriptor_dimension,
)
from gen3_gpu.training_landscape_supervised import evaluate_landscape_policy

CONDITIONS = (
    "clean",
    "remove_goal_landmark_pair",
    "add_25pct_irrelevant",
    "add_50pct_irrelevant",
)


def _remove_goal_landmark_pair(tasks):
    """Hide the target useful peak and its co-centered corridor from descriptors."""
    mask = tasks.component_mask.clone()
    row = torch.arange(tasks.batch_size, device=tasks.device)
    peak = 1 + 2 * tasks.goal_index
    corridor = peak + 1
    mask[row, peak] = False
    mask[row, corridor] = False
    return replace(tasks, component_mask=mask)


def _append_irrelevant(tasks, *, count: int, seed: int):
    """Append decoy-family-shaped landmarks that are absent from Q and labels."""
    if count <= 0:
        return tasks
    device, dtype = tasks.device, tasks.dtype
    batch, dimension = tasks.batch_size, tasks.dimension
    centers = torch.empty(batch, count, dimension, device=device, dtype=dtype)
    precisions = torch.empty(
        batch, count, dimension, dimension, device=device, dtype=dtype
    )
    amplitudes = torch.empty(batch, count, device=device, dtype=dtype)
    # Reusing landmarks across starts keeps each task's corruption condition fixed.
    cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for index in range(batch):
        key = (int(tasks.geometry_id[index].cpu()), int(tasks.goal_index[index].cpu()))
        if key not in cache:
            generator = torch.Generator(device=device).manual_seed(
                seed + 10_007 * key[0] + 101 * key[1]
            )
            direction = torch.randn(
                count, dimension, generator=generator, device=device, dtype=dtype
            )
            direction = torch.nn.functional.normalize(direction, dim=-1, eps=1e-8)
            radius = 7.0 * (
                0.90
                + 0.22
                * torch.rand(count, generator=generator, device=device, dtype=dtype)
            )
            center = tasks.anchor[index][None, :] + radius[:, None] * direction
            eye = torch.eye(dimension, device=device, dtype=dtype)[None, :, :]
            covariance = 1.55**2 * direction[:, :, None] * direction[
                :, None, :
            ] + 1.20**2 * (eye - direction[:, :, None] * direction[:, None, :])
            precision = torch.linalg.inv(covariance)
            amplitude = 0.24 + 0.14 * torch.rand(
                count, generator=generator, device=device, dtype=dtype
            )
            cache[key] = (center, precision, amplitude)
        centers[index], precisions[index], amplitudes[index] = cache[key]
    return replace(
        tasks,
        component_centers=torch.cat([tasks.component_centers, centers], dim=1),
        component_precisions=torch.cat([tasks.component_precisions, precisions], dim=1),
        component_amplitudes=torch.cat([tasks.component_amplitudes, amplitudes], dim=1),
        component_mask=torch.cat(
            [
                tasks.component_mask,
                torch.ones(batch, count, dtype=torch.bool, device=device),
            ],
            dim=1,
        ),
    )


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "mean_matched_cosine",
        "proposal_recall",
        "mode_set_recall",
        "useful_region_count",
        "candidate_decoy_rate",
        "candidate_mean_quality",
    )
    result = []
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        output: dict[str, object] = {"condition": condition, "n": len(subset)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset])
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_sd"] = float(values.std(ddof=0))
            output[f"{metric}_ci95"] = float(
                1.96 * values.std(ddof=1) / np.sqrt(len(values))
            )
        result.append(output)
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(*, output: Path, device: torch.device) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config = SymmetryStudyConfig()
    benchmark = _benchmark(config)
    base = benchmark.base.base
    if config.dimension != 32 or config.candidates != 8:
        raise AssertionError("reference corruption is locked to canonical d=32/K=8")
    rows: list[dict[str, object]] = []
    for ordinal, seed in enumerate(STUDY_SEEDS, 1):
        family = _family(benchmark)
        _, _, _, _, test, test_starts = _datasets(family, benchmark, seed)
        packed_test, h_test = _aligned(test, test_starts, device)
        checkpoint = (
            REPO
            / "results"
            / "d32_equivariant_attention"
            / "models"
            / f"EquivariantAttention_seed_{seed}.pt"
        )
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        state = payload["state_dict"]
        mean = state["invariant_mean"]
        scale = state["invariant_scale"]
        model, _, _ = _load(
            checkpoint,
            lambda: _equivariant_model(
                config, seed=seed, device=device, mean=mean, scale=scale
            ),
            device,
        )
        descriptor_tasks = {
            "clean": packed_test,
            "remove_goal_landmark_pair": _remove_goal_landmark_pair(packed_test),
            "add_25pct_irrelevant": _append_irrelevant(
                packed_test, count=3, seed=seed + 141_001
            ),
            "add_50pct_irrelevant": _append_irrelevant(
                packed_test, count=6, seed=seed + 151_001
            ),
        }
        for condition, reference_tasks in descriptor_tasks.items():
            components = build_equivariant_components(reference_tasks, h_test)
            result = evaluate_landscape_policy(
                model,
                packed_test,
                h_test,
                components,
                lambda_alpha=config.lambda_alpha,
                refinement_steps=base.training.refinement_steps,
                refinement_rate=base.training.refinement_rate,
                useful_threshold=base.landscape.useful_threshold,
                component_label_tasks=reference_tasks,
                collect_per_set=False,
                collect_attention_rows=False,
            )
            rows.append({"seed": seed, "condition": condition, **result.summary})
        _write_csv(output / "per_seed.csv", rows)
        print(
            f"reference corruption: seed {seed} ({ordinal}/{len(STUDY_SEEDS)})",
            flush=True,
        )
    _write_csv(output / "summary.csv", _summary(rows))
    manifest = {
        "status": "evaluation-only diagnostic on frozen d=32 Relational CSD checkpoints",
        "seeds": list(STUDY_SEEDS),
        "conditions": {
            "clean": "unchanged 12-record reference set",
            "remove_goal_landmark_pair": (
                "mask the goal-designated useful peak and its co-centered corridor; "
                "landscape and all three supervised targets remain unchanged"
            ),
            "add_25pct_irrelevant": (
                "append three decoy-family-shaped landmark records absent from Q and labels"
            ),
            "add_50pct_irrelevant": (
                "append six decoy-family-shaped landmark records absent from Q and labels"
            ),
        },
        "fixed": {
            "model_weights": True,
            "landscapes": True,
            "starts": True,
            "targets": True,
            "local_refiner": True,
            "candidate_budget": 8,
            "test_tuning": False,
        },
        "mathematical_caution": (
            "after landmark removal, a target direction need not remain in the normalized "
            "positive cone of the supplied displacement vectors"
        ),
        "descriptor_dimension": invariant_descriptor_dimension(32),
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER / "assets" / "source_data" / "reference_corruption",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    run(output=args.output.resolve(), device=torch.device(device_name))


if __name__ == "__main__":
    main()
