"""Pre-refinement directional cone diagnostic for saved d=32 reference deletion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from creativity_toy.config import STUDY_SEEDS
from gen3_gpu.config_symmetry import SymmetryStudyConfig
from gen3_gpu.study_capacity_diagnostic import _aligned
from gen3_gpu.study_d16 import _datasets
from gen3_gpu.study_failure_diagnostics import _family
from gen3_gpu.study_symmetry import _benchmark
from run_scripts.reproduction_support import fresh_output, verify_recorded_sources
from scipy.optimize import nnls


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "n": float(len(values)),
        "mean": float(values.mean()),
        "sd_sample": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "fraction_ge_0_90": float(np.mean(values >= 0.90)),
        "fraction_ge_0_95": float(np.mean(values >= 0.95)),
        "fraction_ge_0_99": float(np.mean(values >= 0.99)),
    }


def run(output: Path) -> None:
    output = fresh_output(output)
    verify_recorded_sources(
        ROOT, ROOT / "new_verification/fair_selector_d32/manifest.json"
    )
    config = SymmetryStudyConfig()
    benchmark = _benchmark(config)
    if config.dimension != 32 or config.candidates != 8:
        raise AssertionError("audit is locked to canonical d=32/K=8")
    rows: list[dict[str, object]] = []
    for seed in STUDY_SEEDS:
        family = _family(benchmark)
        _, _, _, _, test, test_starts = _datasets(family, benchmark, seed)
        packed, starts = _aligned(test, test_starts, torch.device("cpu"))
        for row in range(packed.batch_size):
            goal = int(packed.goal_index[row])
            target_vector = packed.useful_centers[row, goal] - starts[row]
            target = target_vector / torch.linalg.vector_norm(target_vector).clamp_min(
                1e-12
            )
            keep = packed.component_mask[row].clone()
            peak = 1 + 2 * goal
            keep[peak] = False
            keep[peak + 1] = False
            remaining = packed.component_centers[row, keep] - starts[row]
            matrix = remaining.detach().numpy().T.astype(np.float64, copy=False)
            target_np = target.detach().numpy().astype(np.float64, copy=False)
            coefficient, residual_norm = nnls(matrix, target_np)
            projection = matrix @ coefficient
            projection_norm = float(np.linalg.norm(projection))
            cone_cosine = (
                float(np.dot(target_np, projection) / projection_norm)
                if projection_norm > 1e-12
                else 0.0
            )
            remaining_norm = np.linalg.norm(matrix, axis=0)
            valid = remaining_norm > 1e-12
            single_cosine = float(
                np.max(
                    (target_np[:, None] * matrix[:, valid]).sum(axis=0)
                    / remaining_norm[valid]
                )
            )
            rows.append(
                {
                    "seed": int(seed),
                    "set_index": row,
                    "geometry_id": int(packed.geometry_id[row]),
                    "goal_index": goal,
                    "remaining_reference_records": int(keep.sum()),
                    "nnls_nonzero_coefficients": int(np.sum(coefficient > 1e-9)),
                    "cone_max_cosine": cone_cosine,
                    "best_single_reference_cosine": single_cosine,
                    "projection_norm": projection_norm,
                    "nnls_residual_norm": float(residual_norm),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_set.csv", rows)
    per_seed: list[dict[str, object]] = []
    for seed in STUDY_SEEDS:
        subset = [row for row in rows if row["seed"] == seed]
        cone = np.asarray([row["cone_max_cosine"] for row in subset], dtype=np.float64)
        single = np.asarray(
            [row["best_single_reference_cosine"] for row in subset], dtype=np.float64
        )
        per_seed.append(
            {
                "seed": int(seed),
                **{f"cone_{key}": value for key, value in summarize(cone).items()},
                **{f"single_{key}": value for key, value in summarize(single).items()},
            }
        )
    write_csv(output / "per_seed.csv", per_seed)
    cone = np.asarray([row["cone_max_cosine"] for row in rows], dtype=np.float64)
    single = np.asarray(
        [row["best_single_reference_cosine"] for row in rows], dtype=np.float64
    )
    report = {
        "status": "retrospective pre-refinement directional cone diagnostic",
        "question": "Is the removed target direction reachable from the nonnegative cone of remaining references?",
        "seeds": list(STUDY_SEEDS),
        "sets_per_seed": int(len(rows) / len(STUDY_SEEDS)),
        "method": "SciPy NNLS projection of the unit removed-target direction onto remaining reference displacements",
        "cone_max_cosine": summarize(cone),
        "best_single_reference_cosine": summarize(single),
        "interpretation_rule": (
            "Low cosine limits initial target-direction approximation by the remaining positive cone; "
            "it does not certify useful-region unreachability after refinement."
        ),
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy_nnls": True},
    }
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
