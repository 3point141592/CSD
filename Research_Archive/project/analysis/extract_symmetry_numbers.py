"""Extract numeric symmetry evidence from existing canonical CSVs."""

import argparse
import csv
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
)
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
R = args.package_root.resolve(strict=True)
O = (args.output_dir or R / "new_verification/audit_20260909").resolve()
O.mkdir(parents=True, exist_ok=True)
for name in [
    "canonical_relational_symmetry_numbers.csv",
    "canonical_refiner_symmetry_numbers.csv",
]:
    if (O / name).exists():
        raise FileExistsError(
            f"Refusing to replace {name}; choose a fresh --output-dir"
        )
rows = []
refiner = []
for d in [32, 64]:
    source = R / f"results/d{d}_symmetry_audit/equivariance_metrics.csv"
    records = list(csv.DictReader(source.open(newline="", encoding="utf-8-sig")))
    for method in ["EquivariantAttention", "EquivariantAttentionAugmented"]:
        for transform in [
            "exact_rotated",
            "exact_translated",
            "exact_rotation_translation",
        ]:
            selected = [
                r
                for r in records
                if r["condition"] == method and r["transform"] == transform
            ]
            rows.append(
                dict(
                    dimension=d,
                    method=method,
                    transform=transform,
                    n_seeds=len(selected),
                    starts_per_seed=30,
                    candidates_per_start=8,
                    mean_slot_direction_cosine=statistics.mean(
                        float(r["direct_slot_equivariance_cosine"]) for r in selected
                    ),
                    min_seed_mean_slot_direction_cosine=min(
                        float(r["direct_slot_equivariance_cosine"]) for r in selected
                    ),
                    mean_absolute_alpha_error=statistics.mean(
                        float(r["alpha_invariance_error"]) for r in selected
                    ),
                    sd_seed_mean_absolute_alpha_error=statistics.stdev(
                        float(r["alpha_invariance_error"]) for r in selected
                    ),
                    max_pointwise_absolute_alpha_error=max(
                        float(r["alpha_invariance_max_error"]) for r in selected
                    ),
                    mean_matched_angular_error_degrees=statistics.mean(
                        float(r["equivariance_angular_error_degrees"]) for r in selected
                    ),
                    source=source.relative_to(R).as_posix(),
                )
            )
    source = R / f"results/d{d}_symmetry_audit/symmetry_audit_per_seed.csv"
    records = list(csv.DictReader(source.open(newline="", encoding="utf-8-sig")))
    refiner.append(
        dict(
            dimension=d,
            n_seeds=len(records),
            mean_Euclidean_refiner_coordinate_error=statistics.mean(
                float(r["refinement_equivariance_mean_error"]) for r in records
            ),
            mean_of_seed_max_Euclidean_refiner_coordinate_error=statistics.mean(
                float(r["refinement_equivariance_max_error"]) for r in records
            ),
            max_Euclidean_refiner_coordinate_error=max(
                float(r["refinement_equivariance_max_error"]) for r in records
            ),
            mean_of_seed_max_quality_error=statistics.mean(
                float(r["quality_invariance_max_error"]) for r in records
            ),
            max_quality_error=max(
                float(r["quality_invariance_max_error"]) for r in records
            ),
            mean_region_agreement=statistics.mean(
                float(r["region_classification_agreement"]) for r in records
            ),
            source=source.relative_to(R).as_posix(),
            scope="Refiner tested on original Direct proposals and their exact rigid transforms, not learned Relational proposal coordinate errors",
        )
    )
for name, records in [
    ("canonical_relational_symmetry_numbers.csv", rows),
    ("canonical_refiner_symmetry_numbers.csv", refiner),
]:
    with (O / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
print(
    json.dumps(
        {
            "relational": [r for r in rows if r["method"] == "EquivariantAttention"],
            "refiner": refiner,
        },
        indent=2,
    )
)
