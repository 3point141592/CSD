"""Compare new verification rows with archived rows; timings are excluded."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(a, b):
    with a.open(newline="", encoding="utf-8-sig") as f:
        old = list(csv.DictReader(f))
    with b.open(newline="", encoding="utf-8-sig") as f:
        new = list(csv.DictReader(f))
    result = {
        "archived": a.relative_to(ROOT).as_posix(),
        "replayed": b.relative_to(ROOT).as_posix(),
        "old_rows": len(old),
        "new_rows": len(new),
        "excluded_timing_columns": [],
        "different_cells": 0,
        "max_absolute_numeric_difference": 0.0,
        "examples": [],
    }
    if len(old) != len(new):
        raise ValueError(result)
    for i, (x, y) in enumerate(zip(old, new)):
        if x.keys() != y.keys():
            raise ValueError("column mismatch")
        for k in x:
            if "elapsed" in k or "seconds" in k:
                if k not in result["excluded_timing_columns"]:
                    result["excluded_timing_columns"].append(k)
                continue
            if x[k] != y[k]:
                try:
                    diff = abs(float(x[k]) - float(y[k]))
                except ValueError:
                    diff = None
                if diff == 0:
                    continue
                result["different_cells"] += 1
                if diff is not None:
                    result["max_absolute_numeric_difference"] = max(
                        result["max_absolute_numeric_difference"], diff
                    )
                if len(result["examples"]) < 5:
                    result["examples"].append(
                        {
                            "row": i,
                            "column": k,
                            "old": x[k],
                            "new": y[k],
                            "absolute_difference": diff,
                        }
                    )
    result["scientific_values_exact"] = result["different_cells"] == 0
    return result


def main():
    cases = [
        (
            "fair_selector_d32",
            "engineering_fair_selector_30",
            ["per_seed.csv", "per_set.csv", "summary.csv", "scorer_training.csv"],
        ),
        (
            "corruption_reachability_d32",
            "engineering_cone_900",
            ["per_seed.csv", "per_set.csv"],
        ),
        (
            "reference_corruption_replay_d32",
            "engineering_corruption_30",
            ["per_seed.csv", "summary.csv"],
        ),
    ]
    results = []
    for old, new, files in cases:
        for name in files:
            target = ROOT / "reproduction_runs" / new / name
            if target.exists():
                results.append(compare(ROOT / "new_verification" / old / name, target))
    out = ROOT / "audit_records/engineering_20260909/numerical_replay_comparisons.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
