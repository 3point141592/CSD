"""Compute paired-seed contrasts between the two Direct feature-width variants."""

import csv
import random
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / "per_seed.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 240
lookup = {
    (int(r["dimension"]), int(r["seed"]), r["split"], r["model"]): float(
        r["mode_set_recall"]
    )
    for r in rows
}
assert len(lookup) == 240
seeds = sorted({int(r["seed"]) for r in rows})
assert len(seeds) == 30
rng = random.Random(20261025)
output = []
for dimension in (32, 64):
    for split in ("iid", "exact_rotated"):
        differences = [
            lookup[dimension, seed, split, "DirectRawWide"]
            - lookup[dimension, seed, split, "DirectPreparedNarrow"]
            for seed in seeds
        ]
        bootstrap = sorted(
            sum(differences[rng.randrange(30)] for _ in range(30)) / 30
            for _ in range(20000)
        )
        output.append(
            {
                "dimension": dimension,
                "split": split,
                "contrast": "DirectRawWide_minus_DirectPreparedNarrow",
                "n_paired_seeds": 30,
                "mean_difference": statistics.mean(differences),
                "sample_sd": statistics.stdev(differences),
                "ci95_low": bootstrap[500],
                "ci95_high": bootstrap[19499],
                "method": "paired_seed_percentile_bootstrap_20000",
                "analysis_seed": 20261025,
            }
        )
with (root / "new_pair_contrasts.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(output[0]))
    writer.writeheader()
    writer.writerows(output)
print("Wrote", len(output), "paired contrasts")
