"""Independent final arithmetic and artifact completeness checks; no training."""

import csv
import json

import numpy as np
import torch
from experiment import (
    MODELS,
    ORIGINAL,
    OUT,
    check_lock,
    job_valid,
    matched_config,
    read_json,
    tensor_hash,
    write_json,
)

check_lock()
rows = list(csv.DictReader((OUT / "per_seed.csv").open(encoding="utf-8")))
summary = list(csv.DictReader((OUT / "comparison_summary.csv").open(encoding="utf-8")))
assert len(rows) == 240 and all(r["status"] == "complete" for r in rows)
keys = {(int(r["dimension"]), int(r["seed"]), r["model"], r["split"]) for r in rows}
expected = {
    (d, s, m, split)
    for d in (32, 64)
    for s in matched_config(d).seeds
    for m in MODELS
    for split in ("iid", "exact_rotated")
}
assert keys == expected and len(keys) == len(rows)
primary = []
artifacts = []
for d in (32, 64):
    seeds = matched_config(d).seeds
    for split in ("iid", "exact_rotated"):
        lookup = {
            (int(r["seed"]), r["model"]): r
            for r in rows
            if int(r["dimension"]) == d and r["split"] == split
        }
        a = np.asarray(
            [float(lookup[s, "DirectPrepared"]["mode_set_recall"]) for s in seeds]
        )
        b = np.asarray(
            [float(lookup[s, "Relational"]["mode_set_recall"]) for s in seeds]
        )
        delta = a - b
        indices = np.random.default_rng(20260921 + d).integers(0, 30, size=(20000, 30))
        low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
        row = next(
            r
            for r in summary
            if int(r["dimension"]) == d
            and r["split"] == split
            and r["endpoint"] == "mode_set_recall"
            and r["comparison"] == "DirectPrepared_minus_Relational"
        )
        for col, value in [
            ("mean", delta.mean()),
            ("sample_sd", delta.std(ddof=1)),
            ("ci95_low", low),
            ("ci95_high", high),
        ]:
            assert abs(float(row[col]) - value) < 1e-14, (d, split, col)
        primary.append(
            {
                "dimension": d,
                "split": split,
                "direct_mean": float(a.mean()),
                "relational_mean": float(b.mean()),
                "paired_mean": float(delta.mean()),
                "paired_sd": float(delta.std(ddof=1)),
                "ci95": [float(low), float(high)],
                "n": 30,
            }
        )
    for seed in seeds:
        for model in MODELS:
            folder = OUT / f"jobs/d{d}/seed_{seed}/{model}"
            assert job_valid(folder, f"d{d}_s{seed}_{model}")
            payload = torch.load(
                folder / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            assert (
                payload["metadata"]["steps"] == 1500
                and payload["metadata"]["examples_seen"] == 1_440_000
            )
            assert [r["step"] for r in payload["history"]] == [
                25,
                50,
                100,
                250,
                500,
                1000,
                1500,
            ]
            assert all(torch.isfinite(v).all() for v in payload["state_dict"].values())
            field = "descriptor" if model == "DirectPrepared" else "invariant"
            provenance = read_json(OUT / f"data/d{d}_s{seed}/provenance.json")
            for i, suffix in enumerate(["mean", "scale"]):
                assert (
                    tensor_hash(payload["state_dict"][field + "_" + suffix])
                    == provenance["feature_hashes"][model + "_" + str(i)]
                )
            same = None
            if model == "Relational":
                old = torch.load(
                    ORIGINAL / f"d{d}/models/EquivariantAttention_seed_{seed}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                same = all(
                    torch.equal(v, old["state_dict"][k])
                    for k, v in payload["state_dict"].items()
                )
            artifacts.append(
                {
                    "dimension": d,
                    "seed": seed,
                    "model": model,
                    "final_checkpoint_and_curves_complete": True,
                    "finite_state": True,
                    "training_only_normalization_hashes_match": True,
                    "fresh_relational_identical_to_historical_state": same,
                }
            )
historical = list(
    csv.DictReader((OUT / "historical_context.csv").open(encoding="utf-8"))
)
assert len(historical) == 360
assert (
    len({(r["dimension"], r["seed"], r["condition"], r["split"]) for r in historical})
    == 360
)
write_json(
    OUT / "verification/final_independent_validation.json",
    {
        "passed": True,
        "planned_models": 120,
        "evaluation_rows": 240,
        "historical_replay_rows": 360,
        "primary_and_rotation_arithmetic": primary,
        "artifact_checks": artifacts,
        "fresh_relational_identical_count": sum(
            r["fresh_relational_identical_to_historical_state"] is True
            for r in artifacts
        ),
    },
)
print(
    json.dumps(
        {
            "passed": True,
            "primary_and_rotation": primary,
            "models": 120,
            "historical_rows": 360,
        },
        indent=2,
    )
)
