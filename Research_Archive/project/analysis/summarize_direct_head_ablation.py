"""Consolidate interruption-safe ablation outputs and paired seed statistics."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def readcsv(p):
    with Path(p).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def writecsv(p, rows):
    with Path(p).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def writejson(p, value):
    Path(p).write_text(json.dumps(value, indent=2), encoding="utf-8")


def consolidate(out):
    parts = [
        ROOT / "reproduction_runs/direct_head_ablation_20260909",
        ROOT / "reproduction_runs/direct_head_ablation_resume_20260909",
    ]
    resume = json.loads((parts[1] / "completion.json").read_text(encoding="utf-8"))
    assert resume["complete"] and resume["wide_models"] == 12
    assert not out.exists(), out
    progress = json.loads((parts[0] / "progress.json").read_text(encoding="utf-8"))
    assert progress["wide_models_completed"] == 48
    metadata = sum(
        [
            json.loads((p / "model_metadata.json").read_text(encoding="utf-8"))
            for p in parts
        ],
        [],
    )
    wide = {
        (int(r["dimension"]), int(r["seed"]))
        for r in metadata
        if r["condition"] == "DirectWide"
    }
    protocol = json.loads((parts[0] / "protocol.json").read_text(encoding="utf-8"))
    expected = {(d, s) for d in [32, 64] for s in protocol["protocol"]["seeds"]}
    assert (
        wide == expected and sum(r["condition"] == "DirectWide" for r in metadata) == 60
    )
    assert sum(r["condition"] == "Direct16_retrained_check" for r in metadata) == 2
    out.mkdir(parents=True)
    for name in ["per_seed.csv", "training_fit.csv", "training_curves.csv"]:
        rows = sum([readcsv(p / name) for p in parts], [])
        writecsv(out / name, rows)
    rows = readcsv(out / "per_seed.csv")
    assert len(rows) == 364
    assert (
        len({(r["dimension"], r["seed"], r["condition"], r["split"]) for r in rows})
        == 364
    )
    checkpoint_hashes = {}
    for part in parts:
        for path in part.rglob("*.pt"):
            rel = path.relative_to(part)
            dest = out / rel
            assert not dest.exists(), dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            checkpoint_hashes[rel.as_posix()] = sha(dest)
    assert len(checkpoint_hashes) == 62
    for d in [32, 64]:
        shutil.copy2(parts[0] / f"d{d}/config.json", out / f"d{d}/config.json")
    shutil.copy2(parts[0] / "protocol.json", out / "protocol.json")
    writejson(out / "model_metadata.json", metadata)
    writejson(
        out / "runtime_records.json",
        [json.loads((p / "runtime.json").read_text(encoding="utf-8")) for p in parts],
    )
    preserve = json.loads(
        (
            ROOT
            / "audit_records/engineering_20260909/ablation_interruption_resume.json"
        ).read_text(encoding="utf-8")
    )
    assert all(
        sha(ROOT / k) == v for k, v in preserve["prior_checkpoint_hashes"].items()
    )
    writejson(
        out / "completion.json",
        {
            "complete": True,
            "wide_models": 60,
            "pipeline_check_models": 2,
            "result_rows": 364,
            "checkpoints": checkpoint_hashes,
            "original_protocol_sha256": sha(parts[0] / "protocol.json"),
            "source_parts": [p.relative_to(ROOT).as_posix() for p in parts],
            "original_partial_checkpoints_unchanged": True,
            "interruption_record": "audit_records/engineering_20260909/ablation_interruption_resume.json",
        },
    )


def summaries(out, supplement=None):
    records = readcsv(out / "per_seed.csv")
    summary = []
    contrasts = []
    train = []
    rng = np.random.default_rng(20260909)
    metrics = [
        "mode_set_recall",
        "mean_matched_cosine",
        "candidate_decoy_rate",
        "proposal_recall",
        "candidate_mean_quality",
        "mean_alpha_absolute_error",
    ]
    supplement_rows = (
        readcsv(supplement / "per_seed.csv")
        if supplement and (supplement / "per_seed.csv").exists()
        else []
    )
    lookup = {
        (int(r["dimension"]), int(r["seed"]), r["condition"], r["split"]): r
        for r in records
    }
    supplementary = {
        (int(r["dimension"]), int(r["seed"]), r["condition"], r["split"]): r
        for r in supplement_rows
    }
    for d in [32, 64]:
        seeds = sorted(
            {
                int(r["seed"])
                for r in records
                if int(r["dimension"]) == d and r["condition"] == "DirectWide"
            }
        )
        assert len(seeds) == 30
        indices = rng.integers(0, 30, size=(100000, 30))
        for split in ["iid", "exact_rotated"]:
            allmetrics = metrics + (
                ["before_candidate_useful_rate", "after_candidate_useful_rate"]
                if supplement_rows
                else []
            )
            for metric in allmetrics:
                source = (
                    supplementary
                    if metric.startswith(("before_", "after_"))
                    else lookup
                )
                vectors = {
                    label: np.array(
                        [float(source[(d, s, label, split)][metric]) for s in seeds]
                    )
                    for label in [
                        "Direct16_archived",
                        "DirectWide",
                        "Relational_archived",
                    ]
                }
                assert all(np.isfinite(v).all() for v in vectors.values())
                for label, vector in vectors.items():
                    summary.append(
                        {
                            "dimension": d,
                            "condition": label,
                            "split": split,
                            "metric": metric,
                            "n_seeds": 30,
                            "mean": float(vector.mean()),
                            "sample_sd": float(vector.std(ddof=1)),
                        }
                    )
                for a, b in [
                    ("DirectWide", "Direct16_archived"),
                    ("Relational_archived", "DirectWide"),
                ]:
                    diff = vectors[a] - vectors[b]
                    boot = diff[indices].mean(axis=1)
                    lo, hi = np.quantile(boot, [0.025, 0.975])
                    contrasts.append(
                        {
                            "dimension": d,
                            "split": split,
                            "metric": metric,
                            "contrast": a + " minus " + b,
                            "n_paired_seeds": 30,
                            "mean_difference": float(diff.mean()),
                            "sample_sd_difference": float(diff.std(ddof=1)),
                            "bootstrap_lower": float(lo),
                            "bootstrap_upper": float(hi),
                            "bootstrap_replicates": 100000,
                            "bootstrap_seed": 20260909,
                        }
                    )
    fit = readcsv(out / "training_fit.csv")
    for d in [32, 64]:
        for metric in [
            "mean_matched_cosine",
            "mode_set_recall",
            "candidate_decoy_rate",
        ]:
            x = np.array(
                [
                    float(r[metric])
                    for r in fit
                    if int(r["dimension"]) == d and r["condition"] == "DirectWide"
                ]
            )
            train.append(
                {
                    "dimension": d,
                    "condition": "DirectWide",
                    "evaluation": "training tasks, zero refinement",
                    "metric": metric,
                    "n_seeds": len(x),
                    "mean": float(x.mean()),
                    "sample_sd": float(x.std(ddof=1)),
                }
            )
    replay = []
    checks = []
    for d in [32, 64]:
        old = readcsv(
            ROOT
            / f"paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d{d}/per_seed.csv"
        )
        for r in old:
            label = (
                "Direct16_archived"
                if r["condition"] == "CurrentAttention"
                else "Relational_archived"
            )
            got = lookup[(d, int(r["seed"]), label, r["split"])]
            for metric, value in r.items():
                if metric not in got or metric in [
                    "dimension",
                    "seed",
                    "condition",
                    "paper_name",
                    "split",
                ]:
                    continue
                if float(value) != float(got[metric]):
                    replay.append(
                        {
                            "dimension": d,
                            "seed": int(r["seed"]),
                            "condition": label,
                            "split": r["split"],
                            "metric": metric,
                            "old": value,
                            "new": got[metric],
                            "absolute_difference": abs(
                                float(value) - float(got[metric])
                            ),
                        }
                    )
        for split in ["iid", "exact_rotated"]:
            oldrow = lookup[(d, 7, "Direct16_archived", split)]
            newrow = lookup[(d, 7, "Direct16_retrained_check", split)]
            diffs = {m: abs(float(newrow[m]) - float(oldrow[m])) for m in metrics}
            checks.append(
                {
                    "dimension": d,
                    "seed": 7,
                    "split": split,
                    "all_reported_metrics_exact": all(
                        float(newrow[k]) == float(oldrow[k])
                        for k in newrow
                        if k not in ["dimension", "seed", "condition", "split"]
                    ),
                    "selected_metric_max_abs_difference": max(diffs.values()),
                    "differences": diffs,
                }
            )
    meta = json.loads((out / "model_metadata.json").read_text(encoding="utf-8"))
    counts = []
    for d in [32, 64]:
        for label in [
            "Direct16_archived",
            "DirectWide",
            "Relational_archived",
            "Direct16_retrained_check",
        ]:
            subset = [
                r for r in meta if int(r["dimension"]) == d and r["condition"] == label
            ]
            times = [float(r["training_seconds"]) for r in subset]
            counts.append(
                {
                    "dimension": d,
                    "condition": label,
                    "models": len(subset),
                    "parameter_counts": sorted(
                        set(int(r["parameter_count"]) for r in subset)
                    ),
                    "mean_training_seconds": float(np.mean(times)),
                    "total_training_seconds": float(np.sum(times)),
                }
            )
    writecsv(out / "summary.csv", summary)
    writecsv(out / "paired_bootstrap.csv", contrasts)
    writecsv(out / "training_fit_summary.csv", train)
    report = {
        "complete": True,
        "n_paired_seeds": 30,
        "bootstrap": {
            "replicates": 100000,
            "seed": 20260909,
            "method": "paired seed resampling, percentile 95% interval; PCG64; one shared index matrix per dimension",
        },
        "primary_endpoint": [
            r
            for r in contrasts
            if r["split"] == "iid"
            and r["metric"] == "mode_set_recall"
            and r["contrast"] == "DirectWide minus Direct16_archived"
        ],
        "parameter_counts_and_runtime": counts,
        "archive_evaluation_differences": replay,
        "archive_evaluation_values_exact": not replay,
        "narrow_seed7_training_checks": checks,
        "supplemental_candidate_fraction_included": bool(supplement_rows),
        "interpretation_limits": [
            "retrospective capacity ablation, not parameter matched",
            "same inspected task generator and canonical seeds",
            "training fit rows have zero refinement",
            "secondary intervals are descriptive, no familywise adjustment",
        ],
        "source_hashes": {
            "per_seed.csv": sha(out / "per_seed.csv"),
            "training_fit.csv": sha(out / "training_fit.csv"),
            "model_metadata.json": sha(out / "model_metadata.json"),
            "summary_program": sha(__file__),
        },
    }
    writejson(out / "SUMMARY_REPORT.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--supplement", type=Path)
    args = parser.parse_args()
    if args.consolidate:
        consolidate(args.output)
    summaries(args.output, args.supplement)


if __name__ == "__main__":
    main()
