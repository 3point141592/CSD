"""Read-only independent CSV recomputation; no model loading or source writes."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
)
_parser.add_argument("--output-dir", type=Path)
_args = _parser.parse_args()
B = _args.package_root.resolve(strict=True)
O = (_args.output_dir or B / "new_verification/audit_20260909").resolve()
O.mkdir(parents=True, exist_ok=True)
for _name in ["recomputed_statistics.json"]:
    if (O / _name).exists():
        raise FileExistsError(
            f"Refusing to replace {_name}; choose a fresh --output-dir"
        )


def read(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stats(v):
    a = np.array(v, dtype=float)
    return dict(
        n=len(a),
        mean=float(a.mean()),
        sd=float(a.std(ddof=1)),
        min=float(a.min()),
        max=float(a.max()),
    )


def paired(a, b):
    x = np.array(a, dtype=float) - np.array(b, dtype=float)
    rng = np.random.default_rng(20260909)
    boot = x[rng.integers(0, len(x), size=(100000, len(x)))].mean(axis=1)
    return dict(
        **stats(x),
        bootstrap_percentile_95=np.quantile(boot, [0.025, 0.975]).tolist(),
        positive=int((x > 1e-7).sum()),
        negative=int((x < -1e-7).sum()),
        zero=int((abs(x) <= 1e-7).sum()),
    )


fair = B / "new_verification/fair_selector_d32"
ps = read(fair / "per_seed.csv")
sets = read(fair / "per_set.csv")
saved = read(fair / "summary.csv")
by = defaultdict(list)
for r in ps:
    by[r["method"]].append(r)
checks = []
for r in saved:
    sub = by[r["method"]]
    for k in r:
        for suffix, f in [
            ("_mean", np.mean),
            ("_sd_sample", lambda a: np.std(a, ddof=1)),
            ("_min", np.min),
            ("_max", np.max),
        ]:
            if k.endswith(suffix):
                metric = k[: -len(suffix)]
                checks.append(
                    abs(float(r[k]) - float(f([float(s[metric]) for s in sub])))
                )
out = {
    "fair_summary_checks": len(checks),
    "fair_summary_max_abs_error": max(checks),
    "fair_rows": len(ps),
    "fair_set_rows": len(sets),
    "sets_per_method_seed": sorted(
        set(
            sum(r["seed"] == p["seed"] and r["method"] == p["method"] for r in sets)
            for p in ps
        )
    ),
}
metrics = [
    "before_mode_set_recall",
    "after_proposal_recall",
    "after_mode_set_recall",
    "after_candidate_useful_rate",
    "after_candidate_decoy_rate",
    "after_candidate_mean_quality",
    "exact_repeat_pair_rate",
]
out["fair_recomputed"] = {
    m: {k: stats([r[k] for r in rows]) for k in metrics} for m, rows in by.items()
}
csd = sorted(by["relational_csd"], key=lambda r: int(r["seed"]))
amp = sorted(by["amplitude_top3_repeat"], key=lambda r: int(r["seed"]))
out["amplitude_minus_relational_paired"] = {
    k: paired([r[k] for r in amp], [r[k] for r in csd]) for k in metrics
}
set_checks = []
for p in ps:
    rows = [r for r in sets if r["seed"] == p["seed"] and r["method"] == p["method"]]
    for k in rows[0]:
        if k not in ["seed", "method", "set_index"]:
            set_checks.append(abs(float(p[k]) - np.mean([float(r[k]) for r in rows])))
out["fair_set_to_seed_max_abs_error"] = float(max(set_checks))
out["relational_coverage_failures"] = [
    r
    for r in sets
    if r["method"] == "relational_csd" and float(r["after_mode_set_recall"]) < 0.999999
]
out["scaling"] = {}
for d in [2, 4, 8, 16, 32, 64]:
    rows = read(
        B
        / f"original_results/recovered_attachment/results/matched_scaling/d{d}/per_seed.csv"
    )
    out["scaling"][d] = {}
    for condition in ["CurrentAttention", "EquivariantAttention"]:
        rs = [r for r in rows if r["condition"] == condition and r["split"] == "iid"]
        out["scaling"][d][condition] = {
            k: stats([r[k] for r in rs])
            for k in [
                "mean_matched_cosine",
                "proposal_recall",
                "mode_set_recall",
                "candidate_decoy_rate",
            ]
        }
    a = sorted(
        [
            r
            for r in rows
            if r["condition"] == "EquivariantAttention" and r["split"] == "iid"
        ],
        key=lambda r: int(r["seed"]),
    )
    b = sorted(
        [
            r
            for r in rows
            if r["condition"] == "CurrentAttention" and r["split"] == "iid"
        ],
        key=lambda r: int(r["seed"]),
    )
    out["scaling"][d]["coverage_paired"] = paired(
        [r["mode_set_recall"] for r in a], [r["mode_set_recall"] for r in b]
    )
reach = read(B / "new_verification/corruption_reachability_d32/per_set.csv")
reachseed = read(B / "new_verification/corruption_reachability_d32/per_seed.csv")
out["cone_case_stats"] = stats([r["cone_max_cosine"] for r in reach])
out["cone_seed_means_stats"] = stats([r["cone_mean"] for r in reachseed])
(O / "recomputed_statistics.json").write_text(
    json.dumps(out, indent=2), encoding="utf8"
)
print(
    json.dumps(
        {
            k: out[k]
            for k in [
                "fair_summary_checks",
                "fair_summary_max_abs_error",
                "fair_rows",
                "fair_set_rows",
                "sets_per_method_seed",
                "fair_set_to_seed_max_abs_error",
                "amplitude_minus_relational_paired",
                "relational_coverage_failures",
                "cone_case_stats",
                "cone_seed_means_stats",
            ]
        },
        indent=2,
    )
)
