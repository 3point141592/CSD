"""Automatic fixed analysis; seed is the independent training replicate."""

import numpy as np
from experiment import (
    MODELS,
    OUT,
    check_lock,
    job_valid,
    matched_config,
    now,
    read_json,
    write_csv,
    write_json,
)

ENDPOINTS = [
    "mode_set_recall",
    "mean_matched_cosine",
    "proposal_recall",
    "useful_candidate_fraction",
    "candidate_decoy_rate",
    "train_loss",
    "train_mean_matched_cosine",
    "output_raw_rank",
    "output_direction_rank",
    "parameter_count",
    "optimizer_seconds",
    "job_wall_seconds",
    "training_peak_allocated_bytes",
    "training_peak_reserved_bytes",
]


def analyze(require_complete=True):
    check_lock()
    rows = []
    missing = []
    historical = []
    curves = []
    manifest = read_json(OUT / "manifest.json")
    for d in (32, 64):
        for seed in matched_config(d).seeds:
            for model in MODELS:
                folder = OUT / "jobs" / f"d{d}" / f"seed_{seed}" / model
                key = f"d{d}_s{seed}_{model}"
                if job_valid(folder, key):
                    rows.extend(read_json(folder / "metrics.json"))
                    curves.extend(
                        {"dimension": d, "seed": seed, "model": model, **r}
                        for r in read_json(folder / "curves.json")
                    )
                else:
                    status = manifest["jobs"][key]["status"]
                    missing.append(
                        {"dimension": d, "seed": seed, "model": model, "status": status}
                    )
                    rows.extend(
                        {
                            "dimension": d,
                            "seed": seed,
                            "model": model,
                            "split": split,
                            "status": status,
                        }
                        for split in ("iid", "exact_rotated")
                    )
            hist = OUT / "historical" / f"d{d}_s{seed}.json"
            if hist.exists():
                historical.extend(read_json(hist)["rows"])
    write_csv(OUT / "per_seed.csv", rows)
    write_csv(OUT / "training_curves.csv", curves)
    write_csv(OUT / "historical_context.csv", historical)
    write_json(
        OUT / "analysis_completeness.json",
        {
            "utc": now(),
            "expected_models": 120,
            "completed_models": 120 - len(missing),
            "missing": missing,
        },
    )
    failed_attempts = [
        {"attempt_directory": str(p.parent.relative_to(OUT)), **read_json(p)}
        for p in (OUT / "attempts").glob("*/failure.json")
    ]
    write_json(OUT / "failed_attempts.json", failed_attempts)
    if require_complete:
        assert not missing, ("Not all planned results are available", missing)
    summary = []
    for d in (32, 64):
        for split in ("iid", "exact_rotated"):
            look = {
                (r["seed"], r["model"]): r
                for r in rows
                if r["dimension"] == d
                and r["split"] == split
                and r["status"] == "complete"
            }
            seeds = [
                s
                for s in matched_config(d).seeds
                if all((s, m) in look for m in MODELS)
            ]
            if not seeds:
                continue
            n = len(seeds)
            rng = np.random.default_rng(20260921 + d)
            indices = rng.integers(0, n, size=(20000, n))
            for metric in ENDPOINTS:
                a = np.array(
                    [look[s, "DirectPrepared"][metric] for s in seeds], dtype=np.float64
                )
                b = np.array(
                    [look[s, "Relational"][metric] for s in seeds], dtype=np.float64
                )
                for label, values in [
                    ("DirectPrepared", a),
                    ("Relational", b),
                    ("DirectPrepared_minus_Relational", a - b),
                ]:
                    boot = values[indices].mean(1)
                    lo, hi = np.quantile(boot, [0.025, 0.975])
                    summary.append(
                        {
                            "dimension": d,
                            "split": split,
                            "endpoint": metric,
                            "comparison": label,
                            "n_paired_seeds": n,
                            "expected_seeds": 30,
                            "status": "complete"
                            if n == 30
                            else "provisional_incomplete",
                            "mean": float(values.mean()),
                            "sample_sd": float(values.std(ddof=1)) if n > 1 else "",
                            "ci95_low": float(lo) if n > 1 else "",
                            "ci95_high": float(hi) if n > 1 else "",
                            "ci_method": "paired_seed_percentile_bootstrap_20000",
                            "analysis_rng_seed": 20260921 + d,
                        }
                    )
    write_csv(OUT / "comparison_summary.csv", summary)
    hs = []
    for d in (32, 64):
        for split in ("iid", "exact_rotated"):
            for kind in ("CurrentAttention", "DirectWide", "EquivariantAttention"):
                selected = [
                    r
                    for r in historical
                    if r["dimension"] == d
                    and r["split"] == split
                    and r["condition"] == kind
                ]
                for metric in (
                    "mode_set_recall",
                    "mean_matched_cosine",
                    "proposal_recall",
                    "useful_candidate_fraction",
                    "candidate_decoy_rate",
                    "output_raw_rank",
                    "parameter_count",
                ):
                    if selected:
                        x = np.asarray([r[metric] for r in selected], dtype=float)
                        hs.append(
                            {
                                "dimension": d,
                                "split": split,
                                "condition": kind,
                                "endpoint": metric,
                                "n": len(x),
                                "mean": float(x.mean()),
                                "sample_sd": float(x.std(ddof=1)) if len(x) > 1 else "",
                                "context": "historical_saved_checkpoint_replay",
                            }
                        )
    write_csv(OUT / "historical_summary.csv", hs)
    if missing:
        report = f"Experiment in progress: {120 - len(missing)}/120 planned models completed.\n\nNo final conclusion is available. `per_seed.csv` retains all planned rows; unfinished and failed jobs have blank metrics, never zero coverage. `comparison_summary.csv` is explicitly provisional. The final report will replace this progress note only after all planned jobs pass completeness checks.\n\nThe manuscript has not been edited. See PLAN.md, manifest.json and the worker logs.\n"
    else:

        def item(d, split, metric, kind):
            return next(
                r
                for r in summary
                if r["dimension"] == d
                and r["split"] == split
                and r["endpoint"] == metric
                and r["comparison"] == kind
            )

        report = "All 120 planned final models completed under the fixed retrospective follow-up protocol. No seeds or checkpoints were selected by performance, and the manuscript was not edited.\n\n"
        if failed_attempts:
            report += f"There were {len(failed_attempts)} failed attempts, preserved in attempts/ and failed_attempts.json. The first pilot required a metadata serialization repair before safe checkpoint reload; its fixed seed and training settings were repeated. Failed attempts were never entered as zero coverage.\n\n"
        report += "IID useful-region coverage (mean across 30 independently trained seeds; difference is DirectPrepared minus contemporaneous Relational):\n\n| Dimension | Direct prepared | Relational | Paired difference | 95% paired-seed CI | SD of paired differences |\n|---|---:|---:|---:|---:|---:|\n"
        for d in (32, 64):
            a = item(d, "iid", "mode_set_recall", "DirectPrepared")
            b = item(d, "iid", "mode_set_recall", "Relational")
            diff = item(d, "iid", "mode_set_recall", "DirectPrepared_minus_Relational")
            report += f"| {d} | {a['mean']:.4f} | {b['mean']:.4f} | {diff['mean']:+.4f} | [{diff['ci95_low']:+.4f}, {diff['ci95_high']:+.4f}] | {diff['sample_sd']:.4f} |\n"
        report += "\nThese 95% percentile bootstrap intervals resample paired training seeds, not starts or candidates. SD measures variability between seeds; it is not the confidence interval. Dimensions are analyzed separately; intervals have no familywise adjustment. Full-precision values and every planned seed are in the CSVs.\n\n"
        for d in (32, 64):
            diff = item(d, "iid", "mode_set_recall", "DirectPrepared_minus_Relational")
            rot = item(
                d, "exact_rotated", "mode_set_recall", "DirectPrepared_minus_Relational"
            )
            if diff["ci95_high"] < 0:
                verdict = "Relational retains an IID coverage advantage over this specified stronger baseline."
            elif diff["ci95_low"] > 0:
                verdict = "Prepared-feature Direct has higher IID coverage in this comparison."
            else:
                verdict = "The IID paired interval includes zero; this alone does not establish equivalence."
            report += f"At {d}D, {verdict} The exact-rotation coverage difference is {rot['mean']:+.4f} (95% CI [{rot['ci95_low']:+.4f}, {rot['ci95_high']:+.4f}]). "
            for model in MODELS:
                rank = item(d, "iid", "output_raw_rank", model)
                params = item(d, "iid", "parameter_count", model)
                report += f"{model}: {int(params['mean']):,} parameters; mean measured raw-output rank {rank['mean']:.2f}. "
            report += "\n\n"
        report += "Historical checkpoint context, re-evaluated on the same test records:\n\n| Dimension | Original Direct IID | Wider-head Direct IID | Historical matched Relational IID |\n|---|---:|---:|---:|\n"
        for d in (32, 64):
            means = [
                next(
                    r["mean"]
                    for r in hs
                    if r["dimension"] == d
                    and r["split"] == "iid"
                    and r["condition"] == kind
                    and r["endpoint"] == "mode_set_recall"
                )
                for kind in ("CurrentAttention", "DirectWide", "EquivariantAttention")
            ]
            report += f"| {d} | {means[0]:.4f} | {means[1]:.4f} | {means[2]:.4f} |\n"
        report += "\nAny substantial closing of the old gap shows that its size depended on the restricted original implementation. Comparable observed scores do not by themselves establish statistical equivalence. Where Relational remains better, the defensible claim is an advantage over this specified stronger baseline under this fixed protocol; other architecture and optimization differences remain.\n\n"
        report += "Both models receive benchmark-supplied reference centers. Direct receives the same invariant descriptors and relative vectors, but retains absolute-state inputs and a learned coordinate decoder. Relational uses its relative-vector output. Their widths and architectures differ, so this is neither parameter matched nor a sole-output-rule causal test. Any improvement over historical Direct combines feature and width changes. The results concern this inspected synthetic task family; they do not establish a general Relational advantage or improved LLM reasoning.\n\n"
        report += "Historical checkpoints are labeled separately in historical_context.csv and historical_summary.csv. Saved-model evaluation is distinct from retraining. The reported CPU retraining coverage 0.8333 versus archived seed-7 coverage 0.6333 is not rounding; the exploratory prototype/checkpoint was not present in the searched local folder/archives, so its causal explanation remains unverified. See verification/historical_replay_cpu.json and historical_replay_cuda.json for actual saved-model replay results and numerical deltas.\n\n"
        timing = [read_json(p) for p in (OUT / "timing").glob("*.json")]
        if timing:
            report += f"Measured sum of pair wall times (including data/features, training, diagnostics, historical replay and saving): {sum(r['pair_wall_seconds'] for r in timing) / 3600:.2f} hours. Peak sampled host RSS: {max(r['peak_process_rss_bytes'] for r in timing) / 1024**3:.2f} GiB; peak CUDA allocation: {max(r['peak_gpu_allocated_bytes'] for r in timing) / 1024**3:.2f} GiB. Runtime versions, TF32 settings, per-job time/memory and pilot estimate are saved in runtime.json, pilot.json and timing/.\n\n"
        report += "After verification, update follow-up methods/results, the abstract, Section 7.1, the conclusion and evidence availability. Explain the stronger baseline, retrospective design, benchmark references, architecture differences, dimension-specific outcomes and the limits on causal interpretation. Do not insert the historical CPU probe as a reproduced result.\n\nAll configurations, frozen code hashes, exact task/start/rotation arrays, feature hashes, final checkpoints, curves, checks and reproduction commands are retained here. See COMMANDS.md. This completes the fixed experiment; no further experimental expansion was launched.\n"
    (OUT / "RESULTS_FOR_KRIVAN.md").write_text(report, encoding="utf-8")
    return summary


if __name__ == "__main__":
    analyze(require_complete=True)
