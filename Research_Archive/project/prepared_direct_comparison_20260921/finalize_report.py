"""Write a concise report describing the completed prepared-feature comparison."""

import csv
import shutil
from pathlib import Path

from experiment import OUT, atomic_replace, read_json, sha, write_json

validation = read_json(OUT / "verification/final_independent_validation.json")
assert validation["passed"]
rows = list(csv.DictReader((OUT / "comparison_summary.csv").open(encoding="utf-8")))


def get(d, split, metric, model):
    r = next(
        r
        for r in rows
        if int(r["dimension"]) == d
        and r["split"] == split
        and r["endpoint"] == metric
        and r["comparison"] == model
    )
    assert int(r["n_paired_seeds"]) == 30
    return {k: float(r[k]) for k in ["mean", "sample_sd", "ci95_low", "ci95_high"]}


for d in (32, 64):
    x = get(d, "iid", "mode_set_recall", "DirectPrepared_minus_Relational")
    assert x["ci95_low"] < 0 < x["ci95_high"] and abs(x["mean"]) < 0.01, (
        "Review interpretation for changed results"
    )
    assert (
        get(d, "exact_rotated", "mode_set_recall", "DirectPrepared_minus_Relational")[
            "ci95_high"
        ]
        < -0.30
    )
text = """The large IID useful-region coverage gap is gone in this comparison. Relational retains a large advantage after exact rotation.

All 120 planned final models completed: two models, two dimensions, and the original 30 paired seeds. This was a fixed retrospective follow-up on an already inspected synthetic task family, not an independent preregistered discovery. The manuscript was not edited.

Each row below uses 30 paired training seeds. Differences are prepared-feature Direct minus contemporaneous Relational.

| Evaluation | Direct | Relational | Paired difference | 95% paired CI | SD of paired differences |
|---|---:|---:|---:|---:|---:|
"""
for split, label in [("iid", "IID"), ("exact_rotated", "Exact rotation")]:
    for d in (32, 64):
        a = get(d, split, "mode_set_recall", "DirectPrepared")
        b = get(d, split, "mode_set_recall", "Relational")
        delta = get(d, split, "mode_set_recall", "DirectPrepared_minus_Relational")
        text += f"| {label}, {d}D | {a['mean']:.4f} | {b['mean']:.4f} | {delta['mean']:+.4f} | [{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] | {delta['sample_sd']:.4f} |\n"
text += """
Intervals use 20,000 bootstrap resamples of paired seeds, separately by dimension, without multiplicity adjustment. Starts and candidates are not training replicates. SD describes seed variability, not a confidence interval. Similar IID scores do not establish formal equivalence; performance is near the benchmark ceiling.

Secondary IID results are mixed. In the table, each cell is Direct / Relational; complete SDs and confidence intervals are in comparison_summary.csv.

| Metric | 32D | 64D |
|---|---:|---:|
"""
for metric, label, fmt in [
    ("mean_matched_cosine", "Initial matched cosine", ".4f"),
    ("proposal_recall", "Any-useful success", ".4f"),
    ("useful_candidate_fraction", "Useful-candidate fraction", ".4f"),
    ("candidate_decoy_rate", "Decoy fraction", ".4f"),
    ("train_loss", "Final training loss", ".7f"),
]:
    values = []
    for d in (32, 64):
        a = get(d, "iid", metric, "DirectPrepared")["mean"]
        b = get(d, "iid", metric, "Relational")["mean"]
        values.append(format(a, fmt) + " / " + format(b, fmt))
    text += f"| {label} | {values[0]} | {values[1]} |\n"
text += """
Relational produces a higher useful-candidate fraction at both dimensions. The paired decoy contrast is clear at 32D; its 64D interval includes zero. Direct fits the training objective better. Thus, closing the coverage gap does not mean every metric is equal.

Direct has 152,740 / 189,700 parameters at 32D / 64D; Relational has 44,665 / 148,117. Measured IID raw-output ranks average 32 / 64 for Direct and 32 / 41.23 for Relational. These are output diagnostics, not capacity bounds or quality scores; spectra are saved.

For Paper B, the original IID gap depended on the restricted original Direct implementation. Historical original Direct averaged 0.6085 / 0.0007 coverage; wider-head Direct averaged 0.7700 / 0.0000. Those checkpoints were re-evaluated separately and are not the primary comparison.

The defensible surviving claim is better rotation transfer, plus some candidate-quality advantages, for this specified Relational architecture under this protocol. Both models receive benchmark-supplied reference centers. Direct retains absolute-state inputs and a learned coordinate decoder; architectures and parameter counts differ. Features and widths changed together relative to historical Direct, so neither a feature-only effect nor the output rule as sole cause is identified. These results do not establish universal superiority or improved LLM reasoning.

Independent verification passed for all checkpoints, curves, normalization hashes and paired intervals. Saved-model CPU replay reproduces archived seed-7 coverage of 0.6333. The reported 0.8333 CPU retraining result remains untraced because its probe/checkpoint was not found; that gap is not rounding.

One pilot retry produced identical weights and curves. Diagnostic and file-write repairs preserved completed results and left numerical training/evaluation unchanged. Failures and source versions are documented in RUN_RECORD.md; no failure became zero coverage.

"""
timing = [read_json(p) for p in (OUT / "timing").glob("*.json")]
assert len(timing) == 60
pilot = read_json(OUT / "pilot.json")
text += f"The full pilot took {pilot['total_pilot_wall_seconds']:.1f} seconds and estimated {pilot['remaining_estimated_seconds'] / 60:.1f} minutes remaining. Total measured pair work was {sum(x['pair_wall_seconds'] for x in timing) / 60:.1f} minutes; waiting, repairs and external GPU contention added elapsed time. Peak CUDA allocation / host RSS: {max(x['peak_gpu_allocated_bytes'] for x in timing) / 1024**3:.2f} / {max(x['peak_process_rss_bytes'] for x in timing) / 1024**3:.2f} GiB. Environment: RTX 5090, PyTorch 2.12.1+cu132, float32, TF32 disabled. Per-model costs remain in the CSVs and metadata.\n\n"
text += """After verification, the manuscript sections needing updates are:

- Follow-up methods and results: this fixed stronger baseline, all paired results, retrospective status and benchmark-supplied references.
- Abstract and conclusion: remove any general IID superiority interpretation; state the narrower rotation-transfer result and remaining differences.
- Section 7.1: keep the head-only study as historical context and distinguish it from this joint feature/width follow-up.
- Evidence availability: include this directory's plan, source/configuration hashes, exact data, final checkpoints, curves, CSVs and verification records.

All numbers are retained without display rounding in per_seed.csv and comparison_summary.csv. See COMMANDS.md and RUN_RECORD.md for exact reproduction and process-control commands. The fixed comparison is complete; no further experiments, manuscript edits, publishing or submission were performed.
"""
original = OUT / "RESULTS_FOR_KRIVAN.md"
backup = OUT / "RESULTS_AUTOGENERATED.md"
if not backup.exists():
    shutil.copy2(original, backup)
temp = OUT / "RESULTS_FOR_KRIVAN.md.final.tmp"
temp.write_text(text, encoding="utf-8")
atomic_replace(temp, original)
write_json(
    OUT / "report_manifest.json",
    {
        "report_sha256": sha(original),
        "per_seed_sha256": sha(OUT / "per_seed.csv"),
        "comparison_summary_sha256": sha(OUT / "comparison_summary.csv"),
        "independent_validation_sha256": sha(
            OUT / "verification/final_independent_validation.json"
        ),
        "report_generator_sha256": sha(Path(__file__)),
        "report_scope": "reviewed interpretation of the completed original fixed comparison",
    },
)
print("Reviewed report written; all numerical tables read from verified CSVs.")
