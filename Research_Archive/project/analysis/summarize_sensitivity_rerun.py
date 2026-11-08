"""Summarize saved sensitivity-rerun records without loading models.

The script computes statistics using the standard library and NumPy, writing
to a fresh directory beneath the completed rerun. It performs arithmetic
verification of those records; it does not train or evaluate models, reconstruct
earlier macOS results, or tune models on test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (
    7,
    19,
    43,
    71,
    101,
    131,
    163,
    197,
    229,
    263,
    293,
    331,
    367,
    401,
    439,
    479,
    521,
    563,
    607,
    653,
    701,
    751,
    809,
    863,
    919,
    977,
    1031,
    1091,
    1153,
    1217,
)
DIMENSIONS = (32, 64)
METHODS = ("Direct16", "DirectWide", "Relational", "Multiplicity")
BUDGETS = (0, 3, 9, 18)
QCUTS = (0.60, 0.62, 0.65)
RCUTS = (2.0, 2.5, 3.0)
CAP_METHODS = ("amplitude_top3_repeat", "amplitude_unique8", "random_top3_repeat")
METRICS = (
    "proposal_recall",
    "mode_set_recall",
    "useful_region_count",
    "candidate_useful_rate",
    "candidate_decoy_rate",
    "candidate_anchor_rate",
    "candidate_background_rate",
    "candidate_mean_quality",
    "exact_repeat_pair_rate",
    "candidate_useful_below_threshold_rate",
)
KEY_FIELDS = (
    "dimension",
    "seed",
    "method",
    "refinement_steps",
    "quality_cutoff",
    "region_cutoff",
)
PAPER_B2 = {
    (32, "Direct16"): (0.0000, 0.4333, 0.6085, 0.6148),
    (32, "DirectWide"): (0.0000, 0.6011, 0.7700, 0.7770),
    (32, "Relational"): (0.9711, 0.9941, 0.9967, 0.9967),
    (32, "Multiplicity"): (1.0, 1.0, 1.0, 1.0),
    (64, "Direct16"): (0.0000, 0.0000, 0.0007, 0.0007),
    (64, "DirectWide"): (0.0000, 0.0000, 0.0000, 0.0015),
    (64, "Relational"): (0.9719, 0.9930, 0.9978, 0.9978),
    (64, "Multiplicity"): (1.0, 1.0, 1.0, 1.0),
}
LABELS = {
    "Direct16": "Direct 16",
    "DirectWide": "Direct wide",
    "Relational": "Relational",
    "Multiplicity": "Multiplicity",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing empty output: {path}")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) != 30 or not np.isfinite(values).all():
        raise ValueError("Every summary requires exactly 30 finite seed values")
    return {
        "mean": float(values.mean()),
        "sd_sample": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def key(row):
    return (
        int(row["dimension"]),
        int(row["seed"]),
        row["method"],
        int(row["refinement_steps"]),
        float(row["quality_cutoff"]),
        float(row["region_cutoff"]),
    )


def finite_numeric_rows(rows):
    text_fields = {"method", "cap_condition"}
    for row in rows:
        for column, value in row.items():
            if column not in text_fields and not math.isfinite(float(value)):
                raise ValueError(f"Non-finite numeric field {column}: {value}")


def validate_grid(rows, caps):
    finite_numeric_rows(rows)
    finite_numeric_rows(caps)
    expected = set(itertools.product(DIMENSIONS, SEEDS, METHODS, BUDGETS, QCUTS, RCUTS))
    observed = [key(row) for row in rows]
    if len(observed) != 8640 or len(set(observed)) != 8640 or set(observed) != expected:
        raise ValueError(
            "Sensitivity grid missing, duplicated, or outside the declared 8,640 combinations"
        )
    cap_expected = set(itertools.product(SEEDS, CAP_METHODS, ("uncapped", "cap6")))
    cap_observed = [
        (int(row["seed"]), row["method"], row["cap_condition"]) for row in caps
    ]
    if (
        len(cap_observed) != 180
        or len(set(cap_observed)) != 180
        or set(cap_observed) != cap_expected
    ):
        raise ValueError(
            "Cap grid missing, duplicated, or outside the declared 180 combinations"
        )
    for row in rows + caps:
        if int(row["n_sets"]) != 30 or int(row["n_candidates"]) != 8:
            raise ValueError("Unexpected candidate/set denominator")
    for row in rows:
        if int(row["gradient_queries_per_set"]) != int(row["refinement_steps"]) * 8:
            raise ValueError("Refinement query budget mismatch")
        validate_metrics(row, "")
    for row in caps:
        validate_metrics(row, "before_")
        validate_metrics(row, "after_")
        if (
            row["cap_condition"] == "cap6"
            and float(row["used_max_displacement"]) > 6.00001
        ):
            raise ValueError("Six-unit cap violated")
    lookup = {key(row): row for row in rows}
    monotonic_checks = 0
    for d, seed, method, step in itertools.product(DIMENSIONS, SEEDS, METHODS, BUDGETS):
        for qcut in QCUTS:
            values = [
                float(lookup[(d, seed, method, step, qcut, r)]["mode_set_recall"])
                for r in RCUTS
            ]
            if any(a > b + 1e-6 for a, b in zip(values, values[1:])):
                raise ValueError(
                    "Coverage decreases under a larger region scoring cutoff"
                )
            monotonic_checks += 1
        for rcut in RCUTS:
            values = [
                float(lookup[(d, seed, method, step, q, rcut)]["mode_set_recall"])
                for q in QCUTS
            ]
            if any(a + 1e-6 < b for a, b in zip(values, values[1:])):
                raise ValueError("Coverage increases under a stricter quality cutoff")
            monotonic_checks += 1
    return lookup, {
        "sensitivity_rows": len(rows),
        "cap_rows": len(caps),
        "grid_exact": True,
        "no_duplicate_keys": True,
        "seeds_per_condition": 30,
        "sets_per_seed": 30,
        "candidates_per_set": 8,
        "threshold_monotonicity_checks": monotonic_checks,
        "finite_numeric_fields": True,
        "fraction_partition_checks": True,
        "initial_movement_caps_pass": True,
    }


def validate_metrics(row, prefix):
    for metric in METRICS:
        value = float(row[prefix + metric])
        limit = 3.0 if metric == "useful_region_count" else 1.0
        if not -1e-6 <= value <= limit + 1e-6:
            raise ValueError(f"Out-of-range metric {prefix + metric}")
    if (
        abs(
            float(row[prefix + "useful_region_count"])
            - 3 * float(row[prefix + "mode_set_recall"])
        )
        > 2e-6
    ):
        raise ValueError("Coverage denominator must be three useful regions")
    parts = (
        "candidate_useful_rate",
        "candidate_decoy_rate",
        "candidate_anchor_rate",
        "candidate_background_rate",
        "candidate_useful_below_threshold_rate",
    )
    if abs(sum(float(row[prefix + metric]) for metric in parts) - 1.0) > 2e-6:
        raise ValueError(
            "All-candidate outcome fractions do not partition the candidate set"
        )


def summaries(lookup, caps):
    refinement = []
    for d, method, step in itertools.product(DIMENSIONS, METHODS, BUDGETS):
        row = {
            "dimension": d,
            "method": method,
            "refinement_steps": step,
            "quality_cutoff": 0.62,
            "region_cutoff": 2.5,
            "n_seeds": 30,
        }
        for metric in METRICS + ("initial_matched_cosine",):
            summary = stats(
                [float(lookup[(d, s, method, step, 0.62, 2.5)][metric]) for s in SEEDS]
            )
            row.update({f"{metric}_{name}": value for name, value in summary.items()})
        refinement.append(row)
    thresholds = []
    full_grid_summary = []
    for d, method, step in itertools.product(DIMENSIONS, METHODS, BUDGETS):
        for metric in METRICS:
            across = []
            for qcut, rcut in itertools.product(QCUTS, RCUTS):
                result = stats(
                    [
                        float(lookup[(d, s, method, step, qcut, rcut)][metric])
                        for s in SEEDS
                    ]
                )
                full_grid_summary.append(
                    {
                        "dimension": d,
                        "method": method,
                        "refinement_steps": step,
                        "quality_cutoff": qcut,
                        "region_cutoff": rcut,
                        "metric": metric,
                        "n_seeds": 30,
                        **result,
                    }
                )
                across.append(result["mean"])
            thresholds.append(
                {
                    "dimension": d,
                    "method": method,
                    "refinement_steps": step,
                    "metric": metric,
                    "n_seeds_per_threshold": 30,
                    "threshold_combinations": 9,
                    "min_seed_mean_across_thresholds": min(across),
                    "max_seed_mean_across_thresholds": max(across),
                    "range_width": max(across) - min(across),
                }
            )
    cap_lookup = {(int(r["seed"]), r["method"], r["cap_condition"]): r for r in caps}
    cap_metrics = tuple(
        f"{stage}_{metric}" for stage in ("before", "after") for metric in METRICS
    )
    cap_metrics += (
        "uncapped_max_displacement",
        "uncapped_fraction_above6",
        "used_max_displacement",
    )
    cap_summary, changes, change_rows = [], [], []
    for method in CAP_METHODS:
        for condition in ("uncapped", "cap6"):
            row = {"method": method, "cap_condition": condition, "n_seeds": 30}
            for metric in cap_metrics:
                result = stats(
                    [float(cap_lookup[(s, method, condition)][metric]) for s in SEEDS]
                )
                row.update(
                    {f"{metric}_{name}": value for name, value in result.items()}
                )
            cap_summary.append(row)
        for metric in cap_metrics:
            deltas = [
                float(cap_lookup[(s, method, "cap6")][metric])
                - float(cap_lookup[(s, method, "uncapped")][metric])
                for s in SEEDS
            ]
            changes.append(
                {
                    "method": method,
                    "metric": metric,
                    "comparison": "cap6_minus_uncapped",
                    "n_paired_seeds": 30,
                    **stats(deltas),
                    "max_abs_change": max(abs(v) for v in deltas),
                    "seeds_changed_over_1e-7": sum(abs(v) > 1e-7 for v in deltas),
                }
            )
            change_rows.extend(
                {
                    "seed": s,
                    "method": method,
                    "metric": metric,
                    "cap6_minus_uncapped": delta,
                }
                for s, delta in zip(SEEDS, deltas)
            )
    return refinement, thresholds, full_grid_summary, cap_summary, changes, change_rows


def compare_saved(lookup):
    sources = [
        (
            "matched_d32",
            ROOT
            / "paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d32/per_seed.csv",
            False,
        ),
        (
            "matched_d64",
            ROOT
            / "paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d64/per_seed.csv",
            False,
        ),
        (
            "wide_study_and_comparators",
            ROOT
            / "reproduction_runs/direct_head_ablation_complete_20260909/per_seed.csv",
            False,
        ),
        (
            "secondary_all_candidate_replay",
            ROOT / "reproduction_runs/ablation_secondary_replay_20260909/per_seed.csv",
            True,
        ),
    ]
    condition_map = {
        "CurrentAttention": "Direct16",
        "EquivariantAttention": "Relational",
        "Direct16_archived": "Direct16",
        "Relational_archived": "Relational",
        "DirectWide": "DirectWide",
    }
    details, source_info = [], []
    for label, path, secondary in sources:
        rows = read_csv(path)
        covered = set()
        for row in rows:
            if row["split"] != "iid" or row["condition"] not in condition_map:
                continue
            d = int(row.get("dimension") or label.removeprefix("matched_d"))
            if d not in DIMENSIONS:
                continue
            seed, method = int(row["seed"]), condition_map[row["condition"]]
            identity = (d, seed, method)
            if identity in covered:
                raise ValueError(f"Duplicate archived baseline identity in {path}")
            covered.add(identity)
            pairs = [("initial_matched_cosine", "mean_matched_cosine", 0)]
            if secondary:
                pairs += [
                    (metric, f"{stage}_{metric}", step)
                    for stage, step in (("before", 0), ("after", 9))
                    for metric in METRICS
                    if metric
                    not in (
                        "exact_repeat_pair_rate",
                        "candidate_useful_below_threshold_rate",
                    )
                ]
                pairs += [("exact_repeat_pair_rate", "exact_repeat_pair_rate", 0)]
            else:
                pairs += [
                    (metric, metric, 9)
                    for metric in (
                        "proposal_recall",
                        "mode_set_recall",
                        "useful_region_count",
                        "candidate_decoy_rate",
                        "candidate_anchor_rate",
                        "candidate_mean_quality",
                    )
                ]
            for metric, old_metric, step in pairs:
                if old_metric not in row:
                    raise ValueError(
                        f"Expected archived metric {old_metric} absent from {path}"
                    )
                new = float(lookup[(d, seed, method, step, 0.62, 2.5)][metric])
                old = float(row[old_metric])
                details.append(
                    {
                        "source": label,
                        "dimension": d,
                        "seed": seed,
                        "method": method,
                        "refinement_steps": step,
                        "metric": metric,
                        "archived_metric": old_metric,
                        "archived_value": old,
                        "new_cpu_value": new,
                        "new_minus_archived": new - old,
                        "absolute_difference": abs(new - old),
                    }
                )
        expected = 60 if label.startswith("matched_") else 180
        if len(covered) != expected:
            raise ValueError(
                f"{label}: expected {expected} seed/model baselines, found {len(covered)}"
            )
        source_info.append(
            {
                "label": label,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha(path),
                "iid_seed_model_rows": len(covered),
            }
        )
    difference_summaries = []
    for group in sorted(
        {(r["source"], r["metric"], r["refinement_steps"]) for r in details}
    ):
        selected = [
            r
            for r in details
            if (r["source"], r["metric"], r["refinement_steps"]) == group
        ]
        diffs = np.asarray(
            [r["new_minus_archived"] for r in selected], dtype=np.float64
        )
        worst = max(selected, key=lambda r: r["absolute_difference"])
        difference_summaries.append(
            {
                "source": group[0],
                "metric": group[1],
                "refinement_steps": group[2],
                "n_comparisons": len(selected),
                "mean_difference": float(diffs.mean()),
                "max_absolute_difference": float(np.abs(diffs).max()),
                "comparisons_over_1e-7": int(np.sum(np.abs(diffs) > 1e-7)),
                "worst_dimension": worst["dimension"],
                "worst_seed": worst["seed"],
                "worst_method": worst["method"],
            }
        )
    return details, source_info, difference_summaries


def table_b2(refinement):
    lookup = {
        (r["dimension"], r["method"], r["refinement_steps"]): r for r in refinement
    }
    rows = []
    comparisons = []
    tex = [
        r"% New Windows CPU retrospective rerun; original Mac values are comparison-only.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"$d$ & Method & 0 steps & 3 steps & 9 steps & 18 steps \\",
        r"\midrule",
    ]
    for d, method in itertools.product(DIMENSIONS, METHODS):
        row = {
            "dimension": d,
            "method": method,
            "n_seeds": 30,
            "quality_cutoff": 0.62,
            "region_cutoff": 2.5,
        }
        printed = []
        for index, step in enumerate(BUDGETS):
            r = lookup[(d, method, step)]
            value, sd = r["mode_set_recall_mean"], r["mode_set_recall_sd_sample"]
            row[f"coverage_{step}_mean"] = value
            row[f"coverage_{step}_sd_sample"] = sd
            printed.append(f"{value:.4f}")
            paper = PAPER_B2[(d, method)][index]
            comparisons.append(
                {
                    "dimension": d,
                    "method": method,
                    "refinement_steps": step,
                    "new_cpu_mean": value,
                    "paper_reported_rounded_mean": paper,
                    "new_minus_paper_rounded": value - paper,
                    "same_at_four_decimals": f"{value:.4f}" == f"{paper:.4f}",
                    "paper_value_status": "Manuscript transcription only; original Mac rows unavailable",
                }
            )
        rows.append(row)
        tex.append(f"{d} & {LABELS[method]} & " + " & ".join(printed) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    return rows, comparisons, "\n".join(tex)


def render_report(
    manifest, refinement, thresholds, changes, comparison, table_comparisons
):
    ref = {(r["dimension"], r["method"], r["refinement_steps"]): r for r in refinement}
    threshold = {
        (r["dimension"], r["method"], r["refinement_steps"], r["metric"]): r
        for r in thresholds
    }
    lines = [
        "# Paper B: rerun of the three missing follow-up checks",
        "",
        "## What was done",
        "",
        "Existing saved models were evaluated again on the original reconstructed test tasks. "
        "No models were trained, no checkpoint was replaced, and no settings were adjusted to obtain a desired result. "
        "This is a new Windows CPU run, not recovery of the unavailable Mac supplement and not a fresh independent benchmark.",
        "",
        "The full grid contains 8,640 seed-level rows: two dimensions, four methods, 30 seeds, "
        "four refinement budgets, and nine scoring-cutoff pairs. A separate movement-cap comparison "
        "contains 180 rows for three selectors, 30 seeds and two cap conditions. "
        "Each seed contains 30 sets of eight candidates. Uncertainty below uses sample SD across seeds.",
        "",
        "## 1. Does extra local improvement change the conclusions?",
        "",
        "Coverage is the fraction of the three useful regions reached by the eight candidates. "
        "Zero steps measures the model or selector alone; later columns add ordinary local optimization.",
        "",
        "| Dimension | Method | 0 steps | 3 steps | 9 steps | 18 steps |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for d, method in itertools.product(DIMENSIONS, METHODS):
        values = [ref[(d, method, b)]["mode_set_recall_mean"] for b in BUDGETS]
        lines.append(
            f"| {d} | {LABELS[method]} | "
            + " | ".join(f"{v:.4f}" for v in values)
            + " |"
        )
    lines += [
        "",
        "These are fixed finite budgets, not convergence tests. In particular, low direct-model "
        "coverage before refinement must remain visible: much of its eventual coverage may come from the local optimizer.",
        "",
    ]
    for d in DIMENSIONS:
        for method in ("Direct16", "DirectWide", "Relational"):
            first = ref[(d, method, 0)]["mode_set_recall_mean"]
            nine = ref[(d, method, 9)]["mode_set_recall_mean"]
            eighteen = ref[(d, method, 18)]["mode_set_recall_mean"]
            lines.append(
                f"- {d}D {LABELS[method]}: zero to nine steps changes coverage by "
                f"{100 * (nine - first):+.2f} percentage points; nine to eighteen adds "
                f"{100 * (eighteen - nine):+.2f} points."
            )
    lines += [
        "",
        "## 2. Does changing the success definition change the ordering?",
        "",
        "The same candidate endpoints were rescored using quality cutoffs 0.60/0.62/0.65 "
        "and distance cutoffs 2.0/2.5/3.0. Scoring changes never changed the underlying landscape or optimization path.",
        "",
        "| Dimension | Method | Nine-step coverage range across scoring rules |",
        "|---|---|---:|",
    ]
    for d, method in itertools.product(DIMENSIONS, METHODS):
        r = threshold[(d, method, 9, "mode_set_recall")]
        lines.append(
            f"| {d} | {LABELS[method]} | {r['min_seed_mean_across_thresholds']:.6f}–{r['max_seed_mean_across_thresholds']:.6f} |"
        )
    lines += [
        "",
        "The tested cutoff range is limited. Stability within it is not evidence that every possible definition of usefulness gives the same result.",
        "",
        "## 3. Does equal travel distance alter the simple-selector result?",
        "",
        "Each capped and uncapped pair chooses exactly the same references. Only the initial move is "
        "shortened, if needed, to six units. The nine-step refinement afterwards is unchanged. "
        "The random selector is paired within this CPU rerun; its draws are not assumed identical to the old GPU run.",
        "",
        "| Selector | Mean change in after-refinement coverage | Mean change in useful-candidate fraction | Mean quality change |",
        "|---|---:|---:|---:|",
    ]
    cmap = {(r["method"], r["metric"]): r for r in changes}
    for method in CAP_METHODS:
        vals = [
            cmap[(method, metric)]["mean"]
            for metric in (
                "after_mode_set_recall",
                "after_candidate_useful_rate",
                "after_candidate_mean_quality",
            )
        ]
        lines.append(f"| {method} | " + " | ".join(f"{v:+.8f}" for v in vals) + " |")
    important = [
        r
        for r in changes
        if r["metric"]
        in (
            "before_mode_set_recall",
            "after_mode_set_recall",
            "before_candidate_useful_rate",
            "after_candidate_useful_rate",
            "before_candidate_decoy_rate",
            "after_candidate_decoy_rate",
        )
    ]
    changed = [r for r in important if r["max_abs_change"] > 1e-7]
    if changed:
        lines += [
            "",
            "At least one seed-level coverage/usefulness/decoy value changed by more than 1e-7. "
            "The previous blanket statement that these metrics are unchanged is not assumed; see paired_selector_cap_changes.csv.",
        ]
    else:
        lines += [
            "",
            "No before/after coverage, usefulness or decoy value changed by more than 1e-7 under the cap. "
            "This does not imply quality or travel distances were unchanged; those are reported separately.",
        ]
    lines += ["", "## Comparison with the paper and archived GPU rows", ""]
    matching = sum(r["same_at_four_decimals"] for r in table_comparisons)
    lines.append(
        f"{matching} of 32 Table B2 cells match the manuscript at four decimals. "
        "Those manuscript numbers are comparison-only transcriptions, never substitutes for new measurements."
    )
    lines.append(
        "The archived matched study, complete wider-head study, and all-candidate replay were compared "
        "separately. Their overlapping models are not counted as independent evidence."
    )
    for metric in (
        "mode_set_recall",
        "candidate_decoy_rate",
        "candidate_useful_rate",
        "candidate_mean_quality",
    ):
        selected = [
            r
            for r in comparison["difference_summaries"]
            if r["metric"] == metric and r["refinement_steps"] == 9
        ]
        if selected:
            largest = max(r["max_absolute_difference"] for r in selected)
            lines.append(
                f"- Largest absolute nine-step CPU-versus-archived difference for `{metric}`: {largest:.10g}."
            )
    lines += [
        "",
        "Every individual difference is retained in baseline_differences.csv. Differences in floating-point "
        "arithmetic and random-number generation are reported, not silently rounded away.",
        "",
        "## Interpretation and limitations",
        "",
        "These checks separate the initial proposal, finite local improvement and metric definition. "
        "They do not establish broad optimization superiority. Multiplicity selection exploits duplicated "
        "coordinates in this particular supplied-reference generator; a strong result for this simple rule "
        "remains contrary evidence to claims that learned reference selection is necessary here.",
        "",
        "The same previously inspected task instances and saved models are reused. There is no new training, "
        "no independent held-out confirmation and no reconstruction of missing original Mac execution logs. "
        "The current paper and original result files were not edited.",
        "",
        "## Verification and runtime",
        "",
        f"- Run status: {manifest['status']}",
        f"- Training steps: {manifest['training_steps']}",
        f"- Checkpoint mutation check: {manifest['mutation_check']}",
        f"- Checkpoints checked: {manifest['checkpoint_count']}",
        f"- Runtime: {json.dumps(manifest['runtime'], ensure_ascii=True)}",
        "",
        "See validation.json for grid, denominator and scoring invariants. "
        "See summary_manifest.json for the hashes of this arithmetic program and its inputs.",
        "",
    ]
    return "\n".join(lines)


def run(run_path, output=None):
    run_path = Path(run_path).resolve(strict=True)
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not str(manifest.get("status", "")).startswith("COMPLETE"):
        raise RuntimeError("Refusing to summarize an incomplete evaluation")
    if manifest.get("checkpoint_count") != 180 or manifest.get("training_steps") != 0:
        raise ValueError("Run checkpoint count or zero-training contract violated")
    if (
        manifest.get("mutation_check") != "PASS"
        or manifest["checkpoint_hashes_before"] != manifest["checkpoint_hashes_after"]
    ):
        raise ValueError("Checkpoint integrity failed")
    protocol_path = run_path / "protocol.json"
    if sha(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))["protocol"]
    for name, expected in (
        ("dimensions", DIMENSIONS),
        ("seeds", SEEDS),
        ("methods", METHODS),
    ):
        if tuple(protocol[name]) != expected:
            raise ValueError(f"Unexpected protocol {name}")
    sensitivity_path, caps_path = (
        run_path / "sensitivity_per_seed.csv",
        run_path / "selector_budget_per_seed.csv",
    )
    rows, caps = read_csv(sensitivity_path), read_csv(caps_path)
    lookup, validation = validate_grid(rows, caps)
    refinement, thresholds, all_thresholds, cap_summary, changes, paired = summaries(
        lookup, caps
    )
    details, source_info, difference_summaries = compare_saved(lookup)
    b2, table_comparisons, tex = table_b2(refinement)
    comparison = {
        "qualification": "New Windows CPU versus archived GPU evaluations; missing Mac rows are not available for direct comparison",
        "shared_model_warning": "Overlapping comparator files are separate arithmetic checks, not independent replications",
        "sources": source_info,
        "difference_summaries": difference_summaries,
        "paper_table_B2_comparison": table_comparisons,
    }
    output = Path(output).resolve() if output else run_path / "summary"
    if not output.is_relative_to(run_path) or output == run_path:
        raise ValueError("Summary output must be a fresh subdirectory of this run")
    output.mkdir(parents=True, exist_ok=False)
    for filename, records in (
        ("refinement_summary.csv", refinement),
        ("threshold_range_summary.csv", thresholds),
        ("full_threshold_summary.csv", all_thresholds),
        ("selector_budget_summary.csv", cap_summary),
        ("paired_selector_cap_changes.csv", changes),
        ("paired_selector_cap_per_seed.csv", paired),
        ("table_B2_reproduction.csv", b2),
        ("table_B2_paper_comparison.csv", table_comparisons),
        ("baseline_differences.csv", details),
    ):
        write_csv(output / filename, records)
    (output / "table_B2_reproduction.tex").write_text(tex, encoding="utf-8")
    write_json(output / "baseline_comparison.json", comparison)
    write_json(output / "validation.json", validation)
    report = render_report(
        manifest, refinement, thresholds, changes, comparison, table_comparisons
    )
    (output / "RERUN_REPORT.md").write_text(report, encoding="utf-8")
    files = sorted(p for p in output.iterdir() if p.is_file())
    write_json(
        output / "summary_manifest.json",
        {
            "status": "COMPLETE arithmetic summaries of completed rerun",
            "analysis_sha256": sha(__file__),
            "numpy_version": np.__version__,
            "run_relative_to_package": run_path.relative_to(ROOT).as_posix(),
            "input_hashes": {
                p.relative_to(run_path).as_posix(): sha(p)
                for p in (manifest_path, protocol_path, sensitivity_path, caps_path)
            },
            "archived_comparison_sources": source_info,
            "output_hashes": {p.name: sha(p) for p in files},
            "training_steps": 0,
            "evaluation_steps": 0,
            "arithmetic_only": True,
            "validation": validation,
        },
    )
    print(
        json.dumps(
            {
                "summary_directory": str(output),
                "sensitivity_rows": 8640,
                "cap_rows": 180,
                "paper_B2_cells_matching_at_four_decimals": sum(
                    r["same_at_four_decimals"] for r in table_comparisons
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="Fresh subdirectory of --run (default: summary)"
    )
    args = parser.parse_args()
    run(args.run, args.output)
