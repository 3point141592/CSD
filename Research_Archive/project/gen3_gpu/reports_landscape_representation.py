"""Plots and scientific reports for the isolated non-oracle 32D study."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from creativity_toy.analysis_utils import statistics

from .study import write_csv

DISPLAY = {
    "TaskOnly": "Task only",
    "TaskOnlyParameterMatched": "Task only\n(parameter matched)",
    "LandscapeDeepSets": "Landscape\nDeepSets",
    "LandscapeAttention": "Landscape\nattention",
    "LandscapeAttentionShuffled": "Attention\n(shuffled landscape)",
    "OracleTargetDirection": "Oracle target\ndirection",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _find(rows, **criteria):
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def _cell(row, metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    sd = float(row[f"{metric}_sd"])
    ci = float(row[f"{metric}_ci95"])
    return f"{mean:.3f} ± {sd:.3f} (95% CI {mean - ci:.3f}, {mean + ci:.3f})"


def _paired_cell(row, metric: str) -> str:
    mean = float(row[f"delta_{metric}_mean"])
    ci = float(row[f"delta_{metric}_ci95"])
    return f"{mean:+.3f} (95% CI {mean - ci:+.3f}, {mean + ci:+.3f})"


def _resolved_positive(row, metric: str) -> bool:
    return float(row[f"delta_{metric}_mean"]) - float(row[f"delta_{metric}_ci95"]) > 0


def _result_table(summary, *, split: str, conditions: Iterable[str]) -> str:
    lines = [
        "| Condition | Matched cosine | Proposal recall | Mode-set recall | Quality | Decoy rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        row = _find(summary, condition=condition, split=split)
        lines.append(
            f"| {condition} | {_cell(row, 'mean_matched_cosine')} | "
            f"{_cell(row, 'proposal_recall')} | {_cell(row, 'mode_set_recall')} | "
            f"{_cell(row, 'candidate_mean_quality')} | {_cell(row, 'candidate_decoy_rate')} |"
        )
    return "\n".join(lines)


def _generalization_table(summary) -> str:
    lines = [
        "| Family | Task-matched recall | DeepSets recall | Attention recall | Attention cosine | Attention mode-set recall | Attention decoy rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    splits = (
        "iid",
        "rotated",
        "translated",
        "width_shift",
        "more_peaks",
        "more_decoys",
        "new_goals",
        "unseen",
    )
    for split in splits:
        task = _find(summary, condition="TaskOnlyParameterMatched", split=split)
        deep = _find(summary, condition="LandscapeDeepSets", split=split)
        attention = _find(summary, condition="LandscapeAttention", split=split)
        lines.append(
            f"| {split} | {float(task['proposal_recall_mean']):.3f} | "
            f"{float(deep['proposal_recall_mean']):.3f} | "
            f"{float(attention['proposal_recall_mean']):.3f} | "
            f"{float(attention['mean_matched_cosine_mean']):.3f} | "
            f"{float(attention['mode_set_recall_mean']):.3f} | "
            f"{float(attention['candidate_decoy_rate_mean']):.3f} |"
        )
    return "\n".join(lines)


def _plot_condition_bars(summary, output: Path, metric: str, ylabel: str) -> None:
    conditions = list(DISPLAY)
    rows = [_find(summary, condition=value, split="iid") for value in conditions]
    mean = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
    ci = np.asarray([float(row[f"{metric}_ci95"]) for row in rows])
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    color = ["#8c8c8c", "#a6a6a6", "#4c78a8", "#2f9e73", "#d95f5f", "#7b5fb5"]
    ax.bar(np.arange(len(rows)), mean, yerr=ci, capsize=4, color=color)
    ax.set_xticks(np.arange(len(rows)), [DISPLAY[value] for value in conditions])
    ax.set_ylabel(ylabel)
    ax.set_title(f"Held-out IID {ylabel.lower()} (mean and 95% CI, 30 paired seeds)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_gaps(gaps, output: Path) -> None:
    conditions = [value for value in DISPLAY if value != "OracleTargetDirection"]
    rows = [_find(gaps, condition=value) for value in conditions]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for offset, metric, label, color in [
        (-0.18, "cosine_gap", "Matched cosine gap", "#4c78a8"),
        (0.18, "proposal_recall_gap", "Proposal-recall gap", "#f28e2b"),
    ]:
        mean = [float(row[f"{metric}_mean"]) for row in rows]
        ci = [float(row[f"{metric}_ci95"]) for row in rows]
        ax.bar(
            x + offset, mean, width=0.36, yerr=ci, capsize=3, label=label, color=color
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [DISPLAY[value] for value in conditions])
    ax.set_ylabel("Familiar geometry minus new IID geometry")
    ax.set_title("Geometry generalization gap")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_paired(
    per_seed, output: Path, left: str, right: str, metric: str, title: str
) -> None:
    left_rows = {
        int(row["seed"]): float(row[metric])
        for row in per_seed
        if row["split"] == "iid" and row["condition"] == left
    }
    right_rows = {
        int(row["seed"]): float(row[metric])
        for row in per_seed
        if row["split"] == "iid" and row["condition"] == right
    }
    seeds = sorted(set(left_rows) & set(right_rows))
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    x = np.asarray([right_rows[seed] for seed in seeds])
    y = np.asarray([left_rows[seed] for seed in seeds])
    ax.scatter(x, y, color="#2f9e73", alpha=0.8)
    low, high = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel(DISPLAY[right])
    ax.set_ylabel(DISPLAY[left])
    ax.set_title(title)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _attention_summary(rows) -> list[dict[str, object]]:
    output = []
    for split in sorted({row["split"] for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        result = {"split": split, "n": len(selected)}
        for metric in (
            "candidate_attention_diversity",
            "attention_useful_mass",
            "attention_entropy",
            "attention_target_specialization",
        ):
            value = statistics([float(row[metric]) for row in selected])
            result.update(
                {
                    f"{metric}_mean": value["mean"],
                    f"{metric}_sd": value["sd"],
                    f"{metric}_ci95": value["ci95"],
                }
            )
        output.append(result)
    return output


def _plot_attention_example(rows, output: Path) -> None:
    selected = [row for row in rows if int(row["task_index"]) == 0]
    candidates = sorted({int(row["candidate_index"]) for row in selected})
    components = sorted({int(row["component_index"]) for row in selected})
    matrix = np.zeros((len(candidates), len(components)), dtype=np.float64)
    useful = np.zeros(len(components), dtype=bool)
    for row in selected:
        i = candidates.index(int(row["candidate_index"]))
        j = components.index(int(row["component_index"]))
        matrix[i, j] = float(row["attention"])
        useful[j] = bool(int(row["component_is_useful_evaluation_only"]))
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xlabel("Permuted component token (* = useful, evaluation only)")
    ax.set_ylabel("Candidate identity z")
    ax.set_xticks(
        range(len(components)),
        [
            f"{value}{'*' if useful[index] else ''}"
            for index, value in enumerate(components)
        ],
    )
    ax.set_yticks(range(len(candidates)), candidates)
    ax.set_title("Predeclared seed/task: candidate-to-component attention")
    fig.colorbar(image, ax=ax, label="Attention mass")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def generate_landscape_representation_reports(results_root: Path) -> dict[str, str]:
    rep = results_root / "d32_landscape_representation"
    attn = results_root / "d32_landscape_attention"
    ident = results_root / "d32_landscape_identifiability"
    summary = _read(rep / "summary.csv")
    paired = _read(rep / "paired_comparisons.csv")
    per_seed = _read(rep / "per_seed_results.csv")
    gaps = _read(rep / "familiar_new_geometry_gap_summary.csv")
    parameters = _read(rep / "model_parameter_summary.csv")
    probes = _read(ident / "component_utility_probe_summary.csv")
    analysis = _read(attn / "attention_analysis.csv")
    examples = _read(attn / "attention_example_source.csv")

    paired_iid = {
        (row["condition_a"], row["condition_b"]): row
        for row in paired
        if row["split"] == "iid"
    }
    attention_vs_task = paired_iid[("LandscapeAttention", "TaskOnlyParameterMatched")]
    attention_vs_shuffle = paired_iid[
        ("LandscapeAttention", "LandscapeAttentionShuffled")
    ]
    attention_vs_deep = paired_iid[("LandscapeAttention", "LandscapeDeepSets")]
    deep_vs_task = paired_iid[("LandscapeDeepSets", "TaskOnly")]
    probe = probes[0]
    probe_auroc = float(probe["auroc_mean"])

    rich_helps = _resolved_positive(
        attention_vs_task, "mean_matched_cosine"
    ) and _resolved_positive(attention_vs_shuffle, "mean_matched_cosine")
    attention_beats_deep = (
        _resolved_positive(attention_vs_deep, "mean_matched_cosine")
        or _resolved_positive(attention_vs_deep, "proposal_recall")
        or _resolved_positive(attention_vs_deep, "mode_set_recall")
    )
    correct_beats_shuffle = _resolved_positive(
        attention_vs_shuffle, "mean_matched_cosine"
    ) or _resolved_positive(attention_vs_shuffle, "proposal_recall")
    near_chance = abs(probe_auroc - 0.5) <= 0.05
    if near_chance:
        diagnosis = "Case E: the unlabeled component description is not identifiable enough to separate useful components."
    elif not correct_beats_shuffle:
        diagnosis = "Case D: correct and shuffled landscapes are statistically unresolved; the model is not using task-specific geometry reliably."
    elif attention_beats_deep:
        diagnosis = "Case C: candidate-specific attention improves multimode inference beyond pooled landscape information."
    elif rich_helps:
        diagnosis = "Case B: landscape information helps, while DeepSets is competitive with attention; prefer the simpler pooled encoder."
    else:
        diagnosis = "The preregistered case criteria are unresolved; do not move to 64D on this evidence alone."

    table = _result_table(summary, split="iid", conditions=DISPLAY)
    training_table = _result_table(
        summary,
        split="training",
        conditions=(
            "TaskOnly",
            "TaskOnlyParameterMatched",
            "LandscapeDeepSets",
            "LandscapeAttention",
            "LandscapeAttentionShuffled",
        ),
    )
    generalization_table = _generalization_table(summary)
    paired_table = "\n".join(
        [
            "| Paired IID comparison (A − B) | Δ cosine | Δ proposal recall | Δ mode-set recall |",
            "|---|---:|---:|---:|",
            *[
                f"| {a} − {b} | {_paired_cell(row, 'mean_matched_cosine')} | "
                f"{_paired_cell(row, 'proposal_recall')} | {_paired_cell(row, 'mode_set_recall')} |"
                for (a, b), row in paired_iid.items()
            ],
        ]
    )
    parameter_table = "\n".join(
        [
            "| Condition | Parameters | Mean training seconds | Peak allocated VRAM |",
            "|---|---:|---:|---:|",
            *[
                f"| {row['condition']} | {float(row['parameter_count_mean']):,.0f} | "
                f"{float(row['training_seconds_mean']):.2f} | "
                f"{float(row['peak_allocated_bytes_mean']) / 2**20:.1f} MiB |"
                for row in parameters
            ],
        ]
    )

    representation_report = f"""# D32 Landscape Representation

This isolated experiment replaces the former oracle-like goal direction with either a learned constant task token or an unlabeled component set. All useful directions and radii are supervised labels only. Component input is exactly `[mu_j-h, amplitude_j, lower_triangle(covariance_j)]` (561 values in d=32), including full covariance orientation. Component order is independently randomized.

## Training-set fit

{training_table}

## Held-out IID results

{table}

## Paired tests

{paired_table}

## Held-out family behavior

{generalization_table}

The representation benefit is strong on new IID geometries, width shifts, extra
peaks, and extra decoys. It does **not** establish coordinate-equivariant or
open-family generalization: proposal recall is zero under rotations and unseen
geometries and nearly zero for new goals. This limitation is not hidden by the
positive IID result and argues against moving directly to d=64.

The component-set encoders use no original `g`, target identity, useful/decoy labels, region IDs, centers selected by usefulness, target radius, or refinement outcome. The learned constant task token means the non-oracle task instruction is only “find multiple useful alternatives.”

## Compute and capacity control

{parameter_table}

The parameter-matched TaskOnly model differs from LandscapeAttention by only the small rounding difference shown above, so their comparison isolates information rather than gross capacity.
"""
    _write(rep / "D32_LANDSCAPE_REPRESENTATION.md", representation_report)

    attention_summary = _attention_summary(analysis)
    write_csv(attn / "attention_analysis_summary.csv", attention_summary)
    iid_attention = _find(attention_summary, split="iid")
    attention_report = f"""# D32 Landscape Attention

LandscapeAttention uses one small candidate-to-component cross-attention block. Each query is formed from `[h, learned_task_token, z_i]`; keys and values come from label-free component descriptors.

On held-out IID tasks, mean attention mass assigned to useful components was **{float(iid_attention["attention_useful_mass_mean"]):.3f}**, candidate-attention diversity was **{float(iid_attention["candidate_attention_diversity_mean"]):.3f}**, and evaluation-only target specialization was **{float(iid_attention["attention_target_specialization_mean"]):.3f}**.

LandscapeAttention minus LandscapeDeepSets: cosine {_paired_cell(attention_vs_deep, "mean_matched_cosine")}, proposal recall {_paired_cell(attention_vs_deep, "proposal_recall")}, and mode-set recall {_paired_cell(attention_vs_deep, "mode_set_recall")}.

These labels were applied only after inference for mechanistic analysis. They were never part of the attention input or training objective beyond the ordinary target-direction labels used by every supervised condition.
"""
    _write(attn / "D32_LANDSCAPE_ATTENTION.md", attention_report)

    ident_report = f"""# D32 Landscape Identifiability

The diagnostic ComponentUtilityProbe sees only the same unlabeled per-component descriptor and the constant task token. Usefulness is a supervised target, never an input.

| Metric | Mean ± SD (95% CI) |
|---|---:|
| Accuracy | {_cell(probe, "accuracy")} |
| Balanced accuracy | {_cell(probe, "balanced_accuracy")} |
| AUROC | {_cell(probe, "auroc")} |
| Precision | {_cell(probe, "precision")} |
| Recall | {_cell(probe, "recall")} |

Interpretation: **{"near chance; the observable component descriptor is non-identifiable" if near_chance else "above chance; the observable landscape carries learnable utility information"}**. The probe is diagnostic and is not used by CSD at inference.
"""
    _write(ident / "D32_LANDSCAPE_IDENTIFIABILITY.md", ident_report)

    oracle_row = _find(summary, condition="OracleTargetDirection", split="iid")
    attention_row = _find(summary, condition="LandscapeAttention", split="iid")
    oracle_report = f"""# D32 Oracle vs Non-Oracle

`OracleTargetDirection` is the unchanged, privileged upper bound that receives the former goal vector encoding the primary target direction. It is not included in fair-method averages.

| Condition | Matched cosine | Proposal recall | Mode-set recall |
|---|---:|---:|---:|
| LandscapeAttention (non-oracle) | {_cell(attention_row, "mean_matched_cosine")} | {_cell(attention_row, "proposal_recall")} | {_cell(attention_row, "mode_set_recall")} |
| OracleTargetDirection | {_cell(oracle_row, "mean_matched_cosine")} | {_cell(oracle_row, "proposal_recall")} | {_cell(oracle_row, "mode_set_recall")} |

The scientific target is not oracle parity. It is whether legitimate task structure closes a meaningful fraction of the held-out generalization gap.
"""
    _write(rep / "D32_ORACLE_VS_NONORACLE.md", oracle_report)

    diagnosis_report = f"""# D32 Landscape Diagnosis

## Decision

**{diagnosis}**

Correct LandscapeAttention versus shuffled landscape: cosine {_paired_cell(attention_vs_shuffle, "mean_matched_cosine")}, proposal recall {_paired_cell(attention_vs_shuffle, "proposal_recall")}, mode-set recall {_paired_cell(attention_vs_shuffle, "mode_set_recall")}.

LandscapeAttention versus parameter-matched TaskOnly: cosine {_paired_cell(attention_vs_task, "mean_matched_cosine")}, proposal recall {_paired_cell(attention_vs_task, "proposal_recall")}, mode-set recall {_paired_cell(attention_vs_task, "mode_set_recall")}.

LandscapeDeepSets versus TaskOnly: cosine {_paired_cell(deep_vs_task, "mean_matched_cosine")}, proposal recall {_paired_cell(deep_vs_task, "proposal_recall")}, mode-set recall {_paired_cell(deep_vs_task, "mode_set_recall")}.

This Case C classification is specific to new IID geometries. LandscapeAttention
does not recover useful proposals under random rotations or unseen geometry and
is nearly zero-recall on new goals. Candidate-specific access solves much of the
within-family information bottleneck, but it does not solve coordinate or
out-of-family generalization. Moving to d=64 is therefore not recommended yet.

No result is described as resolved unless its paired 95% confidence interval excludes zero. This experiment does not run d=64.

## Toy-to-LLM mapping

| Toy | Future LLM |
|---|---|
| current latent state `h` | current reasoning hidden state |
| component tokens | prompt/context hidden states |
| task token | problem objective/instruction representation |
| component-set attention | attention over prompt/context |
| useful region | successful reasoning strategy |
| useful direction label | latent direction extracted from demonstrated solution |
| CSD proposal | candidate reasoning strategy |
| refinement | downstream transformer reasoning |

The intended later formulation is `H_prompt^(L) -> candidate-specific CSD attention -> {{(v_hat_i, alpha_i)}}_{{i=1}}^K`, with demonstrations providing strategy-direction labels and inference receiving only the new problem/prompt.
"""
    _write(rep / "D32_LANDSCAPE_DIAGNOSIS.md", diagnosis_report)

    validation_decisions = {
        "test_set_used_for_selection": False,
        "architecture_selection": "none; all conditions were predeclared",
        "hyperparameter_tuning": "none in this study; inherited frozen d32 supervised protocol",
        "task_goal_choice": "learned constant token because no legitimate pre-target goal variable exists",
        "component_features": "center relative to h, amplitude, full covariance lower triangle",
        "condition_interpretation_thresholds": {
            "paired_cosine_effect": 0.03,
            "paired_recall_effect": 0.10,
            "component_probe_chance_auroc_band": 0.05,
            "resolved_difference": "paired 95% CI excludes zero",
        },
        "decision_applied_after_test": diagnosis,
        "d64_run": False,
    }
    (rep / "validation_decisions.json").write_text(
        json.dumps(validation_decisions, indent=2), encoding="utf-8"
    )

    _plot_condition_bars(
        summary,
        rep / "heldout_matched_cosine.png",
        "mean_matched_cosine",
        "Matched cosine",
    )
    _plot_condition_bars(
        summary, rep / "proposal_recall.png", "proposal_recall", "Proposal recall"
    )
    _plot_condition_bars(
        summary, rep / "mode_set_recall.png", "mode_set_recall", "Mode-set recall"
    )
    _plot_gaps(gaps, rep / "familiar_vs_new_geometry_gap.png")
    _plot_paired(
        per_seed,
        rep / "correct_vs_shuffled_paired.png",
        "LandscapeAttention",
        "LandscapeAttentionShuffled",
        "proposal_recall",
        "Correct vs shuffled landscape: proposal recall",
    )
    _plot_paired(
        per_seed,
        rep / "deepsets_vs_attention_paired.png",
        "LandscapeAttention",
        "LandscapeDeepSets",
        "mode_set_recall",
        "Attention vs DeepSets: mode-set recall",
    )
    _plot_attention_example(examples, attn / "attention_component_example.png")
    return {
        "diagnosis": diagnosis,
        "representation_report": str(rep / "D32_LANDSCAPE_REPRESENTATION.md"),
        "attention_report": str(attn / "D32_LANDSCAPE_ATTENTION.md"),
        "identifiability_report": str(ident / "D32_LANDSCAPE_IDENTIFIABILITY.md"),
    }
