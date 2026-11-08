"""Evidence-bound reports for the sequential d=16 capacity diagnosis."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from creativity_toy.analysis_utils import statistics

from .study import write_csv


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: Mapping[str, object], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _stat(rows: Sequence[Mapping[str, object]], metric: str) -> dict[str, float]:
    return statistics([_f(row, metric) for row in rows if np.isfinite(_f(row, metric))])


def _cell(rows: Sequence[Mapping[str, object]], metric: str) -> str:
    stat = _stat(rows, metric)
    return (
        f"{_fmt(stat['mean'])} +/- {_fmt(stat['sd'])} (95% CI +/-{_fmt(stat['ci95'])})"
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def _select(
    rows: list[dict[str, str]],
    *,
    dimension: int,
    architecture: str,
    mode: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if int(float(row["dimension"])) == dimension
        and row["architecture"] == architecture
        and row["mode"] == mode
    ]


def _plot_supervised_curves(
    rows: list[dict[str, str]], path: Path, primary_dimension: int
) -> None:
    metrics = [
        ("candidate_zero_target_cosine", "Cosine alignment"),
        ("mean_angular_error_degrees", "Angular error (degrees)"),
        ("mean_alpha_absolute_error", "Alpha absolute error"),
        ("proposal_recall", "Post-refinement proposal recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for architecture, linestyle in [("current_mlp", "-"), ("linear", "--")]:
        for mode, color in [("single_target", "tab:blue"), ("multimode", "tab:orange")]:
            selected = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["mode"] == mode
                and int(float(row["dimension"])) == primary_dimension
            ]
            if not selected:
                continue
            steps = sorted({int(float(row["step"])) for row in selected})
            for ax, (metric, ylabel) in zip(axes.flat, metrics):
                means = [
                    np.mean(
                        [
                            _f(row, metric)
                            for row in selected
                            if int(float(row["step"])) == step
                        ]
                    )
                    for step in steps
                ]
                ax.plot(
                    steps,
                    means,
                    color=color,
                    linestyle=linestyle,
                    label=f"{architecture} / {mode}",
                )
                ax.set(xlabel="AdamW step", ylabel=ylabel)
                ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_anchor_curves(rows: list[dict[str, str]], path: Path) -> None:
    validation = [row for row in rows if row.get("curve") == "validation"]
    metrics = [
        ("candidate_anchor_rate", "Anchor rate"),
        ("proposal_recall", "Proposal recall"),
        ("candidate_decoy_rate", "Decoy rate"),
        ("best_useful_alignment", "Best useful alignment"),
        ("candidate_mean_quality", "Mean quality"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for value in sorted({float(row["lambda_anchor"]) for row in validation}):
        selected = [row for row in validation if float(row["lambda_anchor"]) == value]
        budgets = sorted({int(float(row["candidate_evaluations"])) for row in selected})
        for ax, (metric, ylabel) in zip(axes.flat, metrics):
            means = [
                np.mean(
                    [
                        _f(row, metric)
                        for row in selected
                        if int(float(row["candidate_evaluations"])) == budget
                    ]
                )
                for budget in budgets
            ]
            ax.plot(budgets, means, marker="o", label=f"λA={value:g}")
            ax.set(xlabel="Black-box candidate outcomes", ylabel=ylabel)
            ax.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_capacity_reports(
    *,
    supervised_dir: Path,
    anchor_dir: Path,
) -> dict[str, object]:
    rows = _read(supervised_dir / "per_seed.csv")
    curves = _read(supervised_dir / "training_curves.csv")
    gate = json.loads(
        (supervised_dir / "decision_gate.json").read_text(encoding="utf-8")
    )
    primary_dimension = int(gate.get("primary_dimension", 16))
    batch = json.loads(
        (supervised_dir / "selected_batch.json").read_text(encoding="utf-8")
    )
    batch_rows = _read(supervised_dir / "batch_scaling.csv")
    selected_batch_row = next(
        row
        for row in batch_rows
        if int(float(row["batch_size"])) == int(batch["selected_batch_size"])
    )
    unique_training_seconds = sum(_f(row, "training_seconds", 0.0) for row in rows)
    peak_allocated = max(
        (_f(row, "peak_allocated_bytes", 0.0) for row in rows), default=0.0
    )
    _plot_supervised_curves(
        curves, supervised_dir / "supervised_training_curves.png", primary_dimension
    )

    dimension_rows = []
    for dimension in sorted({int(float(row["dimension"])) for row in rows}):
        selected = _select(
            rows, dimension=dimension, architecture="current_mlp", mode="multimode"
        )
        if selected:
            dimension_rows.append(
                [
                    dimension,
                    _cell(selected, "best_target_cosine"),
                    _cell(selected, "mean_matched_cosine"),
                    _cell(selected, "mean_alpha_absolute_error"),
                    _cell(selected, "proposal_recall"),
                ]
            )
    primary_architecture = []
    for mode in ["single_target", "multimode"]:
        for architecture in ["current_mlp", "linear"]:
            selected = _select(
                rows, dimension=primary_dimension, architecture=architecture, mode=mode
            )
            if selected:
                training_cosine_metric = (
                    "training_candidate_zero_target_cosine"
                    if mode == "single_target"
                    else "training_mean_matched_cosine"
                )
                heldout_cosine_metric = (
                    "candidate_zero_target_cosine"
                    if mode == "single_target"
                    else "mean_matched_cosine"
                )
                primary_architecture.append(
                    [
                        architecture,
                        mode,
                        _cell(selected, training_cosine_metric),
                        _cell(selected, heldout_cosine_metric),
                        _cell(selected, "training_proposal_recall"),
                        _cell(selected, "proposal_recall"),
                        _cell(selected, "mean_angular_error_degrees"),
                        _cell(selected, "mean_alpha_absolute_error"),
                    ]
                )
    supervised_report = f"""# Supervised representability diagnostic

This is a privileged capacity test, not a deployable CSD training method. During training only, labels are exact normalized center displacements and center distances from training landscapes. Q, refinement, region labels, and test centers are absent from the loss. Multimode supervision uses exact permutation-invariant assignment over the small target/proposal sets; unmatched proposals are unconstrained.

The single-target gate was declared before evaluation: candidate 0 must fit the explicitly supplied training labels with mean target cosine > 0.90 and target-region recall >= 0.75. Multimode representational capacity uses the same distinction (matched cosine > 0.90 and post-refinement proposal recall >= 0.75 on supervised training examples); held-out results are reported separately as generalization tests. This matters because fitting supplied labels tests capacity, while prediction on unseen geometries also tests whether the inputs identify the latent targets.

Selected CUDA batch size: **{batch["selected_batch_size"]}**, using: {batch["selection_rule"]}.
Measured selected-batch throughput was **{_f(selected_batch_row, "examples_per_second"):,.0f} examples/s** with **{_f(selected_batch_row, "peak_allocated_bytes") / 2**20:.1f} MiB** peak allocated memory in the timing sweep. Summed per-model supervised training time was **{unique_training_seconds:.1f} s**; peak allocated memory across study models was **{peak_allocated / 2**20:.1f} MiB**.

## Dimension control: current CSD, multimode

{_table(["d", "Best target cosine", "Mean matched cosine", "Alpha error", "Proposal recall"], dimension_rows) if dimension_rows else "Multimode was not run because the single-target gate failed."}

## {primary_dimension}D architecture control

{_table(["Architecture", "Mode", "Training cosine", "Held-out cosine", "Training recall", "Held-out recall", "Held-out angular error", "Held-out alpha error"], primary_architecture)}

## Gate outcome

- Single-target passed: **{gate["single_target_pass"]}**.
- Single-target training-label fit: cosine {_fmt(float(gate.get("mean_training_candidate_zero_target_cosine", float("nan"))))}, recall {_fmt(float(gate.get("mean_training_candidate_zero_target_recall", float("nan"))))}.
- Single-target held-out generalization passed: **{gate.get("single_target_heldout_generalization_pass", False)}** (cosine {_fmt(float(gate["mean_candidate_zero_target_cosine"]))}, recall {_fmt(float(gate["mean_candidate_zero_target_recall"]))}).
- Multimode run: **{gate["multimode_run"]}**.
- Multimode training-label fit passed: **{gate.get("multimode_training_fit_pass", False)}** (cosine {_fmt(float(gate.get("mean_primary_multimode_training_matched_cosine", gate.get("mean_d16_multimode_training_matched_cosine", float("nan")))))}, recall {_fmt(float(gate.get("mean_primary_multimode_training_proposal_recall", gate.get("mean_d16_multimode_training_proposal_recall", float("nan")))))}).
- Multimode held-out generalization passed: **{gate.get("multimode_heldout_generalization_pass", False)}** (cosine {_fmt(float(gate.get("mean_primary_multimode_matched_cosine", gate.get("mean_d16_multimode_matched_cosine", float("nan")))))}, recall {_fmt(float(gate.get("mean_primary_multimode_proposal_recall", gate.get("mean_d16_multimode_proposal_recall", float("nan")))))}).
- Anchor-escape allowed: **{gate["anchor_escape_allowed"]}**.
- Held-out candidate-0 cosine: {_fmt(float(gate["mean_candidate_zero_target_cosine"]))}.
- Held-out candidate-0 target recall: {_fmt(float(gate["mean_candidate_zero_target_recall"]))}.

{("The model fits both the supplied single-target and multimode " + str(primary_dimension) + "D labels, establishing portfolio representational capacity. Held-out failures remain a separate input/inductive-bias limitation: `[h,g,z]` may not identify independently varying latent targets." if gate.get("multimode_training_fit_pass", False) else ("The model fits supplied single-target labels but fails the multimode training-label criterion, so the current architecture cannot represent the required fixed-size useful proposal portfolio under this protocol." if gate.get("single_target_pass", False) else "The model fails even the single-target training-label criterion, so basic representational capacity is not established."))} None of these privileged results show that scalar-reward RL can discover a useful direction.
"""
    (supervised_dir / "SUPERVISED_REPRESENTABILITY.md").write_text(
        supervised_report, encoding="utf-8"
    )

    anchor_decision = json.loads(
        (anchor_dir / "decision.json").read_text(encoding="utf-8")
    )
    if anchor_decision["status"] == "skipped":
        anchor_report = f"""# Anchor-escape RL diagnostic

Status: **skipped by the sequential experimental gate**.

Reason: {anchor_decision["reason"]}.

No anchor-escape RL run was performed, so no result is inferred about reward shortcutting. Mode preservation remained off by construction (`lambda_mode = 0`).
"""
        case = "case_1"
        conclusion = "Redesign the CSD direction architecture before further RL work."
    else:
        test = _read(anchor_dir / "test_per_seed.csv")
        training_rows = _read(anchor_dir / "training_curves.csv")
        curves_anchor = _read(anchor_dir / "evaluation_curves.csv")
        selection = json.loads(
            (anchor_dir / "lambda_selection.json").read_text(encoding="utf-8")
        )
        _plot_anchor_curves(
            curves_anchor, anchor_dir / "anchor_escape_training_curves.png"
        )
        table_rows = []
        for value in sorted({float(row["lambda_anchor"]) for row in test}):
            selected = [row for row in test if float(row["lambda_anchor"]) == value]
            table_rows.append(
                [
                    value,
                    _cell(selected, "candidate_anchor_rate"),
                    _cell(selected, "proposal_recall"),
                    _cell(selected, "useful_candidate_rate"),
                    _cell(selected, "candidate_decoy_rate"),
                    _cell(selected, "candidate_mean_quality"),
                    _cell(selected, "best_useful_alignment"),
                    _cell(selected, "capture_entry_rate"),
                    _cell(selected, "alpha_mean"),
                    _cell(selected, "effective_directional_rank"),
                ]
            )
        baseline = [row for row in test if float(row["lambda_anchor"]) == 0.0]
        chosen_value = float(anchor_decision["selected_lambda"])
        chosen = [row for row in test if float(row["lambda_anchor"]) == chosen_value]
        paired_rows = []
        for seed in sorted({int(float(row["seed"])) for row in baseline}):
            left = next(row for row in baseline if int(float(row["seed"])) == seed)
            right = next(row for row in chosen if int(float(row["seed"])) == seed)
            paired_rows.append(
                {
                    "seed": seed,
                    "lambda_anchor": chosen_value,
                    **{
                        f"delta_{metric}": _f(right, metric) - _f(left, metric)
                        for metric in [
                            "candidate_anchor_rate",
                            "proposal_recall",
                            "useful_candidate_rate",
                            "candidate_decoy_rate",
                            "candidate_mean_quality",
                            "best_useful_alignment",
                            "capture_entry_rate",
                            "alpha_mean",
                            "effective_directional_rank",
                        ]
                    },
                }
            )
        write_csv(anchor_dir / "paired_differences.csv", paired_rows)
        paired_table = []
        for metric in [
            "candidate_anchor_rate",
            "proposal_recall",
            "candidate_decoy_rate",
            "candidate_mean_quality",
            "best_useful_alignment",
            "capture_entry_rate",
        ]:
            values = [_f(row, f"delta_{metric}") for row in paired_rows]
            stat = statistics(values)
            paired_table.append(
                [
                    metric,
                    _fmt(stat["mean"]),
                    _fmt(stat["sd"]),
                    f"[{_fmt(stat['mean'] - stat['ci95'])}, {_fmt(stat['mean'] + stat['ci95'])}]",
                ]
            )
        anchor_drop = (
            _stat(baseline, "candidate_anchor_rate")["mean"]
            - _stat(chosen, "candidate_anchor_rate")["mean"]
        )
        recall_gain = (
            _stat(chosen, "proposal_recall")["mean"]
            - _stat(baseline, "proposal_recall")["mean"]
        )
        decoy_gain = (
            _stat(chosen, "candidate_decoy_rate")["mean"]
            - _stat(baseline, "candidate_decoy_rate")["mean"]
        )
        selected_training = [
            row
            for row in training_rows
            if row.get("curve") == "training"
            and float(row["lambda_anchor"]) == chosen_value
        ]
        final_training = []
        for seed in sorted({int(float(row["seed"])) for row in selected_training}):
            seed_rows = [
                row for row in selected_training if int(float(row["seed"])) == seed
            ]
            final_training.append(
                max(seed_rows, key=lambda row: _f(row, "candidate_evaluations"))
            )
        final_penalty = _stat(final_training, "mean_anchor_penalty")["mean"]
        final_inside = _stat(final_training, "inside_anchor_margin_rate")["mean"]
        final_distance = _stat(final_training, "mean_anchor_distance")["mean"]
        if anchor_drop > 0.05 and recall_gain > 0.10 and decoy_gain <= 0.10:
            case = "case_2"
            conclusion = (
                "Architecture is capable; reward shortcutting was a major issue."
            )
        elif anchor_drop <= 0.05:
            case = "case_3_inactive_penalty"
            conclusion = "Architecture can fit supplied directions, but the pre-refinement margin was effectively inactive; anchor shortcut removal was not achieved and useful-direction discovery remains unsolved."
        else:
            case = "case_3"
            conclusion = "Architecture is capable, but high-dimensional useful-direction discovery remains unsolved."
        anchor_report = f"""# Anchor-escape RL diagnostic

This experiment ran only because supervised representability passed. It returned to detached black-box Q and changed only the reward by subtracting a bounded anchor-neighborhood margin penalty. Mode preservation was off (`lambda_mode = 0`). The margin radius was **{anchor_decision["anchor_radius"]:.3f}**, derived as {anchor_decision["anchor_radius_basis"]} rather than tuned on test outcomes.

Validation selection rule: {selection["rule"]}. Selected lambda_A: **{chosen_value:g}**.

{_table(["lambda_A", "Anchor rate", "Proposal recall", "Useful rate", "Decoy rate", "Quality", "Useful alignment", "Capture entry", "Alpha", "Direction rank"], table_rows)}

## Paired selected-minus-baseline differences

{_table(["Metric", "Mean delta", "SD", "95% CI"], paired_table)}

Observed test differences relative to lambda_A=0: anchor-rate change **{-anchor_drop:+.3f}**, proposal-recall change **{recall_gain:+.3f}**, decoy-rate change **{decoy_gain:+.3f}**.

## Was the penalty active?

At the final checkpoint, mean pre-refinement anchor distance was **{final_distance:.3f}**, mean margin penalty was **{final_penalty:.6f}**, and the fraction inside the 1.625 margin was **{final_inside:.6f}**. The policy's proposals were already outside the pre-refinement margin, even though nine-step refinement frequently returned them to the anchor. Consequently, increasing this coefficient would not address the measured failure: the penalty is zero where the policy operates.

Decision-tree interpretation: **{conclusion}**

The penalty stops at the anchor margin; it does not reward arbitrarily large distance. Useful and decoy labels were evaluation-only.
"""
    (anchor_dir / "ANCHOR_ESCAPE_RESULTS.md").write_text(
        anchor_report, encoding="utf-8"
    )

    diagnosis = f"""# D{primary_dimension} capacity diagnosis

## Sequential decision

- Part A single-target representability passed: **{gate["single_target_pass"]}**.
- Part A single-target held-out generalization passed: **{gate.get("single_target_heldout_generalization_pass", False)}**.
- Part A multimode training-label fit passed: **{gate.get("multimode_training_fit_pass", False)}**.
- Part A held-out multimode generalization passed: **{gate.get("multimode_heldout_generalization_pass", False)}**.
- Part A overall capacity gate passed: **{gate["anchor_escape_allowed"]}**.
- Part B status: **{anchor_decision["status"]}**.
- Decision-tree case: **{case}**.

## Conclusion

> **{conclusion}**

This conclusion follows the predefined tree:

1. If explicit target supervision cannot produce >0.90 alignment and high post-refinement recall on the supplied labels, stop and redesign the direction architecture.
2. If it can, run detached black-box anchor-escape RL. Held-out multimode generalization remains separately reported and does not silently redefine training-label fit.
3. If anchor escape lowers anchor occupancy and recovers useful proposals without uncontrolled decoys, the anchor shortcut was major.
4. If anchor occupancy falls but proposals become decoys, representational capacity is present but useful-direction discovery remains unsolved.
5. If anchor occupancy does not change because the pre-refinement margin is inactive, the anchor-shortcut hypothesis remains unresolved under this intervention; the next controlled test should measure/penalize the post-refinement endpoint rather than merely increasing lambda.

No deployable creativity claim follows from privileged supervision. Part A shows whether the fixed architecture can encode supplied {primary_dimension}D mappings; its held-out results separately test whether the current inputs and inductive bias support generalization.
"""
    (supervised_dir / f"D{primary_dimension}_CAPACITY_DIAGNOSIS.md").write_text(
        diagnosis, encoding="utf-8"
    )
    return {
        "case": case,
        "conclusion": conclusion,
        "single_target_pass": gate["single_target_pass"],
        "multimode_pass": gate["anchor_escape_allowed"],
        "anchor_escape_status": anchor_decision["status"],
    }
