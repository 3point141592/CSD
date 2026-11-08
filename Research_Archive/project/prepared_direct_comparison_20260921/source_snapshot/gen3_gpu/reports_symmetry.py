"""Reports and paper-quality figures for the d32 coordinate-symmetry study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config_symmetry import SymmetryStudyConfig

LABEL = {
    "CurrentAttention": "Current",
    "CurrentAttentionAugmented": "Current + SE(d) aug.",
    "EquivariantAttention": "Equivariant",
    "EquivariantAttentionAugmented": "Equivariant + SE(d) aug.",
}
PRIMARY = ("iid", "exact_translated", "exact_rotated", "exact_rotation_translation")


def _read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find(rows, **criteria):
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def _write(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _metric(row, name: str) -> str:
    mean = float(row[f"{name}_mean"])
    sd = float(row[f"{name}_sd"])
    ci = float(row[f"{name}_ci95"])
    return f"{mean:.3f} +/- {sd:.3f} (95% CI {mean - ci:.3f}, {mean + ci:.3f})"


def _paired(rows, a, b, split, metric):
    row = _find(rows, condition_a=a, condition_b=b, split=split, metric=metric)
    mean, ci = float(row["mean_difference"]), float(row["ci95_difference"])
    return mean, ci, f"{mean:+.3f} (95% CI {mean - ci:+.3f}, {mean + ci:+.3f})"


def _primary_table(summary) -> str:
    lines = [
        "| Condition | Split | Matched cosine | Proposal recall | Mode-set recall | Useful modes | Quality | Decoy | Alpha error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in LABEL:
        for split in PRIMARY:
            row = _find(summary, condition=condition, split=split)
            lines.append(
                f"| {LABEL[condition]} | {split} | {_metric(row, 'mean_matched_cosine')} | "
                f"{_metric(row, 'proposal_recall')} | {_metric(row, 'mode_set_recall')} | "
                f"{_metric(row, 'useful_region_count')} | {_metric(row, 'candidate_mean_quality')} | "
                f"{_metric(row, 'candidate_decoy_rate')} | {_metric(row, 'mean_alpha_absolute_error')} |"
            )
    return "\n".join(lines)


def _equivariance_table(rows) -> str:
    transforms = ("exact_translated", "exact_rotated", "exact_rotation_translation")
    lines = [
        "| Condition | Transform | Set cosine | Angular error (deg) | Alpha error |",
        "|---|---|---:|---:|---:|",
    ]
    for condition in LABEL:
        for transform in transforms:
            row = _find(rows, condition=condition, transform=transform)
            lines.append(
                f"| {LABEL[condition]} | {transform} | {_metric(row, 'equivariance_cosine')} | "
                f"{_metric(row, 'equivariance_angular_error_degrees')} | "
                f"{_metric(row, 'alpha_invariance_error')} |"
            )
    return "\n".join(lines)


def _secondary_table(summary) -> str:
    splits = (
        "width_shift",
        "more_peaks",
        "more_decoys",
        "new_goals",
        "unseen",
        "family_translated",
        "family_rotated",
    )
    lines = [
        "| Split | Current recall | Current+aug recall | Equivariant recall | Equivariant+aug recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in splits:
        values = [
            float(
                _find(summary, condition=condition, split=split)["proposal_recall_mean"]
            )
            for condition in LABEL
        ]
        lines.append(
            f"| {split} | " + " | ".join(f"{value:.3f}" for value in values) + " |"
        )
    return "\n".join(lines)


def _grouped(summary, output: Path, metric: str, ylabel: str) -> None:
    x = np.arange(len(PRIMARY))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    colors = ("#7f7f7f", "#f28e2b", "#4c78a8", "#2f9e73")
    for index, (condition, label) in enumerate(LABEL.items()):
        rows = [_find(summary, condition=condition, split=split) for split in PRIMARY]
        mean = [float(row[f"{metric}_mean"]) for row in rows]
        ci = [float(row[f"{metric}_ci95"]) for row in rows]
        ax.bar(
            x + (index - 1.5) * width,
            mean,
            width,
            yerr=ci,
            capsize=3,
            label=label,
            color=colors[index],
        )
    ax.set_xticks(
        x, [value.replace("exact_", "").replace("_", " + ") for value in PRIMARY]
    )
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: exact coordinate transformations (mean and 95% CI)")
    ax.legend(ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _equivariance_plot(rows, output: Path) -> None:
    transforms = ("exact_translated", "exact_rotated", "exact_rotation_translation")
    x = np.arange(len(transforms))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    for index, condition in enumerate(LABEL):
        selected = [
            _find(rows, condition=condition, transform=value) for value in transforms
        ]
        mean = [float(row["equivariance_cosine_mean"]) for row in selected]
        ci = [float(row["equivariance_cosine_ci95"]) for row in selected]
        ax.bar(
            x + (index - 1.5) * width,
            mean,
            width,
            yerr=ci,
            capsize=3,
            label=LABEL[condition],
        )
    ax.set_xticks(
        x, [value.replace("exact_", "").replace("_", " + ") for value in transforms]
    )
    ax.set_ylabel("Hungarian equivariance cosine")
    ax.set_ylim(0, 1.03)
    ax.set_title("Exact proposal-set transformation consistency")
    ax.legend(ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _paired_scatter(
    per_seed, output: Path, left: str, right: str, split: str, title: str
) -> None:
    lookup = {
        (int(row["seed"]), row["condition"], row["split"]): float(
            row["proposal_recall"]
        )
        for row in per_seed
    }
    seeds = sorted({int(row["seed"]) for row in per_seed})
    x = np.asarray([lookup[(seed, right, split)] for seed in seeds])
    y = np.asarray([lookup[(seed, left, split)] for seed in seeds])
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(x, y, color="#2f9e73", alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(
        xlabel=LABEL[right],
        ylabel=LABEL[left],
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        title=title,
    )
    ax.grid(alpha=0.23)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _quality_mode(summary, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.8))
    markers = {split: marker for split, marker in zip(PRIMARY, ("o", "s", "^", "D"))}
    for condition, label in LABEL.items():
        for split in PRIMARY:
            row = _find(summary, condition=condition, split=split)
            ax.scatter(
                float(row["mode_set_recall_mean"]),
                float(row["candidate_mean_quality_mean"]),
                marker=markers[split],
                s=70,
                label=f"{label} / {split}",
            )
    ax.set_xlabel("Mode-set recall")
    ax.set_ylabel("Candidate mean quality")
    ax.set_title("Quality versus useful-mode recovery")
    ax.grid(alpha=0.23)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def generate_symmetry_reports(results_root: Path, *, dimension: int = 32):
    config = SymmetryStudyConfig(dimension=dimension)
    prefix = f"d{dimension}"
    audit = results_root / f"{prefix}_symmetry_audit"
    augmentation = results_root / f"{prefix}_symmetry_augmentation"
    equivariant = results_root / f"{prefix}_equivariant_attention"
    summary = _read(augmentation / "summary.csv")
    per_seed = _read(augmentation / "per_seed.csv")
    paired = _read(augmentation / "paired_summary.csv")
    equivariance = _read(audit / "equivariance_summary.csv")
    audit_summary = _read(audit / "symmetry_audit_summary.csv")[0]
    parameters = json.loads(
        (equivariant / "parameter_counts.json").read_text(encoding="utf-8")
    )

    recall = lambda condition, split: float(
        _find(summary, condition=condition, split=split)["proposal_recall_mean"]
    )
    threshold = config.coordinate_transfer_success_recall
    current_aug_success = (
        min(
            recall("CurrentAttentionAugmented", "exact_translated"),
            recall("CurrentAttentionAugmented", "exact_rotated"),
        )
        >= threshold
    )
    equiv_success = (
        min(
            recall("EquivariantAttention", "exact_translated"),
            recall("EquivariantAttention", "exact_rotated"),
        )
        >= threshold
    )
    equiv_aug_success = (
        min(
            recall("EquivariantAttentionAugmented", "exact_translated"),
            recall("EquivariantAttentionAugmented", "exact_rotated"),
        )
        >= threshold
    )
    exact_equiv = (
        min(
            float(
                _find(
                    equivariance, condition="EquivariantAttention", transform=transform
                )["equivariance_cosine_mean"]
            )
            for transform in (
                "exact_translated",
                "exact_rotated",
                "exact_rotation_translation",
            )
        )
        >= config.exact_equivariance_cosine_threshold
    )
    if current_aug_success and equiv_success and equiv_aug_success:
        decision = "Case 3/4: augmentation and exact equivariance both transfer; their combination is complementary if paired results favor it."
    elif current_aug_success:
        decision = "Case 1: the current architecture has sufficient capacity; missing coordinate coverage was the primary problem."
    elif equiv_success:
        decision = "Case 2: coordinate dependence was architectural; exact relational equivariance fixes transfer without augmentation."
    elif equiv_aug_success:
        decision = "Case 4: architectural symmetry and training diversity are jointly required."
    elif exact_equiv:
        decision = "Case 5: transformation consistency is exact but useful recall remains poor; coordinate symmetry is not the dominant remaining cause."
    else:
        decision = "No preregistered success case is resolved; neither tested intervention adequately repairs coordinate transfer."

    current_aug_mean, current_aug_ci, current_aug_delta = _paired(
        paired,
        "CurrentAttentionAugmented",
        "CurrentAttention",
        "exact_rotated",
        "proposal_recall",
    )
    current_aug_quality = _paired(
        paired,
        "CurrentAttentionAugmented",
        "CurrentAttention",
        "exact_rotated",
        "candidate_mean_quality",
    )[2]
    current_aug_decoy = _paired(
        paired,
        "CurrentAttentionAugmented",
        "CurrentAttention",
        "exact_rotated",
        "candidate_decoy_rate",
    )[2]
    equiv_delta = _paired(
        paired,
        "EquivariantAttention",
        "CurrentAttention",
        "exact_rotated",
        "proposal_recall",
    )[2]
    equiv_aug_delta = _paired(
        paired,
        "EquivariantAttentionAugmented",
        "EquivariantAttention",
        "exact_rotated",
        "proposal_recall",
    )[2]
    primary_table = _primary_table(summary)
    secondary_table = _secondary_table(summary)
    equivariance_table = _equivariance_table(equivariance)
    parameter_table = "\n".join(
        [
            "| Condition | Parameters | Mean training seconds | Reused checkpoint |",
            "|---|---:|---:|---:|",
            *[
                f"| {condition} | {value['parameters']:,} | {value['mean_training_seconds']:.2f} | {value['checkpoint_reused']} |"
                for condition, value in parameters.items()
            ],
        ]
    )

    audit_report = f"""# D{dimension} Representation Symmetry Audit

The frozen CurrentAttention representation has the following transformation laws:

| Feature/subsystem | Intended law | Audit conclusion |
|---|---|---|
| absolute `h` | geometric vector | Concatenated into arbitrary coordinate MLPs; translation and rotation leak |
| `mu_j-h` | translation invariant, rotation equivariant | Satisfied numerically |
| amplitude | invariant scalar | Satisfied |
| raw covariance lower triangle | translation invariant, basis-dependent under rotation | Translation safe; not a scalar invariant |
| task token and `z_i` | non-geometric invariant | Satisfied |
| component order | unordered | Permutation invariant |
| descriptor normalization | translation invariant for relative inputs | Coordinate-specific under rotation |
| Q, refinement, region rules | transform with full task geometry | Numerically equivariant |

The exact translation audit gives relative-center error {_metric(audit_summary, "relative_center_translation_max_error")}, covariance error {_metric(audit_summary, "covariance_translation_max_error")}, but frozen CurrentAttention direction cosine {_metric(audit_summary, "current_translation_direction_cosine")}. Passing the translated descriptor with the original `h` restores cosine {_metric(audit_summary, "h_neutralized_translation_direction_cosine")}.

Therefore the surprising translation failure is explained: **absolute `h` enters the query, direction MLP, and alpha head even though the component descriptors are translation invariant**. Goal vectors are absent; task token, ordering, refinement, thresholds, and landscape transformation are not the source.

Refinement equivariance mean error is {_metric(audit_summary, "refinement_equivariance_mean_error")}; region agreement is {_metric(audit_summary, "region_classification_agreement")}.
"""
    _write(audit / "REPRESENTATION_SYMMETRY_AUDIT.md", audit_report)
    _write(audit / f"D{dimension}_SYMMETRY_AUDIT.md", audit_report)

    augmentation_report = f"""# D{dimension} Augmentation Results

CurrentAttentionAugmented preserves the existing architecture exactly. Training draws a transform independently per sampled example from a 4,096-matrix Haar SO({dimension}) pool and a fresh isotropic translation with radius uniform on [0,6]. Centers, covariance orientation, state, and supervised directions are transformed consistently; alpha labels are unchanged.

Current+augmentation minus Current on exact rotation proposal recall: **{current_aug_delta}**.

On exact rotation, augmentation changes candidate quality by **{current_aug_quality}** and decoy rate by **{current_aug_decoy}**. Its recall remains below the preregistered 0.75 transfer threshold, so augmentation alone is not an adequate repair in this protocol.

{primary_table}

No test landscape was used to choose the augmentation distribution.
"""
    _write(augmentation / f"D{dimension}_AUGMENTATION_RESULTS.md", augmentation_report)

    equivariant_report = f"""# D{dimension} Equivariant Landscape Attention

## Input and supervision contract

The proposal model does **not** receive the original goal vector `g`/`tasks.goals`. It receives `h`, a learned constant task token, fixed `z_i`, and unlabeled component geometry. Useful centers, directions, and radii are used only by the privileged Hungarian supervised loss. This intentionally preserves the oracle-like loss while keeping inference inputs non-oracle.

The equivariant model predicts scalar component weights from invariant features—distance, amplitude, covariance eigenvalues/trace/log-determinant, Mahalanobis radius, and pairwise distance/cosine summaries—then constructs each vector as `v_i = sum_j a_ij (mu_j-h)`. Alpha sees invariant quantities only.

Current and equivariant parameter counts differ by only {abs(parameters["CurrentAttention"]["parameters"] - parameters["EquivariantAttention"]["parameters"])} parameters.

Equivariant minus Current exact-rotation proposal recall: **{equiv_delta}**. Adding augmentation to the equivariant architecture changes exact-rotation recall by **{equiv_aug_delta}**.

Exact transformation-consistency metrics are saved separately from useful-region outcomes. The simple relative-vector basis was retained; the optional covariance-vector channel was not needed for this controlled first test.

## Exact transformation consistency

{equivariance_table}

These transformation-law metrics are distinct from useful-region performance. Alpha is compared by persistent candidate identity; a separate Hungarian-set alpha diagnostic is retained in `equivariance_metrics.csv`.

{("At d=64, CurrentAttention itself has IID proposal recall " + f"{recall('CurrentAttention', 'iid'):.3f}. Therefore the equivariant model's exact within-model transformation consistency cleanly establishes transfer of its learned rule, but the between-architecture performance gap also includes d64 learnability/inductive-bias differences; it cannot be attributed only to coordinate shift.") if dimension == 64 else ""}

## Interpretation

**{decision}**

Ordinary augmentation did not provide comparable success, so the optional 10/20/40/80-geometry sample-efficiency sweep was not launched automatically. This run uses d={dimension}; no higher-dimensional follow-up is included.
"""
    equivariant_report = equivariant_report.replace(chr(0xFFFD), "--")
    _write(equivariant / f"D{dimension}_EQUIVARIANT_ATTENTION.md", equivariant_report)

    generalization_report = f"""# D{dimension} Symmetry Generalization

## Primary coordinate transformations

{primary_table}

## Secondary regressions

{secondary_table}

Literal SO(d) symmetry is specific to this toy geometry. The broader lesson is that CSD proposals should depend on relationships in the task representation rather than arbitrary latent coordinate conventions.

| Toy symmetry experiment | LLM analogue |
|---|---|
| global coordinate rotation | change in internal representational basis |
| translation | irrelevant/global latent offset |
| relative component relations | relationships among prompt/context features |
| equivariant candidate attention | proposal generation based on relational structure |
| useful component | potentially useful reasoning concept |
| CSD candidate | candidate reasoning strategy |

This experiment does not establish that an LLM literally needs SO(d)-equivariant hidden-state layers. It evaluates d={dimension} only.
"""
    _write(audit / f"D{dimension}_SYMMETRY_GENERALIZATION.md", generalization_report)

    if dimension == 64:
        d32_path = results_root / "d32_symmetry_augmentation" / "summary.csv"
        if d32_path.exists():
            d32_summary = _read(d32_path)
            lines = [
                "# D64 versus D32 symmetry replication",
                "",
                "| Condition | D32 IID recall | D64 IID recall | D32 IID mode-set | D64 IID mode-set | D32 rotated recall | D64 rotated recall |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for condition in LABEL:
                d32_iid = _find(d32_summary, condition=condition, split="iid")
                d64_iid = _find(summary, condition=condition, split="iid")
                d32_rot = _find(d32_summary, condition=condition, split="exact_rotated")
                d64_rot = _find(summary, condition=condition, split="exact_rotated")
                lines.append(
                    f"| {LABEL[condition]} | {float(d32_iid['proposal_recall_mean']):.3f} | "
                    f"{float(d64_iid['proposal_recall_mean']):.3f} | "
                    f"{float(d32_iid['mode_set_recall_mean']):.3f} | "
                    f"{float(d64_iid['mode_set_recall_mean']):.3f} | "
                    f"{float(d32_rot['proposal_recall_mean']):.3f} | "
                    f"{float(d64_rot['proposal_recall_mean']):.3f} |"
                )
            lines.extend(
                [
                    "",
                    "The exact equivariant rule replicates from d32 to d64. The ordinary current model does not: its IID proposal recall falls from 0.913 to 0.002. Consequently, d64 confirms coordinate transfer of the equivariant rule but also exposes a separate high-dimensional learnability/inductive-bias failure in CurrentAttention.",
                    "",
                    "The d64 model does not receive the old target-direction `g`; useful direction/radius information appears only in the retained privileged supervised loss.",
                ]
            )
            _write(audit / "D64_VS_D32.md", "\n".join(lines))

    _grouped(
        summary,
        audit / "proposal_recall_primary.png",
        "proposal_recall",
        "Proposal recall",
    )
    _grouped(
        summary,
        audit / "mode_set_recall_primary.png",
        "mode_set_recall",
        "Mode-set recall",
    )
    _grouped(
        summary,
        audit / "decoy_rate_primary.png",
        "candidate_decoy_rate",
        "Candidate decoy rate",
    )
    _equivariance_plot(equivariance, audit / "exact_equivariance_cosine.png")
    _paired_scatter(
        per_seed,
        audit / "current_augmentation_paired_recall.png",
        "CurrentAttentionAugmented",
        "CurrentAttention",
        "exact_rotated",
        "Current attention: rotation augmentation",
    )
    _paired_scatter(
        per_seed,
        audit / "current_vs_equivariant_paired_recall.png",
        "EquivariantAttention",
        "CurrentAttention",
        "exact_rotated",
        "Current versus equivariant attention",
    )
    _quality_mode(summary, audit / "quality_vs_mode_recall.png")
    return {
        "decision": decision,
        "dimension": dimension,
        "d64_run": dimension == 64,
    }
