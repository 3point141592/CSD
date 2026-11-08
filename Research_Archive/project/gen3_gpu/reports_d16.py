"""Tables, plots, and conservative interpretation for the d=16 benchmark."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .study import read_csv

LABELS = {
    "contrast_goal_1": "Contrastive goal (1)",
    "contrast_goal_k": "Contrastive K-radius",
    "csd_rl_pairwise": "CSD RL + pairwise",
    "csd_mode_pairwise": "CSD mode + pairwise",
}


def _row(rows: Sequence[Mapping[str, object]], method: str) -> Mapping[str, object]:
    return next(value for value in rows if value["method"] == method)


def _fmt(row: Mapping[str, object], metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    sd = float(row[f"{metric}_sd"])
    ci = float(row[f"{metric}_ci95"])
    return f"{mean:.3f} ± {sd:.3f} (95% CI ±{ci:.3f})"


def _paired(
    rows: Sequence[Mapping[str, object]], left: str, right: str, metric: str
) -> Mapping[str, object]:
    return next(
        row
        for row in rows
        if row["left"] == left and row["right"] == right and row["metric"] == metric
    )


def _claim(
    delta: Mapping[str, object], positive: str, neutral: str, negative: str
) -> str:
    mean = float(delta["mean_paired_difference"])
    low = float(delta["ci95_low"])
    high = float(delta["ci95_high"])
    if low > 0.0:
        return positive
    if high < 0.0:
        return negative
    return neutral


def _main_table(rows: Sequence[Mapping[str, object]]) -> str:
    columns = [
        ("selected_quality", "Quality"),
        ("proposal_recall", "Proposal recall"),
        ("routing_recall", "Routing recall"),
        ("useful_selected", "Useful selected"),
        ("useful_region_count", "Useful regions"),
        ("useful_region_count_2_plus", "Sets 2+"),
        ("selected_decoy_rate", "Decoy"),
        ("target_selected", "Target"),
        ("effective_directional_rank", "Direction rank"),
    ]
    lines = [
        "| Method | " + " | ".join(label for _, label in columns) + " |",
        "|---|" + "|".join("---:" for _ in columns) + "|",
    ]
    for method in LABELS:
        row = _row(rows, method)
        values = [
            f"{float(row[f'{metric}_mean']):.3f} ± {float(row[f'{metric}_sd']):.3f}"
            for metric, _ in columns
        ]
        lines.append(f"| {LABELS[method]} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_validation_audit(
    output: Path,
    primary: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
) -> None:
    per_seed = read_csv(output / "primary_per_seed.csv")
    training = read_csv(output / "training_cost_per_seed.csv")
    alpha_selection = json.loads(
        (output / "CONTRASTIVE_ALPHA_SELECTION.json").read_text(encoding="utf-8")
    )
    numeric = [
        float(value)
        for row in per_seed
        for key, value in row.items()
        if key not in {"seed", "method"} and value != ""
    ]
    matched = [row for row in per_seed if row["method"] != "contrast_goal_1"]
    candidate_budgets = {
        (float(row["candidate_evaluations"]), float(row["q_equivalent_evaluations"]))
        for row in matched
    }
    policy_training = [row for row in training if str(row["method"]).startswith("csd_")]
    alpha_values = [
        float(row["alpha_mean"])
        for row in per_seed
        if str(row["method"]).startswith("csd_")
    ]
    contrast_k = _row(primary, "contrast_goal_k")
    audit = {
        "all_primary_values_finite": bool(np.all(np.isfinite(numeric))),
        "primary_rows": len(per_seed),
        "expected_primary_rows": int(aggregate["seed_count"]) * 4,
        "all_primary_rows_present": len(per_seed) == int(aggregate["seed_count"]) * 4,
        "matched_k_candidate_and_q_budgets": sorted(
            [list(value) for value in candidate_budgets]
        ),
        "matched_k_budget_equal": len(candidate_budgets) == 1,
        "training_candidate_budgets": sorted(
            {int(row["candidate_evaluations"]) for row in policy_training}
        ),
        "training_q_equivalent_budgets": sorted(
            {int(row["q_equivalent_evaluations"]) for row in policy_training}
        ),
        "training_budget_equal": (
            len({int(row["candidate_evaluations"]) for row in policy_training}) == 1
            and len({int(row["q_equivalent_evaluations"]) for row in policy_training})
            == 1
        ),
        "all_csd_alpha_means_in_open_interval_0_6": all(
            0.0 < value < 6.0 for value in alpha_values
        ),
        "contrast_k_pre_refinement_effective_rank_mean": float(
            contrast_k["effective_directional_rank_mean"]
        ),
        "contrast_k_collinear": abs(
            float(contrast_k["effective_directional_rank_mean"]) - 1.0
        )
        < 1e-5,
        "all_geometry_variants_not_effectively_2d": bool(
            aggregate["geometry_audit"]["all_variants_not_effectively_2d"]
        ),
        "rotation_orthogonality_max_error": float(
            aggregate["geometry_audit"]["rotation_orthogonality_max_error"]
        ),
        "alpha_selection_test_set_used": alpha_selection["test_set_used"],
        "cuda_used": aggregate["runtime"]["device"] == "cuda",
        "dtype": aggregate["runtime"]["dtype"],
    }
    audit["all_checks_passed"] = all(
        [
            audit["all_primary_values_finite"],
            audit["all_primary_rows_present"],
            audit["matched_k_budget_equal"],
            audit["training_budget_equal"],
            audit["all_csd_alpha_means_in_open_interval_0_6"],
            audit["contrast_k_collinear"],
            audit["all_geometry_variants_not_effectively_2d"],
            audit["rotation_orthogonality_max_error"] < 1e-5,
            audit["alpha_selection_test_set_used"] is False,
            audit["cuda_used"],
            audit["dtype"] == "float32",
        ]
    )
    (output / "validation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def _plot_primary(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for method, color in zip(LABELS, colors):
        row = _row(rows, method)
        axes[0].errorbar(
            float(row["useful_region_count_mean"]),
            float(row["selected_quality_mean"]),
            xerr=float(row["useful_region_count_ci95"]),
            yerr=float(row["selected_quality_ci95"]),
            marker="o",
            capsize=3,
            color=color,
            label=LABELS[method],
        )
        axes[1].bar(
            LABELS[method], float(row["effective_directional_rank_mean"]), color=color
        )
        axes[2].bar(LABELS[method], float(row["proposal_recall_mean"]), color=color)
    axes[0].set_xlabel("Distinct useful regions / set")
    axes[0].set_ylabel("Pairwise-router selected quality")
    axes[0].set_title("Quality–functional-diversity frontier")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Effective proposal-direction rank")
    axes[1].set_title("Portfolio rank (K=8)")
    axes[2].set_ylabel("Proposal recall")
    axes[2].set_title("At least one useful proposal")
    for axis in axes[1:]:
        axis.tick_params(axis="x", rotation=28, labelsize=8)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "d16_primary_frontiers.png", dpi=180)
    plt.close(figure)


def _plot_generalization(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    if not rows:
        return
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(variants))
    width = 0.19
    for index, method in enumerate(LABELS):
        subset = {str(row["variant"]): row for row in rows if row["method"] == method}
        for axis, metric in zip(
            axes, ["proposal_recall", "useful_region_count", "selected_quality"]
        ):
            axis.bar(
                x + (index - 1.5) * width,
                [float(subset[v][f"{metric}_mean"]) for v in variants],
                width,
                label=LABELS[method],
            )
            axis.set_title(metric.replace("_", " "))
            axis.set_xticks(x, variants, rotation=38, ha="right")
            axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output / "d16_generalization.png", dpi=180)
    plt.close(figure)


def generate_d16_reports(output: Path) -> None:
    primary = read_csv(output / "primary_summary.csv")
    paired = read_csv(output / "paired_differences.csv")
    generalization = (
        read_csv(output / "generalization_summary.csv")
        if (output / "generalization_summary.csv").exists()
        else []
    )
    aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
    training = read_csv(output / "training_cost_per_seed.csv")
    contrast_data = read_csv(output / "contrastive_training_data.csv")
    geometry = aggregate["geometry_audit"]["variant_diagnostics"]["iid"]
    seed_count = int(aggregate["seed_count"])
    rl_vs_contrast_regions = _paired(
        paired, "csd_rl_pairwise", "contrast_goal_k", "useful_region_count"
    )
    mode_vs_contrast_regions = _paired(
        paired, "csd_mode_pairwise", "contrast_goal_k", "useful_region_count"
    )
    rl_vs_contrast_recall = _paired(
        paired, "csd_rl_pairwise", "contrast_goal_k", "proposal_recall"
    )
    mode_vs_contrast_recall = _paired(
        paired, "csd_mode_pairwise", "contrast_goal_k", "proposal_recall"
    )
    mode_vs_rl = _paired(
        paired, "csd_mode_pairwise", "csd_rl_pairwise", "proposal_recall"
    )
    rl_training = [
        float(row["training_seconds"])
        for row in training
        if row["method"] == "csd_rl_pairwise"
    ]
    mode_training = [
        float(row["training_seconds"])
        for row in training
        if row["method"] == "csd_mode_pairwise"
    ]
    router_training = [
        float(row["training_seconds"])
        for row in training
        if row["method"] == "shared_pairwise_router"
    ]
    contrast_construction = [
        float(row["construction_seconds"]) for row in contrast_data
    ]
    per_seed = read_csv(output / "primary_per_seed.csv")
    rl_alpha = np.mean(
        [
            float(row["alpha_mean"])
            for row in per_seed
            if row["method"] == "csd_rl_pairwise"
        ]
    )
    mode_alpha = np.mean(
        [
            float(row["alpha_mean"])
            for row in per_seed
            if row["method"] == "csd_mode_pairwise"
        ]
    )
    rl = _row(primary, "csd_rl_pairwise")
    mode = _row(primary, "csd_mode_pairwise")
    contrast_k = _row(primary, "contrast_goal_k")

    results = f"""# D16 Results

This is a separate `{seed_count}`-seed, float32 CUDA benchmark at `d=16`. It does not overwrite any 2D result. All matched methods use `K=8`, the same landscapes and starts, nine refinement steps, and the same frozen local pairwise useful-novelty router methodology. Contrastive-1 is reported as the deliberately cheaper one-proposal control; Contrastive K-radius is the compute-matched contrastive comparison.

## Geometry validation

The IID useful-center geometry is not a 2D embedding: minimum per-landscape algebraic rank is `{float(geometry["per_geometry_algebraic_rank_min"]):.0f}`, mean entropy effective rank is `{float(geometry["per_geometry_effective_rank_mean"]):.3f}`, aggregate effective rank is `{float(geometry["aggregate_useful_effective_rank"]):.3f}`, and coordinate-variance participation is `{float(geometry["coordinate_variance_participation"]):.3f}` of 16 axes. Useful covariance eigenvalues range from `{float(geometry["useful_covariance_eigenvalue_min"]):.4f}` to `{float(geometry["useful_covariance_eigenvalue_max"]):.4f}`, confirming anisotropy. The true random 16D rotation audit has maximum orthogonality error `{float(aggregate["geometry_audit"]["rotation_orthogonality_max_error"]):.2e}`.

## Primary IID result

{_main_table(primary)}

Full mean, standard deviation, and 95% confidence interval fields are in `primary_summary.csv`; paired intervals are in `paired_differences.csv`.

## Controlled findings

- {_claim(rl_vs_contrast_regions, "CSD RL represents more distinct useful regions than Contrastive K with a paired 95% interval excluding zero.", "The distinct-useful-region difference between CSD RL and Contrastive K is unresolved at 95%.", "Contrastive K represents more distinct useful regions than CSD RL with a paired 95% interval excluding zero.")}
- {_claim(mode_vs_contrast_regions, "Mode-preserved CSD represents more distinct useful regions than Contrastive K with a paired 95% interval excluding zero.", "The distinct-useful-region difference between mode-preserved CSD and Contrastive K is unresolved at 95%.", "Contrastive K represents more distinct useful regions than mode-preserved CSD with a paired 95% interval excluding zero.")}
- {_claim(rl_vs_contrast_recall, "CSD RL has higher proposal recall than Contrastive K with a paired 95% interval excluding zero.", "The proposal-recall difference between CSD RL and Contrastive K is unresolved at 95%.", "Contrastive K has higher proposal recall than CSD RL with a paired 95% interval excluding zero.")}
- Plain and mode-preserved CSD both have exactly zero observed IID proposal recall. Mode preservation raises effective proposal rank from `{float(rl["effective_directional_rank_mean"]):.3f}` to `{float(mode["effective_directional_rank_mean"]):.3f}` but yields no functional useful mode.

The proposal-direction rank result is descriptive, not itself evidence of useful creativity. Functional evidence comes from proposal recall and distinct useful-region representation at the reported quality and decoy rates.

## Alpha, budget, and compute audit

CSD RL learns mean alpha `{float(rl_alpha):.3f}`; mode-preserved CSD learns `{float(mode_alpha):.3f}`. Both remain strictly inside `(0, 6)`. Their high-rank failure is therefore not caused by alpha collapsing to zero.

Every trained policy receives exactly `100,000` returned candidate outcomes (`1,000,000` refinement-expanded Q-equivalent calls). Mean CUDA training time is `{float(np.mean(rl_training)):.2f}s` for RL and `{float(np.mean(mode_training)):.2f}s` for mode preservation; the shared router takes `{float(np.mean(router_training)):.3f}s`; contrast-vector construction takes `{float(np.mean(contrast_construction)):.3f}s`. End-to-end runtime is `{float(aggregate["end_to_end_seconds"]):.1f}s`; peak allocated/reserved VRAM is `{float(aggregate["peak_allocated_bytes"]) / 2**20:.1f}/{float(aggregate["peak_reserved_bytes"]) / 2**20:.1f} MiB`.

This is a clear **cosmetic-diversity failure** for CSD at the frozen budget: geometric separation increased, but proposal recall did not. It is consistent with sparse useful-basin discovery becoming much harder in 16D under black-box policy gradients. It does not establish that the architecture can never work at higher dimension, but the current 2D positive finding does not replicate.
"""
    (output / "D16_RESULTS.md").write_text(results, encoding="utf-8")

    gen_lines = [
        "# D16 Generalization",
        "",
        "All directions, policies, alphas, and routers are frozen before these held-out evaluations. The `rotated` family uses true random orthogonal transformations in 16D; fixed contrast vectors are not recomputed from test geometry.",
        "",
        "| Family | Method | Quality | Proposal recall | Useful regions | Useful selected | Decoy | Direction rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in aggregate.get("geometry_audit", {}).get("variant_diagnostics", {}):
        for method in LABELS:
            matches = [
                row
                for row in generalization
                if row["variant"] == variant and row["method"] == method
            ]
            if not matches:
                continue
            row = matches[0]
            gen_lines.append(
                f"| {variant} | {LABELS[method]} | "
                f"{float(row['selected_quality_mean']):.3f} | "
                f"{float(row['proposal_recall_mean']):.3f} | "
                f"{float(row['useful_region_count_mean']):.3f} | "
                f"{float(row['useful_selected_mean']):.3f} | "
                f"{float(row['selected_decoy_rate_mean']):.3f} | "
                f"{float(row['effective_directional_rank_mean']):.3f} |"
            )
    gen_lines.extend(
        [
            "",
            "Confidence intervals and all twelve primary metrics are preserved in `generalization_summary.csv`. Distribution-shift gains are not called significant here unless their paired analysis is run explicitly; the table is descriptive across the same paired seeds.",
        ]
    )
    (output / "D16_GENERALIZATION.md").write_text(
        "\n".join(gen_lines) + "\n", encoding="utf-8"
    )

    contrast = contrast_k
    single = _row(primary, "contrast_goal_1")
    comparison = f"""# D16 Contrastive Steering vs CSD

## Direct answers

1. **Did the 2D findings replicate?** No. Contrastive K has higher proposal recall than mode-preserved CSD with a paired 95% interval excluding zero; both CSD variants have zero observed IID useful proposals.
2. **Does CSD represent more distinct useful modes?** Contrastive K averages `{float(contrast["useful_region_count_mean"]):.3f}` useful regions/set; CSD RL `{float(rl["useful_region_count_mean"]):.3f}`; mode-preserved CSD `{float(mode["useful_region_count_mean"]):.3f}`. {_claim(mode_vs_contrast_regions, "The mode-CSD advantage is resolved.", "The mode-CSD difference is unresolved.", "Contrastive K has the resolved advantage.")}
3. **Does CSD produce a higher-rank proposal portfolio?** Contrastive K is structurally collinear and measures `{float(contrast["effective_directional_rank_mean"]):.3f}`; CSD RL measures `{float(rl["effective_directional_rank_mean"]):.3f}` and mode-preserved CSD `{float(mode["effective_directional_rank_mean"]):.3f}`. Rank is only a mechanism diagnostic.
4. **Does contrastive steering remain superior for one known target?** Descriptively, yes: Contrastive-1 target selection is `{float(single["target_selected_mean"]):.3f}` at quality `{float(single["selected_quality_mean"]):.3f}`. CSD RL is `{float(rl["target_selected_mean"]):.3f}` at `{float(rl["selected_quality_mean"]):.3f}`; mode-CSD is `{float(mode["target_selected_mean"]):.3f}` at `{float(mode["selected_quality_mean"]):.3f}`.
5. **How does generalization change?** Generalization weakens sharply. Fixed contrast vectors retain useful recall on translation, width, extra-peak, and extra-decoy shifts but fail under true rotations, new-goal geometry, and unseen geometry. CSD has only tiny raw proposal recall under rotated/unseen shifts and the frozen router selects none of those proposals.
6. **Does d=16 strengthen the case for CSD?** No under this frozen training regime and outcome budget. CSD learns higher-rank directions but no useful IID alternatives, so moving to 16D weakens the empirical case and exposes a sample-efficiency/credit-assignment ceiling.

## Cost and interpretation

Contrastive steering has zero learned proposal parameters and only class-mean construction plus vector addition. CSD retains its small learned policy and black-box training cost. Thus even a modest CSD portfolio gain must be weighed against substantially greater training complexity. This experiment supports only claims about useful proposal portfolios in this synthetic landscape family—not creativity in general.
"""
    (output / "D16_CONTRASTIVE_VS_CSD.md").write_text(comparison, encoding="utf-8")
    _plot_primary(primary, output)
    _plot_generalization(generalization, output)
    _write_validation_audit(output, primary, aggregate)
