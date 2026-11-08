"""Plots for the paired Gen2 direction-normalization study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _mean_ci(values: Sequence[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(data))
    ci = (
        float(1.96 * np.std(data, ddof=1) / np.sqrt(len(data)))
        if len(data) > 1
        else 0.0
    )
    return mean, ci


def plot_alpha_normalization_comparison(
    rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    metrics = [
        ("mean_quality", "Mean quality"),
        ("coverage", "Coverage"),
        ("decoy_rate", "Decoy rate"),
        ("useful_exploration_rate", "Useful exploration"),
        ("actual_perturbation_magnitude", "Actual perturbation magnitude"),
    ]
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    alphas = sorted({float(row["alpha"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    axes_flat = axes.ravel()
    for ax, (metric, title) in zip(axes_flat, metrics):
        for variant in variants:
            means = []
            errors = []
            for alpha in alphas:
                values = [
                    float(row[metric])
                    for row in rows
                    if str(row["variant"]) == variant and float(row["alpha"]) == alpha
                ]
                mean, ci = _mean_ci(values)
                means.append(mean)
                errors.append(ci)
            ax.errorbar(
                alphas,
                means,
                yerr=errors,
                marker="o",
                capsize=2.5,
                linewidth=1.7,
                label=variant.replace("gen2_", ""),
            )
        if metric == "actual_perturbation_magnitude":
            ax.plot(
                alphas,
                alphas,
                color="black",
                linestyle="--",
                alpha=0.55,
                label="y = alpha",
            )
        ax.set_title(title)
        ax.set_xlabel("Fixed alpha")
        ax.grid(alpha=0.25)
    axes_flat[0].set_ylabel("Quality")
    axes_flat[1].set_ylabel("Fraction")
    axes_flat[2].set_ylabel("Fraction")
    axes_flat[3].set_ylabel("Fraction")
    axes_flat[4].set_ylabel("L2 norm")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    handles_last, labels_last = axes_flat[4].get_legend_handles_labels()
    axes_flat[5].axis("off")
    axes_flat[5].legend(
        handles_last or handles, labels_last or labels, loc="center", frameon=False
    )
    fig.suptitle("Fixed-alpha sweep: unnormalized versus normalized directions")
    _save(fig, path)


def plot_perturbation_distributions(
    rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    metrics = [
        ("raw_norm", "Raw direction norm"),
        ("normalized_norm", "F.normalize output norm"),
        ("perturbation_norm", "Applied perturbation norm"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, (metric, title) in zip(axes, metrics):
        all_values = np.asarray([float(row[metric]) for row in rows])
        upper = max(float(np.quantile(all_values, 0.995)), 1e-6)
        bins = np.linspace(0.0, upper * 1.02, 45)
        for variant in variants:
            values = [
                float(row[metric]) for row in rows if str(row["variant"]) == variant
            ]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=variant.replace("gen2_", ""),
            )
        ax.set_title(title)
        ax.set_xlabel("L2 norm")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    fig.suptitle("Direction-scale diagnostic at the training alpha")
    _save(fig, path)


def plot_selector_comparison(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    selectors = ["learned_critic", "oracle", "random"]
    metrics = [
        ("selected_quality", "Selected quality"),
        ("target_hit_rate", "Target hit"),
        ("useful_exploration_rate", "Useful exploration"),
        ("decoy_rate", "Decoy rate"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.2))
    x = np.arange(len(selectors), dtype=np.float64)
    width = 0.36
    for ax, (metric, title) in zip(axes, metrics):
        for variant_index, variant in enumerate(variants):
            means = []
            errors = []
            for selector in selectors:
                values = [
                    float(row[metric])
                    for row in rows
                    if str(row["variant"]) == variant
                    and str(row["selector"]) == selector
                ]
                mean, ci = _mean_ci(values)
                means.append(mean)
                errors.append(ci)
            offset = (variant_index - 0.5 * (len(variants) - 1)) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=errors,
                capsize=2.5,
                label=variant.replace("gen2_", ""),
            )
        ax.set_xticks(
            x, [name.replace("_", " ") for name in selectors], rotation=30, ha="right"
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Selectors applied to identical generated candidate batches")
    _save(fig, path)
