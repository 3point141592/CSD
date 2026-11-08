"""Generation-3 learned-alpha and old-norm diagnostic plots."""

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
    error = (
        float(1.96 * np.std(data, ddof=1) / np.sqrt(len(data)))
        if len(data) > 1
        else 0.0
    )
    return mean, error


def plot_gen3_main_comparison(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    metrics = [
        ("selected_quality", "Selected quality"),
        ("coverage", "Coverage"),
        ("useful_exploration_rate", "Useful exploration"),
        ("decoy_rate", "Decoy rate"),
    ]
    labels = [
        name.replace("gen2_", "G2 ")
        .replace("gen3_", "G3 ")
        .replace("_alpha", "")
        .replace("_", " ")
        for name in variants
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.4))
    for ax, (metric, title) in zip(axes, metrics):
        means = []
        errors = []
        for variant in variants:
            values = [float(row[metric]) for row in rows if row["variant"] == variant]
            mean, error = _mean_ci(values)
            means.append(mean)
            errors.append(error)
        ax.bar(np.arange(len(variants)), means, yerr=errors, capsize=2.5)
        ax.set_xticks(np.arange(len(variants)), labels, rotation=38, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Generation 3: explicit magnitude parameterizations")
    _save(fig, path)


def plot_learned_alpha_histograms(
    records: Sequence[Mapping[str, object]], path: Path
) -> None:
    variants = [
        variant
        for variant in dict.fromkeys(str(row["variant"]) for row in records)
        if variant.startswith("gen3_")
    ]
    fig, axes = plt.subplots(
        1, len(variants), figsize=(4.4 * len(variants), 4.0), sharey=True
    )
    if len(variants) == 1:
        axes = [axes]
    bins = np.linspace(0.0, 6.0, 49)
    for ax, variant in zip(axes, variants):
        values = [
            float(row["candidate_alpha"])
            for row in records
            if row["variant"] == variant
        ]
        ax.hist(values, bins=bins, density=True, alpha=0.82)
        ax.axvline(
            3.0, color="black", linestyle="--", linewidth=1.2, label="initial alpha=3"
        )
        ax.set_title(
            variant.replace("gen3_", "").replace("_alpha", "").replace("_", " ")
        )
        ax.set_xlabel("Learned alpha")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.suptitle("Learned exploration-magnitude distributions")
    _save(fig, path)


def plot_alpha_by_candidate_and_goal(
    records: Sequence[Mapping[str, object]], variant: str, path: Path
) -> None:
    subset = [row for row in records if row["variant"] == variant]
    candidate_ids = sorted({int(row["candidate_id"]) for row in subset})
    goals = sorted({int(row["goal_index"]) for row in subset})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, groups, title in [
        (axes[0], "candidate_id", candidate_ids, "Alpha by candidate identity"),
        (axes[1], "goal_index", goals, "Alpha by goal"),
    ]:
        means = []
        errors = []
        for group in groups:
            values = [
                float(row["candidate_alpha"])
                for row in subset
                if int(row[key]) == group
            ]
            mean, error = _mean_ci(values)
            means.append(mean)
            errors.append(error)
        ax.bar(np.arange(len(groups)), means, yerr=errors, capsize=2.5)
        ax.axhline(3.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xticks(np.arange(len(groups)), [str(group) for group in groups])
        ax.set_xlabel(key.replace("_", " "))
        ax.set_ylabel("Mean learned alpha")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def _scatter_with_bins(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    stride = max(1, len(x) // 3500)
    ax.scatter(x[::stride], y[::stride], s=8, alpha=0.12)
    edges = np.quantile(x, np.linspace(0.0, 1.0, 11))
    centers = []
    means = []
    for index in range(10):
        mask = (x >= edges[index]) & (
            x <= edges[index + 1] if index == 9 else x < edges[index + 1]
        )
        if np.any(mask):
            centers.append(float(np.mean(x[mask])))
            means.append(float(np.mean(y[mask])))
    ax.plot(
        centers,
        means,
        marker="o",
        color="black",
        linewidth=2.0,
        label="quantile-bin mean",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)


def plot_alpha_relationships(
    records: Sequence[Mapping[str, object]], variant: str, path: Path
) -> None:
    subset = [row for row in records if row["variant"] == variant]
    alpha = np.asarray([float(row["candidate_alpha"]) for row in subset])
    quality = np.asarray([float(row["true_utility"]) for row in subset])
    critic = np.asarray([float(row["critic_score"]) for row in subset])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    _scatter_with_bins(
        axes[0],
        alpha,
        quality,
        xlabel="Learned alpha",
        ylabel="Refined candidate quality",
        title="Alpha versus candidate quality",
    )
    _scatter_with_bins(
        axes[1],
        alpha,
        critic,
        xlabel="Learned alpha",
        ylabel="Critic score",
        title="Alpha versus critic score",
    )
    _save(fig, path)


def plot_alpha_by_category(
    records: Sequence[Mapping[str, object]], variant: str, path: Path
) -> None:
    categories = ["anchor", "useful_non_target", "target_hit", "decoy", "other"]
    data = [
        [
            float(row["candidate_alpha"])
            for row in records
            if row["variant"] == variant and row.get("category") == category
        ]
        for category in categories
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    positions = [index + 1 for index, values in enumerate(data) if values]
    present = [values for values in data if values]
    present_labels = [category for category, values in zip(categories, data) if values]
    ax.boxplot(present, positions=positions, showfliers=False, whis=(5, 95))
    ax.axhline(3.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(
        positions,
        [label.replace("_", " ") for label in present_labels],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Learned alpha")
    ax.set_title("Contextual linear alpha by candidate outcome")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_old_norm_quality(records: Sequence[Mapping[str, object]], path: Path) -> None:
    norm = np.asarray([float(row["raw_norm"]) for row in records])
    quality = np.asarray([float(row["true_utility"]) for row in records])
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    _scatter_with_bins(
        ax,
        norm,
        quality,
        xlabel="Old unnormalized direction norm",
        ylabel="Refined candidate quality",
        title="Old vector magnitude versus candidate quality",
    )
    _save(fig, path)


def plot_old_norm_categories(
    records: Sequence[Mapping[str, object]], path: Path
) -> None:
    categories = ["anchor", "useful_non_target", "target_hit", "decoy", "other"]
    values = [
        [float(row["raw_norm"]) for row in records if row["category"] == category]
        for category in categories
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot(values, showfliers=False, whis=(5, 95))
    ax.set_xticks(
        np.arange(1, len(categories) + 1),
        [category.replace("_", " ") for category in categories],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Old direction norm")
    ax.set_title("Old magnitude by disjoint candidate type")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_old_norm_identity_goal(
    identity_rows: Sequence[Mapping[str, object]],
    goal_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, rows, key, title in [
        (axes[0], identity_rows, "candidate_id", "Old norm by candidate identity"),
        (axes[1], goal_rows, "goal_index", "Old norm by goal"),
    ]:
        ordered = sorted(rows, key=lambda row: int(row[key]))
        ax.bar(
            np.arange(len(ordered)),
            [float(row["mean_norm"]) for row in ordered],
        )
        ax.set_xticks(np.arange(len(ordered)), [str(int(row[key])) for row in ordered])
        ax.set_xlabel(key.replace("_", " "))
        ax.set_ylabel("Mean old direction norm")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, path)
