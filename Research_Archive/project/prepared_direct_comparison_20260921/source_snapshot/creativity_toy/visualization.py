"""Plotting helpers. Metrics are always computed in the original space."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .landscapes import GaussianIslandLandscape


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    histories: Mapping[str, Sequence[Mapping[str, float]]], path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fields = ["reward", "quality", "gated_diversity", "semantic_penalty"]
    labels = [
        "training reward",
        "refined quality",
        "quality-gated diversity",
        "semantic penalty",
    ]
    for ax, field, label in zip(axes.ravel(), fields, labels):
        for name, history in histories.items():
            ax.plot([row[field] for row in history], label=name, alpha=0.85)
        ax.set_xlabel("ES iteration")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    _save(fig, path)


def plot_alpha_metric(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), float(row["alpha"]))].append(float(row[metric]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = sorted({key[0] for key in grouped})
    for method in methods:
        alphas = sorted(alpha for candidate, alpha in grouped if candidate == method)
        means = np.array([np.mean(grouped[(method, alpha)]) for alpha in alphas])
        errors = np.array(
            [
                1.96
                * np.std(grouped[(method, alpha)], ddof=1)
                / np.sqrt(len(grouped[(method, alpha)]))
                if len(grouped[(method, alpha)]) > 1
                else 0.0
                for alpha in alphas
            ]
        )
        ax.plot(alphas, means, marker="o", label=method.replace("_", " "))
        ax.fill_between(alphas, means - errors, means + errors, alpha=0.14)
    ax.set_xlabel(r"creativity coefficient $\alpha$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_quality_diversity_frontier(
    rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    grouped: dict[tuple[str, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), float(row["alpha"]))].append(row)
    markers = {"baseline": "o", "random": "s", "pure_divergence": "^", "creative": "*"}
    for (method, alpha), values in sorted(grouped.items()):
        quality = np.mean([float(row["mean_quality"]) for row in values])
        diversity = np.mean([float(row["euclidean_diversity"]) for row in values])
        ax.scatter(
            diversity,
            quality,
            marker=markers.get(method, "o"),
            s=45 + 30 * alpha,
            alpha=0.8,
            label=method.replace("_", " ")
            if alpha == min(k[1] for k in grouped if k[0] == method)
            else None,
        )
        if method == "creative":
            ax.annotate(f"{alpha:g}", (diversity, quality), fontsize=7)
    ax.set_xlabel("high-quality Euclidean diversity")
    ax.set_ylabel("mean solution quality")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_2d_discoveries(
    landscape: GaussianIslandLandscape,
    method_records: Mapping[str, Sequence[Mapping[str, object]]],
    path: Path,
) -> None:
    if landscape.dimension != 2:
        raise ValueError("2D discovery plot requires a two-dimensional landscape")
    centers = landscape.centers
    margin = 2.0
    x_min, y_min = np.min(centers, axis=0) - margin
    x_max, y_max = np.max(centers, axis=0) + margin
    x_grid = np.linspace(x_min, x_max, 180)
    y_grid = np.linspace(y_min, y_max, 180)
    xx, yy = np.meshgrid(x_grid, y_grid)
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    zz = landscape.quality(points).reshape(xx.shape)

    methods = list(method_records)
    fig, axes = plt.subplots(
        1, len(methods), figsize=(4.1 * len(methods), 4), sharex=True, sharey=True
    )
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        ax.contourf(xx, yy, zz, levels=20, cmap="viridis", alpha=0.82)
        ax.scatter(
            centers[:, 0], centers[:, 1], marker="X", c="white", edgecolor="black", s=80
        )
        finals = np.stack([np.asarray(row["final"]) for row in method_records[method]])
        ax.scatter(
            finals[:, 0],
            finals[:, 1],
            c="tomato",
            s=24,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_title(method.replace("_", " "))
        ax.set_xlabel("latent dimension 1")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("latent dimension 2")
    fig.suptitle("2D solution islands and discovered endpoints")
    _save(fig, path)


def plot_low_rank(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    labels = [
        "full" if int(row["rank"]) < 0 else f"r={int(row['rank'])}" for row in rows
    ]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, metric, title in zip(
        axes,
        ["mean_quality", "euclidean_diversity", "coverage"],
        ["quality", "diversity", "coverage"],
    ):
        ax.bar(x, [float(row[metric]) for row in rows], color="#4c78a8")
        ax.set_xticks(x, labels, rotation=30)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_generalization(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    variants = [str(row["variant"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(
        axes,
        ["mean_quality", "euclidean_diversity", "coverage"],
        ["quality", "diversity", "coverage"],
    ):
        ax.bar(x, [float(row[metric]) for row in rows], color="#59a14f")
        ax.set_xticks(x, variants, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, path)
