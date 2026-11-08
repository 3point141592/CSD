"""Generation-2 visualizations for geometry, tradeoffs, and ablations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .landscapes_v2 import V2Task


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_method_geometry_v2(
    task: V2Task,
    records_by_method: Mapping[str, Sequence[Mapping[str, object]]],
    path: Path,
) -> None:
    if task.dimension != 2:
        return
    all_centers = np.vstack(
        [task.anchor[None, :], task.useful_centers, task.decoy_centers]
    )
    low = np.min(all_centers, axis=0) - 2.0
    high = np.max(all_centers, axis=0) + 2.0
    xx, yy = np.meshgrid(
        np.linspace(low[0], high[0], 230),
        np.linspace(low[1], high[1], 230),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    zz = task.quality(grid).reshape(xx.shape)

    methods = list(records_by_method)
    fig, axes = plt.subplots(
        1, len(methods), figsize=(4.0 * len(methods), 4.2), sharex=True, sharey=True
    )
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        ax.contourf(xx, yy, zz, levels=24, cmap="viridis", alpha=0.84)
        ax.contour(
            xx,
            yy,
            zz,
            levels=[0.25, 0.5, 0.75, 0.9],
            colors="white",
            linewidths=0.4,
            alpha=0.6,
        )
        ax.scatter(
            task.anchor[0],
            task.anchor[1],
            marker="*",
            s=150,
            c="gold",
            edgecolor="black",
            label="obvious",
        )
        ax.scatter(
            task.useful_centers[:, 0],
            task.useful_centers[:, 1],
            marker="o",
            s=75,
            c=[
                "lime" if i == task.goal_index else "cyan" for i in range(task.n_useful)
            ],
            edgecolor="black",
            label="useful",
        )
        ax.scatter(
            task.decoy_centers[:, 0],
            task.decoy_centers[:, 1],
            marker="X",
            s=70,
            c="crimson",
            edgecolor="black",
            label="decoy",
        )
        rows = records_by_method[method]
        starts = np.stack([np.asarray(row["start"]) for row in rows])
        finals = np.stack([np.asarray(row["final"]) for row in rows])
        ax.scatter(
            starts[:, 0], starts[:, 1], marker="s", c="white", edgecolor="black", s=24
        )
        ax.scatter(finals[:, 0], finals[:, 1], marker=".", c="magenta", s=24, alpha=0.8)
        selected = [row for row in rows if bool(row["selected"])]
        if selected:
            points = np.stack([np.asarray(row["final"]) for row in selected])
            ax.scatter(
                points[:, 0],
                points[:, 1],
                marker="P",
                c="orange",
                edgecolor="black",
                s=75,
            )
        for row in rows[: min(len(rows), 16)]:
            trajectory = np.asarray(row["trajectory"])
            if trajectory.size:
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color="white",
                    alpha=0.24,
                    linewidth=0.7,
                )
        ax.arrow(
            task.anchor[0],
            task.anchor[1],
            task.goal[0],
            task.goal[1],
            width=0.025,
            color="gold",
            length_includes_head=True,
        )
        ax.set_title(method.replace("_", " "))
        ax.set_xlabel("state dimension 1")
    axes[0].set_ylabel("state dimension 2")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Asymmetric goal-conditioned landscape: endpoints and trajectories")
    _save(fig, path)


def plot_quality_diversity_v2(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        x = np.array([float(row["euclidean_diversity"]) for row in subset])
        y = np.array([float(row["mean_quality"]) for row in subset])
        ax.scatter(np.mean(x), np.mean(y), s=100, label=method.replace("_", " "))
        ax.errorbar(
            np.mean(x),
            np.mean(y),
            xerr=1.96 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0,
            yerr=1.96 * np.std(y, ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0,
            fmt="none",
            capsize=3,
            alpha=0.8,
        )
    ax.set_xlabel("high-quality Euclidean diversity")
    ax.set_ylabel("mean quality")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_quality_displacement_v2(
    records: Sequence[Mapping[str, object]], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    methods = list(dict.fromkeys(str(row["method"]) for row in records))
    for method in methods:
        subset = [row for row in records if row["method"] == method]
        if len(subset) > 1800:
            subset = subset[:: max(1, len(subset) // 1800)]
        ax.scatter(
            [float(row["displacement"]) for row in subset],
            [float(row["quality"]) for row in subset],
            s=10,
            alpha=0.22,
            label=method.replace("_", " "),
        )
    ax.set_xlabel("final displacement from obvious state")
    ax.set_ylabel("goal-conditioned quality")
    ax.set_title("Farther is not necessarily better")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_metric_bars_v2(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    metrics = ["coverage", "useful_exploration_rate", "decoy_rate", "collapse_rate"]
    titles = ["coverage", "useful exploration", "decoy rate", "collapse"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric, title in zip(axes, metrics, titles):
        means = []
        errors = []
        for method in methods:
            values = np.array(
                [float(row[metric]) for row in rows if row["method"] == method]
            )
            means.append(np.mean(values))
            errors.append(
                1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
                if len(values) > 1
                else 0
            )
        ax.bar(np.arange(len(methods)), means, yerr=errors, capsize=3)
        ax.set_xticks(
            np.arange(len(methods)),
            [m.replace("_", " ") for m in methods],
            rotation=45,
            ha="right",
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_goal_conditioning_v2(
    records: Sequence[Mapping[str, object]], useful_peaks: int, path: Path
) -> None:
    matrix = np.zeros((useful_peaks, useful_peaks + 2), dtype=np.float64)
    for row in records:
        if not bool(row["selected"]):
            continue
        goal = int(row["goal_index"])
        region = int(row["region"])
        if 1 <= region <= useful_peaks:
            column = region - 1
        elif region > useful_peaks:
            column = useful_peaks
        else:
            column = useful_peaks + 1
        matrix[goal, column] += 1
    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(
        np.arange(useful_peaks + 2),
        [f"useful {i}" for i in range(useful_peaks)] + ["decoy", "anchor/other"],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(np.arange(useful_peaks), [f"goal {i}" for i in range(useful_peaks)])
    ax.set_xlabel("critic-selected region")
    ax.set_ylabel("task goal")
    fig.colorbar(image, ax=ax, label="selection probability")
    _save(fig, path)


def plot_training_v2(
    histories: Mapping[str, Sequence[Mapping[str, float]]], path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(
        axes.ravel(),
        ["reward", "quality", "weighted_diversity", "low_quality_distance"],
    ):
        for name, history in histories.items():
            ax.plot([float(row[metric]) for row in history], label=name)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("ES iteration")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    _save(fig, path)


def plot_grouped_study_v2(
    rows: Sequence[Mapping[str, object]],
    group_key: str,
    path: Path,
) -> None:
    groups = list(dict.fromkeys(str(row[group_key]) for row in rows))
    metrics = ["selected_quality", "coverage", "decoy_rate", "selected_target_rate"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        values = [
            np.mean(
                [float(row[metric]) for row in rows if str(row[group_key]) == group]
            )
            for group in groups
        ]
        ax.bar(np.arange(len(groups)), values)
        ax.set_xticks(np.arange(len(groups)), groups, rotation=40, ha="right")
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    _save(fig, path)
