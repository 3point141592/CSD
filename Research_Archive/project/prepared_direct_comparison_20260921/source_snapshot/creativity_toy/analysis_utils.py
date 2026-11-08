"""Small dependency-free statistical helpers shared by Gen3 diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .normalization_diagnostics import spearman_rank_correlation


def statistics(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        return {"mean": 0.0, "sd": 0.0, "ci95": 0.0, "n": 0.0}
    sd = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    return {
        "mean": float(np.mean(data)),
        "sd": sd,
        "ci95": float(1.96 * sd / np.sqrt(len(data))) if len(data) > 1 else 0.0,
        "n": float(len(data)),
    }


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    return spearman_rank_correlation(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )


def classify_candidate(row: Mapping[str, object], useful_threshold: float) -> str:
    """Disjoint precedence: target hit, useful non-target, decoy, anchor, other."""
    quality = float(row.get("true_utility", row.get("quality", 0.0)))
    is_target = str(row.get("is_target", "False")).lower() == "true" or bool(
        row.get("is_target") is True
    )
    is_useful = str(row.get("is_useful", "False")).lower() == "true" or bool(
        row.get("is_useful") is True
    )
    is_decoy = str(row.get("is_decoy", "False")).lower() == "true" or bool(
        row.get("is_decoy") is True
    )
    region = int(float(row.get("region", -1)))
    if is_target and quality >= useful_threshold:
        return "target_hit"
    if is_useful and quality >= useful_threshold:
        return "useful_non_target"
    if is_decoy:
        return "decoy"
    if region == 0:
        return "anchor"
    return "other"
