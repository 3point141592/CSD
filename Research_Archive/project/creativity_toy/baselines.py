"""Budget-matched baseline and creative-operator evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import TrainingConfig
from .landscapes import GaussianIslandLandscape
from .operators import (
    CreativeOperator,
    LinearCritic,
    pure_divergence_candidates,
    random_candidates,
)

Array = np.ndarray


def evaluate_method(
    *,
    method: str,
    landscapes: Sequence[GaussianIslandLandscape],
    starts: Sequence[Sequence[Array]],
    operator: CreativeOperator,
    alpha: float,
    config: TrainingConfig,
    seed: int,
    critic: LinearCritic | None = None,
) -> list[dict[str, object]]:
    """Evaluate a method with the creative portfolio's candidate budget."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    heads = operator.heads

    for landscape_id, (landscape, landscape_starts) in enumerate(
        zip(landscapes, starts)
    ):
        for trial, h in enumerate(landscape_starts):
            creative_candidates = operator.apply(h, alpha)
            magnitudes = np.linalg.norm(creative_candidates - h[None, :], axis=-1)

            if method == "baseline":
                candidates = np.repeat(h[None, :], heads, axis=0)
            elif method == "random":
                candidates = random_candidates(h, magnitudes, rng)
            elif method == "pure_divergence":
                candidates = pure_divergence_candidates(h, magnitudes, rng)
            elif method == "creative":
                candidates = creative_candidates
            elif method in {"critic_selected", "critic_mixture"}:
                if critic is None:
                    raise ValueError(f"{method} requires a fitted critic")
                features = LinearCritic.features(
                    h, creative_candidates, np.mean(landscape.centers, axis=0)
                )
                if method == "critic_selected":
                    chosen = int(np.argmax(critic.predict(features)))
                    candidates = creative_candidates[chosen : chosen + 1]
                else:
                    candidates = critic.combine(creative_candidates, features)[None, :]
            else:
                raise ValueError(f"Unknown method {method!r}")

            refined = landscape.refine(
                candidates,
                steps=config.refinement_steps,
                rate=config.refinement_rate,
            )
            qualities = landscape.quality(refined)
            islands = landscape.island(refined)
            perturbations = np.linalg.norm(candidates - h[None, :], axis=-1)
            for candidate_id in range(candidates.shape[0]):
                records.append(
                    {
                        "method": method,
                        "alpha": float(alpha),
                        "landscape_id": int(landscape_id),
                        "trial": int(trial),
                        "candidate_id": int(candidate_id),
                        "n_islands": int(landscape.n_islands),
                        "quality": float(qualities[candidate_id]),
                        "island": int(islands[candidate_id]),
                        "perturbation": float(perturbations[candidate_id]),
                        "start": np.asarray(h).copy(),
                        "candidate": candidates[candidate_id].copy(),
                        "final": refined[candidate_id].copy(),
                        # Counting refinement and quality queries keeps method budgets comparable.
                        "evaluation_budget": int(config.refinement_steps + 1),
                    }
                )
    return records
