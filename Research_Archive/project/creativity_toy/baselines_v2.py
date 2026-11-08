"""Budget-matched V2 baseline and operator evaluation."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from .config_v2 import TrainingV2Config
from .creative_operator_v2 import BaseCreativeOperatorV2
from .critics import GoalConditionalRidgeCritic
from .landscapes_v2 import V2Task

Array = np.ndarray


def random_candidates_v2(
    h: Array,
    candidates: int,
    radius: float,
    rng: np.random.Generator,
) -> Array:
    epsilon = rng.normal(size=(candidates, h.size)) * (radius / np.sqrt(h.size))
    return h[None, :] + epsilon


def pure_divergence_candidates_v2(
    h: Array,
    candidates: int,
    radius: float,
    rng: np.random.Generator,
) -> Array:
    """Maximally spread, fixed-radius directions with no goal query."""
    if h.size == 2:
        offset = rng.uniform(0.0, 2.0 * np.pi)
        angles = offset + np.linspace(0.0, 2.0 * np.pi, candidates, endpoint=False)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    else:
        directions = rng.normal(size=(candidates, h.size))
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
        )
    return h[None, :] + radius * directions


def evaluate_method_v2(
    *,
    method: str,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[Array]],
    config: TrainingV2Config,
    seed: int,
    candidates: int,
    operator: BaseCreativeOperatorV2 | None = None,
    critic: GoalConditionalRidgeCritic | None = None,
    alpha: float | None = None,
    record_trajectories: bool = True,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    active_alpha = config.alpha if alpha is None else alpha
    records: list[dict[str, object]] = []
    start_time = time.perf_counter()

    for task_id, (task, task_starts) in enumerate(zip(tasks, starts)):
        for trial, h in enumerate(task_starts):
            if method == "baseline":
                raw_candidates = np.repeat(h[None, :], candidates, axis=0)
            elif method == "random":
                raw_candidates = random_candidates_v2(
                    h, candidates, config.random_radius, rng
                )
            elif method == "pure_divergence":
                raw_candidates = pure_divergence_candidates_v2(
                    h, candidates, config.pure_divergence_radius, rng
                )
            else:
                if operator is None:
                    raise ValueError(f"Method {method!r} requires an operator")
                raw_candidates = operator.apply(h, task.goal, active_alpha)

            if record_trajectories:
                refined, trajectories = task.refine(
                    raw_candidates,
                    steps=config.refinement_steps,
                    rate=config.refinement_rate,
                    return_trajectory=True,
                )
            else:
                refined = task.refine(
                    raw_candidates,
                    steps=config.refinement_steps,
                    rate=config.refinement_rate,
                )
                trajectories = np.empty((len(raw_candidates), 0, task.dimension))
            qualities = task.quality(refined)
            regions = task.region(refined)
            oracle_id = int(np.argmax(qualities))
            if critic is None or method.endswith("no_critic"):
                selected_id = 0
            else:
                features = GoalConditionalRidgeCritic.features(
                    h, raw_candidates, task.goal
                )
                selected_id = critic.select(features)

            for candidate_id in range(len(raw_candidates)):
                region = int(regions[candidate_id])
                records.append(
                    {
                        "method": method,
                        "task_id": int(task_id),
                        "geometry_id": int(task.geometry_id),
                        "goal_index": int(task.goal_index),
                        "target_region": int(task.target_region),
                        "trial": int(trial),
                        "candidate_id": int(candidate_id),
                        "n_useful": int(task.n_useful),
                        "quality": float(qualities[candidate_id]),
                        "region": region,
                        "is_useful": bool(1 <= region <= task.n_useful),
                        "is_decoy": bool(region > task.n_useful),
                        "is_target": bool(region == task.target_region),
                        "selected": bool(candidate_id == selected_id),
                        "oracle_best": bool(candidate_id == oracle_id),
                        "perturbation": float(
                            np.linalg.norm(raw_candidates[candidate_id] - h)
                        ),
                        "displacement": float(
                            np.linalg.norm(refined[candidate_id] - h)
                        ),
                        "objective_evaluations": int(config.refinement_steps + 1),
                        "optimization_steps": int(config.refinement_steps),
                        "start": np.asarray(h).copy(),
                        "goal": task.goal.copy(),
                        "candidate": raw_candidates[candidate_id].copy(),
                        "final": np.asarray(refined[candidate_id]).copy(),
                        "trajectory": trajectories[candidate_id].copy(),
                    }
                )
    elapsed = time.perf_counter() - start_time
    for record in records:
        record["method_wall_clock_seconds"] = float(elapsed)
    return records
