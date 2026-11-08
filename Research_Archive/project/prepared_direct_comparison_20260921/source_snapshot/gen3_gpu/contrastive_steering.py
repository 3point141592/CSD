"""Sample-derived contrast vectors and matched-budget candidate construction.

Region labels are permitted only in :func:`build_contrastive_vectors`, whose
``split`` argument rejects validation/test data.  Inference uses only the
current state, goal, frozen contrast vectors, and validation-selected radii.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from creativity_toy.landscapes_v2 import V2Task

from .config_contrastive import ContrastiveDataConfig

ALLOWED_TRAINING_SPLITS = {"train_iid", "train_rotated"}


@dataclass
class ContrastiveVectors:
    global_direction: torch.Tensor
    goal_directions: torch.Tensor
    target_anchor_directions: torch.Tensor
    useful_decoy_directions: torch.Tensor
    goal_prototypes: torch.Tensor
    raw_norms: dict[str, list[float]]
    sample_counts: dict[str, int]
    split: str
    task_fingerprint: str
    sampling_seconds: float
    vector_seconds: float
    construction_seconds: float

    def direction(self, kind: str, goals: torch.Tensor) -> torch.Tensor:
        if kind == "global":
            return self.global_direction.to(goals).expand(goals.shape[0], -1)
        tables = {
            "goal": self.goal_directions,
            "target_anchor": self.target_anchor_directions,
            "useful_decoy": self.useful_decoy_directions,
        }
        if kind not in tables:
            raise ValueError(f"unknown contrastive direction kind {kind!r}")
        prototypes = self.goal_prototypes.to(goals)
        normalized_goals = F.normalize(goals, p=2, dim=-1, eps=1e-8)
        match = torch.argmax(normalized_goals @ prototypes.T, dim=-1)
        return tables[kind].to(goals).index_select(0, match)


@dataclass
class ContrastiveCandidates:
    candidates: torch.Tensor
    directions: torch.Tensor
    alphas: torch.Tensor


def _task_fingerprint(tasks: Sequence[V2Task]) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        for value in [
            task.anchor,
            task.useful_centers,
            task.decoy_centers,
            task.goal,
            task.component_amplitudes,
        ]:
            array = np.asarray(value, dtype=np.float64)
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        digest.update(str(task.geometry_id).encode("ascii"))
        digest.update(str(task.goal_index).encode("ascii"))
        digest.update(task.variant.encode("utf-8"))
    return digest.hexdigest()


def _sample_matching_region(
    task: V2Task,
    *,
    center: np.ndarray,
    covariance: np.ndarray,
    region: int,
    count: int,
    rng: np.random.Generator,
    config: ContrastiveDataConfig,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    covariance = np.asarray(covariance, dtype=np.float64) * config.sample_scale**2
    for _ in range(config.max_rejection_batches):
        draw = rng.multivariate_normal(
            np.asarray(center, dtype=np.float64),
            covariance,
            size=max(config.rejection_batch, count),
        )
        accepted = draw[np.asarray(task.region(draw)) == region]
        rows.extend(accepted)
        if len(rows) >= count:
            return np.asarray(rows[:count], dtype=np.float64)
    raise RuntimeError(f"could not sample {count} states from region {region}")


def _sample_background(
    task: V2Task,
    *,
    count: int,
    rng: np.random.Generator,
    config: ContrastiveDataConfig,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    radius = task.config.decoy_radius * config.background_radius_scale
    for _ in range(config.max_rejection_batches):
        draw = task.anchor + rng.uniform(
            -radius, radius, size=(max(config.rejection_batch, count), task.dimension)
        )
        accepted = draw[np.asarray(task.region(draw)) == -1]
        rows.extend(accepted)
        if len(rows) >= count:
            return np.asarray(rows[:count], dtype=np.float64)
    raise RuntimeError(f"could not sample {count} background states")


def _normalize(array: np.ndarray) -> tuple[torch.Tensor, float]:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    norm = float(torch.linalg.vector_norm(tensor))
    if norm <= 1e-8:
        raise RuntimeError("contrastive class means produced a zero direction")
    return F.normalize(tensor, p=2, dim=-1, eps=1e-8), norm


def build_contrastive_vectors(
    tasks: Sequence[V2Task],
    *,
    split: str,
    seed: int,
    config: ContrastiveDataConfig,
) -> ContrastiveVectors:
    """Build vectors only from explicitly marked training tasks.

    Sampling is balanced through region/category centroids: every useful or
    decoy basin contributes equally within its category, and negative category
    means are averaged before the class contrast is formed.
    """
    if split not in ALLOWED_TRAINING_SPLITS:
        raise ValueError(
            f"contrast vectors may only use {sorted(ALLOWED_TRAINING_SPLITS)}, got {split!r}"
        )
    if not tasks:
        raise ValueError("the controlled contrastive benchmark requires nonempty tasks")
    dimension = tasks[0].dimension
    if any(task.dimension != dimension for task in tasks):
        raise ValueError("all contrastive training tasks must have the same dimension")
    started = perf_counter()
    rng = np.random.default_rng(seed)
    n = config.samples_per_region
    global_positive: list[np.ndarray] = []
    global_negative: list[np.ndarray] = []
    per_goal: dict[int, dict[str, list[np.ndarray]]] = {}
    counts = {
        "anchor": 0,
        "useful": 0,
        "target": 0,
        "non_target_useful": 0,
        "decoy": 0,
        "background": 0,
    }
    for task in tasks:
        anchor = _sample_matching_region(
            task,
            center=task.anchor,
            covariance=np.eye(dimension) * task.config.anchor_width**2,
            region=0,
            count=n,
            rng=rng,
            config=config,
        )
        useful = [
            _sample_matching_region(
                task,
                center=task.useful_centers[index],
                covariance=task.useful_covariances[index],
                region=1 + index,
                count=n,
                rng=rng,
                config=config,
            )
            for index in range(task.n_useful)
        ]
        decoys = [
            _sample_matching_region(
                task,
                center=task.decoy_centers[index],
                covariance=task.decoy_covariances[index],
                region=1 + task.n_useful + index,
                count=n,
                rng=rng,
                config=config,
            )
            for index in range(task.n_decoys)
        ]
        background = _sample_background(task, count=n, rng=rng, config=config)
        anchor_mean = np.mean(anchor, axis=0)
        useful_means = np.stack([np.mean(value, axis=0) for value in useful])
        decoy_mean = np.mean(
            np.stack([np.mean(value, axis=0) for value in decoys]), axis=0
        )
        background_mean = np.mean(background, axis=0)
        target_mean = useful_means[task.goal_index]
        non_target_mean = np.mean(
            np.delete(useful_means, task.goal_index, axis=0), axis=0
        )

        global_positive.extend(useful_means)
        global_negative.append(
            np.mean([anchor_mean, decoy_mean, background_mean], axis=0)
        )
        bucket = per_goal.setdefault(
            task.goal_index,
            {
                "prototype": [],
                "positive": [],
                "negative": [],
                "anchor": [],
                "decoy": [],
            },
        )
        bucket["prototype"].append(task.goal)
        bucket["positive"].append(target_mean)
        bucket["negative"].append(
            np.mean([anchor_mean, non_target_mean, decoy_mean, background_mean], axis=0)
        )
        bucket["anchor"].append(anchor_mean)
        bucket["decoy"].append(decoy_mean)
        counts["anchor"] += len(anchor)
        counts["useful"] += sum(len(value) for value in useful)
        counts["target"] += len(useful[task.goal_index])
        counts["non_target_useful"] += sum(
            len(value) for index, value in enumerate(useful) if index != task.goal_index
        )
        counts["decoy"] += sum(len(value) for value in decoys)
        counts["background"] += len(background)

    sampling_seconds = perf_counter() - started
    global_direction, global_norm = _normalize(
        np.mean(global_positive, axis=0) - np.mean(global_negative, axis=0)
    )
    goal_directions, ta_directions, ud_directions, prototypes = [], [], [], []
    goal_norms, ta_norms, ud_norms = [], [], []
    for goal_index in sorted(per_goal):
        bucket = per_goal[goal_index]
        goal, goal_norm = _normalize(
            np.mean(bucket["positive"], axis=0) - np.mean(bucket["negative"], axis=0)
        )
        target_anchor, ta_norm = _normalize(
            np.mean(bucket["positive"], axis=0) - np.mean(bucket["anchor"], axis=0)
        )
        useful_decoy, ud_norm = _normalize(
            np.mean(bucket["positive"], axis=0) - np.mean(bucket["decoy"], axis=0)
        )
        prototype, _ = _normalize(np.mean(bucket["prototype"], axis=0))
        goal_directions.append(goal)
        ta_directions.append(target_anchor)
        ud_directions.append(useful_decoy)
        prototypes.append(prototype)
        goal_norms.append(goal_norm)
        ta_norms.append(ta_norm)
        ud_norms.append(ud_norm)
    counts["total_raw_states"] = (
        counts["anchor"] + counts["useful"] + counts["decoy"] + counts["background"]
    )
    construction_seconds = perf_counter() - started
    return ContrastiveVectors(
        global_direction=global_direction,
        goal_directions=torch.stack(goal_directions),
        target_anchor_directions=torch.stack(ta_directions),
        useful_decoy_directions=torch.stack(ud_directions),
        goal_prototypes=torch.stack(prototypes),
        raw_norms={
            "global": [global_norm],
            "goal": goal_norms,
            "target_anchor": ta_norms,
            "useful_decoy": ud_norms,
        },
        sample_counts=counts,
        split=split,
        task_fingerprint=_task_fingerprint(tasks),
        sampling_seconds=sampling_seconds,
        vector_seconds=construction_seconds - sampling_seconds,
        construction_seconds=construction_seconds,
    )


def contrastive_candidates(
    starts: torch.Tensor,
    goals: torch.Tensor,
    vectors: ContrastiveVectors,
    *,
    kind: str,
    alpha: float,
    form: str,
    candidate_count: int,
) -> ContrastiveCandidates:
    direction = vectors.direction(kind, goals)
    if form == "single":
        radii = torch.full((1,), float(alpha), dtype=starts.dtype, device=starts.device)
    elif form == "k_radius":
        radii = torch.linspace(
            float(alpha) / candidate_count,
            float(alpha),
            candidate_count,
            dtype=starts.dtype,
            device=starts.device,
        )
    else:
        raise ValueError(f"unknown proposal form {form!r}")
    directions = direction[:, None, :].expand(-1, len(radii), -1)
    alphas = radii[None, :].expand(starts.shape[0], -1)
    candidates = starts[:, None, :] + alphas.unsqueeze(-1) * directions
    return ContrastiveCandidates(
        candidates=candidates, directions=directions, alphas=alphas
    )


def vectors_to_numpy(vectors: ContrastiveVectors) -> dict[str, np.ndarray]:
    return {
        "global_direction": vectors.global_direction.cpu().numpy(),
        "goal_directions": vectors.goal_directions.cpu().numpy(),
        "target_anchor_directions": vectors.target_anchor_directions.cpu().numpy(),
        "useful_decoy_directions": vectors.useful_decoy_directions.cpu().numpy(),
        "goal_prototypes": vectors.goal_prototypes.cpu().numpy(),
    }
