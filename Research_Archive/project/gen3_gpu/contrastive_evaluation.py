"""Common candidate-set evaluation for contrastive and frozen CSD methods."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from creativity_toy.baselines_v2 import (
    pure_divergence_candidates_v2,
    random_candidates_v2,
)
from creativity_toy.config_gen3 import Gen3StudyConfig
from creativity_toy.landscapes_v2 import V2Task

from .config_router import RouterExperimentConfig
from .contrastive_steering import ContrastiveCandidates
from .critic import BatchedRidgeCritic, critic_features
from .evaluation_mode import FrozenPairwiseRouter
from .router_useful_novelty import FrozenCandidateBatch, normalized_candidate_novelty
from .runtime import synchronize
from .task_batch import pack_tasks

SELECTORS = (
    "random",
    "quality_router",
    "frozen_pairwise_router",
    "quality_oracle",
    "useful_oracle",
)


@dataclass
class CandidateEvaluation:
    batch: FrozenCandidateBatch
    directions: torch.Tensor
    portfolio: dict[str, float]
    selector_rows: list[dict[str, object]]
    set_rows: list[dict[str, object]]
    elapsed_seconds: float
    candidate_evaluations: int
    q_equivalent_evaluations: int


def flatten_starts(
    starts: Sequence[Sequence[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, task_ids, trials = [], [], []
    for task_id, task_starts in enumerate(starts):
        for trial, point in enumerate(task_starts):
            points.append(np.asarray(point, dtype=np.float64))
            task_ids.append(task_id)
            trials.append(trial)
    return np.stack(points), np.asarray(task_ids), np.asarray(trials)


def established_baseline_candidates(
    starts: np.ndarray,
    *,
    method: str,
    candidates: int,
    random_radius: float,
    pure_divergence_radius: float,
    seed: int,
    device: torch.device,
) -> ContrastiveCandidates:
    rng = np.random.default_rng(seed)
    rows = []
    for point in starts:
        if method == "random":
            rows.append(random_candidates_v2(point, candidates, random_radius, rng))
        elif method == "pure_divergence":
            rows.append(
                pure_divergence_candidates_v2(
                    point, candidates, pure_divergence_radius, rng
                )
            )
        else:
            raise ValueError(f"unknown established baseline {method!r}")
    candidate_tensor = torch.as_tensor(
        np.stack(rows), dtype=torch.float32, device=device
    )
    start_tensor = torch.as_tensor(starts, dtype=torch.float32, device=device)
    delta = candidate_tensor - start_tensor[:, None, :]
    alphas = torch.linalg.vector_norm(delta, dim=-1)
    directions = F.normalize(delta, p=2, dim=-1, eps=1e-8)
    return ContrastiveCandidates(candidate_tensor, directions, alphas)


@torch.no_grad()
def make_evaluated_batch(
    *,
    source: str,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    proposal: ContrastiveCandidates,
    base: Gen3StudyConfig,
    router_config: RouterExperimentConfig,
    device: torch.device,
    refinement: bool,
) -> tuple[FrozenCandidateBatch, int, int, float]:
    start_array, task_ids_array, trials_array = flatten_starts(starts)
    packed = pack_tasks(tasks, device=device, dtype=torch.float32)
    task_ids = torch.as_tensor(task_ids_array, dtype=torch.long, device=device)
    trials = torch.as_tensor(trials_array, dtype=torch.long, device=device)
    batch_tasks = packed.index_select(task_ids)
    h = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    candidates = proposal.candidates.detach()
    synchronize(device)
    started = perf_counter()
    raw_quality = batch_tasks.quality(candidates).detach()
    if refinement:
        refined = batch_tasks.refine(
            candidates,
            steps=base.base.training.refinement_steps,
            rate=base.base.training.refinement_rate,
        ).detach()
        quality = batch_tasks.quality(refined).detach()
        q_calls_per_candidate = base.base.training.refinement_steps + 1
    else:
        refined = candidates
        quality = raw_quality
        q_calls_per_candidate = 1
    regions = batch_tasks.regions(refined).detach()
    novelty = normalized_candidate_novelty(
        candidates,
        diversity_scale=router_config.diversity_scale,
        novelty_clip=router_config.novelty_clip,
    ).detach()
    synchronize(device)
    elapsed = perf_counter() - started
    count = int(candidates.shape[0] * candidates.shape[1])
    batch = FrozenCandidateBatch(
        starts=h,
        goals=batch_tasks.goals,
        candidates=candidates,
        refined=refined,
        raw_quality=raw_quality,
        quality=quality,
        regions=regions,
        alphas=proposal.alphas.detach(),
        task_ids=task_ids,
        trials=trials,
        n_useful=batch_tasks.n_useful,
        target_regions=batch_tasks.target_region,
        novelty=novelty,
        source=source,
    )
    return batch, count, count * q_calls_per_candidate, elapsed


def _selected(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, indices[:, None]).squeeze(1)


def _coverage_collapse(
    batch: FrozenCandidateBatch,
    indices: torch.Tensor,
    useful_threshold: float,
) -> tuple[float, float]:
    regions = _selected(batch.regions, indices)
    quality = _selected(batch.quality, indices)
    coverages, collapses = [], []
    for task_id in torch.unique(batch.task_ids).detach().cpu().tolist():
        mask = batch.task_ids == int(task_id)
        n_useful = int(batch.n_useful[mask][0].cpu())
        good = (
            (regions[mask] >= 1)
            & (regions[mask] <= n_useful)
            & (quality[mask] >= useful_threshold)
        )
        hits = regions[mask][good].detach().cpu().tolist()
        coverages.append(len(set(int(value) for value in hits)) / n_useful)
        if len(hits) < 2:
            collapses.append(1.0)
        else:
            counts = Counter(int(value) for value in hits)
            same = sum(value * (value - 1) for value in counts.values())
            collapses.append(same / (len(hits) * (len(hits) - 1)))
    return float(np.mean(coverages)), float(np.mean(collapses))


@torch.no_grad()
def evaluate_candidate_batch(
    *,
    method: str,
    batch: FrozenCandidateBatch,
    directions: torch.Tensor,
    quality_router: BatchedRidgeCritic,
    pairwise_router: FrozenPairwiseRouter,
    useful_threshold: float,
    random_seed: int,
    elapsed_seconds: float,
    candidate_evaluations: int,
    q_equivalent_evaluations: int,
) -> CandidateEvaluation:
    useful = (
        (batch.regions >= 1)
        & (batch.regions <= batch.n_useful[:, None])
        & (batch.quality >= useful_threshold)
    )
    target = (batch.regions == batch.target_regions[:, None]) & (
        batch.quality >= useful_threshold
    )
    decoy = batch.regions > batch.n_useful[:, None]
    anchor = batch.regions == 0
    proposal_hit = torch.any(useful, dim=1)
    useful_scores = batch.quality + useful.float() + target.float()
    rng = np.random.default_rng(random_seed)
    random_indices = torch.as_tensor(
        rng.integers(0, batch.candidate_count, size=batch.batch_size),
        dtype=torch.long,
        device=batch.quality.device,
    )
    selector_indices = {
        "random": random_indices,
        "quality_router": torch.argmax(
            quality_router.predict(
                critic_features(batch.starts, batch.candidates, batch.goals)
            ),
            dim=1,
        ),
        "frozen_pairwise_router": torch.argmax(
            pairwise_router.scores(batch.starts, batch.candidates, batch.goals), dim=1
        ),
        "quality_oracle": torch.argmax(batch.quality, dim=1),
        "useful_oracle": torch.argmax(useful_scores, dim=1),
    }
    useful_counts = []
    set_rows: list[dict[str, object]] = []
    for index in range(batch.batch_size):
        represented = torch.unique(batch.regions[index][useful[index]])
        count = int(represented.numel())
        useful_counts.append(count)
        set_rows.append(
            {
                "method": method,
                "set_index": index,
                "task_id": int(batch.task_ids[index].cpu()),
                "trial": int(batch.trials[index].cpu()),
                "proposal_recall": float(proposal_hit[index].cpu()),
                "useful_region_count": count,
                "candidate_mean_quality": float(torch.mean(batch.quality[index]).cpu()),
                "candidate_decoy_rate": float(torch.mean(decoy[index].float()).cpu()),
                "candidate_target_rate": float(torch.mean(target[index].float()).cpu()),
                "candidate_anchor_rate": float(torch.mean(anchor[index].float()).cpu()),
            }
        )
    useful_count = torch.as_tensor(
        useful_counts, dtype=torch.float32, device=batch.quality.device
    )
    if batch.candidate_count > 1:
        upper = torch.triu_indices(
            batch.candidate_count,
            batch.candidate_count,
            offset=1,
            device=batch.quality.device,
        )
        cosine = torch.matmul(directions, directions.transpose(-1, -2))
        cosine_distance = torch.clamp(1.0 - cosine[:, upper[0], upper[1]], 0.0, 2.0)
        spatial = torch.cdist(batch.refined, batch.refined)[:, upper[0], upper[1]]
        mean_cosine_distance = float(torch.mean(cosine_distance).cpu())
        mean_spatial = float(torch.mean(spatial).cpu())
    else:
        mean_cosine_distance = 0.0
        mean_spatial = 0.0
    singular = torch.linalg.svdvals(directions)
    energy = singular.square()
    probability = energy / torch.clamp(
        torch.sum(energy, dim=1, keepdim=True), min=1e-12
    )
    effective_rank = torch.exp(
        -torch.sum(
            torch.where(
                probability > 0,
                probability * torch.log(probability),
                torch.zeros_like(probability),
            ),
            dim=1,
        )
    )
    selector_rows: list[dict[str, object]] = []
    for selector, indices in selector_indices.items():
        selected_quality = _selected(batch.quality, indices)
        selected_useful = _selected(useful, indices).float()
        selected_target = _selected(target, indices).float()
        selected_decoy = _selected(decoy, indices).float()
        selected_anchor = _selected(anchor, indices).float()
        coverage, collapse = _coverage_collapse(batch, indices, useful_threshold)
        routing_recall = torch.sum(
            selected_useful * proposal_hit.float()
        ) / torch.clamp(torch.sum(proposal_hit.float()), min=1.0)
        selector_rows.append(
            {
                "method": method,
                "selector": selector,
                "selected_quality": float(torch.mean(selected_quality).cpu()),
                "proposal_recall": float(torch.mean(proposal_hit.float()).cpu()),
                "routing_recall": float(routing_recall.cpu()),
                "useful_selected": float(torch.mean(selected_useful).cpu()),
                "target_selected": float(torch.mean(selected_target).cpu()),
                "decoy_selected": float(torch.mean(selected_decoy).cpu()),
                "anchor_selected": float(torch.mean(selected_anchor).cpu()),
                "coverage": coverage,
                "collapse": collapse,
            }
        )
    pairwise = next(
        row for row in selector_rows if row["selector"] == "frozen_pairwise_router"
    )
    portfolio = {
        "candidate_count": float(batch.candidate_count),
        "candidate_mean_quality": float(torch.mean(batch.quality).cpu()),
        "proposal_recall": float(torch.mean(proposal_hit.float()).cpu()),
        "useful_region_count": float(torch.mean(useful_count).cpu()),
        "useful_region_count_2_plus": float(
            torch.mean((useful_count >= 2).float()).cpu()
        ),
        "candidate_decoy_rate": float(torch.mean(decoy.float()).cpu()),
        "candidate_target_rate": float(torch.mean(target.float()).cpu()),
        "candidate_anchor_rate": float(torch.mean(anchor.float()).cpu()),
        "candidate_useful_rate": float(torch.mean(useful.float()).cpu()),
        "candidate_spatial_diversity": mean_spatial,
        "direction_cosine_distance": mean_cosine_distance,
        "effective_directional_rank": float(torch.mean(effective_rank).cpu()),
        "quality_oracle_quality": float(
            torch.mean(torch.amax(batch.quality, dim=1)).cpu()
        ),
        "useful_oracle_success": float(torch.mean(proposal_hit.float()).cpu()),
        "selected_quality": float(pairwise["selected_quality"]),
        "useful_selected": float(pairwise["useful_selected"]),
        "target_selected": float(pairwise["target_selected"]),
        "selected_decoy_rate": float(pairwise["decoy_selected"]),
        "routing_recall": float(pairwise["routing_recall"]),
        "alpha_mean": float(torch.mean(batch.alphas).cpu()),
        "alpha_sd": float(torch.std(batch.alphas, correction=0).cpu()),
        "evaluation_seconds": elapsed_seconds,
        "candidate_evaluations": float(candidate_evaluations),
        "q_equivalent_evaluations": float(q_equivalent_evaluations),
    }
    return CandidateEvaluation(
        batch=batch,
        directions=directions,
        portfolio=portfolio,
        selector_rows=selector_rows,
        set_rows=set_rows,
        elapsed_seconds=elapsed_seconds,
        candidate_evaluations=candidate_evaluations,
        q_equivalent_evaluations=q_equivalent_evaluations,
    )
