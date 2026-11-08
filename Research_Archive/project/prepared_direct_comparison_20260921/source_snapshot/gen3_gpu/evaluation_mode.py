"""Generator-side portfolio metrics with a frozen downstream pairwise router."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from creativity_toy.landscapes_v2 import V2Task

from .black_box_env import BlackBoxLandscapeEnvironment
from .creative_policy_rl import StochasticCreativePolicy
from .critic import critic_features
from .runtime import synchronize
from .task_batch import TorchTaskBatch

SELECTORS = ("random", "frozen_pairwise_router", "quality_oracle", "useful_oracle")


@dataclass
class FrozenPairwiseRouter:
    coefficients: torch.Tensor
    feature_mode: str
    target_mode: str
    parameter_hash: str

    @property
    def parameter_count(self) -> int:
        return int(self.coefficients.numel())

    def scores(
        self, starts: torch.Tensor, candidates: torch.Tensor, goals: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(
            critic_features(starts, candidates, goals), self.coefficients
        )


@dataclass
class ModeEvaluationResult:
    summary: dict[str, float]
    selector_rows: list[dict[str, object]]
    set_rows: list[dict[str, object]]
    candidate_rows: list[dict[str, object]]
    elapsed_seconds: float
    tensors: dict[str, torch.Tensor]


def _tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_frozen_pairwise_router(
    seed: int, device: torch.device
) -> FrozenPairwiseRouter:
    path = (
        Path("results/router_useful_novelty_main/routers")
        / f"seed_{seed}_rl_gated_useful_novelty_pairwise_router.npz"
    )
    payload = np.load(path, allow_pickle=False)
    coefficients = torch.as_tensor(
        payload["coefficients"], dtype=torch.float32, device=device
    )
    return FrozenPairwiseRouter(
        coefficients=coefficients,
        feature_mode=str(payload["feature_mode"]),
        target_mode=str(payload["target_mode"]),
        parameter_hash=_tensor_hash(coefficients),
    )


def router_integrity(
    router: FrozenPairwiseRouter, before_hash: str
) -> dict[str, object]:
    after = _tensor_hash(router.coefficients)
    return {
        "router_hash_before": before_hash,
        "router_hash_after": after,
        "router_unchanged": before_hash == after,
        "maximum_router_parameter_difference": 0.0
        if before_hash == after
        else float("nan"),
    }


def _flatten_starts(
    starts: Sequence[Sequence[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, task_ids, trials = [], [], []
    for task_id, rows in enumerate(starts):
        for trial, point in enumerate(rows):
            points.append(np.asarray(point, dtype=np.float64))
            task_ids.append(task_id)
            trials.append(trial)
    return np.stack(points), np.asarray(task_ids), np.asarray(trials)


def _upper_values(matrix: torch.Tensor) -> torch.Tensor:
    count = matrix.shape[-1]
    upper = torch.triu_indices(count, count, offset=1, device=matrix.device)
    return matrix[:, upper[0], upper[1]]


def _selected(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, indices[:, None]).squeeze(1)


@torch.no_grad()
def evaluate_mode_policy(
    *,
    variant: str,
    policy: StochasticCreativePolicy,
    environment: BlackBoxLandscapeEnvironment,
    router: FrozenPairwiseRouter,
    tasks: Sequence[V2Task],
    packed_tasks: TorchTaskBatch,
    starts: Sequence[Sequence[np.ndarray]],
    action_seed: int,
    random_seed: int,
    useful_threshold: float,
    deterministic: bool = False,
) -> ModeEvaluationResult:
    start_array, task_ids, trials = _flatten_starts(starts)
    device = next(policy.parameters()).device
    h = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    task_index = torch.as_tensor(task_ids, dtype=torch.long, device=device)
    batch_tasks = packed_tasks.index_select(task_index)
    generator = torch.Generator(device=device)
    generator.manual_seed(action_seed)
    synchronize(device)
    started = perf_counter()
    actions = policy.act(
        h,
        batch_tasks.goals,
        generator=None if deterministic else generator,
        deterministic=deterministic,
    )
    outcome = environment.evaluate(actions.candidate, batch_tasks)
    regions = batch_tasks.regions(outcome.refined)
    useful_region = (regions >= 1) & (regions <= batch_tasks.n_useful[:, None])
    good_useful = useful_region & (outcome.quality >= useful_threshold)
    target_good = (regions == batch_tasks.target_region[:, None]) & (
        outcome.quality >= useful_threshold
    )
    decoy = regions > batch_tasks.n_useful[:, None]
    anchor = regions == 0
    proposal = torch.any(good_useful, dim=1)
    scores = router.scores(h, actions.candidate, batch_tasks.goals)
    pairwise_index = torch.argmax(scores, dim=1)
    quality_index = torch.argmax(outcome.quality, dim=1)
    useful_scores = outcome.quality + good_useful.float() + target_good.float()
    useful_index = torch.argmax(useful_scores, dim=1)
    rng = np.random.default_rng(random_seed)
    random_index = torch.as_tensor(
        rng.integers(0, policy.candidates, size=len(h)),
        dtype=torch.long,
        device=device,
    )
    selector_indices = {
        "random": random_index,
        "frozen_pairwise_router": pairwise_index,
        "quality_oracle": quality_index,
        "useful_oracle": useful_index,
    }

    cosine = torch.matmul(actions.direction, actions.direction.transpose(-1, -2))
    cosine_values = _upper_values(cosine)
    cosine_distance_values = torch.clamp(1.0 - cosine_values, min=0.0, max=2.0)
    spatial_values = _upper_values(torch.cdist(outcome.refined, outcome.refined, p=2))
    singular = torch.linalg.svdvals(actions.direction)
    energy = singular * singular
    probabilities = energy / torch.clamp(
        torch.sum(energy, dim=1, keepdim=True), min=1e-12
    )
    effective_rank = torch.exp(
        -torch.sum(
            torch.where(
                probabilities > 0,
                probabilities * torch.log(probabilities),
                torch.zeros_like(probabilities),
            ),
            dim=1,
        )
    )

    useful_counts: list[int] = []
    collapse_values: list[float] = []
    set_rows: list[dict[str, object]] = []
    for set_index in range(len(h)):
        represented = torch.unique(regions[set_index][good_useful[set_index]])
        useful_count = int(represented.numel())
        useful_counts.append(useful_count)
        hit_regions = regions[set_index][good_useful[set_index]].detach().cpu().tolist()
        if len(hit_regions) < 2:
            collapse = 1.0
        else:
            counts = Counter(int(value) for value in hit_regions)
            same = sum(value * (value - 1) for value in counts.values())
            collapse = same / (len(hit_regions) * (len(hit_regions) - 1))
        collapse_values.append(collapse)
        row: dict[str, object] = {
            "variant": variant,
            "set_index": set_index,
            "task_id": int(task_ids[set_index]),
            "trial": int(trials[set_index]),
            "geometry_id": int(tasks[int(task_ids[set_index])].geometry_id),
            "goal_index": int(tasks[int(task_ids[set_index])].goal_index),
            "proposal_recall": float(proposal[set_index].cpu()),
            "useful_region_count": useful_count,
            "useful_region_fraction": useful_count
            / int(batch_tasks.n_useful[set_index].cpu()),
            "direction_cosine_similarity": float(
                torch.mean(cosine_values[set_index]).cpu()
            ),
            "direction_cosine_distance": float(
                torch.mean(cosine_distance_values[set_index]).cpu()
            ),
            "minimum_direction_distance": float(
                torch.min(cosine_distance_values[set_index]).cpu()
            ),
            "effective_directional_rank": float(effective_rank[set_index].cpu()),
            "candidate_spatial_diversity": float(
                torch.mean(spatial_values[set_index]).cpu()
            ),
            "candidate_mean_quality": float(
                torch.mean(outcome.quality[set_index]).cpu()
            ),
            "candidate_decoy_rate": float(torch.mean(decoy[set_index].float()).cpu()),
            "candidate_useful_rate": float(
                torch.mean(good_useful[set_index].float()).cpu()
            ),
            "candidate_target_rate": float(
                torch.mean(target_good[set_index].float()).cpu()
            ),
            "alpha_mean": float(torch.mean(actions.alpha[set_index]).cpu()),
            "alpha_sd": float(torch.std(actions.alpha[set_index], correction=0).cpu()),
            "portfolio_collapse": collapse,
        }
        for selector, indices in selector_indices.items():
            index = int(indices[set_index].cpu())
            row[f"{selector}_index"] = index
            row[f"{selector}_quality"] = float(outcome.quality[set_index, index].cpu())
            row[f"{selector}_useful"] = float(good_useful[set_index, index].cpu())
            row[f"{selector}_target"] = float(target_good[set_index, index].cpu())
            row[f"{selector}_decoy"] = float(decoy[set_index, index].cpu())
            row[f"{selector}_anchor"] = float(anchor[set_index, index].cpu())
        set_rows.append(row)

    selector_rows: list[dict[str, object]] = []
    for selector, indices in selector_indices.items():
        selected_quality = _selected(outcome.quality, indices)
        selected_useful = _selected(good_useful, indices).float()
        selected_target = _selected(target_good, indices).float()
        selected_decoy = _selected(decoy, indices).float()
        selected_anchor = _selected(anchor, indices).float()
        proposal_count = torch.sum(proposal.float())
        routing_recall = torch.sum(selected_useful * proposal.float()) / torch.clamp(
            proposal_count, min=1.0
        )
        selector_rows.append(
            {
                "variant": variant,
                "selector": selector,
                "selected_quality": float(torch.mean(selected_quality).cpu()),
                "proposal_recall": float(torch.mean(proposal.float()).cpu()),
                "routing_recall": float(routing_recall.cpu()),
                "useful_selected": float(torch.mean(selected_useful).cpu()),
                "target_selected": float(torch.mean(selected_target).cpu()),
                "decoy_selected": float(torch.mean(selected_decoy).cpu()),
                "anchor_selected": float(torch.mean(selected_anchor).cpu()),
            }
        )

    candidate_rows: list[dict[str, object]] = []
    direction_cpu = actions.direction.detach().cpu().numpy()
    final_cpu = outcome.refined.detach().cpu().numpy()
    candidate_cpu = actions.candidate.detach().cpu().numpy()
    for set_index in range(len(h)):
        task = tasks[int(task_ids[set_index])]
        for candidate_id in range(policy.candidates):
            region = int(regions[set_index, candidate_id].cpu())
            quality = float(outcome.quality[set_index, candidate_id].cpu())
            is_target = bool(target_good[set_index, candidate_id].cpu())
            is_useful = bool(good_useful[set_index, candidate_id].cpu())
            if is_target:
                category = "target"
            elif is_useful:
                category = "useful_non_target"
            elif region > task.n_useful:
                category = "decoy"
            elif region == 0:
                category = "anchor"
            else:
                category = "other"
            candidate_rows.append(
                {
                    "variant": variant,
                    "set_index": set_index,
                    "task_id": int(task_ids[set_index]),
                    "trial": int(trials[set_index]),
                    "geometry_id": int(task.geometry_id),
                    "goal_index": int(task.goal_index),
                    "target_region": int(task.target_region),
                    "candidate_id": candidate_id,
                    "region": region,
                    "category": category,
                    "quality": quality,
                    "is_useful": is_useful,
                    "is_target": is_target,
                    "is_decoy": bool(decoy[set_index, candidate_id].cpu()),
                    "is_anchor": bool(anchor[set_index, candidate_id].cpu()),
                    "alpha": float(actions.alpha[set_index, candidate_id].cpu()),
                    "direction_0": float(direction_cpu[set_index, candidate_id, 0]),
                    "direction_1": float(direction_cpu[set_index, candidate_id, 1]),
                    "candidate_0": float(candidate_cpu[set_index, candidate_id, 0]),
                    "candidate_1": float(candidate_cpu[set_index, candidate_id, 1]),
                    "final_0": float(final_cpu[set_index, candidate_id, 0]),
                    "final_1": float(final_cpu[set_index, candidate_id, 1]),
                    "router_selected": candidate_id
                    == int(pairwise_index[set_index].cpu()),
                    "quality_oracle_selected": candidate_id
                    == int(quality_index[set_index].cpu()),
                    "useful_oracle_selected": candidate_id
                    == int(useful_index[set_index].cpu()),
                }
            )
    synchronize(device)
    elapsed = perf_counter() - started
    useful_count_tensor = torch.as_tensor(
        useful_counts, dtype=torch.float32, device=device
    )
    pairwise_selector = next(
        row for row in selector_rows if row["selector"] == "frozen_pairwise_router"
    )
    summary = {
        "candidate_mean_quality": float(torch.mean(outcome.quality).cpu()),
        "proposal_recall": float(torch.mean(proposal.float()).cpu()),
        "useful_region_count": float(torch.mean(useful_count_tensor).cpu()),
        "useful_region_fraction": float(
            np.mean([row["useful_region_fraction"] for row in set_rows])
        ),
        "useful_region_count_0": float(
            torch.mean((useful_count_tensor == 0).float()).cpu()
        ),
        "useful_region_count_1": float(
            torch.mean((useful_count_tensor == 1).float()).cpu()
        ),
        "useful_region_count_2_plus": float(
            torch.mean((useful_count_tensor >= 2).float()).cpu()
        ),
        "candidate_decoy_rate": float(torch.mean(decoy.float()).cpu()),
        "candidate_useful_rate": float(torch.mean(good_useful.float()).cpu()),
        "candidate_target_rate": float(torch.mean(target_good.float()).cpu()),
        "direction_cosine_similarity": float(torch.mean(cosine_values).cpu()),
        "direction_cosine_distance": float(torch.mean(cosine_distance_values).cpu()),
        "minimum_direction_distance": float(
            torch.mean(torch.min(cosine_distance_values, dim=1).values).cpu()
        ),
        "effective_directional_rank": float(torch.mean(effective_rank).cpu()),
        "candidate_spatial_diversity": float(torch.mean(spatial_values).cpu()),
        "alpha_mean": float(torch.mean(actions.alpha).cpu()),
        "alpha_sd": float(torch.std(actions.alpha, correction=0).cpu()),
        "within_set_alpha_sd": float(
            torch.mean(torch.std(actions.alpha, dim=1, correction=0)).cpu()
        ),
        "portfolio_collapse": float(np.mean(collapse_values)),
        "quality_oracle_quality": float(
            torch.mean(torch.amax(outcome.quality, dim=1)).cpu()
        ),
        "useful_oracle_success": float(
            torch.mean(_selected(good_useful, useful_index).float()).cpu()
        ),
        "selected_quality": float(pairwise_selector["selected_quality"]),
        "routing_recall": float(pairwise_selector["routing_recall"]),
        "useful_selected": float(pairwise_selector["useful_selected"]),
        "target_selected": float(pairwise_selector["target_selected"]),
        "selected_decoy_rate": float(pairwise_selector["decoy_selected"]),
        "selected_anchor_rate": float(pairwise_selector["anchor_selected"]),
        "evaluation_seconds": elapsed,
    }
    tensors = {
        "starts": h.detach(),
        "directions": actions.direction.detach(),
        "alphas": actions.alpha.detach(),
        "candidates": actions.candidate.detach(),
        "refined": outcome.refined.detach(),
        "quality": outcome.quality.detach(),
        "base_reward": outcome.reward.detach(),
        "novelty": outcome.novelty.detach(),
        "regions": regions.detach(),
        "proposal": proposal.detach(),
    }
    return ModeEvaluationResult(
        summary=summary,
        selector_rows=selector_rows,
        set_rows=set_rows,
        candidate_rows=candidate_rows,
        elapsed_seconds=elapsed,
        tensors=tensors,
    )
