"""Geometric, angular, radial, capture, random-hit, and batch diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from creativity_toy.landscapes_v2 import V2Task

from .black_box_env import BlackBoxLandscapeEnvironment
from .config_rl import RLTrainingConfig
from .contrastive_evaluation import CandidateEvaluation, flatten_starts
from .contrastive_steering import ContrastiveCandidates
from .creative_policy_rl import StochasticCreativePolicy
from .runtime import synchronize
from .task_batch import pack_tasks
from .training_rl_gpu import group_relative_advantage


@torch.no_grad()
def candidate_geometry_rows(
    *,
    dimension: int,
    seed: int,
    method: str,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    proposal: ContrastiveCandidates,
    evaluation: CandidateEvaluation,
    device: torch.device,
    useful_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    start_array, task_ids, trials = flatten_starts(starts)
    packed = pack_tasks(tasks, device=device, dtype=torch.float32)
    indices = torch.as_tensor(task_ids, dtype=torch.long, device=device)
    aligned = packed.index_select(indices)
    h = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    useful_delta = aligned.useful_centers - h[:, None, :]
    useful_distance = torch.linalg.vector_norm(useful_delta, dim=-1)
    useful_distance = torch.where(
        aligned.useful_mask,
        useful_distance,
        torch.full_like(useful_distance, torch.inf),
    )
    useful_direction = F.normalize(useful_delta, p=2, dim=-1, eps=1e-8)
    target_center = torch.gather(
        aligned.useful_centers,
        1,
        aligned.goal_index[:, None, None].expand(-1, 1, dimension),
    ).squeeze(1)
    target_distance = torch.linalg.vector_norm(target_center - h, dim=-1)
    decoy_distance = torch.linalg.vector_norm(
        aligned.decoy_centers - h[:, None, :], dim=-1
    )
    decoy_distance = torch.where(
        aligned.decoy_mask, decoy_distance, torch.full_like(decoy_distance, torch.inf)
    )
    pre_delta = (
        proposal.candidates[:, :, None, :] - aligned.useful_centers[:, None, :, :]
    )
    pre_distance = torch.linalg.vector_norm(pre_delta, dim=-1)
    pre_distance = torch.where(
        aligned.useful_mask[:, None, :],
        pre_distance,
        torch.full_like(pre_distance, torch.inf),
    )
    alignment = torch.einsum("bkd,bud->bku", proposal.directions, useful_direction)
    alignment = torch.where(
        aligned.useful_mask[:, None, :],
        alignment,
        torch.full_like(alignment, -torch.inf),
    )
    target_alignment = torch.gather(
        alignment,
        2,
        aligned.goal_index[:, None, None].expand(-1, proposal.candidates.shape[1], 1),
    ).squeeze(-1)
    final_regions = evaluation.batch.regions
    final_quality = evaluation.batch.quality
    final_useful = (
        (final_regions >= 1)
        & (final_regions <= aligned.n_useful[:, None])
        & (final_quality >= useful_threshold)
    )
    pre_regions = aligned.regions(proposal.candidates)
    pre_useful = (pre_regions >= 1) & (pre_regions <= aligned.n_useful[:, None])
    candidate_rows: list[dict[str, object]] = []
    for batch_index in range(len(h)):
        for candidate_index in range(proposal.candidates.shape[1]):
            candidate_rows.append(
                {
                    "dimension": dimension,
                    "seed": seed,
                    "method": method,
                    "set_index": batch_index,
                    "task_id": int(task_ids[batch_index]),
                    "trial": int(trials[batch_index]),
                    "candidate_id": candidate_index,
                    "alpha": float(proposal.alphas[batch_index, candidate_index].cpu()),
                    "alpha_to_nearest_useful_ratio": float(
                        (
                            proposal.alphas[batch_index, candidate_index]
                            / torch.clamp(
                                torch.amin(useful_distance[batch_index]), min=1e-8
                            )
                        ).cpu()
                    ),
                    "pre_nearest_useful_distance": float(
                        torch.amin(pre_distance[batch_index, candidate_index]).cpu()
                    ),
                    "best_useful_alignment": float(
                        torch.amax(alignment[batch_index, candidate_index]).cpu()
                    ),
                    "target_alignment": float(
                        target_alignment[batch_index, candidate_index].cpu()
                    ),
                    "pre_useful": bool(pre_useful[batch_index, candidate_index].cpu()),
                    "post_useful": bool(
                        final_useful[batch_index, candidate_index].cpu()
                    ),
                    "refinement_created_useful_hit": bool(
                        final_useful[batch_index, candidate_index].cpu()
                        and not pre_useful[batch_index, candidate_index].cpu()
                    ),
                    "quality": float(final_quality[batch_index, candidate_index].cpu()),
                }
            )
    task_rows = [
        {
            "dimension": dimension,
            "seed": seed,
            "method": method,
            "set_index": index,
            "task_id": int(task_ids[index]),
            "trial": int(trials[index]),
            "nearest_useful_distance": float(torch.amin(useful_distance[index]).cpu()),
            "mean_useful_distance": float(
                torch.mean(useful_distance[index][aligned.useful_mask[index]]).cpu()
            ),
            "target_distance": float(target_distance[index].cpu()),
            "nearest_decoy_distance": float(torch.amin(decoy_distance[index]).cpu()),
        }
        for index in range(len(h))
    ]
    return candidate_rows, task_rows


@torch.no_grad()
def empirical_random_hit_rate(
    *,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    alpha_samples: torch.Tensor,
    outcomes: int,
    refinement_steps: int,
    refinement_rate: float,
    useful_threshold: float,
    seed: int,
    device: torch.device,
    chunk_size: int = 4096,
) -> dict[str, float]:
    start_array, task_ids, _ = flatten_starts(starts)
    packed = pack_tasks(tasks, device=device, dtype=torch.float32)
    h_all = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    task_ids_tensor = torch.as_tensor(task_ids, dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    hit_count = 0
    quality_sum = 0.0
    quality_max = 0.0
    processed = 0
    synchronize(device)
    started = perf_counter()
    while processed < outcomes:
        count = min(chunk_size, outcomes - processed)
        set_index = torch.arange(processed, processed + count, device=device) % len(
            h_all
        )
        h = h_all.index_select(0, set_index)
        task_index = task_ids_tensor.index_select(0, set_index)
        aligned = packed.index_select(task_index)
        direction = F.normalize(
            torch.randn(
                (count, 1, h.shape[-1]),
                generator=generator,
                device=device,
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
            eps=1e-8,
        )
        alpha_index = torch.randint(
            len(alpha_samples), (count,), generator=generator, device=device
        )
        alpha = alpha_samples.to(device).index_select(0, alpha_index)
        candidates = h[:, None, :] + alpha[:, None, None] * direction
        refined = aligned.refine(
            candidates, steps=refinement_steps, rate=refinement_rate
        )
        quality = aligned.quality(refined)
        regions = aligned.regions(refined)
        useful = (
            (regions[:, 0] >= 1)
            & (regions[:, 0] <= aligned.n_useful)
            & (quality[:, 0] >= useful_threshold)
        )
        hit_count += int(torch.sum(useful).cpu())
        quality_sum += float(torch.sum(quality).cpu())
        quality_max = max(quality_max, float(torch.amax(quality).cpu()))
        processed += count
    synchronize(device)
    elapsed = perf_counter() - started
    return {
        "outcomes": float(outcomes),
        "useful_hit_rate": hit_count / outcomes,
        "mean_quality": quality_sum / outcomes,
        "max_quality": quality_max,
        "elapsed_seconds": elapsed,
        "outcomes_per_second": outcomes / elapsed,
    }


def benchmark_batch_sizes(
    *,
    make_policy,
    environment: BlackBoxLandscapeEnvironment,
    tasks,
    starts: torch.Tensor,
    training: RLTrainingConfig,
    batch_sizes: Sequence[int],
    warmup_steps: int,
    timing_steps: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        try:
            policy: StochasticCreativePolicy = make_policy()
            optimizer = torch.optim.AdamW(
                policy.parameters(),
                lr=training.learning_rate,
                weight_decay=training.weight_decay,
            )
            batch_generator = torch.Generator(device=device).manual_seed(seed)
            action_generator = torch.Generator(device=device).manual_seed(seed + 1)

            def step() -> None:
                indices = torch.randint(
                    tasks.batch_size,
                    (batch_size,),
                    generator=batch_generator,
                    device=device,
                )
                batch_tasks = tasks.index_select(indices)
                h = starts.index_select(0, indices)
                actions = policy.act(
                    h,
                    batch_tasks.goals,
                    generator=action_generator,
                    deterministic=False,
                )
                outcome = environment.evaluate(actions.candidate, batch_tasks)
                advantage = group_relative_advantage(
                    outcome.reward, epsilon=training.advantage_epsilon
                )
                loss = -torch.mean(advantage * actions.log_prob)
                loss = loss - training.entropy_coefficient * torch.mean(actions.entropy)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=training.gradient_clip
                )
                optimizer.step()

            for _ in range(warmup_steps):
                step()
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = perf_counter()
            for _ in range(timing_steps):
                step()
            synchronize(device)
            elapsed = perf_counter() - started
            outcomes = batch_size * policy.candidates * timing_steps
            rows.append(
                {
                    "batch_size": batch_size,
                    "status": "ok",
                    "elapsed_seconds": elapsed,
                    "steps_per_second": timing_steps / elapsed,
                    "candidate_outcomes_per_second": outcomes / elapsed,
                    "peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    ),
                    "peak_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(device))
                        if device.type == "cuda"
                        else 0
                    ),
                }
            )
        except torch.cuda.OutOfMemoryError as error:
            rows.append(
                {
                    "batch_size": batch_size,
                    "status": "oom",
                    "error": str(error),
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
    return rows
