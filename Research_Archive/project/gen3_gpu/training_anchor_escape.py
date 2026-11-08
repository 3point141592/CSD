"""Black-box RL diagnostic with a bounded anchor-neighborhood penalty.

This is deliberately separate from the established RL and mode-preservation
trainers.  It changes only the detached scalar reward returned to REINFORCE.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as F

from .black_box_env import BlackBoxLandscapeEnvironment
from .config_rl import RLTrainingConfig
from .creative_policy_rl import StochasticCreativePolicy
from .runtime import synchronize
from .task_batch import TorchTaskBatch
from .training_rl_gpu import _gradient_norm, group_relative_advantage


@dataclass
class AnchorEscapeTrainingResult:
    policy: StochasticCreativePolicy
    history: list[dict[str, float]]
    validation_history: list[dict[str, float]]
    elapsed_seconds: float
    optimizer_steps: int
    candidate_evaluations: int
    q_equivalent_evaluations: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def anchor_margin_penalty(
    candidates: torch.Tensor,
    anchor: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(
        candidates.detach() - anchor[:, None, :].detach(), dim=-1
    )
    return (
        torch.relu(
            torch.as_tensor(radius, dtype=distance.dtype, device=distance.device)
            - distance
        )
        .square()
        .detach()
    )


def train_policy_anchor_escape(
    policy: StochasticCreativePolicy,
    environment: BlackBoxLandscapeEnvironment,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    config: RLTrainingConfig,
    *,
    seed: int,
    lambda_anchor: float,
    anchor_radius: float,
    validation_callback: Callable[[StochasticCreativePolicy, int], dict[str, float]]
    | None = None,
) -> AnchorEscapeTrainingResult:
    """Group-relative REINFORCE with one candidate-specific margin penalty."""
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batch_generator = torch.Generator(device=starts.device).manual_seed(seed)
    action_generator = torch.Generator(device=starts.device).manual_seed(seed + 1)
    parameters = list(policy.parameters())
    budgets = sorted(
        set(config.checkpoint_budgets + [config.max_candidate_evaluations])
    )
    history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    candidate_evaluations = 0
    q_equivalent_evaluations = 0
    optimizer_steps = 0
    next_checkpoint = 0
    if budgets and budgets[0] == 0:
        if validation_callback is not None:
            validation_history.append(
                {
                    "candidate_evaluations": 0.0,
                    **validation_callback(policy, 0),
                }
            )
        next_checkpoint = 1
    device = starts.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = perf_counter()
    while candidate_evaluations < config.max_candidate_evaluations:
        next_budget = (
            budgets[next_checkpoint]
            if next_checkpoint < len(budgets)
            else config.max_candidate_evaluations
        )
        target = min(next_budget, config.max_candidate_evaluations)
        remaining_groups = max((target - candidate_evaluations) // policy.candidates, 1)
        active_batch = min(config.batch_size, tasks.batch_size, remaining_groups)
        indices = torch.randperm(
            tasks.batch_size, generator=batch_generator, device=device
        )[:active_batch]
        batch_tasks = tasks.index_select(indices)
        h = starts.index_select(0, indices)
        actions = policy.act(
            h, batch_tasks.goals, generator=action_generator, deterministic=False
        )
        outcome = environment.evaluate(actions.candidate, batch_tasks)
        if outcome.reward.requires_grad or outcome.quality.requires_grad:
            raise RuntimeError("black-box outcome must be detached")
        penalty = anchor_margin_penalty(
            actions.candidate, batch_tasks.anchor, anchor_radius
        )
        reward = (outcome.reward - lambda_anchor * penalty).detach()
        advantage = group_relative_advantage(reward, epsilon=config.advantage_epsilon)
        policy_loss = -torch.mean(advantage * actions.log_prob)
        entropy = torch.mean(actions.entropy)
        loss = policy_loss - config.entropy_coefficient * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_before = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=config.gradient_clip
        )
        gradient_after = _gradient_norm(parameters)
        optimizer.step()
        policy.clamp_standard_deviations(
            direction_min=config.direction_std_min,
            direction_max=config.direction_std_max,
            alpha_min=config.alpha_std_min,
            alpha_max=config.alpha_std_max,
        )
        candidate_evaluations += outcome.candidate_evaluations
        q_equivalent_evaluations += outcome.q_equivalent_evaluations
        optimizer_steps += 1
        crossed = False
        while (
            next_checkpoint < len(budgets)
            and candidate_evaluations >= budgets[next_checkpoint]
        ):
            checkpoint = budgets[next_checkpoint]
            if validation_callback is not None:
                validation_history.append(
                    {
                        "candidate_evaluations": float(candidate_evaluations),
                        "requested_budget": float(checkpoint),
                        **validation_callback(policy, checkpoint),
                    }
                )
            next_checkpoint += 1
            crossed = True
        should_log = (
            optimizer_steps == 1
            or optimizer_steps % config.log_every_steps == 0
            or candidate_evaluations >= config.max_candidate_evaluations
            or crossed
        )
        if should_log:
            with torch.no_grad():
                distance = torch.linalg.vector_norm(
                    actions.candidate - batch_tasks.anchor[:, None, :], dim=-1
                )
                history.append(
                    {
                        "optimizer_step": float(optimizer_steps),
                        "candidate_evaluations": float(candidate_evaluations),
                        "q_equivalent_evaluations": float(q_equivalent_evaluations),
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "mean_base_reward": float(torch.mean(outcome.reward).cpu()),
                        "mean_anchor_penalty": float(torch.mean(penalty).cpu()),
                        "mean_weighted_anchor_penalty": float(
                            torch.mean(lambda_anchor * penalty).cpu()
                        ),
                        "mean_total_reward": float(torch.mean(reward).cpu()),
                        "mean_quality": float(torch.mean(outcome.quality).cpu()),
                        "mean_anchor_distance": float(torch.mean(distance).cpu()),
                        "inside_anchor_margin_rate": float(
                            torch.mean((distance < anchor_radius).float()).cpu()
                        ),
                        "alpha_mean": float(torch.mean(actions.alpha).cpu()),
                        "alpha_sd": float(torch.std(actions.alpha, correction=0).cpu()),
                        "direction_std": float(policy.direction_std().cpu()),
                        "alpha_action_std": float(policy.alpha_std().cpu()),
                        "policy_entropy": float(entropy.detach().cpu()),
                        "gradient_norm_before_clip": float(
                            torch.as_tensor(gradient_before).cpu()
                        ),
                        "gradient_norm_after_clip": float(gradient_after.cpu()),
                    }
                )
    synchronize(device)
    elapsed = perf_counter() - started
    return AnchorEscapeTrainingResult(
        policy=policy,
        history=history,
        validation_history=validation_history,
        elapsed_seconds=elapsed,
        optimizer_steps=optimizer_steps,
        candidate_evaluations=candidate_evaluations,
        q_equivalent_evaluations=q_equivalent_evaluations,
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        peak_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    )


@torch.no_grad()
def evaluate_anchor_policy(
    policy: StochasticCreativePolicy,
    environment: BlackBoxLandscapeEnvironment,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    *,
    useful_threshold: float,
    capture_distance: float,
    action_seed: int,
    deterministic: bool = False,
) -> dict[str, float]:
    generator = torch.Generator(device=starts.device).manual_seed(action_seed)
    actions = policy.act(
        starts,
        tasks.goals,
        generator=None if deterministic else generator,
        deterministic=deterministic,
    )
    outcome = environment.evaluate(actions.candidate, tasks)
    regions = tasks.regions(outcome.refined)
    useful = (
        (regions >= 1)
        & (regions <= tasks.n_useful[:, None])
        & (outcome.quality >= useful_threshold)
    )
    decoy = regions > tasks.n_useful[:, None]
    anchor = regions == 0
    delta = tasks.useful_centers - starts[:, None, :]
    useful_direction = F.normalize(delta, p=2, dim=-1, eps=1e-8)
    alignment = torch.einsum("bkd,bmd->bkm", actions.direction, useful_direction)
    alignment = torch.where(
        tasks.useful_mask[:, None, :], alignment, torch.full_like(alignment, -torch.inf)
    )
    pre_distance = torch.linalg.vector_norm(
        actions.candidate[:, :, None, :] - tasks.useful_centers[:, None, :, :], dim=-1
    )
    pre_distance = torch.where(
        tasks.useful_mask[:, None, :],
        pre_distance,
        torch.full_like(pre_distance, torch.inf),
    )
    cosine = torch.matmul(actions.direction, actions.direction.transpose(-1, -2))
    upper = torch.triu_indices(
        policy.candidates, policy.candidates, offset=1, device=starts.device
    )
    singular = torch.linalg.svdvals(actions.direction)
    energy = singular.square()
    probability = energy / torch.clamp(
        torch.sum(energy, dim=1, keepdim=True), min=1e-12
    )
    rank = torch.exp(
        -torch.sum(
            torch.where(
                probability > 0,
                probability * torch.log(probability),
                torch.zeros_like(probability),
            ),
            dim=1,
        )
    )
    useful_counts = torch.as_tensor(
        [
            float(torch.unique(regions[index][useful[index]]).numel())
            for index in range(len(starts))
        ],
        device=starts.device,
    )
    proposal = torch.any(useful, dim=1)
    return {
        "proposal_recall": float(torch.mean(proposal.float()).cpu()),
        "useful_candidate_rate": float(torch.mean(useful.float()).cpu()),
        "useful_region_count": float(torch.mean(useful_counts).cpu()),
        "candidate_decoy_rate": float(torch.mean(decoy.float()).cpu()),
        "candidate_anchor_rate": float(torch.mean(anchor.float()).cpu()),
        "candidate_mean_quality": float(torch.mean(outcome.quality).cpu()),
        "best_useful_alignment": float(
            torch.mean(torch.amax(alignment, dim=(1, 2))).cpu()
        ),
        "capture_entry_rate": float(
            torch.mean(
                (torch.amin(pre_distance, dim=-1) <= capture_distance).float()
            ).cpu()
        ),
        "alpha_mean": float(torch.mean(actions.alpha).cpu()),
        "alpha_sd": float(torch.std(actions.alpha, correction=0).cpu()),
        "effective_directional_rank": float(torch.mean(rank).cpu()),
        "direction_cosine_distance": float(
            torch.mean(1.0 - cosine[:, upper[0], upper[1]]).cpu()
        ),
    }
