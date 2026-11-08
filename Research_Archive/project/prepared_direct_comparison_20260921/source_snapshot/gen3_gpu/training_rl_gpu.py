"""Group-relative REINFORCE + AdamW for black-box Generation-3 rewards."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from creativity_toy.landscapes_v2 import V2Task

from .black_box_env import BlackBoxLandscapeEnvironment, BlackBoxOutcome
from .config_rl import RLTrainingConfig
from .creative_policy_rl import PolicyActionBatch, StochasticCreativePolicy
from .runtime import synchronize
from .task_batch import TorchTaskBatch


@dataclass
class RLTrainingResult:
    policy: StochasticCreativePolicy
    history: list[dict[str, float]]
    checkpoints: dict[int, dict[str, torch.Tensor]]
    checkpoint_metrics: list[dict[str, float]]
    elapsed_seconds: float
    optimizer_steps: int
    candidate_evaluations: int
    q_equivalent_evaluations: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def group_relative_advantage(
    reward: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    with torch.no_grad():
        centered = reward - torch.mean(reward, dim=1, keepdim=True)
        scale = torch.std(reward, dim=1, keepdim=True, correction=0)
        return (centered / (scale + epsilon)).detach()


def _upper_pair_mean(matrix: torch.Tensor) -> torch.Tensor:
    count = matrix.shape[-1]
    upper = torch.triu(
        torch.ones((count, count), dtype=torch.bool, device=matrix.device),
        diagonal=1,
    )
    return torch.mean(matrix[..., upper])


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    values = [
        torch.sum(parameter.grad.detach() ** 2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return torch.tensor(0.0)
    return torch.sqrt(torch.sum(torch.stack(values)))


@torch.no_grad()
def policy_batch_diagnostics(
    actions: PolicyActionBatch,
    outcome: BlackBoxOutcome,
    config: RLTrainingConfig,
) -> dict[str, float]:
    direction_cosine = torch.matmul(
        actions.direction, actions.direction.transpose(-1, -2)
    )
    spatial = torch.cdist(outcome.refined, outcome.refined, p=2)
    sample_mean_cosine = torch.sum(actions.direction * actions.mean_direction, dim=-1)
    near = (sample_mean_cosine >= config.nearly_deterministic_direction_cosine) & (
        torch.abs(actions.alpha - actions.mean_alpha)
        <= config.nearly_deterministic_alpha_delta
    )
    return {
        "mean_reward": float(torch.mean(outcome.reward).cpu()),
        "reward_std": float(torch.std(outcome.reward, correction=0).cpu()),
        "mean_quality": float(torch.mean(outcome.quality).cpu()),
        "mean_novelty": float(torch.mean(outcome.novelty).cpu()),
        "mean_quality_gate": float(torch.mean(outcome.quality_gate).cpu()),
        "candidate_direction_cosine": float(_upper_pair_mean(direction_cosine).cpu()),
        "candidate_spatial_diversity": float(_upper_pair_mean(spatial).cpu()),
        "within_set_alpha_sd": float(
            torch.mean(torch.std(actions.alpha, dim=1, correction=0)).cpu()
        ),
        "alpha_mean": float(torch.mean(actions.alpha).cpu()),
        "alpha_sd": float(torch.std(actions.alpha, correction=0).cpu()),
        "fraction_nearly_deterministic": float(torch.mean(near.float()).cpu()),
    }


@torch.no_grad()
def portfolio_snapshot(
    policy: StochasticCreativePolicy,
    environment: BlackBoxLandscapeEnvironment,
    tasks: Sequence[V2Task],
    packed_tasks: TorchTaskBatch,
    starts: torch.Tensor,
    *,
    deterministic: bool,
    seed: int,
    useful_threshold: float,
    high_quality_threshold: float,
) -> dict[str, float]:
    generator = torch.Generator(device=starts.device)
    generator.manual_seed(seed)
    actions = policy.act(
        starts,
        packed_tasks.goals,
        generator=None if deterministic else generator,
        deterministic=deterministic,
    )
    outcome = environment.evaluate(actions.candidate, packed_tasks)
    regions = packed_tasks.regions(outcome.refined)
    useful = (regions >= 1) & (regions <= packed_tasks.n_useful[:, None])
    decoy = regions > packed_tasks.n_useful[:, None]
    target = regions == packed_tasks.target_region[:, None]
    good_useful = useful & (outcome.quality >= useful_threshold)
    target_good = target & (outcome.quality >= useful_threshold)
    coverages: list[float] = []
    collapses: list[float] = []
    diversities: list[float] = []
    strict_diversities: list[float] = []
    for task_index, task in enumerate(tasks):
        mask = good_useful[task_index]
        discovered = torch.unique(regions[task_index][mask])
        coverages.append(float(len(discovered)) / task.n_useful)
        points = outcome.refined[task_index][mask]
        diversities.append(
            float(torch.pdist(points).mean().cpu()) if len(points) >= 2 else 0.0
        )
        strict = mask & (outcome.quality[task_index] >= high_quality_threshold)
        strict_points = outcome.refined[task_index][strict]
        strict_diversities.append(
            float(torch.pdist(strict_points).mean().cpu())
            if len(strict_points) >= 2
            else 0.0
        )
        if int(torch.sum(mask)) < 2:
            collapses.append(1.0)
        else:
            selected_regions = regions[task_index][mask]
            counts = torch.bincount(selected_regions, minlength=task.n_useful + 1)
            same = torch.sum(counts * (counts - 1))
            total = len(selected_regions) * (len(selected_regions) - 1)
            collapses.append(float(same.cpu()) / total)
    diagnostics = policy_batch_diagnostics(actions, outcome, RLTrainingConfig())
    diagnostics.update(
        {
            "coverage": float(np.mean(coverages)),
            "collapse_rate": float(np.mean(collapses)),
            "euclidean_diversity": float(np.mean(diversities)),
            "strict_high_quality_diversity": float(np.mean(strict_diversities)),
            "useful_exploration_rate": float(torch.mean(good_useful.float()).cpu()),
            "decoy_rate": float(torch.mean(decoy.float()).cpu()),
            "anchor_escape_rate": float(torch.mean((regions != 0).float()).cpu()),
            "target_hit_rate": float(
                torch.mean(torch.any(target_good, dim=1).float()).cpu()
            ),
            "oracle_best_quality": float(
                torch.mean(torch.amax(outcome.quality, dim=1)).cpu()
            ),
            "evaluation_mode_deterministic": float(deterministic),
        }
    )
    return diagnostics


def _clone_state_dict(policy: StochasticCreativePolicy) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in policy.state_dict().items()
    }


def train_policy_reinforce(
    policy: StochasticCreativePolicy,
    environment: BlackBoxLandscapeEnvironment,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    config: RLTrainingConfig,
    *,
    seed: int,
    checkpoint_callback: Callable[[int, int], dict[str, float]] | None = None,
) -> RLTrainingResult:
    """Train only through log probabilities; rewards remain detached."""
    if tasks.batch_size != len(starts):
        raise ValueError("tasks and starts must have equal length")
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batch_generator = torch.Generator(device=starts.device)
    batch_generator.manual_seed(seed)
    action_generator = torch.Generator(device=starts.device)
    action_generator.manual_seed(seed + 1)
    parameters = list(policy.parameters())
    history: list[dict[str, float]] = []
    checkpoints: dict[int, dict[str, torch.Tensor]] = {0: _clone_state_dict(policy)}
    checkpoint_metrics: list[dict[str, float]] = []
    budgets = sorted(set(config.checkpoint_budgets))
    next_checkpoint = 0
    if budgets and budgets[0] == 0:
        if checkpoint_callback is not None:
            checkpoint_metrics.append(
                {
                    "candidate_evaluations": 0.0,
                    "requested_budget": 0.0,
                    **checkpoint_callback(0, 0),
                }
            )
        next_checkpoint = 1
    candidate_evaluations = 0
    q_equivalent_evaluations = 0
    optimizer_steps = 0
    device = starts.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start_time = perf_counter()

    while candidate_evaluations < config.max_candidate_evaluations:
        next_budget = (
            budgets[next_checkpoint]
            if next_checkpoint < len(budgets)
            else config.max_candidate_evaluations
        )
        target_evaluations = min(next_budget, config.max_candidate_evaluations)
        remaining_groups = max(
            (target_evaluations - candidate_evaluations) // policy.candidates,
            1,
        )
        active_batch_size = min(config.batch_size, tasks.batch_size, remaining_groups)
        indices = torch.randperm(
            tasks.batch_size, generator=batch_generator, device=device
        )[:active_batch_size]
        batch_tasks = tasks.index_select(indices)
        batch_starts = starts.index_select(0, indices)
        actions = policy.act(
            batch_starts,
            batch_tasks.goals,
            generator=action_generator,
            deterministic=False,
        )
        outcome = environment.evaluate(actions.candidate, batch_tasks)
        if outcome.quality.requires_grad or outcome.reward.requires_grad:
            raise RuntimeError(
                "black-box environment outcome unexpectedly requires grad"
            )
        advantage = group_relative_advantage(
            outcome.reward, epsilon=config.advantage_epsilon
        )
        old_log_prob = actions.log_prob.detach()
        policy_loss = -torch.mean(advantage * actions.log_prob)
        entropy = torch.mean(actions.entropy)
        total_loss = policy_loss - config.entropy_coefficient * entropy
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
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
        with torch.no_grad():
            new_log_prob, _, _ = policy.log_prob_of_actions(
                batch_starts,
                batch_tasks.goals,
                actions.raw_direction_action,
                actions.alpha_raw_action,
            )
            approximate_kl = torch.mean(old_log_prob - new_log_prob)
        candidate_evaluations += outcome.candidate_evaluations
        q_equivalent_evaluations += outcome.q_equivalent_evaluations
        optimizer_steps += 1
        should_log = (
            optimizer_steps == 1
            or optimizer_steps % config.log_every_steps == 0
            or candidate_evaluations >= config.max_candidate_evaluations
        )
        crossed_checkpoint = False
        while (
            next_checkpoint < len(budgets)
            and candidate_evaluations >= budgets[next_checkpoint]
        ):
            budget = budgets[next_checkpoint]
            checkpoints[budget] = _clone_state_dict(policy)
            if checkpoint_callback is not None:
                checkpoint_metrics.append(
                    {
                        "candidate_evaluations": float(candidate_evaluations),
                        "requested_budget": float(budget),
                        **checkpoint_callback(budget, optimizer_steps),
                    }
                )
            next_checkpoint += 1
            crossed_checkpoint = True
        if should_log or crossed_checkpoint:
            batch_stats = policy_batch_diagnostics(actions, outcome, config)
            history.append(
                {
                    "optimizer_step": float(optimizer_steps),
                    "candidate_evaluations": float(candidate_evaluations),
                    "q_equivalent_evaluations": float(q_equivalent_evaluations),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    "mean_advantage_magnitude": float(
                        torch.mean(torch.abs(advantage)).cpu()
                    ),
                    "advantage_group_mean_abs": float(
                        torch.mean(torch.abs(torch.mean(advantage, dim=1))).cpu()
                    ),
                    "policy_entropy": float(entropy.detach().cpu()),
                    "direction_std": float(policy.direction_std().detach().cpu()),
                    "alpha_action_std": float(policy.alpha_std().detach().cpu()),
                    "gradient_norm_before_clip": float(
                        torch.as_tensor(gradient_before).detach().cpu()
                    ),
                    "gradient_norm_after_clip": float(gradient_after.detach().cpu()),
                    "approximate_kl": float(approximate_kl.detach().cpu()),
                    **batch_stats,
                }
            )
    synchronize(device)
    elapsed = perf_counter() - start_time
    return RLTrainingResult(
        policy=policy,
        history=history,
        checkpoints=checkpoints,
        checkpoint_metrics=checkpoint_metrics,
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
