"""Black-box RL with a bounded penalty on the final refined endpoint.

This is intentionally separate from every established trainer.  Q, refinement,
and the endpoint penalty are detached scalar feedback; mode preservation is off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import torch

from .black_box_env import BlackBoxLandscapeEnvironment
from .config_rl import RLTrainingConfig
from .creative_policy_rl import StochasticCreativePolicy
from .runtime import synchronize
from .task_batch import TorchTaskBatch
from .training_rl_gpu import _gradient_norm, group_relative_advantage


@dataclass
class EndpointAnchorTrainingResult:
    policy: StochasticCreativePolicy
    history: list[dict[str, float]]
    validation_history: list[dict[str, float]]
    elapsed_seconds: float
    optimizer_steps: int
    candidate_evaluations: int
    q_equivalent_evaluations: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def endpoint_anchor_penalty(
    refined: torch.Tensor,
    anchor: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Squared margin penalty using only detached final endpoints."""
    distance = torch.linalg.vector_norm(
        refined.detach() - anchor[:, None, :].detach(), dim=-1
    )
    return (
        torch.relu(
            torch.as_tensor(radius, dtype=distance.dtype, device=distance.device)
            - distance
        )
        .square()
        .detach()
    )


def train_policy_endpoint_anchor(
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
) -> EndpointAnchorTrainingResult:
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
        penalty = endpoint_anchor_penalty(
            outcome.refined, batch_tasks.anchor, anchor_radius
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
        if (
            optimizer_steps == 1
            or optimizer_steps % config.log_every_steps == 0
            or candidate_evaluations >= config.max_candidate_evaluations
            or crossed
        ):
            with torch.no_grad():
                final_distance = torch.linalg.vector_norm(
                    outcome.refined - batch_tasks.anchor[:, None, :], dim=-1
                )
                pre_distance = torch.linalg.vector_norm(
                    actions.candidate - batch_tasks.anchor[:, None, :], dim=-1
                )
                history.append(
                    {
                        "optimizer_step": float(optimizer_steps),
                        "candidate_evaluations": float(candidate_evaluations),
                        "q_equivalent_evaluations": float(q_equivalent_evaluations),
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "mean_base_reward": float(torch.mean(outcome.reward).cpu()),
                        "mean_endpoint_penalty": float(torch.mean(penalty).cpu()),
                        "mean_weighted_endpoint_penalty": float(
                            torch.mean(lambda_anchor * penalty).cpu()
                        ),
                        "mean_total_reward": float(torch.mean(reward).cpu()),
                        "mean_quality": float(torch.mean(outcome.quality).cpu()),
                        "mean_pre_anchor_distance": float(
                            torch.mean(pre_distance).cpu()
                        ),
                        "mean_final_anchor_distance": float(
                            torch.mean(final_distance).cpu()
                        ),
                        "inside_final_anchor_margin_rate": float(
                            torch.mean((final_distance < anchor_radius).float()).cpu()
                        ),
                        "alpha_mean": float(torch.mean(actions.alpha).cpu()),
                        "alpha_sd": float(torch.std(actions.alpha, correction=0).cpu()),
                        "policy_entropy": float(entropy.detach().cpu()),
                        "gradient_norm_before_clip": float(
                            torch.as_tensor(gradient_before).cpu()
                        ),
                        "gradient_norm_after_clip": float(gradient_after.cpu()),
                    }
                )
    synchronize(device)
    elapsed = perf_counter() - started
    return EndpointAnchorTrainingResult(
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
