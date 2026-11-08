"""Mode-preserving extension of the unchanged black-box CUDA RL trainer."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import torch

from .black_box_env import BlackBoxLandscapeEnvironment
from .config_mode_preservation import ModeRewardConfig
from .config_rl import RLTrainingConfig
from .creative_policy_rl import StochasticCreativePolicy
from .mode_reward import (
    batch_relative_advantage,
    broadcast_set_reward,
    quality_gated_directional_repulsion,
)
from .runtime import synchronize
from .task_batch import TorchTaskBatch
from .training_rl_gpu import (
    RLTrainingResult,
    _clone_state_dict,
    _gradient_norm,
    group_relative_advantage,
    policy_batch_diagnostics,
    train_policy_reinforce,
)


def train_policy_mode_preservation(
    policy: StochasticCreativePolicy,
    environment: BlackBoxLandscapeEnvironment,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    config: RLTrainingConfig,
    mode: ModeRewardConfig,
    *,
    seed: int,
    checkpoint_callback: Callable[[int, int], dict[str, float]] | None = None,
) -> RLTrainingResult:
    """Train with detached mode reward; lambda=0 delegates to the exact baseline."""

    if mode.lambda_mode == 0.0:
        return train_policy_reinforce(
            policy,
            environment,
            tasks,
            starts,
            config,
            seed=seed,
            checkpoint_callback=checkpoint_callback,
        )
    if tasks.batch_size != len(starts):
        raise ValueError("tasks and starts must have equal length")
    if mode.allocation not in {"shared_joint", "candidate_specific"}:
        raise ValueError(f"unknown mode allocation {mode.allocation!r}")

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
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
            (target_evaluations - candidate_evaluations) // policy.candidates, 1
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
        mode_batch = quality_gated_directional_repulsion(
            actions.direction,
            outcome.quality,
            quality_threshold=config.reward.quality_threshold,
            quality_gate_beta=config.reward.quality_gate_beta,
            quality_gated=mode.quality_gated,
            epsilon=mode.epsilon,
        )
        base_advantage = group_relative_advantage(
            outcome.reward, epsilon=config.advantage_epsilon
        )
        base_policy_loss = -torch.mean(base_advantage * actions.log_prob)
        if mode.allocation == "shared_joint":
            # A shared reward cancels under within-set centering. A task-batch baseline
            # preserves its learning signal for the joint proposal-set action;
            # averaging log probabilities normalizes that contribution by K.
            mode_advantage = batch_relative_advantage(
                mode_batch.set_reward, epsilon=config.advantage_epsilon
            )
            joint_log_prob = torch.mean(actions.log_prob, dim=1)
            mode_policy_loss = -torch.mean(mode_advantage * joint_log_prob)
            policy_loss = base_policy_loss + mode.lambda_mode * mode_policy_loss
            total_reward = broadcast_set_reward(
                outcome.reward, mode_batch.set_reward, mode.lambda_mode
            )
            advantage = base_advantage
        else:
            total_reward = (
                outcome.reward + mode.lambda_mode * mode_batch.candidate_reward
            ).detach()
            advantage = group_relative_advantage(
                total_reward, epsilon=config.advantage_epsilon
            )
            policy_loss = -torch.mean(advantage * actions.log_prob)
            mode_policy_loss = policy_loss - base_policy_loss
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
            approximate_kl = torch.mean(actions.log_prob.detach() - new_log_prob)
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
                    "base_policy_loss": float(base_policy_loss.detach().cpu()),
                    "mode_policy_loss": float(mode_policy_loss.detach().cpu()),
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
                    "base_reward_mean": float(torch.mean(outcome.reward).cpu()),
                    "mode_reward_mean": float(torch.mean(mode_batch.set_reward).cpu()),
                    "weighted_mode_reward_mean": float(
                        mode.lambda_mode * torch.mean(mode_batch.set_reward).cpu()
                    ),
                    "total_reward_mean": float(torch.mean(total_reward).cpu()),
                    "mode_reward_to_base_ratio": float(
                        (
                            mode.lambda_mode
                            * torch.mean(mode_batch.set_reward)
                            / torch.clamp(
                                torch.mean(torch.abs(outcome.reward)), min=1e-8
                            )
                        ).cpu()
                    ),
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
