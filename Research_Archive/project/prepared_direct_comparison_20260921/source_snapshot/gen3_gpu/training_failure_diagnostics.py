"""Instrumented black-box RL and privileged direct-gradient diagnostic trainers."""

from __future__ import annotations

from collections.abc import Sequence
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
class DiagnosticTrainingResult:
    policy: StochasticCreativePolicy
    encounter_rows: list[dict[str, float]]
    history: list[dict[str, float]]
    elapsed_seconds: float
    optimizer_steps: int
    candidate_evaluations: int
    q_equivalent_evaluations: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


class EncounterRecorder:
    def __init__(
        self,
        *,
        capacity: int,
        checkpoints: Sequence[int],
        thresholds: Sequence[float],
        quantiles: Sequence[float],
        device: torch.device,
    ) -> None:
        self.values = torch.empty(capacity, dtype=torch.float32, device=device)
        self.capacity = capacity
        self.count = 0
        self.checkpoints = sorted(
            set(int(value) for value in checkpoints if value <= capacity)
        )
        if not self.checkpoints or self.checkpoints[-1] != capacity:
            self.checkpoints.append(capacity)
        self.thresholds = list(thresholds)
        self.quantiles = list(quantiles)
        self.next_checkpoint = 0
        self.rows: list[dict[str, float]] = []

    @torch.no_grad()
    def add(self, quality: torch.Tensor, optimizer_step: int) -> None:
        flat = quality.detach().reshape(-1)
        keep = min(len(flat), self.capacity - self.count)
        if keep <= 0:
            return
        self.values[self.count : self.count + keep] = flat[:keep]
        self.count += keep
        while (
            self.next_checkpoint < len(self.checkpoints)
            and self.count >= self.checkpoints[self.next_checkpoint]
        ):
            budget = self.checkpoints[self.next_checkpoint]
            active = self.values[:budget]
            row: dict[str, float] = {
                "candidate_evaluations": float(budget),
                "optimizer_step": float(optimizer_step),
                "quality_mean": float(torch.mean(active).cpu()),
                "quality_max": float(torch.amax(active).cpu()),
            }
            for threshold in self.thresholds:
                row[f"p_q_gt_{threshold:g}"] = float(
                    torch.mean((active > threshold).float()).cpu()
                )
            quantile_tensor = torch.as_tensor(
                self.quantiles, dtype=active.dtype, device=active.device
            )
            values = torch.quantile(active, quantile_tensor)
            for quantile, value in zip(self.quantiles, values):
                row[f"q_{quantile:g}"] = float(value.cpu())
            self.rows.append(row)
            self.next_checkpoint += 1


def _sample_indices(
    count: int,
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if batch_size <= count:
        return torch.randperm(count, generator=generator, device=device)[:batch_size]
    return torch.randint(count, (batch_size,), generator=generator, device=device)


def differentiable_policy_candidates(
    policy: StochasticCreativePolicy,
    h: torch.Tensor,
    goals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_direction = policy.direction_means(h, goals)
    direction = F.normalize(raw_direction, p=2, dim=-1, eps=1e-8)
    raw_alpha = policy.alpha_means(h, goals, direction)
    alpha = policy.alpha_max * torch.sigmoid(raw_alpha)
    candidates = h[:, None, :] + alpha.unsqueeze(-1) * direction
    return candidates, direction, alpha


def train_instrumented_rl(
    policy: StochasticCreativePolicy,
    environment: BlackBoxLandscapeEnvironment,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    config: RLTrainingConfig,
    *,
    seed: int,
    budget: int,
    batch_size: int,
    encounter_checkpoints: Sequence[int],
    thresholds: Sequence[float],
    quantiles: Sequence[float],
) -> DiagnosticTrainingResult:
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batch_generator = torch.Generator(device=starts.device).manual_seed(seed)
    action_generator = torch.Generator(device=starts.device).manual_seed(seed + 1)
    recorder = EncounterRecorder(
        capacity=budget,
        checkpoints=encounter_checkpoints,
        thresholds=thresholds,
        quantiles=quantiles,
        device=starts.device,
    )
    parameters = list(policy.parameters())
    candidate_evaluations = 0
    q_equivalent = 0
    optimizer_steps = 0
    history: list[dict[str, float]] = []
    device = starts.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = perf_counter()
    while candidate_evaluations < budget:
        remaining_groups = max((budget - candidate_evaluations) // policy.candidates, 1)
        active_batch = min(batch_size, remaining_groups)
        indices = _sample_indices(
            tasks.batch_size,
            active_batch,
            generator=batch_generator,
            device=device,
        )
        batch_tasks = tasks.index_select(indices)
        h = starts.index_select(0, indices)
        actions = policy.act(
            h, batch_tasks.goals, generator=action_generator, deterministic=False
        )
        outcome = environment.evaluate(actions.candidate, batch_tasks)
        if outcome.quality.requires_grad:
            raise RuntimeError("instrumented RL received a differentiable reward")
        advantage = group_relative_advantage(
            outcome.reward, epsilon=config.advantage_epsilon
        )
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
        count = outcome.candidate_evaluations
        candidate_evaluations += count
        q_equivalent += outcome.q_equivalent_evaluations
        optimizer_steps += 1
        recorder.add(outcome.quality, optimizer_steps)
        if recorder.rows and (
            not history
            or recorder.rows[-1]["candidate_evaluations"]
            != history[-1]["candidate_evaluations"]
        ):
            history.append(
                {
                    **recorder.rows[-1],
                    "loss": float(loss.detach().cpu()),
                    "policy_entropy": float(entropy.detach().cpu()),
                    "direction_std": float(policy.direction_std().detach().cpu()),
                    "alpha_action_std": float(policy.alpha_std().detach().cpu()),
                    "alpha_mean": float(torch.mean(actions.alpha).cpu()),
                    "alpha_sd": float(torch.std(actions.alpha, correction=0).cpu()),
                    "direction_effective_rank": float(
                        torch.mean(_effective_rank(actions.direction)).cpu()
                    ),
                    "gradient_norm_before_clip": float(
                        torch.as_tensor(gradient_before).cpu()
                    ),
                    "gradient_norm_after_clip": float(gradient_after.cpu()),
                }
            )
    synchronize(device)
    elapsed = perf_counter() - started
    return DiagnosticTrainingResult(
        policy=policy,
        encounter_rows=recorder.rows,
        history=history,
        elapsed_seconds=elapsed,
        optimizer_steps=optimizer_steps,
        candidate_evaluations=candidate_evaluations,
        q_equivalent_evaluations=q_equivalent,
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        peak_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    )


def _effective_rank(direction: torch.Tensor) -> torch.Tensor:
    singular = torch.linalg.svdvals(direction)
    energy = singular.square()
    probability = energy / torch.clamp(
        torch.sum(energy, dim=1, keepdim=True), min=1e-12
    )
    return torch.exp(
        -torch.sum(
            torch.where(
                probability > 0,
                probability * torch.log(probability),
                torch.zeros_like(probability),
            ),
            dim=1,
        )
    )


def train_direct_gradient(
    policy: StochasticCreativePolicy,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    *,
    refinement_steps: int,
    refinement_rate: float,
    seed: int,
    budget: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    encounter_checkpoints: Sequence[int],
    thresholds: Sequence[float],
    quantiles: Sequence[float],
) -> DiagnosticTrainingResult:
    """Privileged control: optimize mean refined Q through the environment."""
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    batch_generator = torch.Generator(device=starts.device).manual_seed(seed)
    recorder = EncounterRecorder(
        capacity=budget,
        checkpoints=encounter_checkpoints,
        thresholds=thresholds,
        quantiles=quantiles,
        device=starts.device,
    )
    parameters = list(policy.parameters())
    candidate_evaluations = 0
    optimizer_steps = 0
    history: list[dict[str, float]] = []
    device = starts.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = perf_counter()
    while candidate_evaluations < budget:
        remaining_groups = max((budget - candidate_evaluations) // policy.candidates, 1)
        active_batch = min(batch_size, remaining_groups)
        indices = _sample_indices(
            tasks.batch_size,
            active_batch,
            generator=batch_generator,
            device=device,
        )
        batch_tasks = tasks.index_select(indices)
        h = starts.index_select(0, indices)
        candidates, direction, alpha = differentiable_policy_candidates(
            policy, h, batch_tasks.goals
        )
        refined = batch_tasks.refine(
            candidates, steps=refinement_steps, rate=refinement_rate
        )
        quality = batch_tasks.quality(refined)
        if not quality.requires_grad:
            raise RuntimeError("direct-gradient diagnostic lost the Q autograd path")
        loss = -torch.mean(quality)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_before = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=gradient_clip
        )
        gradient_after = _gradient_norm(parameters)
        optimizer.step()
        count = int(active_batch * policy.candidates)
        candidate_evaluations += count
        optimizer_steps += 1
        recorder.add(quality.detach(), optimizer_steps)
        if recorder.rows and (
            not history
            or recorder.rows[-1]["candidate_evaluations"]
            != history[-1]["candidate_evaluations"]
        ):
            history.append(
                {
                    **recorder.rows[-1],
                    "loss": float(loss.detach().cpu()),
                    "alpha_mean": float(torch.mean(alpha).detach().cpu()),
                    "alpha_sd": float(torch.std(alpha.detach(), correction=0).cpu()),
                    "direction_effective_rank": float(
                        torch.mean(_effective_rank(direction.detach())).cpu()
                    ),
                    "gradient_norm_before_clip": float(
                        torch.as_tensor(gradient_before).detach().cpu()
                    ),
                    "gradient_norm_after_clip": float(gradient_after.detach().cpu()),
                }
            )
    synchronize(device)
    elapsed = perf_counter() - started
    return DiagnosticTrainingResult(
        policy=policy,
        encounter_rows=recorder.rows,
        history=history,
        elapsed_seconds=elapsed,
        optimizer_steps=optimizer_steps,
        candidate_evaluations=candidate_evaluations,
        q_equivalent_evaluations=candidate_evaluations * (refinement_steps + 1),
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        peak_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    )
