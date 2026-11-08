"""Models and optimization helpers for the isolated 32D diagnosis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .runtime import synchronize
from .supervised_capacity import supervised_objective, useful_targets
from .task_batch import TorchTaskBatch


class NoZSetPolicy(nn.Module):
    """Diagnostic set predictor: [h,g] emits all K directions simultaneously."""

    def __init__(
        self,
        *,
        dimension: int,
        candidates: int,
        hidden_dimension: int,
        alpha_max: float,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.candidates = candidates
        self.alpha_max = float(alpha_max)
        generator = torch.Generator(device=device).manual_seed(seed)
        self.w1 = nn.Parameter(
            torch.randn(
                hidden_dimension, 2 * dimension, generator=generator, device=device
            )
            * 0.04
        )
        self.b1 = nn.Parameter(torch.zeros(hidden_dimension, device=device))
        self.w2 = nn.Parameter(
            torch.randn(
                candidates * dimension,
                hidden_dimension,
                generator=generator,
                device=device,
            )
            * 0.04
        )
        self.b2 = nn.Parameter(torch.zeros(candidates * dimension, device=device))
        self.alpha_weight = nn.Parameter(
            torch.zeros(candidates, hidden_dimension, device=device)
        )
        self.alpha_bias = nn.Parameter(torch.zeros(candidates, device=device))

    def direction_means(self, h: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(F.linear(torch.cat([h, goal], dim=-1), self.w1, self.b1))
        return F.linear(hidden, self.w2, self.b2).reshape(
            len(h), self.candidates, self.dimension
        )

    def alpha_means(
        self, h: torch.Tensor, goal: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        del direction
        hidden = torch.tanh(F.linear(torch.cat([h, goal], dim=-1), self.w1, self.b1))
        return F.linear(hidden, self.alpha_weight, self.alpha_bias)


class DirectionProbe(nn.Module):
    """Small direction predictor used only to measure input identifiability."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        *,
        hidden_dimension: int | None,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(seed)
        if hidden_dimension is None:
            self.network = nn.Linear(input_dimension, output_dimension, device=device)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dimension, hidden_dimension, device=device),
                nn.Tanh(),
                nn.Linear(hidden_dimension, output_dimension, device=device),
            )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    layer.weight.copy_(
                        torch.randn(
                            layer.weight.shape, generator=generator, device=device
                        )
                        * 0.04
                    )
                    layer.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(inputs), p=2, dim=-1, eps=1e-8)


@dataclass
class DiagnosticTrainingResult:
    history: list[dict[str, float]]
    elapsed_seconds: float
    examples_seen: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def train_supervised_variant(
    model: nn.Module,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    *,
    mode: Literal["single_target", "multimode"],
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    lambda_alpha: float,
    seed: int,
    curve_steps: list[int],
    permute_codes: bool = False,
    evaluation_callback: Callable[[nn.Module, int], dict[str, float]] | None = None,
) -> DiagnosticTrainingResult:
    """Same supervised objective, with an optional code-slot permutation control."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device=starts.device).manual_seed(seed)
    history: list[dict[str, float]] = []
    curve = set(value for value in curve_steps if value <= steps)
    curve.add(steps)
    original_codes = (
        model.codes.detach().clone()
        if permute_codes and hasattr(model, "codes")
        else None
    )
    if starts.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(starts.device)
    synchronize(starts.device)
    started = perf_counter()
    for step in range(1, steps + 1):
        count = min(batch_size, len(starts))
        indices = torch.randperm(
            len(starts), generator=generator, device=starts.device
        )[:count]
        if original_codes is not None:
            permutation = torch.randperm(
                len(original_codes), generator=generator, device=starts.device
            )
            with torch.no_grad():
                model.codes.copy_(original_codes.index_select(0, permutation))
        loss, metrics = supervised_objective(
            model,
            starts.index_select(0, indices),
            tasks.index_select(indices),
            mode=mode,
            lambda_alpha=lambda_alpha,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if original_codes is not None:
            with torch.no_grad():
                model.codes.copy_(original_codes)
        if step in curve:
            point = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                **{
                    name: float(value.detach().cpu()) for name, value in metrics.items()
                },
            }
            if evaluation_callback is not None:
                point.update(evaluation_callback(model, step))
            history.append(point)
    synchronize(starts.device)
    return DiagnosticTrainingResult(
        history=history,
        elapsed_seconds=perf_counter() - started,
        examples_seen=steps * min(batch_size, len(starts)),
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(starts.device))
            if starts.device.type == "cuda"
            else 0
        ),
        peak_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(starts.device))
            if starts.device.type == "cuda"
            else 0
        ),
    )


def probe_inputs(
    kind: str,
    starts: torch.Tensor,
    tasks: TorchTaskBatch,
) -> torch.Tensor:
    if kind == "goal":
        return tasks.goals
    if kind == "state":
        return starts
    if kind in {"state_goal_linear", "state_goal_mlp"}:
        return torch.cat([starts, tasks.goals], dim=-1)
    if kind == "privileged_geometry":
        _, _, target_direction, _ = useful_targets(starts, tasks)
        return target_direction
    raise ValueError(kind)


def train_direction_probe(
    probe: DirectionProbe,
    inputs: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[list[dict[str, float]], float]:
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    log_steps = {1, max(1, steps // 4), max(1, steps // 2), steps}
    history = []
    synchronize(inputs.device)
    started = perf_counter()
    for step in range(1, steps + 1):
        count = min(batch_size, len(inputs))
        indices = torch.randperm(
            len(inputs), generator=generator, device=inputs.device
        )[:count]
        prediction = probe(inputs.index_select(0, indices))
        cosine = torch.sum(
            prediction * target_direction.index_select(0, indices), dim=-1
        )
        loss = torch.mean(1.0 - cosine)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in log_steps:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "cosine": float(torch.mean(cosine).detach().cpu()),
                }
            )
    synchronize(inputs.device)
    return history, perf_counter() - started


@torch.no_grad()
def evaluate_direction_probe(
    probe: DirectionProbe,
    inputs: torch.Tensor,
    starts: torch.Tensor,
    tasks: TorchTaskBatch,
    *,
    refinement_steps: int,
    refinement_rate: float,
    useful_threshold: float,
) -> dict[str, float]:
    _, _, target_direction, target_alpha = useful_targets(starts, tasks)
    prediction = probe(inputs)
    cosine = torch.sum(prediction * target_direction, dim=-1)
    angular = torch.rad2deg(torch.acos(torch.clamp(cosine, -1.0, 1.0)))
    candidate = starts + target_alpha[:, None] * prediction
    refined = tasks.refine(
        candidate[:, None, :], steps=refinement_steps, rate=refinement_rate
    )
    quality = tasks.quality(refined)[:, 0]
    region = tasks.regions(refined)[:, 0]
    target_hit = (region == tasks.target_region) & (quality >= useful_threshold)
    return {
        "heldout_cosine": float(torch.mean(cosine).cpu()),
        "angular_error_degrees": float(torch.mean(angular).cpu()),
        "recall": float(torch.mean(target_hit.float()).cpu()),
        "parameter_count": float(sum(p.numel() for p in probe.parameters())),
        "uses_oracle_alpha": 1.0,
    }
