"""Augmented supervised training and exact consistency metrics for d32 symmetry."""

from __future__ import annotations

from time import perf_counter

import torch

from .landscape_representation import (
    ComponentDescriptorBatch,
    random_component_permutation,
)
from .runtime import synchronize
from .supervised_capacity import exact_assignment_indices
from .symmetry import (
    EquivariantComponentBatch,
    SETransform,
    random_translations,
    transform_current_descriptors,
    transform_equivariant_components,
    transform_points,
    transform_tasks,
    transform_vectors,
)
from .task_batch import TorchTaskBatch
from .training_landscape_supervised import (
    LandscapeTrainingResult,
    landscape_supervised_objective,
)

ComponentBatch = ComponentDescriptorBatch | EquivariantComponentBatch


def _permute_components(
    components: ComponentBatch,
    *,
    generator: torch.Generator,
) -> ComponentBatch:
    permutation = random_component_permutation(components.mask, generator=generator)
    return components.permute(permutation)


def train_symmetry_policy(
    model: torch.nn.Module,
    tasks: TorchTaskBatch,
    starts: torch.Tensor,
    components: ComponentBatch,
    *,
    component_kind: str,
    augmented: bool,
    rotation_pool: torch.Tensor,
    translation_radius_max: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    lambda_alpha: float,
    seed: int,
    curve_steps: list[int],
) -> LandscapeTrainingResult:
    if component_kind not in {"current", "equivariant"}:
        raise ValueError(component_kind)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    batch_generator = torch.Generator(device=starts.device).manual_seed(seed)
    permutation_generator = torch.Generator(device=starts.device).manual_seed(seed + 1)
    transform_generator = torch.Generator(device=starts.device).manual_seed(seed + 2)
    log_steps = {value for value in curve_steps if 0 < value <= steps}
    log_steps.add(steps)
    history: list[dict[str, float]] = []
    if starts.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(starts.device)
    synchronize(starts.device)
    started = perf_counter()
    count = min(batch_size, len(starts))
    for step in range(1, steps + 1):
        index = torch.randperm(
            len(starts), generator=batch_generator, device=starts.device
        )[:count]
        batch_tasks = tasks.index_select(index)
        batch_h = starts.index_select(0, index)
        batch_components = components.index_select(index)
        if augmented:
            pool_indices = torch.randint(
                len(rotation_pool),
                (count,),
                generator=transform_generator,
                device=starts.device,
            )
            rotation = rotation_pool.index_select(0, pool_indices)
            translation = random_translations(
                count,
                starts.shape[-1],
                translation_radius_max,
                generator=transform_generator,
                device=starts.device,
                dtype=starts.dtype,
            )
            transform = SETransform(rotation, translation)
            batch_tasks = transform_tasks(batch_tasks, transform)
            batch_h = transform_points(batch_h, rotation, translation)
            if component_kind == "current":
                batch_components = transform_current_descriptors(
                    batch_components, rotation, dimension=starts.shape[-1]
                )
            else:
                batch_components = transform_equivariant_components(
                    batch_components, rotation
                )
        batch_components = _permute_components(
            batch_components, generator=permutation_generator
        )
        loss, metrics, _ = landscape_supervised_objective(
            model,
            batch_h,
            batch_tasks,
            batch_components,
            lambda_alpha=lambda_alpha,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip
        )
        optimizer.step()
        if step in log_steps:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "augmented": float(augmented),
                    **{
                        name: float(value.detach().cpu())
                        for name, value in metrics.items()
                    },
                }
            )
    synchronize(starts.device)
    return LandscapeTrainingResult(
        history=history,
        elapsed_seconds=perf_counter() - started,
        examples_seen=steps * count,
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


@torch.no_grad()
def set_equivariance_metrics(
    base_proposal,
    transformed_proposal,
    rotation: torch.Tensor,
) -> dict[str, float]:
    expected = transform_vectors(base_proposal.direction, rotation)
    cosine = torch.einsum("bkd,bmd->bkm", transformed_proposal.direction, expected)
    assignment = exact_assignment_indices(1.0 - cosine, expected.shape[1])
    target = torch.arange(expected.shape[1], device=expected.device)[None, :].expand(
        len(expected), -1
    )
    batch = torch.arange(len(expected), device=expected.device)[:, None]
    matched_cosine = cosine[batch, assignment, target]
    matched_alpha = transformed_proposal.alpha[batch, assignment]
    hungarian_alpha_error = torch.abs(matched_alpha - base_proposal.alpha)
    direct_cosine = torch.sum(transformed_proposal.direction * expected, dim=-1)
    direct_alpha = torch.abs(transformed_proposal.alpha - base_proposal.alpha)
    angular = torch.rad2deg(torch.acos(torch.clamp(matched_cosine, -1.0, 1.0)))
    return {
        "equivariance_cosine": float(torch.mean(matched_cosine).cpu()),
        "equivariance_angular_error_degrees": float(torch.mean(angular).cpu()),
        # Slotwise alpha comparisons preserve the persistent z_i candidate identity.
        # Hungarian matching remains a separate diagnostic because nearly identical
        # directions can swap assignments without preserving that identity.
        "alpha_invariance_error": float(torch.mean(direct_alpha).cpu()),
        "alpha_invariance_max_error": float(torch.max(direct_alpha).cpu()),
        "hungarian_set_alpha_error": float(torch.mean(hungarian_alpha_error).cpu()),
        "direct_slot_equivariance_cosine": float(torch.mean(direct_cosine).cpu()),
        "direct_slot_alpha_error": float(torch.mean(direct_alpha).cpu()),
    }
