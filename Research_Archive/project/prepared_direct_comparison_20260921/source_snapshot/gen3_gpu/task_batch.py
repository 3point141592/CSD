"""Batched torch representation of the unchanged Generation-2 landscapes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
import torch
from creativity_toy.landscapes_v2 import V2Task


@dataclass
class TorchTaskBatch:
    """A task-aligned tensor batch.

    Point tensors have shape ``[P, B, K, d]`` or ``[B, K, d]``.  ``P`` is the
    ES population, ``B`` the landscape/start batch, and ``K`` the candidates.
    Components are padded only for non-IID diagnostic use and masked exactly.
    """

    anchor: torch.Tensor
    goals: torch.Tensor
    component_centers: torch.Tensor
    component_precisions: torch.Tensor
    component_amplitudes: torch.Tensor
    component_mask: torch.Tensor
    useful_centers: torch.Tensor
    useful_precisions: torch.Tensor
    useful_mask: torch.Tensor
    decoy_centers: torch.Tensor
    decoy_precisions: torch.Tensor
    decoy_mask: torch.Tensor
    goal_index: torch.Tensor
    geometry_id: torch.Tensor
    n_useful: torch.Tensor
    target_region: torch.Tensor
    region_threshold: float

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype

    @property
    def batch_size(self) -> int:
        return int(self.anchor.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.anchor.shape[-1])

    def tensors(self) -> list[torch.Tensor]:
        return [
            getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), torch.Tensor)
        ]

    def index_select(self, indices: torch.Tensor) -> TorchTaskBatch:
        values: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.index_select(0, indices.to(value.device))
                if isinstance(value, torch.Tensor)
                else value
            )
        return TorchTaskBatch(**values)  # type: ignore[arg-type]

    def repeat_interleave(self, repeats: int) -> TorchTaskBatch:
        values: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.repeat_interleave(repeats, dim=0)
                if isinstance(value, torch.Tensor)
                else value
            )
        return TorchTaskBatch(**values)  # type: ignore[arg-type]

    def _ensure_population(self, points: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if points.ndim == 3:
            return points.unsqueeze(0), True
        if points.ndim != 4:
            raise ValueError("points must have shape [B,K,d] or [P,B,K,d]")
        return points, False

    def _log_component_scores(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        points4, squeezed = self._ensure_population(points)
        difference = points4.unsqueeze(-2) - self.component_centers.unsqueeze(
            0
        ).unsqueeze(2)
        quadratic = torch.einsum(
            "pbkci,bcij,pbkcj->pbkc",
            difference,
            self.component_precisions,
            difference,
        )
        logs = (
            torch.log(self.component_amplitudes).unsqueeze(0).unsqueeze(2)
            - 0.5 * quadratic
        )
        logs = torch.where(
            self.component_mask.unsqueeze(0).unsqueeze(2),
            logs,
            torch.full_like(logs, -torch.inf),
        )
        return logs, difference, squeezed

    def component_scores(self, points: torch.Tensor) -> torch.Tensor:
        logs, _, squeezed = self._log_component_scores(points)
        scores = torch.exp(torch.clamp(logs, min=-745.0, max=0.0))
        return scores.squeeze(0) if squeezed else scores

    def quality(self, points: torch.Tensor) -> torch.Tensor:
        logs, _, squeezed = self._log_component_scores(points)
        scores = torch.exp(torch.clamp(logs, min=-745.0, max=0.0))
        scores = torch.minimum(scores, torch.full_like(scores, 1.0 - 1e-12))
        log_product = torch.sum(torch.log1p(-scores), dim=-1)
        result = -torch.expm1(log_product)
        return result.squeeze(0) if squeezed else result

    def gradient_log_quality(self, points: torch.Tensor) -> torch.Tensor:
        logs, difference, squeezed = self._log_component_scores(points)
        scores = torch.exp(torch.clamp(logs, min=-745.0, max=0.0))
        scores = torch.minimum(scores, torch.full_like(scores, 1.0 - 1e-12))
        grad_log_component = -torch.einsum(
            "bcij,pbkcj->pbkci", self.component_precisions, difference
        )
        grad_score = scores.unsqueeze(-1) * grad_log_component
        log_product = torch.sum(torch.log1p(-scores), dim=-1)
        product = torch.exp(log_product)
        quality = -torch.expm1(log_product)
        denominator = torch.clamp(1.0 - scores, min=1e-12).unsqueeze(-1)
        grad_quality = product.unsqueeze(-1) * torch.sum(
            grad_score / denominator, dim=-2
        )
        result = grad_quality / torch.clamp(quality, min=1e-12).unsqueeze(-1)

        far = quality < 1e-10
        maximum = torch.amax(logs, dim=-1, keepdim=True)
        weights = torch.exp(logs - maximum)
        weights = weights / torch.clamp(
            torch.sum(weights, dim=-1, keepdim=True), min=1e-12
        )
        mixture_gradient = torch.sum(weights.unsqueeze(-1) * grad_log_component, dim=-2)
        result = torch.where(far.unsqueeze(-1), mixture_gradient, result)
        return result.squeeze(0) if squeezed else result

    def refine(
        self,
        points: torch.Tensor,
        *,
        steps: int,
        rate: float,
        max_gradient_norm: float = 4.0,
    ) -> torch.Tensor:
        """Retain the sequential recurrence while batching all independent axes."""
        result = points.clone()
        for _ in range(steps):
            gradient = self.gradient_log_quality(result)
            norm = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True)
            scale = torch.minimum(
                torch.ones_like(norm),
                torch.as_tensor(max_gradient_norm, dtype=norm.dtype, device=norm.device)
                / torch.clamp(norm, min=1e-12),
            )
            result = result + rate * gradient * scale
        return result

    def regions(self, points: torch.Tensor) -> torch.Tensor:
        """Vectorized exact region rule for evaluation points [B,K,d]."""
        if points.ndim != 3:
            raise ValueError("region points must have shape [B,K,d]")
        anchor_difference = points - self.anchor[:, None, :]
        anchor_width_precision = self.component_precisions[:, 0]
        anchor_q = torch.einsum(
            "bki,bij,bkj->bk",
            anchor_difference,
            anchor_width_precision,
            anchor_difference,
        )
        useful_difference = points.unsqueeze(-2) - self.useful_centers.unsqueeze(1)
        useful_q = torch.einsum(
            "bkui,buij,bkuj->bku",
            useful_difference,
            self.useful_precisions,
            useful_difference,
        )
        useful_q = torch.where(
            self.useful_mask.unsqueeze(1),
            useful_q,
            torch.full_like(useful_q, torch.inf),
        )
        decoy_difference = points.unsqueeze(-2) - self.decoy_centers.unsqueeze(1)
        decoy_q = torch.einsum(
            "bkci,bcij,bkcj->bkc",
            decoy_difference,
            self.decoy_precisions,
            decoy_difference,
        )
        decoy_q = torch.where(
            self.decoy_mask.unsqueeze(1), decoy_q, torch.full_like(decoy_q, torch.inf)
        )
        distances = torch.sqrt(
            torch.clamp(
                torch.cat([anchor_q.unsqueeze(-1), useful_q, decoy_q], dim=-1), min=0.0
            )
        )
        minimum, nearest = torch.min(distances, dim=-1)
        return torch.where(
            minimum <= self.region_threshold,
            nearest,
            torch.full_like(nearest, -1),
        )


def _pad_rows(
    rows: Sequence[np.ndarray], count: int, tail: tuple[int, ...]
) -> np.ndarray:
    result = np.zeros((len(rows), count, *tail), dtype=np.float64)
    for index, row in enumerate(rows):
        result[index, : len(row)] = row
    return result


def pack_tasks(
    tasks: Sequence[V2Task],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> TorchTaskBatch:
    if not tasks:
        raise ValueError("cannot pack an empty task sequence")
    dimension = tasks[0].dimension
    components = max(len(task.component_centers) for task in tasks)
    useful = max(task.n_useful for task in tasks)
    decoys = max(task.n_decoys for task in tasks)
    component_centers = _pad_rows(
        [task.component_centers for task in tasks], components, (dimension,)
    )
    component_precisions = _pad_rows(
        [task.component_precisions for task in tasks],
        components,
        (dimension, dimension),
    )
    component_amplitudes = np.ones((len(tasks), components), dtype=np.float64)
    component_mask = np.zeros((len(tasks), components), dtype=bool)
    useful_mask = np.zeros((len(tasks), useful), dtype=bool)
    decoy_mask = np.zeros((len(tasks), decoys), dtype=bool)
    for index, task in enumerate(tasks):
        component_amplitudes[index, : len(task.component_amplitudes)] = (
            task.component_amplitudes
        )
        component_mask[index, : len(task.component_amplitudes)] = True
        useful_mask[index, : task.n_useful] = True
        decoy_mask[index, : task.n_decoys] = True

    def floating(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device)

    def integer(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(array), dtype=torch.long, device=device
        )

    return TorchTaskBatch(
        anchor=floating(np.stack([task.anchor for task in tasks])),
        goals=floating(np.stack([task.goal for task in tasks])),
        component_centers=floating(component_centers),
        component_precisions=floating(component_precisions),
        component_amplitudes=floating(component_amplitudes),
        component_mask=torch.as_tensor(component_mask, dtype=torch.bool, device=device),
        useful_centers=floating(
            _pad_rows([task.useful_centers for task in tasks], useful, (dimension,))
        ),
        useful_precisions=floating(
            _pad_rows(
                [np.linalg.inv(task.useful_covariances) for task in tasks],
                useful,
                (dimension, dimension),
            )
        ),
        useful_mask=torch.as_tensor(useful_mask, dtype=torch.bool, device=device),
        decoy_centers=floating(
            _pad_rows([task.decoy_centers for task in tasks], decoys, (dimension,))
        ),
        decoy_precisions=floating(
            _pad_rows(
                [np.linalg.inv(task.decoy_covariances) for task in tasks],
                decoys,
                (dimension, dimension),
            )
        ),
        decoy_mask=torch.as_tensor(decoy_mask, dtype=torch.bool, device=device),
        goal_index=integer(np.asarray([task.goal_index for task in tasks])),
        geometry_id=integer(np.asarray([task.geometry_id for task in tasks])),
        n_useful=integer(np.asarray([task.n_useful for task in tasks])),
        target_region=integer(np.asarray([task.target_region for task in tasks])),
        region_threshold=float(tasks[0].config.region_mahalanobis_threshold),
    )
