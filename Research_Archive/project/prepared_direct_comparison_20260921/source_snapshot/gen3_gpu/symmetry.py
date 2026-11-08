"""Exact SE(d) transforms and a compact equivariant landscape-attention policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

import torch
import torch.nn.functional as F

from .landscape_representation import (
    ComponentDescriptorBatch,
    LandscapeProposalBatch,
    _NonOracleBase,
)
from .task_batch import TorchTaskBatch


@dataclass
class SETransform:
    rotation: torch.Tensor  # A shared [B,d,d] rotation preserves each task's internal geometry.
    translation: torch.Tensor  # Sharing one [B,d] offset across a task preserves relative point positions.

    def index_select(self, indices: torch.Tensor) -> SETransform:
        return SETransform(
            self.rotation.index_select(0, indices),
            self.translation.index_select(0, indices),
        )


@torch.no_grad()
def proper_orthogonal_matrices(
    count: int,
    dimension: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    raw = torch.randn(
        count, dimension, dimension, generator=generator, device=device, dtype=dtype
    )
    q, triangular = torch.linalg.qr(raw)
    diagonal = torch.diagonal(triangular, dim1=-2, dim2=-1)
    signs = torch.where(
        diagonal >= 0, torch.ones_like(diagonal), -torch.ones_like(diagonal)
    )
    q = q * signs[:, None, :]
    determinant = torch.linalg.det(q)
    correction = torch.where(
        determinant >= 0, torch.ones_like(determinant), -torch.ones_like(determinant)
    )
    q = q.clone()
    q[:, :, -1] *= correction[:, None]
    return q


@torch.no_grad()
def random_translations(
    count: int,
    dimension: int,
    radius_max: float,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    direction = F.normalize(
        torch.randn(count, dimension, generator=generator, device=device, dtype=dtype),
        dim=-1,
        eps=1e-8,
    )
    radius = (
        torch.rand(count, 1, generator=generator, device=device, dtype=dtype)
        * radius_max
    )
    return direction * radius


def transform_vectors(vector: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,b...j->b...i", rotation, vector)


def transform_points(
    point: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    transformed = transform_vectors(point, rotation)
    extra = point.ndim - 2
    return transformed + translation.reshape(
        len(point), *([1] * extra), point.shape[-1]
    )


def transform_matrices(matrix: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bia,b...ac,bjc->b...ij", rotation, matrix, rotation)


@torch.no_grad()
def transform_tasks(tasks: TorchTaskBatch, transform: SETransform) -> TorchTaskBatch:
    """Materialize a fully consistent geometric task transformation."""
    r, t = transform.rotation, transform.translation
    return replace(
        tasks,
        anchor=transform_points(tasks.anchor, r, t),
        goals=transform_vectors(tasks.goals, r),
        component_centers=transform_points(tasks.component_centers, r, t),
        component_precisions=transform_matrices(tasks.component_precisions, r),
        useful_centers=transform_points(tasks.useful_centers, r, t),
        useful_precisions=transform_matrices(tasks.useful_precisions, r),
        decoy_centers=transform_points(tasks.decoy_centers, r, t),
        decoy_precisions=transform_matrices(tasks.decoy_precisions, r),
    )


@torch.no_grad()
def transform_supervised_view(
    tasks: TorchTaskBatch, transform: SETransform
) -> TorchTaskBatch:
    """Transform every field consumed by supervised labels or non-oracle inputs.

    Component descriptors are transformed separately from their lossless
    covariance form. Redundant useful/decoy precision copies are not consumed by
    supervised training; full transformation is used for all evaluation/audits.
    """
    r, t = transform.rotation, transform.translation
    return replace(
        tasks,
        anchor=transform_points(tasks.anchor, r, t),
        goals=transform_vectors(tasks.goals, r),
        component_centers=transform_points(tasks.component_centers, r, t),
        useful_centers=transform_points(tasks.useful_centers, r, t),
        decoy_centers=transform_points(tasks.decoy_centers, r, t),
    )


def _covariance_from_lower(values: torch.Tensor, dimension: int) -> torch.Tensor:
    lower = torch.tril_indices(dimension, dimension, device=values.device)
    covariance = torch.zeros(
        *values.shape[:-1],
        dimension,
        dimension,
        dtype=values.dtype,
        device=values.device,
    )
    covariance[..., lower[0], lower[1]] = values
    covariance[..., lower[1], lower[0]] = values
    return covariance


@torch.no_grad()
def transform_current_descriptors(
    components: ComponentDescriptorBatch,
    rotation: torch.Tensor,
    *,
    dimension: int,
) -> ComponentDescriptorBatch:
    relative = components.values[..., :dimension]
    amplitude = components.values[..., dimension : dimension + 1]
    covariance_lower = components.values[..., dimension + 1 :]
    covariance = _covariance_from_lower(covariance_lower, dimension)
    relative_rotated = transform_vectors(relative, rotation)
    covariance_rotated = transform_matrices(covariance, rotation)
    lower = torch.tril_indices(dimension, dimension, device=rotation.device)
    values = torch.cat(
        [
            relative_rotated,
            amplitude,
            covariance_rotated[..., lower[0], lower[1]],
        ],
        dim=-1,
    )
    values = torch.where(
        components.mask.unsqueeze(-1), values, torch.zeros_like(values)
    )
    return ComponentDescriptorBatch(
        values.detach(), components.mask, source_indices=components.source_indices
    )


@dataclass
class EquivariantComponentBatch:
    invariant: torch.Tensor  # Component indexing [B,J,F] keeps scoring features paired with their vectors.
    relative: torch.Tensor  # Component indexing [B,J,d] preserves the weight-vector correspondence.
    mask: torch.Tensor
    permutation: torch.Tensor | None = None

    def index_select(self, indices: torch.Tensor) -> EquivariantComponentBatch:
        return EquivariantComponentBatch(
            self.invariant.index_select(0, indices),
            self.relative.index_select(0, indices),
            self.mask.index_select(0, indices),
            self.permutation.index_select(0, indices)
            if self.permutation is not None
            else None,
        )

    def permute(self, permutation: torch.Tensor) -> EquivariantComponentBatch:
        scalar_index = permutation.unsqueeze(-1).expand(
            -1, -1, self.invariant.shape[-1]
        )
        vector_index = permutation.unsqueeze(-1).expand(-1, -1, self.relative.shape[-1])
        return EquivariantComponentBatch(
            torch.gather(self.invariant, 1, scalar_index),
            torch.gather(self.relative, 1, vector_index),
            torch.gather(self.mask, 1, permutation),
            permutation,
        )


def invariant_descriptor_dimension(dimension: int) -> int:
    return dimension + 11


@torch.no_grad()
def build_equivariant_components(
    tasks: TorchTaskBatch,
    h: torch.Tensor,
    *,
    eigen_batch_size: int | None = None,
) -> EquivariantComponentBatch:
    relative = tasks.component_centers - h[:, None, :]
    mask = tasks.component_mask
    dimension = h.shape[-1]
    eye = torch.eye(dimension, dtype=h.dtype, device=h.device)
    # CUDA eigvalsh has a large temporary workspace at d=64. Components are
    # independent across the batch, so chunking this exact computation changes
    # neither examples nor mathematics and prevents a >32 GiB workspace request.
    if eigen_batch_size is None:
        eigen_batch_size = 256 if dimension >= 64 else len(h)
    eigenvalue_chunks = []
    for start in range(0, len(h), eigen_batch_size):
        stop = min(start + eigen_batch_size, len(h))
        chunk_mask = mask[start:stop]
        precision_chunk = torch.where(
            chunk_mask[:, :, None, None],
            tasks.component_precisions[start:stop],
            eye[None, None, :, :],
        )
        eigenvalue_chunks.append(torch.linalg.eigvalsh(precision_chunk).clamp_min(1e-8))
    precision_eigenvalues = torch.cat(eigenvalue_chunks, dim=0)
    covariance_eigenvalues = torch.sort(1.0 / precision_eigenvalues, dim=-1).values
    trace = torch.sum(covariance_eigenvalues, dim=-1, keepdim=True)
    logdet = torch.sum(torch.log(covariance_eigenvalues), dim=-1, keepdim=True)
    norm = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
    quadratic = torch.einsum(
        "bji,bjik,bjk->bj", relative, tasks.component_precisions, relative
    ).unsqueeze(-1)

    delta = relative[:, :, None, :] - relative[:, None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    unit = F.normalize(relative, dim=-1, eps=1e-8)
    cosine = torch.matmul(unit, unit.transpose(-1, -2))
    pair_mask = mask[:, :, None] & mask[:, None, :]
    diagonal = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[
        None, :, :
    ]
    pair_mask = pair_mask & ~diagonal
    count = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    distance_mean = (
        torch.sum(
            torch.where(pair_mask, distance, torch.zeros_like(distance)),
            dim=-1,
            keepdim=True,
        )
        / count
    )
    cosine_mean = (
        torch.sum(
            torch.where(pair_mask, cosine, torch.zeros_like(cosine)),
            dim=-1,
            keepdim=True,
        )
        / count
    )
    distance_min = torch.amin(
        torch.where(pair_mask, distance, torch.full_like(distance, torch.inf)),
        dim=-1,
        keepdim=True,
    )
    distance_max = torch.amax(
        torch.where(pair_mask, distance, torch.full_like(distance, -torch.inf)),
        dim=-1,
        keepdim=True,
    )
    cosine_min = torch.amin(
        torch.where(pair_mask, cosine, torch.full_like(cosine, torch.inf)),
        dim=-1,
        keepdim=True,
    )
    cosine_max = torch.amax(
        torch.where(pair_mask, cosine, torch.full_like(cosine, -torch.inf)),
        dim=-1,
        keepdim=True,
    )
    relation = torch.cat(
        [
            distance_mean,
            distance_min,
            distance_max,
            cosine_mean,
            cosine_min,
            cosine_max,
        ],
        dim=-1,
    )
    invariant = torch.cat(
        [
            tasks.component_amplitudes.unsqueeze(-1),
            norm,
            trace,
            logdet,
            quadratic,
            covariance_eigenvalues,
            relation,
        ],
        dim=-1,
    )
    invariant = torch.where(mask.unsqueeze(-1), invariant, torch.zeros_like(invariant))
    relative = torch.where(mask.unsqueeze(-1), relative, torch.zeros_like(relative))
    return EquivariantComponentBatch(
        invariant.detach(), relative.detach(), mask.detach()
    )


@torch.no_grad()
def invariant_normalization(
    components: EquivariantComponentBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = components.invariant[components.mask]
    return (
        selected.mean(dim=0),
        selected.std(dim=0, correction=0).clamp_min(1e-4),
    )


@torch.no_grad()
def transform_equivariant_components(
    components: EquivariantComponentBatch, rotation: torch.Tensor
) -> EquivariantComponentBatch:
    return EquivariantComponentBatch(
        components.invariant,
        transform_vectors(components.relative, rotation),
        components.mask,
        components.permutation,
    )


class EquivariantLandscapeAttention(_NonOracleBase):
    """Invariant scalar attention with an exactly equivariant vector output."""

    condition_name = "EquivariantAttention"

    def __init__(
        self,
        *,
        invariant_dimension: int,
        component_hidden_dimension: int,
        component_token_dimension: int,
        invariant_mean: torch.Tensor,
        invariant_scale: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        token = component_token_dimension
        self.component_token_dimension = token
        self.phi1, self.phi1_bias = self._linear(
            component_hidden_dimension, invariant_dimension
        )
        self.phi2, self.phi2_bias = self._linear(token, component_hidden_dimension)
        query_input = self.task_token_dimension + self.latent_dimension
        self.query_weight, self.query_bias = self._linear(token, query_input)
        self.key_weight, self.key_bias = self._linear(token, token)
        self.value_weight, self.value_bias = self._linear(token, token)
        self.alpha_weight, self.alpha_bias = self._linear(
            1, query_input + token + 1, zero=True
        )
        self.register_buffer("invariant_mean", invariant_mean.detach().clone())
        self.register_buffer("invariant_scale", invariant_scale.detach().clone())

    def forward(
        self, h: torch.Tensor, components: EquivariantComponentBatch | None = None
    ) -> LandscapeProposalBatch:
        if components is None:
            raise ValueError(
                "EquivariantLandscapeAttention requires component geometry"
            )
        normalized = (
            components.invariant - self.invariant_mean[None, None, :]
        ) / self.invariant_scale[None, None, :]
        normalized = torch.where(
            components.mask.unsqueeze(-1), normalized, torch.zeros_like(normalized)
        )
        hidden = F.silu(F.linear(normalized, self.phi1, self.phi1_bias))
        token = F.linear(hidden, self.phi2, self.phi2_bias)
        token = torch.where(
            components.mask.unsqueeze(-1), token, torch.zeros_like(token)
        )
        key = F.linear(token, self.key_weight, self.key_bias)
        value = F.linear(token, self.value_weight, self.value_bias)
        _, task, codes = self._expanded_base(h)
        query_input = torch.cat([task, codes], dim=-1)
        query = F.linear(query_input, self.query_weight, self.query_bias)
        score = torch.einsum("bkd,bjd->bkj", query, key) / sqrt(
            self.component_token_dimension
        )
        score = torch.where(
            components.mask[:, None, :], score, torch.full_like(score, -torch.inf)
        )
        attention = torch.softmax(score, dim=-1)
        readout = torch.einsum("bkj,bjd->bkd", attention, value)
        raw = torch.einsum("bkj,bjd->bkd", attention, components.relative)
        direction = F.normalize(raw, dim=-1, eps=1e-8)
        raw_norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        alpha_input = torch.cat([query_input, readout, raw_norm], dim=-1)
        raw_alpha = F.linear(alpha_input, self.alpha_weight, self.alpha_bias).squeeze(
            -1
        )
        alpha = self.alpha_max * torch.sigmoid(raw_alpha)
        candidate = h[:, None, :] + alpha.unsqueeze(-1) * direction
        return LandscapeProposalBatch(raw, direction, alpha, candidate, attention)
