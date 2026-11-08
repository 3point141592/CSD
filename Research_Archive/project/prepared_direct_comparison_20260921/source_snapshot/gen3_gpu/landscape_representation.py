"""Leakage-safe component sets and compact non-oracle CSD architectures."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from .operator import BatchedCreativeOperator
from .task_batch import TorchTaskBatch

ALLOWED_COMPONENT_INPUT_FIELDS = (
    "component_centers",
    "component_amplitudes",
    "component_precisions",
    "component_mask",
)
FORBIDDEN_NONORACLE_INPUT_FIELDS = (
    "goals",
    "useful_centers",
    "useful_precisions",
    "useful_mask",
    "decoy_centers",
    "decoy_precisions",
    "decoy_mask",
    "goal_index",
    "target_region",
)


@dataclass
class ComponentDescriptorBatch:
    values: torch.Tensor
    mask: torch.Tensor
    permutation: torch.Tensor | None = None
    source_indices: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def components(self) -> int:
        return int(self.values.shape[1])

    def index_select(self, indices: torch.Tensor) -> ComponentDescriptorBatch:
        return ComponentDescriptorBatch(
            values=self.values.index_select(0, indices),
            mask=self.mask.index_select(0, indices),
            permutation=(
                self.permutation.index_select(0, indices)
                if self.permutation is not None
                else None
            ),
            source_indices=(
                self.source_indices.index_select(0, indices)
                if self.source_indices is not None
                else None
            ),
        )

    def permute(self, permutation: torch.Tensor) -> ComponentDescriptorBatch:
        if permutation.shape != self.mask.shape:
            raise ValueError("component permutation must have shape [B,J]")
        gather = permutation.unsqueeze(-1).expand(-1, -1, self.values.shape[-1])
        return ComponentDescriptorBatch(
            values=torch.gather(self.values, 1, gather),
            mask=torch.gather(self.mask, 1, permutation),
            permutation=permutation,
            source_indices=self.source_indices,
        )


def component_descriptor_dimension(dimension: int) -> int:
    return dimension + 1 + dimension * (dimension + 1) // 2


@torch.no_grad()
def build_component_descriptors(
    tasks: TorchTaskBatch,
    h: torch.Tensor,
    *,
    source_indices: torch.Tensor | None = None,
) -> ComponentDescriptorBatch:
    """Build [mu-h, amplitude, lower-triangle(covariance)] without labels."""
    source = tasks if source_indices is None else tasks.index_select(source_indices)
    if len(h) != source.batch_size:
        raise ValueError("states and component sources must be batch aligned")
    mask = source.component_mask
    dimension = h.shape[-1]
    eye = torch.eye(dimension, dtype=h.dtype, device=h.device)
    safe_precision = torch.where(
        mask[:, :, None, None],
        source.component_precisions,
        eye[None, None, :, :],
    )
    covariance = torch.linalg.inv(safe_precision)
    lower = torch.tril_indices(dimension, dimension, device=h.device)
    covariance_features = covariance[:, :, lower[0], lower[1]]
    relative_center = source.component_centers - h[:, None, :]
    values = torch.cat(
        [
            relative_center,
            source.component_amplitudes.unsqueeze(-1),
            covariance_features,
        ],
        dim=-1,
    )
    values = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
    return ComponentDescriptorBatch(
        values=values.detach(),
        mask=mask.detach(),
        source_indices=source_indices,
    )


@torch.no_grad()
def descriptor_normalization(
    components: ComponentDescriptorBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = components.values[components.mask]
    if selected.numel() == 0:
        raise ValueError("cannot normalize an empty component set")
    mean = torch.mean(selected, dim=0)
    scale = torch.std(selected, dim=0, correction=0).clamp_min(1e-4)
    return mean, scale


def random_component_permutation(
    mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.rand(
        mask.shape, dtype=torch.float32, device=mask.device, generator=generator
    )
    return torch.argsort(noise, dim=1)


@dataclass
class LandscapeProposalBatch:
    raw_direction: torch.Tensor
    direction: torch.Tensor
    alpha: torch.Tensor
    candidate: torch.Tensor
    attention: torch.Tensor | None


class _NonOracleBase(nn.Module):
    condition_name = "NonOracleBase"
    uses_original_goal = False
    uses_refinement_input = False
    allowed_component_fields = ALLOWED_COMPONENT_INPUT_FIELDS
    forbidden_input_fields = FORBIDDEN_NONORACLE_INPUT_FIELDS

    def __init__(
        self,
        *,
        dimension: int,
        candidates: int,
        latent_dimension: int,
        task_token_dimension: int,
        alpha_max: float,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.candidates = candidates
        self.latent_dimension = latent_dimension
        self.task_token_dimension = task_token_dimension
        self.alpha_max = float(alpha_max)
        self.task_token = nn.Parameter(torch.zeros(task_token_dimension, device=device))
        reference = BatchedCreativeOperator(
            dimension=dimension,
            candidates=candidates,
            latent_dimension=latent_dimension,
            hidden_dimension=16,
            seed=seed,
            variant="gen3_contextual_linear_alpha",
            device=device,
            dtype=torch.float32,
            alpha_max=alpha_max,
        )
        self.register_buffer("codes", reference.codes.detach().clone())
        self._generator = torch.Generator(device=device).manual_seed(seed + 313)

    def _linear(
        self, output: int, inputs: int, *, zero: bool = False
    ) -> tuple[nn.Parameter, nn.Parameter]:
        weight = torch.zeros(output, inputs, device=self.task_token.device)
        if not zero:
            weight.normal_(mean=0.0, std=0.04, generator=self._generator)
        return nn.Parameter(weight), nn.Parameter(
            torch.zeros(output, device=weight.device)
        )

    def _expanded_base(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(h)
        return (
            h[:, None, :].expand(batch, self.candidates, self.dimension),
            self.task_token[None, None, :].expand(
                batch, self.candidates, self.task_token_dimension
            ),
            self.codes[None, :, :].expand(
                batch, self.candidates, self.latent_dimension
            ),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


class TaskOnlyPolicy(_NonOracleBase):
    condition_name = "TaskOnly"

    def __init__(self, *, hidden_dimension: int, **kwargs) -> None:
        super().__init__(**kwargs)
        input_dimension = (
            self.dimension + self.task_token_dimension + self.latent_dimension
        )
        self.w1, self.b1 = self._linear(hidden_dimension, input_dimension)
        self.w2, self.b2 = self._linear(self.dimension, hidden_dimension)
        alpha_inputs = input_dimension + self.dimension
        self.alpha_weight, self.alpha_bias = self._linear(1, alpha_inputs, zero=True)

    def forward(
        self, h: torch.Tensor, components: ComponentDescriptorBatch | None = None
    ) -> LandscapeProposalBatch:
        del components
        state, task, codes = self._expanded_base(h)
        base = torch.cat([state, task, codes], dim=-1)
        hidden = torch.tanh(F.linear(base, self.w1, self.b1))
        raw = F.linear(hidden, self.w2, self.b2)
        direction = F.normalize(raw, p=2, dim=-1, eps=1e-8)
        raw_alpha = F.linear(
            torch.cat([base, direction], dim=-1),
            self.alpha_weight,
            self.alpha_bias,
        ).squeeze(-1)
        alpha = self.alpha_max * torch.sigmoid(raw_alpha)
        candidate = state + alpha.unsqueeze(-1) * direction
        return LandscapeProposalBatch(raw, direction, alpha, candidate, None)


class TaskOnlyParameterMatchedPolicy(TaskOnlyPolicy):
    condition_name = "TaskOnlyParameterMatched"

    @classmethod
    def for_target_parameters(cls, target_parameters: int, **kwargs):
        dimension = int(kwargs["dimension"])
        task_dimension = int(kwargs["task_token_dimension"])
        latent = int(kwargs["latent_dimension"])
        base_input = dimension + task_dimension + latent
        fixed = task_dimension + dimension + (base_input + dimension) + 1
        per_hidden = base_input + 1 + dimension
        hidden = max(1, int(round((target_parameters - fixed) / per_hidden)))
        return cls(hidden_dimension=hidden, **kwargs)


class _ComponentPolicyBase(_NonOracleBase):
    def __init__(
        self,
        *,
        descriptor_dimension: int,
        component_hidden_dimension: int,
        component_token_dimension: int,
        descriptor_mean: torch.Tensor,
        descriptor_scale: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor_dimension = descriptor_dimension
        self.component_token_dimension = component_token_dimension
        self.phi1, self.phi1_bias = self._linear(
            component_hidden_dimension, descriptor_dimension
        )
        self.phi2, self.phi2_bias = self._linear(
            component_token_dimension, component_hidden_dimension
        )
        self.register_buffer("descriptor_mean", descriptor_mean.detach().clone())
        self.register_buffer("descriptor_scale", descriptor_scale.detach().clone())

    def encode_components(self, components: ComponentDescriptorBatch) -> torch.Tensor:
        normalized = (
            components.values - self.descriptor_mean[None, None, :]
        ) / self.descriptor_scale[None, None, :]
        normalized = torch.where(
            components.mask.unsqueeze(-1), normalized, torch.zeros_like(normalized)
        )
        hidden = F.silu(F.linear(normalized, self.phi1, self.phi1_bias))
        tokens = F.linear(hidden, self.phi2, self.phi2_bias)
        return torch.where(
            components.mask.unsqueeze(-1), tokens, torch.zeros_like(tokens)
        )


class LandscapeDeepSetsPolicy(_ComponentPolicyBase):
    condition_name = "LandscapeDeepSets"

    def __init__(self, *, proposal_hidden_dimension: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rho_weight, self.rho_bias = self._linear(
            self.task_token_dimension,
            self.task_token_dimension + self.component_token_dimension,
        )
        base_input = self.dimension + self.task_token_dimension + self.latent_dimension
        self.w1, self.b1 = self._linear(proposal_hidden_dimension, base_input)
        self.w2, self.b2 = self._linear(self.dimension, proposal_hidden_dimension)
        self.alpha_weight, self.alpha_bias = self._linear(
            1, base_input + self.dimension, zero=True
        )

    def forward(
        self, h: torch.Tensor, components: ComponentDescriptorBatch | None = None
    ) -> LandscapeProposalBatch:
        if components is None:
            raise ValueError("LandscapeDeepSets requires component descriptors")
        tokens = self.encode_components(components)
        count = components.mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = tokens.sum(dim=1) / count
        task = self.task_token[None, :].expand(len(h), -1)
        rich = torch.tanh(
            F.linear(torch.cat([task, pooled], dim=-1), self.rho_weight, self.rho_bias)
        )
        state, _, codes = self._expanded_base(h)
        rich_expanded = rich[:, None, :].expand(-1, self.candidates, -1)
        base = torch.cat([state, rich_expanded, codes], dim=-1)
        hidden = torch.tanh(F.linear(base, self.w1, self.b1))
        raw = F.linear(hidden, self.w2, self.b2)
        direction = F.normalize(raw, dim=-1, eps=1e-8)
        raw_alpha = F.linear(
            torch.cat([base, direction], dim=-1),
            self.alpha_weight,
            self.alpha_bias,
        ).squeeze(-1)
        alpha = self.alpha_max * torch.sigmoid(raw_alpha)
        candidate = state + alpha.unsqueeze(-1) * direction
        return LandscapeProposalBatch(raw, direction, alpha, candidate, None)


class LandscapeAttentionPolicy(_ComponentPolicyBase):
    condition_name = "LandscapeAttention"

    def __init__(self, *, proposal_hidden_dimension: int, **kwargs) -> None:
        super().__init__(**kwargs)
        query_input = self.dimension + self.task_token_dimension + self.latent_dimension
        token = self.component_token_dimension
        self.query_weight, self.query_bias = self._linear(token, query_input)
        self.key_weight, self.key_bias = self._linear(token, token)
        self.value_weight, self.value_bias = self._linear(token, token)
        proposal_input = query_input + token
        self.w1, self.b1 = self._linear(proposal_hidden_dimension, proposal_input)
        self.w2, self.b2 = self._linear(self.dimension, proposal_hidden_dimension)
        self.alpha_weight, self.alpha_bias = self._linear(
            1, proposal_input + self.dimension, zero=True
        )

    def forward(
        self, h: torch.Tensor, components: ComponentDescriptorBatch | None = None
    ) -> LandscapeProposalBatch:
        if components is None:
            raise ValueError("LandscapeAttention requires component descriptors")
        tokens = self.encode_components(components)
        keys = F.linear(tokens, self.key_weight, self.key_bias)
        values = F.linear(tokens, self.value_weight, self.value_bias)
        state, task, codes = self._expanded_base(h)
        query_input = torch.cat([state, task, codes], dim=-1)
        query = F.linear(query_input, self.query_weight, self.query_bias)
        score = torch.einsum("bkd,bjd->bkj", query, keys) / sqrt(
            self.component_token_dimension
        )
        score = torch.where(
            components.mask[:, None, :], score, torch.full_like(score, -torch.inf)
        )
        attention = torch.softmax(score, dim=-1)
        readout = torch.einsum("bkj,bjd->bkd", attention, values)
        proposal_input = torch.cat([query_input, readout], dim=-1)
        hidden = torch.tanh(F.linear(proposal_input, self.w1, self.b1))
        raw = F.linear(hidden, self.w2, self.b2)
        direction = F.normalize(raw, dim=-1, eps=1e-8)
        raw_alpha = F.linear(
            torch.cat([proposal_input, direction], dim=-1),
            self.alpha_weight,
            self.alpha_bias,
        ).squeeze(-1)
        alpha = self.alpha_max * torch.sigmoid(raw_alpha)
        candidate = state + alpha.unsqueeze(-1) * direction
        return LandscapeProposalBatch(raw, direction, alpha, candidate, attention)


class ComponentUtilityProbe(nn.Module):
    """Diagnostic classifier; labels are targets and never model inputs."""

    def __init__(
        self,
        descriptor_dimension: int,
        hidden_dimension: int,
        *,
        mean: torch.Tensor,
        scale: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(seed)
        self.linear1 = nn.Linear(descriptor_dimension, hidden_dimension, device=device)
        self.linear2 = nn.Linear(hidden_dimension, 1, device=device)
        with torch.no_grad():
            self.linear1.weight.copy_(
                torch.randn(
                    self.linear1.weight.shape, generator=generator, device=device
                )
                * 0.04
            )
            self.linear1.bias.zero_()
            self.linear2.weight.copy_(
                torch.randn(
                    self.linear2.weight.shape, generator=generator, device=device
                )
                * 0.04
            )
            self.linear2.bias.zero_()
        self.register_buffer("descriptor_mean", mean.detach().clone())
        self.register_buffer("descriptor_scale", scale.detach().clone())

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        normalized = (descriptor - self.descriptor_mean) / self.descriptor_scale
        return self.linear2(F.silu(self.linear1(normalized))).squeeze(-1)
