"""Stochastic action-space policy for the unchanged Generation-3 architecture."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .operator import BatchedCreativeOperator, OperatorOutput

LOG_TWO_PI = float(np.log(2.0 * np.pi))


@dataclass
class PolicyActionBatch:
    direction_mean: torch.Tensor
    raw_direction_action: torch.Tensor
    direction: torch.Tensor
    alpha_raw_mean: torch.Tensor
    alpha_raw_action: torch.Tensor
    alpha: torch.Tensor
    candidate: torch.Tensor
    log_prob_direction: torch.Tensor
    log_prob_alpha: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    mean_direction: torch.Tensor
    mean_alpha: torch.Tensor


class StochasticCreativePolicy(nn.Module):
    """Gen3 contextual-linear policy plus two shared Gaussian log-stds.

    The Gaussian likelihood is evaluated in pre-normalization action space. The
    sampled direction action is then deterministically mapped to the unit sphere.
    """

    def __init__(
        self,
        *,
        dimension: int,
        candidates: int,
        latent_dimension: int,
        hidden_dimension: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        alpha_max: float = 6.0,
        direction_std_init: float = 0.20,
        alpha_std_init: float = 0.35,
    ) -> None:
        super().__init__()
        if direction_std_init <= 0.0 or alpha_std_init <= 0.0:
            raise ValueError("initial policy standard deviations must be positive")
        self.dimension = dimension
        self.candidates = candidates
        self.latent_dimension = latent_dimension
        self.hidden_dimension = hidden_dimension
        self.input_dimension = 2 * dimension + latent_dimension
        self.alpha_input_dimension = 3 * dimension + latent_dimension
        self.alpha_max = float(alpha_max)
        self.dtype = dtype

        reference = BatchedCreativeOperator(
            dimension=dimension,
            candidates=candidates,
            latent_dimension=latent_dimension,
            hidden_dimension=hidden_dimension,
            seed=seed,
            variant="gen3_contextual_linear_alpha",
            device=device,
            dtype=dtype,
            alpha_max=alpha_max,
        )
        values = reference.initial_parameters
        offset = 0
        count = hidden_dimension * self.input_dimension
        self.w1 = nn.Parameter(
            values[offset : offset + count]
            .reshape(hidden_dimension, self.input_dimension)
            .clone()
        )
        offset += count
        self.b1 = nn.Parameter(values[offset : offset + hidden_dimension].clone())
        offset += hidden_dimension
        count = dimension * hidden_dimension
        self.w2 = nn.Parameter(
            values[offset : offset + count].reshape(dimension, hidden_dimension).clone()
        )
        offset += count
        self.b2 = nn.Parameter(values[offset : offset + dimension].clone())
        offset += dimension
        self.alpha_weight = nn.Parameter(
            values[offset : offset + self.alpha_input_dimension]
            .reshape(1, self.alpha_input_dimension)
            .clone()
        )
        offset += self.alpha_input_dimension
        self.alpha_bias = nn.Parameter(values[offset : offset + 1].clone())
        offset += 1
        if offset != values.numel():
            raise RuntimeError(
                "Generation-3 parameter parsing did not consume all values"
            )
        self.direction_log_std = nn.Parameter(
            torch.tensor(log(direction_std_init), dtype=dtype, device=device)
        )
        self.alpha_log_std = nn.Parameter(
            torch.tensor(log(alpha_std_init), dtype=dtype, device=device)
        )
        self.register_buffer("codes", reference.codes.detach().clone())

    @property
    def architecture_parameter_count(self) -> int:
        return int(
            sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if name not in {"direction_log_std", "alpha_log_std"}
            )
        )

    @property
    def policy_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def direction_std(self) -> torch.Tensor:
        return torch.exp(self.direction_log_std)

    def alpha_std(self) -> torch.Tensor:
        return torch.exp(self.alpha_log_std)

    def _base_inputs(self, h: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        batch = h.shape[0]
        return torch.cat(
            [
                h[:, None, :].expand(batch, self.candidates, self.dimension),
                goal[:, None, :].expand(batch, self.candidates, self.dimension),
                self.codes[None, :, :].expand(
                    batch, self.candidates, self.latent_dimension
                ),
            ],
            dim=-1,
        )

    def direction_means(self, h: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        inputs = self._base_inputs(h, goal)
        hidden = torch.tanh(F.linear(inputs, self.w1, self.b1))
        return F.linear(hidden, self.w2, self.b2)

    def alpha_means(
        self, h: torch.Tensor, goal: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        batch = h.shape[0]
        inputs = torch.cat(
            [
                h[:, None, :].expand(batch, self.candidates, self.dimension),
                goal[:, None, :].expand(batch, self.candidates, self.dimension),
                self.codes[None, :, :].expand(
                    batch, self.candidates, self.latent_dimension
                ),
                direction,
            ],
            dim=-1,
        )
        return F.linear(inputs, self.alpha_weight, self.alpha_bias).squeeze(-1)

    @staticmethod
    def _normal_log_prob(
        action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        variance_term = ((action - mean) / torch.exp(log_std)) ** 2
        return -0.5 * (variance_term + 2.0 * log_std + LOG_TWO_PI)

    @staticmethod
    def _normal_entropy(log_std: torch.Tensor) -> torch.Tensor:
        return log_std + 0.5 * (1.0 + LOG_TWO_PI)

    def log_prob_of_actions(
        self,
        h: torch.Tensor,
        goal: torch.Tensor,
        raw_direction_action: torch.Tensor,
        alpha_raw_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction_mean = self.direction_means(h, goal)
        direction = F.normalize(raw_direction_action, p=2, dim=-1, eps=1e-8)
        alpha_mean = self.alpha_means(h, goal, direction)
        direction_log_prob = torch.sum(
            self._normal_log_prob(
                raw_direction_action, direction_mean, self.direction_log_std
            ),
            dim=-1,
        )
        alpha_log_prob = self._normal_log_prob(
            alpha_raw_action, alpha_mean, self.alpha_log_std
        )
        return direction_log_prob + alpha_log_prob, direction_mean, alpha_mean

    def act(
        self,
        h: torch.Tensor,
        goal: torch.Tensor,
        *,
        generator: torch.Generator | None,
        deterministic: bool = False,
    ) -> PolicyActionBatch:
        direction_mean = self.direction_means(h, goal)
        if deterministic:
            raw_direction = direction_mean.detach()
        else:
            noise = torch.randn(
                direction_mean.shape,
                generator=generator,
                dtype=direction_mean.dtype,
                device=direction_mean.device,
            )
            # Detachment is essential for score-function rather than pathwise
            # gradients. The environment sees this sampled action as data.
            raw_direction = (direction_mean + self.direction_std() * noise).detach()
        direction = F.normalize(raw_direction, p=2, dim=-1, eps=1e-8)
        alpha_mean = self.alpha_means(h, goal, direction)
        if deterministic:
            alpha_raw = alpha_mean.detach()
        else:
            noise = torch.randn(
                alpha_mean.shape,
                generator=generator,
                dtype=alpha_mean.dtype,
                device=alpha_mean.device,
            )
            alpha_raw = (alpha_mean + self.alpha_std() * noise).detach()
        alpha = self.alpha_max * torch.sigmoid(alpha_raw)
        candidate = h[:, None, :] + alpha.unsqueeze(-1) * direction
        log_prob, _, _ = self.log_prob_of_actions(h, goal, raw_direction, alpha_raw)
        direction_log_prob = torch.sum(
            self._normal_log_prob(
                raw_direction, direction_mean, self.direction_log_std
            ),
            dim=-1,
        )
        alpha_log_prob = self._normal_log_prob(
            alpha_raw, alpha_mean, self.alpha_log_std
        )
        entropy_value = self.dimension * self._normal_entropy(
            self.direction_log_std
        ) + self._normal_entropy(self.alpha_log_std)
        entropy = entropy_value.expand_as(log_prob)
        mean_direction = F.normalize(direction_mean, p=2, dim=-1, eps=1e-8)
        mean_alpha = self.alpha_max * torch.sigmoid(
            self.alpha_means(h, goal, mean_direction)
        )
        return PolicyActionBatch(
            direction_mean=direction_mean,
            raw_direction_action=raw_direction,
            direction=direction,
            alpha_raw_mean=alpha_mean,
            alpha_raw_action=alpha_raw,
            alpha=alpha,
            candidate=candidate,
            log_prob_direction=direction_log_prob,
            log_prob_alpha=alpha_log_prob,
            log_prob=log_prob,
            entropy=entropy,
            mean_direction=mean_direction,
            mean_alpha=mean_alpha,
        )

    @torch.no_grad()
    def clamp_standard_deviations(
        self,
        *,
        direction_min: float,
        direction_max: float,
        alpha_min: float,
        alpha_max: float,
    ) -> None:
        self.direction_log_std.clamp_(log(direction_min), log(direction_max))
        self.alpha_log_std.clamp_(log(alpha_min), log(alpha_max))

    def deterministic_parameter_vector(self) -> torch.Tensor:
        """Return exact BatchedCreativeOperator contextual-linear ordering."""
        return torch.cat(
            [
                self.w1.reshape(-1),
                self.b1.reshape(-1),
                self.w2.reshape(-1),
                self.b2.reshape(-1),
                self.alpha_weight.reshape(-1),
                self.alpha_bias.reshape(-1),
            ]
        )


class PolicyEvaluationAdapter:
    """Adapter allowing the unchanged batched evaluation/critic infrastructure."""

    def __init__(
        self,
        policy: StochasticCreativePolicy,
        *,
        deterministic: bool,
        seed: int,
    ) -> None:
        self.policy = policy
        self.deterministic = deterministic
        self.device = next(policy.parameters()).device
        self.dtype = next(policy.parameters()).dtype
        self.dimension = policy.dimension
        self.candidates = policy.candidates
        self.variant = (
            "rl_adamw_deterministic" if deterministic else "rl_adamw_stochastic"
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def forward(
        self, theta: torch.Tensor, h: torch.Tensor, goal: torch.Tensor
    ) -> OperatorOutput:
        del theta
        with torch.no_grad():
            actions = self.policy.act(
                h,
                goal,
                generator=None if self.deterministic else self.generator,
                deterministic=self.deterministic,
            )
        return OperatorOutput(
            raw_directions=actions.raw_direction_action.unsqueeze(0),
            directions=actions.direction.unsqueeze(0),
            alphas=actions.alpha.unsqueeze(0),
            candidates=actions.candidate.unsqueeze(0),
        )
