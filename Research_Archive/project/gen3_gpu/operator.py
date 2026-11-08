"""Functional population-batched copy of the unchanged Generation-3 operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from creativity_toy.creative_operator_v2 import ConditionalMLPOperatorV2
from creativity_toy.creative_operator_v3 import ConditionalLearnedAlphaOperatorV3

Variant = Literal[
    "gen2_unnormalized",
    "gen2_normalized_fixed_alpha3",
    "gen3_global_alpha",
    "gen3_contextual_linear_alpha",
    "gen3_contextual_mlp_alpha",
]


@dataclass
class OperatorOutput:
    raw_directions: torch.Tensor
    directions: torch.Tensor
    alphas: torch.Tensor
    candidates: torch.Tensor


class BatchedCreativeOperator:
    """Evaluate one parameter vector or an ES population without Python loops."""

    def __init__(
        self,
        *,
        dimension: int,
        candidates: int,
        latent_dimension: int,
        hidden_dimension: int,
        seed: int,
        variant: Variant,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        alpha_max: float = 6.0,
        alpha_hidden_dimension: int = 32,
    ) -> None:
        self.dimension = dimension
        self.candidates = candidates
        self.latent_dimension = latent_dimension
        self.hidden_dimension = hidden_dimension
        self.variant = variant
        self.device = device
        self.dtype = dtype
        self.alpha_max = float(alpha_max)
        self.alpha_hidden_dimension = int(alpha_hidden_dimension)
        self.input_dimension = 2 * dimension + latent_dimension
        self.alpha_input_dimension = 3 * dimension + latent_dimension
        self._hidden_size = hidden_dimension * self.input_dimension + hidden_dimension
        self.direction_parameter_count = (
            hidden_dimension * self.input_dimension
            + hidden_dimension
            + dimension * hidden_dimension
            + dimension
        )

        if variant.startswith("gen3_"):
            mode = {
                "gen3_global_alpha": "global",
                "gen3_contextual_linear_alpha": "contextual_linear",
                "gen3_contextual_mlp_alpha": "contextual_mlp",
            }[variant]
            reference = ConditionalLearnedAlphaOperatorV3(
                dimension,
                candidates,
                latent_dimension,
                hidden_dimension,
                seed,
                magnitude_mode=mode,
                alpha_max=alpha_max,
                alpha_hidden_dimension=alpha_hidden_dimension,
            )
        else:
            reference = ConditionalMLPOperatorV2(
                dimension,
                candidates,
                latent_dimension,
                hidden_dimension,
                seed,
                normalize_direction=variant == "gen2_normalized_fixed_alpha3",
            )
        self.codes = torch.as_tensor(
            np.ascontiguousarray(reference.codes), dtype=dtype, device=device
        )
        self.initial_parameters = torch.as_tensor(
            np.ascontiguousarray(reference.get_parameters()), dtype=dtype, device=device
        )
        self.parameter_count = int(self.initial_parameters.numel())
        self.additional_alpha_parameters = (
            self.parameter_count - self.direction_parameter_count
        )

    def parameters_from_numpy(self, values: np.ndarray) -> torch.Tensor:
        if values.size != self.parameter_count:
            raise ValueError(
                f"parameter size {values.size} does not match {self.parameter_count}"
            )
        return torch.as_tensor(
            np.ascontiguousarray(values), dtype=self.dtype, device=self.device
        )

    def parameters_to_numpy(self, values: torch.Tensor) -> np.ndarray:
        return values.detach().to(device="cpu", dtype=torch.float64).numpy().copy()

    def _split_direction(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        population = theta.shape[0]
        offset = 0
        count = self.hidden_dimension * self.input_dimension
        w1 = theta[:, offset : offset + count].reshape(
            population, self.hidden_dimension, self.input_dimension
        )
        offset += count
        b1 = theta[:, offset : offset + self.hidden_dimension]
        offset += self.hidden_dimension
        count = self.dimension * self.hidden_dimension
        w2 = theta[:, offset : offset + count].reshape(
            population, self.dimension, self.hidden_dimension
        )
        offset += count
        b2 = theta[:, offset : offset + self.dimension]
        offset += self.dimension
        return w1, b1, w2, b2, offset

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

    def forward(
        self,
        theta: torch.Tensor,
        h: torch.Tensor,
        goal: torch.Tensor,
    ) -> OperatorOutput:
        """Return tensors shaped [P,B,K,*], retaining the population axis."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        if theta.ndim != 2 or theta.shape[1] != self.parameter_count:
            raise ValueError(
                "theta must have shape [parameters] or [population,parameters]"
            )
        w1, b1, w2, b2, offset = self._split_direction(theta)
        inputs = self._base_inputs(h, goal)
        hidden = torch.tanh(
            torch.einsum("bki,phi->pbkh", inputs, w1) + b1[:, None, None, :]
        )
        raw = torch.einsum("pbkh,pdh->pbkd", hidden, w2) + b2[:, None, None, :]

        if self.variant == "gen2_unnormalized":
            direction = raw
            alphas = torch.ones(raw.shape[:-1], dtype=self.dtype, device=self.device)
        else:
            direction = F.normalize(raw, p=2, dim=-1, eps=1e-8)
            if self.variant == "gen2_normalized_fixed_alpha3":
                alphas = torch.full(
                    raw.shape[:-1], 3.0, dtype=self.dtype, device=self.device
                )
            else:
                population, batch = theta.shape[0], h.shape[0]
                if self.variant == "gen3_global_alpha":
                    raw_alpha = theta[:, offset : offset + 1]
                    offset += 1
                    raw_alpha = raw_alpha[:, None, :].expand(
                        population, batch, self.candidates
                    )
                else:
                    alpha_inputs = torch.cat(
                        [
                            h[None, :, None, :].expand(
                                population, batch, self.candidates, self.dimension
                            ),
                            goal[None, :, None, :].expand(
                                population, batch, self.candidates, self.dimension
                            ),
                            self.codes[None, None, :, :].expand(
                                population,
                                batch,
                                self.candidates,
                                self.latent_dimension,
                            ),
                            direction,
                        ],
                        dim=-1,
                    )
                    if self.variant == "gen3_contextual_linear_alpha":
                        count = self.alpha_input_dimension
                        weight = theta[:, offset : offset + count].reshape(
                            population, self.alpha_input_dimension
                        )
                        offset += count
                        bias = theta[:, offset : offset + 1]
                        offset += 1
                        raw_alpha = (
                            torch.einsum("pbka,pa->pbk", alpha_inputs, weight)
                            + bias[:, None, :]
                        )
                    else:
                        count = self.alpha_hidden_dimension * self.alpha_input_dimension
                        first_weight = theta[:, offset : offset + count].reshape(
                            population,
                            self.alpha_hidden_dimension,
                            self.alpha_input_dimension,
                        )
                        offset += count
                        first_bias = theta[
                            :, offset : offset + self.alpha_hidden_dimension
                        ]
                        offset += self.alpha_hidden_dimension
                        count = self.alpha_hidden_dimension
                        final_weight = theta[:, offset : offset + count]
                        offset += count
                        final_bias = theta[:, offset : offset + 1]
                        offset += 1
                        alpha_hidden = F.silu(
                            torch.einsum("pbka,pha->pbkh", alpha_inputs, first_weight)
                            + first_bias[:, None, None, :]
                        )
                        raw_alpha = (
                            torch.einsum("pbkh,ph->pbk", alpha_hidden, final_weight)
                            + final_bias[:, None, :]
                        )
                alphas = self.alpha_max * torch.sigmoid(raw_alpha)
        if offset != self.parameter_count:
            raise RuntimeError(
                f"operator consumed {offset} parameters, expected {self.parameter_count}"
            )
        candidates = h[None, :, None, :] + alphas.unsqueeze(-1) * direction
        return OperatorOutput(raw, direction, alphas, candidates)

    def exploration_mask(
        self, iteration: int, iterations: int
    ) -> tuple[np.ndarray, np.ndarray]:
        active = np.zeros(self.parameter_count, dtype=np.float64)
        scale = np.ones(self.parameter_count, dtype=np.float64)
        if iteration < max(3, iterations // 4) or iteration % 2 == 0:
            active[self._hidden_size : self.direction_parameter_count] = 1.0
            scale[self._hidden_size : self.direction_parameter_count] = 5.0
        else:
            active[: self._hidden_size] = 1.0
            scale[: self._hidden_size] = 0.7
        if self.additional_alpha_parameters:
            active[self.direction_parameter_count :] = 1.0
            scale[self.direction_parameter_count :] = 1.0
        return active, scale

    def direction_regularization(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        direction = theta[:, : self.direction_parameter_count]
        return torch.mean(direction * direction, dim=-1)
