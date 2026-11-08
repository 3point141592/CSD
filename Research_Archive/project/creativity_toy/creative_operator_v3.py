"""Generation-3 operators that separate normalized direction and magnitude."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .creative_operator_v2 import ConditionalMLPOperatorV2

Array = np.ndarray


class ConditionalLearnedAlphaOperatorV3(ConditionalMLPOperatorV2):
    """Normalized V2 direction generator plus a bounded learned alpha head."""

    MODES = {"global", "contextual_linear", "contextual_mlp"}

    def __init__(
        self,
        dimension: int,
        candidates: int,
        latent_dimension: int,
        hidden_dimension: int,
        seed: int,
        *,
        magnitude_mode: str = "contextual_linear",
        alpha_max: float = 6.0,
        alpha_hidden_dimension: int = 32,
    ) -> None:
        if magnitude_mode not in self.MODES:
            raise ValueError(
                f"Unknown magnitude mode {magnitude_mode!r}; choose {sorted(self.MODES)}"
            )
        if alpha_max <= 0.0:
            raise ValueError("alpha_max must be positive")
        super().__init__(
            dimension,
            candidates,
            latent_dimension,
            hidden_dimension,
            seed,
            use_goal=True,
            normalize_direction=True,
        )
        self.magnitude_mode = magnitude_mode
        self.alpha_max = float(alpha_max)
        self.alpha_hidden_dimension = int(alpha_hidden_dimension)
        self.alpha_input_dimension = 3 * dimension + latent_dimension
        self._direction_parameter_count = super().get_parameters().size

        if magnitude_mode == "global":
            self.raw_global_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float64))
            self.alpha_head: nn.Module | None = None
        elif magnitude_mode == "contextual_linear":
            head = nn.Linear(
                self.alpha_input_dimension, 1, bias=True, dtype=torch.float64
            )
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.alpha_head = head
            self.raw_global_alpha = None
        else:
            first = nn.Linear(
                self.alpha_input_dimension,
                self.alpha_hidden_dimension,
                bias=True,
                dtype=torch.float64,
            )
            final = nn.Linear(
                self.alpha_hidden_dimension, 1, bias=True, dtype=torch.float64
            )
            rng = np.random.default_rng(seed + 7_303)
            bound = 1.0 / np.sqrt(self.alpha_input_dimension)
            with torch.no_grad():
                first.weight.copy_(
                    torch.as_tensor(
                        rng.uniform(-bound, bound, size=tuple(first.weight.shape)),
                        dtype=torch.float64,
                    )
                )
                first.bias.copy_(
                    torch.as_tensor(
                        rng.uniform(-bound, bound, size=tuple(first.bias.shape)),
                        dtype=torch.float64,
                    )
                )
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            self.alpha_head = nn.Sequential(first, nn.SiLU(), final)
            self.raw_global_alpha = None
        self.name = f"generation3_{magnitude_mode}_alpha"

    @property
    def additional_alpha_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self._alpha_parameters()))

    @property
    def direction_parameter_count(self) -> int:
        return int(self._direction_parameter_count)

    def _alpha_parameters(self) -> list[nn.Parameter]:
        if self.magnitude_mode == "global":
            assert self.raw_global_alpha is not None
            return [self.raw_global_alpha]
        assert self.alpha_head is not None
        return list(self.alpha_head.parameters())

    def _alpha_inputs(self, h: Array, goal: Array, direction: Array) -> torch.Tensor:
        rows = np.concatenate(
            [
                np.repeat(np.asarray(h)[None, :], self.candidates, axis=0),
                np.repeat(np.asarray(goal)[None, :], self.candidates, axis=0),
                self.codes,
                direction,
            ],
            axis=1,
        )
        return torch.as_tensor(
            np.ascontiguousarray(rows), dtype=torch.float64, device="cpu"
        )

    def learned_alphas(
        self, h: Array, goal: Array, direction: Array | None = None
    ) -> Array:
        if direction is None:
            direction = self.directions(h, goal)
        if self.magnitude_mode == "global":
            assert self.raw_global_alpha is not None
            raw = self.raw_global_alpha.expand(self.candidates)
        else:
            assert self.alpha_head is not None
            raw = self.alpha_head(self._alpha_inputs(h, goal, direction)).squeeze(-1)
        values = self.alpha_max * torch.sigmoid(raw)
        return values.detach().cpu().numpy()

    def apply(self, h: Array, goal: Array, alpha: float) -> Array:
        del alpha
        direction = self.directions(h, goal)
        magnitudes = self.learned_alphas(h, goal, direction)
        return np.asarray(h)[None, :] + magnitudes[:, None] * direction

    def direction_diagnostics(
        self, h: Array, goal: Array, alpha: float
    ) -> dict[str, Array]:
        del alpha
        raw = self.raw_directions(h, goal)
        normalized = self.unit_directions(raw)
        magnitudes = self.learned_alphas(h, goal, normalized)
        perturbation = magnitudes[:, None] * normalized
        return {
            "raw_direction": raw,
            "normalized_direction": normalized,
            "applied_direction": normalized,
            "alpha_values": magnitudes,
            "raw_norm": np.linalg.norm(raw, axis=-1),
            "normalized_norm": np.linalg.norm(normalized, axis=-1),
            "applied_direction_norm": np.linalg.norm(normalized, axis=-1),
            "perturbation_norm": np.linalg.norm(perturbation, axis=-1),
        }

    def get_parameters(self) -> Array:
        direction = super().get_parameters()
        alpha = np.concatenate(
            [
                parameter.detach().cpu().numpy().ravel()
                for parameter in self._alpha_parameters()
            ]
        )
        return np.concatenate([direction, alpha])

    def set_parameters(self, parameters: Array) -> None:
        values = np.asarray(parameters, dtype=np.float64)
        direction = values[: self._direction_parameter_count]
        super().set_parameters(direction)
        offset = self._direction_parameter_count
        with torch.no_grad():
            for target in self._alpha_parameters():
                end = offset + target.numel()
                target.copy_(
                    torch.as_tensor(
                        values[offset:end].reshape(tuple(target.shape)),
                        dtype=torch.float64,
                    )
                )
                offset = end
        if offset != values.size:
            raise ValueError("Unexpected Generation-3 parameter count")

    def exploration_mask(self, iteration: int, iterations: int) -> tuple[Array, Array]:
        active = np.zeros(self.parameter_count, dtype=np.float64)
        scale = np.ones(self.parameter_count, dtype=np.float64)
        if iteration < max(3, iterations // 4) or iteration % 2 == 0:
            active[self._hidden_size : self._direction_parameter_count] = 1.0
            scale[self._hidden_size : self._direction_parameter_count] = 5.0
        else:
            active[: self._hidden_size] = 1.0
            scale[: self._hidden_size] = 0.7
        active[self._direction_parameter_count :] = 1.0
        scale[self._direction_parameter_count :] = 1.0
        return active, scale

    def regularization(self) -> float:
        """Preserve V2 direction regularization; alpha has its own optional term."""
        direction = super().get_parameters()
        return float(np.mean(direction * direction))

    def alpha_regularization(
        self, h: Array, goal: Array, alpha_reference: float
    ) -> float:
        values = self.learned_alphas(h, goal)
        return float(np.mean((values - alpha_reference) ** 2))

    def save(self, path: Path, **metadata: object) -> None:
        super().save(
            path,
            magnitude_mode=self.magnitude_mode,
            alpha_max=self.alpha_max,
            additional_alpha_parameters=self.additional_alpha_parameters,
            **metadata,
        )
