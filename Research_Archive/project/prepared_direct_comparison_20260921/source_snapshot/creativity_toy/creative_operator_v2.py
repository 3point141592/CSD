"""Generation-1 and Generation-2 creative exploration operators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

Array = np.ndarray


def _latent_codes(candidates: int, dimension: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    if dimension == 1:
        return np.linspace(-1.0, 1.0, candidates)[:, None]
    codes = rng.normal(size=(candidates, dimension))
    codes -= np.mean(codes, axis=0, keepdims=True)
    codes /= np.maximum(np.linalg.norm(codes, axis=1, keepdims=True), 1e-12)
    return codes


class BaseCreativeOperatorV2:
    dimension: int
    candidates: int
    name: str

    @property
    def parameter_count(self) -> int:
        return int(self.get_parameters().size)

    def apply(self, h: Array, goal: Array, alpha: float) -> Array:
        raise NotImplementedError

    def get_parameters(self) -> Array:
        raise NotImplementedError

    def set_parameters(self, parameters: Array) -> None:
        raise NotImplementedError

    def exploration_mask(self, iteration: int, iterations: int) -> tuple[Array, Array]:
        ones = np.ones(self.parameter_count, dtype=np.float64)
        return ones, ones

    def regularization(self) -> float:
        parameters = self.get_parameters()
        return float(np.mean(parameters * parameters))

    def save(self, path: Path, **metadata: object) -> None:
        payload: dict[str, object] = {
            "name": self.name,
            "dimension": self.dimension,
            "candidates": self.candidates,
            "parameters": self.get_parameters(),
        }
        if hasattr(self, "normalize_direction"):
            payload["normalize_direction"] = bool(self.normalize_direction)
        payload.update(metadata)
        np.savez_compressed(path, **payload)


class Generation1OperatorV2(BaseCreativeOperatorV2):
    """Goal-blind Generation-1 heads: h_c = h + alpha W_k h."""

    def __init__(
        self,
        dimension: int,
        candidates: int,
        seed: int,
        initialization_scale: float = 1e-3,
    ) -> None:
        self.dimension = dimension
        self.candidates = candidates
        self.name = "generation1"
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            scale=initialization_scale / np.sqrt(dimension),
            size=(candidates, dimension, dimension),
        )

    def apply(self, h: Array, goal: Array, alpha: float) -> Array:
        del goal
        delta = np.einsum("kij,j->ki", self.weights, np.asarray(h))
        return np.asarray(h)[None, :] + alpha * delta

    def get_parameters(self) -> Array:
        return self.weights.ravel().copy()

    def set_parameters(self, parameters: Array) -> None:
        self.weights[...] = np.asarray(parameters).reshape(self.weights.shape)

    def exploration_mask(self, iteration: int, iterations: int) -> tuple[Array, Array]:
        del iteration, iterations
        return (
            np.ones(self.parameter_count),
            np.full(self.parameter_count, 5.0),
        )


class ConditionalMLPOperatorV2(BaseCreativeOperatorV2):
    """Tiny shared MLP f(h, g, z_k) producing K conditional directions."""

    def __init__(
        self,
        dimension: int,
        candidates: int,
        latent_dimension: int,
        hidden_dimension: int,
        seed: int,
        *,
        use_goal: bool = True,
        normalize_direction: bool = False,
    ) -> None:
        self.dimension = dimension
        self.candidates = candidates
        self.latent_dimension = latent_dimension
        self.hidden_dimension = hidden_dimension
        self.use_goal = use_goal
        self.normalize_direction = normalize_direction
        if use_goal:
            self.name = (
                "generation2_normalized"
                if normalize_direction
                else "generation2_unnormalized"
            )
        else:
            self.name = (
                "generation2_no_goal_normalized"
                if normalize_direction
                else "generation2_no_goal"
            )
        rng = np.random.default_rng(seed)
        self.codes = _latent_codes(candidates, latent_dimension, seed + 991)
        self.input_dimension = 2 * dimension + latent_dimension
        self.w1 = rng.normal(
            scale=0.28 / np.sqrt(self.input_dimension),
            size=(hidden_dimension, self.input_dimension),
        )
        self.b1 = rng.normal(scale=0.02, size=hidden_dimension)
        self.w2 = rng.normal(
            scale=1e-3 / np.sqrt(hidden_dimension),
            size=(dimension, hidden_dimension),
        )
        self.b2 = np.zeros(dimension)
        self._hidden_size = self.w1.size + self.b1.size

    def _inputs(self, h: Array, goal: Array) -> Array:
        h = np.asarray(h, dtype=np.float64)
        goal = (
            np.asarray(goal, dtype=np.float64) if self.use_goal else np.zeros_like(goal)
        )
        return np.concatenate(
            [
                np.repeat(h[None, :], self.candidates, axis=0),
                np.repeat(goal[None, :], self.candidates, axis=0),
                self.codes,
            ],
            axis=1,
        )

    def raw_directions(self, h: Array, goal: Array) -> Array:
        """Return the MLP output before any direction normalization."""
        inputs = self._inputs(h, goal)
        hidden = np.tanh(inputs @ self.w1.T + self.b1[None, :])
        return hidden @ self.w2.T + self.b2[None, :]

    @staticmethod
    def unit_directions(raw_directions: Array) -> Array:
        """Normalize with PyTorch exactly as specified by the experiment."""
        raw_tensor = torch.as_tensor(
            np.ascontiguousarray(raw_directions), dtype=torch.float64, device="cpu"
        )
        normalized = F.normalize(raw_tensor, p=2, dim=-1, eps=1e-8)
        return normalized.detach().cpu().numpy()

    def directions(self, h: Array, goal: Array) -> Array:
        raw = self.raw_directions(h, goal)
        return self.unit_directions(raw) if self.normalize_direction else raw

    def direction_diagnostics(
        self, h: Array, goal: Array, alpha: float
    ) -> dict[str, Array]:
        raw = self.raw_directions(h, goal)
        normalized = self.unit_directions(raw)
        applied = normalized if self.normalize_direction else raw
        perturbation = alpha * applied
        return {
            "raw_direction": raw,
            "normalized_direction": normalized,
            "applied_direction": applied,
            "raw_norm": np.linalg.norm(raw, axis=-1),
            "normalized_norm": np.linalg.norm(normalized, axis=-1),
            "applied_direction_norm": np.linalg.norm(applied, axis=-1),
            "perturbation_norm": np.linalg.norm(perturbation, axis=-1),
        }

    def apply(self, h: Array, goal: Array, alpha: float) -> Array:
        return np.asarray(h)[None, :] + alpha * self.directions(h, goal)

    def get_parameters(self) -> Array:
        return np.concatenate(
            [self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()]
        )

    def set_parameters(self, parameters: Array) -> None:
        values = np.asarray(parameters, dtype=np.float64)
        offset = 0
        for target in [self.w1, self.b1, self.w2, self.b2]:
            end = offset + target.size
            target[...] = values[offset:end].reshape(target.shape)
            offset = end
        if offset != values.size:
            raise ValueError("Unexpected conditional MLP parameter count")

    def exploration_mask(self, iteration: int, iterations: int) -> tuple[Array, Array]:
        active = np.zeros(self.parameter_count, dtype=np.float64)
        scale = np.ones(self.parameter_count, dtype=np.float64)
        if iteration < max(3, iterations // 4) or iteration % 2 == 0:
            active[self._hidden_size :] = 1.0
            scale[self._hidden_size :] = 5.0
        else:
            active[: self._hidden_size] = 1.0
            scale[: self._hidden_size] = 0.7
        return active, scale


class ConditionalLinearOperatorV2(BaseCreativeOperatorV2):
    """Ablation: epsilon = W [h; g; z] + b."""

    def __init__(
        self,
        dimension: int,
        candidates: int,
        latent_dimension: int,
        seed: int,
        *,
        use_goal: bool = True,
    ) -> None:
        self.dimension = dimension
        self.candidates = candidates
        self.latent_dimension = latent_dimension
        self.use_goal = use_goal
        self.name = "generation2_linear" if use_goal else "linear_no_goal"
        self.input_dimension = 2 * dimension + latent_dimension
        rng = np.random.default_rng(seed)
        self.codes = _latent_codes(candidates, latent_dimension, seed + 313)
        self.weights = rng.normal(
            scale=1e-3 / np.sqrt(self.input_dimension),
            size=(dimension, self.input_dimension),
        )
        self.bias = np.zeros(dimension)

    def apply(self, h: Array, goal: Array, alpha: float) -> Array:
        h = np.asarray(h, dtype=np.float64)
        goal = (
            np.asarray(goal, dtype=np.float64) if self.use_goal else np.zeros_like(goal)
        )
        inputs = np.concatenate(
            [
                np.repeat(h[None, :], self.candidates, axis=0),
                np.repeat(goal[None, :], self.candidates, axis=0),
                self.codes,
            ],
            axis=1,
        )
        return h[None, :] + alpha * (inputs @ self.weights.T + self.bias[None, :])

    def get_parameters(self) -> Array:
        return np.concatenate([self.weights.ravel(), self.bias.ravel()])

    def set_parameters(self, parameters: Array) -> None:
        values = np.asarray(parameters, dtype=np.float64)
        split = self.weights.size
        self.weights[...] = values[:split].reshape(self.weights.shape)
        self.bias[...] = values[split:].reshape(self.bias.shape)

    def exploration_mask(self, iteration: int, iterations: int) -> tuple[Array, Array]:
        del iteration, iterations
        return (
            np.ones(self.parameter_count),
            np.full(self.parameter_count, 5.0),
        )
