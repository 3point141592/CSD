"""Linear creative operators and the optional lightweight critic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

Array = np.ndarray


class CreativeOperator:
    """Multi-head h_c = h + alpha W_C h, with full or low-rank W_C."""

    def __init__(
        self,
        dimension: int,
        heads: int = 1,
        rank: int | None = None,
        seed: int = 0,
        initialization_scale: float = 1e-3,
    ) -> None:
        self.dimension = dimension
        self.heads = heads
        self.rank = rank
        rng = np.random.default_rng(seed)
        if rank is None:
            self.full = rng.normal(
                scale=initialization_scale / np.sqrt(dimension),
                size=(heads, dimension, dimension),
            )
            self.a = None
            self.b = None
        else:
            if rank <= 0 or rank > dimension:
                raise ValueError("rank must be in [1, dimension]")
            # If both factors start near zero, antithetic ES mutations have the
            # symmetry (+A)(+B)^T = (-A)(-B)^T and provide almost no signal.
            # A tiny A and orthonormal B keep W tiny while breaking that symmetry.
            self.a = rng.normal(
                scale=initialization_scale / np.sqrt(dimension),
                size=(heads, dimension, rank),
            )
            bases = []
            for _ in range(heads):
                basis, _ = np.linalg.qr(
                    rng.normal(size=(dimension, rank)), mode="reduced"
                )
                bases.append(basis)
            self.b = np.stack(bases)
            self.full = None

    @property
    def parameter_count(self) -> int:
        if self.rank is None:
            return self.heads * self.dimension * self.dimension
        return 2 * self.heads * self.dimension * int(self.rank)

    def matrices(self) -> Array:
        if self.full is not None:
            return self.full
        assert self.a is not None and self.b is not None and self.rank is not None
        return np.einsum("kir,kjr->kij", self.a, self.b)

    def apply(self, h: Array, alpha: float) -> Array:
        h = np.asarray(h, dtype=np.float64)
        delta = np.einsum("kij,...j->...ki", self.matrices(), h)
        return h[..., None, :] + alpha * delta

    def perturbation(self, h: Array, alpha: float) -> Array:
        h = np.asarray(h, dtype=np.float64)
        return alpha * np.einsum("kij,...j->...ki", self.matrices(), h)

    def get_parameters(self) -> Array:
        if self.full is not None:
            return self.full.ravel().copy()
        assert self.a is not None and self.b is not None
        return np.concatenate([self.a.ravel(), self.b.ravel()])

    def set_parameters(self, parameters: Array) -> None:
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.size != self.parameter_count:
            raise ValueError(
                f"Expected {self.parameter_count} parameters, got {parameters.size}"
            )
        if self.full is not None:
            self.full[...] = parameters.reshape(self.full.shape)
            return
        assert self.a is not None and self.b is not None
        split = self.a.size
        self.a[...] = parameters[:split].reshape(self.a.shape)
        self.b[...] = parameters[split:].reshape(self.b.shape)

    def initialize_first_head(self, other: CreativeOperator) -> None:
        """Warm-start a multi-head full operator from a trained single head."""
        if self.rank is not None or other.rank is not None:
            return
        if other.heads != 1 or other.dimension != self.dimension:
            raise ValueError("Warm start requires a matching single-head operator")
        assert self.full is not None and other.full is not None
        self.full[0] = other.full[0]

    def save(self, path: Path, **metadata: object) -> None:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "heads": self.heads,
            "rank": -1 if self.rank is None else self.rank,
            "parameters": self.get_parameters(),
        }
        payload.update(metadata)
        np.savez_compressed(path, **payload)


def random_candidates(
    h: Array,
    rms_magnitudes: Array,
    rng: np.random.Generator,
) -> Array:
    """Gaussian perturbations with the requested RMS norm budget."""
    h = np.asarray(h, dtype=np.float64)
    magnitudes = np.asarray(rms_magnitudes, dtype=np.float64)
    epsilon = rng.normal(size=(magnitudes.size, h.size)) * (
        magnitudes[:, None] / np.sqrt(h.size)
    )
    return h[None, :] + epsilon


def pure_divergence_candidates(
    h: Array,
    magnitudes: Array,
    rng: np.random.Generator,
) -> Array:
    """Quality-blind, maximally spread directions at a matched norm budget."""
    h = np.asarray(h, dtype=np.float64)
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    heads = magnitudes.size
    # Orthogonal directions spread this baseline's candidates without using Q.
    if heads > h.size:
        # More heads than dimensions cannot be mutually orthogonal. QR on the
        # transposed layout gives a deterministic spherical frame instead.
        raw = rng.normal(size=(heads, h.size))
        directions, _ = np.linalg.qr(raw, mode="reduced")
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
        )
        if np.linalg.norm(h) > 1e-12:
            directions[0] = -h / np.linalg.norm(h)
        return h[None, :] + magnitudes[:, None] * directions

    raw = rng.normal(size=(h.size, heads))
    directions, _ = np.linalg.qr(raw, mode="reduced")
    directions = directions.T
    # A fixed counter-direction tests divergence without landscape information.
    if np.linalg.norm(h) > 1e-12:
        directions[0] = -h / np.linalg.norm(h)
    return h[None, :] + magnitudes[:, None] * directions


@dataclass
class LinearCritic:
    """Small ridge-regression critic for head quality from candidate + goal summary."""

    coefficients: Array | None = None
    temperature: float = 0.15

    @staticmethod
    def features(h: Array, candidates: Array, goal_center: Array) -> Array:
        h = np.asarray(h, dtype=np.float64)
        c = np.asarray(candidates, dtype=np.float64)
        goal = np.asarray(goal_center, dtype=np.float64)
        delta = c - h[None, :]
        h_norm = np.linalg.norm(h)
        c_norm = np.linalg.norm(c, axis=-1)
        delta_norm = np.linalg.norm(delta, axis=-1)
        cosine = np.sum(c * h[None, :], axis=-1) / np.maximum(c_norm * h_norm, 1e-12)
        return np.concatenate(
            [
                np.ones((c.shape[0], 1)),
                c,
                delta,
                c * h[None, :],
                c * goal[None, :],
                c_norm[:, None],
                delta_norm[:, None],
                cosine[:, None],
            ],
            axis=1,
        )

    def fit(self, features: Array, targets: Array, ridge: float = 1e-3) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        identity = np.eye(x.shape[1])
        identity[0, 0] = 0.0
        self.coefficients = np.linalg.solve(x.T @ x + ridge * identity, x.T @ y)

    def predict(self, features: Array) -> Array:
        if self.coefficients is None:
            raise RuntimeError("Critic must be fitted before prediction")
        return np.asarray(features) @ self.coefficients

    def weights(self, features: Array) -> Array:
        scores = self.predict(features) / max(self.temperature, 1e-6)
        scores -= np.max(scores)
        weights = np.exp(scores)
        return weights / np.sum(weights)

    def combine(self, candidates: Array, features: Array) -> Array:
        return np.sum(self.weights(features)[:, None] * candidates, axis=0)
