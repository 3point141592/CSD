"""Small goal-conditioned candidate critic used by V2 selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

Array = np.ndarray


@dataclass
class GoalConditionalRidgeCritic:
    ridge: float = 0.01
    temperature: float = 0.12
    coefficients: Array | None = None

    @staticmethod
    def features(h: Array, candidates: Array, goal: Array) -> Array:
        h = np.asarray(h, dtype=np.float64)
        candidates = np.asarray(candidates, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        delta = candidates - h[None, :]
        candidate_norm = np.linalg.norm(candidates, axis=1)
        delta_norm = np.linalg.norm(delta, axis=1)
        alignment = np.sum(delta * goal[None, :], axis=1)
        cosine = alignment / np.maximum(delta_norm * np.linalg.norm(goal), 1e-12)
        return np.concatenate(
            [
                np.ones((len(candidates), 1)),
                np.repeat(h[None, :], len(candidates), axis=0),
                candidates,
                np.repeat(goal[None, :], len(candidates), axis=0),
                delta,
                candidates * goal[None, :],
                delta * goal[None, :],
                candidates * candidates,
                delta * delta,
                candidate_norm[:, None],
                (candidate_norm * candidate_norm)[:, None],
                delta_norm[:, None],
                (delta_norm * delta_norm)[:, None],
                alignment[:, None],
                (alignment * alignment)[:, None],
                (alignment * delta_norm)[:, None],
                cosine[:, None],
            ],
            axis=1,
        )

    def fit(self, features: Array, targets: Array) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        identity = np.eye(x.shape[1])
        identity[0, 0] = 0.0
        self.coefficients = np.linalg.solve(
            x.T @ x + self.ridge * identity,
            x.T @ y,
        )

    def predict(self, features: Array) -> Array:
        if self.coefficients is None:
            raise RuntimeError("Critic has not been fitted")
        return np.asarray(features) @ self.coefficients

    def weights(self, features: Array) -> Array:
        scores = self.predict(features) / max(self.temperature, 1e-8)
        scores -= np.max(scores)
        weights = np.exp(scores)
        return weights / np.sum(weights)

    def select(self, features: Array) -> int:
        return int(np.argmax(self.predict(features)))

    def combine(self, candidates: Array, features: Array) -> Array:
        return np.sum(self.weights(features)[:, None] * candidates, axis=0)

    def save(self, path: Path) -> None:
        if self.coefficients is None:
            raise RuntimeError("Cannot save an unfitted critic")
        np.savez_compressed(
            path,
            coefficients=self.coefficients,
            ridge=self.ridge,
            temperature=self.temperature,
        )
