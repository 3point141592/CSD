"""Stable import name for the existing vectorized ES baseline.

No ES logic is duplicated or changed; the implementation remains in training.py.
"""

from .training import (
    ObjectiveResult,
    TrainingResult,
    operator_objective_batched,
    rank_utilities,
    train_operator_batched,
)

__all__ = [
    "ObjectiveResult",
    "TrainingResult",
    "operator_objective_batched",
    "rank_utilities",
    "train_operator_batched",
]
