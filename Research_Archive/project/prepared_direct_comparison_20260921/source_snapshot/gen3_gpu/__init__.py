"""Separate vectorized PyTorch implementation of the Generation-3 study.

The reference CPU implementation remains in :mod:`creativity_toy`.  Modules in
this package reuse its configuration and deterministic NumPy landscape sampler,
then pack the resulting immutable tasks into batched torch tensors.
"""

from .operator import BatchedCreativeOperator, OperatorOutput
from .task_batch import TorchTaskBatch, pack_tasks

__all__ = [
    "BatchedCreativeOperator",
    "OperatorOutput",
    "TorchTaskBatch",
    "pack_tasks",
]
