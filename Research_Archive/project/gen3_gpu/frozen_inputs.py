"""Serializable stochastic schedules for strict identical-input validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from creativity_toy.config_v2 import TrainingV2Config

from .operator import BatchedCreativeOperator


@dataclass
class FrozenESSchedule:
    batch_indices: np.ndarray
    noise: np.ndarray
    seed: int
    source: str = "numpy_frozen"

    @property
    def iterations(self) -> int:
        return int(self.batch_indices.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            batch_indices=self.batch_indices,
            noise=self.noise,
            seed=np.asarray(self.seed),
            source=np.asarray(self.source),
        )

    @classmethod
    def load(cls, path: Path) -> FrozenESSchedule:
        payload = np.load(path, allow_pickle=False)
        return cls(
            batch_indices=payload["batch_indices"],
            noise=payload["noise"],
            seed=int(payload["seed"]),
            source=str(payload["source"]),
        )

    def tensors(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(self.batch_indices, dtype=torch.long, device=device),
            torch.as_tensor(
                np.ascontiguousarray(self.noise), dtype=dtype, device=device
            ),
        )


def make_numpy_schedule(
    *,
    operator: BatchedCreativeOperator,
    task_count: int,
    config: TrainingV2Config,
    iterations: int,
    seed: int,
) -> FrozenESSchedule:
    """Reproduce the reference NumPy RNG call order exactly."""
    rng = np.random.default_rng(seed)
    population = config.population + config.population % 2
    half = population // 2
    batch = min(config.batch_size, task_count)
    batch_indices = np.empty((iterations, batch), dtype=np.int64)
    noise = np.empty(
        (iterations, population, operator.parameter_count), dtype=np.float64
    )
    for iteration in range(iterations):
        batch_indices[iteration] = rng.choice(task_count, size=batch, replace=False)
        active, scale = operator.exploration_mask(iteration, iterations)
        half_noise = rng.normal(size=(half, operator.parameter_count))
        half_noise *= scale[None, :]
        half_noise *= active[None, :]
        noise[iteration] = np.concatenate([half_noise, -half_noise], axis=0)
    return FrozenESSchedule(batch_indices, noise, seed, "numpy_frozen")


def make_torch_native_schedule(
    *,
    operator: BatchedCreativeOperator,
    task_count: int,
    config: TrainingV2Config,
    iterations: int,
    seed: int,
    generator_device: torch.device,
) -> FrozenESSchedule:
    """Mode-B schedule: same seed, native CPU/CUDA generator, different samples."""
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    population = config.population + config.population % 2
    half = population // 2
    batch = min(config.batch_size, task_count)
    index_rows: list[np.ndarray] = []
    noise_rows: list[np.ndarray] = []
    for iteration in range(iterations):
        indices = torch.randperm(
            task_count, generator=generator, device=generator_device
        )[:batch]
        active, scale = operator.exploration_mask(iteration, iterations)
        half_noise = torch.randn(
            (half, operator.parameter_count),
            generator=generator,
            device=generator_device,
            dtype=torch.float32,
        )
        half_noise = half_noise * torch.as_tensor(
            scale * active, dtype=torch.float32, device=generator_device
        )
        full_noise = torch.cat([half_noise, -half_noise], dim=0)
        index_rows.append(indices.cpu().numpy())
        noise_rows.append(full_noise.cpu().numpy())
    return FrozenESSchedule(
        np.stack(index_rows),
        np.stack(noise_rows).astype(np.float32),
        seed,
        f"torch_native_{generator_device.type}",
    )


def save_validation_inputs(
    path: Path,
    *,
    starts: np.ndarray,
    goals: np.ndarray,
    initial_parameters: np.ndarray,
    latent_codes: np.ndarray,
    task_arrays: dict[str, np.ndarray],
) -> None:
    """Save all non-schedule tensors that define an identical-input check."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        starts=starts,
        goals=goals,
        initial_parameters=initial_parameters,
        latent_codes=latent_codes,
        **{f"task_{key}": value for key, value in task_arrays.items()},
    )
