"""Full-rank 16D extension of the Generation-2 landscape family.

This module is deliberately separate from :mod:`landscapes_v2`: the legacy
``dimension > 2`` path embeds all centres in a random two-dimensional plane.
Here useful and decoy displacements are constructed from a full 16D frame, so
the benchmark changes dimension without mutating any established 2D result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config_v2 import LandscapeV2Config
from .landscapes_v2 import V2Task

Array = np.ndarray
D16_DIMENSION = 16


def random_orthogonal(rng: np.random.Generator, dimension: int) -> Array:
    """Draw a deterministic Haar-distributed proper orthogonal matrix."""
    raw = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(raw)
    signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    q = q * signs[None, :]
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _near_identity_rotation(
    rng: np.random.Generator, dimension: int, scale: float
) -> Array:
    raw = np.eye(dimension) + scale * rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(raw)
    signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    return q * signs[None, :]


def _unit(vector: Array) -> Array:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _anisotropic_covariance(
    direction: Array, radial_width: float, lateral_width: float
) -> Array:
    unit = _unit(np.asarray(direction, dtype=np.float64))
    projector = np.outer(unit, unit)
    identity = np.eye(unit.size, dtype=np.float64)
    return radial_width**2 * projector + lateral_width**2 * (identity - projector)


def effective_rank(values: Array, *, squared: bool = True) -> float:
    """Entropy effective rank of singular values (or covariance eigenvalues)."""
    spectrum = np.asarray(values, dtype=np.float64)
    spectrum = np.square(spectrum) if squared else np.maximum(spectrum, 0.0)
    total = float(spectrum.sum())
    if total <= 1e-15:
        return 0.0
    probabilities = spectrum / total
    probabilities = probabilities[probabilities > 1e-15]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


@dataclass
class _D16Geometry:
    anchor: Array
    useful_centers: Array
    useful_covariances: Array
    useful_directions: Array
    decoy_centers: Array
    decoy_covariances: Array
    decoy_amplitudes: Array
    geometry_id: int
    variant: str
    width_scale: float
    orthogonal_error: float


class D16LandscapeFamily:
    """Generation-2 semantics with full-rank, asymmetric 16D geometry."""

    def __init__(self, config: LandscapeV2Config) -> None:
        if config.dimension != D16_DIMENSION:
            raise ValueError(
                f"D16LandscapeFamily requires dimension=16, got {config.dimension}"
            )
        self.config = config
        rng = np.random.default_rng(config.family_seed)
        self.base_frame = random_orthogonal(rng, config.dimension)

    def _orientation(
        self, rng: np.random.Generator, variant: str
    ) -> tuple[Array, float]:
        dimension = self.config.dimension
        if variant in {"rotated", "unseen"}:
            rotation = random_orthogonal(rng, dimension)
        elif variant == "new_goals":
            rotation = _near_identity_rotation(rng, dimension, 0.28)
        else:
            # IID geometries share a family frame but vary in all 16 axes. A
            # non-trivial rotation scale prevents the aggregate training
            # distribution from collapsing to only three fixed coordinates.
            rotation = _near_identity_rotation(rng, dimension, 0.12)
        frame = rotation @ self.base_frame
        error = float(np.max(np.abs(frame.T @ frame - np.eye(dimension))))
        return frame, error

    @staticmethod
    def _useful_directions(frame: Array, count: int, jitter: Array) -> Array:
        dimension = frame.shape[0]
        directions: list[Array] = []
        for index in range(count):
            # The dominant axes differ, while two small asymmetric components
            # avoid a symmetric coordinate-axis construction.
            vector = (
                frame[:, index]
                + (0.14 + 0.02 * index) * frame[:, (index + 5) % dimension]
                - (0.07 + 0.01 * (index % 3)) * frame[:, (2 * index + 9) % dimension]
                + jitter[index]
            )
            directions.append(_unit(vector))
        return np.stack(directions)

    @staticmethod
    def _decoy_directions(
        frame: Array, useful_directions: Array, count: int, jitter: Array
    ) -> Array:
        dimension = frame.shape[0]
        useful_count = len(useful_directions)
        directions: list[Array] = []
        for index in range(count):
            primary = frame[:, (useful_count + index + 2) % dimension]
            misleading = useful_directions[index % useful_count]
            secondary = frame[:, (3 * index + 11) % dimension]
            vector = primary + 0.34 * misleading - 0.18 * secondary + jitter[index]
            directions.append(_unit(vector))
        return np.stack(directions)

    def _sample_geometry(
        self,
        rng: np.random.Generator,
        geometry_id: int,
        variant: str,
    ) -> _D16Geometry:
        cfg = self.config
        useful_count = cfg.useful_peaks
        decoy_count = cfg.decoys
        width_scale = 1.0
        if variant == "width_shift":
            width_scale = 1.45
        elif variant == "more_peaks":
            useful_count += 2
        elif variant == "more_decoys":
            decoy_count += 4
        elif variant == "unseen":
            useful_count += 1
            decoy_count += 2
            width_scale = float(rng.uniform(0.75, 1.35))
        elif variant not in {
            "iid",
            "rotated",
            "translated",
            "new_goals",
        }:
            raise ValueError(f"Unknown D16 landscape variant {variant!r}")

        frame, orthogonal_error = self._orientation(rng, variant)
        anchor_direction = _unit(rng.normal(size=cfg.dimension))
        anchor = anchor_direction * rng.uniform(0.0, cfg.train_translation_scale)
        if variant == "translated":
            anchor = anchor + 4.5 * frame[:, 12 % cfg.dimension]

        jitter_scale = 0.025 if variant != "unseen" else 0.08
        useful_jitter = rng.normal(
            scale=jitter_scale, size=(useful_count, cfg.dimension)
        )
        useful_directions = self._useful_directions(frame, useful_count, useful_jitter)
        radii = cfg.alternative_radius * rng.uniform(0.90, 1.11, size=useful_count)
        if variant == "unseen":
            radii *= rng.uniform(0.82, 1.22, size=useful_count)
        useful_centers = anchor[None, :] + radii[:, None] * useful_directions
        width_jitter = rng.uniform(0.90, 1.10, size=useful_count)
        useful_covariances = np.stack(
            [
                _anisotropic_covariance(
                    direction,
                    cfg.useful_radial_width * width_scale * width_jitter[index],
                    cfg.useful_lateral_width * width_scale / width_jitter[index],
                )
                for index, direction in enumerate(useful_directions)
            ]
        )

        decoy_jitter = rng.normal(
            scale=0.055 if variant != "unseen" else 0.11,
            size=(decoy_count, cfg.dimension),
        )
        decoy_directions = self._decoy_directions(
            frame, useful_directions, decoy_count, decoy_jitter
        )
        decoy_radii = cfg.decoy_radius * rng.uniform(0.89, 1.13, size=decoy_count)
        if variant == "unseen":
            decoy_radii *= rng.uniform(0.84, 1.20, size=decoy_count)
        decoy_centers = anchor[None, :] + decoy_radii[:, None] * decoy_directions
        decoy_covariances = np.stack(
            [
                _anisotropic_covariance(
                    direction,
                    cfg.decoy_radial_width * width_scale,
                    cfg.decoy_lateral_width * width_scale,
                )
                for direction in decoy_directions
            ]
        )
        decoy_amplitudes = rng.uniform(
            cfg.decoy_quality_min, cfg.decoy_quality_max, size=decoy_count
        )
        return _D16Geometry(
            anchor=anchor,
            useful_centers=useful_centers,
            useful_covariances=useful_covariances,
            useful_directions=useful_directions,
            decoy_centers=decoy_centers,
            decoy_covariances=decoy_covariances,
            decoy_amplitudes=decoy_amplitudes,
            geometry_id=geometry_id,
            variant=variant,
            width_scale=width_scale,
            orthogonal_error=orthogonal_error,
        )

    def _task_from_geometry(self, geometry: _D16Geometry, goal_index: int) -> V2Task:
        cfg = self.config
        centers: list[Array] = [geometry.anchor]
        covariances: list[Array] = [
            np.eye(cfg.dimension) * (cfg.anchor_width * geometry.width_scale) ** 2
        ]
        amplitudes: list[float] = [cfg.anchor_quality]
        kinds: list[str] = ["anchor"]

        for index, (center, covariance, direction) in enumerate(
            zip(
                geometry.useful_centers,
                geometry.useful_covariances,
                geometry.useful_directions,
            )
        ):
            is_target = index == goal_index
            centers.append(center)
            covariances.append(covariance)
            amplitudes.append(
                cfg.target_quality if is_target else cfg.secondary_quality
            )
            kinds.append("useful_target" if is_target else "useful_secondary")
            centers.append(center)
            covariances.append(
                _anisotropic_covariance(
                    direction,
                    cfg.corridor_radial_width * geometry.width_scale,
                    cfg.corridor_lateral_width * geometry.width_scale,
                )
            )
            amplitudes.append(
                cfg.corridor_target_quality
                if is_target
                else cfg.corridor_secondary_quality
            )
            kinds.append("corridor_target" if is_target else "corridor_secondary")

        for center, covariance, amplitude in zip(
            geometry.decoy_centers,
            geometry.decoy_covariances,
            geometry.decoy_amplitudes,
        ):
            centers.append(center)
            covariances.append(covariance)
            amplitudes.append(float(amplitude))
            kinds.append("decoy")

        covariance_array = np.stack(covariances)
        return V2Task(
            anchor=geometry.anchor.copy(),
            useful_centers=geometry.useful_centers.copy(),
            useful_covariances=geometry.useful_covariances.copy(),
            decoy_centers=geometry.decoy_centers.copy(),
            decoy_covariances=geometry.decoy_covariances.copy(),
            goal=geometry.useful_directions[goal_index].copy(),
            goal_index=goal_index,
            geometry_id=geometry.geometry_id,
            component_centers=np.stack(centers),
            component_precisions=np.linalg.inv(covariance_array),
            component_amplitudes=np.asarray(amplitudes, dtype=np.float64),
            component_kinds=tuple(kinds),
            config=cfg,
            variant=geometry.variant,
        )

    def sample_tasks(
        self,
        *,
        geometries: int,
        seed: int,
        variant: str = "iid",
        all_goals: bool = True,
    ) -> list[V2Task]:
        rng = np.random.default_rng(seed)
        tasks: list[V2Task] = []
        for geometry_id in range(geometries):
            geometry = self._sample_geometry(rng, geometry_id, variant)
            goal_indices: Sequence[int]
            if all_goals:
                goal_indices = range(len(geometry.useful_centers))
            else:
                goal_indices = [int(rng.integers(len(geometry.useful_centers)))]
            tasks.extend(
                self._task_from_geometry(geometry, goal_index)
                for goal_index in goal_indices
            )
        return tasks


class DimensionalLandscapeFamily(D16LandscapeFamily):
    """The same full-rank construction for controlled d=4/8/16/32/64 sweeps.

    The strict D16 public boundary remains unchanged; this diagnostic-only
    family reuses its mathematics at intermediate dimensions. Dimension two
    continues to use the established V2 family.
    """

    def __init__(self, config: LandscapeV2Config) -> None:
        if config.dimension not in {4, 8, 16, 32, 64}:
            raise ValueError(
                "DimensionalLandscapeFamily supports only diagnostic d=4/8/16/32/64"
            )
        self.config = config
        rng = np.random.default_rng(config.family_seed)
        self.base_frame = random_orthogonal(rng, config.dimension)


def geometry_diagnostics(tasks: Sequence[V2Task]) -> dict[str, float | list[float]]:
    """Audit rank, coordinate spread and anisotropic covariance structure."""
    unique: dict[int, V2Task] = {}
    for task in tasks:
        unique.setdefault(task.geometry_id, task)

    per_geometry_rank: list[float] = []
    per_geometry_algebraic_rank: list[float] = []
    all_directions: list[Array] = []
    useful_eigenvalues: list[Array] = []
    for task in unique.values():
        displacements = task.useful_centers - task.anchor[None, :]
        singular_values = np.linalg.svd(displacements, compute_uv=False)
        per_geometry_rank.append(effective_rank(singular_values))
        tolerance = max(displacements.shape) * singular_values[0] * np.finfo(float).eps
        per_geometry_algebraic_rank.append(float(np.sum(singular_values > tolerance)))
        all_directions.append(
            displacements / np.linalg.norm(displacements, axis=1, keepdims=True)
        )
        useful_eigenvalues.extend(np.linalg.eigvalsh(task.useful_covariances))

    direction_matrix = np.concatenate(all_directions, axis=0)
    aggregate_singular_values = np.linalg.svd(direction_matrix, compute_uv=False)
    coordinate_variance = np.var(direction_matrix, axis=0)
    coordinate_participation = effective_rank(coordinate_variance, squared=False)
    eigenvalues = np.asarray(useful_eigenvalues)
    return {
        "dimension": float(tasks[0].dimension if tasks else 0),
        "unique_geometries": float(len(unique)),
        "per_geometry_effective_rank_mean": float(np.mean(per_geometry_rank)),
        "per_geometry_effective_rank_min": float(np.min(per_geometry_rank)),
        "per_geometry_algebraic_rank_min": float(np.min(per_geometry_algebraic_rank)),
        "aggregate_useful_effective_rank": effective_rank(aggregate_singular_values),
        "aggregate_useful_singular_values": aggregate_singular_values.tolist(),
        "coordinate_variance_participation": coordinate_participation,
        "coordinate_variance": coordinate_variance.tolist(),
        "useful_covariance_eigenvalue_min": float(eigenvalues.min()),
        "useful_covariance_eigenvalue_median": float(np.median(eigenvalues)),
        "useful_covariance_eigenvalue_max": float(eigenvalues.max()),
        "useful_covariance_condition_mean": float(
            np.mean(eigenvalues[:, -1] / eigenvalues[:, 0])
        ),
        "not_effectively_2d": bool(
            np.min(per_geometry_rank) > 2.0
            and effective_rank(aggregate_singular_values) > 2.0
            and coordinate_participation > 2.0
        ),
    }
