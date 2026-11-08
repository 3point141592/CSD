"""Asymmetric, goal-conditional landscapes for Generation 2.

Training code sees only smooth quality values and gradients. Region labels are
kept for post-hoc evaluation of useful alternatives and decoys.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config_v2 import LandscapeV2Config

Array = np.ndarray


def _anisotropic_covariance(
    direction: Array, radial_width: float, lateral_width: float
) -> Array:
    direction = np.asarray(direction, dtype=np.float64)
    unit = direction / np.linalg.norm(direction)
    identity = np.eye(unit.size)
    projector = np.outer(unit, unit)
    return radial_width**2 * projector + lateral_width**2 * (identity - projector)


@dataclass
class V2Task:
    anchor: Array
    useful_centers: Array
    useful_covariances: Array
    decoy_centers: Array
    decoy_covariances: Array
    goal: Array
    goal_index: int
    geometry_id: int
    component_centers: Array
    component_precisions: Array
    component_amplitudes: Array
    component_kinds: tuple[str, ...]
    config: LandscapeV2Config
    variant: str = "iid"

    @property
    def dimension(self) -> int:
        return int(self.anchor.size)

    @property
    def n_useful(self) -> int:
        return len(self.useful_centers)

    @property
    def n_decoys(self) -> int:
        return len(self.decoy_centers)

    @property
    def target_region(self) -> int:
        return 1 + self.goal_index

    def _log_component_scores(self, x: Array) -> tuple[Array, Array]:
        x = np.asarray(x, dtype=np.float64)
        difference = x[..., None, :] - self.component_centers
        quadratic = np.einsum(
            "...ni,nij,...nj->...n",
            difference,
            self.component_precisions,
            difference,
        )
        logs = np.log(self.component_amplitudes) - 0.5 * quadratic
        return logs, difference

    def component_scores(self, x: Array) -> Array:
        logs, _ = self._log_component_scores(x)
        return np.exp(np.clip(logs, -745.0, 0.0))

    def quality(self, x: Array) -> Array:
        """Smooth probabilistic union of anisotropic positive components."""
        scores = np.minimum(self.component_scores(x), 1.0 - 1e-12)
        log_product = np.sum(np.log1p(-scores), axis=-1)
        return -np.expm1(log_product)

    def gradient_log_quality(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        logs, difference = self._log_component_scores(x)
        scores = np.exp(np.clip(logs, -745.0, 0.0))
        scores = np.minimum(scores, 1.0 - 1e-12)
        grad_log_component = -np.einsum(
            "nij,...nj->...ni", self.component_precisions, difference
        )
        grad_score = scores[..., :, None] * grad_log_component
        log_product = np.sum(np.log1p(-scores), axis=-1)
        product = np.exp(log_product)
        quality = -np.expm1(log_product)
        grad_quality = product[..., None] * np.sum(
            grad_score / np.maximum(1.0 - scores, 1e-12)[..., :, None], axis=-2
        )
        result = grad_quality / np.maximum(quality, 1e-12)[..., None]

        # The log-mixture direction avoids vanishing gradients far from every component;
        # it preserves the modes and reported Q while allowing refinement to proceed.
        far = quality < 1e-10
        if np.any(far):
            maximum = np.max(logs, axis=-1, keepdims=True)
            weights = np.exp(logs - maximum)
            weights /= np.sum(weights, axis=-1, keepdims=True)
            mixture_gradient = np.sum(
                weights[..., :, None] * grad_log_component, axis=-2
            )
            result = np.where(far[..., None], mixture_gradient, result)
        return result

    def refine(
        self,
        x: Array,
        steps: int,
        rate: float,
        *,
        return_trajectory: bool = False,
        max_gradient_norm: float = 4.0,
    ) -> Array | tuple[Array, Array]:
        result = np.asarray(x, dtype=np.float64).copy()
        trajectory = [result.copy()]
        for _ in range(steps):
            gradient = self.gradient_log_quality(result)
            norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
            gradient *= np.minimum(1.0, max_gradient_norm / np.maximum(norm, 1e-12))
            result += rate * gradient
            if return_trajectory:
                trajectory.append(result.copy())
        if return_trajectory:
            return result, np.stack(trajectory, axis=-2)
        return result

    @staticmethod
    def _mahalanobis(x: Array, centers: Array, covariances: Array) -> Array:
        difference = np.asarray(x)[..., None, :] - centers
        precisions = np.linalg.inv(covariances)
        return np.sqrt(
            np.maximum(
                np.einsum("...ni,nij,...nj->...n", difference, precisions, difference),
                0.0,
            )
        )

    def region(self, x: Array) -> Array:
        """0=anchor, 1..U=useful, U+1..=decoy, -1=irrelevant."""
        points = np.asarray(x, dtype=np.float64)
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, self.dimension)
        anchor_cov = np.eye(self.dimension) * self.config.anchor_width**2
        anchor_distance = self._mahalanobis(
            flat, self.anchor[None, :], anchor_cov[None, :, :]
        )[:, 0]
        useful_distance = self._mahalanobis(
            flat, self.useful_centers, self.useful_covariances
        )
        decoy_distance = self._mahalanobis(
            flat, self.decoy_centers, self.decoy_covariances
        )
        combined = np.concatenate(
            [anchor_distance[:, None], useful_distance, decoy_distance], axis=1
        )
        nearest = np.argmin(combined, axis=1)
        minimum = np.min(combined, axis=1)
        nearest = np.where(
            minimum <= self.config.region_mahalanobis_threshold, nearest, -1
        )
        return nearest.reshape(original_shape)

    def is_useful_region(self, region: Array) -> Array:
        return (region >= 1) & (region <= self.n_useful)

    def is_decoy_region(self, region: Array) -> Array:
        return region > self.n_useful


@dataclass
class _Geometry:
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


class V2LandscapeFamily:
    def __init__(self, config: LandscapeV2Config) -> None:
        self.config = config
        rng = np.random.default_rng(config.family_seed)
        if config.dimension == 2:
            self.plane = np.eye(2)
        else:
            raw = rng.normal(size=(config.dimension, 2))
            self.plane, _ = np.linalg.qr(raw, mode="reduced")

    def _direction(self, angle: float, plane: Array | None = None) -> Array:
        active_plane = self.plane if plane is None else plane
        return active_plane @ np.array([np.cos(angle), np.sin(angle)])

    def _sample_geometry(
        self,
        rng: np.random.Generator,
        geometry_id: int,
        variant: str,
    ) -> _Geometry:
        cfg = self.config
        useful_count = cfg.useful_peaks
        decoy_count = cfg.decoys
        width_scale = 1.0
        plane = self.plane
        rotation = rng.uniform(-cfg.train_rotation_range, cfg.train_rotation_range)
        translation_scale = cfg.train_translation_scale

        if variant == "rotated":
            rotation += np.pi / 2.0
            if cfg.dimension > 2:
                raw = rng.normal(size=(cfg.dimension, 2))
                plane, _ = np.linalg.qr(raw, mode="reduced")
        elif variant == "translated":
            translation_scale = 0.25
        elif variant == "width_shift":
            width_scale = 1.45
        elif variant == "more_peaks":
            useful_count += 2
        elif variant == "more_decoys":
            decoy_count += 4
        elif variant == "new_goals":
            rotation += 0.58
        elif variant == "unseen":
            useful_count = cfg.useful_peaks + 1
            decoy_count = cfg.decoys + 2
            width_scale = rng.uniform(0.75, 1.35)
            rotation = rng.uniform(-np.pi, np.pi)
        elif variant != "iid":
            raise ValueError(f"Unknown V2 landscape variant {variant!r}")

        anchor = rng.uniform(-translation_scale, translation_scale, size=cfg.dimension)
        if variant == "translated":
            anchor += self._direction(0.35, plane) * 4.5

        if useful_count == 3:
            base_angles = np.deg2rad(np.array([28.0, 154.0, 276.0]))
        else:
            base_angles = np.linspace(
                0.15, 2.0 * np.pi + 0.15, useful_count, endpoint=False
            )
        irregularity = rng.normal(
            scale=0.07 if variant != "unseen" else 0.16, size=useful_count
        )
        useful_angles = rotation + base_angles + irregularity
        useful_directions = np.stack(
            [self._direction(angle, plane) for angle in useful_angles]
        )
        useful_radii = cfg.alternative_radius * rng.uniform(
            0.92, 1.08, size=useful_count
        )
        if variant == "unseen":
            useful_radii *= rng.uniform(0.8, 1.25, size=useful_count)
        useful_centers = anchor[None, :] + useful_radii[:, None] * useful_directions
        useful_covariances = np.stack(
            [
                _anisotropic_covariance(
                    direction,
                    cfg.useful_radial_width * width_scale,
                    cfg.useful_lateral_width * width_scale,
                )
                for direction in useful_directions
            ]
        )

        decoy_angles = (
            rotation
            + np.linspace(-0.35, 2.0 * np.pi - 0.35, decoy_count, endpoint=False)
            + rng.normal(scale=0.12, size=decoy_count)
        )
        decoy_directions = np.stack(
            [self._direction(angle, plane) for angle in decoy_angles]
        )
        decoy_radii = cfg.decoy_radius * rng.uniform(0.90, 1.12, size=decoy_count)
        if variant == "unseen":
            decoy_radii *= rng.uniform(0.85, 1.2, size=decoy_count)
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
        return _Geometry(
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
        )

    def _task_from_geometry(self, geometry: _Geometry, goal_index: int) -> V2Task:
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
            target = index == goal_index
            centers.append(center)
            covariances.append(covariance)
            amplitudes.append(cfg.target_quality if target else cfg.secondary_quality)
            kinds.append("useful_target" if target else "useful_secondary")

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
                if target
                else cfg.corridor_secondary_quality
            )
            kinds.append("corridor_target" if target else "corridor_secondary")

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


def starting_state_v2(
    task: V2Task,
    rng: np.random.Generator,
    noise: float = 0.16,
    pre_steps: int = 5,
    pre_rate: float = 0.20,
) -> Array:
    state = task.anchor + rng.normal(scale=noise, size=task.dimension)
    return task.refine(state, steps=pre_steps, rate=pre_rate)  # type: ignore[return-value]
