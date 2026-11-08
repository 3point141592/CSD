"""Synthetic multi-modal idea landscapes with known solution islands.

The training family has shared latent transformations between an anchor idea and
alternative ideas.  This is deliberate: a context-free linear operator cannot
infer independently random peak locations.  OOD variants break parts of that
structure to expose memorization and equivariance failures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class GaussianIslandLandscape:
    centers: Array
    widths: Array
    amplitudes: Array
    power: float = 2.0
    anchor_index: int = 0
    name: str = "gaussian"

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=np.float64)
        self.widths = np.asarray(self.widths, dtype=np.float64)
        self.amplitudes = np.asarray(self.amplitudes, dtype=np.float64)
        if self.centers.ndim != 2:
            raise ValueError("centers must have shape (islands, dimension)")
        if np.any(self.widths <= 0) or np.any(self.amplitudes <= 0):
            raise ValueError("widths and amplitudes must be positive")
        # A peak-relative scale lets quality thresholds transfer between tasks.
        center_logs = self._component_log_scores(self.centers)
        row_max = np.max(center_logs, axis=-1, keepdims=True)
        values = np.exp(row_max[..., 0]) * np.sum(
            np.exp(center_logs - row_max), axis=-1
        )
        self._normalizer = float(np.max(values))

    @property
    def dimension(self) -> int:
        return int(self.centers.shape[1])

    @property
    def n_islands(self) -> int:
        return int(self.centers.shape[0])

    @property
    def anchor(self) -> Array:
        return self.centers[self.anchor_index].copy()

    def _component_log_scores(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        diff = x[..., None, :] - self.centers
        distance = np.linalg.norm(diff, axis=-1)
        scaled = distance / self.widths
        return np.log(self.amplitudes) - 0.5 * np.power(scaled, self.power)

    def quality(self, x: Array) -> Array:
        """Smooth mixture quality, approximately in [0, 1]."""
        logs = self._component_log_scores(x)
        maximum = np.max(logs, axis=-1, keepdims=True)
        mixture = np.exp(maximum[..., 0]) * np.sum(np.exp(logs - maximum), axis=-1)
        return np.clip(mixture / self._normalizer, 0.0, 1.0)

    def component_quality(self, x: Array) -> Array:
        return np.exp(self._component_log_scores(x))

    def island(self, x: Array) -> Array:
        """Return the dominant island ID; quality gating belongs in metrics."""
        return np.argmax(self._component_log_scores(x), axis=-1)

    def gradient_log_quality(self, x: Array) -> Array:
        """Stable gradient of log(sum of generalized radial-basis peaks)."""
        x = np.asarray(x, dtype=np.float64)
        diff = x[..., None, :] - self.centers
        distance = np.linalg.norm(diff, axis=-1)
        logs = self._component_log_scores(x)
        maximum = np.max(logs, axis=-1, keepdims=True)
        weights = np.exp(logs - maximum)
        weights /= np.sum(weights, axis=-1, keepdims=True)

        safe_distance = np.maximum(distance, 1e-12)
        coefficient = (
            -0.5
            * self.power
            * np.power(safe_distance, self.power - 2.0)
            / np.power(self.widths, self.power)
        )
        component_grad = coefficient[..., None] * diff
        return np.sum(weights[..., None] * component_grad, axis=-2)

    def refine(
        self,
        x: Array,
        steps: int,
        rate: float,
        max_gradient_norm: float = 5.0,
    ) -> Array:
        """Fixed-budget gradient ascent on log quality."""
        result = np.asarray(x, dtype=np.float64).copy()
        # For Gaussian peaks grad(log Q) scales as 1 / width^2. This simple
        # preconditioner keeps the local optimizer budget comparable when the
        # generalization study changes peak widths.
        effective_rate = rate * float(np.median(self.widths) ** 2)
        for _ in range(steps):
            gradient = self.gradient_log_quality(result)
            norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
            gradient *= np.minimum(1.0, max_gradient_norm / np.maximum(norm, 1e-12))
            result += effective_rate * gradient
        return result


class StructuredLandscapeFamily:
    """Landscapes whose alternatives share transformations in a latent subspace."""

    def __init__(
        self,
        dimension: int,
        intrinsic_dimension: int,
        family_seed: int,
        max_peaks: int = 10,
        embedding_seed: int | None = None,
    ) -> None:
        self.dimension = dimension
        self.intrinsic_dimension = min(intrinsic_dimension, dimension)
        self.family_seed = family_seed
        self.max_peaks = max_peaks
        rng = np.random.default_rng(
            family_seed if embedding_seed is None else embedding_seed
        )
        if dimension == 2:
            self.embedding = np.eye(2)
            base_count = max(1, (max_peaks + 1) // 2)
            base_angles = np.linspace(0.0, 2.0 * np.pi, base_count, endpoint=False)
            midpoint_angles = base_angles + np.pi / base_count
            angles = np.concatenate([base_angles, midpoint_angles])[:max_peaks]
            self.transforms = []
            for angle in angles:
                c, s = np.cos(angle), np.sin(angle)
                self.transforms.append(np.array([[c, -s], [s, c]]))
        else:
            raw = rng.normal(size=(dimension, self.intrinsic_dimension))
            self.embedding, _ = np.linalg.qr(raw, mode="reduced")
            self.transforms = [np.eye(self.intrinsic_dimension)]
            # Shared transforms give the operator a reusable relation across tasks.
            while len(self.transforms) < max_peaks:
                candidate, _ = np.linalg.qr(
                    rng.normal(
                        size=(self.intrinsic_dimension, self.intrinsic_dimension)
                    )
                )
                # Proper rotations exclude reflections; separation avoids nearly unchanged maps.
                if np.linalg.det(candidate) < 0:
                    candidate[:, 0] *= -1
                separation = np.linalg.norm(
                    candidate - np.eye(self.intrinsic_dimension)
                )
                if separation > 1.5:
                    self.transforms.append(candidate)

    def sample(
        self,
        rng: np.random.Generator,
        n_peaks: int,
        radius: float,
        width: float,
        *,
        width_jitter: float = 0.08,
        power: float = 2.0,
        shift: Array | None = None,
        name: str = "iid",
    ) -> GaussianIslandLandscape:
        if n_peaks > len(self.transforms):
            raise ValueError("Requested more peaks than the family provides")
        latent = rng.normal(size=self.intrinsic_dimension)
        latent *= radius / np.linalg.norm(latent)
        centers = np.stack(
            [
                self.embedding @ (transform @ latent)
                for transform in self.transforms[:n_peaks]
            ]
        )
        if shift is not None:
            centers = centers + np.asarray(shift)
        widths = width * (1.0 + rng.uniform(-width_jitter, width_jitter, size=n_peaks))
        amplitudes = 0.92 + 0.08 * rng.random(n_peaks)
        amplitudes[0] = 1.0
        return GaussianIslandLandscape(
            centers=centers,
            widths=widths,
            amplitudes=amplitudes,
            power=power,
            name=name,
        )


def make_landscapes(
    *,
    count: int,
    seed: int,
    dimension: int,
    intrinsic_dimension: int,
    family_seed: int,
    max_family_peaks: int,
    n_peaks: int,
    radius: float,
    width: float,
    variant: str = "iid",
) -> list[GaussianIslandLandscape]:
    """Create a deterministic train/test family or an OOD variant."""
    rng = np.random.default_rng(seed)
    embedding_seed = family_seed
    actual_width = width
    width_jitter = 0.08
    power = 2.0
    shift = None

    if variant == "more_peaks":
        n_peaks = max(n_peaks, max_family_peaks)
    elif variant == "width_shift":
        actual_width = width * 1.45
        width_jitter = 0.25
    elif variant == "shifted":
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        shift = direction * (0.65 * radius)
    elif variant == "rotated":
        embedding_seed = family_seed + 99_991
    elif variant == "non_gaussian":
        power = 4.0
        actual_width = width * 1.2
    elif variant != "iid":
        raise ValueError(f"Unknown landscape variant {variant!r}")

    family = StructuredLandscapeFamily(
        dimension=dimension,
        intrinsic_dimension=intrinsic_dimension,
        family_seed=family_seed,
        max_peaks=max_family_peaks,
        embedding_seed=embedding_seed,
    )
    return [
        family.sample(
            rng,
            n_peaks=n_peaks,
            radius=radius,
            width=actual_width,
            width_jitter=width_jitter,
            power=power,
            shift=shift,
            name=variant,
        )
        for _ in range(count)
    ]


def starting_point(
    landscape: GaussianIslandLandscape,
    rng: np.random.Generator,
    noise: float,
    pre_steps: int = 5,
    pre_rate: float = 0.32,
) -> Array:
    """Create and locally optimize the deliberately obvious/anchor idea."""
    raw = landscape.anchor + rng.normal(scale=noise, size=landscape.dimension)
    return landscape.refine(raw, steps=pre_steps, rate=pre_rate)
