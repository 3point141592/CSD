from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def cone_cosine(matrix: np.ndarray, target: np.ndarray) -> float:
    coefficient, _ = nnls(matrix, target)
    projection = matrix @ coefficient
    norm = np.linalg.norm(projection)
    return float(np.dot(target, projection) / norm) if norm > 1e-12 else 0.0


def test_target_inside_positive_cone() -> None:
    matrix = np.eye(2)
    target = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    assert abs(cone_cosine(matrix, target) - 1.0) < 1e-12


def test_target_outside_positive_cone() -> None:
    matrix = np.eye(2)
    target = np.asarray([-1.0, 0.0])
    assert cone_cosine(matrix, target) == 0.0
