"""Predeclared configuration for the isolated d=64 symmetry replication."""

from __future__ import annotations

from creativity_toy.config import STUDY_SEEDS

from .config_symmetry import SymmetryStudyConfig


def symmetry_d64_config() -> SymmetryStudyConfig:
    config = SymmetryStudyConfig(
        name="d64_landscape_attention_symmetry",
        dimension=64,
        # At d=64 this gives 148,161 parameters, versus 148,164 in the
        # CurrentAttention control. The three-parameter difference is smaller
        # than the d=32 match and avoids a capacity confound.
        invariant_component_hidden_dimension=981,
        invariant_component_token_dimension=64,
    )
    config.seeds = list(STUDY_SEEDS)
    return config


def smoke_symmetry_d64_config() -> SymmetryStudyConfig:
    config = symmetry_d64_config()
    config.name = "d64_landscape_attention_symmetry_smoke"
    config.training_geometries = 10
    config.starts_per_task = 4
    config.supervised_steps = 150
    config.batch_size = 128
    config.seeds = list(STUDY_SEEDS[:3])
    config.curve_steps = [0, 25, 50, 100, 150]
    config.augmentation.rotation_pool_size = 256
    return config
