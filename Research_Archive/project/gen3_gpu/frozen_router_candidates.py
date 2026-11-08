"""Load immutable trained generators and cache candidate tensors once per split."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from creativity_toy.config_gen3 import Gen3StudyConfig
from creativity_toy.landscapes_v2 import V2Task

from .config_rl import RLStudyConfig
from .config_router import RouterExperimentConfig
from .creative_policy_rl import PolicyEvaluationAdapter, StochasticCreativePolicy
from .critic import BatchedRidgeCritic, make_numpy_broad_candidates
from .operator import BatchedCreativeOperator
from .router_useful_novelty import FrozenCandidateBatch, normalized_candidate_novelty
from .study_rl import _load_es
from .task_batch import pack_tasks


@dataclass
class FrozenGenerator:
    kind: str
    operator: object
    theta: torch.Tensor
    quality_router: BatchedRidgeCritic
    parameter_snapshot: list[torch.Tensor]
    parameter_hash: str


def _hash_tensors(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _rl_policy(
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    seed: int,
    device: torch.device,
) -> StochasticCreativePolicy:
    policy = StochasticCreativePolicy(
        dimension=base.base.landscape.dimension,
        candidates=base.base.training.candidates,
        latent_dimension=base.base.training.latent_dimension,
        hidden_dimension=base.base.training.hidden_dimension,
        seed=seed + 10_020,
        device=device,
        dtype=torch.float32,
        alpha_max=rl.alpha_max,
        direction_std_init=rl.training.direction_std_init,
        alpha_std_init=rl.training.alpha_std_init,
    )
    payload = torch.load(
        Path("results/rl_adamw_main/models") / f"seed_{seed}_rl_adamw.pt",
        map_location=device,
        weights_only=True,
    )
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    return policy


def load_frozen_generator(
    kind: str,
    *,
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    seed: int,
    device: torch.device,
) -> FrozenGenerator:
    if kind == "rl":
        policy = _rl_policy(base, rl, seed, device)
        critic_data = np.load(
            Path("results/rl_adamw_main/models") / f"seed_{seed}_rl_adamw_critic.npz",
            allow_pickle=False,
        )
        critic = BatchedRidgeCritic(
            coefficients=torch.as_tensor(
                critic_data["coefficients"], dtype=torch.float32, device=device
            ),
            ridge=float(critic_data["ridge"]),
            temperature=float(critic_data["temperature"]),
        )
        tensors = [value.detach().clone() for value in policy.state_dict().values()]
        return FrozenGenerator(
            kind=kind,
            operator=policy,
            theta=torch.empty(0, dtype=torch.float32, device=device),
            quality_router=critic,
            parameter_snapshot=tensors,
            parameter_hash=_hash_tensors(tensors),
        )
    if kind == "es":
        operator, theta, critic = _load_es(base, seed, device)
        tensors = [theta.detach().clone(), operator.codes.detach().clone()]
        return FrozenGenerator(
            kind=kind,
            operator=operator,
            theta=theta,
            quality_router=critic,
            parameter_snapshot=tensors,
            parameter_hash=_hash_tensors(tensors),
        )
    raise ValueError(f"unknown generator kind {kind!r}")


def current_generator_tensors(generator: FrozenGenerator) -> list[torch.Tensor]:
    if generator.kind == "rl":
        policy = generator.operator
        assert isinstance(policy, StochasticCreativePolicy)
        return [value.detach() for value in policy.state_dict().values()]
    operator = generator.operator
    assert isinstance(operator, BatchedCreativeOperator)
    return [generator.theta.detach(), operator.codes.detach()]


def generator_integrity(generator: FrozenGenerator) -> dict[str, object]:
    current = current_generator_tensors(generator)
    maximum = max(
        float(torch.max(torch.abs(before - after)).cpu())
        for before, after in zip(generator.parameter_snapshot, current)
    )
    return {
        "generator": generator.kind,
        "parameter_hash_before": generator.parameter_hash,
        "parameter_hash_after": _hash_tensors(current),
        "maximum_parameter_difference": maximum,
        "unchanged": maximum == 0.0
        and generator.parameter_hash == _hash_tensors(current),
    }


def _flatten_starts(
    starts: Sequence[Sequence[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat, task_ids, trials = [], [], []
    for task_id, values in enumerate(starts):
        for trial, value in enumerate(values):
            flat.append(np.asarray(value, dtype=np.float64))
            task_ids.append(task_id)
            trials.append(trial)
    return np.stack(flat), np.asarray(task_ids), np.asarray(trials)


@torch.no_grad()
def generate_frozen_candidates(
    generator: FrozenGenerator,
    *,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    action_seed: int,
    base: Gen3StudyConfig,
    router_config: RouterExperimentConfig,
    device: torch.device,
    source: str,
) -> FrozenCandidateBatch:
    start_array, task_ids_array, trials_array = _flatten_starts(starts)
    packed = pack_tasks(tasks, device=device, dtype=torch.float32)
    task_ids = torch.as_tensor(task_ids_array, dtype=torch.long, device=device)
    trials = torch.as_tensor(trials_array, dtype=torch.long, device=device)
    batch_tasks = packed.index_select(task_ids)
    h = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    if generator.kind == "rl":
        policy = generator.operator
        assert isinstance(policy, StochasticCreativePolicy)
        adapter = PolicyEvaluationAdapter(policy, deterministic=False, seed=action_seed)
        output = adapter.forward(generator.theta, h, batch_tasks.goals)
    else:
        operator = generator.operator
        assert isinstance(operator, BatchedCreativeOperator)
        output = operator.forward(generator.theta, h, batch_tasks.goals)
    candidates = output.candidates.squeeze(0).detach()
    alphas = output.alphas.squeeze(0).detach()
    refined = batch_tasks.refine(
        candidates,
        steps=base.base.training.refinement_steps,
        rate=base.base.training.refinement_rate,
    ).detach()
    raw_quality = batch_tasks.quality(candidates).detach()
    quality = batch_tasks.quality(refined).detach()
    regions = batch_tasks.regions(refined).detach()
    novelty = normalized_candidate_novelty(
        candidates,
        diversity_scale=router_config.diversity_scale,
        novelty_clip=router_config.novelty_clip,
    ).detach()
    return FrozenCandidateBatch(
        starts=h,
        goals=batch_tasks.goals,
        candidates=candidates,
        refined=refined,
        raw_quality=raw_quality,
        quality=quality,
        regions=regions,
        alphas=alphas,
        task_ids=task_ids,
        trials=trials,
        n_useful=batch_tasks.n_useful,
        target_regions=batch_tasks.target_region,
        novelty=novelty,
        source=source,
    )


@torch.no_grad()
def generate_broad_training_candidates(
    *,
    tasks: Sequence[V2Task],
    starts: Sequence[Sequence[np.ndarray]],
    broad_seed: int,
    base: Gen3StudyConfig,
    router_config: RouterExperimentConfig,
    device: torch.device,
    source: str,
) -> FrozenCandidateBatch:
    start_array, task_ids_array, trials_array = _flatten_starts(starts)
    packed = pack_tasks(tasks, device=device, dtype=torch.float32)
    task_ids = torch.as_tensor(task_ids_array, dtype=torch.long, device=device)
    trials = torch.as_tensor(trials_array, dtype=torch.long, device=device)
    batch_tasks = packed.index_select(task_ids)
    h = torch.as_tensor(start_array, dtype=torch.float32, device=device)
    candidates = torch.as_tensor(
        make_numpy_broad_candidates(
            start_array,
            candidates=base.base.training.candidates,
            pure_divergence_radius=base.base.training.pure_divergence_radius,
            seed=broad_seed,
        ),
        dtype=torch.float32,
        device=device,
    )
    refined = batch_tasks.refine(
        candidates,
        steps=base.base.training.refinement_steps,
        rate=base.base.training.refinement_rate,
    ).detach()
    raw_quality = batch_tasks.quality(candidates).detach()
    quality = batch_tasks.quality(refined).detach()
    regions = batch_tasks.regions(refined).detach()
    novelty = normalized_candidate_novelty(
        candidates,
        diversity_scale=router_config.diversity_scale,
        novelty_clip=router_config.novelty_clip,
    ).detach()
    alphas = torch.linalg.vector_norm(candidates - h[:, None, :], dim=-1)
    return FrozenCandidateBatch(
        starts=h,
        goals=batch_tasks.goals,
        candidates=candidates,
        refined=refined,
        raw_quality=raw_quality,
        quality=quality,
        regions=regions,
        alphas=alphas,
        task_ids=task_ids,
        trials=trials,
        n_useful=batch_tasks.n_useful,
        target_regions=batch_tasks.target_region,
        novelty=novelty,
        source=source,
    )


def save_candidate_batch(path: Path, batch: FrozenCandidateBatch) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: getattr(batch, name).detach().cpu()
        for name in [
            "starts",
            "goals",
            "candidates",
            "refined",
            "raw_quality",
            "quality",
            "regions",
            "alphas",
            "task_ids",
            "trials",
            "n_useful",
            "target_regions",
            "novelty",
        ]
    }
    payload["source"] = batch.source
    torch.save(payload, path)
    return _hash_tensors([payload["candidates"]])


def load_candidate_batch(path: Path, device: torch.device) -> FrozenCandidateBatch:
    payload = torch.load(path, map_location=device, weights_only=True)
    return FrozenCandidateBatch(
        **{
            name: payload[name].to(device)
            for name in [
                "starts",
                "goals",
                "candidates",
                "refined",
                "raw_quality",
                "quality",
                "regions",
                "alphas",
                "task_ids",
                "trials",
                "n_useful",
                "target_regions",
                "novelty",
            ]
        },
        source=str(payload["source"]),
    )
