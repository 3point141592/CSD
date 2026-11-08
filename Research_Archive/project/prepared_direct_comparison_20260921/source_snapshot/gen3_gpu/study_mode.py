"""Validation selection and paired 30-seed generator mode-preservation study."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from creativity_toy.analysis_utils import statistics
from creativity_toy.config_gen3 import Gen3StudyConfig
from creativity_toy.experiments.study_v2 import _make_starts
from creativity_toy.landscapes_v2 import V2LandscapeFamily, V2Task

from .black_box_env import BlackBoxLandscapeEnvironment
from .config_mode_preservation import ModePreservationConfig, ModeRewardConfig
from .config_rl import RLStudyConfig
from .evaluation_mode import (
    ModeEvaluationResult,
    evaluate_mode_policy,
    load_frozen_pairwise_router,
    router_integrity,
)
from .frozen_router_candidates import load_candidate_batch
from .mode_reward import quality_gated_directional_repulsion
from .runtime import RuntimeInfo
from .study import write_csv
from .study_rl import _datasets, _policy
from .task_batch import pack_tasks
from .training_mode_gpu import train_policy_mode_preservation

PRIMARY_VARIANTS = (
    "baseline_lambda_0",
    "gated_shared_selected",
    "ungated_shared_ablation",
    "gated_candidate_specific_ablation",
)


def frozen_rl_config() -> RLStudyConfig:
    """Load optimizer/reward settings from the completed 30-seed RL study."""

    payload = json.loads(
        Path("results/rl_adamw_main/config.json").read_text(encoding="utf-8")
    )
    saved = payload["rl"]["training"]
    rl = RLStudyConfig()
    rl.training.learning_rate = float(saved["learning_rate"])
    rl.training.weight_decay = float(saved["weight_decay"])
    rl.training.batch_size = int(saved["batch_size"])
    rl.training.max_candidate_evaluations = int(saved["max_candidate_evaluations"])
    rl.training.checkpoint_budgets = [
        int(value) for value in saved["checkpoint_budgets"]
    ]
    rl.training.direction_std_init = float(saved["direction_std_init"])
    rl.training.alpha_std_init = float(saved["alpha_std_init"])
    rl.training.entropy_coefficient = float(saved["entropy_coefficient"])
    rl.training.gradient_clip = float(saved["gradient_clip"])
    rl.training.reward.lambda_novelty = float(saved["reward"]["lambda_novelty"])
    rl.training.reward.quality_threshold = float(saved["reward"]["quality_threshold"])
    rl.training.reward.quality_gate_beta = float(saved["reward"]["quality_gate_beta"])
    rl.training.reward.diversity_scale = float(saved["reward"]["diversity_scale"])
    rl.training.reward.novelty_clip = float(saved["reward"]["novelty_clip"])
    return rl


def _nested(starts: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
    return [[np.asarray(value, dtype=np.float64)] for value in starts]


def _state_hash(policy) -> str:
    digest = hashlib.sha256()
    for key, value in policy.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_existing_policy(
    base: Gen3StudyConfig, rl: RLStudyConfig, seed: int, device: torch.device
):
    policy = _policy(base, rl, seed=seed, device=device)
    payload = torch.load(
        Path("results/rl_adamw_main/models") / f"seed_{seed}_rl_adamw.pt",
        map_location=device,
        weights_only=True,
    )
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    return policy


def _maximum_state_difference(left, right) -> float:
    values = []
    for key, tensor in left.state_dict().items():
        values.append(torch.max(torch.abs(tensor - right.state_dict()[key])).detach())
    return float(torch.max(torch.stack(values)).cpu())


def _save_policy(path: Path, policy, metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in policy.state_dict().items()
            },
            "architecture_parameter_count": policy.architecture_parameter_count,
            "policy_parameter_count": policy.policy_parameter_count,
            **metadata,
        },
        path,
    )


def _mode_metrics(
    evaluation: ModeEvaluationResult,
    mode: ModeRewardConfig,
    rl: RLStudyConfig,
) -> dict[str, float]:
    reward = quality_gated_directional_repulsion(
        evaluation.tensors["directions"],
        evaluation.tensors["quality"],
        quality_threshold=rl.training.reward.quality_threshold,
        quality_gate_beta=rl.training.reward.quality_gate_beta,
        quality_gated=mode.quality_gated,
        epsilon=mode.epsilon,
    )
    base = evaluation.tensors["base_reward"]
    return {
        "base_reward": float(torch.mean(base).cpu()),
        "mode_reward": float(torch.mean(reward.set_reward).cpu()),
        "weighted_mode_reward": float(
            mode.lambda_mode * torch.mean(reward.set_reward).cpu()
        ),
        "total_reward": float(
            torch.mean(base + mode.lambda_mode * reward.set_reward[:, None]).cpu()
        ),
        "mode_reward_to_base_ratio": float(
            (
                mode.lambda_mode
                * torch.mean(reward.set_reward)
                / torch.clamp(torch.mean(torch.abs(base)), min=1e-8)
            ).cpu()
        ),
    }


def _evaluate_validation(
    *,
    policy,
    environment,
    router,
    tasks: Sequence[V2Task],
    packed_tasks,
    starts: Sequence[np.ndarray],
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    mode: ModeRewardConfig,
    seed: int,
    rollouts: int,
) -> dict[str, float]:
    rows = []
    for rollout in range(rollouts):
        evaluation = evaluate_mode_policy(
            variant="validation",
            policy=policy,
            environment=environment,
            router=router,
            tasks=tasks,
            packed_tasks=packed_tasks,
            starts=_nested(starts),
            action_seed=seed + 50_000 + rollout,
            random_seed=seed + 50_500 + rollout,
            useful_threshold=base.base.landscape.useful_threshold,
        )
        rows.append({**evaluation.summary, **_mode_metrics(evaluation, mode, rl)})
    output: dict[str, float] = {}
    for key in rows[0]:
        output[key] = float(np.mean([float(row[key]) for row in rows]))
    return output


def _checkpoint_callback(
    *,
    policy,
    environment,
    router,
    tasks,
    packed_tasks,
    starts,
    base,
    rl,
    mode,
    seed: int,
):
    def callback(budget: int, step: int) -> dict[str, float]:
        evaluation = evaluate_mode_policy(
            variant="checkpoint",
            policy=policy,
            environment=environment,
            router=router,
            tasks=tasks,
            packed_tasks=packed_tasks,
            starts=_nested(starts),
            action_seed=seed + 45_000 + budget,
            random_seed=seed + 45_500 + budget,
            useful_threshold=base.base.landscape.useful_threshold,
        )
        return {
            "optimizer_step": float(step),
            **evaluation.summary,
            **_mode_metrics(evaluation, mode, rl),
        }

    return callback


def run_mode_sweep(
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    config: ModePreservationConfig,
    *,
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
) -> float:
    """Sweep only lambda_mode on training-family validation landscapes."""

    output.mkdir(parents=True, exist_ok=True)
    family = V2LandscapeFamily(base.base.landscape)
    seeds = base.base.seeds[: config.sweep.seeds]
    trial_rows: list[dict[str, object]] = []
    for lambda_mode in config.sweep.lambda_modes:
        for seed in seeds:
            train, train_starts, validation, validation_starts, _, _ = _datasets(
                family, base, rl, seed
            )
            packed_train = pack_tasks(train, device=device, dtype=torch.float32)
            packed_validation = pack_tasks(
                validation, device=device, dtype=torch.float32
            )
            training_starts = torch.as_tensor(
                np.stack(train_starts), dtype=torch.float32, device=device
            )
            policy = _policy(base, rl, seed=seed, device=device)
            router = load_frozen_pairwise_router(seed, device)
            environment = BlackBoxLandscapeEnvironment.from_configs(
                base.base.training, rl.training.reward
            )
            training = copy.deepcopy(rl.training)
            training.max_candidate_evaluations = config.sweep.candidate_evaluations
            training.checkpoint_budgets = [0, config.sweep.candidate_evaluations]
            mode = ModeRewardConfig(
                lambda_mode=lambda_mode,
                allocation="shared_joint",
                quality_gated=True,
            )
            result = train_policy_mode_preservation(
                policy,
                environment,
                packed_train,
                training_starts,
                training,
                mode,
                seed=seed + 40_021,
            )
            validation_metrics = _evaluate_validation(
                policy=policy,
                environment=environment,
                router=router,
                tasks=validation,
                packed_tasks=packed_validation,
                starts=validation_starts,
                base=base,
                rl=rl,
                mode=mode,
                seed=seed,
                rollouts=config.sweep.stochastic_validation_rollouts,
            )
            trial_rows.append(
                {
                    "seed": seed,
                    "lambda_mode": lambda_mode,
                    "allocation": mode.allocation,
                    "quality_gated": mode.quality_gated,
                    "candidate_evaluations": result.candidate_evaluations,
                    "q_equivalent_evaluations": result.q_equivalent_evaluations,
                    "optimizer_steps": result.optimizer_steps,
                    "training_seconds": result.elapsed_seconds,
                    "peak_allocated_bytes": result.peak_allocated_bytes,
                    **validation_metrics,
                }
            )
            print(
                f"[mode sweep] lambda={lambda_mode:g} seed={seed} "
                f"proposal={validation_metrics['proposal_recall']:.3f} "
                f"quality={validation_metrics['candidate_mean_quality']:.3f} "
                f"decoy={validation_metrics['candidate_decoy_rate']:.3f}",
                flush=True,
            )
    summary_rows: list[dict[str, object]] = []
    summary_metrics = [
        "candidate_mean_quality",
        "selected_quality",
        "proposal_recall",
        "routing_recall",
        "useful_selected",
        "useful_region_count",
        "candidate_decoy_rate",
        "selected_decoy_rate",
        "direction_cosine_distance",
        "alpha_sd",
        "base_reward",
        "mode_reward",
        "weighted_mode_reward",
        "total_reward",
        "mode_reward_to_base_ratio",
        "training_seconds",
    ]
    for lambda_mode in config.sweep.lambda_modes:
        rows = [row for row in trial_rows if float(row["lambda_mode"]) == lambda_mode]
        result: dict[str, object] = {"lambda_mode": lambda_mode, "n_seeds": len(rows)}
        for metric in summary_metrics:
            stat = statistics([float(row[metric]) for row in rows])
            result[metric] = stat["mean"]
            result[f"{metric}_sd"] = stat["sd"]
            result[f"{metric}_ci95"] = stat["ci95"]
        summary_rows.append(result)
    baseline = next(row for row in summary_rows if float(row["lambda_mode"]) == 0.0)
    eligible = [
        row
        for row in summary_rows
        if float(row["candidate_mean_quality"])
        >= float(baseline["candidate_mean_quality"]) - config.sweep.quality_tolerance
        and float(row["candidate_decoy_rate"])
        <= float(baseline["candidate_decoy_rate"]) + config.sweep.decoy_tolerance
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["proposal_recall"]),
            float(row["useful_region_count"]),
            float(row["useful_selected"]),
            float(row["candidate_mean_quality"]),
            -float(row["lambda_mode"]),
        ),
    )
    selected_lambda = float(selected["lambda_mode"])
    selection = {
        "selection_data": "held-out training-family validation landscapes",
        "test_set_used": False,
        "only_swept_parameter": "lambda_mode",
        "grid": config.sweep.lambda_modes,
        "selection_rule": (
            "eligible when candidate mean quality >= lambda0-0.02 and candidate "
            "decoy rate <= lambda0+0.02; maximize proposal recall, then useful-region "
            "count, useful selected, quality, and prefer lower lambda"
        ),
        "selected_lambda_mode": selected_lambda,
        "quality_gate_threshold": rl.training.reward.quality_threshold,
        "quality_gate_beta": rl.training.reward.quality_gate_beta,
        "shared_credit_assignment": (
            "batch-relative joint-action REINFORCE; a broadcast shared reward would "
            "cancel under the unchanged within-set advantage"
        ),
        "all_candidates": summary_rows,
        "config": config.to_dict(),
        "runtime": runtime.to_dict(),
    }
    write_csv(output / "lambda_sweep_trials.csv", trial_rows)
    write_csv(output / "lambda_sweep_summary.csv", summary_rows)
    (output / "MODE_LAMBDA_SELECTION.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    return selected_lambda


def load_selected_lambda(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_set_used") is not False:
        raise ValueError("mode lambda selection was not validation-only")
    return float(payload["selected_lambda_mode"])


def _attach(
    rows: list[dict[str, object]],
    *,
    seed: int,
    variant: str,
    lambda_mode: float,
    allocation: str,
    quality_gated: bool,
) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "variant": variant,
            "lambda_mode": lambda_mode,
            "allocation": allocation,
            "quality_gated": quality_gated,
            **row,
        }
        for row in rows
    ]


def run_mode_main(
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    config: ModePreservationConfig,
    *,
    selected_lambda: float,
    seeds: Sequence[int],
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    family = V2LandscapeFamily(base.base.landscape)
    payload = {
        "generation3": base.to_dict(),
        "rl_frozen": rl.to_dict(),
        "mode": config.to_dict(),
        "selected_lambda_mode": selected_lambda,
        "seeds": list(seeds),
        "runtime": runtime.to_dict(),
        "black_box_contract": "Q/refinement detached; region labels evaluation only",
        "router_contract": "saved pairwise router loaded once per seed and never fitted",
    }
    (output / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    primary_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    generalization_rows: list[dict[str, object]] = []
    integrity_rows: list[dict[str, object]] = []
    reproduction_rows: list[dict[str, object]] = []

    specs = [
        ("baseline_lambda_0", ModeRewardConfig(lambda_mode=0.0)),
        (
            "gated_shared_selected",
            ModeRewardConfig(
                lambda_mode=selected_lambda,
                allocation="shared_joint",
                quality_gated=True,
            ),
        ),
        (
            "ungated_shared_ablation",
            ModeRewardConfig(
                lambda_mode=selected_lambda,
                allocation="shared_joint",
                quality_gated=False,
            ),
        ),
        (
            "gated_candidate_specific_ablation",
            ModeRewardConfig(
                lambda_mode=selected_lambda,
                allocation="candidate_specific",
                quality_gated=True,
            ),
        ),
    ]

    for seed_index, seed in enumerate(seeds):
        print(f"[mode main {seed_index + 1}/{len(seeds)}] seed {seed}", flush=True)
        train, train_starts, validation, validation_starts, test, test_starts = (
            _datasets(family, base, rl, seed)
        )
        packed_train = pack_tasks(train, device=device, dtype=torch.float32)
        packed_validation = pack_tasks(validation, device=device, dtype=torch.float32)
        packed_test = pack_tasks(test, device=device, dtype=torch.float32)
        train_tensor = torch.as_tensor(
            np.stack(train_starts), dtype=torch.float32, device=device
        )
        environment = BlackBoxLandscapeEnvironment.from_configs(
            base.base.training, rl.training.reward
        )
        router = load_frozen_pairwise_router(seed, device)
        router_before = router.parameter_hash
        evaluated_policies: dict[str, object] = {}
        for variant, mode in specs:
            if "ablation" in variant and seed_index >= config.ablation_seeds:
                continue
            policy = _policy(base, rl, seed=seed, device=device)
            initial_hash = _state_hash(policy)
            callback = _checkpoint_callback(
                policy=policy,
                environment=environment,
                router=router,
                tasks=validation,
                packed_tasks=packed_validation,
                starts=validation_starts,
                base=base,
                rl=rl,
                mode=mode,
                seed=seed,
            )
            training = train_policy_mode_preservation(
                policy,
                environment,
                packed_train,
                train_tensor,
                rl.training,
                mode,
                seed=seed + 40_021,
                checkpoint_callback=callback,
            )
            policy.eval()
            evaluated_policies[variant] = policy
            for row in training.history:
                history_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "lambda_mode": mode.lambda_mode,
                        "allocation": mode.allocation,
                        "quality_gated": mode.quality_gated,
                        **row,
                    }
                )
            for row in training.checkpoint_metrics:
                checkpoint_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "lambda_mode": mode.lambda_mode,
                        **row,
                    }
                )
            _save_policy(
                model_dir / f"seed_{seed}_{variant}.pt",
                policy,
                {
                    "seed": seed,
                    "variant": variant,
                    "lambda_mode": mode.lambda_mode,
                    "allocation": mode.allocation,
                    "quality_gated": mode.quality_gated,
                },
            )
            evaluation = evaluate_mode_policy(
                variant=variant,
                policy=policy,
                environment=environment,
                router=router,
                tasks=test,
                packed_tasks=packed_test,
                starts=test_starts,
                action_seed=seed + 41_004,
                random_seed=seed + 41_104,
                useful_threshold=base.base.landscape.useful_threshold,
            )
            reward_metrics = _mode_metrics(evaluation, mode, rl)
            primary_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "lambda_mode": mode.lambda_mode,
                    "allocation": mode.allocation,
                    "quality_gated": mode.quality_gated,
                    "training_seconds": training.elapsed_seconds,
                    "candidate_outcomes_per_second": training.candidate_evaluations
                    / training.elapsed_seconds,
                    "candidate_evaluations": training.candidate_evaluations,
                    "q_equivalent_evaluations": training.q_equivalent_evaluations,
                    "optimizer_steps": training.optimizer_steps,
                    "peak_allocated_bytes": training.peak_allocated_bytes,
                    "peak_reserved_bytes": training.peak_reserved_bytes,
                    "architecture_parameter_count": policy.architecture_parameter_count,
                    "policy_parameter_count": policy.policy_parameter_count,
                    "direction_std": float(policy.direction_std().detach().cpu()),
                    "alpha_action_std": float(policy.alpha_std().detach().cpu()),
                    **evaluation.summary,
                    **reward_metrics,
                }
            )
            selector_rows.extend(
                _attach(
                    evaluation.selector_rows,
                    seed=seed,
                    variant=variant,
                    lambda_mode=mode.lambda_mode,
                    allocation=mode.allocation,
                    quality_gated=mode.quality_gated,
                )
            )
            set_rows.extend(
                _attach(
                    evaluation.set_rows,
                    seed=seed,
                    variant=variant,
                    lambda_mode=mode.lambda_mode,
                    allocation=mode.allocation,
                    quality_gated=mode.quality_gated,
                )
            )
            candidate_rows.extend(
                _attach(
                    evaluation.candidate_rows,
                    seed=seed,
                    variant=variant,
                    lambda_mode=mode.lambda_mode,
                    allocation=mode.allocation,
                    quality_gated=mode.quality_gated,
                )
            )
            final_hash = _state_hash(policy)
            integrity_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "initial_policy_hash": initial_hash,
                    "final_policy_hash": final_hash,
                    "architecture_parameter_count": policy.architecture_parameter_count,
                    "policy_parameter_count": policy.policy_parameter_count,
                    "candidate_count": policy.candidates,
                    "alpha_max": policy.alpha_max,
                    "candidate_evaluations": training.candidate_evaluations,
                    **router_integrity(router, router_before),
                }
            )
            if variant == "baseline_lambda_0":
                existing = _load_existing_policy(base, rl, seed, device)
                state_difference = _maximum_state_difference(policy, existing)
                cached = load_candidate_batch(
                    Path("results/router_useful_novelty_main/candidate_cache")
                    / f"seed_{seed}_rl_test.pt",
                    device,
                )
                reproduction_rows.append(
                    {
                        "seed": seed,
                        "maximum_policy_parameter_difference": state_difference,
                        "maximum_candidate_difference": float(
                            torch.max(
                                torch.abs(
                                    evaluation.tensors["candidates"] - cached.candidates
                                )
                            ).cpu()
                        ),
                        "maximum_refined_difference": float(
                            torch.max(
                                torch.abs(
                                    evaluation.tensors["refined"] - cached.refined
                                )
                            ).cpu()
                        ),
                        "maximum_quality_difference": float(
                            torch.max(
                                torch.abs(
                                    evaluation.tensors["quality"] - cached.quality
                                )
                            ).cpu()
                        ),
                    }
                )

        for variant in ["baseline_lambda_0", "gated_shared_selected"]:
            policy = evaluated_policies[variant]
            for variant_index, landscape_variant in enumerate(
                config.generalization_variants
            ):
                variant_tasks = family.sample_tasks(
                    geometries=base.base.test_geometries,
                    seed=seed + 30_000 + variant_index,
                    variant=landscape_variant,
                    all_goals=True,
                )
                variant_starts = _make_starts(
                    variant_tasks,
                    trials=base.base.trials_per_task,
                    seed=seed + 30_100 + variant_index,
                )
                packed_variant = pack_tasks(
                    variant_tasks, device=device, dtype=torch.float32
                )
                evaluation = evaluate_mode_policy(
                    variant=variant,
                    policy=policy,
                    environment=environment,
                    router=router,
                    tasks=variant_tasks,
                    packed_tasks=packed_variant,
                    starts=variant_starts,
                    action_seed=seed + 30_500 + 20 * variant_index + 1,
                    random_seed=seed + 30_700 + 20 * variant_index,
                    useful_threshold=base.base.landscape.useful_threshold,
                )
                generalization_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "lambda_mode": 0.0
                        if variant == "baseline_lambda_0"
                        else selected_lambda,
                        "landscape_variant": landscape_variant,
                        **evaluation.summary,
                    }
                )

    write_csv(output / "primary_per_seed.csv", primary_rows)
    write_csv(output / "selector_per_seed.csv", selector_rows)
    write_csv(output / "candidate_set_diagnostics.csv", set_rows)
    write_csv(output / "candidate_diagnostics.csv", candidate_rows)
    write_csv(output / "training_history.csv", history_rows)
    write_csv(output / "checkpoint_metrics.csv", checkpoint_rows)
    write_csv(output / "generalization_per_seed.csv", generalization_rows)
    write_csv(output / "integrity.csv", integrity_rows)
    write_csv(output / "baseline_reproduction.csv", reproduction_rows)
    aggregate = {
        "seed_count": len(seeds),
        "selected_lambda_mode": selected_lambda,
        "maximum_baseline_policy_difference": max(
            float(row["maximum_policy_parameter_difference"])
            for row in reproduction_rows
        ),
        "maximum_baseline_candidate_difference": max(
            float(row["maximum_candidate_difference"]) for row in reproduction_rows
        ),
        "maximum_router_parameter_difference": max(
            float(row["maximum_router_parameter_difference"]) for row in integrity_rows
        ),
        "black_box_training": True,
        "region_labels_used_for_training": False,
        "router_retrained": False,
        "test_set_used_for_lambda_selection": False,
    }
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
