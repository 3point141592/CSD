"""Hyperparameter selection and 30-seed black-box RL study orchestration."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from creativity_toy.analysis_utils import classify_candidate, statistics
from creativity_toy.config_gen3 import Gen3StudyConfig
from creativity_toy.experiments.study_gen3 import _enrich_alpha_summary
from creativity_toy.experiments.study_v2 import _make_starts, _training_starts
from creativity_toy.landscapes_v2 import V2LandscapeFamily, V2Task

from .black_box_env import BlackBoxLandscapeEnvironment
from .config_rl import RLStudyConfig
from .creative_policy_rl import PolicyEvaluationAdapter, StochasticCreativePolicy
from .critic import BatchedRidgeCritic, fit_critic_batched
from .evaluation import evaluate_batched
from .operator import BatchedCreativeOperator
from .runtime import RuntimeInfo, synchronize
from .study import write_csv
from .task_batch import pack_tasks
from .training_rl_gpu import portfolio_snapshot, train_policy_reinforce

RL_MODES = ("rl_adamw_stochastic", "rl_adamw_deterministic")
MAIN_METRICS = [
    "mean_quality",
    "selected_quality",
    "oracle_best_quality",
    "euclidean_diversity",
    "coverage",
    "collapse_rate",
    "useful_exploration_rate",
    "decoy_rate",
    "anchor_escape_rate",
    "target_hit_rate",
    "selected_target_rate",
    "critic_regret",
    "mean_alpha",
    "alpha_sd",
]


def _policy(
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    *,
    seed: int,
    device: torch.device,
) -> StochasticCreativePolicy:
    return StochasticCreativePolicy(
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


def _datasets(
    family: V2LandscapeFamily,
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    seed: int,
) -> tuple[
    list[V2Task],
    list[np.ndarray],
    list[V2Task],
    list[np.ndarray],
    list[V2Task],
    list[list[np.ndarray]],
]:
    train = family.sample_tasks(
        geometries=base.base.train_geometries,
        seed=seed + 10_000,
        variant="iid",
        all_goals=True,
    )
    train_starts = _training_starts(_make_starts(train, trials=1, seed=seed + 10_001))
    validation = family.sample_tasks(
        geometries=rl.validation_geometries,
        seed=seed + 15_000,
        variant="iid",
        all_goals=True,
    )
    validation_starts = _training_starts(
        _make_starts(validation, trials=1, seed=seed + 15_001)
    )
    test = family.sample_tasks(
        geometries=base.base.test_geometries,
        seed=seed + 20_000,
        variant="iid",
        all_goals=True,
    )
    test_starts = _make_starts(
        test,
        trials=base.base.trials_per_task,
        seed=seed + 20_001,
    )
    return train, train_starts, validation, validation_starts, test, test_starts


def _snapshot_callback(
    *,
    policy: StochasticCreativePolicy,
    environment: BlackBoxLandscapeEnvironment,
    tasks: list[V2Task],
    packed_tasks,
    starts: torch.Tensor,
    base: Gen3StudyConfig,
    seed: int,
):
    def callback(budget: int, step: int) -> dict[str, float]:
        output: dict[str, float] = {}
        for deterministic, label in [(False, "stochastic"), (True, "deterministic")]:
            metrics = portfolio_snapshot(
                policy,
                environment,
                tasks,
                packed_tasks,
                starts,
                deterministic=deterministic,
                seed=seed + 45_000 + budget + (1 if deterministic else 0),
                useful_threshold=base.base.landscape.useful_threshold,
                high_quality_threshold=base.base.landscape.high_quality_threshold,
            )
            output.update({f"{label}_{key}": value for key, value in metrics.items()})
        direction_entropy = policy.dimension * (
            policy.direction_log_std.detach() + 0.5 * (1.0 + np.log(2.0 * np.pi))
        )
        alpha_entropy = policy.alpha_log_std.detach() + 0.5 * (
            1.0 + np.log(2.0 * np.pi)
        )
        output.update(
            {
                "optimizer_step": float(step),
                "policy_entropy": float((direction_entropy + alpha_entropy).cpu()),
                "direction_std": float(policy.direction_std().detach().cpu()),
                "alpha_action_std": float(policy.alpha_std().detach().cpu()),
            }
        )
        return output

    return callback


def _validation_score(
    policy: StochasticCreativePolicy,
    environment: BlackBoxLandscapeEnvironment,
    tasks: list[V2Task],
    packed_tasks,
    starts: torch.Tensor,
    base: Gen3StudyConfig,
    *,
    seed: int,
    stochastic_rollouts: int,
) -> dict[str, float]:
    rewards: list[float] = []
    qualities: list[float] = []
    for rollout in range(stochastic_rollouts):
        snapshot = portfolio_snapshot(
            policy,
            environment,
            tasks,
            packed_tasks,
            starts,
            deterministic=False,
            seed=seed + 50_000 + rollout,
            useful_threshold=base.base.landscape.useful_threshold,
            high_quality_threshold=base.base.landscape.high_quality_threshold,
        )
        rewards.append(snapshot["mean_reward"])
        qualities.append(snapshot["mean_quality"])
    deterministic = portfolio_snapshot(
        policy,
        environment,
        tasks,
        packed_tasks,
        starts,
        deterministic=True,
        seed=seed + 50_999,
        useful_threshold=base.base.landscape.useful_threshold,
        high_quality_threshold=base.base.landscape.high_quality_threshold,
    )
    return {
        "stochastic_validation_reward": float(np.mean(rewards)),
        "stochastic_validation_reward_sd": float(np.std(rewards, ddof=0)),
        "stochastic_validation_quality": float(np.mean(qualities)),
        "deterministic_validation_reward": deterministic["mean_reward"],
        "deterministic_validation_quality": deterministic["mean_quality"],
        "deterministic_validation_coverage": deterministic["coverage"],
        "deterministic_validation_useful": deterministic["useful_exploration_rate"],
        "deterministic_validation_decoy": deterministic["decoy_rate"],
    }


def run_hyperparameter_sweep(
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    *,
    output: Path,
    device: torch.device,
) -> tuple[float, float, list[dict[str, object]]]:
    """Select only on held-out training-family validation reward, never test."""
    output.mkdir(parents=True, exist_ok=True)
    family = V2LandscapeFamily(base.base.landscape)
    trial_rows: list[dict[str, object]] = []
    sweep_seeds = base.base.seeds[: rl.sweep.seeds]
    for learning_rate in rl.sweep.learning_rates:
        for weight_decay in rl.sweep.weight_decays:
            for seed in sweep_seeds:
                train, train_starts, validation, validation_starts, _, _ = _datasets(
                    family, base, rl, seed
                )
                packed_train = pack_tasks(train, device=device, dtype=torch.float32)
                packed_validation = pack_tasks(
                    validation, device=device, dtype=torch.float32
                )
                train_tensor = torch.as_tensor(
                    np.stack(train_starts), dtype=torch.float32, device=device
                )
                validation_tensor = torch.as_tensor(
                    np.stack(validation_starts), dtype=torch.float32, device=device
                )
                training = copy.deepcopy(rl.training)
                training.learning_rate = learning_rate
                training.weight_decay = weight_decay
                training.max_candidate_evaluations = rl.sweep.candidate_evaluations
                training.checkpoint_budgets = [0, rl.sweep.candidate_evaluations]
                policy = _policy(base, rl, seed=seed, device=device)
                environment = BlackBoxLandscapeEnvironment.from_configs(
                    base.base.training, training.reward
                )
                result = train_policy_reinforce(
                    policy,
                    environment,
                    packed_train,
                    train_tensor,
                    training,
                    seed=seed + 40_021,
                )
                score = _validation_score(
                    policy,
                    environment,
                    validation,
                    packed_validation,
                    validation_tensor,
                    base,
                    seed=seed,
                    stochastic_rollouts=rl.sweep.stochastic_validation_rollouts,
                )
                trial_rows.append(
                    {
                        "seed": seed,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "training_candidate_evaluations": result.candidate_evaluations,
                        "optimizer_steps": result.optimizer_steps,
                        "training_seconds": result.elapsed_seconds,
                        **score,
                    }
                )
                print(
                    f"[RL sweep] lr={learning_rate:g} wd={weight_decay:g} "
                    f"seed={seed} val_reward={score['stochastic_validation_reward']:.4f}",
                    flush=True,
                )
    summary: list[dict[str, object]] = []
    for learning_rate in rl.sweep.learning_rates:
        for weight_decay in rl.sweep.weight_decays:
            rows = [
                row
                for row in trial_rows
                if row["learning_rate"] == learning_rate
                and row["weight_decay"] == weight_decay
            ]
            reward_stats = statistics(
                [float(row["stochastic_validation_reward"]) for row in rows]
            )
            summary.append(
                {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "validation_reward_mean": reward_stats["mean"],
                    "validation_reward_sd": reward_stats["sd"],
                    "validation_reward_ci95": reward_stats["ci95"],
                    "validation_quality_mean": float(
                        np.mean(
                            [
                                float(row["stochastic_validation_quality"])
                                for row in rows
                            ]
                        )
                    ),
                    "deterministic_coverage_mean": float(
                        np.mean(
                            [
                                float(row["deterministic_validation_coverage"])
                                for row in rows
                            ]
                        )
                    ),
                    "n_seeds": len(rows),
                }
            )
    selected = max(
        summary,
        key=lambda row: (
            float(row["validation_reward_mean"]),
            float(row["validation_quality_mean"]),
        ),
    )
    write_csv(output / "hyperparameter_trials.csv", trial_rows)
    write_csv(output / "hyperparameter_summary.csv", summary)
    selection = {
        "selection_data": "held-out training-family validation landscapes",
        "selection_metric": "mean stochastic useful-novelty reward",
        "test_set_used": False,
        "selected_learning_rate": selected["learning_rate"],
        "selected_weight_decay": selected["weight_decay"],
        "all_candidates": summary,
    }
    (output / "HYPERPARAMETER_SELECTION.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    return (
        float(selected["learning_rate"]),
        float(selected["weight_decay"]),
        trial_rows,
    )


def _load_es(
    base: Gen3StudyConfig,
    seed: int,
    device: torch.device,
) -> tuple[BatchedCreativeOperator, torch.Tensor, BatchedRidgeCritic]:
    operator = BatchedCreativeOperator(
        dimension=base.base.landscape.dimension,
        candidates=base.base.training.candidates,
        latent_dimension=base.base.training.latent_dimension,
        hidden_dimension=base.base.training.hidden_dimension,
        seed=seed + 10_020,
        variant="gen3_contextual_linear_alpha",
        device=device,
        dtype=torch.float32,
        alpha_max=base.alpha_max,
        alpha_hidden_dimension=base.alpha_hidden_dimension,
    )
    root = Path("results/gen3_gpu/models")
    model = np.load(
        root / f"seed_{seed}_gen3_contextual_linear_alpha.npz",
        allow_pickle=False,
    )
    critic_data = np.load(
        root / f"seed_{seed}_gen3_contextual_linear_alpha_critic.npz",
        allow_pickle=False,
    )
    theta = operator.parameters_from_numpy(model["parameters"])
    critic = BatchedRidgeCritic(
        coefficients=torch.as_tensor(
            critic_data["coefficients"], dtype=torch.float32, device=device
        ),
        ridge=float(critic_data["ridge"]),
        temperature=float(critic_data["temperature"]),
    )
    return operator, theta, critic


def _save_policy(
    path: Path,
    policy: StochasticCreativePolicy,
    *,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in policy.state_dict().items()
            },
            "seed": seed,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "architecture_parameter_count": policy.architecture_parameter_count,
            "policy_parameter_count": policy.policy_parameter_count,
            "alpha_max": policy.alpha_max,
        },
        path,
    )


def _aggregate(primary: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for mode in RL_MODES:
        rows = [row for row in primary if row["evaluation_mode"] == mode]
        result: dict[str, object] = {"evaluation_mode": mode}
        for metric in MAIN_METRICS + [
            "training_seconds",
            "candidate_evaluations",
            "q_equivalent_evaluations",
            "optimizer_steps",
            "direction_std",
            "alpha_action_std",
            "policy_entropy",
            "training_mean_reward",
            "training_reward_std",
        ]:
            stats = statistics([float(row[metric]) for row in rows])
            for key, value in stats.items():
                result[f"{metric}_{key}"] = value
        output.append(result)
    return output


def _plot_learning_curves(rows: list[dict[str, object]], path: Path) -> None:
    metrics = [
        ("stochastic_mean_reward", "reward"),
        ("stochastic_mean_quality", "quality"),
        ("stochastic_coverage", "coverage"),
        ("stochastic_useful_exploration_rate", "useful exploration"),
        ("stochastic_decoy_rate", "decoy rate"),
        ("stochastic_alpha_mean", "alpha mean"),
        ("stochastic_alpha_sd", "alpha std"),
        ("policy_entropy", "policy entropy"),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    budgets = sorted({int(float(row["requested_budget"])) for row in rows})
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        means, lows, highs = [], [], []
        for budget in budgets:
            values = [
                float(row[metric])
                for row in rows
                if int(float(row["requested_budget"])) == budget
            ]
            stats = statistics(values)
            means.append(stats["mean"])
            lows.append(stats["mean"] - stats["ci95"])
            highs.append(stats["mean"] + stats["ci95"])
        axis.plot(budgets, means, marker="o", label="GPU RL stochastic")
        axis.fill_between(budgets, lows, highs, alpha=0.2)
        axis.set_title(label)
        axis.set_xlabel("training candidate/Q outcomes")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_training_diagnostics(rows: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    x = [float(row["candidate_evaluations"]) for row in rows]
    series = [
        ("mean_reward", "training reward"),
        ("policy_entropy", "Gaussian entropy"),
        ("gradient_norm_before_clip", "gradient norm before clip"),
        ("candidate_direction_cosine", "direction cosine"),
        ("candidate_spatial_diversity", "spatial diversity"),
        ("within_set_alpha_sd", "within-set alpha SD"),
    ]
    for axis, (key, title) in zip(axes.ravel(), series):
        axis.plot(x, [float(row[key]) for row in rows])
        axis.set_title(title)
        axis.set_xlabel("training candidate/Q outcomes")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def run_rl_study(
    base: Gen3StudyConfig,
    rl: RLStudyConfig,
    *,
    output: Path,
    device: torch.device,
    runtime: RuntimeInfo,
    seeds: Sequence[int],
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    payload = {
        "generation3": base.to_dict(),
        "rl": rl.to_dict(),
        "seeds": list(seeds),
        "runtime": runtime.to_dict(),
        "black_box_contract": (
            "policy sees detached scalar reward only; refinement and Q execute in no_grad"
        ),
    }
    (output / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    family = V2LandscapeFamily(base.base.landscape)
    primary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    generalization_rows: list[dict[str, object]] = []

    for seed_index, seed in enumerate(seeds):
        print(f"[RL main {seed_index + 1}/{len(seeds)}] seed {seed}", flush=True)
        (
            train,
            train_starts_list,
            validation,
            validation_starts_list,
            test,
            test_starts,
        ) = _datasets(family, base, rl, seed)
        packed_train = pack_tasks(train, device=device, dtype=torch.float32)
        packed_validation = pack_tasks(validation, device=device, dtype=torch.float32)
        packed_test = pack_tasks(test, device=device, dtype=torch.float32)
        train_starts = torch.as_tensor(
            np.stack(train_starts_list), dtype=torch.float32, device=device
        )
        validation_starts = torch.as_tensor(
            np.stack(validation_starts_list), dtype=torch.float32, device=device
        )
        policy = _policy(base, rl, seed=seed, device=device)
        environment = BlackBoxLandscapeEnvironment.from_configs(
            base.base.training, rl.training.reward
        )
        callback = _snapshot_callback(
            policy=policy,
            environment=environment,
            tasks=validation,
            packed_tasks=packed_validation,
            starts=validation_starts,
            base=base,
            seed=seed,
        )
        training = train_policy_reinforce(
            policy,
            environment,
            packed_train,
            train_starts,
            rl.training,
            seed=seed + 40_021,
            checkpoint_callback=callback,
        )
        for row in training.history:
            history_rows.append({"seed": seed, **row})
        for row in training.checkpoint_metrics:
            checkpoint_rows.append({"seed": seed, **row})
        _save_policy(
            model_dir / f"seed_{seed}_rl_adamw.pt",
            policy,
            seed=seed,
            learning_rate=rl.training.learning_rate,
            weight_decay=rl.training.weight_decay,
        )

        stochastic_train_adapter = PolicyEvaluationAdapter(
            policy, deterministic=False, seed=seed + 40_022
        )
        dummy_theta = torch.empty(0, device=device)
        synchronize(device)
        critic_start = perf_counter()
        critic, _ = fit_critic_batched(
            stochastic_train_adapter,  # type: ignore[arg-type]
            dummy_theta,
            packed_train,
            train_starts,
            base.base.training,
            seed=seed + 40_023,
        )
        synchronize(device)
        critic_seconds = perf_counter() - critic_start
        np.savez_compressed(
            model_dir / f"seed_{seed}_rl_adamw_critic.npz",
            coefficients=critic.coefficients.detach().cpu().to(torch.float64).numpy(),
            ridge=critic.ridge,
            temperature=critic.temperature,
        )

        final_training = training.history[-1]
        for mode_index, deterministic in enumerate([False, True]):
            mode = RL_MODES[mode_index]
            adapter = PolicyEvaluationAdapter(
                policy,
                deterministic=deterministic,
                seed=seed + 41_004 + mode_index,
            )
            evaluation = evaluate_batched(
                variant=mode,
                operator=adapter,  # type: ignore[arg-type]
                theta=dummy_theta,
                critic=critic,
                tasks=test,
                packed_tasks=packed_test,
                starts=test_starts,
                config=base.base.training,
                seed=seed + 41_104 + mode_index,
                useful_threshold=base.base.landscape.useful_threshold,
                high_quality_threshold=base.base.landscape.high_quality_threshold,
            )
            for row in evaluation.records:
                row["seed"] = seed
                row["evaluation_mode"] = mode
                row["category"] = classify_candidate(
                    row, base.base.landscape.useful_threshold
                )
            for row in evaluation.candidate_sets:
                row["seed"] = seed
                row["evaluation_mode"] = mode
            for row in evaluation.selectors:
                row["seed"] = seed
                row["evaluation_mode"] = mode
            _enrich_alpha_summary(evaluation.summary, evaluation.records)
            entropy = (
                policy.dimension
                * (
                    policy.direction_log_std.detach()
                    + 0.5 * (1.0 + np.log(2.0 * np.pi))
                )
                + policy.alpha_log_std.detach()
                + 0.5 * (1.0 + np.log(2.0 * np.pi))
            )
            primary_rows.append(
                {
                    "seed": seed,
                    "training": "gpu_rl_adamw",
                    "feedback": "detached_scalar_useful_novelty_reward",
                    "evaluation_mode": mode,
                    "training_seconds": training.elapsed_seconds,
                    "critic_fit_seconds": critic_seconds,
                    "candidate_evaluations": training.candidate_evaluations,
                    "q_equivalent_evaluations": training.q_equivalent_evaluations,
                    "critic_fit_candidate_evaluations": len(train)
                    * (
                        base.base.training.candidates
                        + max(2 * base.base.training.candidates, 8)
                    ),
                    "optimizer_steps": training.optimizer_steps,
                    "architecture_parameter_count": policy.architecture_parameter_count,
                    "policy_parameter_count": policy.policy_parameter_count,
                    "direction_std": float(policy.direction_std().detach().cpu()),
                    "alpha_action_std": float(policy.alpha_std().detach().cpu()),
                    "policy_entropy": float(entropy.cpu()),
                    "training_mean_reward": float(final_training["mean_reward"]),
                    "training_reward_std": float(final_training["reward_std"]),
                    "peak_allocated_bytes": training.peak_allocated_bytes,
                    "peak_reserved_bytes": training.peak_reserved_bytes,
                    **evaluation.summary,
                }
            )
            candidate_rows.extend(evaluation.records)
            set_rows.extend(evaluation.candidate_sets)
            selector_rows.extend(evaluation.selectors)

        # Keep IID-trained routers fixed so task-shift results measure transfer.
        es_operator, es_theta, es_critic = _load_es(base, seed, device)
        for variant_index, variant in enumerate(rl.generalization_variants):
            variant_tasks = family.sample_tasks(
                geometries=base.base.test_geometries,
                seed=seed + 30_000 + variant_index,
                variant=variant,
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
            methods = [
                ("gpu_es", es_operator, es_theta, es_critic),
                (
                    "gpu_rl_stochastic",
                    PolicyEvaluationAdapter(
                        policy,
                        deterministic=False,
                        seed=seed + 30_500 + 20 * variant_index + 1,
                    ),
                    dummy_theta,
                    critic,
                ),
                (
                    "gpu_rl_deterministic",
                    PolicyEvaluationAdapter(
                        policy,
                        deterministic=True,
                        seed=seed + 30_500 + 20 * variant_index + 2,
                    ),
                    dummy_theta,
                    critic,
                ),
            ]
            for method, operator, theta, method_critic in methods:
                evaluation = evaluate_batched(
                    variant=method,
                    operator=operator,  # type: ignore[arg-type]
                    theta=theta,
                    critic=method_critic,
                    tasks=variant_tasks,
                    packed_tasks=packed_variant,
                    starts=variant_starts,
                    config=base.base.training,
                    seed=seed + 30_700 + 20 * variant_index,
                    useful_threshold=base.base.landscape.useful_threshold,
                    high_quality_threshold=base.base.landscape.high_quality_threshold,
                )
                generalization_rows.append(
                    {
                        "seed": seed,
                        "landscape_variant": variant,
                        "method": method,
                        **evaluation.summary,
                    }
                )

    table = _aggregate(primary_rows)
    write_csv(output / "primary_per_seed.csv", primary_rows)
    write_csv(output / "candidate_diagnostics.csv", candidate_rows)
    write_csv(output / "candidate_set_diagnostics.csv", set_rows)
    write_csv(output / "selector_per_seed.csv", selector_rows)
    write_csv(output / "training_history.csv", history_rows)
    write_csv(output / "checkpoint_metrics.csv", checkpoint_rows)
    write_csv(output / "generalization_metrics.csv", generalization_rows)
    write_csv(output / "RL_TABLE.csv", table)
    _plot_learning_curves(checkpoint_rows, output / "rl_learning_curves.png")
    first_seed_history = [row for row in history_rows if row["seed"] == seeds[0]]
    _plot_training_diagnostics(
        first_seed_history, output / "rl_training_diagnostics_first_seed.png"
    )
    aggregate = {
        "seed_count": len(seeds),
        "runtime": runtime.to_dict(),
        "table": table,
        "black_box_training": True,
        "region_labels_used_for_training": False,
        "oracle_selector_used_for_training": False,
        "test_set_used_for_hyperparameter_selection": False,
    }
    (output / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate
