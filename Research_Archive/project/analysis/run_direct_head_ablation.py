"""Retrospective ablation of Direct-Vector output-head capacity.

Train Direct-Vector models with wider output heads under the original matched
protocol; all model weights are optimized. Reevaluate the two archived models
on the same tasks and run a seed-7 width-16 pipeline check. This is a capacity
ablation, not a parameter-matched architecture comparison. Scientific outputs
are saved separately from the archived results.
"""

import argparse
import datetime
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from gen3_gpu.config_landscape_representation import LandscapeRepresentationStudyConfig
from gen3_gpu.landscape_representation import LandscapeAttentionPolicy
from gen3_gpu.training_landscape_supervised import evaluate_landscape_policy
from run_scripts.reproduction_support import fresh_output, sha256

SPEC = importlib.util.spec_from_file_location(
    "matched_driver",
    ROOT / "paper_csd_toy_revision5/scripts/run_matched_scaling_study.py",
)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def direct_model(config, seed, device, mean, scale, width):
    base = LandscapeRepresentationStudyConfig().representation
    return LandscapeAttentionPolicy(
        dimension=config.dimension,
        candidates=config.candidates,
        latent_dimension=config.latent_dimension,
        task_token_dimension=config.task_token_dimension,
        alpha_max=config.alpha_max,
        seed=seed + 10_020,
        device=device,
        descriptor_dimension=mean.numel(),
        component_hidden_dimension=base.component_hidden_dimension,
        component_token_dimension=base.component_token_dimension,
        descriptor_mean=mean,
        descriptor_scale=scale,
        proposal_hidden_dimension=width,
    )


def freeze_protocol():
    protocol = {
        "title": "Retrospective Direct output-head capacity ablation",
        "status": "frozen before any new ablation outcome; exploratory task families already inspected",
        "dimensions": [32, 64],
        "seeds": list(driver.STUDY_SEEDS),
        "changed_factor": "Direct proposal_hidden_dimension=dimension instead of 16",
        "unchanged": [
            "original coordinate/descriptor encoder",
            "component token dimension 32",
            "candidate count 8",
            "original task/start generation and effective batch seed",
            "80 training geometries",
            "1500 optimizer updates",
            "batch size 960",
            "AdamW learning rate 0.001 and weight decay 0.0001",
            "gradient clip 1.0",
            "lambda_alpha 0.1",
            "nine refinement steps at 0.28",
            "no augmentation",
            "IID and exact-rotation evaluation",
        ],
        "primary_endpoint": "paired seed-level IID useful-region coverage: wider Direct minus archived width-16 Direct",
        "secondary_endpoints": [
            "matched initial target cosine",
            "candidate useful fraction",
            "candidate decoy fraction",
            "training losses and full-training-set initial cosine",
            "runtime",
            "actual parameter count",
        ],
        "comparators": [
            "replayed archived Direct-16",
            "replayed archived Relational CSD",
        ],
        "pipeline_checks": "fresh Direct-16 seed7 at d32 and d64 under same1500update protocol",
        "stopping_rule": "complete all60 wider models plus2 width16 checks; retain every outcome; no tuning, early stopping, seed changes, or outcome-driven reruns",
        "claim_limit": "capacity ablation, not a parameter-matched architecture ranking or independent confirmatory task benchmark",
        "precision": "float32; TF32 disabled using original configure_runtime",
        "training_initialization_note": "larger parameter tensors change the within-model RNG consumption; same canonical construction seed is used",
        "source_hashes": {
            p.relative_to(ROOT).as_posix(): sha256(p)
            for p in [
                Path(__file__),
                ROOT / "paper_csd_toy_revision5/scripts/run_matched_scaling_study.py",
                ROOT / "gen3_gpu/landscape_representation.py",
                ROOT / "gen3_gpu/training_symmetry.py",
                ROOT / "gen3_gpu/training_landscape_supervised.py",
                ROOT / "gen3_gpu/study_symmetry.py",
                ROOT / "gen3_gpu/config_landscape_representation.py",
            ]
        },
    }
    return protocol


def run(out, device):
    out = fresh_output(out)
    lock = (
        ROOT / "audit_records/engineering_20260909/direct_head_ablation_protocol.json"
    )
    recorded = json.loads(lock.read_text(encoding="utf-8"))
    if recorded["protocol"] != freeze_protocol():
        raise RuntimeError("frozen protocol/source drift")
    out.mkdir(parents=True)
    (out / "protocol.json").write_text(
        lock.read_text(encoding="utf-8"), encoding="utf-8"
    )
    runtime = driver.runtime_info(device, torch.float32)
    driver._atomic_json(out / "runtime.json", runtime.to_dict())
    rows = []
    trainrows = []
    curves = []
    metadata = []
    input_hashes = {}
    for d in [32, 64]:
        config = driver.matched_config(d)
        benchmark = driver._benchmark(config)
        driver._atomic_json(out / f"d{d}" / "config.json", config.to_dict())
        for ordinal, seed in enumerate(driver.STUDY_SEEDS, 1):
            start = time.perf_counter()
            train, _, _, _, test, starts = driver._datasets(
                driver._family(benchmark), benchmark, seed
            )
            nested = driver._make_starts(
                train, trials=config.starts_per_task, seed=seed + 41_001
            )
            packed, h = driver._aligned(train, nested, device)
            comp = driver.build_component_descriptors(packed, h)
            mean, scale = driver.descriptor_normalization(comp)
            packed_test, h_test = driver._aligned(test, starts, device)
            transform = driver._exact_transforms(config, h_test, seed=seed + 81_001)[
                "exact_rotated"
            ]
            evals = {
                "iid": (packed_test, h_test),
                "exact_rotated": (
                    driver.transform_tasks(packed_test, transform),
                    driver.transform_points(
                        h_test, transform.rotation, transform.translation
                    ),
                ),
            }
            conditions = ["Direct16_archived", "Relational_archived", "DirectWide"] + (
                ["Direct16_retrained_check"] if seed == 7 else []
            )
            for label in conditions:
                learned = label in ["DirectWide", "Direct16_retrained_check"]
                relational = label == "Relational_archived"
                if learned:
                    width = d if label == "DirectWide" else 16
                    model = direct_model(config, seed, device, mean, scale, width)
                    driver._assert_nonoracle_model(model)
                    trained = driver.train_symmetry_policy(
                        model,
                        packed,
                        h,
                        comp,
                        component_kind="current",
                        augmented=False,
                        rotation_pool=torch.empty(0, d, d, device=device),
                        translation_radius_max=config.augmentation.translation_radius_max,
                        steps=config.supervised_steps,
                        batch_size=min(config.batch_size, len(h)),
                        learning_rate=config.learning_rate,
                        weight_decay=config.weight_decay,
                        gradient_clip=config.gradient_clip,
                        lambda_alpha=config.lambda_alpha,
                        seed=seed + 61_101,
                        curve_steps=config.curve_steps,
                    )
                    history = trained.history
                    meta = {
                        "dimension": d,
                        "seed": seed,
                        "condition": label,
                        "proposal_hidden_dimension": width,
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "training_seconds": trained.elapsed_seconds,
                        "examples_seen": trained.examples_seen,
                        "steps": config.supervised_steps,
                        "effective_batch_seed": seed + 61_101,
                        "protocol_sha256": sha256(lock),
                        "source_hashes": recorded["protocol"]["source_hashes"],
                    }
                    driver._save(
                        out / f"d{d}" / "models" / f"{label}_seed_{seed}.pt",
                        model,
                        history=history,
                        metadata=meta,
                    )
                    curves.extend(
                        {"dimension": d, "seed": seed, "condition": label, **point}
                        for point in history
                    )
                    # Omit refinement here so training fit measures the frozen model's own proposals.
                    fit = evaluate_landscape_policy(
                        model,
                        packed,
                        h,
                        comp,
                        lambda_alpha=config.lambda_alpha,
                        refinement_steps=0,
                        refinement_rate=benchmark.base.base.training.refinement_rate,
                        useful_threshold=benchmark.base.base.landscape.useful_threshold,
                        collect_per_set=False,
                        collect_attention_rows=False,
                    )
                    trainrows.append(
                        {
                            "dimension": d,
                            "seed": seed,
                            "condition": label,
                            **fit.summary,
                        }
                    )
                else:
                    kind = "EquivariantAttention" if relational else "CurrentAttention"
                    path = (
                        ROOT
                        / f"paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d{d}/models/{kind}_seed_{seed}.pt"
                    )
                    key = path.relative_to(ROOT).as_posix()
                    input_hashes[key] = sha256(path)
                    payload = torch.load(path, map_location=device, weights_only=True)
                    state = payload["state_dict"]
                    if relational:
                        factory = lambda: driver._equivariant_model(
                            config,
                            seed=seed,
                            device=device,
                            mean=state["invariant_mean"],
                            scale=state["invariant_scale"],
                        )
                    else:
                        factory = lambda: direct_model(
                            config, seed, device, mean, scale, 16
                        )
                    model, history, meta = driver._load(path, factory, device)
                    meta = {
                        **meta,
                        "dimension": d,
                        "seed": seed,
                        "condition": label,
                        "checkpoint_path": key,
                        "checkpoint_sha256": input_hashes[key],
                    }
                model.eval()
                metadata.append(meta)
                for split, (tasks, evalh) in evals.items():
                    metrics = driver._evaluate(
                        "EquivariantAttention" if relational else "CurrentAttention",
                        model,
                        tasks,
                        evalh,
                        benchmark,
                        config,
                    )
                    rows.append(
                        {
                            "dimension": d,
                            "seed": seed,
                            "condition": label,
                            "split": split,
                            **metrics,
                        }
                    )
                del model
            driver.write_csv(out / "per_seed.csv", rows)
            driver.write_csv(out / "training_fit.csv", trainrows)
            driver.write_csv(out / "training_curves.csv", curves)
            driver._atomic_json(out / "model_metadata.json", metadata)
            driver._atomic_json(
                out / "progress.json",
                {
                    "dimension": d,
                    "seed": seed,
                    "completed_in_dimension": ordinal,
                    "required_per_dimension": 30,
                    "wide_models_completed": sum(
                        m["condition"] == "DirectWide" for m in metadata
                    ),
                },
            )
            print(
                f"direct-head ablation d={d} seed={seed} ({ordinal}/30), {time.perf_counter() - start:.1f}s",
                flush=True,
            )
    changed = [k for k, v in input_hashes.items() if sha256(ROOT / k) != v]
    if changed:
        raise RuntimeError(f"archived checkpoint mutation: {changed}")
    driver._atomic_json(
        out / "completion.json",
        {
            "complete": True,
            "wide_models": 60,
            "pipeline_check_models": 2,
            "result_rows": len(rows),
            "input_checkpoint_hashes": input_hashes,
            "input_checkpoints_unchanged": True,
            "protocol_sha256": sha256(lock),
            "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.freeze_protocol:
        dest = (
            ROOT
            / "audit_records/engineering_20260909/direct_head_ablation_protocol.json"
        )
        if dest.exists():
            raise FileExistsError(dest)
        protocol = freeze_protocol()
        body = {
            "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "protocol": protocol,
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(body, indent=2), encoding="utf-8")
        (dest.with_suffix(".sha256")).write_text(sha256(dest) + "\n", encoding="utf-8")
        print(f"frozen protocol: {sha256(dest)}")
        return
    if args.output is None:
        parser.error("--output is required")
    run(args.output, driver.configure_runtime(args.device))


if __name__ == "__main__":
    main()
