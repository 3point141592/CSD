"""Evaluation-only replay of packaged checkpoints. This program cannot train."""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from gen3_gpu import study_symmetry as study
from gen3_gpu.config_symmetry import SymmetryStudyConfig
from gen3_gpu.config_symmetry_d64 import symmetry_d64_config
from run_scripts.reproduction_support import fresh_output, sha256

SPEC = importlib.util.spec_from_file_location(
    "ablation_driver", ROOT / "analysis/run_direct_head_ablation.py"
)
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)
driver = ablation.driver
from analysis.run_fair_selector_verification import Protocol, evaluate_proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=["matched", "canonical", "ablation"], required=True
    )
    parser.add_argument("--dimensions", nargs="+", type=int, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=ROOT / "reproduction_runs/direct_head_ablation_complete_20260909",
    )
    args = parser.parse_args()
    allowed = (
        [32, 64]
        if args.family in ["canonical", "ablation"]
        else list(driver.DIMENSIONS)
    )
    if any(d not in allowed for d in args.dimensions):
        parser.error(f"allowed dimensions for {args.family}: {allowed}")
    args.ablation_root = args.ablation_root.resolve()
    if ROOT not in args.ablation_root.parents:
        parser.error("--ablation-root must be inside the reproduction package")
    out = fresh_output(args.output)
    out.mkdir(parents=True)
    device = driver.configure_runtime(args.device)
    rows = []
    inputs = {}
    for d in args.dimensions:
        config = (
            (SymmetryStudyConfig() if d == 32 else symmetry_d64_config())
            if args.family == "canonical"
            else driver.matched_config(d)
        )
        benchmark = driver._benchmark(config)
        conditions = (
            list(config.conditions)
            if args.family == "canonical"
            else (
                ["CurrentAttention", "EquivariantAttention"]
                if args.family == "matched"
                else ["Direct16_archived", "Relational_archived", "DirectWide"]
            )
        )
        for ordinal, seed in enumerate(driver.STUDY_SEEDS, 1):
            _, _, _, _, test, starts = driver._datasets(
                driver._family(benchmark), benchmark, seed
            )
            tasks, h = driver._aligned(test, starts, device)
            transforms = driver._exact_transforms(config, h, seed=seed + 81_001)
            evals = {"iid": (tasks, h)}
            for key, transform in transforms.items():
                if args.family != "canonical" and key != "exact_rotated":
                    continue
                evals[key] = (
                    driver.transform_tasks(tasks, transform),
                    driver.transform_points(
                        h, transform.rotation, transform.translation
                    ),
                )
            for label in conditions:
                rel = label.startswith("Equivariant") or label == "Relational_archived"
                kind = "EquivariantAttention" if rel else "CurrentAttention"
                if args.family == "canonical":
                    if d == 32 and label == "CurrentAttention":
                        path = (
                            ROOT
                            / f"results/d32_landscape_representation/models/LandscapeAttention_seed_{seed}.pt"
                        )
                    else:
                        path = (
                            ROOT
                            / f"results/d{d}_{'equivariant_attention' if rel else 'symmetry_augmentation'}/models/{label}_seed_{seed}.pt"
                        )
                elif label == "DirectWide":
                    path = args.ablation_root / f"d{d}/models/DirectWide_seed_{seed}.pt"
                else:
                    path = (
                        ROOT
                        / f"paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d{d}/models/{kind}_seed_{seed}.pt"
                    )
                key = path.relative_to(ROOT).as_posix()
                inputs[key] = sha256(path)
                payload = torch.load(path, map_location=device, weights_only=True)
                state = payload["state_dict"]
                if rel:
                    factory = lambda: driver._equivariant_model(
                        config,
                        seed=seed,
                        device=device,
                        mean=state["invariant_mean"],
                        scale=state["invariant_scale"],
                    )
                else:
                    factory = lambda: ablation.direct_model(
                        config,
                        seed,
                        device,
                        state["descriptor_mean"],
                        state["descriptor_scale"],
                        int(state["w1"].shape[0]),
                    )
                model, _, _ = driver._load(path, factory, device)
                for split, (task, hs) in evals.items():
                    if args.family == "ablation":
                        components = study._components(kind, task, hs)
                        with torch.no_grad():
                            proposal = model(hs, components)
                        metrics, _ = evaluate_proposal(
                            proposal, task, hs, protocol=Protocol(dimension=d)
                        )
                    else:
                        metrics = driver._evaluate(
                            kind, model, task, hs, benchmark, config
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
            driver.write_csv(out / "per_seed.csv", rows)
            print(f"{args.family} replay d={d} seed={seed} ({ordinal}/30)", flush=True)
    changed = [key for key, value in inputs.items() if sha256(ROOT / key) != value]
    if changed:
        raise RuntimeError(f"input checkpoints changed: {changed}")
    driver._atomic_json(
        out / "manifest.json",
        {
            "family": args.family,
            "dimensions": args.dimensions,
            "rows": len(rows),
            "optimizer_steps": 0,
            "all_checkpoints_unchanged": True,
            "checkpoint_hashes": inputs,
            "source_sha256": sha256(__file__),
            "runtime": driver.runtime_info(device, torch.float32).to_dict(),
            "splits": "four primary coordinate transforms only"
            if args.family == "canonical"
            else "IID and exact_rotated",
        },
    )


if __name__ == "__main__":
    main()
