"""Retrospective CPU-only decoder-span audit of saved Direct models.

No training or task redesign. IID tasks and starts are reconstructed from the
same NumPy generator and seed offsets as the archived matched study.
Projection is a necessary directional upper bound before refinement, not a
claim about endpoint reachability or a proof that wider heads will succeed.
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
)
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--dimensions", type=int, nargs="+", default=[32, 64])
parser.add_argument("--seed-limit", type=int, default=0)
args = parser.parse_args()
ROOT = args.package_root.resolve(strict=True)
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

torch.set_num_threads(2)
from creativity_toy.config import STUDY_SEEDS
from creativity_toy.experiments.study_v2 import _make_starts
from gen3_gpu.config_symmetry import SymmetryStudyConfig
from gen3_gpu.study_symmetry import _benchmark, _family

out = (args.output_dir or ROOT / "new_verification/audit_20260909").resolve()
out.mkdir(parents=True, exist_ok=True)
for name in [
    "direct_rank_per_seed.csv",
    "direct_rank_per_target.csv",
    "direct_rank_summary.json",
]:
    if (out / name).exists():
        raise FileExistsError(
            f"Refusing to replace {name}; choose a fresh --output-dir"
        )
seeds = list(STUDY_SEEDS)[: args.seed_limit or None]
seed_rows = []
target_rows = []
for d in args.dimensions:
    config = SymmetryStudyConfig(dimension=d)
    benchmark = _benchmark(config)
    base = benchmark.base.base
    for seed in seeds:
        model_path = (
            ROOT
            / f"paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d{d}/models/CurrentAttention_seed_{seed}.pt"
        )
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        state = payload["state_dict"]
        matrix = torch.cat([state["w2"].double(), state["b2"].double()[:, None]], dim=1)
        u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
        rank = int((s > s[0] * 1e-6).sum())
        orthogonal = u[:, :rank].numpy()
        family = _family(benchmark)
        tasks = family.sample_tasks(
            geometries=base.test_geometries,
            seed=seed + 20000,
            variant="iid",
            all_goals=True,
        )
        starts = _make_starts(tasks, trials=base.trials_per_task, seed=seed + 20001)
        all_bounds = []
        for task_index, (task, task_starts) in enumerate(zip(tasks, starts)):
            for start_index, start in enumerate(task_starts):
                # Match evaluator rounding so the geometric comparison uses the same displacements.
                displacement = (
                    task.useful_centers.astype(np.float32)
                    - np.asarray(start, dtype=np.float32)[None, :]
                )
                target = displacement / np.linalg.norm(
                    displacement, axis=1, keepdims=True
                )
                bounds = np.linalg.norm(target.astype(np.float64) @ orthogonal, axis=1)
                all_bounds.extend(bounds.tolist())
                for target_index, bound in enumerate(bounds):
                    target_rows.append(
                        dict(
                            dimension=d,
                            seed=seed,
                            task_index=task_index,
                            start_index=start_index,
                            target_index=target_index,
                            max_direction_cosine_in_decoder_span=float(bound),
                        )
                    )
        archive = list(
            csv.DictReader(
                (model_path.parent.parent / "per_seed.csv").open(
                    newline="", encoding="utf-8-sig"
                )
            )
        )
        observed = next(
            r
            for r in archive
            if int(r["seed"]) == seed
            and r["condition"] == "CurrentAttention"
            and r["split"] == "iid"
        )
        row = dict(
            dimension=d,
            seed=seed,
            w2_rows=state["w2"].shape[0],
            w2_columns=state["w2"].shape[1],
            rank_w2_b2=rank,
            target_count=len(all_bounds),
            mean_span_cosine_upper_bound=statistics.mean(all_bounds),
            min_span_cosine_upper_bound=min(all_bounds),
            max_span_cosine_upper_bound=max(all_bounds),
            observed_saved_matched_cosine=float(observed["mean_matched_cosine"]),
            observed_saved_coverage=float(observed["mode_set_recall"]),
            checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            checkpoint=model_path.relative_to(ROOT).as_posix(),
        )
        seed_rows.append(row)
    print(f"d={d}: {len(seeds)} saved checkpoints inspected", flush=True)
for name, rows in [
    ("direct_rank_per_seed.csv", seed_rows),
    ("direct_rank_per_target.csv", target_rows),
]:
    with (out / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
summary = []
for d in args.dimensions:
    rows = [r for r in seed_rows if r["dimension"] == d]
    summary.append(
        dict(
            dimension=d,
            n_seeds=len(rows),
            n_targets=sum(r["target_count"] for r in rows),
            ranks=sorted(set(r["rank_w2_b2"] for r in rows)),
            mean_span_cosine_upper_bound=statistics.mean(
                r["mean_span_cosine_upper_bound"] for r in rows
            ),
            sd_span_cosine_upper_bound=statistics.stdev(
                r["mean_span_cosine_upper_bound"] for r in rows
            )
            if len(rows) > 1
            else None,
            mean_saved_matched_cosine=statistics.mean(
                r["observed_saved_matched_cosine"] for r in rows
            ),
            mean_saved_coverage=statistics.mean(
                r["observed_saved_coverage"] for r in rows
            ),
        )
    )
manifest = dict(
    status="NEW retrospective saved-checkpoint diagnostic; no training, CPU only",
    span_theorem="raw=W2*tanh(W1*z+b1)+b2; scalar normalization and alpha preserve span(W2,b2); rank<=17 for a 16-unit hidden layer",
    caveat="Upper bound on initial directional cosine only. It is not a post-refinement coverage bound and not a causal claim that widening alone fixes performance.",
    dtype="loaded float32 weights projected using float64 SVD; rtol=1e-6",
    iid_generator="original study_d16._datasets test branch, same family config, seeds+20000, starts+20001, five geometries/all goals/two starts",
    torch_version=torch.__version__,
    numpy_version=np.__version__,
    summary=summary,
)
(out / "direct_rank_summary.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
