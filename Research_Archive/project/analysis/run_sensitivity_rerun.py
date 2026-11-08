"""Evaluate the documented sensitivity grid on Windows CPU.

The rerun uses the canonical task implementation and documented grid. The
earlier macOS program and runtime were unavailable for this rerun, so these
outputs are independently generated sensitivity results. Evaluation performs
no optimizer updates, backward passes, parameter writes or model saves.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import run_fair_selector_verification as fair
from gen3_gpu.config_symmetry import SymmetryStudyConfig
from gen3_gpu.landscape_representation import LandscapeProposalBatch
from gen3_gpu.study_symmetry import _components
from gen3_gpu.symmetry import build_equivariant_components
from gen3_gpu.training_landscape_supervised import _matching

SPEC = importlib.util.spec_from_file_location(
    "sensitivity_ablation_loader", ROOT / "analysis/run_direct_head_ablation.py"
)
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)
driver = ablation.driver

DIMENSIONS = (32, 64)
METHODS = ("Direct16", "DirectWide", "Relational", "Multiplicity")
BUDGETS = (0, 3, 9, 18)
QUALITY_CUTOFFS = (0.60, 0.62, 0.65)
REGION_CUTOFFS = (2.0, 2.5, 3.0)
CAP_METHODS = ("amplitude_top3_repeat", "amplitude_unique8", "random_top3_repeat")
METRICS = (
    "proposal_recall",
    "mode_set_recall",
    "useful_region_count",
    "candidate_useful_rate",
    "candidate_decoy_rate",
    "candidate_anchor_rate",
    "candidate_background_rate",
    "candidate_mean_quality",
    "exact_repeat_pair_rate",
    "candidate_useful_below_threshold_rate",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def csv_write(path, rows):
    if not rows:
        raise ValueError("Empty result rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def npz_write(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def checkpoint_path(d, seed, method):
    if method == "DirectWide":
        return (
            ROOT
            / f"reproduction_runs/direct_head_ablation_complete_20260909/d{d}/models/DirectWide_seed_{seed}.pt"
        )
    kind = "EquivariantAttention" if method == "Relational" else "CurrentAttention"
    return (
        ROOT
        / f"paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d{d}/models/{kind}_seed_{seed}.pt"
    )


def protocol():
    return {
        "title": "Paper B saved-model sensitivity and selector movement-cap rerun",
        "version": "windows-cpu-reconstruction-v1",
        "evidence_status": "New retrospective rerun; not recovered Mac artifacts or fresh held-out confirmation",
        "dimensions": list(DIMENSIONS),
        "seeds": list(driver.STUDY_SEEDS),
        "methods": list(METHODS),
        "neural_checkpoints": 180,
        "component_token_width": 32,
        "candidates": 8,
        "evaluation": {
            "geometries_per_seed": 5,
            "goals_per_geometry": 3,
            "starts_per_goal": 2,
            "task_seed_offset": 20000,
            "start_seed_offset": 20001,
            "split": "previously inspected IID",
        },
        "refinement": {
            "steps": list(BUDGETS),
            "rate": 0.28,
            "max_gradient_norm": 4.0,
            "algorithm": "unchanged TorchTaskBatch.refine; stabilized grad logQ, Q<1e-10 mixture fallback",
        },
        "scoring": {
            "quality_cutoffs": list(QUALITY_CUTOFFS),
            "mahalanobis_cutoffs": list(REGION_CUTOFFS),
            "coverage_denominator": 3,
            "thresholds_affect_only_scoring": True,
        },
        "multiplicity": {
            "rule": "coordinates only, group Euclidean distance <=1e-5, select groups of multiplicity>=2",
            "slots": "cycle discovered groups in representative-record order; no supplied useful-mode count",
            "arithmetic": "same float32 Torch task/refiner as neural methods, not original NumPy float64 diagnostic",
        },
        "cap_audit": {
            "dimension": 32,
            "family": "canonical IID",
            "methods": list(CAP_METHODS),
            "caps": ["uncapped", 6.0],
            "initial_displacement_only": True,
            "refinement_steps": 9,
            "quality_cutoff": 0.62,
            "mahalanobis_cutoff": 2.5,
            "center_tolerance": 1e-5,
            "anchor": "nearest observable center excluded by preserved grouping code",
            "top3_slots": list(fair.ALLOCATION),
            "random_seed_offset": 231001,
            "random_device": "CPU; paired same selected centers before/after cap",
            "scorer_training": False,
        },
        "precision": "checkpoint/model/task float32; summary mean/sample SD float64",
        "execution": {
            "device": "cpu",
            "torch_threads": 1,
            "training_steps": 0,
            "stopping_rule": "all60 dimension/seed units and all30 cap seeds, all outcomes retained",
        },
        "historical_runtime_unavailable": "Mac Python3.12.14 / torch2.14.0 / NumPy2.5.2; use installed recorded Windows environment and report actual differences",
        "trajectory_caching": "one deterministic 18-step trajectory per proposal, score cached 0/3/9/18 snapshots at all thresholds",
        "sources": "all gen3_gpu/creativity_toy Python files, inspected loaders, and this script hashed before execution",
    }


def source_paths():
    paths = [
        Path(__file__),
        ROOT / "analysis/run_direct_head_ablation.py",
        ROOT / "analysis/run_fair_selector_verification.py",
        ROOT / "paper_csd_toy_revision5/scripts/run_matched_scaling_study.py",
        ROOT / "run_scripts/reproduction_support.py",
    ]
    paths += list((ROOT / "creativity_toy").rglob("*.py")) + list(
        (ROOT / "gen3_gpu").rglob("*.py")
    )
    return sorted(set(paths))


def runtime():
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": torch.get_num_threads(),
    }


def make_test(config, seed):
    benchmark = driver._benchmark(config)
    family = driver._family(benchmark)
    tasks = family.sample_tasks(
        geometries=benchmark.base.base.test_geometries,
        seed=seed + 20000,
        variant="iid",
        all_goals=True,
    )
    starts = driver._make_starts(
        tasks, trials=benchmark.base.base.trials_per_task, seed=seed + 20001
    )
    packed, h = driver._aligned(tasks, starts, torch.device("cpu"))
    assert packed.batch_size == 30 and h.shape == (30, config.dimension)
    assert (packed.n_useful == 3).all()
    return packed, h


def tensor_digest(tasks, h):
    out = {}
    for name, v in [("starts", h)] + [
        (f.name, getattr(tasks, f.name)) for f in dataclasses.fields(tasks)
    ]:
        if isinstance(v, torch.Tensor):
            out[name] = hashlib.sha256(
                v.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
    return out


def load_model(config, seed, method):
    path = checkpoint_path(config.dimension, seed, method)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    if method == "Relational":
        model = driver._equivariant_model(
            config,
            seed=seed,
            device=torch.device("cpu"),
            mean=state["invariant_mean"],
            scale=state["invariant_scale"],
        )
    else:
        width = config.dimension if method == "DirectWide" else 16
        assert state["w1"].shape[0] == width
        model = ablation.direct_model(
            config,
            seed,
            torch.device("cpu"),
            state["descriptor_mean"],
            state["descriptor_scale"],
            width,
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    assert all(
        p.dtype == torch.float32 and not p.requires_grad for p in model.parameters()
    )
    return model


def multiplicity_proposal(tasks, h):
    # Use only coordinate repetition so the selector cannot exploit labels or a supplied mode count.
    proposals = []
    selections = []
    for row in range(tasks.batch_size):
        centers = tasks.component_centers[row, tasks.component_mask[row]]
        same = (
            torch.linalg.vector_norm(centers[:, None] - centers[None, :], dim=-1)
            <= 1e-5
        )
        reps = [i for i in range(len(centers)) if not bool(same[i, :i].any())]
        chosen = [i for i in reps if int(same[i].sum()) >= 2]
        if not chosen:
            raise RuntimeError(
                "No duplicate-reference group; no hidden-label fallback allowed"
            )
        index = torch.tensor([chosen[k % len(chosen)] for k in range(8)])
        proposals.append(centers[index])
        selections.append(index)
    points = torch.stack(proposals)
    raw = points - h[:, None, :]
    return LandscapeProposalBatch(
        raw,
        F.normalize(raw, dim=-1, eps=1e-8),
        torch.linalg.vector_norm(raw, dim=-1),
        points,
        None,
    ), torch.stack(selections)


def cap_proposal(proposal, h, cap):
    raw = proposal.candidate - h[:, None, :]
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    used = raw * torch.clamp(cap / torch.clamp(norm, min=1e-8), max=1.0)
    points = h[:, None, :] + used
    return LandscapeProposalBatch(
        used,
        F.normalize(used, dim=-1, eps=1e-8),
        torch.linalg.vector_norm(used, dim=-1),
        points,
        None,
    )


def endpoints(tasks, points, budgets):
    current = points.clone()
    previous = 0
    result = {}
    for step in budgets:
        current = tasks.refine(
            current, steps=step - previous, rate=0.28, max_gradient_norm=4.0
        )
        if not torch.isfinite(current).all():
            raise FloatingPointError("Non-finite refinement")
        result[step] = current.clone()
        previous = step
    return result


def score_points(tasks, points, qcut, rcut):
    scored = dataclasses.replace(tasks, region_threshold=rcut)
    quality = scored.quality(points)
    region = scored.regions(points)
    if not torch.isfinite(quality).all():
        raise FloatingPointError("Non-finite quality")
    useful = (region >= 1) & (region <= tasks.n_useful[:, None]) & (quality >= qcut)
    ids = torch.arange(1, int(tasks.n_useful.max()) + 1)
    count = (
        (useful[:, :, None] & (region[:, :, None] == ids[None, None, :]))
        .any(dim=1)
        .sum(dim=1)
        .float()
    )
    upper = torch.triu_indices(points.shape[1], points.shape[1], offset=1)
    distance = torch.linalg.vector_norm(points[:, :, None] - points[:, None, :], dim=-1)
    per_set = {
        "proposal_recall": useful.any(dim=1).float(),
        "mode_set_recall": count / tasks.n_useful.float(),
        "useful_region_count": count,
        "candidate_useful_rate": useful.float().mean(dim=1),
        "candidate_decoy_rate": (region > tasks.n_useful[:, None]).float().mean(dim=1),
        "candidate_anchor_rate": (region == 0).float().mean(dim=1),
        "candidate_background_rate": (region < 0).float().mean(dim=1),
        "candidate_useful_below_threshold_rate": (
            (region >= 1) & (region <= tasks.n_useful[:, None]) & (quality < qcut)
        )
        .float()
        .mean(dim=1),
        "candidate_mean_quality": quality.mean(dim=1),
        "exact_repeat_pair_rate": (distance[:, upper[0], upper[1]] <= 1e-5)
        .float()
        .mean(dim=1),
    }
    summary = {k: float(v.mean()) for k, v in per_set.items()}
    return summary, per_set, quality, region


def unit_sensitivity(out, d, seed):
    config = driver.matched_config(d)
    tasks, h = make_test(config, seed)
    allrows = []
    setrows = []
    metadata = {
        "dimension": d,
        "seed": seed,
        "input_hashes": tensor_digest(tasks, h),
        "config": config.to_dict(),
        "methods": {},
    }
    for method in METHODS:
        begin = time.perf_counter()
        if method == "Multiplicity":
            proposal, selected = multiplicity_proposal(tasks, h)
        else:
            model = load_model(config, seed, method)
            kind = (
                "EquivariantAttention" if method == "Relational" else "CurrentAttention"
            )
            proposal = model(h, _components(kind, tasks, h))
        if not torch.isfinite(proposal.candidate).all():
            raise FloatingPointError("Non-finite proposals")
        matching = _matching(proposal, h, tasks, lambda_alpha=0.1)
        cosine = float(matching["matched_cosine"].mean())
        points = endpoints(tasks, proposal.candidate, BUDGETS)
        arrays = {
            "starts": h.numpy(),
            "geometry_id": tasks.geometry_id.numpy(),
            "goal_index": tasks.goal_index.numpy(),
            "alpha": proposal.alpha.numpy(),
            "direction": proposal.direction.numpy(),
        }
        if method == "Multiplicity":
            arrays["selected_record_indices"] = selected.numpy()
        for step in BUDGETS:
            arrays[f"points_{step}"] = points[step].numpy()
            for qcut in QUALITY_CUTOFFS:
                for rcut in REGION_CUTOFFS:
                    metrics, per_set, q, r = score_points(
                        tasks, points[step], qcut, rcut
                    )
                    row = {
                        "dimension": d,
                        "seed": seed,
                        "method": method,
                        "refinement_steps": step,
                        "quality_cutoff": qcut,
                        "region_cutoff": rcut,
                        "n_sets": 30,
                        "n_candidates": 8,
                        "initial_matched_cosine": cosine,
                        **metrics,
                        "gradient_queries_per_set": step * 8,
                    }
                    allrows.append(row)
                    if qcut == 0.62 and rcut == 2.5:
                        arrays[f"quality_{step}"] = q.numpy()
                        arrays[f"region_{step}"] = r.numpy()
                        for i in range(30):
                            setrows.append(
                                {
                                    "dimension": d,
                                    "seed": seed,
                                    "method": method,
                                    "refinement_steps": step,
                                    "set_index": i,
                                    "geometry_id": int(tasks.geometry_id[i]),
                                    "goal_index": int(tasks.goal_index[i]),
                                    **{k: float(v[i]) for k, v in per_set.items()},
                                }
                            )
        npz_write(out / f"endpoints/d{d}/seed_{seed}_{method}.npz", **arrays)
        metadata["methods"][method] = {
            "elapsed_seconds": time.perf_counter() - begin,
            "model_path": None
            if method == "Multiplicity"
            else str(checkpoint_path(d, seed, method).relative_to(ROOT)),
        }
    csv_write(out / f"units/sensitivity_d{d}_seed_{seed}.csv", allrows)
    csv_write(out / f"units/sensitivity_sets_d{d}_seed_{seed}.csv", setrows)
    atomic_json(out / f"units/sensitivity_d{d}_seed_{seed}.json", metadata)


def unit_caps(out, seed):
    tasks, h = make_test(SymmetryStudyConfig(), seed)
    groups = fair.group_references(tasks, h, build_equivariant_components(tasks, h))
    rows = []
    sets = []
    metadata = {"seed": seed, "input_hashes": tensor_digest(tasks, h), "methods": {}}
    for method in CAP_METHODS:
        original, selected = fair.selector_proposal(
            method,
            h,
            groups,
            scorer=None,
            mean=None,
            scale=None,
            random_seed=seed + 231001,
        )
        lengths = torch.linalg.vector_norm(original.candidate - h[:, None, :], dim=-1)
        arrays = {
            "starts": h.numpy(),
            "selected_group_indices": selected.numpy(),
            "uncapped_lengths": lengths.numpy(),
        }
        for label, proposal in [
            ("uncapped", original),
            ("cap6", cap_proposal(original, h, 6.0)),
        ]:
            points = endpoints(tasks, proposal.candidate, (0, 9))
            summaries = {}
            for step in [0, 9]:
                m, per_set, q, r = score_points(tasks, points[step], 0.62, 2.5)
                summaries.update(
                    {
                        ("before_" if step == 0 else "after_") + k: v
                        for k, v in m.items()
                    }
                )
                arrays[f"{label}_points_{step}"] = points[step].numpy()
                arrays[f"{label}_quality_{step}"] = q.numpy()
                arrays[f"{label}_region_{step}"] = r.numpy()
                for i in range(30):
                    sets.append(
                        {
                            "seed": seed,
                            "method": method,
                            "cap_condition": label,
                            "refinement_steps": step,
                            "set_index": i,
                            "geometry_id": int(tasks.geometry_id[i]),
                            "goal_index": int(tasks.goal_index[i]),
                            **{k: float(v[i]) for k, v in per_set.items()},
                        }
                    )
            usedlength = torch.linalg.vector_norm(
                proposal.candidate - h[:, None, :], dim=-1
            )
            if label == "cap6" and float(usedlength.max()) > 6.0 + 1e-5:
                raise AssertionError("Cap violated")
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "cap_condition": label,
                    "n_sets": 30,
                    "n_candidates": 8,
                    "uncapped_max_displacement": float(lengths.max()),
                    "uncapped_fraction_above6": float((lengths > 6.0).float().mean()),
                    "used_max_displacement": float(usedlength.max()),
                    **summaries,
                }
            )
        npz_write(out / f"endpoints/caps/seed_{seed}_{method}.npz", **arrays)
    csv_write(out / f"units/caps_seed_{seed}.csv", rows)
    csv_write(out / f"units/cap_sets_seed_{seed}.csv", sets)
    atomic_json(out / f"units/caps_seed_{seed}.json", metadata)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def complete_unit(out, key, csvnames):
    paths = [out / p for p in csvnames]
    atomic_json(
        out / f"completion/{key}.json",
        {"files": {str(p.relative_to(out)): sha(p) for p in paths}},
    )


def unit_done(out, key):
    marker = out / f"completion/{key}.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    for rel, wanted in record["files"].items():
        if sha(out / rel) != wanted:
            raise RuntimeError("Completed unit integrity failed: " + rel)
    return True


def run(out, resume):
    out = out.resolve()
    allowed = ROOT / "reproduction_runs"
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("Use a fresh experiment folder inside reproduction_runs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_paths()}
    inputs = {
        str(checkpoint_path(d, s, m).relative_to(ROOT)): sha(checkpoint_path(d, s, m))
        for d in DIMENSIONS
        for s in driver.STUDY_SEEDS
        for m in METHODS[:3]
    }
    lock = {
        "protocol": protocol(),
        "source_hashes": sources,
        "checkpoint_hashes_before": inputs,
        "runtime": runtime(),
    }
    if out.exists():
        if not resume:
            raise FileExistsError("Use --resume only for this exact frozen run")
        saved = json.loads((out / "protocol.json").read_text())
        if saved != lock:
            raise RuntimeError("Resume protocol/source/checkpoint/runtime mismatch")
    else:
        if resume:
            raise FileNotFoundError(out)
        out.mkdir(parents=True)
        atomic_json(out / "protocol.json", lock)
        atomic_json(
            out / "started.json",
            {
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "authorization": "User requested rerun of the three missing evaluation checks; zero training",
            },
        )
    started = time.perf_counter()
    with torch.inference_mode():
        for d in DIMENSIONS:
            for ordinal, seed in enumerate(driver.STUDY_SEEDS, 1):
                key = f"d{d}_seed_{seed}"
                if not unit_done(out, key):
                    unit_sensitivity(out, d, seed)
                    names = [
                        str(p.relative_to(out))
                        for p in sorted(
                            (out / "endpoints" / f"d{d}").glob(f"seed_{seed}_*.npz")
                        )
                    ]
                    names += [
                        f"units/sensitivity_d{d}_seed_{seed}.csv",
                        f"units/sensitivity_sets_d{d}_seed_{seed}.csv",
                        f"units/sensitivity_d{d}_seed_{seed}.json",
                    ]
                    complete_unit(out, key, names)
                print(f"sensitivity d={d} seed={seed} {ordinal}/30", flush=True)
        for ordinal, seed in enumerate(driver.STUDY_SEEDS, 1):
            key = f"cap_seed_{seed}"
            if not unit_done(out, key):
                unit_caps(out, seed)
                names = [
                    str(p.relative_to(out))
                    for p in sorted((out / "endpoints/caps").glob(f"seed_{seed}_*.npz"))
                ]
                names += [
                    f"units/caps_seed_{seed}.csv",
                    f"units/cap_sets_seed_{seed}.csv",
                    f"units/caps_seed_{seed}.json",
                ]
                complete_unit(out, key, names)
            print(f"cap seed={seed} {ordinal}/30", flush=True)
    sensitivity = [
        r
        for d in DIMENSIONS
        for s in driver.STUDY_SEEDS
        for r in read_csv(out / f"units/sensitivity_d{d}_seed_{s}.csv")
    ]
    caps = [
        r
        for s in driver.STUDY_SEEDS
        for r in read_csv(out / f"units/caps_seed_{s}.csv")
    ]
    assert len(sensitivity) == 8640 and len(caps) == 180
    assert (
        len(
            {
                tuple(
                    r[k]
                    for k in [
                        "dimension",
                        "seed",
                        "method",
                        "refinement_steps",
                        "quality_cutoff",
                        "region_cutoff",
                    ]
                )
                for r in sensitivity
            }
        )
        == 8640
    )
    assert (
        len({tuple(r[k] for k in ["seed", "method", "cap_condition"]) for r in caps})
        == 180
    )
    csv_write(out / "sensitivity_per_seed.csv", sensitivity)
    csv_write(out / "selector_budget_per_seed.csv", caps)
    after = {p: sha(ROOT / p) for p in inputs}
    source_after = {p: sha(ROOT / p) for p in sources}
    if after != inputs or source_after != sources:
        raise AssertionError("Source/checkpoint mutation")
    atomic_json(
        out / "manifest.json",
        {
            "status": "COMPLETE new Windows CPU retrospective rerun",
            "protocol_sha256": sha(out / "protocol.json"),
            "rows": {"sensitivity": len(sensitivity), "selector_budget": len(caps)},
            "seeds": list(driver.STUDY_SEEDS),
            "checkpoint_count": len(inputs),
            "checkpoint_hashes_before": inputs,
            "checkpoint_hashes_after": after,
            "source_hashes": sources,
            "mutation_check": "PASS",
            "runtime": runtime(),
            "training_steps": 0,
            "gradient_backwards": 0,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "qualification": "Original Mac source, row files and runtime not recovered. This is an explicit source-derived rerun, not claimed bitwise reproduction.",
        },
    )
    print(
        "COMPLETE: 8640 sensitivity rows + 180 cap rows; zero training; hashes unchanged",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    run(a.output, a.resume)
