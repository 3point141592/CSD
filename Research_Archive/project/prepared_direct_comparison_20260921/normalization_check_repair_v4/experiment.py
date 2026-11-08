"""Fixed prepared-feature comparison; original scientific modules remain unchanged."""

import argparse
import csv
import dataclasses
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SNAP = OUT / "source_snapshot"
sys.path.append(str(ROOT))
sys.path.insert(0, str(SNAP))
import numpy as np
import psutil
import torch
from creativity_toy.experiments.study_v2 import _make_starts
from gen3_gpu.landscape_representation import (
    ALLOWED_COMPONENT_INPUT_FIELDS,
    ComponentDescriptorBatch,
    LandscapeAttentionPolicy,
    build_component_descriptors,
    descriptor_normalization,
    random_component_permutation,
)
from gen3_gpu.runtime import configure_runtime, runtime_info, synchronize
from gen3_gpu.study_symmetry import (
    _aligned,
    _assert_nonoracle_model,
    _benchmark,
    _current_model,
    _datasets,
    _equivariant_model,
    _evaluate,
    _exact_transforms,
    _family,
)
from gen3_gpu.symmetry import (
    build_equivariant_components,
    invariant_normalization,
    transform_points,
    transform_tasks,
)
from gen3_gpu.task_batch import pack_tasks
from gen3_gpu.training_landscape_supervised import (
    evaluate_landscape_policy,
    landscape_supervised_objective,
)
from gen3_gpu.training_symmetry import train_symmetry_policy
from run_matched_scaling_study import matched_config

ORIGINAL = ROOT / "paper_csd_toy_revision5/assets/source_data/matched_scaling_study"
WIDE = (
    ROOT
    / "Paper_B_Publication_Materials_20260912/03_Reproduction/reproduction_runs/direct_head_ablation_complete_20260909"
)
MODELS = ("DirectPrepared", "Relational")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def fingerprint():
    paths = [
        OUT / "PLAN.md",
        OUT / "configurations.json",
        OUT / "experiment.py",
        OUT / "analyze.py",
    ] + sorted(SNAP.rglob("*.py"))
    return {str(p.relative_to(OUT)).replace("\\", "/"): sha(p) for p in paths}


def check_lock():
    lock = read_json(OUT / "lock.json")
    assert lock["hashes"] == fingerprint(), (
        "Frozen code/config changed; refusing resume"
    )
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in ("gen3_gpu", "creativity_toy") and getattr(
            module, "__file__", None
        ):
            assert Path(module.__file__).resolve().is_relative_to(SNAP), (
                name,
                module.__file__,
            )
    return lock


def freeze():
    assert not (OUT / "lock.json").exists(), "Already frozen"
    discovery = read_json(OUT / "discovery_review.json")
    assert discovery["equivalent_completed_runs"] == 0
    write_json(
        OUT / "lock.json",
        {
            "frozen_utc": now(),
            "hashes": fingerprint(),
            "study": "retrospective_follow_up",
            "jobs": 120,
            "discovery": sha(OUT / "discovery_review.json"),
        },
    )
    write_json(
        OUT / "manifest.json",
        {
            "created_utc": now(),
            "status": "awaiting_verification",
            "jobs": {
                f"d{d}_s{s}_{m}": {"status": "pending"}
                for d in (32, 64)
                for s in matched_config(d).seeds
                for m in MODELS
            },
        },
    )


def update_manifest(key=None, **values):
    obj = read_json(OUT / "manifest.json")
    obj["updated_utc"] = now()
    if key:
        obj["jobs"][key].update(values)
    else:
        obj.update(values)
    write_json(OUT / "manifest.json", obj)


def finite(value):
    if isinstance(value, torch.Tensor):
        assert bool(torch.isfinite(value).all()), "Nonfinite tensor"
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite(v)
    elif isinstance(value, float):
        assert math.isfinite(value), "Nonfinite number"


def tensor_hash(x):
    a = x.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def direct_features(eq):
    return ComponentDescriptorBatch(
        torch.cat([eq.relative, eq.invariant], dim=-1), eq.mask
    )


def factory(kind, c, seed, device, stats):
    mean, scale = stats
    if kind == "Relational":
        return _equivariant_model(c, seed=seed, device=device, mean=mean, scale=scale)
    if kind == "HistoricalDirect":
        return _current_model(c, seed=seed, device=device, mean=mean, scale=scale)
    prepared = kind == "DirectPrepared"
    return LandscapeAttentionPolicy(
        dimension=c.dimension,
        candidates=c.candidates,
        latent_dimension=c.latent_dimension,
        task_token_dimension=c.task_token_dimension,
        alpha_max=c.alpha_max,
        seed=seed + 10020,
        device=device,
        descriptor_dimension=mean.numel(),
        component_hidden_dimension=256 if prepared else 64,
        component_token_dimension=128 if prepared else 32,
        proposal_hidden_dimension=256 if prepared else c.dimension,
        descriptor_mean=mean,
        descriptor_scale=scale,
    )


def runtime(device):
    info = runtime_info(device, torch.float32).to_dict()
    info.update(
        executable=sys.executable,
        numpy=np.__version__,
        psutil=psutil.__version__,
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        cudnn_version=torch.backends.cudnn.version(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(),
        pid=os.getpid(),
        host_memory=psutil.virtual_memory()._asdict(),
        disk=dict(zip(("total", "used", "free"), shutil.disk_usage(OUT))),
    )
    info["nvidia_smi"] = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
    ).stdout
    info["driver_version"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return info


def verify_runtime_identity(current):
    identity_keys = (
        "python",
        "platform",
        "torch_version",
        "torch_cuda_version",
        "device",
        "device_name",
        "dtype",
        "tf32_matmul",
        "tf32_cudnn",
        "executable",
        "numpy",
        "threads",
        "interop_threads",
        "cudnn_version",
        "deterministic_algorithms",
        "matmul_precision",
        "driver_version",
    )
    baseline = OUT / "runtime.json"
    if baseline.exists():
        previous = read_json(baseline)
        differences = {
            k: [previous.get(k), current.get(k)]
            for k in identity_keys
            if previous.get(k) != current.get(k)
        }
        assert not differences, (
            "Training environment changed; refusing to mix primary runs",
            differences,
        )
    else:
        write_json(baseline, current)
    write_json(
        OUT
        / "runtime_sessions"
        / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json",
        current,
    )


class Monitor:
    def __init__(self, device):
        self.device = device
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        if self.device.type == "cuda":
            self.peak_allocated = max(
                self.peak_allocated, torch.cuda.max_memory_allocated(self.device)
            )
            self.peak_reserved = max(
                self.peak_reserved, torch.cuda.max_memory_reserved(self.device)
            )

    def loop(self):
        while not self.stop.wait(0.5):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()

    def result(self):
        return {
            "peak_process_rss_bytes": self.peak_rss,
            "peak_gpu_allocated_bytes": self.peak_allocated,
            "peak_gpu_reserved_bytes": self.peak_reserved,
        }


def data(d, seed, device, save=True):
    start = time.perf_counter()
    c = matched_config(d)
    b = _benchmark(c)
    assert c.to_dict() == read_json(OUT / "configurations.json")[str(d)]["matched"]
    train, _, _, _, test, test_starts = _datasets(_family(b), b, seed)
    starts = _make_starts(train, trials=32, seed=seed + 41001)
    assert len(train) == 240 and len(test) == 15 and all(len(x) == 32 for x in starts)
    tr, h = _aligned(train, starts, device)
    te, ht = _aligned(test, test_starts, device)
    assert (
        len(h) == 7680
        and len(ht) == 30
        and set(tr.goal_index.cpu().tolist()) == {0, 1, 2}
    )
    transform = _exact_transforms(c, ht, seed=seed + 81001)["exact_rotated"]
    rotated = (
        transform_tasks(te, transform),
        transform_points(ht, transform.rotation, transform.translation),
    )
    eq = build_equivariant_components(tr, h)
    direct = direct_features(eq)
    assert eq.invariant.shape[-1] == d + 11 and direct.values.shape[-1] == 2 * d + 11
    stats = {
        "DirectPrepared": descriptor_normalization(direct),
        "Relational": invariant_normalization(eq),
    }
    feature_hashes = {
        k: tensor_hash(v)
        for k, v in {
            "relative": eq.relative,
            "phi": eq.invariant,
            "mask": eq.mask,
            "train_starts": h,
            "test_starts": ht,
            **{
                m + "_" + str(i): x for m, ss in stats.items() for i, x in enumerate(ss)
            },
        }.items()
    }
    if save:
        folder = OUT / "data" / f"d{d}_s{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {
            "train_starts": np.asarray(starts, dtype=np.float32),
            "test_starts": np.asarray(test_starts, dtype=np.float32),
            "exact_rotation": transform.rotation.cpu().numpy(),
            "exact_translation": transform.translation.cpu().numpy(),
        }
        for label, tasks in [("train", train), ("test", test)]:
            unique = pack_tasks(tasks, device=torch.device("cpu"), dtype=torch.float32)
            for f in dataclasses.fields(unique):
                x = getattr(unique, f.name)
                arrays[label + "_" + f.name] = (
                    x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
                )
        path = folder / "records.npz"
        if path.exists():
            with np.load(path) as old:
                assert set(old.files) == set(arrays)
                for key, a in arrays.items():
                    assert np.array_equal(old[key], a), (
                        "Regenerated data mismatch",
                        key,
                    )
            assert (
                read_json(folder / "provenance.json")["feature_hashes"]
                == feature_hashes
            )
        else:
            tmp = folder / "records.tmp.npz"
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)
            write_json(
                folder / "provenance.json",
                {
                    "dimension": d,
                    "seed": seed,
                    "records_sha256": sha(path),
                    "feature_hashes": feature_hashes,
                    "train_tasks": 240,
                    "train_records": 7680,
                    "test_tasks": 15,
                    "test_records": 30,
                    "config_sha256": sha(OUT / "configurations.json"),
                    "generator": "unchanged _datasets, _make_starts, _aligned; source_snapshot",
                },
            )
    synchronize(device)
    return (
        c,
        b,
        tr,
        h,
        eq,
        direct,
        stats,
        {"iid": (te, ht), "exact_rotated": rotated},
        feature_hashes,
        time.perf_counter() - start,
    )


@torch.no_grad()
def evaluate(kind, model, tasks, h, b, c, components=None):
    if components is None:
        if kind in ("HistoricalDirect", "HistoricalWide"):
            components = build_component_descriptors(tasks, h)
        else:
            eq = build_equivariant_components(tasks, h)
            components = direct_features(eq) if kind == "DirectPrepared" else eq
    summary = evaluate_landscape_policy(
        model,
        tasks,
        h,
        components,
        lambda_alpha=c.lambda_alpha,
        refinement_steps=b.base.base.training.refinement_steps,
        refinement_rate=b.base.base.training.refinement_rate,
        useful_threshold=b.base.base.landscape.useful_threshold,
        collect_per_set=False,
        collect_attention_rows=False,
    ).summary
    proposal = model(h, components)
    finite(
        [proposal.raw_direction, proposal.direction, proposal.alpha, proposal.candidate]
    )
    refined = tasks.refine(proposal.candidate, steps=9, rate=0.28)
    region = tasks.regions(refined)
    quality = tasks.quality(refined)
    useful = (region >= 1) & (region <= tasks.n_useful[:, None]) & (quality >= 0.62)
    summary["useful_candidate_fraction"] = float(useful.float().mean().cpu())
    diagnostic = {}
    for name, x in [("raw", proposal.raw_direction), ("direction", proposal.direction)]:
        matrix = x.reshape(-1, c.dimension).detach().cpu()
        sv = torch.linalg.svdvals(matrix)
        tol = max(matrix.shape) * torch.finfo(torch.float32).eps * float(sv[0])
        rank = int((sv > tol).sum())
        summary["output_" + name + "_rank"] = rank
        diagnostic[name] = {
            "shape": list(matrix.shape),
            "singular_values": sv.tolist(),
            "threshold": tol,
            "rank": rank,
        }
    if hasattr(model, "w2"):
        matrix = torch.cat([model.w2, model.b2[:, None]], dim=1).detach().cpu()
        sv = torch.linalg.svdvals(matrix)
        tol = max(matrix.shape) * torch.finfo(torch.float32).eps * float(sv[0])
        summary["decoder_augmented_rank"] = int((sv > tol).sum())
    finite(summary)
    return summary, diagnostic


def verify_features(c, tr, h, eq, direct, stats, seed):
    assert torch.equal(direct.values[:, :, : c.dimension], eq.relative)
    assert torch.equal(direct.values[:, :, c.dimension :], eq.invariant)
    assert torch.equal(direct.mask, eq.mask)
    perm = random_component_permutation(
        eq.mask, generator=torch.Generator(device=h.device).manual_seed(seed + 61102)
    )
    dp = direct.permute(perm)
    ep = eq.permute(perm)
    assert torch.equal(dp.values, direct_features(ep).values) and torch.equal(
        dp.mask, ep.mask
    )
    selected = direct.values[direct.mask]
    assert torch.equal(stats["DirectPrepared"][0], selected.mean(0))
    assert torch.equal(
        stats["DirectPrepared"][1], selected.std(0, correction=0).clamp_min(1e-4)
    )
    assert torch.allclose(
        stats["DirectPrepared"][0][c.dimension :],
        stats["Relational"][0],
        rtol=0,
        atol=1e-5,
    )
    assert torch.allclose(
        stats["DirectPrepared"][1][c.dimension :],
        stats["Relational"][1],
        rtol=0,
        atol=1e-5,
    )
    idx = torch.arange(8, device=h.device)
    small = tr.index_select(idx)
    hs = h[:8]

    class Restricted:
        def __getattr__(self, key):
            assert key in ALLOWED_COMPONENT_INPUT_FIELDS, (
                "Label accessed by feature builder",
                key,
            )
            return getattr(small, key)

    restricted = build_equivariant_components(Restricted(), hs)
    assert torch.allclose(restricted.invariant, eq.invariant[:8], rtol=1e-5, atol=1e-5)
    # Include padding because all-valid IID records cannot reveal masked-value leakage.
    masked = dataclasses.replace(small, component_mask=small.component_mask.clone())
    masked.component_mask[:, -1] = False
    masked_eq = build_equivariant_components(masked, hs)
    masked_d = direct_features(masked_eq)
    assert (
        not masked_d.mask[:, -1].any()
        and torch.count_nonzero(masked_d.values[:, -1]) == 0
    )
    padded_perm = random_component_permutation(
        masked_eq.mask, generator=torch.Generator(device=h.device).manual_seed(919)
    )
    assert torch.equal(
        masked_d.permute(padded_perm).values,
        direct_features(masked_eq.permute(padded_perm)).values,
    )
    records = []
    for kind, comp in [("DirectPrepared", direct), ("Relational", eq)]:
        model = factory(kind, c, seed, h.device, stats[kind])
        _assert_nonoracle_model(model)
        loss, metrics, proposal = landscape_supervised_objective(
            model, hs, small, comp.index_select(idx), lambda_alpha=c.lambda_alpha
        )
        finite([loss, metrics, proposal.raw_direction, proposal.candidate])
        loss.backward()
        finite([p.grad for p in model.parameters() if p.grad is not None])
        # Perturb labels to detect leakage into predictions while confirming the loss uses them.
        changed = dataclasses.replace(small, useful_centers=small.useful_centers + 1)
        changed_loss, _, changed_proposal = landscape_supervised_objective(
            model, hs, changed, comp.index_select(idx), lambda_alpha=c.lambda_alpha
        )
        assert torch.equal(
            proposal.raw_direction, changed_proposal.raw_direction
        ) and not torch.equal(loss, changed_loss)
        records.append(
            {
                "model": kind,
                "initial_loss": float(loss.detach().cpu()),
                "changed_supervision_loss": float(changed_loss.detach().cpu()),
                "finite_gradients": True,
                "parameter_count": sum(p.numel() for p in model.parameters()),
            }
        )
    return {
        "shared_values_masks": True,
        "paired_permutation": True,
        "masked_records": True,
        "normalization_training_records_only": True,
        "feature_builder_allowed_fields_only": True,
        "supervision_changes_loss_not_inputs": True,
        "phi_dimension": c.dimension + 11,
        "direct_dimension": 2 * c.dimension + 11,
        "models": records,
    }


@torch.no_grad()
def training_fit(model, tr, h, comp, c):
    totals = {}
    count = 0
    for start in range(0, len(h), 960):
        idx = torch.arange(start, min(start + 960, len(h)), device=h.device)
        loss, metrics, _ = landscape_supervised_objective(
            model,
            h[idx],
            tr.index_select(idx),
            comp.index_select(idx),
            lambda_alpha=c.lambda_alpha,
        )
        for k, v in {"loss": loss, **metrics}.items():
            totals[k] = totals.get(k, 0.0) + float(v.cpu()) * len(idx)
        count += len(idx)
    result = {k: v / count for k, v in totals.items()}
    finite(result)
    return result


def historic(seed, d, b, c, splits, device, gate=False):
    rows = []
    validation = []
    archived = list(csv.DictReader((ORIGINAL / f"d{d}/per_seed.csv").open()))
    for kind, label, source in [
        (
            "HistoricalDirect",
            "CurrentAttention",
            ORIGINAL / f"d{d}/models/CurrentAttention_seed_{seed}.pt",
        ),
        (
            "Relational",
            "EquivariantAttention",
            ORIGINAL / f"d{d}/models/EquivariantAttention_seed_{seed}.pt",
        ),
        (
            "HistoricalWide",
            "DirectWide",
            WIDE / f"d{d}/models/DirectWide_seed_{seed}.pt",
        ),
    ]:
        if not source.exists():
            validation.append(
                {
                    "condition": label,
                    "status": "missing_checkpoint",
                    "source": str(source),
                }
            )
            continue
        payload = torch.load(source, map_location=device, weights_only=True)
        state = payload["state_dict"]
        stats = (
            (state["invariant_mean"], state["invariant_scale"])
            if kind == "Relational"
            else (state["descriptor_mean"], state["descriptor_scale"])
        )
        model = factory(kind, c, seed, device, stats)
        model.load_state_dict(state, strict=True)
        model.eval()
        for split, (tasks, h) in splits.items():
            metrics, rank = evaluate(kind, model, tasks, h, b, c)
            row = {
                "dimension": d,
                "seed": seed,
                "condition": label,
                "split": split,
                "context": "historical_saved_checkpoint_replay",
                "checkpoint_sha256": sha(source),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                **metrics,
            }
            rows.append(row)
            if kind != "HistoricalWide":
                reference = [
                    r
                    for r in archived
                    if int(r["seed"]) == seed
                    and r["condition"] == label
                    and r["split"] == split
                ][0]
                differences = {
                    k: metrics[k] - float(reference[k])
                    for k in metrics
                    if k in reference
                }
                direct_path = _evaluate(label, model, tasks, h, b, c)
                assert all(metrics[k] == v for k, v in direct_path.items()), (
                    "Evaluator wrapper disagrees with original evaluator"
                )
                coverage_ok = abs(differences["mode_set_recall"]) < 1e-6
                # Near-one acos amplifies rounding, so angular error needs a looser comparison bound.
                numerical_ok = all(
                    abs(v) < (0.02 if k == "mean_angular_error_degrees" else 2e-5)
                    for k, v in differences.items()
                )
                validation.append(
                    {
                        "condition": label,
                        "split": split,
                        "coverage_match": coverage_ok,
                        "numerical_match": numerical_ok,
                        "deltas": differences,
                        "archive_coverage": float(reference["mode_set_recall"]),
                        "replayed_coverage": metrics["mode_set_recall"],
                    }
                )
                if gate:
                    assert coverage_ok and numerical_ok, (
                        "Historical evaluation mismatch",
                        validation[-1],
                    )
    return rows, validation


def verify(device):
    check_lock()
    out = []
    # Reuse seeded test records to isolate runtime differences from training and task sampling.
    for d in (32, 64):
        c = matched_config(d)
        b = _benchmark(c)
        _, _, _, _, test, starts = _datasets(_family(b), b, 7)
        te, h = _aligned(test, starts, device)
        t = _exact_transforms(c, h, seed=81008)["exact_rotated"]
        splits = {"iid": (te, h)}
        if device.type == "cuda":
            splits["exact_rotated"] = (
                transform_tasks(te, t),
                transform_points(h, t.rotation, t.translation),
            )
        rows, validation = historic(7, d, b, c, splits, device, gate=True)
        out.append({"dimension": d, "validation": validation, "metrics": rows})
    write_json(
        OUT / "verification" / f"historical_replay_{device.type}.json",
        {
            "utc": now(),
            "runtime": runtime(device),
            "results": out,
            "passed": True,
            "rotation_note": "CPU audit uses IID only: CPU and GPU RNGs produce different exact-rotation draws. GPU replay uses the original CUDA transform stream.",
        },
    )
    print(now(), "Historical saved-checkpoint replay passed", device, flush=True)


def job_valid(folder, key):
    if not folder.exists():
        return False
    completion = read_json(folder / "completion.json")
    assert completion["key"] == key and completion["lock_sha256"] == sha(
        OUT / "lock.json"
    )
    assert completion["steps"] == 1500 and completion["status"] == "complete"
    assert set(completion["files"]) == {
        "checkpoint.pt",
        "metadata.json",
        "metrics.json",
        "rank_diagnostics.json",
        "training_fit.json",
        "curves.json",
        "configuration.json",
        "verification.json",
    }
    for rel, digest in completion["files"].items():
        assert sha(folder / rel) == digest, ("Incomplete/changed job", key, rel)
    assert len(read_json(folder / "metrics.json")) == 2
    metadata = read_json(folder / "metadata.json")
    data_folder = OUT / "data" / f"d{metadata['dimension']}_s{metadata['seed']}"
    provenance = read_json(data_folder / "provenance.json")
    assert sha(data_folder / "records.npz") == provenance["records_sha256"]
    assert provenance["feature_hashes"] == metadata["feature_hashes"]
    return True


def run_pair(d, seed, device, pilot=False):
    check_lock()
    pair_start = time.perf_counter()
    folders = {m: OUT / "jobs" / f"d{d}" / f"seed_{seed}" / m for m in MODELS}
    needed = []
    for m, folder in folders.items():
        key = f"d{d}_s{seed}_{m}"
        if job_valid(folder, key):
            update_manifest(key, status="complete", reverified_utc=now())
        else:
            needed.append(m)
    if not needed:
        return None
    with Monitor(device) as monitor:
        c, b, tr, h, eq, direct, stats, splits, feature_hashes, prep_seconds = data(
            d, seed, device
        )
        checks = verify_features(c, tr, h, eq, direct, stats, seed)
        write_json(OUT / "verification" / f"d{d}_s{seed}_features.json", checks)
        monitor.sample()
        historical_rows, historical_validation = historic(
            seed, d, b, c, splits, device, gate=pilot
        )
        write_json(
            OUT / "historical" / f"d{d}_s{seed}.json",
            {"rows": historical_rows, "validation": historical_validation},
        )
        for kind in needed:
            key = f"d{d}_s{seed}_{kind}"
            start = time.perf_counter()
            stage = OUT / "attempts" / f"{key}_{uuid.uuid4().hex[:8]}"
            stage.mkdir(parents=True)
            update_manifest(
                key,
                status="running",
                started_utc=now(),
                pid=os.getpid(),
                attempt=str(stage.relative_to(OUT)),
            )
            print(now(), "START", key, flush=True)
            try:
                model = factory(kind, c, seed, device, stats[kind])
                comp = direct if kind == "DirectPrepared" else eq
                result = train_symmetry_policy(
                    model,
                    tr,
                    h,
                    comp,
                    component_kind="current"
                    if kind == "DirectPrepared"
                    else "equivariant",
                    augmented=False,
                    rotation_pool=torch.empty(0, d, d, device=device),
                    translation_radius_max=c.augmentation.translation_radius_max,
                    steps=1500,
                    batch_size=960,
                    learning_rate=0.001,
                    weight_decay=0.0001,
                    gradient_clip=1.0,
                    lambda_alpha=0.1,
                    seed=seed + 61101,
                    curve_steps=c.curve_steps,
                )
                finite(result.history)
                finite(list(model.parameters()))
                assert result.history[-1]["step"] == 1500
                model.eval()
                fit = training_fit(model, tr, h, comp, c)
                metadata = {
                    "dimension": d,
                    "seed": seed,
                    "model": kind,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "steps": 1500,
                    "batch_size": 960,
                    "examples_seen": result.examples_seen,
                    "optimizer_seconds": result.elapsed_seconds,
                    "training_peak_allocated_bytes": result.peak_allocated_bytes,
                    "training_peak_reserved_bytes": result.peak_reserved_bytes,
                    "feature_hashes": feature_hashes,
                    "lock_sha256": sha(OUT / "lock.json"),
                    "pilot": pilot,
                    "runtime": json.loads(
                        json.dumps(runtime_info(device, torch.float32).to_dict())
                    ),
                }
                torch.save(
                    {
                        "state_dict": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "history": result.history,
                        "metadata": metadata,
                    },
                    stage / "checkpoint.pt",
                )
                loaded = factory(kind, c, seed, device, stats[kind])
                loaded.load_state_dict(
                    torch.load(
                        stage / "checkpoint.pt", map_location=device, weights_only=True
                    )["state_dict"]
                )
                loaded.eval()
                metrics = []
                ranks = {}
                for split, (tasks, hs) in splits.items():
                    metric, rank = evaluate(kind, model, tasks, hs, b, c)
                    replayed, _ = evaluate(kind, loaded, tasks, hs, b, c)
                    assert metric == replayed, (
                        "Checkpoint replay changed metrics",
                        key,
                        split,
                    )
                    metrics.append(
                        {
                            "dimension": d,
                            "seed": seed,
                            "model": kind,
                            "split": split,
                            "status": "complete",
                            "parameter_count": metadata["parameter_count"],
                            **metric,
                            **{"train_" + k: v for k, v in fit.items()},
                            "optimizer_seconds": result.elapsed_seconds,
                            "training_peak_allocated_bytes": result.peak_allocated_bytes,
                            "training_peak_reserved_bytes": result.peak_reserved_bytes,
                        }
                    )
                    ranks[split] = rank
                synchronize(device)
                metadata["job_wall_seconds"] = time.perf_counter() - start
                for row in metrics:
                    row["job_wall_seconds"] = metadata["job_wall_seconds"]
                write_json(stage / "metadata.json", metadata)
                write_json(stage / "metrics.json", metrics)
                write_json(stage / "rank_diagnostics.json", ranks)
                write_json(stage / "training_fit.json", fit)
                write_json(stage / "curves.json", result.history)
                write_json(
                    stage / "configuration.json",
                    {
                        "matched": c.to_dict(),
                        "architecture": read_json(OUT / "configurations.json")[str(d)][
                            "direct_prepared"
                        ]
                        if kind == "DirectPrepared"
                        else {
                            "hidden": c.invariant_component_hidden_dimension,
                            "token": 32,
                        },
                    },
                )
                write_json(
                    stage / "verification.json",
                    {
                        "checkpoint_round_trip_exact_metrics": True,
                        "finite_parameters_and_logged_loss_gradients": True,
                        "feature_checks": checks,
                    },
                )
                hashes = {
                    str(p.relative_to(stage)): sha(p)
                    for p in stage.iterdir()
                    if p.is_file()
                }
                write_json(
                    stage / "completion.json",
                    {
                        "key": key,
                        "status": "complete",
                        "steps": 1500,
                        "completed_utc": now(),
                        "files": hashes,
                        "lock_sha256": sha(OUT / "lock.json"),
                    },
                )
                folders[kind].parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage, folders[kind])
                assert job_valid(folders[kind], key)
                update_manifest(
                    key,
                    status="complete",
                    completed_utc=now(),
                    path=str(folders[kind].relative_to(OUT)),
                    job_wall_seconds=metadata["job_wall_seconds"],
                )
                print(
                    now(),
                    "COMPLETE",
                    key,
                    "IID coverage",
                    metrics[0]["mode_set_recall"],
                    "wall seconds",
                    metadata["job_wall_seconds"],
                    flush=True,
                )
                del model, loaded
                gc.collect()
                torch.cuda.empty_cache()
            except BaseException:
                failure = {
                    "utc": now(),
                    "key": key,
                    "traceback": traceback.format_exc(),
                }
                write_json(stage / "failure.json", failure)
                update_manifest(key, status="failed", failure=failure)
                raise
        monitor.sample()
    record = {
        "dimension": d,
        "seed": seed,
        "pilot": pilot,
        "pair_wall_seconds": time.perf_counter() - pair_start,
        "preparation_and_feature_seconds": prep_seconds,
        **monitor.result(),
        "disk_bytes": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()),
    }
    write_json(OUT / "timing" / f"d{d}_s{seed}.json", record)
    del tr, h, eq, direct, stats, splits
    gc.collect()
    torch.cuda.empty_cache()
    return record


def run_pilot(device):
    start = time.perf_counter()
    verify(device)
    records = []
    for d in (32, 64):
        row = run_pair(d, 7, device, pilot=True)
        records.append(row or read_json(OUT / "timing" / f"d{d}_s7.json"))
    assert all(
        job_valid(OUT / "jobs" / f"d{d}" / "seed_7" / m, f"d{d}_s7_{m}")
        for d in (32, 64)
        for m in MODELS
    )
    eta = sum(r["pair_wall_seconds"] * 29 for r in records)
    summary = {
        "passed": True,
        "lock_sha256": sha(OUT / "lock.json"),
        "completed_utc": now(),
        "total_pilot_wall_seconds": time.perf_counter() - start,
        "includes": "GPU historical verification, data preparation, features, checks, historical context, training, fit, evaluation, serialization",
        "pairs": records,
        "remaining_estimated_seconds": eta,
        "estimate_range_seconds": [eta * 0.8, eta * 1.5],
        "runtime": runtime(device),
        "disk_free_bytes": shutil.disk_usage(OUT).free,
    }
    assert summary["disk_free_bytes"] > 10 * 1024**3, "Insufficient disk margin"
    write_json(OUT / "pilot.json", summary)
    update_manifest(status="pilot_passed", pilot=summary)
    print(now(), "PILOT PASSED; estimated remaining seconds", eta, flush=True)
    return summary


def wait_for_external_training():
    while True:
        running = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            name = (p.info["name"] or "").lower()
            cmd = " ".join(p.info["cmdline"] or []).lower()
            if "hurricanegputrain" in name or "run_mlp_equal_time_experiment.py" in cmd:
                running.append(p.info)
        if not running:
            break
        update_manifest(status="waiting_for_other_training", external_training=running)
        print(
            now(),
            "Waiting for existing RocketLeague training without interrupting it",
            [p["pid"] for p in running],
            flush=True,
        )
        if (OUT / "STOP_AFTER_JOB").exists():
            raise SystemExit("Stop requested while waiting")
        time.sleep(30)


def pipeline():
    check_lock()
    wait_for_external_training()
    device = configure_runtime("cuda")
    verify_runtime_identity(runtime(device))
    pilot = (
        read_json(OUT / "pilot.json")
        if (OUT / "pilot.json").exists()
        else run_pilot(device)
    )
    assert pilot["passed"] and pilot["lock_sha256"] == sha(OUT / "lock.json")
    assert all(
        job_valid(OUT / "jobs" / f"d{d}" / "seed_7" / m, f"d{d}_s7_{m}")
        for d in (32, 64)
        for m in MODELS
    )
    update_manifest(status="batch_running", worker_pid=os.getpid())
    for d in (32, 64):
        for seed in matched_config(d).seeds:
            if (OUT / "STOP_AFTER_JOB").exists():
                update_manifest(status="stopped_between_pairs")
                return
            run_pair(d, seed, device)
            from analyze import analyze

            analyze(require_complete=False)
    from analyze import analyze

    analyze(require_complete=True)
    update_manifest(status="complete", completed_utc=now())
    print(now(), "ALL 120 MODELS COMPLETE; report written", flush=True)


def acquire_worker_lock():
    path = OUT / "worker.lock.json"
    if path.exists():
        old = read_json(path)
        try:
            process = psutil.Process(old["pid"])
            if abs(process.create_time() - old["create_time"]) < 1:
                raise RuntimeError(
                    f"Worker already active with PID {old['pid']}; no concurrent launch allowed"
                )
        except psutil.NoSuchProcess:
            pass
        os.replace(path, OUT / f"worker.lock.stale.{uuid.uuid4().hex[:8]}.json")
    with path.open("x", encoding="utf-8") as f:
        json.dump(
            {
                "pid": os.getpid(),
                "create_time": psutil.Process().create_time(),
                "utc": now(),
            },
            f,
        )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["freeze", "verify-cpu", "pilot", "run", "status", "analyze"]
    )
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
        return
    if args.phase == "status":
        m = read_json(OUT / "manifest.json")
        counts = {}
        for x in m["jobs"].values():
            counts[x["status"]] = counts.get(x["status"], 0) + 1
        print(
            json.dumps(
                {
                    "status": m["status"],
                    "counts": counts,
                    "updated_utc": m.get("updated_utc"),
                    "running": {
                        k: v for k, v in m["jobs"].items() if v["status"] == "running"
                    },
                },
                indent=2,
            )
        )
        return
    check_lock()
    if args.phase == "verify-cpu":
        verify(configure_runtime("cpu"))
        return
    if args.phase == "analyze":
        from analyze import analyze

        analyze(require_complete=True)
        return
    lock = acquire_worker_lock()
    try:
        if args.phase == "pilot":
            run_pilot(configure_runtime("cuda"))
        else:
            pipeline()
    except BaseException:
        write_json(
            OUT / "worker_failure.json",
            {"utc": now(), "traceback": traceback.format_exc()},
        )
        update_manifest(status="failed_stopped")
        raise
    finally:
        if lock.exists():
            os.replace(lock, OUT / f"worker.lock.ended.{uuid.uuid4().hex[:8]}.json")


if __name__ == "__main__":
    main()
