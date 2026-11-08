"""Train and evaluate two Direct feature-width configurations using the preserved pipeline."""

from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIOR = ROOT / "prepared_direct_comparison_20260921"
sys.path.insert(0, str(PRIOR))
import experiment as prior
import psutil
import torch

KINDS = ("DirectPreparedNarrow", "DirectRawWide")
STEPS = 1500


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return prior.sha(path)


def write(path, obj):
    prior.write_json(path, obj)


def read(path):
    return prior.read_json(path)


def key(d, s, kind):
    return f"d{d}_s{s}_{kind}"


def folder(d, s, kind):
    return HERE / "jobs" / f"d{d}" / f"seed_{s}" / kind


def files_to_lock():
    return [
        HERE / "PLAN.md",
        HERE / "run.py",
        PRIOR / "lock.json",
        PRIOR / "experiment.py",
        PRIOR / "source_snapshot" / "gen3_gpu" / "config_landscape_representation.py",
    ]


def check_lock():
    prior.check_lock()
    lock = read(HERE / "lock.json")
    assert lock["source_hashes"] == {
        str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in files_to_lock()
    }, "Source or plan changed"
    assert lock["prior_lock_sha256"] == digest(PRIOR / "lock.json")
    for old_hash, record in lock.get("compatible_prior_locks", {}).items():
        assert digest(HERE / record["lock_path"]) == old_hash
        previous = read(HERE / record["lock_path"])
        assert (
            digest(HERE / record["source_path"])
            == previous["source_hashes"][
                "direct_feature_width_ablation_20261025/run.py"
            ]
        )
        assert all(
            previous["source_hashes"][p] == lock["source_hashes"][p]
            for p in previous["source_hashes"]
            if p != "direct_feature_width_ablation_20261025/run.py"
        )
    return lock


def freeze():
    assert not (HERE / "lock.json").exists(), "Already initialized"
    prior.check_lock()
    seeds = list(prior.matched_config(32).seeds)
    assert seeds == list(prior.matched_config(64).seeds) and len(seeds) == 30
    base = prior.read_json(PRIOR / "configurations.json")
    for d in (32, 64):
        c = prior.matched_config(d)
        assert c.to_dict() == base[str(d)]["matched"]
        for s in seeds:
            for old in ("DirectPrepared", "Relational"):
                assert prior.job_valid(
                    PRIOR / "jobs" / f"d{d}" / f"seed_{s}" / old, f"d{d}_s{s}_{old}"
                )
    config = {
        "dimensions": [32, 64],
        "seeds": seeds,
        "steps": STEPS,
        "models": {
            "DirectPreparedNarrow": {
                "features": "r_phi",
                "hidden": 64,
                "token": 32,
                "proposal": 16,
            },
            "DirectRawWide": {
                "features": "raw",
                "hidden": 256,
                "token": 128,
                "proposal": 256,
            },
        },
        "comparison": "retrospective_two_missing_cells",
        "prior_dir": str(PRIOR),
        "prior_config_sha256": digest(PRIOR / "configurations.json"),
    }
    write(HERE / "configuration.json", config)
    write(
        HERE / "lock.json",
        {
            "created_utc": stamp(),
            "source_hashes": {
                str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
                for p in files_to_lock()
            },
            "prior_lock_sha256": digest(PRIOR / "lock.json"),
            "configuration_sha256": digest(HERE / "configuration.json"),
        },
    )
    write(
        HERE / "manifest.json",
        {
            "created_utc": stamp(),
            "status": "awaiting_pilot",
            "jobs": {
                key(d, s, k): {"status": "pending"}
                for d in (32, 64)
                for s in seeds
                for k in KINDS
            },
        },
    )


def update(k=None, **values):
    m = read(HERE / "manifest.json")
    if k:
        m["jobs"][k].update(values)
    else:
        m.update(values)
    m["updated_utc"] = stamp()
    write(HERE / "manifest.json", m)


def valid(d, s, kind):
    dest = folder(d, s, kind)
    if not dest.exists():
        return False
    record = read(dest / "completion.json")
    assert (
        record["key"] == key(d, s, kind)
        and record["status"] == "complete"
        and record["steps"] == STEPS
    )
    current = read(HERE / "lock.json")
    assert record["lock_sha256"] == digest(HERE / "lock.json") or record[
        "lock_sha256"
    ] in current.get("compatible_prior_locks", {})
    assert set(record["files"]) == {
        "checkpoint.pt",
        "configuration.json",
        "curves.json",
        "metadata.json",
        "metrics.json",
        "training_fit.json",
        "rank_diagnostics.json",
        "verification.json",
    }
    for name, h in record["files"].items():
        assert digest(dest / name) == h, (dest, name)
    assert len(read(dest / "metrics.json")) == 2
    return True


def model_for(kind, c, seed, device, stats):
    width = {"DirectPreparedNarrow": (64, 32, 16), "DirectRawWide": (256, 128, 256)}[
        kind
    ]
    return prior.LandscapeAttentionPolicy(
        dimension=c.dimension,
        candidates=c.candidates,
        latent_dimension=c.latent_dimension,
        task_token_dimension=c.task_token_dimension,
        alpha_max=c.alpha_max,
        seed=seed + 10020,
        device=device,
        descriptor_dimension=stats[0].numel(),
        component_hidden_dimension=width[0],
        component_token_dimension=width[1],
        proposal_hidden_dimension=width[2],
        descriptor_mean=stats[0],
        descriptor_scale=stats[1],
    )


def external_gpu_job():
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or []).lower()
            if p.pid != os.getpid() and (
                "train.py --device cuda" in cmd
                or "run_mlp_equal_time_experiment.py" in cmd
            ):
                found.append({"pid": p.pid, "command": cmd[:300]})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return found


def prepare(d, s, device):
    c, b, tr, h, eq, prepared, _, splits, feature_hashes, prep_seconds = prior.data(
        d, s, device, save=False
    )
    old = read(PRIOR / "data" / f"d{d}_s{s}" / "provenance.json")
    assert feature_hashes == old["feature_hashes"], (
        "Tasks or prepared features differ from completed study"
    )
    raw = prior.build_component_descriptors(tr, h)
    assert raw.values.shape[-1] == d + 1 + d * (d + 1) // 2
    assert prepared.values.shape[-1] == 2 * d + 11
    assert torch.equal(raw.mask, prepared.mask)
    raw_stats = prior.descriptor_normalization(raw)
    prep_stats = prior.descriptor_normalization(prepared)
    expected = prepared.values[prepared.mask]
    assert torch.equal(prep_stats[0], expected.mean(0))
    assert torch.equal(prep_stats[1], expected.std(0, correction=0).clamp_min(1e-4))
    assert torch.equal(prep_stats[0], prior.descriptor_normalization(prepared)[0])
    perm = prior.random_component_permutation(
        raw.mask, generator=torch.Generator(device=device).manual_seed(s + 61102)
    )
    assert torch.equal(
        raw.permute(perm).values,
        torch.gather(raw.values, 1, perm.unsqueeze(-1).expand_as(raw.values)),
    )
    assert torch.equal(
        prepared.permute(perm).values,
        torch.gather(prepared.values, 1, perm.unsqueeze(-1).expand_as(prepared.values)),
    )
    return (
        c,
        b,
        tr,
        h,
        splits,
        raw,
        prepared,
        raw_stats,
        prep_stats,
        feature_hashes,
        prep_seconds,
    )


def preflight(c, tr, h, raw, prepared, raw_stats, prep_stats, seed):
    idx = torch.arange(8, device=h.device)
    checks = []
    for kind, comp, stats in [
        ("DirectPreparedNarrow", prepared, prep_stats),
        ("DirectRawWide", raw, raw_stats),
    ]:
        model = model_for(kind, c, seed, h.device, stats)
        prior._assert_nonoracle_model(model)
        loss, metrics, p = model_loss(
            model, h[idx], tr.index_select(idx), comp.index_select(idx), c
        )
        prior.finite([loss, metrics, p.raw_direction, p.candidate])
        loss.backward()
        prior.finite([v.grad for v in model.parameters() if v.grad is not None])
        changed = prior.dataclasses.replace(
            tr.index_select(idx), useful_centers=tr.index_select(idx).useful_centers + 1
        )
        other, _, p2 = model_loss(model, h[idx], changed, comp.index_select(idx), c)
        assert torch.equal(p.raw_direction, p2.raw_direction) and not torch.equal(
            loss, other
        )
        checks.append(
            {
                "kind": kind,
                "parameter_count": sum(v.numel() for v in model.parameters()),
                "finite_gradients": True,
                "labels_only_in_loss": True,
            }
        )
    return checks


def model_loss(model, h, tr, comp, c):
    return prior.landscape_supervised_objective(
        model, h, tr, comp, lambda_alpha=c.lambda_alpha
    )


def run_pair(d, s, device, pilot=False):
    check_lock()
    needed = [kind for kind in KINDS if not valid(d, s, kind)]
    if not needed:
        return None
    started = time.perf_counter()
    with prior.Monitor(device) as monitor:
        (
            c,
            b,
            tr,
            h,
            splits,
            raw,
            prepared,
            raw_stats,
            prep_stats,
            feature_hashes,
            prep_seconds,
        ) = prepare(d, s, device)
        checks = preflight(c, tr, h, raw, prepared, raw_stats, prep_stats, s)
        for kind in needed:
            k = key(d, s, kind)
            begin = time.perf_counter()
            attempt = HERE / "attempts" / (k + "_" + uuid.uuid4().hex[:8])
            attempt.mkdir(parents=True)
            update(
                k,
                status="running",
                started_utc=stamp(),
                pid=os.getpid(),
                attempt=str(attempt.relative_to(HERE)),
            )
            print(stamp(), "START", k, flush=True)
            try:
                comp, stats = (
                    (prepared, prep_stats)
                    if kind == "DirectPreparedNarrow"
                    else (raw, raw_stats)
                )
                model = model_for(kind, c, s, device, stats)
                result = prior.train_symmetry_policy(
                    model,
                    tr,
                    h,
                    comp,
                    component_kind="current",
                    augmented=False,
                    rotation_pool=torch.empty(0, d, d, device=device),
                    translation_radius_max=c.augmentation.translation_radius_max,
                    steps=STEPS,
                    batch_size=960,
                    learning_rate=0.001,
                    weight_decay=0.0001,
                    gradient_clip=1.0,
                    lambda_alpha=0.1,
                    seed=s + 61101,
                    curve_steps=c.curve_steps,
                )
                assert result.history[-1]["step"] == STEPS
                prior.finite(result.history)
                prior.finite(list(model.parameters()))
                model.eval()
                fit = prior.training_fit(model, tr, h, comp, c)
                meta = {
                    "dimension": d,
                    "seed": s,
                    "model": kind,
                    "parameter_count": sum(v.numel() for v in model.parameters()),
                    "optimizer_seconds": result.elapsed_seconds,
                    "training_peak_allocated_bytes": result.peak_allocated_bytes,
                    "training_peak_reserved_bytes": result.peak_reserved_bytes,
                    "feature_hashes": feature_hashes,
                    "pilot": pilot,
                    "steps": STEPS,
                    "runtime": json.loads(
                        json.dumps(prior.runtime_info(device, torch.float32).to_dict())
                    ),
                }
                torch.save(
                    {
                        "state_dict": {
                            name: v.detach().cpu()
                            for name, v in model.state_dict().items()
                        },
                        "history": result.history,
                        "metadata": meta,
                    },
                    attempt / "checkpoint.pt",
                )
                loaded = model_for(kind, c, s, device, stats)
                loaded.load_state_dict(
                    torch.load(
                        attempt / "checkpoint.pt",
                        map_location=device,
                        weights_only=True,
                    )["state_dict"]
                )
                loaded.eval()
                rows = []
                rank = {}
                alias = (
                    "DirectPrepared"
                    if kind == "DirectPreparedNarrow"
                    else "HistoricalDirect"
                )
                for split, (tasks, starts) in splits.items():
                    metric, spectrum = prior.evaluate(alias, model, tasks, starts, b, c)
                    replay, _ = prior.evaluate(alias, loaded, tasks, starts, b, c)
                    assert metric == replay, "Checkpoint replay mismatch"
                    rows.append(
                        {
                            "dimension": d,
                            "seed": s,
                            "model": kind,
                            "split": split,
                            "status": "complete",
                            "parameter_count": meta["parameter_count"],
                            **metric,
                            **{"train_" + name: value for name, value in fit.items()},
                            "optimizer_seconds": result.elapsed_seconds,
                            "training_peak_allocated_bytes": result.peak_allocated_bytes,
                            "training_peak_reserved_bytes": result.peak_reserved_bytes,
                        }
                    )
                    rank[split] = spectrum
                prior.synchronize(device)
                meta["job_wall_seconds"] = time.perf_counter() - begin
                for row in rows:
                    row["job_wall_seconds"] = meta["job_wall_seconds"]
                write(attempt / "metadata.json", meta)
                write(attempt / "metrics.json", rows)
                write(attempt / "rank_diagnostics.json", rank)
                write(attempt / "training_fit.json", fit)
                write(attempt / "curves.json", result.history)
                write(
                    attempt / "configuration.json",
                    {
                        "matched": c.to_dict(),
                        "widths": {"hidden": 64, "token": 32, "proposal": 16}
                        if kind == "DirectPreparedNarrow"
                        else {"hidden": 256, "token": 128, "proposal": 256},
                        "features": "r_phi"
                        if kind == "DirectPreparedNarrow"
                        else "raw",
                    },
                )
                write(
                    attempt / "verification.json",
                    {
                        "preflight": checks,
                        "checkpoint_round_trip": True,
                        "train_feature_hashes_match_prior": True,
                    },
                )
                hashes = {p.name: digest(p) for p in attempt.iterdir() if p.is_file()}
                write(
                    attempt / "completion.json",
                    {
                        "key": k,
                        "status": "complete",
                        "steps": STEPS,
                        "files": hashes,
                        "lock_sha256": digest(HERE / "lock.json"),
                        "completed_utc": stamp(),
                    },
                )
                dest = folder(d, s, kind)
                dest.parent.mkdir(parents=True, exist_ok=True)
                prior.atomic_replace(attempt, dest)
                assert valid(d, s, kind)
                update(
                    k,
                    status="complete",
                    completed_utc=stamp(),
                    job_wall_seconds=meta["job_wall_seconds"],
                )
                print(
                    stamp(),
                    "COMPLETE",
                    k,
                    "IID coverage",
                    rows[0]["mode_set_recall"],
                    flush=True,
                )
                del model, loaded
                gc.collect()
                torch.cuda.empty_cache()
            except BaseException:
                failure = {
                    "key": k,
                    "utc": stamp(),
                    "traceback": traceback.format_exc(),
                }
                write(attempt / "failure.json", failure)
                update(k, status="failed", failure=failure)
                raise
    record = {
        "dimension": d,
        "seed": s,
        "pilot": pilot,
        "pair_wall_seconds": time.perf_counter() - started,
        "preparation_and_feature_seconds": prep_seconds,
        **monitor.result(),
    }
    write(HERE / "timing" / f"d{d}_s{s}.json", record)
    return record


def pilot(device):
    began = time.perf_counter()
    records = []
    for d in (32, 64):
        records.append(
            run_pair(d, 7, device, pilot=True)
            or read(HERE / "timing" / f"d{d}_s7.json")
        )
    assert all(valid(d, 7, k) for d in (32, 64) for k in KINDS)
    estimate = 29 * sum(r["pair_wall_seconds"] for r in records)
    result = {
        "passed": True,
        "completed_utc": stamp(),
        "total_wall_seconds": time.perf_counter() - began,
        "includes": "data regeneration, feature construction, checks, training, fit, evaluation, checkpoint replay and serialization",
        "pairs": records,
        "remaining_estimated_seconds": estimate,
        "runtime": prior.runtime(device),
        "disk_free_bytes": shutil.disk_usage(HERE).free,
        "lock_sha256": digest(HERE / "lock.json"),
    }
    assert result["disk_free_bytes"] > 5 * 1024**3
    write(HERE / "pilot.json", result)
    update(status="pilot_passed")
    print(stamp(), "PILOT PASSED; remaining estimate seconds", estimate, flush=True)


def analyze(complete=True):
    import statistics

    lock = check_lock()
    cfg = read(HERE / "configuration.json")
    rows = []
    for d in cfg["dimensions"]:
        for s in cfg["seeds"]:
            for k in KINDS:
                if valid(d, s, k):
                    rows.extend(read(folder(d, s, k) / "metrics.json"))
                elif complete:
                    raise RuntimeError(f"Incomplete {key(d, s, k)}")
    if rows:
        prior.write_csv(HERE / "per_seed.csv", rows)
    summary = []
    for d in cfg["dimensions"]:
        for split in ("iid", "exact_rotated"):
            for k in KINDS:
                selected = [
                    r
                    for r in rows
                    if r["dimension"] == d and r["split"] == split and r["model"] == k
                ]
                if not selected:
                    continue
                vals = [r["mode_set_recall"] for r in selected]
                summary.append(
                    {
                        "dimension": d,
                        "split": split,
                        "model": k,
                        "n": len(vals),
                        "mean_coverage": statistics.mean(vals),
                        "sd_coverage": statistics.stdev(vals) if len(vals) > 1 else "",
                        "mean_matched_cosine": statistics.mean(
                            r["mean_matched_cosine"] for r in selected
                        ),
                        "mean_useful_fraction": statistics.mean(
                            r["useful_candidate_fraction"] for r in selected
                        ),
                        "mean_decoy_fraction": statistics.mean(
                            r["candidate_decoy_rate"] for r in selected
                        ),
                        "parameter_count": selected[0]["parameter_count"],
                    }
                )
    if summary:
        prior.write_csv(HERE / "comparison_summary.csv", summary)
    if complete:
        context = []
        for row in csv.DictReader(
            (PRIOR / "historical_context.csv").open(newline="", encoding="utf-8")
        ):
            if row["condition"] == "CurrentAttention" and row["split"] in (
                "iid",
                "exact_rotated",
            ):
                context.append(
                    {
                        "dimension": int(row["dimension"]),
                        "seed": int(row["seed"]),
                        "model": "OriginalDirect_historical",
                        "split": row["split"],
                        "mode_set_recall": float(row["mode_set_recall"]),
                        "source": "historical_saved_checkpoint_replay",
                    }
                )
        for row in csv.DictReader(
            (PRIOR / "per_seed.csv").open(newline="", encoding="utf-8")
        ):
            if row["model"] == "DirectPrepared" and row["split"] in (
                "iid",
                "exact_rotated",
            ):
                context.append(
                    {
                        "dimension": int(row["dimension"]),
                        "seed": int(row["seed"]),
                        "model": "DirectPreparedWide_prior",
                        "split": row["split"],
                        "mode_set_recall": float(row["mode_set_recall"]),
                        "source": "completed_prepared_direct_comparison",
                    }
                )
        assert len(context) == 240
        prior.write_csv(HERE / "prior_context.csv", context)
        import random

        rng = random.Random(20261025)
        contrasts = []
        lookup = {
            (r["dimension"], r["seed"], r["model"], r["split"]): r["mode_set_recall"]
            for r in rows + context
        }
        comparisons = [
            (
                "PreparedNarrow_minus_OriginalDirect",
                "DirectPreparedNarrow",
                "OriginalDirect_historical",
            ),
            (
                "RawWide_minus_OriginalDirect",
                "DirectRawWide",
                "OriginalDirect_historical",
            ),
            (
                "PreparedWide_minus_PreparedNarrow",
                "DirectPreparedWide_prior",
                "DirectPreparedNarrow",
            ),
            ("PreparedWide_minus_RawWide", "DirectPreparedWide_prior", "DirectRawWide"),
        ]
        for d in cfg["dimensions"]:
            for split in ("iid", "exact_rotated"):
                for label, a, b in comparisons:
                    values = [
                        float(lookup[d, s, a, split]) - float(lookup[d, s, b, split])
                        for s in cfg["seeds"]
                    ]
                    boots = sorted(
                        sum(values[rng.randrange(30)] for _ in range(30)) / 30
                        for _ in range(20000)
                    )
                    contrasts.append(
                        {
                            "dimension": d,
                            "split": split,
                            "contrast": label,
                            "n_paired_seeds": 30,
                            "mean_difference": statistics.mean(values),
                            "sample_sd": statistics.stdev(values),
                            "ci95_low": boots[500],
                            "ci95_high": boots[19499],
                            "ci_method": "paired_seed_percentile_bootstrap_20000",
                            "caution": "historical raw/narrow is not contemporaneous"
                            if b == "OriginalDirect_historical"
                            else "same prior protocol, different run dates",
                        }
                    )
        prior.write_csv(HERE / "contrasts.csv", contrasts)
        write(
            HERE / "analysis_complete.json",
            {
                "utc": stamp(),
                "new_models": 120,
                "context_rows": len(context),
                "contrasts": len(contrasts),
            },
        )
        update(status="complete", completed_utc=stamp())
        print(stamp(), "ALL 120 NEW MODELS COMPLETE", flush=True)


def status():
    m = read(HERE / "manifest.json")
    counts = {}
    for v in m["jobs"].values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print(
        json.dumps(
            {
                "status": m["status"],
                "counts": counts,
                "updated_utc": m.get("updated_utc"),
                "external_gpu_job": external_gpu_job(),
            },
            indent=2,
        )
    )


def run():
    check_lock()
    worker = HERE / "worker.lock.json"
    if worker.exists():
        old = read(worker)
        if psutil.pid_exists(old["pid"]):
            raise RuntimeError(f"Worker already alive: {old['pid']}")
        os.replace(worker, HERE / f"worker.lock.stale.{uuid.uuid4().hex[:8]}.json")
    write(worker, {"pid": os.getpid(), "utc": stamp()})
    try:
        device = prior.configure_runtime("cuda")
        info = prior.runtime(device)
        if (HERE / "runtime.json").exists():
            old = read(HERE / "runtime.json")
            keys = (
                "torch_version",
                "torch_cuda_version",
                "device_name",
                "dtype",
                "tf32_matmul",
                "tf32_cudnn",
            )
            assert all(info[k] == old[k] for k in keys), "Training environment changed"
        else:
            write(HERE / "runtime.json", info)
        if not (HERE / "pilot.json").exists():
            pilot(device)
        else:
            assert read(HERE / "pilot.json")["passed"]
        update(status="batch_running", worker_pid=os.getpid())
        cfg = read(HERE / "configuration.json")
        for d in cfg["dimensions"]:
            for s in cfg["seeds"]:
                if (HERE / "STOP_AFTER_JOB").exists():
                    update(status="stopped_between_pairs")
                    return
                update(status="batch_running", worker_pid=os.getpid())
                run_pair(d, s, device)
                analyze(complete=False)
        analyze(complete=True)
    except BaseException:
        write(
            HERE / "worker_failure.json",
            {"utc": stamp(), "traceback": traceback.format_exc()},
        )
        update(status="failed_stopped")
        raise
    finally:
        if worker.exists():
            os.replace(worker, HERE / f"worker.lock.ended.{uuid.uuid4().hex[:8]}.json")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["freeze", "run", "pilot", "status", "analyze"])
    args = ap.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "status":
        status()
    elif args.phase == "analyze":
        analyze(True)
    elif args.phase == "pilot":
        pilot(prior.configure_runtime("cuda"))
    else:
        run()
