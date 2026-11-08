"""Document a control-file retry repair, retaining all numerical work."""

from experiment import *

archive = OUT / "atomic_io_repair_v5"
prior = read_json(archive / "lock.json")
assert not (OUT / "worker.lock.json").exists()
functions = [
    "direct_features",
    "factory",
    "data",
    "evaluate",
    "training_fit",
    "historic",
    "run_pair",
    "tensor_hash",
]
old_tree = ast.parse((archive / "experiment.py").read_text())
new_tree = ast.parse((OUT / "experiment.py").read_text())
for name in functions:
    a = next(
        n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    b = next(
        n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    assert ast.dump(a, include_attributes=False) == ast.dump(
        b, include_attributes=False
    ), name
hashes = fingerprint()
assert all(hashes[p] == h for p, h in prior["hashes"].items() if p != "experiment.py")
lock = dict(prior)
lock["hashes"] = hashes
lock["frozen_utc"] = now()
lock["revision"] = 5
lock["compatible_prior_locks"] = dict(prior.get("compatible_prior_locks", {}))
lock["compatible_prior_locks"][sha(archive / "lock.json")] = {
    "lock_path": str((archive / "lock.json").relative_to(OUT)),
    "runner_path": str((archive / "experiment.py").relative_to(OUT)),
    "unchanged_numerical_functions": functions,
    "reason": "Atomic JSON/CSV write retry and skipping redundant partial analysis for already complete pairs; numerical work is unchanged.",
}
write_json(OUT / "lock.json", lock)
check_lock()
manifest = read_json(OUT / "manifest.json")
completed = [k for k, v in manifest["jobs"].items() if v["status"] == "complete"]
assert len(completed) == 76
for key in completed:
    d, s, model = key.split("_", 2)
    folder = OUT / "jobs" / d / f"seed_{s[1:]}" / model
    assert job_valid(folder, key)
write_json(
    archive / "repair.json",
    {
        "utc": now(),
        "completed_models_preserved": 76,
        "numerical_function_ASTs_identical": functions,
        "scientific_snapshot_config_plan_analysis_hashes_unchanged": True,
        "repair": "Retry atomic JSON/CSV replacement up to12 times on PermissionError, retaining atomic publication and temporary files on final failure. Skip redundant partial analysis when no new pair completed. No training/evaluation result changed.",
    },
)
update_manifest(status="atomic_io_repaired_ready")
print("Verified all76 committed models; atomic publication repair ready.")
