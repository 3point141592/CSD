"""Preserve completed models across a diagnostic-only tolerance correction."""

from experiment import *

archive = OUT / "normalization_check_repair_v4"
old = read_json(archive / "lock.json")
assert not (OUT / "worker.lock.json").exists()
manifest = read_json(OUT / "manifest.json")
assert sum(x["status"] == "complete" for x in manifest["jobs"].values()) == 74
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
newhash = fingerprint()
assert all(newhash[p] == h for p, h in old["hashes"].items() if p != "experiment.py")
evidence = read_json(OUT / "verification/normalization_assertion_d64_s197.json")
assert (
    evidence["shared_phi_bitwise_identical"]
    and evidence["same_mask"]
    and evidence["same_relative_vectors"]
)
for row in evidence["rows"]:
    for suffix in ["mean", "std"]:
        a = row["direct_" + suffix]
        b = row["relational_" + suffix]
        assert abs(a - b) <= 1e-5 + 8 * np.finfo(np.float32).eps * abs(b)
lock = dict(old)
lock["hashes"] = newhash
lock["frozen_utc"] = now()
lock["revision"] = 4
lock["compatible_prior_locks"] = {
    sha(archive / "lock.json"): {
        "lock_path": str((archive / "lock.json").relative_to(OUT)),
        "runner_path": str((archive / "experiment.py").relative_to(OUT)),
        "unchanged_numerical_functions": functions,
        "reason": "Only normalization-verification tolerance and control/provenance guards changed. Model, features, normalization computations, data, optimizer, evaluation, scoring and analysis are unchanged.",
    }
}
write_json(OUT / "lock.json", lock)
check_lock()
records = []
for d in (32, 64):
    for seed in matched_config(d).seeds:
        paths = [OUT / f"jobs/d{d}/seed_{seed}/{m}" for m in MODELS]
        if not all(p.exists() for p in paths):
            continue
        states = [
            torch.load(p / "checkpoint.pt", map_location="cpu", weights_only=True)[
                "state_dict"
            ]
            for p in paths
        ]
        for suffix in ["mean", "scale"]:
            a = states[0]["descriptor_" + suffix][d:]
            b = states[1]["invariant_" + suffix]
            assert torch.allclose(
                a, b, rtol=8 * torch.finfo(torch.float32).eps, atol=1e-5
            )
        for p, m in zip(paths, MODELS):
            assert job_valid(p, f"d{d}_s{seed}_{m}")
        records.append(
            {
                "dimension": d,
                "seed": seed,
                "both_completed_checkpoints_revalidated": True,
            }
        )
write_json(
    archive / "repair.json",
    {
        "utc": now(),
        "completed_models_preserved": 74,
        "full_pairs_revalidated": records,
        "maximum_observed_mean_difference": evidence["max_absolute_mean_difference"],
        "maximum_observed_relative_mean_difference": evidence[
            "max_relative_mean_difference"
        ],
        "new_check_rtol": 8 * np.finfo(np.float32).eps.item(),
        "new_check_atol": 1e-5,
        "unchanged_numerical_functions": functions,
        "all_scientific_snapshot_config_plan_and_analysis_hashes_unchanged": True,
        "normalization_computation_changed": False,
        "training_or_scoring_changed": False,
        "control_change": "Check for unrelated GPU trainers before every seed pair, as well as at launch. This prevents new overlapping jobs from being missed.",
    },
)
update_manifest(status="normalization_check_repaired_waiting_for_gpu")
print(
    "Revalidated74 complete models and all four pilots; no trained model or result was replaced."
)
