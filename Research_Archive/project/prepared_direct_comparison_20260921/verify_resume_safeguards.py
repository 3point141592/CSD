"""Exercise resume validation in isolated fixtures, without touching real jobs."""

import uuid

import experiment as e

actual = e.OUT
fixture = actual / "verification" / ("resume_fixture_" + uuid.uuid4().hex[:8])
fixture.mkdir(parents=True)
checks = {}
try:
    e.OUT = fixture
    baseline = {
        "python": "test",
        "torch_version": "test",
        "numpy": "test",
        "device_name": "test",
        "dtype": "float32",
        "tf32_matmul": False,
        "tf32_cudnn": False,
    }
    e.verify_runtime_identity(baseline)
    original = e.sha(fixture / "runtime.json")
    try:
        e.verify_runtime_identity({**baseline, "numpy": "changed"})
    except AssertionError:
        checks["changed_runtime_rejected"] = True
    else:
        raise AssertionError("Changed runtime was accepted")
    assert e.sha(fixture / "runtime.json") == original
    e.write_json(fixture / "lock.json", {"fixture": True})
    folder = fixture / "jobs/d32/seed_7/DirectPrepared"
    folder.mkdir(parents=True)
    for name in [
        "checkpoint.pt",
        "metadata.json",
        "metrics.json",
        "rank_diagnostics.json",
        "training_fit.json",
        "curves.json",
        "configuration.json",
        "verification.json",
    ]:
        e.write_json(folder / name, {"fixture": True})
    e.write_json(
        folder / "metadata.json", {"dimension": 32, "seed": 7, "feature_hashes": {}}
    )
    e.write_json(
        folder / "metrics.json", [{"split": "iid"}, {"split": "exact_rotated"}]
    )
    data = fixture / "data/d32_s7"
    data.mkdir(parents=True)
    (data / "records.npz").write_bytes(
        b"isolated validation fixture, not experimental data"
    )
    e.write_json(
        data / "provenance.json",
        {"records_sha256": e.sha(data / "records.npz"), "feature_hashes": {}},
    )
    completion = {
        "key": "fixture",
        "status": "complete",
        "steps": 1500,
        "lock_sha256": e.sha(fixture / "lock.json"),
        "files": {p.name: e.sha(p) for p in folder.iterdir()},
    }
    e.write_json(folder / "completion.json", completion)
    assert e.job_valid(folder, "fixture")
    checks["consistent_fixture_accepted"] = True
    (folder / "checkpoint.pt").write_bytes(b"changed checkpoint")
    try:
        e.job_valid(folder, "fixture")
    except AssertionError:
        checks["changed_checkpoint_rejected"] = True
    else:
        raise AssertionError("Changed checkpoint accepted")
    e.write_json(folder / "checkpoint.pt", {"fixture": True})
    (data / "records.npz").write_bytes(b"changed records")
    try:
        e.job_valid(folder, "fixture")
    except AssertionError:
        checks["changed_data_rejected"] = True
    else:
        raise AssertionError("Changed data accepted")
    checks["missing_job_not_complete"] = not e.job_valid(fixture / "absent", "fixture")
finally:
    e.OUT = actual
e.write_json(
    actual / "verification/resume_safeguards.json",
    {
        "passed": all(checks.values()),
        "checks": checks,
        "fixture": str(fixture.relative_to(actual)),
        "note": "Synthetic file-validation fixtures only; no training/evaluation result.",
    },
)
print(checks)
