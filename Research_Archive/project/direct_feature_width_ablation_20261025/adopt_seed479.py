"""Recover the complete seed-479 payload after its transient OneDrive rename failure."""

import shutil
import uuid

import run

source = run.HERE / "attempts" / "d32_s479_DirectRawWide_5747a63e"
destination = run.folder(32, 479, "DirectRawWide")
run.check_lock()
assert source.exists() and not destination.exists()
completion = run.read(source / "completion.json")
assert completion["key"] == "d32_s479_DirectRawWide"
assert completion["steps"] == 1500 and completion["status"] == "complete"
assert (
    completion["lock_sha256"]
    in run.read(run.HERE / "lock.json")["compatible_prior_locks"]
)
assert len(run.read(source / "metrics.json")) == 2
for name, expected in completion["files"].items():
    assert run.digest(source / name) == expected, (name, "source hash mismatch")

stage = run.HERE / "attempts" / ("adopt_d32_s479_DirectRawWide_" + uuid.uuid4().hex[:8])
stage.mkdir()
for name in list(completion["files"]) + ["completion.json"]:
    shutil.copy2(source / name, stage / name)
for name, expected in completion["files"].items():
    assert run.digest(stage / name) == expected, (name, "copy hash mismatch")
assert run.digest(stage / "completion.json") == run.digest(source / "completion.json")
destination.parent.mkdir(parents=True, exist_ok=True)
run.prior.atomic_replace(stage, destination)
assert run.valid(32, 479, "DirectRawWide")
run.update(
    "d32_s479_DirectRawWide",
    status="complete",
    completed_utc=run.stamp(),
    recovered_from=str(source.relative_to(run.HERE)),
    note="Complete saved payload committed after transient directory rename failure; failed attempt retained",
)
print("Recovered", destination)
