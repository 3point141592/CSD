"""Record pilot failure and neutral serialization repair before the full batch."""

from experiment import *

assert not (OUT / "worker.lock.json").exists()
assert not list((OUT / "jobs").glob("**/completion.json"))
failed = list((OUT / "attempts").glob("*/failure.json"))
assert len(failed) == 1
archive = OUT / "pilot_serialization_repair_v3"
archive.mkdir(exist_ok=False)
for name in ("lock.json", "manifest.json", "worker_failure.json"):
    p = OUT / name
    if p.exists():
        shutil.copy2(p, archive / name)
write_json(
    archive / "repair.json",
    {
        "utc": now(),
        "accepted_models": 0,
        "failed_training_attempts": 1,
        "repair": "Checkpoint metadata runtime values are converted to plain JSON primitives before torch.save. PyTorch TorchVersion is a str subclass that weights_only=True does not allow as a pickle global. Training/model/loss/data/scoring functions and configuration are unchanged. The same d32 seed7 job will be retrained under final code; failed checkpoint and failure record are preserved. Reporting adds explicit failed-attempt accounting and labeled historical context; endpoints and uncertainty method unchanged.",
        "failure_records": [str(p.relative_to(OUT)) for p in failed],
    },
)
lock = read_json(OUT / "lock.json")
lock["hashes"] = fingerprint()
lock["frozen_utc"] = now()
lock["revision"] = 3
lock["prior_lock_sha256"] = sha(archive / "lock.json")
write_json(OUT / "lock.json", lock)
update_manifest(status="pilot_serialization_repair_ready")
print(
    "Repair recorded; full batch still gated by all four pilot runs and checkpoint replay."
)
