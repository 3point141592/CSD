"""Record the pre-training operational amendment while retaining prior protocol records."""

from experiment import *

assert not list((OUT / "jobs").glob("**/checkpoint.pt"))
assert not list((OUT / "attempts").glob("**/checkpoint.pt"))
assert not (OUT / "worker.lock.json").exists()
archive = OUT / "pretraining_revision_1"
archive.mkdir(exist_ok=False)
for name in ("lock.json", "manifest.json", "worker_failure.json"):
    p = OUT / name
    if p.exists():
        shutil.copy2(p, archive / name)
write_json(
    archive / "amendment.json",
    {
        "utc": now(),
        "models_trained": 0,
        "reason": "Operational resume audit: require identical runtime versions/hardware/precision/thread settings on resume and verify pilot lock/checkpoint completeness before full batch. No scientific architecture, data, training or scoring change. Queued worker stopped gracefully; the unrelated RocketLeague job was untouched.",
    },
)
lock = read_json(OUT / "lock.json")
lock["hashes"] = fingerprint()
lock["frozen_utc"] = now()
lock["pretraining_revision"] = 2
lock["prior_lock_sha256"] = sha(archive / "lock.json")
write_json(OUT / "lock.json", lock)
update_manifest(status="awaiting_gpu_after_pretraining_resume_audit")
print(
    "Pre-training operational correction recorded; no trained results exist or were replaced."
)
