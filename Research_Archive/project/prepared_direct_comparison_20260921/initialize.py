import hashlib
import json
import shutil
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SNAP = OUT / "source_snapshot"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


orig = ROOT / "paper_csd_toy_revision5/assets/source_data/matched_scaling_study"
checks = []
for d in (32, 64):
    manifest = json.loads((orig / f"d{d}/manifest.json").read_text())
    for rel, expected in manifest["source_hashes"].items():
        actual = digest(ROOT / rel)
        checks.append(
            {
                "dimension": d,
                "path": rel,
                "expected": expected,
                "actual": actual,
                "match": actual == expected,
            }
        )
assert all(x["match"] for x in checks), checks
copied = []
for package in ("gen3_gpu", "creativity_toy"):
    for p in (ROOT / package).rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(ROOT)
        target = SNAP / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert digest(target) == digest(p)
        else:
            shutil.copy2(p, target)
        copied.append(
            {
                "original": str(rel),
                "snapshot": str(target.relative_to(OUT)),
                "sha256": digest(p),
            }
        )
driver = ROOT / "paper_csd_toy_revision5/scripts/run_matched_scaling_study.py"
target = SNAP / "run_matched_scaling_study.py"
if not target.exists():
    shutil.copy2(driver, target)
assert digest(target) == digest(driver)
copied.append(
    {
        "original": str(driver.relative_to(ROOT)),
        "snapshot": str(target.relative_to(OUT)),
        "sha256": digest(driver),
    }
)
(OUT / "source_origin.json").write_text(
    json.dumps({"archived_checks": checks, "files": copied}, indent=2)
)
sys.path.insert(0, str(SNAP))
from dataclasses import asdict

from gen3_gpu.study_symmetry import _benchmark
from run_matched_scaling_study import matched_config

configs = {}
for d in (32, 64):
    c = matched_config(d)
    b = _benchmark(c)
    archived = json.loads((orig / f"d{d}/manifest.json").read_text())["config"]
    assert c.to_dict() == archived, (c.to_dict(), archived)
    configs[str(d)] = {
        "matched": c.to_dict(),
        "benchmark": asdict(b),
        "direct_prepared": {
            "component_hidden_dimension": 256,
            "component_token_dimension": 128,
            "proposal_hidden_dimension": 256,
            "descriptor_dimension": 2 * d + 11,
        },
    }
    print(
        d,
        "test geometries",
        b.base.base.test_geometries,
        "eval trials",
        b.base.base.trials_per_task,
        "refiner",
        b.base.base.training.refinement_steps,
        b.base.base.training.refinement_rate,
        "threshold",
        b.base.base.landscape.useful_threshold,
    )
(OUT / "configurations.json").write_text(json.dumps(configs, indent=2))
print(
    "Verified archived source hashes and exact d32/d64 configurations; snapshot files:",
    len(copied),
)
