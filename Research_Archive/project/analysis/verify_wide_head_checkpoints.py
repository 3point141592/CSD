"""CPU-only verification of wider decoder rank and narrow pipeline checks.

This inspects saved states and provenance. It performs no model training or
evaluation forward pass. Full output rank removes the particular linear-span
bound; it does not establish universal nonlinear expressivity or useful search.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
)
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
ROOT = args.package_root.resolve(strict=True)
OUT = (args.output_dir or ROOT / "new_verification/audit_20260909").resolve()
OUT.mkdir(parents=True, exist_ok=True)
for name in ["wide_head_rank_per_seed.csv", "wide_head_checkpoint_verification.json"]:
    if (OUT / name).exists():
        raise FileExistsError(
            f"Refusing to replace {name}; choose a fresh --output-dir"
        )
sys.path.insert(0, str(ROOT))
import torch
from creativity_toy.config import STUDY_SEEDS

torch.set_num_threads(2)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


root = ROOT / "reproduction_runs/direct_head_ablation_complete_20260909"
lock = ROOT / "audit_records/engineering_20260909/direct_head_ablation_protocol.json"
protocol = json.loads(lock.read_text(encoding="utf-8"))
expected = json.loads((root / "completion.json").read_text(encoding="utf-8"))
frozen = [
    {
        "path": p,
        "expected_sha256": wanted,
        "actual_sha256": sha(ROOT / p),
        "matches": sha(ROOT / p) == wanted,
    }
    for p, wanted in protocol["protocol"]["source_hashes"].items()
]
rows = []
loaded_hashes = {}
pipeline = []
for d in [32, 64]:
    for seed in STUDY_SEEDS:
        path = root / f"d{d}/models/DirectWide_seed_{seed}.pt"
        digest = sha(path)
        loaded_hashes[path.relative_to(ROOT).as_posix()] = digest
        saved = torch.load(path, map_location="cpu", weights_only=True)
        state = saved["state_dict"]
        matrix = torch.cat([state["w2"].double(), state["b2"].double()[:, None]], dim=1)
        singular = torch.linalg.svdvals(matrix)
        rank = int((singular > singular[0] * 1e-6).sum())
        rows.append(
            {
                "dimension": d,
                "seed": seed,
                "output_hidden_width": state["w2"].shape[1],
                "augmented_rows": matrix.shape[0],
                "augmented_columns": matrix.shape[1],
                "numerical_rank_rtol_1e6": rank,
                "smallest_to_largest_singular_value": float(singular[-1] / singular[0]),
                "checkpoint_sha256": digest,
                "checkpoint": path.relative_to(ROOT).as_posix(),
                "recorded_parameter_count": saved["metadata"]["parameter_count"],
            }
        )
    new_path = root / f"d{d}/models/Direct16_retrained_check_seed_7.pt"
    old_path = (
        ROOT
        / f"paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d{d}/models/CurrentAttention_seed_7.pt"
    )
    new = torch.load(new_path, map_location="cpu", weights_only=True)
    old = torch.load(old_path, map_location="cpu", weights_only=True)
    key_equal = new["state_dict"].keys() == old["state_dict"].keys()
    unequal = [
        key
        for key in new["state_dict"]
        if key not in old["state_dict"]
        or not torch.equal(new["state_dict"][key], old["state_dict"][key])
    ]
    pipeline.append(
        {
            "dimension": d,
            "seed": 7,
            "state_keys_identical": key_equal,
            "unequal_tensor_names": unequal,
            "all_state_tensors_identical": key_equal and not unequal,
            "complete_training_history_identical": new["history"] == old["history"],
            "retrained_checkpoint": new_path.relative_to(ROOT).as_posix(),
            "archived_checkpoint": old_path.relative_to(ROOT).as_posix(),
            "retrained_sha256": sha(new_path),
            "archived_sha256": sha(old_path),
            "scope": "Two specific narrow training pipeline replications, not all historical training",
        }
    )
    loaded_hashes[new_path.relative_to(ROOT).as_posix()] = sha(new_path)

consolidation = []
for rel, wanted in expected["checkpoints"].items():
    path = root / rel
    matching_parts = []
    for part in [
        "direct_head_ablation_20260909",
        "direct_head_ablation_resume_20260909",
    ]:
        origin = ROOT / "reproduction_runs" / part / rel
        if origin.is_file() and sha(origin) == sha(path):
            matching_parts.append(origin.relative_to(ROOT).as_posix())
    consolidation.append(
        {
            "checkpoint": path.relative_to(ROOT).as_posix(),
            "matches_recorded_sha256": sha(path) == wanted,
            "byte_equal_source_parts": matching_parts,
        }
    )
summary = [
    {
        "dimension": d,
        "checkpoints": sum(r["dimension"] == d for r in rows),
        "ranks": sorted(
            set(r["numerical_rank_rtol_1e6"] for r in rows if r["dimension"] == d)
        ),
        "minimum_singular_value_ratio": min(
            r["smallest_to_largest_singular_value"] for r in rows if r["dimension"] == d
        ),
    }
    for d in [32, 64]
]
report = {
    "status": "Saved-checkpoint retrospective audit; CPU float64 SVD; no training",
    "source_sha256": sha(__file__),
    "torch_version": torch.__version__,
    "relative_rank_threshold": 1e-6,
    "protocol_sha256": sha(lock),
    "protocol_frozen_at_utc": protocol["frozen_at_utc"],
    "frozen_source_checks": frozen,
    "summary": summary,
    "pipeline_checks": pipeline,
    "consolidation_checks": consolidation,
    "input_checkpoints_unchanged": all(
        sha(ROOT / p) == h for p, h in loaded_hashes.items()
    ),
    "interpretation": "Full augmented output-matrix rank removes the original rank-17 linear-span limitation only; it does not establish ability to realize every direction or a post-refinement coverage guarantee.",
}
report["all_checks_passed"] = (
    all(x["matches"] for x in frozen)
    and all(r["numerical_rank_rtol_1e6"] == r["dimension"] for r in rows)
    and all(
        x["all_state_tensors_identical"] and x["complete_training_history_identical"]
        for x in pipeline
    )
    and all(
        x["matches_recorded_sha256"] and x["byte_equal_source_parts"]
        for x in consolidation
    )
    and report["input_checkpoints_unchanged"]
    and len(rows) == 60
    and len(consolidation) == 62
)
with (OUT / "wide_head_rank_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / "wide_head_checkpoint_verification.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {
            "all_checks_passed": report["all_checks_passed"],
            "summary": summary,
            "pipeline_checks": pipeline,
            "frozen_files": len(frozen),
            "consolidated_checkpoints": len(consolidation),
        },
        indent=2,
    )
)
sys.exit(0 if report["all_checks_passed"] else 1)
