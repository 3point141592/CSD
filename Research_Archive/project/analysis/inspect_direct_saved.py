"""Read saved model tensor shapes; no training, forward passes or mutations."""

import argparse
import csv
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
)
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
ROOT = args.package_root.resolve(strict=True)
OUT = (args.output_dir or ROOT / "new_verification/audit_20260909").resolve()
OUT.mkdir(parents=True, exist_ok=True)
for name in [
    "direct_saved_tensor_inspection.json",
    "direct_training_curves_summary.csv",
]:
    if (OUT / name).exists():
        raise FileExistsError(
            f"Refusing to replace {name}; choose a fresh --output-dir"
        )
rows = []
sources = [
    (
        "matched32",
        ROOT
        / "paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d32/models/CurrentAttention_seed_7.pt",
    ),
    (
        "matched64",
        ROOT
        / "paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d64/models/CurrentAttention_seed_7.pt",
    ),
    (
        "canonical32",
        ROOT
        / "results/d32_landscape_representation/models/LandscapeAttention_seed_7.pt",
    ),
    (
        "canonical64",
        ROOT / "results/d64_symmetry_augmentation/models/CurrentAttention_seed_7.pt",
    ),
]
for family, path in sources:
    if not path.exists():
        rows.append(
            {
                "family": family,
                "path": path.relative_to(ROOT).as_posix(),
                "missing": True,
            }
        )
        continue
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state_dict"]
    w, b = state["w2"], state["b2"]
    mat = torch.cat([w.double(), b.double()[:, None]], dim=1)
    row = {
        "family": family,
        "path": path.relative_to(ROOT).as_posix(),
        "w1_shape": list(state["w1"].shape),
        "w2_shape": list(w.shape),
        "b2_shape": list(b.shape),
        "displacement_span_bound": min(mat.shape),
        "actual_w2_b2_rank_rtol_1e6": int(torch.linalg.matrix_rank(mat, rtol=1e-6)),
        "metadata": saved.get("metadata", {}),
    }
    rows.append(row)
(OUT / "direct_saved_tensor_inspection.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8"
)
print(json.dumps(rows, indent=2))

curve_rows = []
for dim in [2, 4, 8, 16, 32, 64]:
    source = (
        ROOT
        / f"paper_csd_toy_revision5/assets/source_data/matched_scaling_study/d{dim}/training_curves.csv"
    )
    records = list(csv.DictReader(source.open(newline="", encoding="utf-8-sig")))
    for method in ["CurrentAttention", "EquivariantAttention"]:
        for step in [500, 1000, 1500]:
            subset = [
                r
                for r in records
                if r["condition"] == method and float(r["step"]) == step
            ]
            curve_rows.append(
                {
                    "dimension": dim,
                    "method": method,
                    "step": step,
                    "n": len(subset),
                    "mean_training_minibatch_cosine": sum(
                        float(r["mean_matched_cosine"]) for r in subset
                    )
                    / len(subset),
                    "mean_training_minibatch_loss": sum(
                        float(r["loss"]) for r in subset
                    )
                    / len(subset),
                    "source": source.relative_to(ROOT).as_posix(),
                }
            )
with (OUT / "direct_training_curves_summary.csv").open(
    "w", newline="", encoding="utf-8"
) as stream:
    writer = csv.DictWriter(stream, fieldnames=list(curve_rows[0]))
    writer.writeheader()
    writer.writerows(curve_rows)
