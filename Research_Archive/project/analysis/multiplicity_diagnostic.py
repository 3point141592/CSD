"""Zero-training retrospective input-shortcut audit, NumPy only.

Select centers by multiplicity >=2, with no amplitude, shape, labels, or known
useful count consulted by selector. Labels are used only to score selections.
Reconstructs the canonical family and seed offsets from saved source.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
)
_parser.add_argument("--output-dir", type=Path)
_args = _parser.parse_args()
B = _args.package_root.resolve(strict=True)
O = (_args.output_dir or B / "new_verification/audit_20260909").resolve()
O.mkdir(parents=True, exist_ok=True)
for _name in ["multiplicity_per_set.csv", "multiplicity_summary.json"]:
    if (O / _name).exists():
        raise FileExistsError(
            f"Refusing to replace {_name}; choose a fresh --output-dir"
        )
sys.path.insert(0, str(B))
from creativity_toy.config import STUDY_SEEDS
from creativity_toy.config_v2 import LandscapeV2Config
from creativity_toy.landscapes_d16 import DimensionalLandscapeFamily
from creativity_toy.landscapes_v2 import starting_state_v2

begin = time.perf_counter()
# The matched protocol's task defaults keep selector results comparable;
# only the dimension changes for this 32D audit.
config = LandscapeV2Config(dimension=32)
rows = []
for seed in STUDY_SEEDS:
    family = DimensionalLandscapeFamily(config)
    tasks = family.sample_tasks(
        geometries=5, seed=seed + 20000, variant="iid", all_goals=True
    )
    rng = np.random.default_rng(seed + 20001)
    for task in tasks:
        # Restrict selection to coordinates so usefulness labels cannot guide the rule.
        c = task.component_centers
        same = np.linalg.norm(c[:, None] - c[None, :], axis=-1) <= 1e-5
        representative = np.array([i for i in range(len(c)) if not same[i, :i].any()])
        counts = same[representative].sum(axis=1)
        chosen = representative[counts >= 2]
        chosen_centers = c[chosen]
        # Cycle discovered groups to meet the common eight-candidate budget without a mode-count input.
        points = chosen_centers[np.arange(8) % len(chosen)]
        before_region = task.region(points)
        before_useful = task.is_useful_region(before_region) & (
            task.quality(points) >= config.useful_threshold
        )
        after = task.refine(points, steps=9, rate=0.28)
        after_region = task.region(after)
        after_useful = task.is_useful_region(after_region) & (
            task.quality(after) >= config.useful_threshold
        )
        # Delay label access until scoring so ground truth cannot affect selected groups.
        true_useful = np.any(
            np.linalg.norm(
                c[representative, None] - task.useful_centers[None, :], axis=-1
            )
            <= 1e-5,
            axis=1,
        )
        pred = counts >= 2
        for trial in range(2):
            start = starting_state_v2(task, rng)
            rows.append(
                dict(
                    seed=seed,
                    geometry_id=int(task.geometry_id),
                    goal_index=int(task.goal_index),
                    trial=trial,
                    groups=len(representative),
                    selected=len(chosen),
                    true_useful=int(true_useful.sum()),
                    center_classification_accuracy=float((true_useful == pred).mean()),
                    before_mode_set_recall=len(np.unique(before_region[before_useful]))
                    / task.n_useful,
                    after_mode_set_recall=len(np.unique(after_region[after_useful]))
                    / task.n_useful,
                    after_useful_candidate_rate=float(after_useful.mean()),
                    after_decoy_rate=float((after_region > task.n_useful).mean()),
                    after_mean_quality=float(task.quality(after).mean()),
                    max_selected_travel_distance=float(
                        np.linalg.norm(chosen_centers - start, axis=1).max()
                    ),
                    all_selected_travel_within_six=bool(
                        (np.linalg.norm(chosen_centers - start, axis=1) <= 6).all()
                    ),
                    nearest_center_is_anchor=bool(
                        np.linalg.norm(
                            c[
                                representative[
                                    np.argmin(
                                        np.linalg.norm(
                                            c[representative] - start, axis=1
                                        )
                                    )
                                ]
                            ]
                            - task.anchor
                        )
                        < 1e-5
                    ),
                )
            )
with (O / "multiplicity_per_set.csv").open("w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
report = dict(
    status="NEW retrospective diagnostic by independent reviewer; NumPy family reconstruction, no checkpoint loading",
    selection="coordinate grouping tolerance 1e-5; select all centers with multiplicity>=2, no top-three count supplied; cycle 8 slots",
    independence="30 seeds are independent replication units; 900 set rows are nested under 150 geometries; selector and center outcomes do not depend on starts",
    numerical_scope="CPU NumPy float64 refiner; not a bitwise GPU evaluation replay",
    n_seeds=len(STUDY_SEEDS),
    n_unique_geometries=30 * 5,
    n_goal_conditioned_tasks=len(rows) // 2,
    n_sets=len(rows),
    columns={
        k: dict(
            mean=float(np.mean([r[k] for r in rows])),
            min=float(min(r[k] for r in rows)),
            max=float(max(r[k] for r in rows)),
        )
        for k in rows[0]
        if k not in ["seed", "geometry_id", "goal_index", "trial"]
    },
    source_hashes={
        p.relative_to(B).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            B / "creativity_toy/landscapes_d16.py",
            B / "creativity_toy/landscapes_v2.py",
            B / "creativity_toy/config_v2.py",
            B / "gen3_gpu/config_d16.py",
        ]
    },
    numpy_version=np.__version__,
    seconds=time.perf_counter() - begin,
)
(O / "multiplicity_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf8"
)
print(json.dumps(report, indent=2))
