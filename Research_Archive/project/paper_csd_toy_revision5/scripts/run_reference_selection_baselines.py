"""Protocol-matched d=32 reference-selection baselines for the CSD paper.

This isolated paper experiment reuses the canonical supervised d=32 task
splits, starts, candidate budget, refinement, and region metrics.  It does not
modify repository experiment artifacts.  Four policies point directly toward
ranked component centers: amplitude, amplitude/distance, random ranking, and a
44-parameter logistic scorer trained on invariant per-component descriptors.

The logistic scorer's labels are derived on the *training split only* by
matching component centers to generator-known useful centers.  No useful or
decoy label is consumed at inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
from creativity_toy.config import STUDY_SEEDS
from creativity_toy.experiments.study_v2 import _make_starts
from gen3_gpu.config_symmetry import SymmetryStudyConfig
from gen3_gpu.landscape_representation import LandscapeProposalBatch
from gen3_gpu.study_capacity_diagnostic import _aligned
from gen3_gpu.study_d16 import _datasets
from gen3_gpu.study_failure_diagnostics import _family
from gen3_gpu.study_symmetry import _benchmark
from gen3_gpu.symmetry import (
    EquivariantComponentBatch,
    build_equivariant_components,
    invariant_descriptor_dimension,
    invariant_normalization,
)
from gen3_gpu.training_landscape_supervised import (
    component_evaluation_labels,
    evaluate_landscape_policy,
)
from torch import nn

METHODS = (
    "amplitude_topk",
    "amplitude_distance_topk",
    "descriptor_logistic_topk",
    "random_topk",
)


@dataclass(frozen=True)
class Protocol:
    dimension: int = 32
    candidates: int = 8
    train_geometries: int = 80
    starts_per_task: int = 32
    test_geometries: int = 5
    trials_per_task: int = 2
    refinement_steps: int = 9
    refinement_rate: float = 0.28
    useful_threshold: float = 0.62
    scorer_steps: int = 500
    scorer_batch_tasks: int = 960
    scorer_learning_rate: float = 0.03
    scorer_weight_decay: float = 1e-4
    label_rule: str = "component center within 1e-5 of a useful center"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


class DescriptorLogisticScorer(nn.Module):
    """One affine scalar per component; no candidate or cross-reference path."""

    def __init__(self, feature_count: int, *, device: torch.device) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count, 1, device=device, dtype=torch.float32)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


class RankedReferencePolicy(nn.Module):
    """Select K component records and point directly toward their centers."""

    def __init__(
        self,
        *,
        method: str,
        candidates: int,
        scorer: DescriptorLogisticScorer | None = None,
        mean: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        random_seed: int = 0,
    ) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown ranking method {method!r}")
        self.method = method
        self.candidates = int(candidates)
        self.scorer = scorer
        self.register_buffer(
            "feature_mean", None if mean is None else mean.detach().clone()
        )
        self.register_buffer(
            "feature_scale", None if scale is None else scale.detach().clone()
        )
        self.random_seed = int(random_seed)

    def component_scores(self, components: EquivariantComponentBatch) -> torch.Tensor:
        amplitude = components.invariant[..., 0]
        distance = components.invariant[..., 1]
        if self.method == "amplitude_topk":
            score = amplitude
        elif self.method == "amplitude_distance_topk":
            score = amplitude / torch.clamp(distance, min=1e-8)
        elif self.method == "descriptor_logistic_topk":
            if (
                self.scorer is None
                or self.feature_mean is None
                or self.feature_scale is None
            ):
                raise RuntimeError(
                    "descriptor scorer and training normalization are required"
                )
            normalized = (components.invariant - self.feature_mean) / self.feature_scale
            score = self.scorer(normalized)
        else:
            generator = torch.Generator(device=components.invariant.device).manual_seed(
                self.random_seed
            )
            score = torch.rand(
                components.mask.shape,
                generator=generator,
                device=components.invariant.device,
                dtype=components.invariant.dtype,
            )
        return torch.where(components.mask, score, torch.full_like(score, -torch.inf))

    def selected_indices(self, components: EquivariantComponentBatch) -> torch.Tensor:
        if torch.any(components.mask.sum(dim=1) < self.candidates):
            raise ValueError("every task must provide at least K active references")
        score = self.component_scores(components)
        # Stable sorting makes equal-amplitude ties deterministic and exposes
        # rather than hides the generator's materialized component ordering.
        return torch.argsort(score, dim=1, descending=True, stable=True)[
            :, : self.candidates
        ]

    def forward(
        self, h: torch.Tensor, components: EquivariantComponentBatch | None = None
    ) -> LandscapeProposalBatch:
        if components is None:
            raise ValueError("ranked-reference policies require component descriptors")
        index = self.selected_indices(components)
        gather_vector = index.unsqueeze(-1).expand(
            -1, -1, components.relative.shape[-1]
        )
        relative = torch.gather(components.relative, 1, gather_vector)
        alpha = torch.linalg.vector_norm(relative, dim=-1)
        direction = F.normalize(relative, dim=-1, eps=1e-8)
        candidate = h[:, None, :] + alpha.unsqueeze(-1) * direction
        attention = F.one_hot(index, num_classes=components.mask.shape[1]).to(
            components.invariant.dtype
        )
        return LandscapeProposalBatch(relative, direction, alpha, candidate, attention)


def train_descriptor_scorer(
    components: EquivariantComponentBatch,
    labels: torch.Tensor,
    *,
    mean: torch.Tensor,
    scale: torch.Tensor,
    seed: int,
    steps: int,
    batch_tasks: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[DescriptorLogisticScorer, list[dict[str, float]]]:
    scorer = DescriptorLogisticScorer(
        components.invariant.shape[-1], device=components.invariant.device
    )
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device=components.invariant.device).manual_seed(seed)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        count = min(batch_tasks, len(labels))
        index = torch.randperm(
            len(labels), generator=generator, device=components.invariant.device
        )[:count]
        batch = components.index_select(index)
        target = labels.index_select(0, index).float()
        normalized = (batch.invariant - mean) / scale
        logits = scorer(normalized)
        selected_logits = logits[batch.mask]
        selected_target = target[batch.mask]
        loss = F.binary_cross_entropy_with_logits(selected_logits, selected_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {1, 100, 250, steps}:
            with torch.no_grad():
                accuracy = torch.mean(
                    ((selected_logits >= 0) == selected_target.bool()).float()
                )
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "batch_accuracy": float(accuracy.cpu()),
                }
            )
    scorer.eval()
    return scorer, history


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "mean_matched_cosine",
        "proposal_recall",
        "mode_set_recall",
        "useful_region_count",
        "candidate_decoy_rate",
        "candidate_mean_quality",
    )
    result: list[dict[str, object]] = []
    for method in (*METHODS, "relational_csd"):
        subset = [row for row in rows if row["method"] == method]
        if not subset:
            continue
        summary: dict[str, object] = {"method": method, "n": len(subset)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd"] = float(values.std(ddof=0))
            summary[f"{metric}_ci95"] = float(
                1.96 * values.std(ddof=1) / np.sqrt(len(values))
            )
        result.append(summary)
    return result


def relational_rows() -> list[dict[str, object]]:
    source = REPO / "results" / "d32_symmetry_augmentation" / "per_seed.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    kept = []
    for row in rows:
        if row["condition"] == "EquivariantAttention" and row["split"] == "iid":
            kept.append(
                {
                    "seed": int(row["seed"]),
                    "method": "relational_csd",
                    **{
                        key: float(row[key])
                        for key in (
                            "mean_matched_cosine",
                            "proposal_recall",
                            "mode_set_recall",
                            "useful_region_count",
                            "candidate_decoy_rate",
                            "candidate_mean_quality",
                        )
                    },
                }
            )
    if [int(row["seed"]) for row in kept] != list(STUDY_SEEDS):
        raise AssertionError("canonical Relational CSD rows do not match STUDY_SEEDS")
    return kept


def run(*, output: Path, device: torch.device) -> None:
    protocol = Protocol()
    config = SymmetryStudyConfig()
    if list(config.seeds) != list(STUDY_SEEDS):
        raise AssertionError("canonical symmetry seeds changed")
    benchmark = _benchmark(config)
    base = benchmark.base.base
    observed = Protocol(
        train_geometries=base.train_geometries,
        test_geometries=base.test_geometries,
        trials_per_task=base.trials_per_task,
        refinement_steps=base.training.refinement_steps,
        refinement_rate=base.training.refinement_rate,
        useful_threshold=base.landscape.useful_threshold,
    )
    if observed != protocol:
        raise AssertionError(
            f"canonical protocol drifted: {observed!r} != {protocol!r}"
        )

    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    all_rows: list[dict[str, object]] = []
    scorer_history: list[dict[str, object]] = []

    for ordinal, seed in enumerate(STUDY_SEEDS, 1):
        family = _family(benchmark)
        train, _, _, _, test, test_starts = _datasets(family, benchmark, seed)
        train_nested = _make_starts(
            train, trials=config.starts_per_task, seed=seed + 41_001
        )
        packed_train, h_train = _aligned(train, train_nested, device)
        packed_test, h_test = _aligned(test, test_starts, device)
        train_components = build_equivariant_components(packed_train, h_train)
        test_components = build_equivariant_components(packed_test, h_test)
        mean, scale = invariant_normalization(train_components)
        train_labels, _ = component_evaluation_labels(packed_train)
        scorer, history = train_descriptor_scorer(
            train_components,
            train_labels,
            mean=mean,
            scale=scale,
            seed=seed + 111_001,
            steps=protocol.scorer_steps,
            batch_tasks=protocol.scorer_batch_tasks,
            learning_rate=protocol.scorer_learning_rate,
            weight_decay=protocol.scorer_weight_decay,
        )
        checkpoint = model_dir / f"descriptor_logistic_seed_{seed}.pt"
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in scorer.state_dict().items()
                },
                "mean": mean.detach().cpu(),
                "scale": scale.detach().cpu(),
                "seed": seed,
                "parameter_count": sum(
                    parameter.numel() for parameter in scorer.parameters()
                ),
                "protocol": asdict(protocol),
            },
            checkpoint,
        )
        scorer_history.extend({"seed": seed, **point} for point in history)

        policies = {
            "amplitude_topk": RankedReferencePolicy(
                method="amplitude_topk", candidates=protocol.candidates
            ),
            "amplitude_distance_topk": RankedReferencePolicy(
                method="amplitude_distance_topk", candidates=protocol.candidates
            ),
            "descriptor_logistic_topk": RankedReferencePolicy(
                method="descriptor_logistic_topk",
                candidates=protocol.candidates,
                scorer=scorer,
                mean=mean,
                scale=scale,
            ),
            "random_topk": RankedReferencePolicy(
                method="random_topk",
                candidates=protocol.candidates,
                random_seed=seed + 131_001,
            ),
        }
        for method, policy in policies.items():
            policy.to(device).eval()
            result = evaluate_landscape_policy(
                policy,
                packed_test,
                h_test,
                test_components,
                lambda_alpha=config.lambda_alpha,
                refinement_steps=protocol.refinement_steps,
                refinement_rate=protocol.refinement_rate,
                useful_threshold=protocol.useful_threshold,
                collect_per_set=False,
                collect_attention_rows=False,
            )
            all_rows.append({"seed": seed, "method": method, **result.summary})
        write_csv(output / "per_seed.csv", all_rows)
        write_csv(output / "scorer_training.csv", scorer_history)
        atomic_json(
            output / "progress.json",
            {
                "completed_seeds": [int(value) for value in STUDY_SEEDS[:ordinal]],
                "completed": ordinal,
                "required": len(STUDY_SEEDS),
            },
        )
        print(
            f"reference baselines: seed {seed} ({ordinal}/{len(STUDY_SEEDS)})",
            flush=True,
        )

    all_rows.extend(relational_rows())
    summaries = summarize(all_rows)
    write_csv(output / "per_seed_with_relational_csd.csv", all_rows)
    write_csv(output / "summary.csv", summaries)
    atomic_json(
        output / "manifest.json",
        {
            "status": "new isolated paper diagnostic",
            "scientific_question": "Can per-reference descriptor ranking alone solve the d=32 benchmark?",
            "protocol": asdict(protocol),
            "methods": {
                "amplitude_topk": "rank component records by A_j",
                "amplitude_distance_topk": "rank component records by A_j/(||mu_j-h||+1e-8)",
                "descriptor_logistic_topk": (
                    "44-parameter affine scorer over the same 43 invariant per-component descriptors; "
                    "training-only binary labels identify component centers coincident with useful centers"
                ),
                "random_topk": "uniform pseudorandom component ranking, independently seeded per study seed",
                "relational_csd": "frozen canonical results/d32_symmetry_augmentation/per_seed.csv rows",
            },
            "fairness": {
                "candidate_budget": 8,
                "active_references_per_iid_task": 12,
                "selection_unit": "materialized component record; duplicate peak/corridor centers are not deduplicated",
                "proposal": "point directly to selected component center, alpha=distance",
                "local_refinement": "canonical nine-step gradient ascent on log Q",
                "model_selection": "none",
                "test_tuning": False,
                "test_labels_consumed_by_policies": False,
            },
            "source_hashes": {
                "gen3_gpu/symmetry.py": sha256(REPO / "gen3_gpu" / "symmetry.py"),
                "gen3_gpu/training_landscape_supervised.py": sha256(
                    REPO / "gen3_gpu" / "training_landscape_supervised.py"
                ),
                "gen3_gpu/study_symmetry.py": sha256(
                    REPO / "gen3_gpu" / "study_symmetry.py"
                ),
                "results/d32_symmetry_augmentation/per_seed.csv": sha256(
                    REPO / "results" / "d32_symmetry_augmentation" / "per_seed.csv"
                ),
            },
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU",
            },
        },
    )
    atomic_json(
        output / "completion.json",
        {
            "complete": True,
            "seeds": len(STUDY_SEEDS),
            "new_baseline_rows": len(all_rows) - len(STUDY_SEEDS),
            "relational_reference_rows": len(STUDY_SEEDS),
            "descriptor_parameter_count": invariant_descriptor_dimension(32) + 1,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER / "assets" / "source_data" / "reference_selection_baselines",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    run(output=args.output.resolve(), device=torch.device(device_name))


if __name__ == "__main__":
    main()
